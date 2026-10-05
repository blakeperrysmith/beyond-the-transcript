import json
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(synthetic_artifacts, tmp_path, monkeypatch):
    monkeypatch.setenv("BTT_ARTIFACTS", str(synthetic_artifacts))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'t.db'}")
    import importlib

    import app.main as m

    importlib.reload(m)
    m.RATE_LIMIT = 1000
    with TestClient(m.app) as c:
        yield c, m


def pcm(secs=1.5, sr=16000):
    t = np.arange(int(secs * sr)) / sr
    y = 0.3 * np.sin(2 * np.pi * 180 * t) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))
    return (y * 32767).astype("<i2").tobytes()


def post(c, body, sr="16000", source="live"):
    return c.post("/api/analyze", content=body, headers={"X-Sample-Rate": sr, "X-Source": source})


def test_analyze_roundtrip(client):
    c, _ = client
    r = post(c, pcm())
    assert r.status_code == 200
    j = r.json()
    assert abs(sum(j["probs"].values()) - 1) < 1e-4
    assert set(j["timings"]) >= {"features_ms", "prosody_ms", "tract_ms", "server_ms", "cold"}
    assert j["timings"]["cold"] is True
    assert len(j["prosody"]["f0_contour"]) > 10
    assert j["stats"]["n"] == 1
    j2 = post(c, pcm(), source="sample").json()
    assert j2["timings"]["cold"] is False and j2["stats"]["n"] == 2
    ok = c.post("/api/client-latency", json={"run_id": j2["run_id"], "e2e_ms": 55.0})
    assert ok.status_code == 200 and ok.json()["stats"]["round_trip"]["n"] == 1


def test_bad_inputs(client):
    c, _ = client
    assert post(c, pcm(0.1)).status_code == 400  # too short
    assert post(c, pcm(9.0)).status_code == 413  # too long
    assert post(c, b"abc").status_code == 400  # odd byte count
    assert post(c, pcm(), sr="100").status_code == 400
    assert c.post("/api/client-latency", json={"run_id": 1, "e2e_ms": -5}).status_code == 400


def test_rate_limit(client):
    c, m = client
    m.RATE_LIMIT = 2
    m._hits.clear()
    codes = [post(c, pcm(0.6)).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_no_audio_is_persisted(client, tmp_path):
    c, _ = client
    post(c, pcm())
    leftovers = [p for p in tmp_path.rglob("*") if p.suffix in {".wav", ".pcm", ".raw"}]
    assert not leftovers


def test_info_and_pages(client):
    c, _ = client
    m = c.get("/api/model").json()
    assert m["loaded"] and m["data_source"] == "synthetic"
    assert c.get("/").status_code == 200
    assert c.get("/model-card").status_code == 200
    assert c.get("/api/stats").json()["tract_version"]


def test_unreachable_database_falls_back_and_analysis_still_works(synthetic_artifacts, tmp_path, monkeypatch):
    monkeypatch.setenv("BTT_ARTIFACTS", str(synthetic_artifacts))
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@127.0.0.1:1/nope")  # nothing listens on port 1
    monkeypatch.setenv("BTT_LOCAL_DB", str(tmp_path / "fallback.db"))
    import importlib

    import app.main as m

    importlib.reload(m)
    with TestClient(m.app) as c:
        r = post(c, pcm())
        assert r.status_code == 200
        s = c.get("/api/stats").json()
        assert s["storage"]["persistent"] is False and "could not be reached" in s["storage"]["note"]
        assert s["stats"]["n"] == 1


def test_log_failure_never_breaks_analysis(client):
    c, m = client

    def boom(**_):
        raise RuntimeError("db down")

    m._state["store"].add = boom
    m._state["store"].summary = boom
    r = post(c, pcm())
    assert r.status_code == 200
    j = r.json()
    assert j["run_id"] is None and j["stats"] is None
    assert c.get("/api/stats").json()["stats"] is None


def test_storage_reported(client):
    c, _ = client
    s = c.get("/api/stats").json()["storage"]
    assert s["kind"] == "sqlite" and s["persistent"] is False
