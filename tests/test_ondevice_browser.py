"""End to end: the model running as WebAssembly in a real browser must agree with the server's tract output.

Skipped when Playwright or a Chromium build is not available. Needs the same synthetic artifacts as the other tests.
"""
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

playwright = pytest.importorskip("playwright.sync_api")
ROOT = Path(__file__).resolve().parent.parent


def _have_chromium():
    try:
        with playwright.sync_playwright() as pw:
            return Path(pw.chromium.executable_path).exists()
    except Exception:
        return False


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture()
def server(synthetic_artifacts, tmp_path):
    port = _free_port()
    env = {**os.environ, "BTT_ARTIFACTS": str(synthetic_artifacts), "DATABASE_URL": f"sqlite:///{tmp_path/'b.db'}", "BTT_RATE_LIMIT_PER_MIN": "0"}
    p = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(port), "--log-level", "warning"], cwd=ROOT, env=env)
    import urllib.request

    for _ in range(100):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=1)
            break
        except Exception:
            time.sleep(0.2)
    yield f"http://127.0.0.1:{port}"
    p.terminate()
    p.wait(10)


def voice(secs, sr):
    t = np.arange(int(secs * sr)) / sr
    f0 = 150 + 30 * np.sin(2 * np.pi * 0.8 * t)
    y = sum(np.sin(h * 2 * np.pi * np.cumsum(f0) / sr) / h for h in range(1, 10)) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
    return (0.25 * y / np.abs(y).max()).astype(np.float32)


@pytest.mark.skipif(not _have_chromium(), reason="no Chromium for Playwright")
@pytest.mark.parametrize("sr", [48000, 44100, 16000])
def test_wasm_matches_server(server, sr):
    import json
    import urllib.request

    pcm = voice(2.2, sr)
    i16 = np.where(pcm < 0, pcm.astype(np.float64) * 32768, pcm.astype(np.float64) * 32767).astype(np.int16)
    req = urllib.request.Request(server + "/api/analyze", data=i16.tobytes(), headers={"X-Sample-Rate": str(sr), "Content-Type": "application/octet-stream"})
    want = json.loads(urllib.request.urlopen(req).read())
    info = json.loads(urllib.request.urlopen(server + "/api/model").read())
    classes, temp = info["classes"], info["temperature"]

    with playwright.sync_playwright() as pw:
        b = pw.chromium.launch()
        page = b.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(server + "/")
        got = page.evaluate(
            """async ([samples, sr]) => {
              const mod = await import('/static/ondevice/ondevice.js');
              const eng = await mod.load({ modelUrl: '/model.onnx' });
              return await eng.run(Float32Array.from(samples), sr);
            }""",
            [pcm.tolist(), sr],
        )
        b.close()
    assert not errors, errors
    z = np.array(got["logits"]) / temp
    z = np.exp(z - z.max())
    p = z / z.sum()
    got_probs = dict(zip(classes, p))
    for c in classes:
        assert abs(got_probs[c] - want["probs"][c]) < 2e-3, (c, got_probs[c], want["probs"][c])
    assert got["model_ms"] > 0 and got["features_ms"] > 0
