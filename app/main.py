"""FastAPI server: raw PCM in, delivery probabilities, prosody and timings out.

Audio is decoded in memory and discarded. The only thing written to disk or the
database is a row of timings (see store.py).

    uvicorn app.main:app --host :: --port $PORT
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import mimetypes
import os
import platform
import threading
from contextlib import asynccontextmanager
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from scipy.special import softmax

from btt import __version__
from btt import mdlite
from btt.modelcard import render_model_card
from btt.features import MAX_SECONDS, fit_window, log_mel, to_16k
from btt.prosody import DESCRIPTIONS, measure
from btt.tract_runtime import TractModel, tract_version

from .store import open_store

log = logging.getLogger("btt.app")
mimetypes.add_type("application/wasm", ".wasm")  # streaming WebAssembly compile needs this type
mimetypes.add_type("text/javascript", ".mjs")
ROOT = Path(__file__).resolve().parent
ARTIFACTS = Path(os.environ.get("BTT_ARTIFACTS", ROOT.parent / "artifacts"))
STATIC = ROOT / "static"
MAX_BODY_BYTES = 2 * 1024 * 1024
MIN_SECONDS = 0.4
RATE_LIMIT = int(os.environ.get("BTT_RATE_LIMIT_PER_MIN", "30"))

@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_model()
    default_db = Path(os.environ.get("BTT_LOCAL_DB", ROOT.parent / "latency.db"))
    _state["store"] = open_store(os.environ.get("DATABASE_URL") or f"sqlite:///{default_db}", str(default_db))
    yield


app = FastAPI(title="Beyond the transcript", version=__version__, docs_url=None, redoc_url=None, lifespan=lifespan)

_compute_lock = threading.Lock()  # Praat and the tract runnable are guarded; numpy parts run freely
_state: dict = {"model": None, "meta": None, "metrics": None, "cross": None, "store": None, "served": 0}
_hits: dict[str, deque] = defaultdict(deque)


def _cpu_description() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return f"{line.split(':', 1)[1].strip()} x{os.cpu_count()}"
    except OSError:
        pass
    return f"{platform.processor() or platform.machine()} x{os.cpu_count()}"


def load_model(artifacts: Path | None = None) -> bool:
    d = Path(artifacts or ARTIFACTS)
    onnx, meta = d / "model.onnx", d / "meta.json"
    if not onnx.exists() or not meta.exists():
        _state["model"] = None
        return False
    _state["meta"] = json.loads(meta.read_text())
    mp = d / "metrics.json"
    _state["metrics"] = json.loads(mp.read_text()) if mp.exists() else {}
    cp = d / "metrics_cross_corpus.json"
    try:
        _state["cross"] = json.loads(cp.read_text()) if cp.exists() else None
    except ValueError:
        _state["cross"] = None
    _state["model"] = TractModel(onnx)
    return True


def _client_key(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    ip = fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "")
    day = time.strftime("%Y-%m-%d")
    return hashlib.sha256(f"{day}|{ip}".encode()).hexdigest()[:16]  # held in memory only


def _rate_limit(request: Request) -> None:
    if RATE_LIMIT <= 0:
        return
    key, now = _client_key(request), time.time()
    q = _hits[key]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        raise HTTPException(429, f"Slow down: {RATE_LIMIT} analyses per minute per visitor.")
    q.append(now)
    if len(_hits) > 5000:  # keep memory bounded
        for k in [k for k, v in _hits.items() if not v or now - v[-1] > 60]:
            _hits.pop(k, None)


def _mel_preview(feat: np.ndarray) -> str:
    """(64, 300) log-mel -> 32x100 uint8, base64. For drawing only."""
    f = feat.reshape(32, 2, 100, 3).mean(axis=(1, 3))[::-1]  # low frequencies at the bottom
    lo, hi = np.percentile(f, 2), np.percentile(f, 99.5)
    u8 = np.clip((f - lo) / max(hi - lo, 1e-6), 0, 1) * 255
    return base64.b64encode(u8.astype(np.uint8).tobytes()).decode()


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "model_loaded": _state["model"] is not None}


def _json_safe(o):
    """Starlette refuses NaN and Infinity. Bootstrap intervals are NaN when a group has one speaker."""
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    return o


def _ondevice_kb() -> int | None:
    """Download size of the on-device engine: the runtime files plus the model. None when it is not installed."""
    d = STATIC / "ondevice"
    if not (d / "manifest.json").exists():
        return None
    total = sum(p.stat().st_size for p in d.rglob("*") if p.is_file() and p.name not in {"manifest.json", ".gitkeep"} and not p.name.startswith("LICENSE"))
    model = ARTIFACTS / "model.onnx"
    if model.exists():
        total += model.stat().st_size
    return int(round(total / 1024))


@app.get("/api/model")
def model_info() -> dict:
    meta, metrics = _state["meta"], _state["metrics"] or {}
    if meta is None:
        return {"loaded": False}
    t = metrics.get("test", {})
    return _json_safe({
        "loaded": True,
        "classes": meta["classes"],
        "data_source": meta["data_source"],
        "n_params": meta["n_params"],
        "datasets": meta["datasets"],
        "abstain_threshold": meta["abstain_threshold"],
        "temperature": meta["temperature"],
        "test": {
            "n_speakers": t.get("n_speakers"),
            "n_clips": t.get("n_clips"),
            "accuracy": t.get("accuracy"),
            "accuracy_ci95": t.get("accuracy_ci95_speaker_bootstrap"),
            "chance": t.get("chance_accuracy"),
            "coverage_at_threshold": t.get("coverage_at_threshold"),
            "accuracy_when_answering": t.get("accuracy_when_answering"),
            "groups": {k: v for k, v in (t.get("groups") or {}).items() if k != "sentence"},
            "per_speaker": t.get("per_speaker"),
            "cross_corpus": _state["cross"],
        },
        "ondevice_kb": _ondevice_kb(),
        "tract_version": tract_version(),
        "hardware": _cpu_description(),
        "prosody_descriptions": DESCRIPTIONS,
    })


@app.post("/api/analyze")
async def analyze(request: Request) -> JSONResponse:
    t_req = time.perf_counter()
    if _state["model"] is None:
        raise HTTPException(503, "No model is loaded. Train one and place model.onnx and meta.json in artifacts/.")
    _rate_limit(request)
    body = await request.body()
    if len(body) > MAX_BODY_BYTES:
        raise HTTPException(413, "Audio too large. Send at most 8 seconds of 16-bit mono PCM.")
    if len(body) % 2:
        raise HTTPException(400, "Body must be 16-bit little-endian PCM.")
    try:
        sr = int(request.headers.get("x-sample-rate", "0"))
    except ValueError:
        sr = 0
    if not 8000 <= sr <= 96000:
        raise HTTPException(400, "Set the X-Sample-Rate header (8000 to 96000).")
    y = np.frombuffer(body, dtype="<i2").astype(np.float32) / 32768.0
    dur = y.size / sr
    if dur < MIN_SECONDS:
        raise HTTPException(400, "Recording is too short. Speak for at least half a second.")
    if dur > MAX_SECONDS:
        raise HTTPException(413, f"Recording is {dur:.1f} s. The limit is {MAX_SECONDS:.0f} s.")
    source = request.headers.get("x-source", "live")
    source = source if source in ("live", "sample") else "live"

    meta, store = _state["meta"], _state["store"]
    t0 = time.perf_counter()
    y16 = to_16k(y, sr)
    feat_full = log_mel(y16)
    feat = fit_window(feat_full)
    t1 = time.perf_counter()
    with _compute_lock:
        t2 = time.perf_counter()
        pros = measure(y16, contour=True)
        t3 = time.perf_counter()
        logits = _state["model"].run(feat[None, None])
        t4 = time.perf_counter()
        cold = _state["served"] == 0
        _state["served"] += 1
    probs = softmax(logits / meta["temperature"])
    top = int(np.argmax(probs))
    conf = float(probs[top])
    abstained = conf < meta["abstain_threshold"]
    t_end = time.perf_counter()

    timings = {
        "features_ms": (t1 - t0) * 1000,
        "prosody_ms": (t3 - t2) * 1000,
        "tract_ms": (t4 - t3) * 1000,
        "server_ms": (t_end - t_req) * 1000,
        "cold": cold,
    }
    run_id = None
    try:
        run_id = store.add(source=source, model_version=meta["version"], cold=cold,
                           features_ms=timings["features_ms"], prosody_ms=timings["prosody_ms"],
                           tract_ms=timings["tract_ms"], server_ms=timings["server_ms"])
    except Exception as e:  # the log must never break an analysis
        log.warning("could not log run: %s", type(e).__name__)
    return JSONResponse(
        {
            "run_id": run_id,
            "label": None if abstained else meta["classes"][top],
            "abstained": abstained,
            "confidence": conf,
            "probs": {c: float(p) for c, p in zip(meta["classes"], probs)},
            "prosody": pros,
            "mel": _mel_preview(feat),
            "clip_seconds": float(min(dur, 3.0)),
            "audio_seconds": float(dur),
            "was_cropped": bool(feat_full.shape[1] > feat.shape[1]),
            "timings": timings,
            "stats": _safe_stats(),
        }
    )


def _safe_stats(ttl: float = 2.0):
    try:
        return _state["store"].summary(ttl=ttl)
    except Exception as e:  # noqa: BLE001
        log.warning("could not read stats: %s", type(e).__name__)
        return None


def _storage_info() -> dict:
    st = _state["store"]
    return {"kind": st.kind, "persistent": st.persistent, "note": st.note}


class ClientLatency(BaseModel):
    run_id: int
    e2e_ms: float


@app.post("/api/client-latency")
def client_latency(body: ClientLatency) -> dict:
    if not (0 < body.e2e_ms < 60000):
        raise HTTPException(400, "Implausible round-trip time.")
    try:
        _state["store"].set_e2e(body.run_id, body.e2e_ms)
    except Exception as e:  # noqa: BLE001
        log.warning("could not log round trip: %s", type(e).__name__)
    return {"ok": True, "stats": _safe_stats(ttl=0), "storage": _storage_info()}


@app.get("/api/stats")
def stats() -> dict:
    return {"stats": _safe_stats(), "storage": _storage_info(), "hardware": _cpu_description(), "tract_version": tract_version()}


@app.get("/model.onnx")
def model_file() -> FileResponse:
    """The same file the server runs. The on-device engine downloads it, so it is public like the rest of the repo."""
    f = ARTIFACTS / "model.onnx"
    if not f.exists():
        raise HTTPException(404, "No model yet.")
    return FileResponse(f, media_type="application/octet-stream")


def _card_markdown() -> str:
    """Rebuilt from the metrics files on every request, so the card always matches the numbers the page shows."""
    if _state["meta"] and _state["metrics"] and "test" in _state["metrics"]:
        try:
            return render_model_card(_state["meta"], _state["metrics"], _state["cross"])
        except Exception as e:  # noqa: BLE001
            log.warning("could not render model card from metrics: %s", type(e).__name__)
    card = ARTIFACTS / "MODEL_CARD.md"
    if card.exists():
        return card.read_text()
    raise HTTPException(404, "No model card yet.")


@app.get("/model-card")
def model_card() -> HTMLResponse:
    return HTMLResponse(mdlite.PAGE.format(body=mdlite.to_html(_card_markdown())))


@app.get("/model-card.md")
def model_card_md() -> PlainTextResponse:
    return PlainTextResponse(_card_markdown(), media_type="text/markdown; charset=utf-8")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
