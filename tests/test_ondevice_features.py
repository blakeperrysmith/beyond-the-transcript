"""The browser feature code must agree with btt/features.py. If it does not, the on-device model sees
different input from the one it was trained on and gives confident wrong answers, so this is the gate."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from btt.features import featurize

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")

TOL = 1e-3


def clip(secs, sr, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(int(secs * sr)) / sr
    f0 = 140 + 40 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    y = sum(np.sin(h * phase) / h for h in range(1, 12))
    y = y * (0.6 + 0.4 * np.sin(2 * np.pi * 3.5 * t)) * (t > 0.2) * (t < secs - 0.1)  # silence at both ends
    y = 0.2 * y / np.abs(y).max() + 0.002 * rng.standard_normal(t.size)
    return y.astype(np.float32)


def server_view(y, sr):
    """What the Python server hears: the browser's 16-bit conversion, then /32768."""
    y = y.astype(np.float64)  # JavaScript multiplies in double precision, so the test must too
    q = np.where(y < 0, y * 32768.0, y * 32767.0)
    q = np.trunc(np.clip(q, -32768, 32767)).astype(np.int16)
    return q.astype(np.float32) / 32768.0


CASES = [(1.7, 16000), (0.6, 16000), (2.4, 48000), (1.5, 44100), (5.0, 48000), (2.0, 22050), (3.0, 8000)]


def test_js_log_mel_matches_python(tmp_path):
    jobs = []
    for i, (secs, sr) in enumerate(CASES):
        f = tmp_path / f"c{i}.f32"
        clip(secs, sr, i).tofile(f)
        jobs.append({"file": str(f), "sr": sr})
    (tmp_path / "jobs.json").write_text(json.dumps(jobs))
    subprocess.run(["node", str(ROOT / "tests/js/featurize.mjs"), str(tmp_path / "jobs.json"), str(tmp_path / "out.json")], check=True, timeout=120)
    got = json.loads((tmp_path / "out.json").read_text())
    for (secs, sr), g, i in zip(CASES, got, range(len(CASES))):
        want = featurize(server_view(clip(secs, sr, i), sr), sr)[0]
        have = np.asarray(g["data"], dtype=np.float32).reshape(want.shape)
        err = np.abs(have - want)
        assert err.max() < TOL, f"{secs}s @ {sr} Hz: max abs log-mel error {err.max():.2e} (mean {err.mean():.2e})"
