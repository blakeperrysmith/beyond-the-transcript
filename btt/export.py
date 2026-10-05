"""ONNX export and the PyTorch-vs-tract parity check.

tract has an open issue where ONNX GRU/LSTM attributes it does not read
(custom activations, clip, input_forget) are ignored silently. A plain
``nn.GRU`` does not set them, but a numerical comparison is the only honest way
to know the exported graph computes what PyTorch trained, so it is a test and
not an assumption.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .features import N_MELS, WIN_FRAMES
from .model import DeliveryNet


def export_onnx(model: DeliveryNet, path: str | Path, opset: int = 17) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval().cpu()
    dummy = torch.zeros(1, 1, N_MELS, WIN_FRAMES)
    # The legacy TorchScript exporter maps nn.GRU to a single ONNX GRU op, which
    # is the form tract handles best.
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["logmel"],
        output_names=["logits"],
        opset_version=opset,
        dynamo=False,
    )
    return path


def parity_check(
    model: DeliveryNet,
    onnx_path: str | Path,
    inputs: np.ndarray,
    atol: float = 1e-3,
) -> dict:
    """Compare PyTorch logits with tract logits on ``inputs`` (N, 1, 64, 300)."""
    from .tract_runtime import TractModel

    model = model.eval().cpu()
    tm = TractModel(onnx_path)
    diffs, agree = [], 0
    with torch.no_grad():
        for x in inputs:
            xt = torch.from_numpy(x[None].astype(np.float32))
            ref = model(xt).numpy().reshape(-1)
            got = tm.run(x[None].astype(np.float32))
            diffs.append(float(np.max(np.abs(ref - got))))
            agree += int(np.argmax(ref) == np.argmax(got))
    return {
        "n": int(len(inputs)),
        "max_abs_logit_diff": float(max(diffs)),
        "mean_abs_logit_diff": float(np.mean(diffs)),
        "argmax_agreement": agree / max(len(inputs), 1),
        "atol": atol,
        "passed": bool(max(diffs) <= atol and agree == len(inputs)),
    }


def benchmark(model: DeliveryNet, onnx_path: str | Path, n: int = 200) -> dict:
    """Median and p95 single-clip latency, PyTorch eager vs tract, on this CPU."""
    from .tract_runtime import TractModel

    model = model.eval().cpu()
    torch.set_num_threads(1)
    x = np.random.default_rng(0).normal(-6, 2, size=(1, 1, N_MELS, WIN_FRAMES)).astype(np.float32)
    xt = torch.from_numpy(x)
    tm = TractModel(onnx_path)

    def timed(fn):
        for _ in range(10):
            fn()
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1000)
        return {"p50_ms": float(np.percentile(ts, 50)), "p95_ms": float(np.percentile(ts, 95))}

    with torch.no_grad():
        eager = timed(lambda: model(xt))
    return {"pytorch_eager_1thread": eager, "tract": timed(lambda: tm.run(x)), "n": n}
