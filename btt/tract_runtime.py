"""Thin wrapper over the tract Python bindings (Sonos' inference engine).

The model is exported with a fixed input shape (1, 1, N_MELS, WIN_FRAMES), so
tract can optimise the whole graph up front and each request is one ``run``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tract

from .features import N_MELS, WIN_FRAMES


class TractModel:
    def __init__(self, onnx_path: str | Path):
        self.path = Path(onnx_path)
        inference = tract.onnx().load(str(self.path))
        inference.set_input_fact(0, f"1,1,{N_MELS},{WIN_FRAMES},f32")
        self._runnable = inference.into_model().into_runnable()
        self.version = getattr(tract, "version", lambda: "unknown")
        # One throwaway run so the first real request is not charged for lazy init.
        self.run(np.zeros((1, 1, N_MELS, WIN_FRAMES), dtype=np.float32))

    def run(self, feats: np.ndarray) -> np.ndarray:
        """feats: (1, 1, N_MELS, WIN_FRAMES) float32 -> logits (n_classes,)."""
        x = np.ascontiguousarray(feats, dtype=np.float32)
        out = self._runnable.run([x])[0].to_numpy()
        return np.asarray(out, dtype=np.float32).reshape(-1)


def tract_version() -> str:
    try:
        from importlib.metadata import version

        return version("tract")
    except Exception:  # pragma: no cover
        return "unknown"
