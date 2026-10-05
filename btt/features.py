"""Audio front end shared by training, export tests and the serving app.

Everything here is plain numpy/scipy so the server never needs torch. The same
function produces the training features and the serving features, which removes
one common source of train/serve skew.

Pipeline: resample to 16 kHz -> RMS-normalise the waveform -> STFT (32 ms
window, 10 ms hop) -> 64 mel bands -> log -> fixed 3 s window.
"""
from __future__ import annotations

from functools import lru_cache
from math import gcd

import numpy as np
from scipy.signal import resample_poly

SR = 16000
N_FFT = 512
HOP = 160
N_MELS = 64
FMIN = 50.0
FMAX = 7600.0
WIN_FRAMES = 300  # 3.0 s at a 10 ms hop
LOG_EPS = 1e-6
FLOOR = float(np.log(LOG_EPS))  # value used to pad short clips
MAX_SECONDS = 8.0  # the server refuses anything longer


def to_16k(y: np.ndarray, sr: int) -> np.ndarray:
    """Mono float32 at ``sr`` -> mono float32 at 16 kHz."""
    y = np.asarray(y, dtype=np.float32)
    if sr == SR:
        return y
    g = gcd(SR, int(sr))
    return resample_poly(y, SR // g, int(sr) // g).astype(np.float32)


def rms_normalise(y: np.ndarray, target: float = 0.1) -> np.ndarray:
    """Remove recording gain. Dynamics inside the utterance are untouched."""
    rms = float(np.sqrt(np.mean(np.square(y, dtype=np.float64)))) if y.size else 0.0
    if rms < 1e-8:
        return y.astype(np.float32)
    return (y * (target / rms)).astype(np.float32)


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


@lru_cache(maxsize=1)
def mel_filterbank() -> np.ndarray:
    """(N_MELS, N_FFT//2+1) triangular filterbank, HTK mel scale."""
    n_bins = N_FFT // 2 + 1
    mel_pts = np.linspace(_hz_to_mel(FMIN), _hz_to_mel(FMAX), N_MELS + 2)
    hz_pts = _mel_to_hz(mel_pts)
    bin_freqs = np.linspace(0, SR / 2, n_bins)
    fb = np.zeros((N_MELS, n_bins), dtype=np.float64)
    for i in range(N_MELS):
        lo, ctr, hi = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
        up = (bin_freqs - lo) / max(ctr - lo, 1e-9)
        down = (hi - bin_freqs) / max(hi - ctr, 1e-9)
        fb[i] = np.maximum(0.0, np.minimum(up, down))
    # Area-normalise so wide high bands do not dominate.
    fb /= np.maximum(fb.sum(axis=1, keepdims=True), 1e-9)
    return fb.astype(np.float32)


@lru_cache(maxsize=1)
def _window() -> np.ndarray:
    return np.hanning(N_FFT + 1)[:-1].astype(np.float32)


def log_mel(y16: np.ndarray) -> np.ndarray:
    """Waveform at 16 kHz -> (N_MELS, T) log-mel, T = 1 + len // HOP."""
    y16 = rms_normalise(np.asarray(y16, dtype=np.float32))
    if y16.size < N_FFT:
        y16 = np.pad(y16, (0, N_FFT - y16.size))
    pad = N_FFT // 2
    yp = np.pad(y16, (pad, pad), mode="reflect")
    n_frames = 1 + (yp.size - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    frames = yp[idx] * _window()[None, :]
    power = np.abs(np.fft.rfft(frames, n=N_FFT, axis=1)) ** 2  # (T, bins)
    mel = power @ mel_filterbank().T  # (T, N_MELS)
    return np.log(mel + LOG_EPS).T.astype(np.float32)


def fit_window(feat: np.ndarray, n: int = WIN_FRAMES) -> np.ndarray:
    """Return exactly ``n`` frames.

    Longer clips: the most energetic ``n``-frame window (so leading/trailing
    silence in a live recording is dropped). Shorter clips: padded at the end
    with the log-floor, which is also what training does.
    """
    t = feat.shape[1]
    if t >= n:
        if t == n:
            return feat
        frame_energy = feat.mean(axis=0)
        csum = np.concatenate([[0.0], np.cumsum(frame_energy, dtype=np.float64)])
        sums = csum[n:] - csum[:-n]
        start = int(np.argmax(sums))
        return feat[:, start : start + n]
    out = np.full((feat.shape[0], n), FLOOR, dtype=np.float32)
    out[:, :t] = feat
    return out


def featurize(y: np.ndarray, sr: int) -> np.ndarray:
    """Mono waveform at any sample rate -> (1, N_MELS, WIN_FRAMES) float32."""
    y16 = to_16k(y, sr)
    return fit_window(log_mel(y16))[None, :, :]
