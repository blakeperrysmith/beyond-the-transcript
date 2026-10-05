"""Hand-built prosody measurements with Praat (via Parselmouth).

These are not learned. They are the classical acoustic correlates of delivery
that a transcript discards: pitch level and range, how much of the clip is
voiced, speaking rate, pausing, loudness variation and spectral tilt.

Pitch values are speaker-relative. Compare two clips from the *same* speaker;
a difference between two different people mostly reflects their voices.
"""
from __future__ import annotations

import numpy as np
import parselmouth
from parselmouth.praat import call

SILENCE_DB = 25.0  # frames this far below the loudest frame count as silence
MIN_DIP_DB = 2.0  # de Jong & Wempe: minimum intensity dip between syllable peaks


def _syllable_nuclei(intensity, voiced_times: np.ndarray) -> int:
    """Count intensity peaks that sit in voiced frames (de Jong & Wempe, 2009)."""
    vals = intensity.values[0]
    times = intensity.xs()
    thr = vals.max() - SILENCE_DB
    peaks = []
    for i in range(1, len(vals) - 1):
        if vals[i] > vals[i - 1] and vals[i] >= vals[i + 1] and vals[i] > thr:
            peaks.append(i)
    kept = []
    for i in peaks:
        if kept:
            between = vals[kept[-1] : i + 1].min()
            if min(vals[i], vals[kept[-1]]) - between < MIN_DIP_DB:
                if vals[i] > vals[kept[-1]]:
                    kept[-1] = i
                continue
        kept.append(i)
    if voiced_times.size == 0:
        return 0
    n = 0
    for i in kept:
        j = int(np.argmin(np.abs(voiced_times - times[i])))
        if abs(voiced_times[j] - times[i]) <= 0.02:
            n += 1
    return n


def measure(y16: np.ndarray, contour: bool = False) -> dict:
    """Prosody summary for a mono 16 kHz waveform. Values are None when undefined.

    With ``contour=True`` also returns the pitch contour (Hz, 0 = unvoiced) every 20 ms.
    """
    y16 = np.asarray(y16, dtype=np.float64)
    out: dict = {
        "f0_median_hz": None,
        "f0_range_st": None,
        "voiced_frac": None,
        "syllable_rate": None,
        "pause_frac": None,
        "loudness_range_db": None,
        "spectral_tilt_db": None,
        "duration_s": float(y16.size / 16000.0),
    }
    if y16.size < 1600 or np.max(np.abs(y16)) < 1e-4:
        return out
    snd = parselmouth.Sound(y16, sampling_frequency=16000)
    pitch = snd.to_pitch_ac(time_step=0.01, pitch_floor=75.0, pitch_ceiling=500.0)
    f0 = pitch.selected_array["frequency"]
    ptimes = pitch.xs()
    voiced = f0 > 0
    if contour:
        out["f0_contour"] = [round(float(v), 1) for v in f0[::2]]
    out["voiced_frac"] = float(voiced.mean())
    if voiced.sum() < 5:
        return out
    f0v = f0[voiced]
    out["f0_median_hz"] = float(np.median(f0v))
    st = 12 * np.log2(f0v / np.median(f0v))
    out["f0_range_st"] = float(np.percentile(st, 90) - np.percentile(st, 10))

    intensity = snd.to_intensity(minimum_pitch=75.0, time_step=0.01)
    ivals = intensity.values[0]
    itimes = intensity.xs()
    t_lo, t_hi = ptimes[voiced][0], ptimes[voiced][-1]
    span = (itimes >= t_lo) & (itimes <= t_hi)
    if span.sum() > 3:
        seg = np.maximum(ivals[span], ivals.max() - 60.0)  # digital silence reads as -300 dB; floor it
        out["pause_frac"] = float(np.mean(seg < ivals.max() - SILENCE_DB))
        out["loudness_range_db"] = float(np.percentile(seg, 95) - np.percentile(seg, 5))
        dur = max(t_hi - t_lo, 0.2)
        out["syllable_rate"] = float(_syllable_nuclei(intensity, ptimes[voiced]) / dur)

    spec = snd.to_spectrum()
    low = call(spec, "Get band energy", 50.0, 1000.0)
    high = call(spec, "Get band energy", 1000.0, 5000.0)
    if low > 0 and high > 0:
        out["spectral_tilt_db"] = float(10 * np.log10(high / low))
    return out


# Plain-language labels and short explanations for the UI.
DESCRIPTIONS = {
    "f0_median_hz": ("Pitch (median)", "Hz", "Average voice pitch. Speaker-specific."),
    "f0_range_st": ("Pitch range", "semitones", "Spread between the 10th and 90th percentile of pitch. Livelier speech is wider."),
    "voiced_frac": ("Voiced share", "", "Fraction of the clip with a detectable pitch."),
    "syllable_rate": ("Speaking rate", "syll/s", "Estimated syllable nuclei per second of speech."),
    "pause_frac": ("Pausing", "", "Share of the speech span that is silent."),
    "loudness_range_db": ("Loudness range", "dB", "How much loudness varies inside the clip."),
    "spectral_tilt_db": ("Spectral tilt", "dB", "High-band vs low-band energy. Higher means brighter, more effortful voice."),
}
