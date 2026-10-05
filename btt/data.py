"""Corpus scanning, speaker-disjoint splits and feature caching.

Both corpora encode their labels in file names, so scanning needs no metadata
files except the optional CREMA-D demographics CSV. Demographics are used only
for offline, per-group evaluation. They are never a model input or output.
"""
from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from .features import SR, featurize
from .labels import (
    CLASS_TO_ID,
    CREMAD_EMOTION,
    CREMAD_SENTENCE,
    RAVDESS_EMOTION,
    RAVDESS_STATEMENT,
)


@dataclass
class Item:
    path: str
    dataset: str  # "cremad" | "ravdess"
    speaker: str  # globally unique, e.g. "cremad_1001"
    label: str
    sentence_id: str
    sentence: str
    sex: str = "unknown"
    age: int | None = None
    race: str = "unknown"
    ethnicity: str = "unknown"

    def to_dict(self) -> dict:
        return asdict(self)


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Mono float32 samples in [-1, 1] and the native sample rate."""
    sr, y = wavfile.read(str(path))
    if y.dtype == np.int16:
        y = y.astype(np.float32) / 32768.0
    elif y.dtype == np.int32:
        y = y.astype(np.float32) / 2147483648.0
    elif y.dtype == np.uint8:
        y = (y.astype(np.float32) - 128.0) / 128.0
    else:
        y = y.astype(np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y, int(sr)


_RAVDESS_RE = re.compile(r"^(\d\d)-(\d\d)-(\d\d)-(\d\d)-(\d\d)-(\d\d)-(\d\d)\.wav$")
_CREMAD_RE = re.compile(r"^(\d{4})_([A-Z]{3})_([A-Z]{3})_([A-Z]{2})\.wav$")


def parse_ravdess(path: str) -> Item | None:
    m = _RAVDESS_RE.match(os.path.basename(path))
    if not m:
        return None
    _modality, vocal, emo, _intensity, stmt, _rep, actor = m.groups()
    if vocal != "01" or emo not in RAVDESS_EMOTION:  # speech only; calm/surprised dropped
        return None
    return Item(
        path=path,
        dataset="ravdess",
        speaker=f"ravdess_{actor}",
        label=RAVDESS_EMOTION[emo],
        sentence_id=f"ravdess_{stmt}",
        sentence=RAVDESS_STATEMENT.get(stmt, ""),
        sex="female" if int(actor) % 2 == 0 else "male",  # documented by the dataset
    )


def parse_cremad(path: str, demo: dict[str, dict] | None = None) -> Item | None:
    m = _CREMAD_RE.match(os.path.basename(path))
    if not m:
        return None
    actor, sent, emo, _level = m.groups()
    if emo not in CREMAD_EMOTION:
        return None
    d = (demo or {}).get(actor, {})
    return Item(
        path=path,
        dataset="cremad",
        speaker=f"cremad_{actor}",
        label=CREMAD_EMOTION[emo],
        sentence_id=f"cremad_{sent}",
        sentence=CREMAD_SENTENCE.get(sent, ""),
        sex=str(d.get("sex", "unknown")).lower(),
        age=d.get("age"),
        race=str(d.get("race", "unknown")),
        ethnicity=str(d.get("ethnicity", "unknown")),
    )


def load_cremad_demographics(csv_path: str | Path | None) -> dict[str, dict]:
    """Parse CREMA-D's VideoDemographics.csv (ActorID, Age, Sex, Race, Ethnicity)."""
    if not csv_path or not Path(csv_path).exists():
        return {}
    out: dict[str, dict] = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            actor = str(row.get("ActorID", "")).strip()
            if not actor:
                continue
            try:
                age = int(float(row.get("Age", "")))
            except ValueError:
                age = None
            out[actor] = {
                "age": age,
                "sex": row.get("Sex", "unknown"),
                "race": row.get("Race", "unknown"),
                "ethnicity": row.get("Ethnicity", "unknown"),
            }
    return out


def scan(
    ravdess_dir: str | Path | None = None,
    cremad_dir: str | Path | None = None,
    demographics_csv: str | Path | None = None,
) -> list[Item]:
    """Walk the corpus folders (layout does not matter) and parse file names."""
    demo = load_cremad_demographics(demographics_csv)
    items: list[Item] = []
    for root, parser in ((ravdess_dir, parse_ravdess), (cremad_dir, None)):
        if not root:
            continue
        for dirpath, _dirs, files in os.walk(root):
            for fn in sorted(files):
                if not fn.lower().endswith(".wav"):
                    continue
                p = os.path.join(dirpath, fn)
                it = parser(p) if parser else parse_cremad(p, demo)
                if it is not None:
                    items.append(it)
    items.sort(key=lambda i: (i.dataset, i.speaker, i.path))
    return items


def speaker_split(
    items: list[Item],
    seed: int = 0,
    fracs: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> dict[str, list[str]]:
    """Speaker-disjoint train/val/test, done within each corpus so both appear
    in every split. No speaker ever appears in more than one split."""
    rng = np.random.default_rng(seed)
    out = {"train": [], "val": [], "test": []}
    for ds in sorted({i.dataset for i in items}):
        spk = sorted({i.speaker for i in items if i.dataset == ds})
        rng.shuffle(spk)
        n = len(spk)
        n_val = max(1, round(n * fracs[1])) if n >= 3 else 0
        n_test = max(1, round(n * fracs[2])) if n >= 3 else 0
        out["test"] += spk[:n_test]
        out["val"] += spk[n_test : n_test + n_val]
        out["train"] += spk[n_test + n_val :]
    assert not (set(out["train"]) & set(out["val"]) | set(out["train"]) & set(out["test"]) | set(out["val"]) & set(out["test"]))
    return out


def subset(items: list[Item], speakers: list[str]) -> list[Item]:
    keep = set(speakers)
    return [i for i in items if i.speaker in keep]


def build_features(items: list[Item], cache: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Features (N, 1, 64, 300) float16 and labels (N,) int64, optionally cached."""
    if cache and Path(cache).exists():
        z = np.load(cache, allow_pickle=False)
        if len(z["y"]) == len(items):
            return z["x"], z["y"]
    xs = np.zeros((len(items), 1, 64, 300), dtype=np.float16)
    ys = np.zeros(len(items), dtype=np.int64)
    for k, it in enumerate(items):
        y, sr = read_wav(it.path)
        xs[k] = featurize(y, sr).astype(np.float16)
        ys[k] = CLASS_TO_ID[it.label]
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, x=xs, y=ys)
    return xs, ys


# ----------------------------------------------------------------------------
# Synthetic corpus: used ONLY to test the pipeline end to end where the real
# datasets cannot be downloaded. Anything trained on it is marked as such in
# meta.json and the app shows a banner.
# ----------------------------------------------------------------------------

_SYN_STYLE = {
    # f0 scale, f0 range (semitones), rate Hz of syllable pulses, tilt (dB/oct), amp, tremolo, noise
    "neutral": (1.00, 2.0, 4.0, -9.0, 0.5, 0.0, 0.01),
    "happy": (1.25, 8.0, 5.0, -5.0, 0.7, 0.0, 0.01),
    "sad": (0.85, 1.0, 2.5, -14.0, 0.3, 0.0, 0.01),
    "angry": (1.10, 5.0, 5.5, -3.0, 0.9, 0.0, 0.02),
    "fearful": (1.35, 6.0, 6.0, -8.0, 0.5, 9.0, 0.01),
    "disgust": (0.90, 2.0, 3.0, -11.0, 0.5, 0.0, 0.03),
}


def synth_clip(label: str, base_f0: float, seed: int, sr: int = SR, dur: float = 2.2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    f0s, rng_st, rate, tilt, amp, trem, noise = _SYN_STYLE[label]
    t = np.arange(int(sr * dur)) / sr
    contour_st = rng_st * np.sin(2 * np.pi * (0.7 + 0.3 * rng.random()) * t + rng.random() * 6.28)
    f0 = base_f0 * f0s * 2 ** (contour_st / 12)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    y = np.zeros_like(t)
    for h in range(1, 20):
        y += (10 ** (tilt * np.log2(h) / 20)) * np.sin(h * phase)
    env = 0.5 * (1 + np.sin(2 * np.pi * rate * t + rng.random() * 6.28))
    env = 0.25 + 0.75 * env
    if trem:
        env = env * (0.7 + 0.3 * np.sin(2 * np.pi * trem * t))
    fade = np.minimum(1, np.minimum(t, dur - t) / 0.1)
    y = amp * y * env * fade + noise * rng.normal(size=t.size)
    return (y / (np.max(np.abs(y)) + 1e-9) * 0.6).astype(np.float32)


def write_synthetic_corpus(root: str | Path, n_speakers: int = 12, seed: int = 0) -> tuple[Path, Path]:
    """Write small CREMA-D-style and RAVDESS-style folders of synthetic wavs."""
    root = Path(root)
    cremad = root / "cremad" / "AudioWAV"
    ravdess = root / "ravdess"
    cremad.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    emo_codes = {v: k for k, v in CREMAD_EMOTION.items()}
    rav_codes = {v: k for k, v in RAVDESS_EMOTION.items()}
    sents = ["IEO", "TIE", "DFA"]
    for s in range(n_speakers):
        actor = 1001 + s
        f0 = float(rng.uniform(95, 230))
        for sent in sents:
            for label in _SYN_STYLE:
                seed_i = int(rng.integers(1 << 30))
                y = synth_clip(label, f0, seed_i)
                fn = f"{actor}_{sent}_{emo_codes[label]}_XX.wav"
                wavfile.write(cremad / fn, SR, (y * 32767).astype(np.int16))
    for a in range(1, n_speakers // 2 + 1):
        d = ravdess / f"Actor_{a:02d}"
        d.mkdir(parents=True, exist_ok=True)
        f0 = float(rng.uniform(95, 230))
        for stmt in ("01", "02"):
            for label in _SYN_STYLE:
                y = synth_clip(label, f0, int(rng.integers(1 << 30)), sr=48000)
                fn = f"03-01-{rav_codes[label]}-01-{stmt}-01-{a:02d}.wav"
                wavfile.write(d / fn, 48000, (y * 32767).astype(np.int16))
    return ravdess, root / "cremad"
