"""Build the click-to-play sample pairs for the demo.

Each pair is one speaker saying one sentence two ways. Clips come only from
speakers in the model's held-out test split, and are chosen from the labels
alone, before looking at what the model says about them, so the demo shows the
model's mistakes as well as its hits.

    python scripts/make_samples.py --ravdess RAV --cremad CREMA --demographics Demo.csv \
        --split artifacts/split.json --out app/static/samples
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from btt.data import read_wav, scan  # noqa: E402
from btt.features import SR, to_16k  # noqa: E402
from btt.labels import LICENSES  # noqa: E402

# Contrasts that are easy to hear, in order of preference.
CONTRASTS = [("neutral", "angry"), ("happy", "sad"), ("neutral", "sad"), ("neutral", "happy"), ("angry", "sad"), ("neutral", "fearful")]
MAX_S = 3.0
TRIM_DB = 35.0
PAD_S = 0.12


def trim(y16: np.ndarray) -> np.ndarray:
    """Cut leading and trailing silence, then keep at most MAX_S."""
    frame = int(0.01 * SR)
    n = len(y16) // frame
    if n == 0:
        return y16
    rms = np.sqrt((y16[: n * frame].reshape(n, frame) ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms / rms.max())
    live = np.where(db > -TRIM_DB)[0]
    if live.size == 0:
        return y16[: int(MAX_S * SR)]
    lo = max(0, live[0] * frame - int(PAD_S * SR))
    hi = min(len(y16), (live[-1] + 1) * frame + int(PAD_S * SR))
    return y16[lo:hi][: int(MAX_S * SR)]


def write_clip(y: np.ndarray, path: Path) -> None:
    peak = float(np.max(np.abs(y))) or 1.0
    y = y * min(1.0, 0.9 / peak)  # never amplify, only prevent clipping
    wavfile.write(str(path), SR, (np.clip(y, -1, 1) * 32767).astype(np.int16))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ravdess")
    ap.add_argument("--cremad")
    ap.add_argument("--demographics")
    ap.add_argument("--split", required=True, help="split.json written by training")
    ap.add_argument("--out", default="app/static/samples")
    ap.add_argument("--n", type=int, default=6, help="number of pairs")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    test = set(json.loads(Path(args.split).read_text())["test"])
    items = [i for i in scan(args.ravdess, args.cremad, args.demographics) if i.speaker in test]
    by = defaultdict(dict)  # (speaker, sentence_id) -> label -> item
    for it in items:
        by[(it.speaker, it.sentence_id)].setdefault(it.label, it)

    rng = random.Random(args.seed)
    cands = []
    for (spk, sid), d in sorted(by.items()):
        for rank, (x, y) in enumerate(CONTRASTS):
            if x in d and y in d:
                cands.append((rank, rng.random(), spk, sid, d[x], d[y]))
    cands.sort(key=lambda c: (c[0], c[1]))

    chosen, used_spk, used_pair, per_corpus = [], set(), set(), defaultdict(int)
    for _, _, spk, sid, a, b in cands:
        if spk in used_spk or (a.label, b.label) in used_pair:
            continue
        if per_corpus[a.dataset] >= (args.n + 1) // 2:
            continue
        chosen.append((a, b))
        used_spk.add(spk); used_pair.add((a.label, b.label)); per_corpus[a.dataset] += 1
        if len(chosen) == args.n:
            break
    if not chosen:
        raise SystemExit("No matched pairs found in the test split.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("pair*.wav"):
        old.unlink()
    pairs, corpora = [], set()
    for n, (a, b) in enumerate(chosen, 1):
        sides = {}
        for tag, it in (("a", a), ("b", b)):
            y, sr = read_wav(it.path)
            fname = f"pair{n:02d}_{tag}.wav"
            write_clip(trim(to_16k(y, sr)), out / fname)
            name = "CREMA-D" if it.dataset == "cremad" else "RAVDESS"
            sides[tag] = {"file": fname, "emotion": it.label, "speaker": it.speaker.split("_", 1)[1], "corpus": name,
                          "caption": f"{it.label.capitalize()}, {name} actor {it.speaker.split('_', 1)[1]}"}
            corpora.add(it.dataset)
        pairs.append({"id": f"pair{n:02d}", "sentence": a.sentence, "title": a.sentence.rstrip(".") + f" ({a.label} / {b.label})", "a": sides["a"], "b": sides["b"]})

    attribution = ("Clips are trimmed to at most 3 seconds and come from actors the model never saw in training. "
                   "Pairs were picked from the labels alone, so the model gets some of them wrong. "
                   + "; ".join(LICENSES[c] for c in sorted(corpora)) + ".")
    (out / "manifest.json").write_text(json.dumps({"attribution": attribution, "pairs": pairs}, indent=1))
    print(f"wrote {len(pairs)} pairs to {out}")


if __name__ == "__main__":
    main()
