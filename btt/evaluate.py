"""Evaluation: calibration, abstention, per-group and per-speaker metrics.

Every interval is a bootstrap over *speakers*, not clips: clips from one speaker
are not independent, so a clip-level interval would be too narrow. With 14 or so
test speakers the intervals are wide, and that is the honest picture.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax, log_softmax

from .data import Item
from .labels import CLASSES


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Single-parameter temperature scaling fit on validation NLL, T kept in [0.5, 5]
    so a tiny or easy validation set cannot produce a degenerate value."""
    def nll(log_t: float) -> float:
        lp = log_softmax(logits / np.exp(log_t), axis=1)
        return float(-lp[np.arange(len(y)), y].mean())

    res = minimize_scalar(nll, bounds=(float(np.log(0.5)), float(np.log(5.0))), method="bounded")
    return float(np.exp(res.x))


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(total)


def pick_abstain_threshold(probs_val: np.ndarray, coverage: float = 0.8) -> float:
    """Confidence threshold at which the model answers on ``coverage`` of validation clips."""
    conf = np.sort(probs_val.max(axis=1))
    k = int(np.floor((1.0 - coverage) * len(conf)))
    return float(conf[min(max(k, 0), len(conf) - 1)])


def macro_f1(y: np.ndarray, pred: np.ndarray, n_classes: int = len(CLASSES)) -> float:
    f1s = []
    for c in range(n_classes):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(f1s))


def confusion(y: np.ndarray, pred: np.ndarray, n_classes: int = len(CLASSES)) -> list[list[int]]:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    return cm.tolist()


def _speaker_table(speakers: list[str], correct: np.ndarray):
    uniq = sorted(set(speakers))
    idx = {s: i for i, s in enumerate(uniq)}
    n = np.zeros(len(uniq))
    c = np.zeros(len(uniq))
    for s, ok in zip(speakers, correct):
        n[idx[s]] += 1
        c[idx[s]] += ok
    return uniq, n, c


def speaker_bootstrap_ci(
    speakers: list[str], correct: np.ndarray, reps: int = 2000, seed: int = 0
) -> tuple[float, float]:
    _u, n, c = _speaker_table(speakers, correct)
    if len(n) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(n), size=(reps, len(n)))
    accs = c[idx].sum(axis=1) / n[idx].sum(axis=1)
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def age_bucket(age: int | None) -> str:
    if age is None:
        return "unknown"
    if age < 30:
        return "20-29"
    if age < 50:
        return "30-49"
    return "50+"


def group_report(items: list[Item], y: np.ndarray, pred: np.ndarray) -> dict:
    """Accuracy with speaker-bootstrap CI for each available grouping."""
    correct = (y == pred).astype(float)
    keys = {
        "dataset": [i.dataset for i in items],
        "sex": [i.sex for i in items],
        "age_bucket": [age_bucket(i.age) for i in items],
        "race": [i.race for i in items],
        "ethnicity": [i.ethnicity for i in items],
        "sentence": [i.sentence_id for i in items],
    }
    speakers = [i.speaker for i in items]
    out: dict[str, dict] = {}
    for name, vals in keys.items():
        vals = np.array(vals)
        groups = {}
        for v in sorted(set(vals)):
            if v == "unknown":
                continue
            m = vals == v
            sp = [s for s, k in zip(speakers, m) if k]
            lo, hi = speaker_bootstrap_ci(sp, correct[m])
            groups[str(v)] = {
                "n_clips": int(m.sum()),
                "n_speakers": len(set(sp)),
                "accuracy": float(correct[m].mean()),
                "ci95": [lo, hi],
                "macro_f1": macro_f1(y[m], pred[m]),
            }
        if len(groups) > 1 or name == "dataset":
            out[name] = groups
    return out


def per_speaker_spread(items: list[Item], y: np.ndarray, pred: np.ndarray) -> dict:
    correct = (y == pred).astype(float)
    uniq, n, c = _speaker_table([i.speaker for i in items], correct)
    acc = c / n
    q = np.percentile(acc, [0, 25, 50, 75, 100])
    worst = np.argsort(acc)[:3]
    return {
        "n_speakers": len(uniq),
        "min": float(q[0]),
        "p25": float(q[1]),
        "median": float(q[2]),
        "p75": float(q[3]),
        "max": float(q[4]),
        "worst_speakers": [{"speaker": uniq[i], "accuracy": float(acc[i])} for i in worst],
    }


def coverage_curve(probs: np.ndarray, y: np.ndarray, thresholds=(0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    rows = []
    for t in thresholds:
        m = conf >= t
        rows.append(
            {
                "threshold": float(t),
                "coverage": float(m.mean()),
                "accuracy_on_answered": float((pred[m] == y[m]).mean()) if m.any() else None,
            }
        )
    return rows


def full_report(
    items: list[Item], logits: np.ndarray, y: np.ndarray, temperature: float, threshold: float
) -> dict:
    probs = softmax(logits / temperature, axis=1)
    pred = probs.argmax(axis=1)
    speakers = [i.speaker for i in items]
    lo, hi = speaker_bootstrap_ci(speakers, (pred == y).astype(float))
    return {
        "n_clips": int(len(y)),
        "n_speakers": len(set(speakers)),
        "accuracy": float((pred == y).mean()),
        "accuracy_ci95_speaker_bootstrap": [lo, hi],
        "macro_f1": macro_f1(y, pred),
        "chance_accuracy": 1.0 / len(CLASSES),
        "ece": ece(probs, y),
        "temperature": temperature,
        "abstain_threshold": threshold,
        "coverage_at_threshold": float((probs.max(axis=1) >= threshold).mean()),
        "accuracy_when_answering": float(
            (pred[probs.max(axis=1) >= threshold] == y[probs.max(axis=1) >= threshold]).mean()
        )
        if (probs.max(axis=1) >= threshold).any()
        else None,
        "coverage_curve": coverage_curve(probs, y),
        "confusion": confusion(y, pred),
        "classes": CLASSES,
        "groups": group_report(items, y, pred),
        "per_speaker": per_speaker_spread(items, y, pred),
    }
