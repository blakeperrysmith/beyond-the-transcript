"""Train, evaluate, calibrate and export.

    python -m btt.train --ravdess DIR --cremad DIR --demographics CSV --out artifacts
    python -m btt.train --synthetic /tmp/syn --out artifacts_smoke      # pipeline test only

The shipped model is the speaker-disjoint, two-corpus run ("main"). The
"cross" mode trains on one corpus and tests on the other, which is the closer
check on whether the model learned delivery or learned a recording setup.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import __version__
from .data import Item, build_features, scan, speaker_split, subset, write_synthetic_corpus
from .evaluate import fit_temperature, full_report, macro_f1, pick_abstain_threshold, speaker_bootstrap_ci
from .export import benchmark, export_onnx, parity_check
from .features import N_MELS, WIN_FRAMES
from .labels import CLASSES, LICENSES
from .model import DeliveryNet, count_params
from .modelcard import write_model_card


def spec_augment(x: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """Two frequency masks and two time masks per clip, filled with the clip mean."""
    x = x.clone()
    b = x.shape[0]
    for i in range(b):
        fill = x[i].mean()
        for _ in range(2):
            w = int(rng.integers(0, 9))
            f0 = int(rng.integers(0, N_MELS - w + 1))
            x[i, :, f0 : f0 + w, :] = fill
        for _ in range(2):
            w = int(rng.integers(0, 31))
            t0 = int(rng.integers(0, WIN_FRAMES - w + 1))
            x[i, :, :, t0 : t0 + w] = fill
    return x


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict_logits(model: nn.Module, x: np.ndarray, bs: int = 128) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(x), bs):
        dev = next(model.parameters()).device
        xb = torch.from_numpy(x[i : i + bs].astype(np.float32)).to(dev)
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(CLASSES)), dtype=np.float32)


def fit(
    xtr: np.ndarray,
    ytr: np.ndarray,
    xva: np.ndarray,
    yva: np.ndarray,
    epochs: int,
    bs: int,
    lr: float,
    seed: int,
    log=print,
) -> tuple[DeliveryNet, dict]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = DeliveryNet()
    xs = xtr.astype(np.float32)
    model.set_norm(float(xs.mean()), float(xs.std()))
    model.to(DEVICE)
    counts = np.bincount(ytr, minlength=len(CLASSES)).astype(np.float64)
    weights = torch.tensor(counts.sum() / (len(CLASSES) * np.maximum(counts, 1)), dtype=torch.float32).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    steps = epochs * max(1, int(np.ceil(len(xs) / bs)))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    best, best_state, history = -1.0, None, []
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(xs))
        tot, n = 0.0, 0
        for i in range(0, len(perm), bs):
            idx = perm[i : i + bs]
            xb = spec_augment(torch.from_numpy(xs[idx]), rng).to(DEVICE)
            yb = torch.from_numpy(ytr[idx]).to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
            n += len(idx)
        va_logits = predict_logits(model, xva)
        va_f1 = macro_f1(yva, va_logits.argmax(1))
        history.append({"epoch": ep + 1, "train_loss": tot / n, "val_macro_f1": va_f1})
        log(f"epoch {ep + 1:3d}/{epochs}  loss {tot / n:.3f}  val macro-F1 {va_f1:.3f}")
        if va_f1 > best:
            best, best_state = va_f1, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.cpu().eval()  # export, parity and benchmark run on CPU
    return model, {"best_val_macro_f1": best, "history": history}


def run_main(args, items: list[Item], out: Path, log=print) -> dict:
    split = speaker_split(items, seed=args.seed)
    parts = {k: subset(items, v) for k, v in split.items()}
    cache = Path(args.cache_dir) if args.cache_dir else None
    feats = {
        k: build_features(v, cache / f"{k}.npz" if cache else None) for k, v in parts.items()
    }
    log({k: len(v) for k, v in parts.items()}, "clips by split")
    (xtr, ytr), (xva, yva), (xte, yte) = feats["train"], feats["val"], feats["test"]
    t0 = time.time()
    model, fit_info = fit(xtr, ytr, xva, yva, args.epochs, args.bs, args.lr, args.seed, log=log)
    train_seconds = time.time() - t0

    va_logits = predict_logits(model, xva)
    temperature = fit_temperature(va_logits, yva)
    from scipy.special import softmax

    threshold = pick_abstain_threshold(softmax(va_logits / temperature, axis=1), coverage=0.8)
    te_logits = predict_logits(model, xte)
    test_report = full_report(parts["test"], te_logits, yte, temperature, threshold)
    val_report = full_report(parts["val"], va_logits, yva, temperature, threshold)

    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    onnx_path = export_onnx(model, out / "model.onnx")
    sel = np.random.default_rng(0).choice(len(xte), size=min(32, len(xte)), replace=False)
    parity = parity_check(model, onnx_path, xte[sel].astype(np.float32))
    bench = benchmark(model, onnx_path, n=150)
    log("parity:", parity)

    meta = {
        "version": __version__,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_source": "synthetic" if args.synthetic else "real",
        "classes": CLASSES,
        "n_params": count_params(model),
        "feature_config": {"sr": 16000, "n_fft": 512, "hop": 160, "n_mels": N_MELS, "win_frames": WIN_FRAMES},
        "temperature": temperature,
        "abstain_threshold": threshold,
        "abstain_target_coverage": 0.8,
        "datasets": sorted({i.dataset for i in items}),
        "licenses": {d: LICENSES[d] for d in sorted({i.dataset for i in items})},
        "split_seed": args.seed,
        "epochs": args.epochs,
        "train_seconds": round(train_seconds, 1),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "train_hardware": f"{platform.processor() or platform.machine()} x{os.cpu_count()}"
        + (f", {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""),
    }
    metrics = {
        "fit": fit_info,
        "validation": {k: v for k, v in val_report.items() if k in ("accuracy", "macro_f1", "ece", "n_clips", "n_speakers")},
        "test": test_report,
        "parity_pytorch_vs_tract": parity,
        "benchmark_build_machine": bench,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "split.json").write_text(json.dumps(split, indent=2))
    write_model_card(out, meta, metrics)
    return {"meta": meta, "metrics": metrics, "model": model}


def run_cross(args, items: list[Item], out: Path, log=print) -> dict:
    """Train on one corpus (speaker-disjoint val inside it), test on the other."""
    results = {}
    for src, dst in (("cremad", "ravdess"), ("ravdess", "cremad")):
        a = [i for i in items if i.dataset == src]
        b = [i for i in items if i.dataset == dst]
        if not a or not b:
            continue
        split = speaker_split(a, seed=args.seed, fracs=(0.85, 0.15, 0.0))
        tr, va = subset(a, split["train"]), subset(a, split["val"])
        xtr, ytr = build_features(tr)
        xva, yva = build_features(va)
        xte, yte = build_features(b)
        log(f"cross {src}->{dst}: train {len(tr)} val {len(va)} test {len(b)}")
        model, _ = fit(xtr, ytr, xva, yva, args.epochs, args.bs, args.lr, args.seed, log=lambda *_: None)
        logits = predict_logits(model, xte)
        pred = logits.argmax(1)
        lo, hi = speaker_bootstrap_ci([i.speaker for i in b], (pred == yte).astype(float))
        results[f"{src}_to_{dst}"] = {
            "n_train_clips": len(tr),
            "n_test_clips": len(b),
            "n_test_speakers": len({i.speaker for i in b}),
            "accuracy": float((pred == yte).mean()),
            "accuracy_ci95_speaker_bootstrap": [lo, hi],
            "macro_f1": macro_f1(yte, pred),
        }
        log(results[f"{src}_to_{dst}"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics_cross_corpus.json").write_text(json.dumps(results, indent=2))
    return results


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ravdess")
    p.add_argument("--cremad")
    p.add_argument("--demographics", help="CREMA-D VideoDemographics.csv (optional, evaluation only)")
    p.add_argument("--synthetic", help="write and use a synthetic corpus in this dir (pipeline test only)")
    p.add_argument("--out", default="artifacts")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache-dir")
    p.add_argument("--mode", choices=["main", "cross", "both"], default="both")
    args = p.parse_args(argv)

    if args.synthetic:
        args.ravdess, args.cremad = map(str, write_synthetic_corpus(args.synthetic))
    items = scan(args.ravdess, args.cremad, args.demographics)
    if not items:
        raise SystemExit("No clips found. Check --ravdess / --cremad paths.")
    print(f"{len(items)} clips, {len({i.speaker for i in items})} speakers")
    out = Path(args.out)
    if args.mode in ("main", "both"):
        run_main(args, items, out)
    if args.mode in ("cross", "both"):
        run_cross(args, items, out)


if __name__ == "__main__":
    main()
