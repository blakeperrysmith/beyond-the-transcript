"""Generate MODEL_CARD.md from metrics.json so the numbers cannot be hand-edited."""
from __future__ import annotations

from pathlib import Path


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _ds(k: str) -> str:
    return {"cremad": "CREMA-D", "ravdess": "RAVDESS"}.get(k, k)


def _ci(ci) -> str:
    if not ci or ci[0] != ci[0]:
        return "n/a"
    return f"{100 * ci[0]:.0f}-{100 * ci[1]:.0f}%"


def render_model_card(meta: dict, metrics: dict, cross: dict | None = None) -> str:
    """Markdown for the model card, built only from the metrics files so the numbers cannot drift."""
    t = metrics["test"]
    lines: list[str] = []
    if meta["data_source"] == "synthetic":
        lines += [
            "> **SYNTHETIC SMOKE-TEST MODEL.** Trained on generated tones to test the pipeline.",
            "> Its numbers say nothing about speech. Retrain on the real corpora before using.",
            "",
        ]
    lines += [
        "# Model card: delivery classifier",
        "",
        "A small CRNN that labels how a sentence was *delivered* (neutral, happy, sad, angry,",
        "fearful, disgust) from the audio signal alone, without using the transcript. It exists to",
        "demonstrate that speech carries information a one-best transcript throws away.",
        "",
        "## Intended use and non-use",
        "- Intended: an educational demo of audio-native modelling, and a worked example of",
        "  speaker-disjoint training, calibration, abstention and per-group evaluation.",
        "- Not intended: judging a real person's emotional state, hiring, screening, surveillance,",
        "  or any decision about people. Acted emotion is not felt emotion.",
        "- The app never infers age, sex or any other demographic attribute. Demographic metadata",
        "  is used only offline, to measure whether accuracy differs across groups.",
        "",
        "## Model",
        f"- {meta['n_params']:,} parameters. Three conv blocks, a bidirectional GRU over time, mean pool, linear head.",
        "- Input: 64-band log-mel, 3 s window, 16 kHz. The feature code is shared by training and serving.",
        "- Trained in PyTorch, exported to ONNX, served with tract (Sonos' inference engine).",
        f"- PyTorch vs tract parity on {metrics['parity_pytorch_vs_tract']['n']} test clips: max logit difference "
        f"{metrics['parity_pytorch_vs_tract']['max_abs_logit_diff']:.1e}, "
        f"argmax agreement {_pct(metrics['parity_pytorch_vs_tract']['argmax_agreement'])}.",
        "",
        "## Data",
        *[f"- {v}" for v in meta["licenses"].values()],
        "- Both corpora are acted speech by adults in North American English. RAVDESS 'calm' and",
        "  'surprised' were dropped because CREMA-D has no counterpart.",
        f"- Split by speaker (seed {meta['split_seed']}), within each corpus, so no speaker appears in more than one split.",
        "- Because RAVDESS is licensed non-commercial, so is this model.",
        "",
        "## Results (held-out speakers)",
        f"- {t['n_clips']} clips from {t['n_speakers']} speakers. Chance is {_pct(t['chance_accuracy'])}.",
        f"- Accuracy {_pct(t['accuracy'])} (95% CI {_ci(t['accuracy_ci95_speaker_bootstrap'])}, bootstrap over speakers), macro-F1 {t['macro_f1']:.2f}.",
        f"- Calibration after temperature scaling (T={t['temperature']:.2f}): ECE {t['ece']:.3f}.",
        f"- Abstention: below confidence {t['abstain_threshold']:.2f} the app says 'not sure'. On test this answers "
        f"{_pct(t['coverage_at_threshold'])} of clips with {_pct(t['accuracy_when_answering'])} accuracy on those.",
        "",
        "### Per group (accuracy, 95% CI over speakers)",
        "Small groups have wide intervals. A gap smaller than the interval is not evidence of a difference.",
        "",
        "| Grouping | Group | Clips | Speakers | Accuracy | 95% CI |",
        "|---|---|---|---|---|---|",
    ]
    for gname, groups in t["groups"].items():
        if gname == "sentence":
            continue
        for gval, g in groups.items():
            lines.append(
                f"| {gname} | {_ds(gval) if gname == 'dataset' else gval} | {g['n_clips']} | {g['n_speakers']} | {_pct(g['accuracy'])} | {_ci(g['ci95'])} |"
            )
    ps = t["per_speaker"]
    lines += [
        "",
        f"Per-speaker accuracy across {ps['n_speakers']} test speakers: min {_pct(ps['min'])}, median {_pct(ps['median'])}, max {_pct(ps['max'])}.",
        "The spread between speakers matters more than the average.",
        "Sex comes from both corpora. Age, race and ethnicity come only from CREMA-D's metadata, so those rows describe CREMA-D speakers.",
        "",
    ]
    cc = {k: v for k, v in (cross or {}).items() if v and v.get("accuracy") is not None}
    if cc:
        lines += [
            "### Trained on one corpus, tested on the other",
            "A harder and more honest test than the speaker-disjoint split above: a new corpus brings new speakers, a new studio and a new script.",
            "",
            "| Train | Test | Test clips | Test speakers | Accuracy | 95% CI |",
            "|---|---|---|---|---|---|",
        ]
        for k, v in cc.items():
            a, b = (_ds(x) for x in k.split("_to_"))
            lines.append(f"| {a} | {b} | {v['n_test_clips']} | {v['n_test_speakers']} | {_pct(v['accuracy'])} | {_ci(v.get('accuracy_ci95_speaker_bootstrap'))} |")
        near = [k for k, v in cc.items() if v["accuracy"] < t["chance_accuracy"] + 0.10]
        note = (" A result near chance means the model has learned corpus-specific cues as well as delivery, and should not be trusted on unfamiliar recording conditions." if near
                else " Compare these with the held-out result above: the gap is the cost of a new corpus.")
        lines += ["", f"Chance is {_pct(t['chance_accuracy'])}." + note, ""]
    lines += [
        "## Limitations and next steps",
        "### Limitations",
        "- Acted, studio-quality, adult speech. Not validated for children, older adults beyond the corpus range,",
        "  non-American accents, other languages, phone audio, background noise or spontaneous conversation.",
        "- The corpora carry no accent labels, so accuracy by accent cannot be measured with them.",
        "- The author's earlier production speech work was machine-directed speech. Spontaneous speech has hesitations,",
        "  self-repair, overlap and blended emotions that this model never saw.",
        "- Six forced categories are a simplification. The abstain state exists because many real utterances fit none.",
        "- The Praat measurements and the classifier are independent readings of the same audio. Nothing combines them.",
        "- Latency figures in the app are from the demo host, not from this build machine.",
        "",
        "### Next steps",
        "- Evaluate on spontaneous, noisy, multi-speaker speech, broken down by accent, age and recording device.",
        "- Compare against a large pretrained speech encoder with a small head, to separate data limits from model limits.",
        "- Use the Praat measurements as a second input, and learn when the two readings disagree.",
        "- Score overlapping one-second windows over a live stream and smooth over time.",
        "- Check calibration group by group, and re-fit the abstain threshold on conversational data.",
        "",
        "## Verification",
        f"- PyTorch and tract agree on the exported model (above). The same ONNX file is what the browser runs on-device.",
        "- The browser's JavaScript feature code and the Python feature code produce the same log-mel to within 1e-3, tested at 8, 16, 22.05, 44.1 and 48 kHz input.",
        "- A headless browser runs the WebAssembly model and the server's tract model on the same clip and requires matching probabilities.",
        "",
        "## Reproduce",
        "`python -m btt.train --ravdess DIR --cremad DIR --demographics CSV --out artifacts`",
        f"Built {meta['created']} with torch {meta['torch']}, Python {meta['python']}.",
    ]
    return "\n".join(lines) + "\n"


def write_model_card(out: Path, meta: dict, metrics: dict, cross: dict | None = None) -> Path:
    path = Path(out) / "MODEL_CARD.md"
    path.write_text(render_model_card(meta, metrics, cross))
    return path
