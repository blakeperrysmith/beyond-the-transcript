"""Draw app/static/pipeline.svg: the two readings of one waveform. Plain SVG, no dependencies.

    python scripts/pipeline_diagram.py

The figures in it (layer sizes, parameter count) come from btt/model.py's constants and the model built
from them, so the picture cannot drift from the code.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xml.sax.saxutils import escape

from btt.features import HOP, N_MELS, WIN_FRAMES
from btt.labels import CLASSES

try:
    from btt.model import DeliveryNet, count_params

    N_PARAMS = count_params(DeliveryNet(n_classes=len(CLASSES)))
except Exception:  # torch not installed: fall back to a number checked in tests
    N_PARAMS = 106934

W, H = 900, 858
INK, MUTED = "#15222c", "#5a6a75"
LEARN, LEARN_BG = "#3a3fa5", "#eceefb"
MEAS, MEAS_BG = "#b9531c", "#fcefe7"
NEUT_BG = "#f4f6f7"

out: list[str] = []


def box(x, y, w, h, title, lines=(), stroke=INK, fill=NEUT_BG, tag=None, dash=False):
    d = ' stroke-dasharray="6 4"' if dash else ""
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="1.6"{d}/>')
    ty = y + 24
    out.append(f'<text x="{x + 14}" y="{ty}" font-size="15" font-weight="700" fill="{INK}">{escape(title)}</text>')
    for ln in lines:
        ty += 20
        out.append(f'<text x="{x + 14}" y="{ty}" font-size="13" fill="{INK if not ln.startswith("~") else MUTED}">{escape(ln.lstrip("~"))}</text>')
    if tag:
        out.append(f'<text x="{x + w - 12}" y="{y + h - 12}" font-size="12" text-anchor="end" fill="{MUTED}">{escape(tag)}</text>')


def arrow(x1, y1, x2, y2, color=INK):
    out.append(f'<path d="M{x1} {y1} L{x2} {y2}" stroke="{color}" stroke-width="1.8" fill="none" marker-end="url(#ah)"/>')


def elbow(points, color=INK):
    d = "M" + " L".join(f"{x} {y}" for x, y in points)
    out.append(f'<path d="{d}" stroke="{color}" stroke-width="1.8" fill="none" stroke-linejoin="round" marker-end="url(#ah)"/>')


out.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" font-family="system-ui, -apple-system, \'Segoe UI\', Roboto, Helvetica, Arial, sans-serif">')
out.append("<title>Two independent readings of one waveform: a learned delivery classifier and Praat measurements</title>")
out.append(f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{INK}"/></marker></defs>')
out.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')

# shared front
box(300, 8, 300, 72, "Audio in", ["Microphone or sample clip", "16-bit PCM at any sample rate"])
arrow(450, 80, 450, 102)
box(300, 102, 300, 60, "Resample to 16 kHz", ["Same code path in training and serving"])
elbow([(450, 162), (450, 184), (235, 184), (235, 208)], LEARN)
elbow([(450, 184), (675, 184), (675, 208)], MEAS)

# lane headers
out.append(f'<rect x="30" y="208" width="410" height="30" rx="15" fill="{LEARN}"/>')
out.append(f'<text x="235" y="228" font-size="14" font-weight="700" fill="#fff" text-anchor="middle">Learned reading</text>')
out.append(f'<rect x="480" y="208" width="390" height="30" rx="15" fill="{MEAS}"/>')
out.append(f'<text x="675" y="228" font-size="14" font-weight="700" fill="#fff" text-anchor="middle">Measured reading</text>')

# learned lane
box(30, 262, 410, 62, "Log-mel features", [f"{N_MELS} bands, 3 s window ({WIN_FRAMES} frames, {HOP // 16} ms hop)"], LEARN, LEARN_BG)
arrow(235, 324, 235, 346, LEARN)
box(30, 346, 410, 214, "DeliveryNet", [
    "3 blocks of conv 3x3, batch norm, ReLU, max-pool",
    "Channels 16, then 32, then 64",
    "Linear projection to 64",
    "Bidirectional GRU, 64 units each way",
    "Mean over time",
    f"Linear to {len(CLASSES)} classes",
    f"{N_PARAMS:,} parameters",
], LEARN, LEARN_BG, tag="tract on the server, ONNX Runtime Web on your device")
arrow(235, 560, 235, 582, LEARN)
box(30, 582, 410, 62, "Temperature scaling, then softmax", [f"Probabilities for {len(CLASSES)} deliveries"], LEARN, LEARN_BG)
arrow(235, 644, 235, 666, LEARN)
box(30, 666, 410, 62, "Abstain below a confidence threshold", ["A label with its confidence, or \"Not sure\""], LEARN, LEARN_BG)

# measured lane
box(480, 262, 390, 62, "Praat (Parselmouth)", ["Pitch and intensity tracking on the same audio"], MEAS, MEAS_BG)
arrow(675, 324, 675, 346, MEAS)
box(480, 346, 390, 214, "Measurements, not learned", [
    "Pitch: median and range",
    "Share of the clip that is voiced",
    "Syllable rate",
    "Share of the clip that is pause",
    "Loudness range",
    "Spectral tilt",
    "Pitch contour, drawn on the spectrogram",
], MEAS, MEAS_BG, tag="server only")
arrow(675, 560, 675, 666, MEAS)
box(480, 666, 390, 62, "Numbers for A and B", ["Shown side by side, with the difference"], MEAS, MEAS_BG)

# no-fusion marker
out.append(f'<path d="M460 262 L460 560" stroke="{MUTED}" stroke-width="1.4" stroke-dasharray="3 5"/>')
out.append(f'<text transform="translate(452 411) rotate(-90)" font-size="12" fill="{MUTED}" text-anchor="middle" dominant-baseline="auto">not connected</text>')

# merge into the page
out.append(f'<path d="M235 728 L235 748 L675 748 M675 728 L675 748" stroke="{INK}" stroke-width="1.8" fill="none" stroke-linejoin="round"/>')
arrow(450, 748, 450, 770)
box(250, 770, 400, 74, "The page: A and B side by side", ["Nothing combines the two readings.", "~Praat is not an input to the model."])
out.append("</svg>")

path = Path(__file__).resolve().parent.parent / "app" / "static" / "pipeline.svg"
path.write_text("\n".join(out) + "\n", encoding="utf-8")
print("wrote", path)
