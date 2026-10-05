"""Draw the model graph from the PyTorch code itself and time each layer.

    python scripts/arch_diagram.py [--out app/static]

Writes architecture.svg (torchview graph of DeliveryNet) and architecture.json
(layer table: shapes, parameters, build-time milliseconds on this machine).
Per-layer times are measured here with forward hooks on one CPU thread, at build
time. They are not the live serving numbers, which are in the page footer.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from btt.model import DeliveryNet, example_input  # noqa: E402


def layer_table(model: torch.nn.Module, x: torch.Tensor, repeats: int = 50) -> list[dict]:
    rows, starts, acc = {}, {}, {}
    hooks = []
    leaves = [(n, m) for n, m in model.named_modules() if n and not list(m.children())]

    def pre(name):
        return lambda m, i: starts.__setitem__(name, time.perf_counter())

    def post(name):
        def f(m, i, o):
            dt = (time.perf_counter() - starts[name]) * 1000
            acc.setdefault(name, []).append(dt)
            out = o[0] if isinstance(o, tuple) else o
            rows[name] = {"layer": name, "type": type(m).__name__, "output": list(out.shape),
                          "params": sum(p.numel() for p in m.parameters())}
        return f

    for n, m in leaves:
        hooks += [m.register_forward_pre_hook(pre(n)), m.register_forward_hook(post(n))]
    torch.set_num_threads(1)
    with torch.no_grad():
        for _ in range(5):
            model(x)
        for v in acc.values():
            v.clear()
        for _ in range(repeats):
            model(x)
    for h in hooks:
        h.remove()
    out = []
    for n, _ in leaves:
        if n in rows:
            t = sorted(acc[n])
            rows[n]["ms_p50"] = round(t[len(t) // 2], 3)
            out.append(rows[n])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="app/static")
    ap.add_argument("--classes", type=int, default=6)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from torchview import draw_graph

    model = DeliveryNet(args.classes).eval()
    x = example_input()
    g = draw_graph(model, input_data=x, device="cpu", depth=3, expand_nested=False, graph_name="DeliveryNet", save_graph=False)
    g.visual_graph.attr(rankdir="LR", bgcolor="transparent")
    g.visual_graph.format = "svg"
    svg = g.visual_graph.pipe(format="svg").decode()
    (out / "architecture.svg").write_text(svg)

    rows = layer_table(model, x)
    total = sum(r["params"] for r in rows)
    (out / "architecture.json").write_text(json.dumps(
        {"input": list(x.shape), "n_params": total, "layers": rows,
         "timing_note": "p50 per layer over 50 runs, one CPU thread, PyTorch eager, measured at build time"}, indent=1))
    print(f"{len(rows)} layers, {total:,} parameters -> {out}")


if __name__ == "__main__":
    main()
