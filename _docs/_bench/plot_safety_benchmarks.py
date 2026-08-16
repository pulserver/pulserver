#!/usr/bin/env python3
"""Plot the mechanical-resonance-vs-PNS cost comparison in ``explanations/benchmarks``.

``bench_safety.py`` writes ``safety.json`` (mechres ``gate_ms`` and PNS
``pns_slew_ms`` + ``pns_model_ms`` per sequence); this turns that measured
data into the bar chart embedded in the benchmarks page, so the "PNS costs
roughly N times mechanical resonance" claim there is read off a plot instead
of asserted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "docs" / "_bench" / "safety.json"
DEFAULT_OUTPUT = ROOT / "docs" / "explanations" / "assets" / "benchmarks" / "safety_scaling.png"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = json.loads(args.input.read_text())["rows"]
    labels = [row["label"] for row in rows]
    x = np.arange(len(rows))
    gate_ms = [row["gate_ms"] for row in rows]
    pns_ms = [row["pns_slew_ms"] + row["pns_model_ms"] for row in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.38
    ax.bar(x - width / 2, gate_ms, width, label="mechanical resonance gate", color="#4e9a6f")
    ax.bar(x + width / 2, pns_ms, width, label="PNS (waveform + model)", color="#c0524a")
    ax.set_yscale("log")
    ax.set_ylabel("predownload cost [ms]")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(args.output)


if __name__ == "__main__":
    main()
