#!/usr/bin/env python3
"""Plot the compact benchmark summary embedded in ``explanations/benchmarks``.

``bench_zoo.py`` writes ``results.json``; this script turns that measured data
into a reproducible, paper-ready overview without duplicating numbers in code.
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
DEFAULT_INPUT = ROOT / "docs" / "_bench" / "results.json"
DEFAULT_OUTPUT = ROOT / "docs" / "explanations" / "assets" / "benchmarks" / "pipeline_scaling.png"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = json.loads(args.input.read_text())
    labels = [f"{row['plugin']}\n{row['label']}" for row in rows]
    x = np.arange(len(rows))
    download_s = [row["design_s"] + row["serialise_s"] + row["parse_ms"] / 1e3 + row["convert_ms"] / 1e3 + row["cache_write_ms"] / 1e3 for row in rows]
    pulsegen_kb = [row["heap_pulsegen_kb"] for row in rows]
    scanloop_kb = [row["heap_scanloop_kb"] for row in rows]

    fig, (ax_time, ax_memory) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax_time.bar(x, download_s, color="#3977af")
    ax_time.set_yscale("log")
    ax_time.set_ylabel("predownload time [s]")
    ax_time.grid(axis="y", alpha=0.25)

    width = 0.38
    ax_memory.bar(x - width / 2, pulsegen_kb, width, label="pulse generation", color="#4e9a6f")
    ax_memory.bar(x + width / 2, scanloop_kb, width, label="scan loop", color="#d5894d")
    ax_memory.set_yscale("log")
    ax_memory.set_ylabel("loaded heap [KB]")
    ax_memory.set_xticks(x, labels, rotation=30, ha="right")
    ax_memory.grid(axis="y", alpha=0.25)
    ax_memory.legend(frameon=False, ncol=2)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(args.output)


if __name__ == "__main__":
    main()
