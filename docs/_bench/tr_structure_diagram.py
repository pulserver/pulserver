#!/usr/bin/env python3
"""Figure for ``explanations/sequence_model/tr_and_segmentation.md``.

Documentation-only tooling; not part of the shipped package.

A schematic of the three levels the page derives, one above the other: the
block stream, the repeating unit found in it, and the segments that unit is
partitioned into. The plots beside it in that page are real sequences; this
one names the relationship they are instances of.

Usage:
    <venv>/bin/python docs/_bench/tr_structure_diagram.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# House style: a figure has to be legible at the width a manual page gives it.
plt.rcParams.update(
    {
        "font.size": 11.0,
        "axes.titlesize": 12.5,
        "axes.labelsize": 11.5,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 10.5,
        "figure.titlesize": 13.0,
    }
)


OUT_DIR = Path(__file__).resolve().parents[1] / "explanations" / "assets" / "segments"

BLOCK = ["#4c72b0", "#55a868", "#c44e52", "#b0b0b0"]
SEG_A = "#dd8452"
SEG_B = "#8172b3"
INK = "#22252a"

#: One repetition: excite, phase encode, read, pad. The pad is the pure delay
#: that splits off into a segment of its own.
PATTERN = ["RF", "PE", "RO", "pad"]
NREP = 6


def cell(ax, x, y, w, h, color, label=None, *, fontsize=9.4, alpha=1.0):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=0.3",
            linewidth=0.7,
            edgecolor=color,
            facecolor=color,
            alpha=alpha,
            zorder=3,
        )
    )
    if label:
        ax.text(
            x + w / 2,
            y + h / 2,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            color="white",
            fontweight="bold",
            zorder=4,
        )


def band(ax, x, y, w, h, color, label):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=0.6",
            linewidth=1.1,
            edgecolor=color,
            facecolor=color,
            alpha=0.13,
            zorder=1,
        )
    )
    ax.text(
        x + w / 2,
        y + h + 0.7,
        label,
        ha="center",
        va="bottom",
        fontsize=10.8,
        color=color,
        fontweight="bold",
        zorder=4,
    )


def brace(ax, x0, x1, y, color, label):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y),
            (x1, y),
            arrowstyle="<->",
            mutation_scale=9,
            linewidth=1.1,
            color=color,
            zorder=4,
        )
    )
    ax.text(
        x0, y - 1.0, label, ha="left", va="top", fontsize=10.5, color=color, zorder=4
    )


def build() -> Path:
    fig, ax = plt.subplots(figsize=(8.6, 6.42))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 46)
    ax.axis("off")

    # -- 1. the block stream --------------------------------------------
    ax.text(
        2,
        43.4,
        "the file",
        ha="left",
        va="center",
        fontsize=12.8,
        color=INK,
        fontweight="bold",
    )
    ax.text(
        13.5,
        43.4,
        "a flat list of blocks",
        ha="left",
        va="center",
        fontsize=10.8,
        color=INK,
        style="italic",
    )

    n = len(PATTERN) * NREP
    w, gap = 3.4, 0.55
    x0 = 2.0
    for i in range(n):
        cell(
            ax,
            x0 + i * (w + gap),
            36.5,
            w,
            3.4,
            BLOCK[i % len(PATTERN)],
            PATTERN[i % len(PATTERN)],
        )
    brace(
        ax,
        x0,
        x0 + len(PATTERN) * (w + gap) - gap,
        34.6,
        INK,
        "the shortest period that holds over the whole list",
    )

    # -- 2. the repeating unit ------------------------------------------
    ax.text(
        2,
        27.2,
        "the TR",
        ha="left",
        va="center",
        fontsize=12.8,
        color=INK,
        fontweight="bold",
    )
    ax.text(
        13.5,
        27.2,
        "recognised from block content — duration and which channels play, "
        "never an annotation",
        ha="left",
        va="center",
        fontsize=10.8,
        color=INK,
        style="italic",
    )

    bw, bgap = 17.0, 2.2
    bx = 6.0
    band(ax, bx - 1.6, 17.0, 2 * (bw + bgap) + bw + 2.6, 7.6, SEG_A, "segment 0")
    band(ax, bx + 3 * (bw + bgap) - 1.6, 17.0, bw + 3.2, 7.6, SEG_B, "segment 1")
    for i, lbl in enumerate(PATTERN):
        cell(ax, bx + i * (bw + bgap), 18.6, bw, 4.4, BLOCK[i], lbl, fontsize=11.6)
    ax.text(
        50,
        15.2,
        "cut where the gradients are zero, so each piece is playable on its own",
        ha="center",
        va="top",
        fontsize=10.5,
        color=INK,
    )

    # -- 3. what the sequencer executes ---------------------------------
    ax.text(
        2,
        9.6,
        "the scan",
        ha="left",
        va="center",
        fontsize=12.8,
        color=INK,
        fontweight="bold",
    )
    ax.text(
        15.5,
        9.6,
        "two segments prepared, twelve instances triggered",
        ha="left",
        va="center",
        fontsize=10.8,
        color=INK,
        style="italic",
    )

    sw, sgap = 5.6, 0.5
    sx = 2.0
    for r in range(NREP):
        base = sx + r * (sw * 4 / 3 + sw / 3 + sgap * 2)
        cell(ax, base, 3.4, sw, 3.4, SEG_A, "0", fontsize=9.9)
        cell(ax, base + sw + sgap, 3.4, sw / 2.2, 3.4, SEG_B, "1", fontsize=9.9)
    ax.text(
        50,
        1.2,
        "one prepared program per segment; a repetition is a pair of triggers "
        "and its own row of parameters",
        ha="center",
        va="center",
        fontsize=10.5,
        color=INK,
        style="italic",
    )

    fig.tight_layout(pad=0.5)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "tr_and_segments_schematic.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(build().relative_to(OUT_DIR.parents[1]))
