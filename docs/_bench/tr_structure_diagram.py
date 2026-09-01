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

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from schematic import fit_text  # noqa: E402

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

SEG_COLOR = ["#dd8452", "#8172b3", "#4c72b0"]
INK = "#22252a"

#: One repetition of an inversion-prepared gradient echo, at a train length of
#: two so the figure stays readable. A gradient echo would not do: its TR is one
#: shot, so every segment in it is played once and the partition shows nothing.
#: This one shows both ways a segment is reused -- the readout replayed inside
#: the TR, and one delay serving two different waits.
PATTERN = ["INV", "TI", "RF", "RO", "RW", "RF", "RO", "RW", "pad"]
BLOCK = {
    "INV": "#937860",
    "TI": "#b0b0b0",
    "RF": "#4c72b0",
    "RO": "#c44e52",
    "RW": "#55a868",
    "pad": "#b0b0b0",
}
#: ``(first block, last block, segment id)`` over one repetition.
SPANS = [(0, 0, 0), (1, 1, 1), (2, 4, 2), (5, 7, 2), (8, 8, 1)]
NSEG = 3
NREP = 2
NSCAN = 3


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
        fit_text(
            ax,
            x + w / 2,
            y + h / 2,
            label,
            width=0.88 * w,
            height=0.8 * h,
            fontsize=fontsize,
            wrap=False,
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            zorder=4,
        )


def band(ax, x, y, w, h, color, label, label_dy=0.7):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=0.6",
            linewidth=1.4,
            edgecolor=color,
            facecolor=color,
            alpha=0.16,
            zorder=1,
        )
    )
    ax.text(
        x + w / 2,
        y + h + label_dy,
        label,
        ha="center",
        va="bottom",
        fontsize=10.2,
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
    fig, ax = plt.subplots(figsize=(9.2, 6.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 48)
    ax.axis("off")

    def heading(y, title, subtitle, gap=13.5):
        ax.text(2, y, title, ha="left", va="center", fontsize=12.8,
                color=INK, fontweight="bold")
        ax.text(gap, y, subtitle, ha="left", va="center", fontsize=10.6,
                color=INK, style="italic")

    # -- 1. the block stream --------------------------------------------
    heading(44.6, "the file", "a flat list of blocks")

    per = len(PATTERN)
    n = per * NREP
    x0, span = 2.0, 96.0
    pitch = span / n
    w = pitch - 0.55
    for i in range(n):
        lbl = PATTERN[i % per]
        cell(ax, x0 + i * pitch, 36.5, w, 3.4, BLOCK[lbl], lbl, fontsize=8.4)
    brace(ax, x0, x0 + per * pitch - 0.55, 34.6, INK,
          "the shortest period that holds over the whole list")

    # -- 2. the repeating unit ------------------------------------------
    heading(29.0, "the TR",
            "recognised from block content — duration and which channels play, "
            "never an annotation")

    bx, bspan = 4.0, 92.0
    bpitch = bspan / per
    bw = bpitch - 1.4
    for k, (lo, hi, seg) in enumerate(SPANS):
        band(ax, bx + lo * bpitch - 0.55, 17.0,
             (hi - lo) * bpitch + bw + 1.1, 7.8,
             SEG_COLOR[seg], f"segment {seg}",
             label_dy=0.6 if k % 2 == 0 else 2.9)
    for i, lbl in enumerate(PATTERN):
        cell(ax, bx + i * bpitch, 18.6, bw, 4.4, BLOCK[lbl], lbl, fontsize=9.6)
    ax.text(50, 14.6,
            "cut where every gradient axis rests at zero, and cut before each "
            "excitation — so a segment is a shot",
            ha="center", va="top", fontsize=10.2, color=INK)

    # -- 3. what the sequencer executes ---------------------------------
    heading(9.6, "the scan",
            f"{NSEG} segments prepared, {len(SPANS)} instances triggered per "
            "repetition", gap=15.5)

    sx, sspan = 2.0, 96.0
    tr_pitch = sspan / NSCAN
    widths = [(hi - lo + 1) for lo, hi, _ in SPANS]
    unit = (tr_pitch - 2.2) / sum(widths)
    for r in range(NSCAN):
        base = sx + r * tr_pitch
        for (lo, hi, seg), width in zip(SPANS, widths):
            cw = width * unit - 0.5
            cell(ax, base, 3.4, cw, 3.4, SEG_COLOR[seg], str(seg), fontsize=9.4)
            base += width * unit
    ax.text(50, 1.1,
            "segment 2 is the readout, replayed inside the TR; segment 1 is "
            "one delay serving two different waits",
            ha="center", va="center", fontsize=10.4, color=INK, style="italic")

    fig.tight_layout(pad=0.5)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "tr_and_segments_schematic.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(build().relative_to(OUT_DIR.parents[1]))
