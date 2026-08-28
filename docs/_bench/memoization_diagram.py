#!/usr/bin/env python3
"""The shared figure for ``explanations/performance/pns.md`` and
``explanations/performance/mechanical_resonance.md``.

A schematic, not a measurement: what the stimulation check and the acoustic
check do with the same canonical TR, where they are the same calculation, and
where they part.  Drawn rather than plotted, so nothing here reads the package.

Usage:
    <venv>/bin/python docs/_bench/memoization_diagram.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_DIR = (
    Path(__file__).resolve().parents[1] / "explanations" / "assets" / "memoization"
)

SHARED = "#4a4a4a"
PNS = "#4c72b0"
MECH = "#dd8452"
INK = "#22252a"


def box(ax, x, y, w, h, color, *, title, body):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=0.9",
            linewidth=1.2,
            edgecolor=color,
            facecolor="white",
            zorder=2,
        )
    )
    ax.text(
        x + w / 2,
        y + h - 1.4,
        title,
        ha="center",
        va="top",
        fontsize=8.6,
        color=color,
        fontweight="bold",
        zorder=3,
    )
    ax.text(
        x + w / 2,
        y + h - 4.1,
        body,
        ha="center",
        va="top",
        fontsize=7.4,
        color=INK,
        zorder=3,
        linespacing=1.5,
    )
    return x + w / 2, y + h / 2


def group(ax, x, y, w, h, color, label):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=1.2",
            linewidth=1.1,
            edgecolor=color,
            facecolor=color,
            alpha=0.07,
            zorder=1,
        )
    )
    ax.text(
        x + w / 2,
        y + h - 1.3,
        label,
        ha="center",
        va="top",
        fontsize=9.0,
        color=color,
        fontweight="bold",
        zorder=3,
    )


def arrow(ax, start, end, color, *, rad=0.0, label=None, label_pos=None):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.2,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=1.5,
            shrinkB=1.5,
            zorder=4,
        )
    )
    if label is not None:
        lx, ly = label_pos or ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        ax.text(
            lx,
            ly,
            label,
            ha="center",
            va="bottom",
            fontsize=7.2,
            color=color,
            zorder=5,
            linespacing=1.4,
        )


def build() -> Path:
    fig, ax = plt.subplots(figsize=(11.0, 6.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")

    group(ax, 3, 43.5, 94, 14, SHARED, "the same calculation for both checks")
    box(
        ax,
        5.5,
        45,
        28,
        9.5,
        SHARED,
        title="walk the canonical TR once",
        body="per block position and per axis,\nthe waveform each repetition\nputs there",
    )
    box(
        ax,
        36,
        45,
        28,
        9.5,
        SHARED,
        title="one identity per waveform",
        body="gradient definition \u00b7 shape id \u00b7\nblock duration \u2014 the identity the\nrepresentation already carries",
    )
    box(
        ax,
        66.5,
        45,
        28,
        9.5,
        SHARED,
        title="one response per waveform",
        body="each distinct waveform convolved\n(or transformed) once, then placed\nand scaled per occurrence",
    )
    arrow(ax, (33.5, 49.75), (36, 49.75), SHARED)
    arrow(ax, (64, 49.75), (66.5, 49.75), SHARED)

    arrow(ax, (40, 43.2), (25.5, 34.4), SHARED, rad=0.10)
    arrow(ax, (60, 43.2), (74.5, 34.4), SHARED, rad=-0.10)
    ax.text(
        50,
        39.0,
        "a block position whose waveform is not the same in every repetition",
        ha="center",
        va="center",
        fontsize=8.2,
        color=INK,
        style="italic",
        zorder=6,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 2.5},
    )

    group(ax, 3, 3, 45, 31, PNS, "stimulation  \u00b7  a peak in time")
    box(
        ax,
        5.5,
        19,
        40,
        10,
        PNS,
        title="one window per shape group",
        body="repetitions are grouped by the shapes their\nwhole TR plays \u2014 at most 64 groups \u2014 and the\nwindow is evaluated once per group",
    )
    box(
        ax,
        5.5,
        5.5,
        40,
        10.4,
        PNS,
        title="then",
        body="keep the worst group; combine the three axes\nby root-sum-square and read the peak",
    )
    arrow(ax, (25.5, 19), (25.5, 15.9), PNS)

    group(ax, 52, 3, 45, 31, MECH, "acoustic  \u00b7  a line spectrum")
    box(
        ax,
        54.5,
        19,
        40,
        10,
        MECH,
        title="a rank basis per position and axis",
        body="the distinct waveforms there are stacked and\ndecomposed \u2014 one transform per basis vector,\nand the truncated tail bounded and added back",
    )
    box(
        ax,
        54.5,
        5.5,
        40,
        10.4,
        MECH,
        title="then",
        body="the largest magnitude among the combinations\nthat occur; the axes sum coherently, because\na rotation mixes them",
    )
    arrow(ax, (74.5, 19), (74.5, 15.9), MECH)

    ax.text(
        50,
        1.1,
        "The identity is shared; what a varying position costs is not. "
        "A set of waveforms occurring at several positions is decomposed once per position.",
        ha="center",
        va="center",
        fontsize=7.6,
        color=INK,
        style="italic",
    )

    fig.tight_layout(pad=0.4)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "shared_memoization.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(build().relative_to(OUT_DIR.parents[1]))
