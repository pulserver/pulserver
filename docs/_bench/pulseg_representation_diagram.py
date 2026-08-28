#!/usr/bin/env python3
"""Figures for ``explanations/background/pulseg.md``.

Documentation-only tooling; not part of the shipped package.

Two pictures of the same move. One takes a single gradient event and shows
which half of it is fixed for the whole scan and which half the playout sets.
The other takes a whole scan and shows what that split does to what has to be
prepared.

Usage:
    <venv>/bin/python docs/_bench/pulseg_representation_diagram.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_DIR = Path(__file__).resolve().parents[1] / "explanations" / "assets" / "pulseg"

DEFN = "#4c72b0"  # prepared once
INST = "#dd8452"  # applied per playout
FLAT = "#8c8c8c"  # pulseq, undifferentiated
INK = "#22252a"

#: rise, flat, fall (us) of the trapezoid both figures are drawn on.
RISE, PLATEAU, FALL = 200.0, 600.0, 200.0


def trapezoid(amplitude):
    """Corner times (us) and amplitudes of one trapezoid."""
    t = np.array([0.0, RISE, RISE + PLATEAU, RISE + PLATEAU + FALL])
    return t, np.array([0.0, amplitude, amplitude, 0.0])


def box(ax, x, y, w, h, color, *, title, body, alpha=1.0):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=0.9",
            linewidth=1.2,
            edgecolor=color,
            facecolor="white",
            alpha=alpha,
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


def build_event_split() -> Path:
    fig, (axw, axb) = plt.subplots(
        1, 2, figsize=(10.6, 3.9), gridspec_kw={"width_ratios": [1.05, 1.0]}
    )

    # -- the waveform, one shape at four amplitudes ----------------------
    for amp, alpha in ((1.0, 1.0), (0.62, 0.5), (0.28, 0.5), (-0.45, 0.5)):
        t, g = trapezoid(amp)
        axw.plot(
            t,
            g,
            color=INST if alpha < 1 else DEFN,
            linewidth=2.0 if alpha == 1.0 else 1.3,
            alpha=alpha,
            zorder=3,
        )
    axw.axhline(0.0, color=INK, linewidth=0.8, zorder=1)

    # the definition: the corner times
    for x0, x1, label in (
        (0.0, RISE, "rise"),
        (RISE, RISE + PLATEAU, "flat"),
        (RISE + PLATEAU, RISE + PLATEAU + FALL, "fall"),
    ):
        axw.annotate(
            "",
            xy=(x1, -0.72),
            xytext=(x0, -0.72),
            arrowprops={"arrowstyle": "<->", "color": DEFN, "lw": 1.1},
        )
        axw.text(
            (x0 + x1) / 2, -0.86, label, ha="center", va="top", fontsize=7.6, color=DEFN
        )

    # the instance parameter: the height
    axw.annotate(
        "",
        xy=(RISE + PLATEAU * 0.55, 1.0),
        xytext=(RISE + PLATEAU * 0.55, 0.0),
        arrowprops={"arrowstyle": "<->", "color": INST, "lw": 1.3},
    )
    axw.text(
        RISE + PLATEAU * 0.55 + 22,
        0.5,
        "amplitude",
        ha="left",
        va="center",
        fontsize=7.8,
        color=INST,
    )

    axw.set_ylim(-1.15, 1.35)
    axw.axis("off")
    axw.text(
        RISE + PLATEAU + FALL + 30,
        -0.42,
        "other\nplayouts",
        ha="left",
        va="center",
        fontsize=7.4,
        color=INST,
        linespacing=1.4,
    )
    axw.set_xlim(-70, RISE + PLATEAU + FALL + 190)
    axw.set_title("one gradient event, four playouts", fontsize=9.0, color=INK, pad=6)

    # -- what each half holds -------------------------------------------
    axb.set_xlim(0, 100)
    axb.set_ylim(0, 40)
    axb.axis("off")
    box(
        axb,
        2,
        23,
        96,
        15,
        DEFN,
        title="definition  \u2014  prepared once",
        body="rise, flat and fall times, or the normalised samples of an\n"
        "arbitrary waveform; an RF envelope; an ADC's dwell and\n"
        "sample count. Converted to hardware units, resampled to\n"
        "the sequencer raster, loaded into pulse-generator memory.",
    )
    box(
        axb,
        2,
        3.5,
        96,
        15,
        INST,
        title="instance parameter  \u2014  applied per playout",
        body="the amplitude; RF amplitude, phase and frequency offset;\n"
        "the shot index that selects a waveform variant; the\n"
        "rotation; whether the ADC acquires this time round.\n"
        "One row of the scan loop.",
    )

    fig.tight_layout(pad=0.6)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "event_split.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def cell(ax, x, y, w, h, color, *, alpha=1.0, label=None, fontsize=6.4):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=0.28",
            linewidth=0.7,
            edgecolor=color,
            facecolor=color,
            alpha=alpha,
            zorder=2,
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
            zorder=3,
            fontweight="bold",
        )


def build_scan_split() -> Path:
    """What the split does to a whole scan."""
    fig, ax = plt.subplots(figsize=(11.0, 4.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 42)
    ax.axis("off")

    events = ["RF", "PE", "RO", "SP"]
    nshot = 6
    cw, ch, gap = 6.2, 3.0, 0.7

    # -- Pulseq: every block a complete record --------------------------
    ax.text(
        17,
        39.5,
        "Pulseq",
        ha="center",
        va="center",
        fontsize=10.0,
        color=FLAT,
        fontweight="bold",
    )
    ax.text(
        17,
        36.4,
        "one record per block, complete every time",
        ha="center",
        va="center",
        fontsize=7.6,
        color=INK,
        style="italic",
    )
    for r in range(nshot):
        y = 30 - r * (ch + gap)
        for c, e in enumerate(events):
            cell(ax, 3 + c * (cw + gap), y, cw, ch, FLAT, alpha=0.75, label=e)
    ax.text(
        17,
        30 - nshot * (ch + gap) - 1.2,
        f"{len(events)} \u00d7 {nshot} records, and one number in each row differs",
        ha="center",
        va="top",
        fontsize=7.4,
        color=INK,
    )

    ax.add_patch(
        FancyArrowPatch(
            (34, 18),
            (44, 18),
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=1.6,
            color=INK,
            zorder=4,
        )
    )
    ax.text(
        39,
        19.4,
        "one split",
        ha="center",
        va="bottom",
        fontsize=8.0,
        color=INK,
        fontweight="bold",
    )

    # -- PulSeg: definitions once, instances per shot --------------------
    ax.text(
        62,
        39.5,
        "PulSeg",
        ha="center",
        va="center",
        fontsize=10.0,
        color=INK,
        fontweight="bold",
    )
    ax.text(
        62,
        36.4,
        "the fixed half once, the varying half per shot",
        ha="center",
        va="center",
        fontsize=7.6,
        color=INK,
        style="italic",
    )

    for c, e in enumerate(events):
        cell(ax, 48 + c * (cw + gap), 30, cw, ch, DEFN, label=e)
    ax.text(
        48 + 4 * (cw + gap) + 2.0,
        31.5,
        "base blocks\nprepared once",
        ha="left",
        va="center",
        fontsize=7.4,
        color=DEFN,
        linespacing=1.4,
    )

    for r in range(nshot):
        y = 25.4 - r * (ch * 0.62 + gap)
        for c in range(len(events)):
            cell(ax, 48 + c * (cw + gap), y, cw, ch * 0.62, INST, alpha=0.85)
    ax.text(
        48 + 4 * (cw + gap) + 2.0,
        25.4 - (nshot - 1) * (ch * 0.62 + gap) / 2.0,
        "instance rows\napplied per shot",
        ha="left",
        va="center",
        fontsize=7.4,
        color=INST,
        linespacing=1.4,
    )

    ax.text(
        50,
        3.0,
        "The left column grows with the scan and every row of it has to be "
        "prepared.\nOn the right only the orange grows, and applying a row is "
        "writing a number.",
        ha="center",
        va="center",
        fontsize=7.6,
        color=INK,
        linespacing=1.6,
    )

    fig.tight_layout(pad=0.5)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "scan_split.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


if __name__ == "__main__":
    for path in (build_event_split(), build_scan_split()):
        print(path.relative_to(OUT_DIR.parents[1]))
