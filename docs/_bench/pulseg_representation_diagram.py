#!/usr/bin/env python3
"""Figures for ``explanations/background/pulseg.md``.

Documentation-only tooling; not part of the shipped package.

Three pictures of the same move. One takes a single gradient event and shows
which half of it is fixed for the whole scan and which half the playout sets.
The second takes two arbitrary waveforms that share a definition and shows what
that means for the memory they occupy. The third takes a whole scan and shows
what the split does to what has to be prepared.

Usage:
    <venv>/bin/python docs/_bench/pulseg_representation_diagram.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from schematic import data_extent, fit_text  # noqa: E402

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
    """A titled box whose prose is wrapped to the room the box actually has."""
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
    pad_x, pad_y = 0.035 * w, 0.09 * h
    title_art = fit_text(
        ax,
        x + w / 2,
        y + h - pad_y,
        title,
        width=w - 2 * pad_x,
        fontsize=12.2,
        wrap=False,
        color=color,
        fontweight="bold",
        ha="center",
        va="top",
        zorder=3,
    )
    _, title_h = data_extent(ax, title_art)
    top = y + h - pad_y - title_h - 0.55 * pad_y
    fit_text(
        ax,
        x + w / 2,
        top,
        body,
        width=w - 2 * pad_x,
        height=top - y - pad_y,
        fontsize=10.5,
        color=INK,
        ha="center",
        va="top",
        linespacing=1.5,
        zorder=3,
    )


def build_event_split() -> Path:
    fig, (axw, axb) = plt.subplots(
        1, 2, figsize=(9.2, 4.72), gridspec_kw={"width_ratios": [0.92, 1.22]}
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
            (x0 + x1) / 2, -0.86, label, ha="center", va="top", fontsize=10.8, color=DEFN
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
        fontsize=11.1,
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
        fontsize=10.5,
        color=INST,
        linespacing=1.4,
    )
    axw.set_xlim(-70, RISE + PLATEAU + FALL + 190)
    axw.set_title("one gradient event, four playouts", fontsize=12.8, color=INK, pad=6)

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
        body="a trapezoid's rise, flat and fall times; an arbitrary\n"
        "gradient's delay, sample count and time shape; an ADC's\n"
        "dwell and sample count. Converted to hardware units,\n"
        "resampled to the sequencer raster, reserved in memory.",
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
        "the waveform shape id and the rotation; whether the ADC\n"
        "acquires this time round. One row of the scan loop.",
    )

    fig.tight_layout(pad=0.6)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "event_split.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


#: Sample count both arms in ``build_definition_sharing`` are written on.
ARM_SAMPLES = 96


def _arm(turn):
    """One spiral arm's gradient on a single axis, as a sampled waveform."""
    tau = np.linspace(0.0, 1.0, ARM_SAMPLES)
    theta = 7.5 * np.pi * tau + turn
    return tau, tau * np.cos(theta)


def build_definition_sharing() -> Path:
    """Two waveforms, one definition: the same reservation, different samples."""
    fig, (axw, axb) = plt.subplots(
        1, 2, figsize=(8.6, 4.37), gridspec_kw={"width_ratios": [1.0, 1.15]}
    )

    for turn, colour, label in ((0.0, DEFN, "shape 12"), (1.9, INST, "shape 37")):
        tau, g = _arm(turn)
        axw.plot(tau, g, color=colour, linewidth=1.4, zorder=3, label=label)
    axw.axhline(0.0, color=INK, linewidth=0.8, zorder=1)
    axw.axvspan(0.0, 1.0, color=FLAT, alpha=0.10, zorder=0)
    axw.annotate(
        "",
        xy=(1.0, -1.28),
        xytext=(0.0, -1.28),
        arrowprops={"arrowstyle": "<->", "color": FLAT, "lw": 1.1},
    )
    axw.text(
        0.5,
        -1.40,
        f"same delay, same raster, same {ARM_SAMPLES} samples",
        ha="center",
        va="top",
        fontsize=10.8,
        color=FLAT,
    )
    axw.legend(frameon=False, fontsize=10.0, loc="lower left", ncol=2,
           bbox_to_anchor=(0.0, 1.01, 1.0, 0.14), mode="expand",
           borderaxespad=0.0, handlelength=1.2)
    axw.set_ylim(-1.75, 1.25)
    axw.set_xlim(-0.06, 1.06)
    axw.axis("off")
    axw.set_title(
        "two written-out spiral arms, one axis", fontsize=12.8, color=INK, pad=26
    )

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
        title="one definition  \u2014  one prepared object",
        body="delay \u00b7 sample count \u00b7 time shape. The samples are not part\n"
        "of it, so both arms map to the same definition id, and the\n"
        "interpreter reserves one buffer of that length for the block\n"
        "position rather than one per arm.",
    )
    box(
        axb,
        2,
        3.5,
        96,
        15,
        INST,
        title="two shape ids  \u2014  what is written into it",
        body="The shape id is an instance parameter, so it is the scan loop\n"
        "that decides which arm the reserved buffer holds. Two arms\n"
        "differing only in their samples cost one reservation and two\n"
        "shapes, not two reservations.",
    )

    fig.tight_layout(pad=0.6)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "definition_sharing.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def cell(ax, x, y, w, h, color, *, alpha=1.0, label=None, fontsize=9.1):
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
    fig, ax = plt.subplots(figsize=(8.6, 5.68))
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
        fontsize=14.2,
        color=FLAT,
        fontweight="bold",
    )
    ax.text(
        17,
        36.4,
        "one record per block, complete every time",
        ha="center",
        va="center",
        fontsize=10.8,
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
        fontsize=10.5,
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
        fontsize=11.4,
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
        fontsize=14.2,
        color=INK,
        fontweight="bold",
    )
    ax.text(
        62,
        36.4,
        "the fixed half once, the varying half per shot",
        ha="center",
        va="center",
        fontsize=10.8,
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
        fontsize=10.5,
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
        fontsize=10.5,
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
        fontsize=10.8,
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
    for path in (build_event_split(), build_definition_sharing(), build_scan_split()):
        print(path.relative_to(OUT_DIR.parents[1]))
