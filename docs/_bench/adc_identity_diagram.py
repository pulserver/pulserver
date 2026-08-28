#!/usr/bin/env python3
"""Figure for ``explanations/sequence_model/tr_and_segmentation.md``.

Documentation-only tooling; not part of the shipped package.

One block position, played by repetitions that digitise it differently or not
at all, and the three separate questions asked of it. Drawn rather than
plotted, so nothing here reads the package.

Usage:
    <venv>/bin/python docs/_bench/adc_identity_diagram.py
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

PLAIN = "#9a9a9a"
ADC_A = "#4c72b0"
ADC_B = "#dd8452"
NONE = "#d8d8d8"
INK = "#22252a"

#: Which readout the acquiring position carries, repetition by repetition:
#: two preparation shots, then two readouts alternating.
READOUT = [None, None, "A", "B", "A", "B"]
PATTERN = ["RF", "PE", "RO", "sp"]


def cell(
    ax, x, y, w, h, color, label=None, *, fontsize=9.1, dashed=False, textcolor="white"
):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=0.28",
            linewidth=1.0 if dashed else 0.7,
            linestyle="--" if dashed else "-",
            edgecolor=INK if dashed else color,
            facecolor="white" if dashed else color,
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
            color=INK if dashed else textcolor,
            fontweight="bold",
            zorder=4,
        )


def panel(ax, x, y, w, h, color, title):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=1.0",
            linewidth=1.1,
            edgecolor=color,
            facecolor=color,
            alpha=0.07,
            zorder=1,
        )
    )
    ax.text(
        x + w / 2,
        y + h - 1.2,
        title,
        ha="center",
        va="top",
        fontsize=12.2,
        color=color,
        fontweight="bold",
        zorder=4,
    )


def build() -> Path:
    fig, ax = plt.subplots(figsize=(8.6, 6.54))
    ax.set_xlim(0, 100)
    ax.set_ylim(1, 54)
    ax.axis("off")

    # -- the scan, repetition by repetition ------------------------------
    ax.text(
        2,
        52.6,
        "as written",
        ha="left",
        va="center",
        fontsize=11.6,
        color=INK,
        fontweight="bold",
    )
    ax.text(
        2,
        50.0,
        "two preparation shots that acquire nothing, then two readouts "
        "of different length, alternating",
        ha="left",
        va="center",
        fontsize=9.8,
        color=INK,
        style="italic",
    )

    w, gap = 3.4, 0.5
    x0 = 2.0
    for r, adc in enumerate(READOUT):
        for c, lbl in enumerate(PATTERN):
            x = x0 + (r * len(PATTERN) + c) * (w + gap)
            if c != 2:
                cell(ax, x, 45.0, w, 3.4, PLAIN, lbl)
            elif adc is None:
                cell(ax, x, 45.0, w, 3.4, NONE, "—", textcolor=INK)
            else:
                cell(ax, x, 45.0, w, 3.4, ADC_A if adc == "A" else ADC_B, "RO" + adc)
    ax.text(
        2,
        43.4,
        "\u2014  no ADC        RO A, RO B  the two readouts",
        ha="left",
        va="top",
        fontsize=10.2,
        color=INK,
    )

    for cx in (18.0, 50.0, 82.0):
        ax.add_patch(
            FancyArrowPatch(
                (cx, 41.6),
                (cx, 33.0),
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1.2,
                color=INK,
                zorder=4,
            )
        )

    pw, ph, py = 31.0, 26.5, 5.5

    # -- 1. the repeating unit ------------------------------------------
    panel(ax, 2.5, py, pw, ph, PLAIN, "what repeats")
    for c, lbl in enumerate(PATTERN):
        cell(
            ax,
            5.0 + c * 6.6,
            25.0,
            5.8,
            3.6,
            PLAIN if c != 2 else "white",
            lbl if c != 2 else "RO",
            dashed=(c == 2),
            fontsize=9.9,
        )
    ax.text(
        18.0,
        23.2,
        "the ADC is left out of the identity",
        ha="center",
        va="top",
        fontsize=9.8,
        color=INK,
        style="italic",
    )
    ax.text(
        18.0,
        19.6,
        "Which readout a block digitises\nwith is carried by the instance,\n"
        "so all six repetitions are the\nsame four blocks. The period is\n"
        "one repetition, not two, and\nnot six.",
        ha="center",
        va="top",
        fontsize=9.4,
        color=INK,
        linespacing=1.6,
    )

    # -- 2. the segments -------------------------------------------------
    panel(ax, 34.5, py, pw, ph, ADC_A, "what is prepared")
    for row, (adc, col) in enumerate((("A", ADC_A), ("B", ADC_B))):
        yy = 25.0 - row * 4.6
        for c, lbl in enumerate(PATTERN):
            x = 36.6 + c * 4.9
            cell(
                ax,
                x,
                yy,
                4.3,
                3.4,
                col if c == 2 else PLAIN,
                ("RO" + adc) if c == 2 else lbl,
                fontsize=8.8,
            )
        ax.text(
            36.6 + 4 * 4.9 + 0.4,
            yy + 1.7,
            f"seg {row}",
            ha="left",
            va="center",
            fontsize=9.4,
            color=col,
        )
    ax.text(
        50.0,
        17.4,
        "A prepared segment binds one\nreceive filter to each of its\n"
        "block positions, so the two\nreadouts cannot share one. Split\n"
        "by the readouts a repetition\nactually plays: two segments.",
        ha="center",
        va="top",
        fontsize=9.4,
        color=INK,
        linespacing=1.6,
    )

    # -- 3. the dummies --------------------------------------------------
    panel(ax, 66.5, py, pw, ph, PLAIN, "where a dummy goes")
    for c, lbl in enumerate(PATTERN):
        cell(
            ax,
            69.0 + c * 6.6,
            25.0,
            5.8,
            3.6,
            PLAIN if c != 2 else NONE,
            lbl if c != 2 else "\u2014",
            fontsize=9.9,
            textcolor="white" if c != 2 else INK,
        )
    ax.add_patch(
        FancyArrowPatch(
            (83.5, 24.8),
            (76.5, 21.8),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.1,
            color=ADC_A,
            zorder=4,
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (86.7, 24.8),
            (93.0, 21.8),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.1,
            color=ADC_B,
            zorder=4,
        )
    )
    ax.text(72.5, 21.0, "segment 0", ha="center", va="top", fontsize=9.4, color=ADC_A)
    ax.text(91.0, 21.0, "segment 1", ha="center", va="top", fontsize=9.4, color=ADC_B)
    ax.text(
        82.0,
        17.4,
        "Either. It acquires nothing\nthere, so neither segment's\n"
        "filter is used for it and both\nplay its gradients and RF\n"
        "identically. The choice is not\nobservable.",
        ha="center",
        va="top",
        fontsize=9.4,
        color=INK,
        linespacing=1.6,
    )

    fig.tight_layout(pad=0.5)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "adc_identities.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(build().relative_to(OUT_DIR.parents[1]))
