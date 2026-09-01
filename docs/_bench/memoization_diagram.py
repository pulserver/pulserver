#!/usr/bin/env python3
"""Memoization figures for the two gradient-side performance pages.

One figure per page, each showing only its own check:

* ``explanations/assets/pns_performance/memoization.png``
* ``explanations/assets/mechanical_resonance/memoization.png``

Both open on the same reading of the TR -- the gradient columns of the block
table, gathered by position and deduplicated on the ``(gx, gy, gz)`` tuple --
and then follow that page's own route to a response.

Schematics, not measurements: drawn rather than plotted, so nothing here reads
the package.

Usage:
    <venv>/bin/python docs/_bench/memoization_diagram.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

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


ASSETS = Path(__file__).resolve().parents[1] / "explanations" / "assets"

SHARED = "#4a4a4a"
PNS = "#4c72b0"
MECH = "#dd8452"
VARY = "#c44e52"
INK = "#22252a"
MUTED = "#7b8288"

MONO = {"family": "DejaVu Sans Mono"}

#: One TR of a stack-of-spirals shot, gradient columns only, as ``def/shape``.
#: Position 1 and 2 are the two readout blocks whose arms are written out shot
#: by shot, so their tuples differ from instance to instance.
TR_POSITIONS = ["excitation", "readout", "readout", "spoiler"]
TR_INSTANCES = [
    ("TR 0", [("—", "—", "3/0"), ("1/1", "2/2", "—"), ("1/1", "2/2", "—"), ("4/0", "4/0", "4/0")]),
    ("TR 1", [("—", "—", "3/0"), ("1/4", "2/5", "—"), ("1/4", "2/5", "—"), ("4/0", "4/0", "4/0")]),
    ("TR k", [("—", "—", "3/0"), ("1/n", "2/m", "—"), ("1/n", "2/m", "—"), ("4/0", "4/0", "4/0")]),
]
#: Distinct tuples each position takes across the whole scan.
TR_MULTIPLICITY = [1, "K", "K", 1]


def box(ax, x, y, w, h, color, *, title, body, title_size=10.0, body_size=8.8):
    """A titled box; returns its centre."""
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
        y + h - 1.3,
        title,
        ha="center",
        va="top",
        fontsize=title_size,
        color=color,
        fontweight="bold",
        zorder=3,
    )
    ax.text(
        x + w / 2,
        y + h - 3.9,
        body,
        ha="center",
        va="top",
        fontsize=body_size,
        color=INK,
        zorder=3,
        linespacing=1.5,
    )
    return x + w / 2, y + h / 2


#: Room a `group` band's label needs inside its own top edge, in axes units.
#: A box row opened above this runs into the label, which is a silent failure:
#: the figure still renders. Both builders derive their first row from it.
GROUP_LABEL_STRIP = 3.0

#: Where the band that encloses the reasoning starts, relative to the head.
GROUP_TOP = -3.0


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
        y + h - 1.2,
        label,
        ha="center",
        va="top",
        fontsize=11.2,
        color=color,
        fontweight="bold",
        zorder=3,
    )


def arrow(ax, start, end, color, *, rad=0.0, label=None, label_pos=None, size=8.0):
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
            va="center",
            fontsize=size,
            color=color,
            zorder=5,
            linespacing=1.4,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.8},
        )


def draw_tr_table(ax, x0, y_top):
    """The gradient columns of one TR, played three times.

    Returns the y of the table's bottom edge.
    """
    row_h = 2.5
    lab_w = 12.0
    col_w = 5.4
    gap = 2.2

    ax.text(
        x0,
        y_top + 5.0,
        "the block table's gradient columns, as  definition / shape",
        ha="left",
        va="bottom",
        fontsize=12.2,
        color=SHARED,
        fontweight="bold",
    )

    # Instance headers and per-axis column headers.
    for inst, (name, _) in enumerate(TR_INSTANCES):
        bx = x0 + lab_w + inst * (3 * col_w + gap)
        ax.text(
            bx + 1.5 * col_w,
            y_top + 1.9,
            name,
            ha="center",
            va="bottom",
            fontsize=11.4,
            color=INK,
        )
        for ax_i, axis in enumerate(("gx", "gy", "gz")):
            ax.text(
                bx + (ax_i + 0.5) * col_w,
                y_top + 0.35,
                axis,
                ha="center",
                va="bottom",
                fontsize=10.2,
                color=MUTED,
                **MONO,
            )

    for row, name in enumerate(TR_POSITIONS):
        y = y_top - (row + 1) * row_h
        varies = TR_MULTIPLICITY[row] != 1
        if row % 2 == 0:
            ax.add_patch(
                Rectangle(
                    (x0, y),
                    lab_w + len(TR_INSTANCES) * (3 * col_w + gap) - gap,
                    row_h,
                    facecolor="#f2f3f5",
                    edgecolor="none",
                    zorder=0,
                )
            )
        ax.text(
            x0 + 0.4,
            y + row_h / 2,
            f"{row}  {name}",
            ha="left",
            va="center",
            fontsize=10.5,
            color=INK,
        )
        for inst, (_, rows) in enumerate(TR_INSTANCES):
            bx = x0 + lab_w + inst * (3 * col_w + gap)
            if varies:
                ax.add_patch(
                    Rectangle(
                        (bx, y + 0.25),
                        3 * col_w,
                        row_h - 0.5,
                        facecolor=VARY,
                        edgecolor="none",
                        alpha=0.13,
                        zorder=1,
                    )
                )
            for ax_i, cell in enumerate(rows[row]):
                ax.text(
                    bx + (ax_i + 0.5) * col_w,
                    y + row_h / 2,
                    cell,
                    ha="center",
                    va="center",
                    fontsize=10.2,
                    color=INK if cell != "—" else MUTED,
                    **MONO,
                )
    return y_top - len(TR_POSITIONS) * row_h


def draw_positions(ax, x0, y_top, accent):
    """One box per block position, carrying its tuple multiplicity."""
    w, h, gap = 44.0, 7.2, 4.0
    for pos, name in enumerate(TR_POSITIONS):
        x = x0 + (pos % 2) * (w + gap)
        y = y_top - h - (pos // 2) * (h + 2.6)
        mult = TR_MULTIPLICITY[pos]
        varies = mult != 1
        colour = accent if varies else SHARED
        box(
            ax,
            x,
            y,
            w,
            h,
            colour,
            title=f"position {pos} \u00b7 {name}",
            body=f"{mult} distinct tuples" if varies else f"{mult} distinct tuple",
            title_size=10.0,
            body_size=9.2,
        )
    return y_top - 2 * h - 2.6


def draw_head(ax, accent, title):
    """The reading of the TR both checks share. Returns the y to build under."""
    ax.text(
        3,
        97.5,
        title,
        ha="left",
        va="top",
        fontsize=13.0,
        color=INK,
        fontweight="bold",
    )
    bottom = draw_tr_table(ax, 4.0, 87.5)

    ax.text(
        50.0,
        bottom - 2.0,
        "gather by position across the instances,\n"
        "deduplicate on the (gx, gy, gz) tuple",
        ha="center",
        va="top",
        fontsize=10.2,
        color=SHARED,
        linespacing=1.5,
    )
    arrow(ax, (50.0, bottom - 8.6), (50.0, bottom - 11.2), SHARED)
    return draw_positions(ax, 4.0, bottom - 12.2, accent)


def new_axes():
    fig, ax = plt.subplots(figsize=(8.6, 10.71))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


def build_pns() -> Path:
    fig, ax = new_axes()
    y = draw_head(ax, PNS, "Stimulation: what one canonical TR is convolved from")

    top = y + GROUP_TOP - GROUP_LABEL_STRIP
    box(
        ax,
        4.5,
        top - 12.5,
        43,
        11.5,
        PNS,
        title="every position with one tuple",
        body="the amplitude is the only thing that varies.\n"
        "The window takes the largest each position\nreaches, sign kept: a bound, because the same\n"
        "shape driven harder is a larger response.",
    )
    box(
        ax,
        52.5,
        top - 12.5,
        43,
        11.5,
        VARY,
        title="any position with K tuples",
        body="there is no amplitude at which one arm's shape\n"
        "covers another's. The instances are classified by\n"
        "their whole tuple sequence into G ≤ 64 groups,\n"
        "and one window is built per group.",
    )

    mid = top - 13.5
    box(
        ax,
        5.5,
        mid - 11.0,
        89,
        11.0,
        PNS,
        title="inside one window: one convolution per distinct waveform",
        body="Each block's slice is keyed on (gradient definition · shape id · block duration), the identity the\n"
        "representation already carries. Each distinct key is convolved with the model kernel once;\n"
        "every occurrence of it is a scaled, shifted add of the stored result.  Time domain, per axis.",
    )

    bot = mid - 13.2
    box(
        ax,
        5.5,
        bot - 8.0,
        89,
        8.0,
        PNS,
        title="the verdict",
        body="Combine the three axes by root-sum-square at every instant — this is the step that does not\n"
        "decompose — read the peak, and keep the worst group.",
    )

    group(ax, 3, bot - 9.6, 94, y + GROUP_TOP - (bot - 9.6), PNS,
          "a peak in time — no position is bounded on its own")

    ax.text(
        50,
        bot - 12.6,
        "Grouping costs a full window per group. It is the honest route:\n"
        "a stimulation peak is a property of the whole window, not of one\n"
        "position, so there is no per-position bound to take instead.",
        ha="center",
        va="center",
        fontsize=9.8,
        color=INK,
        style="italic",
    )

    out = ASSETS / "pns_performance" / "memoization.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.4)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def build_mech() -> Path:
    fig, ax = new_axes()
    y = draw_head(ax, MECH, "Acoustic: what one canonical TR is transformed from")

    top = y + GROUP_TOP - GROUP_LABEL_STRIP
    box(
        ax,
        4.5,
        top - 12.5,
        43,
        11.5,
        MECH,
        title="every position with one tuple",
        body="one complex contribution, so the position joins\n"
        "the coherent sum: its three axes are mixed by its\n"
        "rotation and phased by its start time.",
    )
    box(
        ax,
        52.5,
        top - 12.5,
        43,
        11.5,
        VARY,
        title="any position with K tuples",
        body="no single contribution, so the position is bounded\n"
        "instead: the largest magnitude among the K tuples\nthat really occur — exact for the position,\n"
        "whatever the rotation mixes.",
    )

    mid = top - 13.5
    box(
        ax,
        51.5,
        mid - 12.5,
        43,
        12.5,
        VARY,
        title="and, per axis, a rank basis",
        body="Stack that axis's distinct waveforms and\n"
        "decompose. Taken only at 4 or more sharing one\n"
        "sampling, and only if the rank is at most half of\n"
        "them: rank transforms then replace K.",
    )
    box(
        ax,
        5.5,
        mid - 12.5,
        43,
        12.5,
        MECH,
        title="one transform per distinct waveform",
        body="Base transforms are memoized on the waveform\n"
        "identity across the whole TR, basis vectors\n"
        "included, so a waveform at several positions is\n"
        "transformed once.",
    )

    bot = mid - 14.0
    box(
        ax,
        5.5,
        bot - 8.0,
        89,
        8.0,
        MECH,
        title="the verdict",
        body="At each harmonic of the TR inside a guarded band: the coherent sum, plus the bound of every\n"
        "varying position, plus the truncated tail of every basis. Held against the band's threshold.",
    )

    group(ax, 3, bot - 9.6, 94, y + GROUP_TOP - (bot - 9.6), MECH,
          "a line spectrum — every position bounded on its own")

    ax.text(
        50,
        bot - 12.6,
        "The basis is built per position and per axis, so a family of waveforms\n"
        "appearing at several positions is decomposed once for each of them;\n"
        "only the transforms are shared. A truncated tail is bounded and added\n"
        "back, so a compressed position reads louder, never quieter.",
        ha="center",
        va="center",
        fontsize=9.8,
        color=INK,
        style="italic",
    )

    out = ASSETS / "mechanical_resonance" / "memoization.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.4)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


if __name__ == "__main__":
    for path in (build_pns(), build_mech()):
        print(path.relative_to(ASSETS.parents[1]))
