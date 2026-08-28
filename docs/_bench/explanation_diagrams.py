#!/usr/bin/env python3
"""Schematics for the explanation pages that had none.

Documentation-only tooling; not part of the shipped package. Nothing here
reads the package: each figure draws a structure the prose describes, so it
stays correct as long as the prose does.

Usage:
    <venv>/bin/python docs/_bench/explanation_diagrams.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ASSETS = Path(__file__).resolve().parents[1] / "explanations" / "assets"

INK = "#22252a"
MUTED = "#7b8288"
BLUE = "#4c72b0"
GREEN = "#55a868"
ORANGE = "#dd8452"
RED = "#c0524a"
PURPLE = "#8172b3"
GREY = "#9aa0a6"

FS_TITLE = 13.0
FS_LABEL = 11.5
FS_BODY = 10.5
FS_SMALL = 9.5
MONO = {"family": "DejaVu Sans Mono"}

plt.rcParams.update(
    {
        "font.size": FS_BODY,
        "axes.titlesize": FS_TITLE,
        "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_BODY,
        "ytick.labelsize": FS_BODY,
        "legend.fontsize": FS_BODY,
    }
)


# ---------------------------------------------------------------------------
#  Shared drawing helpers
# ---------------------------------------------------------------------------


def canvas(w, h, xmax=100.0, ymax=100.0):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, colour, *, title=None, body=None, fill="white",
        alpha=1.0, lw=1.4, mono=False, title_size=FS_LABEL, body_size=FS_BODY):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0,rounding_size=0.8",
            facecolor=fill, edgecolor=colour, lw=lw, alpha=alpha, zorder=2,
        )
    )
    top = y + h
    if title is not None:
        top -= 1.8
        ax.text(x + w / 2, top, title, ha="center", va="top",
                fontsize=title_size, color=colour, fontweight="bold", zorder=3)
        top -= 2.4
    if body is not None:
        ax.text(x + w / 2, top if title else y + h - 1.8, body,
                ha="center", va="top", fontsize=body_size, color=INK,
                linespacing=1.45, zorder=3, **(MONO if mono else {}))
    return x + w / 2, y + h / 2


def band(ax, x, y, w, h, colour, label, *, label_size=FS_LABEL):
    ax.add_patch(
        FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0",
                       facecolor=colour, edgecolor="none", alpha=0.10, zorder=1)
    )
    ax.text(x + 1.6, y + h - 1.6, label, ha="left", va="top",
            fontsize=label_size, color=colour, fontweight="bold", zorder=3)


def arrow(ax, a, b, colour, *, dashed=False, lw=1.4, rad=0.0, head="-|>"):
    ax.add_patch(
        FancyArrowPatch(
            a, b, arrowstyle=head, mutation_scale=13, lw=lw, color=colour,
            connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2, zorder=4,
            linestyle=(0, (4, 2)) if dashed else "solid",
        )
    )


def save(fig, rel):
    out = ASSETS / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(out.relative_to(ASSETS.parents[1]))
    return out


# ---------------------------------------------------------------------------
#  background/pulseq.md
# ---------------------------------------------------------------------------


def pulseq_file_structure():
    """The block table, the event libraries it indexes, and the shapes."""
    fig, ax = canvas(8.6, 7.6, ymax=76)

    rows = [
        ("1", "694", "1", "0", "0", "1", "0", "2"),
        ("2", "356", "0", "2", "3", "4", "0", "0"),
        ("3", "268", "0", "5", "0", "0", "1", "0"),
    ]
    cols = ["NUM", "DUR", "RF", "GX", "GY", "GZ", "ADC", "EXT"]
    cw, rh, x0, ytop = 6.0, 4.6, 22.0, 71.0

    ax.text(3.0, ytop + 3.4, "[BLOCKS] — the played order, one row per block",
            fontsize=FS_LABEL, color=BLUE, fontweight="bold", va="bottom")
    for c, name in enumerate(cols):
        ax.text(x0 + (c + 0.5) * cw, ytop + 0.5, name, ha="center", va="bottom",
                fontsize=FS_SMALL, color=MUTED, **MONO)
    for r, cells in enumerate(rows):
        y = ytop - (r + 1) * rh
        if r % 2 == 0:
            ax.add_patch(Rectangle((x0, y), len(cols) * cw, rh,
                                   facecolor="#f2f3f5", edgecolor="none"))
        for c, cell in enumerate(cells):
            ax.text(x0 + (c + 0.5) * cw, y + rh / 2, cell, ha="center",
                    va="center", fontsize=FS_BODY, color=INK, **MONO)

    ax.text(50.0, ytop - 3 * rh - 2.2,
            "every cell is an id — an event used ten thousand times is "
            "written once",
            ha="center", va="top", fontsize=FS_BODY, color=MUTED)

    libs = [
        ("[ADC]", "samples, dwell,\ndelay, offsets", ORANGE, 3.0, 31.0),
        ("[EXTENSIONS]", "one linked list per block:\nlabels, rotation, "
         "trigger", PURPLE, 52.0, 31.0),
        ("[RF]", "amplitude, shape ids,\ndelay, offsets, use", BLUE, 3.0, 15.0),
        ("[TRAP], [GRADIENTS]", "amplitude with\nrise, flat and fall,\n"
         "or a shape id", GREEN, 52.0, 15.0),
    ]
    for name, body, colour, x, y in libs:
        box(ax, x, y, 45.0, 13.0, colour, title=name, body=body,
            title_size=FS_BODY, body_size=FS_SMALL)

    for x in (25.5, 74.5):
        arrow(ax, (x, 49.0), (x, 44.0), MUTED)

    box(ax, 14.0, 0.5, 72.0, 9.5, RED, title="[SHAPES]",
        body="run-length encoded on the derivative, so a thousand-sample\n"
             "linear ramp costs three numbers",
        title_size=FS_BODY, body_size=FS_SMALL)
    arrow(ax, (25.5, 15.0), (32.0, 10.0), RED, rad=0.10)
    arrow(ax, (74.5, 15.0), (68.0, 10.0), RED, rad=-0.10)

    return save(fig, "pulseq/file_structure.png")


def structure_levels():
    """The four levels at which the format leaves the structure implicit."""
    fig, ax = canvas(8.6, 6.29, ymax=52)

    n, w, gap, x0, y = 12, 6.6, 1.0, 3.0, 34.0
    kinds = ["RF", "PE", "RO", "sp"] * 3
    colours = {"RF": BLUE, "PE": GREEN, "RO": RED, "sp": GREY}
    for k in range(n):
        x = x0 + k * (w + gap)
        ax.add_patch(FancyBboxPatch((x, y), w, 6.0,
                                    boxstyle="round,pad=0,rounding_size=0.6",
                                    facecolor=colours[kinds[k]], alpha=0.75,
                                    edgecolor="black", lw=0.6, zorder=2))
        ax.text(x + w / 2, y + 3.0, kinds[k], ha="center", va="center",
                fontsize=FS_SMALL, color="white", fontweight="bold", zorder=3)
    ax.text(x0, y + 12.6, "what the file says: an ordered list of blocks",
            fontsize=FS_LABEL, color=INK, fontweight="bold", va="bottom")

    # One column for every label, to the right of the longest bracket.
    label_x = x0 + 4 * (w + gap) + 2.0

    ax.annotate("", xy=(x0 + w / 2, y + 6.4), xytext=(x0 + w / 2, y + 9.6),
                arrowprops={"arrowstyle": "->", "color": PURPLE, "lw": 1.4})
    ax.text(x0 + w + 2.0, y + 9.6, "within an event — which part is the fixed "
            "skeleton, and which varies per playout", ha="left", va="center",
            fontsize=FS_BODY, color=PURPLE)

    levels = [
        (1, BLUE, "the block — that these events recur together, as a unit"),
        (3, GREEN, "the segment — that this run is preparable once and replayed"),
        (4, ORANGE, "the TR — which period the whole scan is a repetition of"),
    ]
    for i, (span, colour, note) in enumerate(levels):
        yy = 26.0 - i * 8.0
        ax.annotate("", xy=(x0 + span * (w + gap) - gap, yy),
                    xytext=(x0, yy),
                    arrowprops={"arrowstyle": "|-|", "color": colour, "lw": 1.6})
        ax.text(label_x, yy, note, ha="left", va="center", fontsize=FS_BODY,
                color=colour)

    ax.text(50, 1.0, "None of the four is a field in the file. Each is implicit "
            "in the content, and both a segmenting sequencer\nand every safety "
            "check need it before they can do anything.",
            ha="center", va="bottom", fontsize=FS_BODY, color=INK,
            style="italic", linespacing=1.5)
    return save(fig, "pulseq/structure_levels.png")


# ---------------------------------------------------------------------------
#  background/fov_transformation.md
# ---------------------------------------------------------------------------


def transform_fov_walk():
    """What `mr.TransformFOV` does to one block, and what it carries onward."""
    fig, ax = canvas(8.6, 6.6, ymax=68)

    box(ax, 2.0, 56.0, 96.0, 10.0, GREY, title=None,
        body="one block of an already-built sequence:  RF · ADC · Gx · Gy · Gz",
        body_size=FS_BODY)

    box(ax, 2.0, 41.0, 96.0, 12.0, BLUE, title=None,
        body="per axis with a shift — build the piecewise polynomial of the "
             "gradient,\nintegrate it analytically. That integral is the "
             "k-space position the shift\nhas to be paid against.",
        body_size=FS_BODY)
    arrow(ax, (50.0, 56.0), (50.0, 53.0), MUTED)

    ax.text(50.0, 38.0, "is the gradient constant across the event?",
            ha="center", va="center", fontsize=FS_LABEL, color=INK,
            fontweight="bold")

    box(ax, 2.0, 18.0, 45.0, 16.0, GREEN, title="yes — two scalars",
        body="a frequency offset and a\nphase offset on the RF and\n"
             "the ADC. The file needs\nnothing downstream.",
        title_size=FS_BODY, body_size=FS_SMALL)
    box(ax, 53.0, 18.0, 45.0, 16.0, ORANGE, title="no — a phase per sample",
        body="multiplied into rf.signal;\nfor the ADC, stored in\n"
             "adc.phaseModulation for\nthe reconstruction to undo.",
        title_size=FS_BODY, body_size=FS_SMALL)
    arrow(ax, (40.0, 36.6), (24.5, 34.0), GREEN, rad=0.12)
    arrow(ax, (60.0, 36.6), (75.5, 34.0), ORANGE, rad=-0.12)

    box(ax, 2.0, 4.0, 96.0, 10.0, PURPLE, title=None,
        body="carry the accumulated phase into the next block — this is "
             "where absolute k stands,\nand it is what makes the walk "
             "sequential",
        body_size=FS_BODY)
    arrow(ax, (24.5, 18.0), (40.0, 14.0), MUTED, rad=-0.10)
    arrow(ax, (75.5, 18.0), (60.0, 14.0), MUTED, rad=0.10)

    return save(fig, "pulseq/transform_fov_walk.png")


# ---------------------------------------------------------------------------
#  sequence_model/pulseg_representation.md
# ---------------------------------------------------------------------------


def pulseg_mapping():
    """The four PulSeg structures, and the one Pulserver adds above them."""
    fig, ax = canvas(8.6, 6.2, ymax=62)

    band(ax, 2, 2, 96, 38, BLUE, "the specification's four structures")
    items = [
        ("BaseBlock", "one block's definitions\nand its duration", 5.0, 21.0),
        ("VirtualSegment", "an ordered list of\nbase block ids", 52.0, 21.0),
        ("SegmentInstance", "the per-playout row:\namplitudes, offsets,\n"
         "shape id, rotation", 5.0, 4.5),
        ("ExecutionStream", "which instance plays\nwhen, stored as runs",
         52.0, 4.5),
    ]
    for name, body, x, y in items:
        box(ax, x, y, 43.0, 13.0, BLUE, title=name, body=body,
            title_size=FS_BODY, body_size=FS_SMALL)
    arrow(ax, (48.0, 27.5), (52.0, 27.5), BLUE)
    arrow(ax, (48.0, 11.0), (52.0, 11.0), BLUE)
    arrow(ax, (73.5, 21.0), (73.5, 17.5), BLUE)

    band(ax, 2, 42, 96, 18, ORANGE, "what Pulserver adds")
    box(ax, 5.0, 43.5, 90.0, 10.5, ORANGE, title=None,
        body="the structural TR — the shortest block period the normalised "
             "structure repeats\nover. Detected from the content, never read "
             "off an annotation, and written\nback as [DEFINITIONS] TRSize "
             "for whoever reads the file next.",
        body_size=FS_SMALL)
    arrow(ax, (50.0, 43.5), (50.0, 40.0), ORANGE)
    return save(fig, "pulseg/pulserver_mapping.png")


# ---------------------------------------------------------------------------
#  sequence_model/tr_and_segmentation.md
# ---------------------------------------------------------------------------


def segmentation_rules():
    """Where a boundary is allowed, and where one is placed."""
    fig, ax = canvas(8.6, 5.6, ymax=56)

    x0, w, y = 4.0, 9.2, 30.0
    t = np.linspace(0.0, 1.0, 60)
    trap = np.clip(np.minimum(t / 0.25, (1.0 - t) / 0.25), 0.0, 1.0)

    # amplitude, has_rf, has_adc, ends_at_zero
    blocks = [
        (0.0, 1, 0, 1),
        (0.6, 0, 0, 1),
        (1.0, 0, 1, 0),
        (-0.5, 0, 0, 1),
        (0.0, 1, 0, 1),
        (0.6, 0, 0, 1),
        (1.0, 0, 1, 0),
        (-0.5, 0, 0, 1),
    ]
    for k, (amp, has_rf, has_adc, _) in enumerate(blocks):
        x = x0 + k * w
        if k in (2, 6):
            ramp = np.clip(t / 0.15, 0.0, 1.0)
            ax.plot(x + 0.3 + (w - 0.6) * t, y + 6.0 * ramp, color=GREEN,
                    lw=2.0)
        elif k in (3, 7):
            # The rewinder picks the readout up where it left off, so the
            # seam before it carries a live gradient and cannot be cut.
            rew = np.interp(t, [0.0, 0.25, 0.75, 1.0], [1.0, -0.5, -0.5, 0.0])
            ax.plot(x + 0.3 + (w - 0.6) * t, y + 6.0 * rew, color=GREEN, lw=2.0)
        elif amp:
            ax.plot(x + 0.3 + (w - 0.6) * t, y + 6.0 * amp * trap,
                    color=GREEN, lw=2.0)
        ax.plot([x, x + w], [y, y], color=MUTED, lw=0.8)
        if has_rf:
            ax.plot(x + 0.3 + (w - 0.6) * t, y + 5.0 * np.exp(-30 * (t - 0.35) ** 2),
                    color=BLUE, lw=2.0)
            ax.text(x + w / 2, y + 13.0, "RF", ha="center", va="bottom",
                    fontsize=FS_SMALL, color=BLUE, fontweight="bold")
        if has_adc:
            ax.add_patch(Rectangle((x + 0.3, y - 4.0), w - 0.6, 2.4,
                                   facecolor=ORANGE, alpha=0.6,
                                   edgecolor="none"))
            ax.text(x + w / 2, y - 5.2, "ADC", ha="center", va="top",
                    fontsize=FS_SMALL, color=ORANGE, fontweight="bold")

    ax.text(x0, y + 18.0, "one TR of the block table", fontsize=FS_LABEL,
            color=INK, fontweight="bold", va="bottom")

    # Legal joins: every seam where both sides sit at zero.
    legal = [0, 1, 2, 4, 5, 6, 8]
    for k in legal:
        xs = x0 + k * w
        ax.plot([xs], [y - 9.5], marker="v", color=GREEN, ms=7)
    for k in (3, 7):
        xs = x0 + k * w
        ax.plot([xs], [y - 9.5], marker="x", color=RED, ms=8, mew=2.0)
    ax.text(x0, y - 13.0, "▾ a legal join — both sides at zero gradient      "
            "✕ live gradient, no cut possible", ha="left", va="top",
            fontsize=FS_BODY, color=INK)

    # The cuts actually taken: the last legal join before each RF.
    for k in (0, 4):
        xs = x0 + k * w
        ax.annotate("", xy=(xs, y + 16.0), xytext=(xs, y - 7.0),
                    arrowprops={"arrowstyle": "-", "color": PURPLE, "lw": 2.0,
                                "linestyle": (0, (5, 3))})
    ax.text(x0 + 8 * w + 1.0, y + 3.0, "cuts fall\nbefore the RF",
            ha="left", va="center", fontsize=FS_BODY, color=PURPLE,
            linespacing=1.4)

    ax.text(50.0, 1.0,
            "A boundary is where the sequencer can stop one prepared unit and "
            "start the next, so it has to\nsit at zero gradient. Among those, "
            "the cut is placed at the last one before each excitation.",
            ha="center", va="bottom", fontsize=FS_BODY, color=INK,
            style="italic", linespacing=1.5)
    return save(fig, "segments/segmentation_rules.png")


# ---------------------------------------------------------------------------
#  safety/index.md
# ---------------------------------------------------------------------------


def where_checks_run():
    """Which moment each check runs at, on each of the two write paths."""
    fig, ax = canvas(8.6, 7.0, ymax=76)

    lx, c1, c2, cw = 2.0, 26.0, 62.0, 34.0
    ax.text(lx, 68.0, "moment", fontsize=FS_BODY, color=MUTED, va="bottom")
    ax.text(c1 + cw / 2, 68.0, "written for\nthe scanner", ha="center",
            va="bottom", fontsize=FS_LABEL, color=BLUE, fontweight="bold",
            linespacing=1.4)
    ax.text(c2 + cw / 2, 68.0, "written for\nanywhere else", ha="center",
            va="bottom", fontsize=FS_LABEL, color=GREEN, fontweight="bold",
            linespacing=1.4)

    rows = [
        ("add_block", GREY,
         "nothing", MUTED,
         "nothing", MUTED),
        ("write()", BLUE,
         "nothing — the binary\nwriter takes no checks", MUTED,
         "amplitude · slew\ncontinuity · timing\nagainst the declared limits",
         GREEN),
        ("predownload", ORANGE,
         "amplitude · slew\ncontinuity · timing\nagainst the real system,\n"
         "then PNS and resonance", BLUE,
         "never reached", MUTED),
        ("hardware monitor", RED,
         "the only thing that can stop\na scan already running", RED,
         "never reached", MUTED),
    ]
    y = 62.0
    for name, name_colour, left, left_colour, right, right_colour in rows:
        lines = 1 + max(left.count("\n"), right.count("\n"))
        h = 4.5 + 3.6 * lines
        y -= h
        ax.add_patch(Rectangle((lx, y), 96.0 - lx, h - 1.4,
                               facecolor=name_colour, alpha=0.06,
                               edgecolor="none"))
        ax.text(lx + 1.4, y + (h - 1.4) / 2, name, ha="left", va="center",
                fontsize=FS_BODY, color=name_colour, fontweight="bold")
        ax.text(c1 + cw / 2, y + (h - 1.4) / 2, left, ha="center", va="center",
                fontsize=FS_SMALL, color=left_colour, linespacing=1.45)
        ax.text(c2 + cw / 2, y + (h - 1.4) / 2, right, ha="center",
                va="center", fontsize=FS_SMALL, color=right_colour,
                linespacing=1.45)

    ax.annotate("", xy=(lx - 0.2, y), xytext=(lx - 0.2, 62.0),
                arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 1.2})

    ax.text(50.0, 1.0,
            "One C core, called from two places. Design time and predownload "
            "cannot disagree,\nso a file bound for the scanner is written "
            "unchecked and judged where the limits are real.",
            ha="center", va="bottom", fontsize=FS_BODY, color=INK,
            style="italic", linespacing=1.5)
    return save(fig, "safety/where_checks_run.png")


# ---------------------------------------------------------------------------
#  safety/gradient_slew.md
# ---------------------------------------------------------------------------


def vector_limit():
    """Two axes inside their own limit, over it in combination."""
    fig, (ax_v, ax_c) = plt.subplots(
        1, 2, figsize=(8.6, 5.33), gridspec_kw={"width_ratios": [1.0, 1.25]}
    )

    # -- (a) the vector limit ---------------------------------------------
    gmax = 40.0
    th = np.linspace(0, np.pi / 2, 200)
    ax_v.plot(gmax * np.cos(th), gmax * np.sin(th), color=RED, lw=2.0,
              label="the system limit, 40 mT/m")
    ax_v.add_patch(Rectangle((0, 0), 35, 35, facecolor=GREEN, alpha=0.12,
                             edgecolor=GREEN, lw=1.4, ls="--"))
    ax_v.plot([0, 35], [0, 35], color=BLUE, lw=2.2)
    ax_v.plot(35, 35, "o", color=BLUE, ms=9)
    ax_v.annotate("35 on x and 35 on y\nis 49 mT/m", xy=(35, 35),
                  xytext=(17, 44), fontsize=FS_BODY, color=BLUE,
                  arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 1.2})
    ax_v.text(23, 5, "per-axis box\n(35 mT/m each)", fontsize=FS_SMALL,
              color=GREEN)
    ax_v.set_xlim(0, 56)
    ax_v.set_ylim(0, 56)
    ax_v.set_aspect("equal")
    ax_v.set_xlabel("$G_x$  [mT/m]")
    ax_v.set_ylabel("$G_y$  [mT/m]")
    ax_v.set_title("(a) the limit is on the vector", fontsize=FS_TITLE, loc="left")
    for sp in ("top", "right"):
        ax_v.spines[sp].set_visible(False)

    # -- (b) the seam between two blocks ----------------------------------
    t1 = np.array([0.0, 0.2, 1.0, 1.2])
    g1 = np.array([0.0, 20.0, 20.0, 12.0])
    t2 = np.array([1.2, 1.4, 2.2, 2.4])
    ax_c.plot(t1, g1, color=BLUE, lw=2.4)
    ax_c.plot(t2, np.array([0.0, 18.0, 18.0, 0.0]), color=ORANGE, lw=2.4)
    ax_c.plot([1.2, 1.2], [12.0, 0.0], color=RED, lw=2.4, ls=":")
    ax_c.plot(1.2, 12.0, "o", color=BLUE, ms=8)
    ax_c.plot(1.2, 0.0, "o", color=ORANGE, ms=8)
    ax_c.axvline(1.2, color=MUTED, lw=1.0, ls="--")
    ax_c.annotate("block n ends at 12 mT/m,\nblock n+1 starts at 0",
                  xy=(1.2, 6.0), xytext=(1.42, 9.0), fontsize=FS_BODY,
                  color=RED, arrowprops={"arrowstyle": "->", "color": RED,
                                         "lw": 1.2})
    ax_c.text(0.05, 22.0, "block n", fontsize=FS_BODY, color=BLUE)
    ax_c.text(1.9, 20.0, "block n+1", fontsize=FS_BODY, color=ORANGE)
    ax_c.set_xlim(-0.05, 2.9)
    ax_c.set_ylim(-2.0, 26.0)
    ax_c.set_xlabel("time  [ms]")
    ax_c.set_ylabel("$G_x$  [mT/m]")
    ax_c.set_title("(b) a block boundary is not a boundary for the gradient",
                   fontsize=FS_TITLE, loc="left")
    for sp in ("top", "right"):
        ax_c.spines[sp].set_visible(False)

    fig.tight_layout(pad=0.8)
    return save(fig, "gradient_slew/vector_and_seam.png")


# ---------------------------------------------------------------------------
#  performance/conversion.md
# ---------------------------------------------------------------------------


def conversion_stages():
    """The stages between a file arriving and the first block playing."""
    fig, ax = canvas(8.6, 6.4, ymax=64)

    stages = [
        ("parse", "text or binary into raw blocks and shapes",
         "per block", BLUE),
        ("resolve", "event references into definitions and instances",
         "per block, once", BLUE),
        ("detect", "the structural TR, from block identities",
         "per block, integer work", GREEN),
        ("segment", "the TR partitioned into reusable runs",
         "per TR block", GREEN),
        ("cache", "write the resolved tables beside the file",
         "paid once per file", ORANGE),
    ]
    h, gap, x0, w = 8.6, 2.0, 2.0, 96.0
    y = 62.0
    for k, (name, body, cost, colour) in enumerate(stages):
        y -= h
        box(ax, x0, y, w, h, colour, title=None, body=None)
        ax.text(x0 + 2.2, y + h / 2, name, ha="left", va="center",
                fontsize=FS_LABEL, color=colour, fontweight="bold")
        ax.text(x0 + 22.0, y + h / 2, body, ha="left", va="center",
                fontsize=FS_BODY, color=INK)
        ax.text(x0 + w - 2.2, y + h / 2, cost, ha="right", va="center",
                fontsize=FS_SMALL, color=colour)
        if k < len(stages) - 1:
            arrow(ax, (x0 + w / 2, y), (x0 + w / 2, y - gap), MUTED)
        y -= gap
    ax.text(50, 1.0, "Every stage after the first is linear in the block table "
            "and pays nothing per waveform;\nthe cache turns the whole column "
            "into a read on the next download.",
            ha="center", va="bottom", fontsize=FS_BODY, color=INK,
            style="italic", linespacing=1.5)
    return save(fig, "conversion/stages.png")


def raster_alignment():
    """A time the sequencer cannot address, and the one it can."""
    fig, ax = plt.subplots(figsize=(8.6, 3.3))

    raster = 4.0
    ticks = np.arange(0, 33, raster)
    ax.vlines(ticks, 0.0, 1.0, color=MUTED, lw=1.0)
    ax.hlines(0.0, -1.0, 33.0, color=MUTED, lw=1.2)
    for t in ticks:
        ax.text(t, -0.22, f"{int(t)}", ha="center", va="top",
                fontsize=FS_SMALL, color=MUTED, **MONO)
    ax.text(16.0, -0.62, "gradient raster — 4 µs on this system",
            ha="center", va="top", fontsize=FS_BODY, color=MUTED)

    ax.add_patch(Rectangle((12.0, 1.5), 16.0, 1.0, facecolor=GREEN,
                           alpha=0.35, edgecolor=GREEN, lw=1.6))
    ax.plot([12.0], [2.0], "o", color=GREEN, ms=8)
    ax.text(30.0, 2.0, "starts at 12 µs — a multiple of the raster",
            ha="left", va="center", fontsize=FS_BODY, color=GREEN)

    ax.add_patch(Rectangle((14.0, 3.0), 16.0, 1.0, facecolor=RED, alpha=0.30,
                           edgecolor=RED, lw=1.6))
    ax.plot([14.0], [3.5], "o", color=RED, ms=8)
    ax.text(30.0, 3.5, "starts at 14 µs — the hardware cannot begin there",
            ha="left", va="center", fontsize=FS_BODY, color=RED)

    ax.set_xlim(-2.0, 78.0)
    ax.set_ylim(-1.6, 4.6)
    ax.axis("off")
    fig.tight_layout(pad=0.4)
    return save(fig, "gradient_slew/raster_alignment.png")


# ---------------------------------------------------------------------------
#  performance/transform_fov.md
# ---------------------------------------------------------------------------


def base_trajectory_sharing():
    """One stored shape per distinct trajectory, against one per readout."""
    fig, ax = canvas(8.6, 6.4, ymax=64)

    n, w, gap, x0 = 6, 12.0, 3.0, 8.0
    angles = np.linspace(0.0, np.pi * (n - 1) / n, n)

    def arms(y, colour, rotate):
        t = np.linspace(0.0, 1.0, 160)
        for k in range(n):
            x = x0 + k * (w + gap)
            r = t
            th = 2.6 * np.pi * t + (angles[k] if rotate else 0.0)
            ax.plot(x + w / 2 + 0.42 * w * r * np.cos(th),
                    y + 4.2 + 3.4 * r * np.sin(th), color=colour, lw=1.1)

    ax.text(x0, 58.0, "six readouts of one stack of spirals — same arm, "
            "six angles", fontsize=FS_LABEL, color=INK, fontweight="bold",
            va="bottom")
    arms(48.0, BLUE, rotate=True)
    for k in range(n):
        x = x0 + k * (w + gap)
        ax.add_patch(Rectangle((x, 48.0), w, 8.6, facecolor="none",
                               edgecolor=BLUE, lw=1.0))

    box(ax, 2.0, 26.0, 45.0, 16.0, ORANGE, title="a phase per sample",
        body="one array of num_samples\nper readout — six arrays,\n"
             "none of them equal to\nanother",
        title_size=FS_BODY, body_size=FS_SMALL)
    box(ax, 53.0, 26.0, 45.0, 16.0, GREEN, title="the base trajectory",
        body="the arm before its angle\nand its amplitude — one\n"
             "array, and the six rows\ncollapse onto it",
        title_size=FS_BODY, body_size=FS_SMALL)
    arrow(ax, (30.0, 47.0), (24.5, 42.0), ORANGE, rad=0.10)
    arrow(ax, (70.0, 47.0), (75.5, 42.0), GREEN, rad=-0.10)

    t = np.linspace(0.0, 1.0, 200)
    th = 2.6 * np.pi * t
    ax.plot(66.0 + 0.42 * w * t * np.cos(th), 12.0 + 3.4 * t * np.sin(th),
            color=GREEN, lw=1.3)
    ax.add_patch(Rectangle((59.5, 7.5), w, 9.0, facecolor="none",
                           edgecolor=GREEN, lw=1.0))
    ax.text(66.0, 5.6, "one row", ha="center", va="top", fontsize=FS_BODY,
            color=GREEN)
    arrow(ax, (75.5, 26.0), (69.0, 17.0), GREEN, rad=-0.10)

    for k in range(n):
        x = 4.0 + k * 7.2
        ax.add_patch(Rectangle((x, 7.5), 5.6, 9.0, facecolor=ORANGE,
                               alpha=0.20, edgecolor=ORANGE, lw=1.0))
    ax.text(24.0, 5.6, "six rows", ha="center", va="top",
            fontsize=FS_BODY, color=ORANGE)
    arrow(ax, (24.5, 26.0), (24.5, 17.0), ORANGE)

    return save(fig, "transform_fov/base_trajectory.png")


# ---------------------------------------------------------------------------
#  performance/gradient_checks.md
# ---------------------------------------------------------------------------


def library_vs_scan():
    """Why one check scales with the library and the other with the scan."""
    fig, ax = canvas(8.6, 5.57, ymax=46)

    n, w, gap, x0, y = 14, 5.6, 0.9, 3.0, 32.0
    lib_of = [0, 1, 2, 0, 1, 3, 0, 1, 2, 0, 1, 3, 0, 1]
    lib_colour = [BLUE, GREEN, ORANGE, PURPLE]
    for k in range(n):
        x = x0 + k * (w + gap)
        ax.add_patch(Rectangle((x, y), w, 5.4, facecolor=lib_colour[lib_of[k]],
                               alpha=0.75, edgecolor="black", lw=0.5))
        ax.text(x + w / 2, y + 2.7, str(lib_of[k]), ha="center", va="center",
                fontsize=FS_SMALL, color="white", fontweight="bold")
    ax.text(x0, y + 7.6, "the block table — a million rows, each naming a "
            "waveform by id", fontsize=FS_LABEL, color=INK, fontweight="bold",
            va="bottom")

    strip_end = x0 + n * (w + gap) - gap
    ax.annotate("", xy=(strip_end, y - 2.2), xytext=(x0, y - 2.2),
                arrowprops={"arrowstyle": "|-|", "color": RED, "lw": 1.5})
    for k in range(1, n):
        xs = x0 + k * (w + gap) - gap / 2
        ax.plot([xs, xs], [y - 3.0, y - 1.4], color=RED, lw=1.0)
    ax.text(x0, y - 5.0, "continuity — one comparison per seam, so it walks "
            "the table", ha="left", va="top", fontsize=FS_BODY, color=RED)

    ly = 8.0
    for i, colour in enumerate(lib_colour):
        x = x0 + i * 15.0
        ax.add_patch(Rectangle((x, ly), 11.0, 5.4, facecolor=colour,
                               alpha=0.75, edgecolor="black", lw=0.5))
        ax.text(x + 5.5, ly + 2.7, str(i), ha="center", va="center",
                fontsize=FS_BODY, color="white", fontweight="bold")
    ax.text(x0, ly + 7.4, "the gradient library — four entries",
            fontsize=FS_LABEL, color=INK, fontweight="bold", va="bottom")
    ax.text(x0, ly - 2.0,
            "amplitude and slew — one evaluation per entry, whatever the "
            "scan does with it",
            ha="left", va="top", fontsize=FS_BODY, color=BLUE)
    return save(fig, "gradient_checks/library_vs_scan.png")


if __name__ == "__main__":
    pulseq_file_structure()
    structure_levels()
    transform_fov_walk()
    pulseg_mapping()
    segmentation_rules()
    where_checks_run()
    vector_limit()
    raster_alignment()
    conversion_stages()
    base_trajectory_sharing()
    library_vs_scan()
