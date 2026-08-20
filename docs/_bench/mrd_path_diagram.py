#!/usr/bin/env python3
"""Architecture figures for ``explanations/sequence_model/mrd_architecture.md``.

Documentation-only tooling; not part of the shipped package.

The figures are schematics, not measurements: one names the stages the raw data
passes through between the spectrometer and the images, and where each stage
gets what it adds; the other is the real-time round trip.  Both are drawn
rather than plotted, so nothing here reads the package.

Usage:
    <venv>/bin/python docs/_bench/mrd_path_diagram.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_DIR = Path(__file__).resolve().parents[1] / "explanations" / "assets" / "mrd_path"

SCANNER = "#4a4a4a"
CLIENT = "#4c72b0"
SERVER = "#dd8452"
QUEUE = "#8172b3"
INK = "#22252a"


def box(ax, x, y, w, h, color, *, title=None, body=None, dashed=False, fill=None):
    """Draw one rounded box and return its centre."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=0.9",
            linewidth=1.2,
            linestyle="--" if dashed else "-",
            edgecolor=color,
            facecolor=fill if fill is not None else "white",
            zorder=2,
        )
    )
    cx, cy = x + w / 2, y + h / 2
    if title is not None and body is not None:
        ax.text(
            cx, y + h - 1.5, title, ha="center", va="top",
            fontsize=8.6, color=color, fontweight="bold", zorder=3,
        )
        ax.text(
            cx, y + h - 4.2, body, ha="center", va="top",
            fontsize=7.4, color=INK, zorder=3, linespacing=1.45,
        )
    else:
        ax.text(
            cx, cy, title if body is None else body, ha="center", va="center",
            fontsize=8.0, color=INK, zorder=3, linespacing=1.45,
        )
    return cx, cy


def group(ax, x, y, w, h, color, label):
    """Draw a labelled container behind the boxes it holds."""
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
        x + w / 2, y + h - 1.4, label, ha="center", va="top",
        fontsize=9.0, color=color, fontweight="bold", zorder=3,
    )


def arrow(ax, start, end, color, *, dashed=False, rad=0.0, label=None,
          label_pos=None, label_va="bottom", fontsize=7.4):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.2,
            linestyle="--" if dashed else "-",
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
            lx, ly, label, ha="center", va=label_va, fontsize=fontsize,
            color=color, zorder=5, linespacing=1.4,
        )


def build() -> Path:
    fig, ax = plt.subplots(figsize=(11.0, 6.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 58)
    ax.axis("off")

    # -- scanner ---------------------------------------------------------
    group(ax, 1, 24, 18, 32, SCANNER, "scanner")
    box(ax, 2.5, 43.5, 15, 9, SCANNER,
        title="acquisition system",
        body="native raw-data\npackets, as measured")
    box(ax, 2.5, 26.5, 15, 9, SCANNER,
        title="the sequence",
        body="the .seq chain\nthe interpreter played")

    # -- client ----------------------------------------------------------
    group(ax, 22, 12, 20, 44, CLIENT, "client  \u00b7  C++")
    box(ax, 23.5, 43.5, 17, 8.5, CLIENT,
        title="convert", body="native packet \u2192\nMRD acquisition")
    box(ax, 23.5, 27.5, 17, 13, CLIENT,
        title="enrich",
        body="encoding spaces, counters,\nflags and echo position,\n"
             "k-space trajectory,\nsequence description")
    box(ax, 23.5, 14.5, 17, 10, CLIENT,
        title="demodulate",
        body="undo the design-time\nFOV-shift phase")

    arrow(ax, (17.5, 48), (23.5, 47.5), SCANNER)
    arrow(ax, (17.5, 31), (23.5, 34), SCANNER)
    arrow(ax, (32, 43.5), (32, 40.5), CLIENT)
    arrow(ax, (32, 27.5), (32, 24.5), CLIENT)

    # -- the wire --------------------------------------------------------
    arrow(ax, (42, 36), (61.5, 36), CLIENT,
          label="MRD session protocol \u00b7 TCP\nheader \u00b7 acquisitions \u00b7 waveforms",
          label_pos=(51.7, 37.3), fontsize=7.2)
    arrow(ax, (61.5, 29), (42, 29), SERVER,
          label="images \u00b7 DICOM", label_pos=(51.7, 27.6), label_va="top",
          fontsize=7.2)

    # -- server ----------------------------------------------------------
    group(ax, 60, 22, 39, 34, SERVER, "server  \u00b7  Python")
    box(ax, 61.5, 42.5, 36, 10.5, SERVER,
        title="exam cache",
        body="sensitivity maps, bases, plans \u2014 keyed, built once, shared by every\n"
             "scan of one exam; retired the moment the exam identity changes")
    box(ax, 61.5, 25.5, 15.5, 13, SERVER,
        title="recon slot 1",
        body="one plugin instance,\none stream,\nits own thread")
    box(ax, 82, 25.5, 15.5, 13, SERVER,
        title="recon slot N",
        body="N derived from\navailable RAM and\nRAM per recon")
    ax.text(79.5, 32, "\u00b7 \u00b7 \u00b7", ha="center", va="center",
            fontsize=11, color=SERVER, zorder=3)
    arrow(ax, (69.2, 42.5), (69.2, 38.5), SERVER)
    arrow(ax, (89.7, 42.5), (89.7, 38.5), SERVER)

    # -- overflow --------------------------------------------------------
    group(ax, 60, 1.5, 39, 18, QUEUE, "when every slot is busy")
    box(ax, 61.5, 4, 15.5, 11, QUEUE,
        title="drain to disk",
        body="the stream is consumed\nin full to an MRD file\nand a sidecar")
    box(ax, 82, 4, 15.5, 11, QUEUE,
        title="replay worker",
        body="takes the oldest sidecar,\nwaits for a slot, runs the\nrequested recon on the file")
    arrow(ax, (69.2, 22), (69.2, 15), QUEUE, dashed=True)
    arrow(ax, (77, 9.5), (82, 9.5), QUEUE)
    arrow(ax, (89.7, 15), (89.7, 25.5), QUEUE, dashed=True)

    fig.tight_layout(pad=0.4)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "mrd_path.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def build_feedback() -> Path:
    fig, ax = plt.subplots(figsize=(9.0, 2.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 26)
    ax.axis("off")

    box(ax, 4, 9, 32, 12, SCANNER,
        title="interpreter",
        body="the blocks still ahead\nof the playout cursor")
    box(ax, 64, 9, 32, 12, SERVER,
        title="real-time process",
        body="its own port,\nthe same MRD framing")

    arrow(ax, (36, 18.5), (64, 18.5), SCANNER,
          label="one acquisition, as it is measured",
          label_pos=(50, 19.6), fontsize=7.4)
    arrow(ax, (64, 11.5), (36, 11.5), SERVER,
          label="one tagged result,\nbefore the blocks it bears on",
          label_pos=(50, 10.5), label_va="top", fontsize=7.4)

    ax.text(50, 2.6,
            "which tags are recognised, and what each one changes, "
            "is decided on the interpreter side",
            ha="center", va="center", fontsize=7.6, color=INK, style="italic")

    fig.tight_layout(pad=0.4)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "mrd_feedback.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


if __name__ == "__main__":
    for path in (build(), build_feedback()):
        print(path.relative_to(OUT_DIR.parents[1]))
