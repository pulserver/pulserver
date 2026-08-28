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
            fontsize=11.0, color=color, fontweight="bold", zorder=3,
        )
        ax.text(
            cx, y + h - 4.2, body, ha="center", va="top",
            fontsize=9.3, color=INK, zorder=3, linespacing=1.45,
        )
    else:
        ax.text(
            cx, cy, title if body is None else body, ha="center", va="center",
            fontsize=10.2, color=INK, zorder=3, linespacing=1.45,
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
        fontsize=11.4, color=color, fontweight="bold", zorder=3,
    )


def arrow(ax, start, end, color, *, dashed=False, rad=0.0, label=None,
          label_pos=None, label_va="bottom", fontsize=9.3):
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
    fig, ax = plt.subplots(figsize=(8.6, 7.89))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 58)
    ax.axis("off")

    # -- scanner ---------------------------------------------------------
    group(ax, 1, 24, 20.5, 32, SCANNER, "scanner")
    box(ax, 2.5, 43.5, 17.5, 9, SCANNER,
        title="acquisitions",
        body="native packets,\nas measured")
    box(ax, 2.5, 26.5, 17.5, 9, SCANNER,
        title="the sequence",
        body="the .seq chain\nit played")

    # -- client ----------------------------------------------------------
    group(ax, 22, 12, 20, 44, CLIENT, "client  \u00b7  C++")
    box(ax, 23.5, 43.5, 17, 8.5, CLIENT,
        title="convert", body="native packet\n\u2192 MRD")
    box(ax, 23.5, 27.5, 17, 13, CLIENT,
        title="enrich",
        body="encoding spaces,\ncounters and flags,\necho position,\n"
             "k-space trajectory,\nsequence description")
    box(ax, 23.5, 14.5, 17, 10, CLIENT,
        title="demodulate",
        body="undo the design-\ntime FOV phase")

    arrow(ax, (20.0, 48), (23.5, 47.5), SCANNER)
    arrow(ax, (20.0, 31), (23.5, 34), SCANNER)
    arrow(ax, (32, 43.5), (32, 40.5), CLIENT)
    arrow(ax, (32, 27.5), (32, 24.5), CLIENT)

    # -- the wire --------------------------------------------------------
    arrow(ax, (42, 36), (61.5, 36), CLIENT,
          label="MRD session protocol \u00b7 TCP\nheader, acquisitions, waveforms",
          label_pos=(51.7, 37.6), fontsize=9.2)
    arrow(ax, (61.5, 29), (42, 29), SERVER,
          label="images \u00b7 DICOM", label_pos=(51.7, 27.8), label_va="top",
          fontsize=9.2)

    # -- server ----------------------------------------------------------
    group(ax, 60, 22, 39, 34, SERVER, "server  \u00b7  Python")
    box(ax, 61.5, 41.5, 36, 11.5, SERVER,
        title="exam cache",
        body="sensitivity maps, bases, plans \u2014 keyed and built\n"
             "once, shared by every scan of one exam, retired\n"
             "the moment the exam identity changes")
    box(ax, 60.8, 25.5, 17.2, 13, SERVER,
        title="recon slot 1",
        body="one plugin instance,\none stream, its own\nthread")
    box(ax, 81.0, 25.5, 17.0, 13, SERVER,
        title="recon slot N",
        body="N from available\nRAM and RAM per\nrecon")
    ax.text(79.5, 32.0, "\u00b7\n\u00b7\n\u00b7", ha="center", va="center",
            fontsize=13.0, color=SERVER, zorder=3, linespacing=0.9)
    arrow(ax, (69.4, 41.5), (69.4, 38.5), SERVER)
    arrow(ax, (89.5, 41.5), (89.5, 38.5), SERVER)

    # -- overflow --------------------------------------------------------
    group(ax, 60, 1.5, 39, 18, QUEUE, "when every slot is busy")
    box(ax, 61.5, 4, 18.0, 11, QUEUE,
        title="drain to disk",
        body="the stream is read to\nthe end into an MRD\nfile and a sidecar")
    box(ax, 80.5, 4, 17.0, 11, QUEUE,
        title="replay worker",
        body="takes the oldest\nsidecar, waits for a\nslot, runs its recon")
    arrow(ax, (64.5, 22), (64.5, 15), QUEUE, dashed=True)
    arrow(ax, (79.5, 9.5), (80.5, 9.5), QUEUE)

    arrow(ax, (94.5, 15), (94.5, 25.5), QUEUE, dashed=True)

    fig.tight_layout(pad=0.4)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "mrd_path.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def build_feedback() -> Path:
    fig, ax = plt.subplots(figsize=(8.6, 2.79))
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
          label_pos=(50, 19.6), fontsize=10.5)
    arrow(ax, (64, 11.5), (36, 11.5), SERVER,
          label="one tagged result,\nbefore the blocks it bears on",
          label_pos=(50, 10.5), label_va="top", fontsize=10.5)

    ax.text(50, 2.6,
            "which tags are recognised, and what each one changes, "
            "is decided on the interpreter side",
            ha="center", va="center", fontsize=10.8, color=INK, style="italic")

    fig.tight_layout(pad=0.4)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "mrd_feedback.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


if __name__ == "__main__":
    for path in (build(), build_feedback()):
        print(path.relative_to(OUT_DIR.parents[1]))
