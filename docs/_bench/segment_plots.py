#!/usr/bin/env python3
"""TR-and-segment figures for ``explanations/sequence_model/tr_and_segmentation.md``.

Documentation-only tooling; not part of the shipped package.

Each figure is one canonical TR drawn by ``Sequence.plot(tr=...)`` -- the call
the page tells a reader to make -- with the segments the interpreter prepares
shaded behind the waveforms.  The shading is the only ink added here: the
spans come from the per-block segment identity the C core reports, so the
picture and the interpreter's partition are the same answer.

Usage:
    <venv>/bin/python docs/_bench/segment_plots.py
    <venv>/bin/python docs/_bench/segment_plots.py --only=gre
"""

from __future__ import annotations

import argparse
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pulserver.app as app
from pulserver._ext import pulseg

OUT_DIR = Path(__file__).resolve().parents[1] / "explanations" / "assets" / "segments"

#: One hue per segment, held across the figures so a colour means a segment.
SEGMENT_COLORS = ("#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3")

#: Above this many instance boundaries in view, the per-instance rules become a
#: picket fence and the run's outline carries the structure on its own.
INSTANCE_RULE_LIMIT = 8


def tr_runs(seq):
    """One acquiring TR, as ``(instance, runs)``.

    A run is ``(segment_id, edges)``: one consecutive stretch of the same
    segment replayed, ``edges`` holding its instance boundaries in seconds
    from the start of the TR.

    The instance drawn is the first that acquires the most, so the figure
    carries the ADC and the encoding a dummy shot would leave blank. The
    partition itself is the same on every instance.
    """
    structure = seq._structure_for("plot")
    table = pulseg._get_scan_table(structure.collection)

    tr_size = seq.tr_size
    acquiring = np.asarray(table["adc_flag"])[
        : (len(table["adc_flag"]) // tr_size) * tr_size
    ]
    instance = int(np.argmax(acquiring.reshape(-1, tr_size).sum(axis=1)))
    window = slice(instance * tr_size, (instance + 1) * tr_size)

    seg_id = np.asarray(table["segment_id"])[window]
    starts = np.asarray(table["segment_start"])[window]
    edges = np.concatenate(
        [[0.0], np.cumsum(np.asarray(table["duration_us"])[window] * 1e-6)]
    )

    bounds = [i for i in range(tr_size) if starts[i]] + [tr_size]
    runs: list[tuple[int, list[float]]] = []
    for lo, hi in pairwise(bounds):
        this = int(seg_id[lo])
        if runs and runs[-1][0] == this:
            runs[-1][1].append(float(edges[hi]))
        else:
            runs.append((this, [float(edges[lo]), float(edges[hi])]))
    return instance, runs


def shade(plot, runs, t_factor, view):
    """Shade each run of a segment behind every panel and name it above them."""
    axes = plot.ax1
    lo_view, hi_view = view
    for seg_id, edges in runs:
        start, stop = edges[0], edges[-1]
        if stop <= lo_view or start >= hi_view:
            continue
        repeats = len(edges) - 1
        interior = [e for e in edges[1:-1] if lo_view < e < hi_view]
        color = SEGMENT_COLORS[seg_id % len(SEGMENT_COLORS)]
        for ax in axes:
            ax.axvspan(
                start * t_factor,
                stop * t_factor,
                color=color,
                alpha=0.13,
                lw=0,
                zorder=0,
            )
            for edge in (start, stop):
                ax.axvline(edge * t_factor, color=color, alpha=0.6, lw=1.0, zorder=1)
            if len(interior) < INSTANCE_RULE_LIMIT:
                for edge in interior:
                    ax.axvline(
                        edge * t_factor,
                        color=color,
                        alpha=0.75,
                        lw=1.0,
                        ls=":",
                        zorder=1,
                    )
        name = f"segment {seg_id}" + (f"  x{repeats}" if repeats > 1 else "")
        centre = 0.5 * (max(start, lo_view) + min(stop, hi_view))
        axes[0].annotate(
            name,
            xy=(centre * t_factor, 1.06),
            xycoords=("data", "axes fraction"),
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=color,
            clip_on=False,
        )


def save(fig, stem: str) -> Path:
    out = OUT_DIR / f"{stem}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=130)
    plt.close("all")
    return out


def draw(seq, runs, instance, stem, title, *, t_range=None, time_disp="ms"):
    kwargs = {} if t_range is None else {"time_range": t_range}
    plot = seq.plot(
        tr=instance, stacked=True, plot_now=False, time_disp=time_disp, **kwargs
    )
    view = t_range if t_range is not None else (runs[0][1][0], runs[-1][1][-1])
    shade(plot, runs, getattr(plot.fig1, "_seq_t_factor", 1.0), view)
    fig = plot.fig1
    fig.set_size_inches(9.5, 8.5)
    fig.suptitle(title, fontsize=10)
    return save(fig, stem)


def headline(seq, runs) -> str:
    instances = sum(len(edges) - 1 for _, edges in runs)
    return (
        f"{seq.tr_size} blocks, {instances} segment instances "
        f"of {seq.num_segments} segments"
    )


def gre() -> list[Path]:
    seq = app.gre2D_sequence.main(
        n_x=256, n_y=64, n_slices=1, te=None, tr=12e-3, spoiling_cycles=6.0
    )
    instance, runs = tr_runs(seq)
    return [
        draw(
            seq,
            runs,
            instance,
            "gre_2d_tr_segments",
            f"spoiled GRE, one TR -- {headline(seq, runs)}",
        )
    ]


def mprage() -> list[Path]:
    seq = app.mprage3D_sequence.main(
        n_x=256,
        n_y=64,
        n_z=32,
        views_per_segment=8,
        n_acs=16,
        n_acs_z=8,
        ti=75e-3,
        tr_outer=135e-3,
        spoiling_cycles=6.0,
    )
    instance, runs = tr_runs(seq)
    written = [
        draw(
            seq,
            runs,
            instance,
            "mprage_3d_tr_segments",
            f"MPRAGE, one TR -- {headline(seq, runs)}",
        )
    ]

    # One instance of the train, at the scale that shows its four blocks.
    train = max(runs, key=lambda run: len(run[1]))
    written.append(
        draw(
            seq,
            runs,
            instance,
            "mprage_3d_train_segments",
            f"MPRAGE, one of the {len(train[1]) - 1} instances the train replays",
            t_range=(train[1][0], train[1][1]),
        )
    )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("all", "gre", "mprage"), default="all")
    args = parser.parse_args()

    for name, build in (("gre", gre), ("mprage", mprage)):
        if args.only in ("all", name):
            paths = "  ".join(str(p.relative_to(OUT_DIR.parents[1])) for p in build())
            print(f"{name:8} -> {paths}")


if __name__ == "__main__":
    main()
