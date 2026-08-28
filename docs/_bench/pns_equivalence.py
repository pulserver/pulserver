#!/usr/bin/env python3
"""The equivalence figures for ``explanations/performance/pns``.

Documentation-only tooling, and the companion of ``mechres_equivalence.py``:
the two checks share a structure, so the two scripts do too. Every claim the
page makes is computed here rather than stored.

``canonical_tr``
    One window judged for a whole scan, on the same four controlled
    stack-of-spirals scans ``mechres_equivalence`` uses -- nothing varying
    across repetitions, the encode amplitude varying, the readout waveform
    varying, both varying. The reference is the whole scan rendered and
    convolved in one pass, so the window's peak is checked against what every
    repetition really reaches.
``assembly_cost``
    What the evaluation's cost actually depends on: the scan's length, the
    number of distinct shapes in the window, and the window's own duration.
``epi_verdict``
    The verdict itself -- an echo train's stimulation against the 80 % margin
    and the 100 % threshold.

The nerve model throughout is **Irnich rheobase/chronaxie**, written here
straight from its published definition in double precision, so both sides of
every comparison are independent readings rather than two calls into the same
library. ``pns_equivalence_plots.py`` supplies those primitives.

Usage:
    <venv>/bin/python docs/_bench/pns_equivalence.py [name ...]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

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


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pulserver.pypulseq as pp  # noqa: E402
from _bench.pns_equivalence_plots import (  # noqa: E402
    IRNICH,
    assembled_per_shape,
    canonical_window,
    convolved_directly,
    irnich_kernel,
    norm,
)
from _figures import FAINT, INK, MUTED, SERIES, _style  # noqa: E402

ASSETS = (
    Path(__file__).resolve().parents[1] / "explanations" / "assets" / "pns_performance"
)

GAMMA_HZ_PER_T = 42.576e6

#: The two lines every stimulation figure is read against. Unlike a
#: mechanical-resonance band, these are not calibrated against anything: the
#: model reports a percentage of its own threshold, and 100 % is the limit.
THRESHOLD = 100.0
MARGIN = 80.0

#: The same four controlled scans ``mechres_equivalence`` builds, on the
#: weaker of the two gradient systems that script's audit uses and at a
#: readout bandwidth a console would prescribe -- which puts this protocol
#: just under threshold, where the margin is worth looking at.
SYSTEM = pp.Opts(max_grad=33, grad_unit="mT/m", max_slew=120, slew_unit="T/m/s")
STACK = dict(
    n_x=64,
    n_z=8,
    n_arms=16,
    angle_scheme="golden",
    n_dummy=0,
    tr=10e-3,
    readout_bandwidth_hz=83e3,
    use_rotation_ext=False,
)
REPETITIONS = 4

CASES = (
    ("A", "nothing varies", False, False),
    ("B", "the encode amplitude varies", False, True),
    ("C", "the readout waveform varies", True, False),
    ("D", "both vary", True, True),
)


def stack_kernel(**overrides):
    from pulserver.app import gre_stack_of_spirals3D_sequence as plugin

    return plugin, plugin.StackOfSpiralsKernel(SYSTEM, **{**STACK, **overrides})


def controlled_scan(kernel_pair, arm_varies, encode_varies, reps=REPETITIONS):
    """``reps`` repetitions of the shipped kernel, varying only what is asked."""
    plugin, kernel = kernel_pair
    sequence = pp.Sequence(SYSTEM)
    phases = pp.make_rf_spoiling_schedule(reps, increment=np.deg2rad(117.0))
    for m in range(reps):
        arm = m if arm_varies else 0
        # Spread the encode over its whole table however many repetitions are
        # drawn, so a short figure still shows the amplitude range a real scan
        # sweeps rather than the first few steps of it.
        partition = round(m * 7 / max(reps - 1, 1)) if encode_varies else 2
        kernel.readout.rf.phase_offset = phases[m]
        kernel.readout.adc.phase_offset = phases[m]
        plugin.StackShotKernel(
            sequence,
            kernel.readout,
            kernel.readout.arm(kernel.shot_index(arm, partition)),
            None,
            (partition - 4) / 4,
            acquire=True,
        )
    return sequence


# ----------------------------------------------------------------------
# the two paths every figure compares
# ----------------------------------------------------------------------


def scan_waveforms(sequence, tr_samples, reps):
    """Every repetition as it really plays, concatenated, in Hz/m per axis.

    Each repetition is pulled through the *same* extraction the window comes
    from, rather than resampled off the gradient objects: the two grids differ
    by half a raster tick, and on a slew-sensitive spike that alone is worth
    0.8 percentage points of threshold -- which would show up as a bound
    margin that is really a sampling artefact.
    """
    per_instance = [canonical_window(sequence, index)[0] for index in range(reps)]
    return [
        np.concatenate([instance[axis][:tr_samples] for instance in per_instance])
        for axis in range(3)
    ]


def instance_spread(sequence, tr_samples, reps):
    """Largest gradient difference between any two repetitions, in mT/m.

    What makes the four cases genuinely different scans, as opposed to what
    the verdict makes of them.
    """
    per_instance = [canonical_window(sequence, index)[0] for index in range(reps)]
    return (
        max(
            float(
                np.abs(
                    per_instance[i][axis][:tr_samples]
                    - per_instance[0][axis][:tr_samples]
                ).max()
            )
            for i in range(reps)
            for axis in range(3)
        )
        / 42.576e3
    )


def scan_response(sequence, dt, kernel, tr_samples, reps):
    """``(whole-scan response, per-repetition slices)`` as fractions of threshold.

    The naive analysis in full: every repetition as it actually plays,
    concatenated, differentiated and convolved in one pass. Nothing here knows
    the scan has a period.
    """
    total = tr_samples * reps
    response = norm(
        convolved_directly(
            scan_waveforms(sequence, tr_samples, reps), dt, GAMMA_HZ_PER_T, kernel
        )
    )[:total]
    return response, response.reshape(reps, tr_samples)


def window_response(sequence, dt, kernel, tr_samples, tr="worst_case"):
    """The canonical window's response, as a fraction of threshold."""
    axes, _, _ = canonical_window(sequence, tr)
    return norm(convolved_directly(axes, dt, GAMMA_HZ_PER_T, kernel))[:tr_samples]


def window_geometry(sequence):
    """``(dt, samples in one repetition, kernel)`` for a sequence."""
    axes, lengths, dt = canonical_window(sequence, "worst_case")
    structure = sequence._structure_for("bound")
    return dt, round(structure.tr_duration / dt), irnich_kernel(dt, IRNICH)


# ----------------------------------------------------------------------
# drawing helpers
# ----------------------------------------------------------------------


def thresholds(axis, label=False):
    axis.axhline(THRESHOLD, color="#e34948", lw=1.1, ls=(0, (4, 2)), zorder=6)
    axis.axhline(MARGIN, color=MUTED, lw=0.8, ls=(0, (2, 3)), zorder=6)
    if label:
        axis.text(
            0.012,
            THRESHOLD,
            "100 %",
            transform=axis.get_yaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=9.7,
            color="#e34948",
        )
        axis.text(
            0.012,
            MARGIN,
            "80 %",
            transform=axis.get_yaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=9.7,
            color=MUTED,
        )


def legend(target, handles=None, labels=None, **kwargs):
    """A legend above the axes it belongs to, never on top of the data."""
    kwargs.setdefault("fontsize", 9.4)
    kwargs.setdefault("frameon", False)
    kwargs.setdefault("loc", "lower left")
    kwargs.setdefault("bbox_to_anchor", (0.0, 1.14, 1.0, 0.16))
    kwargs.setdefault("mode", "expand")
    kwargs.setdefault("borderaxespad", 0.0)
    kwargs.setdefault("handlelength", 1.2)
    made = target.legend(handles, labels, **kwargs) if handles else target.legend(**kwargs)
    for text in made.get_texts():
        text.set_color(MUTED)
    return made


def save(figure, stem):
    ASSETS.mkdir(parents=True, exist_ok=True)
    out = ASSETS / f"{stem}.png"
    figure.patch.set_facecolor("white")
    figure.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"  {out.relative_to(ASSETS.parents[2])}")
    return out


# ----------------------------------------------------------------------
# 1 -- the canonical window against the scan it stands for
# ----------------------------------------------------------------------


def canonical_tr():
    kernel_pair = stack_kernel()

    figure, axes = plt.subplots(
        len(CASES), 2, figsize=(8.6, 9.4), width_ratios=(1.0, 0.66)
    )
    figure.subplots_adjust(
        hspace=0.70, wspace=0.30, top=0.875, bottom=0.055, left=0.105, right=0.985
    )

    for column, (letter, what, arm_varies, encode_varies) in enumerate(CASES):
        sequence = controlled_scan(kernel_pair, arm_varies, encode_varies)
        dt, tr_samples, kernel = window_geometry(sequence)
        reps = sequence._structure_for("bound").num_trs

        window = window_response(sequence, dt, kernel, tr_samples)
        scan, per_rep = scan_response(sequence, dt, kernel, tr_samples, reps)
        millis = np.arange(scan.size) * dt * 1e3
        period_ms = tr_samples * dt * 1e3

        top = axes[column, 0]
        for edge in range(1, reps):
            top.axvline(edge * period_ms, color=FAINT, lw=0.7, zorder=0)
        top.plot(millis, scan * 100.0, color=SERIES[0], lw=1.4, zorder=3)
        top.plot(
            millis,
            np.tile(window, reps) * 100.0,
            color=INK,
            lw=1.0,
            ls=(0, (3, 2.4)),
            zorder=4,
        )
        thresholds(top, label=column == 0)
        spread = instance_spread(sequence, tr_samples, reps)
        _style(top, f"{letter}   {what}")
        top.text(
            0.98,
            0.95,
            "repetitions identical"
            if spread < 1e-6
            else f"repetitions differ by up to {spread:.0f} mT/m",
            transform=top.transAxes,
            fontsize=9.4,
            color=MUTED,
            va="top",
            ha="right",
        )
        top.set_xlim(0, millis[-1])
        top.set_ylim(0, 124)
        top.set_ylabel("% of threshold", fontsize=10.4)
        if column == len(CASES) - 1:
            top.set_xlabel("time across the scan (ms)", fontsize=10.4)

        peaks = per_rep.max(axis=1) * 100.0
        judged = window.max() * 100.0
        bottom = axes[column, 1]
        for edge in range(1, reps):
            bottom.axvline(edge * period_ms, color=FAINT, lw=0.7, zorder=0)
        bottom.plot(
            (np.arange(reps) + 0.5) * period_ms,
            peaks,
            "o",
            color=SERIES[0],
            ms=4.5,
            mew=0,
            zorder=3,
        )
        bottom.axhline(judged, color=INK, lw=1.2, ls=(0, (4, 2)), zorder=4)
        _style(bottom, f"judged / worst repetition = {judged / peaks.max():.5f}×")
        bottom.set_xlim(0, millis[-1])
        span = max(judged - peaks.min(), 0.06)
        bottom.set_ylim(peaks.min() - 0.45 * span, judged + 0.5 * span)
        bottom.set_ylabel("peak per\nrepetition (%)", fontsize=10.4)
        if column == len(CASES) - 1:
            bottom.set_xlabel("time across the scan (ms)", fontsize=10.4)
        print(
            f"    {letter} {what:28} judged {judged:7.3f} %  worst repetition "
            f"{peaks.max():7.3f} %  ratio {judged / peaks.max():.4f}  "
            f"spread across repetitions {peaks.max() - peaks.min():.3f} pp"
        )

    legend(
        figure,
        [
            Line2D([], [], color=SERIES[0], lw=1.4),
            Line2D([], [], color=INK, lw=1.0, ls=(0, (3, 2.4))),
            Line2D([], [], color="#e34948", lw=1.1, ls=(0, (4, 2))),
        ],
        [
            "the scan as it really plays, straight through",
            "the canonical window, tiled — what the gate judges",
            "threshold",
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncols=2,
        fontsize=9.8,
    )
    figure.suptitle(
        "Each kind of variation switched on alone. In all four, the window's\n"
        "peak is exactly the worst repetition's.",
        x=0.02,
        y=0.995,
        ha="left",
        va="top",
        fontsize=12.0,
        color=INK,
    )
    return save(figure, "canonical_tr")


# ----------------------------------------------------------------------
# 2 -- what the evaluation's cost depends on
# ----------------------------------------------------------------------


def assembly_cost():
    from importlib import import_module

    from pulserver.app import epi2D_sequence

    def fastest(call, repeats=3):
        best = float("inf")
        for _ in range(repeats):
            started = time.perf_counter()
            call()
            best = min(best, time.perf_counter() - started)
        return best * 1e3

    # --- scan length: the timeline against the window -------------------
    scans, timeline_ms, window_ms = [], [], []
    for reps in (3, 12, 48, 144):
        sequence = epi2D_sequence.main(
            plot=False,
            write_seq=False,
            system=SYSTEM,
            n_x=64,
            n_y=64,
            segments=1,
            n_dummy=0,
            n_repetitions=reps,
            readout_bandwidth_hz=250e3,
        )
        sequence = sequence[0] if isinstance(sequence, tuple) else sequence
        dt, tr_samples, kernel = window_geometry(sequence)
        # Every repetition of this EPI time series is identical, so tiling the
        # window is the scan -- and the cost being measured depends on the
        # array's length, not on which of the two paths produced it.
        window_axes, _, _ = canonical_window(sequence, "worst_case")
        full = [np.tile(axis[:tr_samples], reps) for axis in window_axes]
        axes = window_axes
        timeline_ms.append(
            fastest(lambda: convolved_directly(full, dt, GAMMA_HZ_PER_T, kernel))
        )
        window_ms.append(
            fastest(lambda: convolved_directly(axes, dt, GAMMA_HZ_PER_T, kernel))
        )
        scans.append(reps)
        print(
            f"    {reps:4d} repetitions: timeline {timeline_ms[-1]:8.1f} ms   "
            f"window {window_ms[-1]:6.2f} ms"
        )

    # --- distinct shapes: convolved whole against assembled -------------
    families = []
    for label, module, kwargs in (
        ("spiral GRE", "gre_spiral2D_sequence", dict(n_x=64, n_arms=8, tr=None)),
        ("2D GRE", "gre2D_sequence", dict(n_x=128, n_y=64, tr=15e-3)),
        (
            "EPI",
            "epi2D_sequence",
            dict(n_x=64, n_y=64, segments=1, readout_bandwidth_hz=250e3),
        ),
        ("FSE", "fse2D_sequence", dict(n_x=128, n_y=64)),
        ("MPRAGE", "mprage3D_sequence", dict(n_x=96, n_y=32, n_z=16)),
    ):
        built = getattr(import_module("pulserver.app"), module).main(
            plot=False, write_seq=False, system=SYSTEM, n_dummy=0, **kwargs
        )
        sequence = built[0] if isinstance(built, tuple) else built
        axes, lengths, dt = canonical_window(sequence, "worst_case")
        kernel = irnich_kernel(dt, IRNICH)
        _, shapes, occurrences = assembled_per_shape(
            axes, lengths, dt, GAMMA_HZ_PER_T, kernel
        )
        n, k = axes[0].size, kernel.size
        # Convolving each stored shape once, then placing every occurrence
        # twice -- once for the window, once for the wrapped history.
        convolve = sum((stored.size - k + 1) * k for _, stored in shapes)
        place = 2 * sum(shapes[index][1].size for index, _, _ in occurrences)
        families.append(
            {
                "label": label,
                "ms": n * dt * 1e3,
                "shapes": len(shapes),
                "occurrences": len(occurrences),
                "direct": 3.0 * n * k,
                "assembled": float(convolve + place),
            }
        )
        f = families[-1]
        print(
            f"    {label:12} window {f['ms']:8.1f} ms  shapes {f['shapes']:3d}  "
            f"occurrences {f['occurrences']:4d}  whole {f['direct'] / 1e6:8.2f} M  "
            f"assembled {f['assembled'] / 1e6:6.2f} M  speedup "
            f"{f['direct'] / f['assembled']:6.1f}x"
        )

    figure, panels = plt.subplots(1, 3, figsize=(8.6, 4.1), dpi=170)
    figure.subplots_adjust(wspace=0.36, top=0.78, bottom=0.26, left=0.08, right=0.985)

    axis = panels[0]
    axis.plot(
        scans,
        timeline_ms,
        "o-",
        color=SERIES[1],
        lw=1.4,
        ms=4,
        label="the whole timeline",
    )
    axis.plot(scans, window_ms, "o-", color=SERIES[2], lw=1.4, ms=4, label="one window")
    axis.set_xscale("log")
    axis.set_yscale("log")
    _style(axis, "scan length")
    axis.set_xlabel("repetitions", fontsize=10.8)
    axis.set_ylabel("ms", fontsize=10.8)
    legend(axis, ncol=1)

    axis = panels[1]
    index = np.arange(len(families))
    axis.bar(
        index - 0.19,
        [f["direct"] / 1e6 for f in families],
        0.36,
        color=SERIES[1],
        label="convolved whole",
    )
    axis.bar(
        index + 0.19,
        [f["assembled"] / 1e6 for f in families],
        0.36,
        color=SERIES[2],
        label="assembled per shape",
    )
    axis.set_yscale("log")
    axis.set_xticks(index)
    axis.set_xticklabels(
        [f["label"] for f in families], fontsize=9.7, rotation=25, ha="right"
    )
    _style(axis, "one window, two ways")
    axis.set_ylabel("multiply-adds (M)", fontsize=10.8)
    legend(axis, ncol=1)

    axis = panels[2]
    axis.plot(
        [f["ms"] for f in families],
        [f["direct"] / f["assembled"] for f in families],
        "o",
        color=SERIES[3],
        ms=6,
        mew=0,
    )
    for f in families:
        axis.annotate(
            f["label"],
            (f["ms"], f["direct"] / f["assembled"]),
            textcoords="offset points",
            xytext=(6, -3),
            fontsize=9.5,
            color=MUTED,
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(4, 6000)
    _style(axis, "what the assembly buys")
    axis.set_xlabel("window duration (ms)", fontsize=10.8)
    axis.set_ylabel("speedup", fontsize=10.8)

    figure.suptitle(
        "What the stimulation check's cost actually depends on",
        x=0.02,
        y=0.995,
        ha="left",
        va="top",
        fontsize=12.0,
        color=INK,
    )
    return save(figure, "assembly_cost")


# ----------------------------------------------------------------------
# 3 -- the verdict
# ----------------------------------------------------------------------


def epi_verdict():
    from pulserver.app import epi2D_sequence

    sequence = epi2D_sequence.main(
        plot=False,
        write_seq=False,
        system=SYSTEM,
        n_x=64,
        n_y=64,
        segments=1,
        n_dummy=0,
        n_repetitions=16,
        readout_bandwidth_hz=250e3,
    )
    if isinstance(sequence, tuple):
        sequence = sequence[0]
    dt, tr_samples, kernel = window_geometry(sequence)
    window = window_response(sequence, dt, kernel, tr_samples) * 100.0
    millis = np.arange(tr_samples) * dt * 1e3

    figure, axis = plt.subplots(figsize=(8.6, 3.84))
    figure.subplots_adjust(top=0.80, bottom=0.18)
    axis.plot(millis, window, color=SERIES[0], lw=1.0, zorder=3)
    peak = int(np.argmax(window))
    axis.plot(millis[peak], window[peak], "o", color="#e34948", ms=6, mew=0, zorder=6)
    axis.annotate(
        f"{window[peak]:.0f} % of threshold",
        (millis[peak], window[peak]),
        textcoords="offset points",
        xytext=(10, -2),
        fontsize=10.8,
        color="#e34948",
    )
    thresholds(axis, label=True)
    _style(axis, "")
    axis.set_xlim(0, millis[-1])
    axis.set_ylim(0, max(118, 1.12 * window.max()))
    axis.set_xlabel("time within the repetition (ms)", fontsize=10.8)
    axis.set_ylabel("stimulation (% of threshold)", fontsize=10.8)
    figure.suptitle(
        "The verdict: one number, the peak of the combined response over the window",
        x=0.02,
        ha="left",
        fontsize=12.0,
        color=INK,
    )
    print(f"    peak {window.max():.2f} % at {millis[peak]:.2f} ms")
    return save(figure, "epi_verdict")


FIGURES = {
    "canonical_tr": canonical_tr,
    "assembly_cost": assembly_cost,
    "epi_verdict": epi_verdict,
}


def main(names):
    for name in names or FIGURES:
        print(name)
        FIGURES[name]()


if __name__ == "__main__":
    main(sys.argv[1:])
