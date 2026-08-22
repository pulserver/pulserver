#!/usr/bin/env python3
"""The equivalence figures for ``explanations/performance/mechanical_resonance``.

Documentation-only tooling. Every claim on that page is "the fast evaluation
equals the slow one", so every figure here draws both and, where the numbers
are meant to coincide, the residual as well.

``canonical_tr``
    One window judged against every repetition it stands for, in the three
    situations a repetition can differ in: nothing varies, an amplitude or a
    rotation varies, the waveform itself varies.
``basis_equivalence``
    A multishot readout's arms, their singular values, and the arm-by-arm
    agreement between a scan whose arms are turned by a rotation and the same
    scan with them written out as separate waveforms.
``shape_response``
    Per gradient family, the closed-form response the engine evaluates
    against a direct numerical Fourier integral of the rendered repetition.
``finite_reps``
    A finite number of repetitions puts real drive between the harmonics.
    The dense spectrum of the rendered record, the lobes of the Dirichlet
    kernel, and where the probes have to sit to see them.
``epi_comb``
    A single-shot echo train read against two guarded bands: the verdict
    itself, on the lines a band contains.

Two conventions hold across every panel, both so a quiet sequence looks
quiet:

* the vertical axis is equivalent sustained amplitude in mT/m, framed at
  0.1-30 mT/m on every logarithmic axis. The audit
  (``mechres_calibration.py``) measured 0.13 mT/m for the quietest sequence
  in the corpus and 22.5 for the loudest, so a tighter frame would be
  magnifying the difference between two kinds of silence;
* a guarded band is drawn with the threshold the gate actually applies --
  7.5 mT/m where the band states no amplitude, and 0.81 x the stated plateau
  where it states one.

Usage:
    <venv>/bin/python docs/_bench/mechres_equivalence.py [name ...]
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pulserver.pypulseq as pp  # noqa: E402
from _figures import FAINT, INK, MUTED, SERIES, _style  # noqa: E402
from pulserver._ext.pulseg import _calc_mech_resonances  # noqa: E402

ASSETS = Path(__file__).resolve().parents[1] / "explanations" / "assets" / "mechanical_resonance"

GAMMA = 42.576e3  #: Hz/m per mT/m

#: ``SA_AEQ_POLICY_MT_PER_M`` and ``SA_AEQ_TRAIN_SHAPE``. Drawn, not recomputed.
POLICY_MT_PER_M = 7.5
TRAIN_SHAPE = 0.8106

#: The frame every logarithmic A_eq axis is drawn in -- see the module note.
FRAME = (0.1, 30.0)

#: ``pulseg_calc_mech_resonances`` amplitude modes.
BOUND, ACTUAL = 0, 2

SYSTEM = pp.Opts(max_grad=50, grad_unit="mT/m", max_slew=200, slew_unit="T/m/s")
WEAK = pp.Opts(max_grad=33, grad_unit="mT/m", max_slew=120, slew_unit="T/m/s")


# ----------------------------------------------------------------------
# the two paths every figure compares
# ----------------------------------------------------------------------


def engine_lines(sequence, index=0, mode=BOUND, max_freq=2000.0, bands=()):
    """``(f, A_eq[3, n])`` at the TR harmonics, in mT/m, from the C engine."""
    structure = sequence._structure_for("bound")
    spectra = _calc_mech_resonances(
        structure.collection,
        0,
        int(index),
        mode,
        target_resolution_hz=1.0 / structure.tr_duration,
        max_freq_hz=max_freq,
        forbidden_bands=[(lo, hi, amp * GAMMA) for lo, hi, amp in bands],
    )
    freqs = np.asarray(spectra["analytical_peak_freqs"], float)
    amps = np.stack(
        [np.asarray(spectra[f"analytical_peak_amp_g{ax}"], float) for ax in "xyz"]
    )
    return freqs, amps / GAMMA


def engine_envelope(sequence, resolution_hz, max_freq, index=0, mode=BOUND):
    """``(f, A_eq[3, n])`` on a dense uniform grid rather than at the harmonics."""
    structure = sequence._structure_for("bound")
    spectra = _calc_mech_resonances(
        structure.collection,
        0,
        int(index),
        mode,
        target_resolution_hz=resolution_hz,
        max_freq_hz=max_freq,
        forbidden_bands=[],
    )
    freqs = np.asarray(spectra["envelope_freqs_hz"], float)
    amps = np.stack([np.asarray(spectra[f"envelope_amp_g{ax}"], float) for ax in "xyz"])
    return freqs, amps / GAMMA


def rendered(sequence, index=0):
    """One repetition on the gradient raster: ``(dt, g[3, n])`` in mT/m."""
    channels = sequence._structure_for("figure").waveform(int(index)).waveforms()
    times = channels[0][0]
    grid = float(times[1] - times[0])
    # The closing sample repeats the opening one of the next repetition.
    return grid, np.stack([np.asarray(c[1], float)[:-1] for c in channels]) / GAMMA


def numerical_lines(sequence, freqs, index=0, oversample=32):
    """``A_eq[3, n]`` from a direct Fourier integral of the rendered repetition.

    The slow alternative in full: render the repetition, interpolate it the
    way the model says the field behaves between raster samples, and
    integrate ``g(t) e^{-2 pi i f t}`` term by term at each frequency asked
    for. Nothing here shares a line of code with the engine.
    """
    grid, waves = rendered(sequence, index)
    period = grid * waves.shape[1]
    coarse = np.arange(waves.shape[1] + 1) * grid
    fine = np.linspace(0.0, period, waves.shape[1] * oversample + 1)
    out = np.empty((3, freqs.size))
    kernel = np.exp(-2j * np.pi * freqs[:, None] * fine[None, :])
    for axis in range(3):
        closed = np.append(waves[axis], waves[axis][0])
        dense = np.interp(fine, coarse, closed)
        integral = np.trapezoid(dense[None, :] * kernel, fine, axis=1)
        out[axis] = 2.0 * np.abs(integral) / period
    return out


def engine_lines_complex(sequence, index, mode, max_freq=2000.0):
    """``(f, amp[3, n], phase[3, n])`` -- the engine's own complex line values."""
    structure = sequence._structure_for("bound")
    spectra = _calc_mech_resonances(
        structure.collection,
        0,
        int(index),
        mode,
        target_resolution_hz=1.0 / structure.tr_duration,
        max_freq_hz=max_freq,
        forbidden_bands=[],
    )
    freqs = np.asarray(spectra["analytical_peak_freqs"], float)
    amps = np.stack(
        [np.asarray(spectra[f"analytical_peak_amp_g{ax}"], float) for ax in "xyz"]
    )
    phases = np.stack(
        [np.asarray(spectra[f"analytical_peak_phase_g{ax}"], float) for ax in "xyz"]
    )
    return freqs, amps / GAMMA, phases


def scan_at_harmonics(sequence, max_freq=2000.0):
    """``(f, A_eq[n])`` of the whole scan at the TR harmonics, worst axis.

    Summed from the engine's own per-repetition complex lines, so both sides
    of the bound comparison describe the same field. At a TR harmonic the
    inter-repetition phase factor is unity, so the scan's line is simply the
    mean of what the repetitions put there:
    ``A_eq_scan(f_k) = |sum_m A_eq_m(f_k) e^{i phi_m}| / M``.
    """
    reps = sequence._structure_for("bound").num_trs
    total = None
    for m in range(reps):
        freqs, amps, phases = engine_lines_complex(sequence, m, ACTUAL, max_freq)
        term = amps * np.exp(1j * phases)
        total = term if total is None else total + term
    return freqs, (np.abs(total) / reps).max(0)


def record_spectrum(sequence, freqs):
    """``A_eq[n]`` of the whole scan, worst axis, rendered repetition by repetition.

    The naive analysis in full: no period, no line structure, no reuse -- every
    repetition rendered as it actually plays, concatenated, and transformed.
    This is the object the canonical window has to bound, and unlike a
    per-repetition comparison it exists at every frequency, not only at the
    harmonics.
    """
    structure = sequence._structure_for("bound")
    grid, _ = rendered(sequence, 0)
    record = np.concatenate(
        [rendered(sequence, i)[1] for i in range(structure.num_trs)], axis=1
    )
    total = grid * record.shape[1]
    times = np.arange(record.shape[1]) * grid
    kernel = np.exp(-2j * np.pi * freqs[:, None] * times[None, :])
    correction = np.sinc(freqs * grid) ** 2
    return np.stack(
        [2.0 * np.abs(kernel @ record[a]) * grid * correction / total for a in range(3)]
    ).max(0)


def scan_spectrum(sequence, num_reps, freqs, index=0):
    """``A_eq[3, n]`` of ``num_reps`` back-to-back repetitions, rendered and transformed.

    The naive whole-record analysis: no period, no line structure, no reuse.
    The linear interpolant of uniform samples is the sample train convolved
    with a triangle of width ``2 dt``, so its exact transform is the plain
    sum times ``dt sinc^2(f dt)`` -- which keeps this a reference rather than
    a second approximation.
    """
    grid, waves = rendered(sequence, index)
    total = grid * waves.shape[1] * num_reps
    times = np.arange(waves.shape[1] * num_reps) * grid
    kernel = np.exp(-2j * np.pi * freqs[:, None] * times[None, :])
    correction = np.sinc(freqs * grid) ** 2
    out = np.empty((3, freqs.size))
    for axis in range(3):
        record = np.tile(waves[axis], num_reps)
        out[axis] = 2.0 * np.abs(kernel @ record) * grid * correction / total
    return out


# ----------------------------------------------------------------------
# drawing helpers
# ----------------------------------------------------------------------


def frame(axis, title, xmax, ylabel=True):
    _style(axis, title)
    axis.set_yscale("log")
    axis.set_ylim(*FRAME)
    axis.set_xlim(0, xmax)
    axis.set_xlabel("frequency (Hz)", fontsize=8)
    if ylabel:
        axis.set_ylabel("$A_{eq}$ (mT/m)", fontsize=8)
    from matplotlib.ticker import FixedLocator, FuncFormatter

    ticks = [t for t in (0.1, 0.3, 1.0, 3.0, 10.0, 30.0) if FRAME[0] <= t <= FRAME[1]]
    axis.yaxis.set_major_locator(FixedLocator(ticks))
    axis.yaxis.set_minor_locator(FixedLocator([]))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def draw_bands(axis, bands):
    """A guarded band and the threshold the gate holds it to."""
    for low, high, plateau in bands:
        level = TRAIN_SHAPE * plateau if plateau > 0 else POLICY_MT_PER_M
        axis.axvspan(low, high, color="#e34948", alpha=0.08, lw=0, zorder=0)
        axis.plot([low, high], [level] * 2, color="#e34948", lw=1.2, ls=(0, (4, 2)), zorder=6)


def legend(target, handles=None, labels=None, **kwargs):
    kwargs.setdefault("fontsize", 7.6)
    kwargs.setdefault("frameon", False)
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
# the sequences
# ----------------------------------------------------------------------


def build(module, **kwargs):
    from importlib import import_module

    sequence = getattr(import_module("pulserver.app"), module).main(
        plot=False, write_seq=False, n_dummy=0, **kwargs
    )
    return sequence[0] if isinstance(sequence, tuple) else sequence


def epi_series(repetitions=16):
    """Single-shot EPI, played as a time series: every repetition identical."""
    return build(
        "epi2D_sequence",
        system=SYSTEM,
        n_x=64,
        n_y=64,
        segments=1,
        n_repetitions=repetitions,
        readout_bandwidth_hz=250e3,
    )


def cartesian(phase_encodes=32):
    """A spoiled gradient echo: one position steps its amplitude, nothing else."""
    return build(
        "gre2D_sequence",
        system=SYSTEM,
        n_x=128,
        n_y=phase_encodes,
        tr=9e-3,
        te=4.5e-3,
        readout_bandwidth_hz=83e3,
    )


def spiral(arms=8, rotated=False, matrix=192, tr=None):
    """A long spiral readout, its arms either turned by a rotation or written out.

    Long on purpose, and filling its repetition. What makes a line spectrum
    legible is the readout occupying most of the period: a short arm inside a
    long TR beats against everything else in the repetition and swings the
    lines by a decade from one to the next, while an arm that fills its TR
    sweeps a wide range smoothly. The default here is a 36 ms readout in a
    42 ms repetition, which is also what a real protocol looks like. Golden
    angles, so no two arms are a sign flip or an axis swap of another.
    """
    return build(
        "gre_spiral2D_sequence",
        n_x=matrix,
        n_arms=arms,
        tr=tr,
        angle_scheme="golden",
        readout_bandwidth_hz=125e3,
        use_rotation_ext=rotated,
    )


#: Where every inspected vendor band falls -- the only frequencies the
#: verdict looks at. Shaded in the panels so a peak outside it is visibly
#: not the thing being judged.
TERRITORY = (515.0, 1650.0)

#: The controlled scan: one 3D stack-of-spirals repetition, played sixteen
#: times, with the two things that can vary across repetitions switched on
#: and off independently. Sixteen distinct golden-angle arms, which is what a
#: real interleaved scan plays; the structural TR stays at one shot either
#: way, since a waveform's samples are not part of a gradient definition.
STACK = dict(n_x=64, n_z=8, n_arms=16, angle_scheme="golden", n_dummy=0,
             tr=10e-3, readout_bandwidth_hz=250e3, use_rotation_ext=False)
REPETITIONS = 16


def stack_kernel():
    from pulserver.app import gre_stack_of_spirals3D_sequence as plugin

    return plugin, plugin.StackOfSpiralsKernel(SYSTEM, **STACK)


def controlled_scan(kernel_pair, arm_varies, encode_varies):
    """Sixteen repetitions of the shipped kernel, varying only what is asked."""
    plugin, kernel = kernel_pair
    sequence = pp.Sequence(SYSTEM)
    phases = pp.make_rf_spoiling_schedule(REPETITIONS, increment=np.deg2rad(117.0))
    for m in range(REPETITIONS):
        arm = m if arm_varies else 0
        partition = (m % 8) if encode_varies else 2
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
# 1 -- the canonical window against the scan it stands for
# ----------------------------------------------------------------------

CASES = (
    ("A", "nothing varies", False, False),
    ("B", "the encode amplitude varies", False, True),
    ("C", "the readout waveform varies", True, False),
    ("D", "both vary", True, True),
)

MAX_FREQ = 2000.0


def canonical_tr():
    kernel = stack_kernel()

    figure, axes = plt.subplots(2, 4, figsize=(11.0, 6.0), height_ratios=(1.0, 0.52))
    figure.subplots_adjust(hspace=0.55, wspace=0.22, top=0.775, bottom=0.09,
                           left=0.065, right=0.99)

    for column, (letter, what, arm_varies, encode_varies) in enumerate(CASES):
        sequence = controlled_scan(kernel, arm_varies, encode_varies)
        reps = sequence._structure_for("bound").num_trs

        # Both sides of the bound, in the engine's own model, at the harmonics.
        freqs, scan = scan_at_harmonics(sequence, MAX_FREQ)
        _, window = engine_lines(sequence, mode=BOUND, max_freq=MAX_FREQ)
        window = window.max(0)

        # And, as context, where the scan's energy actually sits: the rendered
        # record, which has structure between the harmonics that no single
        # repetition has.
        dense_f, _ = engine_envelope(sequence, 0.5, MAX_FREQ, mode=BOUND)
        dense_scan = record_spectrum(sequence, dense_f)

        top = axes[0, column]
        top.axvspan(*TERRITORY, color=FAINT, alpha=0.30, lw=0, zorder=0)
        top.plot(dense_f, np.maximum(dense_scan, 1e-3), color=SERIES[1], lw=0.5,
                 alpha=0.55, zorder=1)
        top.plot(freqs, scan, color=SERIES[1], lw=1.2, zorder=3)
        top.plot(freqs, window, color=INK, lw=1.3, ls=(0, (4, 2)), zorder=4)
        frame(top, f"{letter}   {what}", MAX_FREQ, column == 0)
        inside = (freqs >= TERRITORY[0]) & (freqs <= TERRITORY[1])
        top.text(0.035, 0.055,
                 f"in band: {window[inside].max():.2f} judged,"
                 f"\n{scan[inside].max():.2f} driven",
                 transform=top.transAxes, fontsize=7.4, color=MUTED)

        ratio = window / np.maximum(scan, 1e-12)
        loud = scan > 0.05 * scan.max()
        bottom = axes[1, column]
        bottom.axvspan(*TERRITORY, color=FAINT, alpha=0.30, lw=0, zorder=0)
        bottom.plot(freqs[loud], ratio[loud], color=SERIES[0], lw=0, marker="o",
                    ms=2.6, mew=0, zorder=2)
        bottom.axhline(1.0, color=MUTED, lw=0.8, ls=(0, (4, 3)))
        _style(bottom, f"never below 1 · median {np.median(ratio[loud]):.3f}×")
        bottom.set_xlim(0, MAX_FREQ)
        bottom.set_ylim(0.97, 1.35)
        bottom.set_xlabel("frequency (Hz)", fontsize=8)
        if column == 0:
            bottom.set_ylabel("judged / driven", fontsize=8)
        print(f"    {letter} {what:28} min {ratio.min():.4f} median {np.median(ratio[loud]):.3f} "
              f"| in band judged {window[inside].max():.3f} driven {scan[inside].max():.3f} "
              f"({window[inside].max()/scan[inside].max():.3f}x)")

    legend(
        figure,
        [Line2D([], [], color=SERIES[1], lw=1.2),
         Line2D([], [], color=INK, lw=1.3, ls=(0, (4, 2))),
         Line2D([], [], color=SERIES[1], lw=0.8, alpha=0.55),
         Line2D([], [], color=FAINT, lw=7)],
        ["what the whole scan drives, at the TR harmonics",
         "the canonical window, which is what the gate judges",
         "the whole scan at every frequency",
         "where vendor bands fall"],
        loc="upper center", bbox_to_anchor=(0.5, 0.905), ncols=4,
    )
    figure.suptitle(
        "One repetition played sixteen times, each kind of variation switched on alone — "
        "the window is above the scan at every line, and by a few percent",
        x=0.02, ha="left", fontsize=10, color=INK,
    )
    return save(figure, "canonical_tr")


# ----------------------------------------------------------------------
# 2 -- the rank basis
# ----------------------------------------------------------------------


def basis_equivalence(arms=8):
    written = spiral(arms, rotated=False)
    turned = spiral(arms, rotated=True)

    figure = plt.figure(figsize=(9.6, 3.4))
    grid = figure.add_gridspec(1, 3, width_ratios=(0.9, 0.9, 1.5), wspace=0.34)

    # -- the arms themselves ------------------------------------------------
    axis = figure.add_subplot(grid[0, 0])
    _, first = rendered(written, 0)
    varies = np.zeros(first.shape[1], bool)
    for index in range(1, arms):
        _, other = rendered(written, index)
        varies |= np.any(np.abs(other - first) > 1e-9, axis=0)
    window = np.flatnonzero(varies)
    span = slice(window[0], window[-1] + 1)
    grid_s, _ = rendered(written, 0)
    times = np.arange(span.start, span.stop) * grid_s * 1e3
    stacked = []
    for index in range(arms):
        _, wave = rendered(written, index)
        stacked.append(wave[:, span])
        axis.plot(times, wave[0, span], color=SERIES[index % len(SERIES)], lw=0.8)
    _style(axis, f"{arms} arms, on the readout axis")
    axis.set_xlabel("time (ms)", fontsize=8)
    axis.set_ylabel("$G_x$ (mT/m)", fontsize=8)
    middle = 0.5 * (times[0] + times[-1])
    axis.set_xlim(middle, middle + 2.0)

    # -- their singular values ---------------------------------------------
    axis = figure.add_subplot(grid[0, 1])
    for physical in range(2):
        matrix = np.stack([wave[physical] for wave in stacked])
        singular = np.linalg.svd(matrix, compute_uv=False)
        singular = singular / singular[0]
        axis.semilogy(
            np.arange(1, singular.size + 1),
            np.maximum(singular, 1e-17),
            "o-",
            color=SERIES[physical],
            lw=1.1,
            ms=3.5,
            label=f"$G_{'xy'[physical]}$",
        )
        print(f"    singular values, G{'xy'[physical]}: {np.round(singular[:4], 12)}")
    axis.axhline(1e-6, color=MUTED, lw=0.8, ls=(0, (4, 3)))
    axis.text(1.1, 2.2e-6, "kept above here", fontsize=7, color=MUTED)
    _style(axis, "and their singular values")
    axis.set_xlabel("index", fontsize=8)
    axis.set_ylabel("$\\sigma_r / \\sigma_1$", fontsize=8)
    axis.set_ylim(1e-11, 5.0)
    legend(axis, loc="lower right")

    # -- the verdict, arm by arm, both encodings ---------------------------
    axis = figure.add_subplot(grid[0, 2])
    for index in range(arms):
        freqs, one = engine_lines(turned, index, ACTUAL)
        _, other = engine_lines(written, index, ACTUAL)
        axis.plot(freqs, one.max(0), color=SERIES[index % len(SERIES)], lw=1.0)
        axis.plot(freqs, other.max(0), color=INK, lw=0, marker="o", ms=2.2, mfc="none", mew=0.6)
        gap = np.abs(one.max(0) - other.max(0)) / np.maximum(one.max(0), 1e-12)
        loud = one.max(0) > 0.05 * one.max()
        print(f"    arm {index}: rotation vs written out, median gap {np.median(gap[loud]):.2e}")
    draw_bands(axis, [(550.0, 700.0, 0.0), (1150.0, 1300.0, 0.0)])
    frame(axis, "the same scan, both encodings, arm by arm", 2000.0)
    legend(
        axis,
        [
            Line2D([], [], color=MUTED, lw=1.0),
            Line2D([], [], color=INK, lw=0, marker="o", ms=2.2, mfc="none", mew=0.6),
        ],
        ["one waveform, turned by a rotation", "the arms written out, compressed to a rank basis"],
        loc="upper left",
    )

    figure.subplots_adjust(top=0.80)
    figure.suptitle(
        "Distinct is not independent: turned arms span two dimensions, however many of them there are",
        x=0.02,
        ha="left",
        fontsize=10,
        color=INK,
    )
    return save(figure, "basis_equivalence")


# ----------------------------------------------------------------------
# 3 -- the response of each gradient family
# ----------------------------------------------------------------------

FAMILIES = (
    ("trapezoids only", "gradient echo, 32 phase encodes", cartesian, 2500.0),
    ("a compressed train of them", "single-shot EPI", epi_series, 2500.0),
    ("one long arbitrary waveform", "spiral, one arm", lambda: spiral(1, matrix=64), 2000.0),
)


def shape_response():
    figure, axes = plt.subplots(2, 3, figsize=(9.8, 5.4), height_ratios=(1.0, 0.55))
    figure.subplots_adjust(hspace=0.52, wspace=0.24, top=0.82, bottom=0.10)

    for column, (title, subtitle, factory, max_freq) in enumerate(FAMILIES):
        sequence = factory()
        start = time.perf_counter()
        freqs, fast = engine_lines(sequence, 0, ACTUAL, max_freq)
        fast_ms = 1e3 * (time.perf_counter() - start)
        start = time.perf_counter()
        slow = numerical_lines(sequence, freqs)
        slow_ms = 1e3 * (time.perf_counter() - start)

        top = axes[0, column]
        top.plot(freqs, slow.max(0), color=SERIES[1], lw=2.4, alpha=0.5)
        top.plot(freqs, fast.max(0), color=INK, lw=0.9)
        frame(top, f"{title}\n{subtitle}", max_freq, column == 0)

        loud = fast.max(0) > 0.01 * fast.max()
        residual = np.abs(fast.max(0) - slow.max(0)) / np.maximum(fast.max(0), 1e-12)
        bottom = axes[1, column]
        bottom.semilogy(freqs[loud], np.maximum(residual[loud], 1e-12), ".", color=SERIES[0], ms=3)
        bottom.set_ylim(1e-9, 1e-2)
        bottom.set_xlim(0, max_freq)
        _style(bottom, "relative difference" if column == 0 else "")
        bottom.text(
            0.98, 0.06,
            f"median {np.median(residual[loud]):.0e}\n{fast_ms:.0f} ms against {slow_ms:.0f} ms",
            transform=bottom.transAxes, ha="right", va="bottom",
            fontsize=7.4, color=MUTED,
        )
        bottom.set_xlabel("frequency (Hz)", fontsize=8)
        if column == 0:
            bottom.set_ylabel("|fast − slow| / fast", fontsize=8)
        print(
            f"    {title:28} median {np.median(residual[loud]):.2e} "
            f"max {residual[loud].max():.2e}  {fast_ms:.1f} ms vs {slow_ms:.0f} ms"
        )

    legend(
        figure,
        [Line2D([], [], color=SERIES[1], lw=2.4, alpha=0.5), Line2D([], [], color=INK, lw=0.9)],
        [
            "a direct Fourier integral of the rendered repetition",
            "the closed forms the engine evaluates",
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncols=2,
    )
    figure.suptitle(
        "Every family's response, against integrating the rendered repetition",
        x=0.02,
        ha="left",
        fontsize=10,
        color=INK,
    )
    return save(figure, "shape_response")


# ----------------------------------------------------------------------
# 4 -- what a finite number of repetitions puts between the harmonics
# ----------------------------------------------------------------------


def finite_reps(repetitions=16):
    sequence = epi_series(repetitions)
    structure = sequence._structure_for("bound")
    period = structure.tr_duration
    spacing = 1.0 / period

    freqs, amps = engine_lines(sequence, 0, ACTUAL, 1200.0)
    inside = (freqs > 500.0) & (freqs < 900.0)
    harmonic = int(np.argmax(np.where(inside, amps.max(0), 0.0))) + 1
    centre = harmonic * spacing

    half = 0.32 * spacing
    dense = np.linspace(centre - half, centre + half, 3000)
    record = scan_spectrum(sequence, repetitions, dense).max(0)

    def probe(offsets):
        """What the engine evaluates at an offset from the harmonic."""
        points = (harmonic + np.asarray(offsets, float)) * spacing
        single = numerical_lines(sequence, points).max(0)
        turns = points * period
        kernel = np.abs(np.sin(repetitions * np.pi * turns))
        kernel = kernel / np.maximum(repetitions * np.abs(np.sin(np.pi * turns)), 1e-300)
        kernel[np.abs(np.sin(np.pi * turns)) < 1e-12] = 1.0
        return points, single * kernel

    lobe_offsets = np.concatenate(
        [(np.arange(4) + 0.5) / repetitions, -(np.arange(4) + 0.5) / repetitions]
    )
    null_offsets = np.concatenate(
        [np.arange(1, 5) / repetitions, -np.arange(1, 5) / repetitions]
    )

    figure, axes = plt.subplots(1, 2, figsize=(9.8, 4.3), width_ratios=(1.35, 1.0))
    figure.subplots_adjust(wspace=0.26, top=0.68, bottom=0.16)

    axis = axes[0]
    axis.plot(dense, record, color=SERIES[1], lw=1.4, alpha=0.65, zorder=1)
    lobe_freqs, on_lobe = probe(lobe_offsets)
    axis.plot(lobe_freqs, on_lobe, "o", color=INK, ms=5, mew=0, zorder=4)
    null_freqs, on_null = probe(null_offsets)
    axis.plot(null_freqs, on_null, "o", color=SERIES[0], ms=6, mfc="none", mew=1.2, zorder=4)
    exact_freq, exact = probe([0.0])
    axis.plot(exact_freq, exact, "o", color="#e34948", ms=6.5, mew=0, zorder=5)
    axis.axhline(POLICY_MT_PER_M, color="#e34948", lw=1.0, ls=(0, (4, 2)), zorder=3)
    axis.text(
        dense[0], POLICY_MT_PER_M * 1.03, "7.5 mT/m", fontsize=7.2, color="#e34948", va="bottom"
    )
    _style(axis, f"one harmonic of a {repetitions}-repetition scan, at {centre:.0f} Hz")
    axis.set_xlabel("frequency (Hz)", fontsize=8)
    axis.set_ylabel("$A_{eq}$ (mT/m)", fontsize=8)
    axis.set_xlim(dense[0], dense[-1])
    axis.set_ylim(0, 1.12 * float(exact[0]))
    legend(
        figure,
        [
            Line2D([], [], color=SERIES[1], lw=1.4, alpha=0.65),
            Line2D([], [], color="#e34948", lw=0, marker="o", ms=6.5),
            Line2D([], [], color=INK, lw=0, marker="o", ms=5),
            Line2D([], [], color=SERIES[0], lw=0, marker="o", ms=6, mfc="none", mew=1.2),
        ],
        [
            "the whole record, rendered and transformed",
            "the harmonic itself",
            "probes at $(j+\\frac{1}{2})/M$",
            "probes at $j/M$ — the kernel's nulls",
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncols=2,
    )

    axis = axes[1]
    turns = np.linspace(0.0, 0.55, 4000)
    kernel = np.abs(np.sin(repetitions * np.pi * turns))
    kernel = kernel / np.maximum(repetitions * np.abs(np.sin(np.pi * turns)), 1e-300)
    kernel[0] = 1.0
    axis.plot(turns, kernel, color=FAINT, lw=1.0)
    peaks = (np.arange(8) + 0.5) / repetitions
    heights = 2.0 / (np.pi * (2 * np.arange(8) + 1))
    axis.plot(peaks, heights, "o", color=INK, ms=4.5, mew=0)
    axis.plot(
        np.arange(1, 9) / repetitions,
        np.zeros(8),
        "o",
        color=SERIES[0],
        ms=5,
        mfc="none",
        mew=1.2,
    )
    _style(axis, "$|D_M|/M$, the kernel those probes sit on")
    axis.set_xlabel("$f\\,T_{TR}$, from one harmonic towards the next", fontsize=8)
    axis.set_ylabel("attenuation", fontsize=8)
    axis.set_xlim(0, 0.55)
    axis.set_ylim(-0.04, 1.08)
    for index in range(4):
        axis.text(
            peaks[index] + 0.006, heights[index] + 0.035,
            f"{heights[index]:.2f}", fontsize=7.2, color=MUTED,
        )
    axis.text(
        0.42, 0.93,
        "peak heights do not depend on $M$,\nonly their spacing does — so four\n"
        "probes a side cover any scan length",
        transform=axis.transAxes, fontsize=7.4, color=MUTED, va="top",
    )

    figure.suptitle(
        "Between the harmonics sits drive a resonance would feel, and only a finite\n"
        "scan has it — which is why the check cannot stop at the harmonics",
        x=0.02, ha="left", fontsize=10, color=INK,
    )

    at_lobes = np.interp(lobe_freqs, dense, record)
    at_nulls = np.interp(null_freqs, dense, record)
    print(f"    harmonic {centre:.1f} Hz, A_eq {float(exact[0]):.3f} mT/m")
    print(f"    probes on the lobes  {np.round(on_lobe, 4)}")
    print(f"    the record there     {np.round(at_lobes, 4)}")
    print(f"    probes on the nulls  {np.round(on_null, 6)}")
    print(f"    the record there     {np.round(at_nulls, 6)}")
    return save(figure, "finite_reps")


# ----------------------------------------------------------------------
# 5 -- the verdict itself
# ----------------------------------------------------------------------

EPI_BANDS = ((650.0, 750.0, 0.0), (1500.0, 1600.0, 0.0))


def epi_comb(repetitions=16):
    sequence = epi_series(repetitions)
    structure = sequence._structure_for("bound")
    spectra = _calc_mech_resonances(
        structure.collection,
        0,
        0,
        BOUND,
        target_resolution_hz=1.0 / structure.tr_duration,
        max_freq_hz=2500.0,
        forbidden_bands=[(lo, hi, amp * GAMMA) for lo, hi, amp in EPI_BANDS],
    )
    freqs = np.asarray(spectra["analytical_peak_freqs"], float)
    amps = np.stack(
        [np.asarray(spectra[f"analytical_peak_amp_g{ax}"], float) for ax in "xyz"]
    ).max(0) / GAMMA
    candidates = np.asarray(spectra["candidate_freqs"], float)
    on_line = np.stack(
        [np.asarray(spectra[f"candidate_amps_g{ax}"], float) for ax in "xyz"]
    ).max(0) / GAMMA
    refused = np.asarray(spectra["candidate_violations"], int).astype(bool)

    figure, axis = plt.subplots(figsize=(9.8, 3.8))
    figure.subplots_adjust(top=0.72, bottom=0.18)
    axis.plot(freqs, amps, color=SERIES[0], lw=1.1, zorder=3)

    # The rest of what the verdict evaluates: the finite-repeat lobes between
    # the harmonics, at the frequencies the probes actually sit on. Drawing
    # them is what keeps the per-harmonic markers from reading as a
    # peak-finding -- the verdict covers the band, not a search over it.
    spacing = 1.0 / structure.tr_duration
    offsets = np.concatenate(
        [(np.arange(4) + 0.5) / repetitions, 1.0 - (np.arange(4) + 0.5) / repetitions]
    )
    probes = np.unique(
        np.concatenate([(k / spacing + offsets) * spacing for k in candidates])
    )
    axis.plot(probes, record_spectrum(sequence, probes), "o", color=SERIES[1],
              ms=2.6, mfc="none", mew=0.8, zorder=4,
              label="the finite-repeat lobes between them, evaluated too")
    draw_bands(axis, EPI_BANDS)

    for mask, colour, label in (
        (~refused, SERIES[2], "harmonic in band, its interval clears the threshold"),
        (refused, "#e34948", "harmonic in band, something in its interval does not"),
    ):
        if mask.any():
            axis.plot(candidates[mask], on_line[mask], "o", color=colour, ms=5,
                      mew=0, label=label, zorder=7)
    frame(axis, "", 2500.0)
    legend(axis, loc="upper center", bbox_to_anchor=(0.5, 1.20), ncols=3)
    axis.set_ylim(0.1, 40.0)
    figure.suptitle(
        "The verdict: every frequency a band covers, against the level that band allows",
        x=0.02, ha="left", fontsize=10, color=INK,
    )
    print(f"    candidates {candidates.size}, refused {int(refused.sum())}")
    for f_hz, a, bad in zip(candidates, on_line, refused):
        if bad:
            print(f"      refused at {f_hz:.1f} Hz (on the harmonic itself: {a:.2f} mT/m)")
    return save(figure, "epi_comb")


FIGURES = {
    "canonical_tr": canonical_tr,
    "basis_equivalence": basis_equivalence,
    "shape_response": shape_response,
    "finite_reps": finite_reps,
    "epi_comb": epi_comb,
}


def main(names):
    for name in names or FIGURES:
        print(name)
        FIGURES[name]()


if __name__ == "__main__":
    main(sys.argv[1:])
