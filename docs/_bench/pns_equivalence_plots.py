#!/usr/bin/env python3
"""The measurements and figures behind :doc:`../explanations/performance/pns`.

Documentation-only tooling, like everything else in this directory. Everything
the page claims about the Irnich rheobase/chronaxie model is computed here
rather than stored, so it can be rechecked rather than believed:

``assembly``
    The nerve response over one canonical TR, computed twice -- once by
    convolving the whole window, once by convolving each distinct gradient
    shape and adding scaled, shifted copies of the result -- against what
    :meth:`~pulserver.pypulseq.Sequence.calculate_pns` returns for the same
    window. The direct convolution is written straight from the kernel the
    Irnich model publishes, in double precision, so it is an independent
    reading of the same definition rather than another call into the library.

``scaling``
    The chronaxie cost of the same protocol at four scan lengths, convolved
    whole over the timeline and over the canonical TR, against what Pulserver
    returns for that window.

``envelope``
    A four-arm spiral gradient echo, whose repetitions play genuinely
    different waveforms, with every interleave's response drawn against the
    canonical TR's. Drawn for both encodings of the same scan: the arms
    turned by a ``ROTATIONS`` extension, and the arms written out as their
    own waveforms.

Usage:
    <venv>/bin/python docs/_bench/pns_equivalence_plots.py
    <venv>/bin/python docs/_bench/pns_equivalence_plots.py --only=scaling
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pulserver.pypulseq as pp

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "python" / "fixtures"
EXAMPLES = REPO_ROOT / "examples" / "sequence"
OUT_DIR = (
    Path(__file__).resolve().parents[1] / "explanations" / "assets" / "pns_performance"
)

# The constants the safety figures are drawn with: a generic body-gradient
# point, not any particular scanner's configuration. Rheobase is in T/m/s,
# the unit the slew waveform reaches a nerve model in.
IRNICH = {"chronaxie_us": 360.0, "rheobase": 20.0, "alpha": 0.333}

#: Chronaxie constants of kernel the Irnich model keeps before truncating the
#: 1/tau^2 tail, and the amount of history it therefore asks to be pre-padded
#: with. ``pulseg_pns_irnich.c`` uses the same number.
KERNEL_TAU = 20.0

#: An occurrence counts as a scaled copy of a template when every sample
#: matches to this fraction of the template's largest sample.
PROPORTIONAL_RTOL = 1e-5

# Categorical slots, assigned in this order and never cycled.
BLUE, ORANGE, AQUA, AMBER = "#2a78d6", "#eb6834", "#1baf7a", "#c98500"
RED, INK, MUTED, GRID = "#e34948", "#0b0b0b", "#52514e", "#d8d7d2"


# --- the model, written from its published definition ---------------------


def irnich_kernel(dt: float, hardware: dict) -> np.ndarray:
    """The Irnich impulse response on a ``dt``-second raster.

    Parameters
    ----------
    dt : float
        Sample spacing, in seconds.
    hardware : dict
        ``chronaxie_us``, ``rheobase`` (T/m/s) and ``alpha``.

    Returns
    -------
    numpy.ndarray
        The kernel, truncated at ``KERNEL_TAU`` chronaxie constants.
    """
    chronaxie = hardware["chronaxie_us"] * 1e-6
    s_min = hardware["rheobase"] / hardware["alpha"]
    taus = np.arange(int(KERNEL_TAU * chronaxie / dt) + 1) * dt
    return (dt / s_min) * chronaxie / (chronaxie + taus) ** 2


def convolved_directly(
    axes: list[np.ndarray],
    dt: float,
    gamma: float,
    kernel: np.ndarray,
    *,
    wrap: bool = True,
) -> np.ndarray:
    """The textbook route: pad, differentiate, convolve the whole thing.

    Parameters
    ----------
    axes : list of numpy.ndarray
        One gradient waveform per axis, in Hz/m, on a uniform ``dt`` raster.
    dt : float
        Sample spacing, in seconds.
    gamma : float
        Gyromagnetic ratio, in Hz/T.
    kernel : numpy.ndarray
        The nerve model's impulse response.
    wrap : bool, default True
        Take the padding from the start of the waveform, which is what a
        repetition played back to back with copies of itself sees. False pads
        with zeros: a scan played once, from rest.

    Returns
    -------
    numpy.ndarray
        ``(N + K - 1, 3)`` response, as a percentage of threshold per axis.
    """
    pad = kernel.size
    responses = []
    for waveform in axes:
        tail = waveform[np.arange(pad) % waveform.size] if wrap else np.zeros(pad)
        slew = np.diff(np.concatenate([waveform, tail])) / gamma / dt
        responses.append(np.convolve(slew, kernel)[: slew.size] * 100.0)
    return np.stack(responses, axis=-1)


def assembled_per_shape(
    axes: list[np.ndarray],
    lengths: list[int],
    dt: float,
    gamma: float,
    kernel: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, np.ndarray]], list[tuple[int, int, float]]]:
    """The assembled route: convolve each distinct shape once, then add.

    Mirrors ``pulseg_pns_memo.c``. Each block's slice of the window
    contributes its slew zero-extended on both sides, so the seam between two
    blocks receives the same pair of floats the direct forward difference
    subtracts.

    Parameters
    ----------
    axes : list of numpy.ndarray
        One gradient waveform per axis, in Hz/m, on a uniform ``dt`` raster.
    lengths : list of int
        Samples owned by each block, summing to the window length.
    dt : float
        Sample spacing, in seconds.
    gamma : float
        Gyromagnetic ratio, in Hz/T.
    kernel : numpy.ndarray
        The nerve model's impulse response.

    Returns
    -------
    response : numpy.ndarray
        ``(N + K - 1, 3)`` response, as a percentage of threshold per axis.
    shapes : list of tuple
        ``(axis, stored response)`` per distinct shape.
    occurrences : list of tuple
        ``(shape, offset, scale)`` for every slice that plays.
    """
    num_samples = axes[0].size
    total = num_samples + kernel.size - 1
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])

    shapes: list[tuple[int, np.ndarray]] = []
    keys: list[tuple[int, int, int, float, int]] = []
    occurrences: list[tuple[int, int, float]] = []

    for axis, waveform in enumerate(axes):
        for offset, length in zip(offsets, lengths, strict=True):
            block = waveform[offset : offset + length]
            if not np.any(block):
                continue
            pivot = int(np.argmax(np.abs(block)))
            match = None
            for index, (its_axis, its_length, its_pivot, value, first) in enumerate(
                keys
            ):
                if (its_axis, its_length, its_pivot) != (axis, length, pivot):
                    continue
                scale = block[pivot] / value
                reference = axes[its_axis][first : first + its_length]
                if np.allclose(
                    block,
                    scale * reference,
                    rtol=0,
                    atol=PROPORTIONAL_RTOL * abs(value),
                ):
                    match = (index, scale)
                    break
            if match is None:
                slew = np.empty(length + 1)
                slew[0] = block[0] / gamma / dt
                slew[1:length] = np.diff(block) / gamma / dt
                slew[length] = -block[length - 1] / gamma / dt
                shapes.append((axis, np.convolve(slew, kernel)))
                keys.append((axis, length, pivot, block[pivot], int(offset)))
                match = (len(shapes) - 1, 1.0)
            occurrences.append((match[0], int(offset) - 1, match[1]))

    response = [np.zeros(total) for _ in axes]

    def add(target: np.ndarray, stored: np.ndarray, at: int, scale: float) -> None:
        first, last = max(0, -at), min(stored.size, total - at)
        if last > first:
            target[at + first : at + last] += scale * stored[first:last]

    # The window is played back to back with copies of itself, so the leading
    # occurrences are replayed one window later: that is the same warmed-up
    # history the direct route gets from wrapping the waveform round.
    for shift in (0, num_samples):
        for shape, offset, scale in occurrences:
            axis, stored = shapes[shape]
            add(response[axis], stored, offset + shift, scale)

    return np.stack(response, axis=-1) * 100.0, shapes, occurrences


# --- window extraction ----------------------------------------------------


def canonical_window(seq, tr) -> tuple[list[np.ndarray], list[int], float]:
    """One TR's uniformly rastered gradient waveform, and its block lengths.

    Parameters
    ----------
    seq : pulserver.pypulseq.Sequence
        The sequence to read.
    tr : str or int
        ``"worst_case"`` for the envelope the gate judges, or an instance
        index.

    Returns
    -------
    axes : list of numpy.ndarray
        One waveform per gradient axis, in Hz/m.
    lengths : list of int
        Samples owned by each block, summing to the window length.
    dt : float
        Sample spacing, in seconds.
    """
    window = seq._structure_for("pns figures").waveform(tr)
    channels = [np.asarray(channel) for channel in window.waveforms()]
    dt = float(channels[0][0][1] - channels[0][0][0])
    lengths = [round(block["duration"] / dt) for block in window._blocks]
    # The extraction leaves one endpoint sample past the last block; the
    # assembly folds it into that block, as the safety core does.
    lengths[-1] += 1
    axes = [channel[1] for channel in channels]
    if sum(lengths) != axes[0].size:
        raise RuntimeError("block lengths do not tile the extracted window")
    return axes, lengths, dt


def whole_scan(seq, dt: float) -> list[np.ndarray]:
    """The whole timeline, on the raster the canonical window uses.

    Parameters
    ----------
    seq : pulserver.pypulseq.Sequence
        The sequence to render.
    dt : float
        Sample spacing, in seconds.

    Returns
    -------
    list of numpy.ndarray
        One gradient waveform per axis, in Hz/m.
    """
    gradients = seq.get_gradients()
    last = max(g.x[-1] for g in gradients if g is not None) - 1e-10
    times = (np.arange(int(np.ceil(last / dt))) + 0.5) * dt
    return [
        np.zeros(times.size) if g is None else np.asarray(g(times)) for g in gradients
    ]


def norm(response: np.ndarray) -> np.ndarray:
    """Per-axis percentages of threshold combined into one, as a fraction."""
    return np.sqrt((response**2).sum(axis=1)) / 100.0


def fastest(call, *args, repeats: int = 3, **kwargs) -> tuple[float, object]:
    """``(seconds, result)`` for the quickest of ``repeats`` runs."""
    best, answer = float("inf"), None
    for _ in range(repeats):
        started = time.perf_counter()
        answer = call(*args, **kwargs)
        best = min(best, time.perf_counter() - started)
    return best, answer


# --- figure 1: the same answer, assembled instead of convolved ------------


def assembly_figure() -> Path:
    """Direct convolution against per-shape assembly, over an EPI shot."""
    seq = pp.read(FIXTURES / "epi_2d.seq")
    axes, lengths, dt = canonical_window(seq, "worst_case")
    gamma = float(seq.system.gamma)
    kernel = irnich_kernel(dt, IRNICH)

    direct = convolved_directly(axes, dt, gamma, kernel)
    assembled, shapes, occurrences = assembled_per_shape(
        axes, lengths, dt, gamma, kernel
    )
    _, library, _, _ = seq.calculate_pns(IRNICH, tr="worst_case", do_plots=False)

    times = np.arange(direct.shape[0]) * dt * 1e3
    peak = norm(direct).max()

    kernel_ops = sum(
        (stored.size - kernel.size + 1) * kernel.size for _, stored in shapes
    )
    # Every occurrence is placed twice: once in the window, once a window
    # later, which is what reproduces the wrapped history.
    placement_ops = 2 * sum(shapes[shape][1].size for shape, _, _ in occurrences)
    print(
        f"  window {axes[0].size} samples, {len(lengths)} blocks, {len(occurrences)} slices, "
        f"{len(shapes)} distinct shapes\n"
        f"  multiply-adds: direct {3 * direct.shape[0] * kernel.size:,}, "
        f"assembled {kernel_ops + placement_ops:,} "
        f"({3 * direct.shape[0] * kernel.size / (kernel_ops + placement_ops):.1f}x fewer)\n"
        f"  peak: direct {100 * peak:.4f} %, assembled {100 * norm(assembled).max():.4f} %, "
        f"Pulserver {100 * library.max():.4f} %"
    )

    fig, (top, middle, bottom) = plt.subplots(
        3, 1, figsize=(9.5, 9.0), gridspec_kw={"height_ratios": [1.0, 1.1, 0.7]}
    )

    # A zoom on the blip train, where one template response is scaled and
    # shifted a few dozen times. The decomposition is per axis: the combined
    # norm below is a root-sum-square, which does not decompose.
    _plot_contributions(top, times, shapes, occurrences, direct, width_ms=1.6)

    middle.plot(
        times,
        100.0 * norm(direct),
        color=BLUE,
        alpha=0.35,
        linewidth=5.0,
        label="convolved directly over the window",
        solid_capstyle="round",
    )
    middle.plot(
        times,
        100.0 * library,
        color=INK,
        linewidth=1.1,
        label="assembled from per-shape responses (Pulserver)",
    )
    _mark_period(middle, axes[0].size * dt * 1e3)
    middle.axhline(100.0, color=RED, linewidth=1.0)
    middle.annotate(
        "100 % of threshold",
        xy=(times[-1], 100.0),
        xytext=(-4, 4),
        textcoords="offset points",
        ha="right",
        color=RED,
        fontsize=8,
    )
    middle.set_ylabel("combined stimulation [% of threshold]")
    middle.set_ylim(-4.0, 160.0)
    middle.legend(loc="upper left", frameon=False, fontsize=9)
    middle.set_title(
        f"{len(occurrences)} slices, {len(shapes)} distinct shapes"
        f"  \u2014  peak {100 * peak:.2f} % either way",
        fontsize=10,
        loc="left",
    )

    exact = 100.0 * (norm(assembled) - norm(direct))
    shipped = 100.0 * (library - norm(direct))
    bottom.plot(
        times,
        exact,
        color=AQUA,
        linewidth=1.2,
        label="assembly minus direct convolution, both in double precision"
        f"  (peak {np.abs(exact).max():.0e} %)",
    )
    bottom.plot(
        times,
        shipped,
        color=ORANGE,
        linewidth=1.0,
        label="Pulserver minus direct convolution, the library in float32"
        f"  (peak {np.abs(shipped).max():.0e} %)",
    )
    bottom.axhline(0.0, color=GRID, linewidth=0.8)
    _mark_period(bottom, axes[0].size * dt * 1e3, label=False)
    reach = np.abs(shipped).max()
    bottom.set_ylim(-2.3 * reach, 1.3 * reach)
    bottom.set_ylabel("difference [% of threshold]")
    bottom.set_xlabel("time within the canonical TR [ms]")
    bottom.legend(loc="lower left", frameon=False, fontsize=9)

    for axis in (top, middle, bottom):
        _style(axis)
    for axis in (middle, bottom):
        axis.set_xlim(0.0, times[-1])

    fig.suptitle(
        "EPI, one canonical TR: the response assembled per shape is the response convolved whole",
        fontsize=11,
    )
    return _save(fig, "assembly_equivalence")


def _plot_contributions(axis, times, shapes, occurrences, direct, width_ms) -> None:
    """One axis's response as the sum of its scaled, shifted per-shape pieces.

    Drawn for the axis with the most repeated slices -- the readout, on an
    echo train -- because the decomposition is per axis: the combined norm is
    a root-sum-square, and that does not decompose.
    """
    total = direct.shape[0]
    played = np.bincount(
        [shapes[shape][0] for shape, _, _ in occurrences], minlength=direct.shape[1]
    )
    carrier = int(np.argmax(played))
    mine = [
        (shape, at, scale)
        for shape, at, scale in occurrences
        if shapes[shape][0] == carrier
    ]

    # The stretch where the most slices start: that is where the sum is least
    # obviously the sum of anything, and where drawing the pieces earns its
    # place.
    width = int(width_ms / (times[1] - times[0]))
    onsets = np.array([max(0, at) for _, at, _ in mine])
    starts = np.arange(0, total - width)
    counts = np.array(
        [np.count_nonzero((onsets >= s) & (onsets < s + width)) for s in starts]
    )
    energy = np.convolve(direct[:, carrier] ** 2, np.ones(width), mode="valid")[
        : starts.size
    ]
    # Most slices starting inside the window; among windows tied on that, the
    # one carrying the most stimulation.
    first = int(starts[int(np.lexsort((energy, counts))[-1])])
    span = (float(times[first]), float(times[first + width]))
    inside = (times >= span[0]) & (times <= span[1])

    label = "one shape's stored response, scaled and shifted"
    drawn = 0
    for shape, at, scale in mine:
        _, stored = shapes[shape]
        piece = np.zeros(total)
        low, high = max(0, -at), min(stored.size, total - at)
        if high <= low:
            continue
        piece[at + low : at + high] = scale * stored[low:high] * 100.0
        if not np.any(np.abs(piece[inside]) > 1e-2):
            continue
        axis.plot(times, piece, color=MUTED, linewidth=0.8, alpha=0.65, label=label)
        label = None
        drawn += 1
    axis.plot(
        times,
        direct[:, carrier],
        color=INK,
        linewidth=1.6,
        label="their sum, which is the directly convolved response",
    )
    axis.set_xlim(*span)
    limit = np.abs(direct[inside, carrier]).max()
    axis.set_ylim(-1.25 * limit, 1.7 * limit)
    axis.set_ylabel(f"${'xyz'[carrier]}$ stimulation [%]")
    axis.set_xlabel("time within the canonical TR [ms]")
    axis.legend(loc="upper left", frameon=False, fontsize=9)
    axis.set_title(
        f"{width_ms:.1f} ms inside the echo train: {drawn} scaled, shifted copies"
        " of stored responses, and their sum",
        fontsize=10,
        loc="left",
    )


def _busiest_span(
    trace: np.ndarray, times: np.ndarray, width_ms: float
) -> tuple[float, float]:
    """The ``width_ms`` window carrying the most stimulation."""
    samples = max(2, int(width_ms / (times[1] - times[0])))
    energy = np.convolve(trace**2, np.ones(samples), mode="valid")
    start = int(np.argmax(energy))
    return float(times[start]), float(times[start + samples - 1])


# --- figure 2: one window that stands for every repetition ----------------


def _spiral(use_rotation_ext: bool):
    """The four-arm spiral gradient echo, in one of its two encodings.

    Golden-angle arms rather than a uniform fan, so that the four interleaves
    are four visibly different waveforms rather than two and their negatives.
    """
    if str(EXAMPLES) not in sys.path:
        sys.path.insert(0, str(EXAMPLES))
    import gre_spiral2D_sequence

    return gre_spiral2D_sequence.main(
        n_x=32,
        n_arms=4,
        angle_scheme="golden",
        n_dummy=2,
        tr=20e-3,
        readout_bandwidth_hz=125e3,
        use_rotation_ext=use_rotation_ext,
    )


def envelope_figure() -> Path:
    """Every repetition's response against the window the check judges."""
    written_out, rotated = _spiral(False), _spiral(True)
    structure = written_out._structure_for("pns figures")

    arms, seen = [], []
    for index in range(structure.num_trs):
        gradients = [np.asarray(c)[1] for c in structure.waveform(index).waveforms()]
        if any(np.array_equal(gradients[0], other[0]) for other in seen):
            continue
        seen.append(gradients)
        arms.append((index, gradients))

    window, _, dt = canonical_window(written_out, "worst_case")
    times = np.arange(window[0].size) * dt * 1e3
    period = times[-1]

    _, envelope, _, _ = written_out.calculate_pns(
        IRNICH, tr="worst_case", do_plots=False
    )
    _, turned, _, _ = rotated.calculate_pns(IRNICH, tr="worst_case", do_plots=False)
    # tr=0 reads the envelope rather than the first repetition, so the
    # repetitions are read from 1 upwards.
    played = [
        (index, written_out.calculate_pns(IRNICH, tr=index, do_plots=False)[1])
        for index in range(1, structure.num_trs)
    ]
    response_times = np.arange(envelope.size) * dt * 1e3

    fig, (top, middle, bottom) = plt.subplots(
        3, 1, figsize=(9.5, 9.0), gridspec_kw={"height_ratios": [1.0, 1.1, 0.9]}
    )
    palette = (BLUE, ORANGE, AQUA, AMBER)

    spread = np.max([np.abs(gradients[0]) for _, gradients in arms], axis=0)
    (live,) = np.nonzero(spread > 0.02 * spread.max())
    readout = slice(live[0], live[-1] + 1)
    for slot, (_index, gradients) in enumerate(arms):
        top.plot(
            times[readout],
            1e-3 * gradients[0][readout],
            color=palette[slot % len(palette)],
            linewidth=1.1,
            label=f"arm {slot}",
        )
    top.plot(
        times[readout],
        1e-3 * window[0][readout],
        color=INK,
        linewidth=1.8,
        linestyle="--",
        label="canonical TR",
    )
    top.set_ylabel("$G_x$ [kHz/m]")
    top.set_xlabel("time within the canonical TR [ms]")
    reach = np.abs([gradients[0][readout] for _, gradients in arms]).max() * 1e-3
    top.set_ylim(-1.15 * reach, 1.75 * reach)
    top.legend(loc="upper center", frameon=False, fontsize=9, ncol=5)
    top.set_title(
        f"{len(arms)} interleaves at one block position, and the window built over each",
        fontsize=10,
        loc="left",
    )

    for slot, (_, trace) in enumerate(played):
        middle.plot(
            response_times,
            100.0 * trace,
            color=BLUE,
            linewidth=1.0,
            label=f"the {len(played)} repetitions, one trace each"
            if slot == 0
            else None,
        )
    middle.plot(
        response_times,
        100.0 * envelope,
        color=INK,
        linewidth=1.7,
        label="worst canonical TR, arms written out",
    )
    middle.plot(
        response_times,
        100.0 * turned,
        color=MUTED,
        linewidth=1.7,
        linestyle=":",
        label="worst canonical TR, arms turned by a rotation",
    )
    _mark_period(middle, period)
    middle.axhline(100.0, color=RED, linewidth=1.0)
    middle.set_ylabel("combined stimulation [% of threshold]")
    middle.set_ylim(-4.0, 100.0 * envelope.max() * 1.45)
    middle.legend(loc="upper left", frameon=False, fontsize=9, ncol=2)

    over = 0.0
    for _, trace in played:
        difference = 100.0 * (trace - envelope)
        over = max(over, float(difference.max()))
        bottom.plot(response_times, difference, color=BLUE, linewidth=1.0)
    bottom.axhline(0.0, color=INK, linewidth=1.0)
    _mark_period(bottom, period, label=False)
    bottom.set_ylabel("repetition minus canonical TR [%]")
    bottom.set_xlabel("time within the canonical TR [ms]")
    bottom.set_title(
        "each repetition against the window it is judged by, sample by sample:"
        " worst excursion above it"
        f" {over:.1e} % of threshold",
        fontsize=10,
        loc="left",
    )

    for axis in (top, middle, bottom):
        _style(axis)
    for axis in (middle, bottom):
        axis.set_xlim(0.0, response_times[-1])

    print(
        f"  {len(arms)} interleaves over {structure.num_trs} repetitions\n"
        f"  peak: canonical TR {100 * envelope.max():.4f} % written out, "
        f"{100 * turned.max():.4f} % rotation-encoded; "
        f"worst repetition {100 * max(t.max() for _, t in played):.4f} %\n"
        f"  repetition minus canonical TR: at most {over:+.2e} % above, "
        f"{min(100.0 * (t - envelope).min() for _, t in played):.3f} % below"
    )

    fig.suptitle(
        "Spiral gradient echo: the repetitions play different waveforms,"
        " so each gets a window and the worst is judged",
        fontsize=11,
    )
    return _save(fig, "multishot_envelope")


def _mark_period(axis, period_ms: float, label: bool = True) -> None:
    """Where the TR ends and the wrapped history begins."""
    axis.axvline(period_ms, color=MUTED, linewidth=0.9, linestyle="--")
    if not label:
        return
    axis.annotate(
        "one TR ends; what follows is the\nnext repetition, wrapped in",
        xy=(period_ms, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(5, -12),
        textcoords="offset points",
        ha="left",
        va="top",
        color=MUTED,
        fontsize=8,
    )


# --- the sweep behind "one window, not the scan" -------------------------

#: The scan lengths the window's independence is shown over. One blipped EPI
#: family, the repetition itself held fixed, only the slice count growing.
SWEEP = ({"n_slices": 1}, {"n_slices": 4}, {"n_slices": 16}, {"n_slices": 48})


def scaling_report() -> None:
    """Chronaxie cost over the scan against chronaxie cost over one window.

    The first two columns share one implementation -- the direct convolution
    the assembly figure is checked against -- so their ratio is the algorithm
    and not two codebases. The third is what the scanner actually runs.
    """
    import pulserver.app as app

    print(
        "| Blocks | TRs | Over the timeline | Over the canonical TR | Pulserver |\n"
        "|---:|---:|---:|---:|---:|"
    )
    peaks = []
    for case in SWEEP:
        seq = app.epi2D_sequence(plot=False, write_seq=False, n_x=96, n_y=96, **case)
        axes, _, dt = canonical_window(seq, "worst_case")
        gamma = float(seq.system.gamma)
        kernel = irnich_kernel(dt, IRNICH)

        scan = whole_scan(seq, dt)
        over_scan, timeline = fastest(
            convolved_directly, scan, dt, gamma, kernel, wrap=False
        )
        over_window, window = fastest(convolved_directly, axes, dt, gamma, kernel)
        shipped, answer = fastest(
            seq.calculate_pns, IRNICH, do_plots=False, tr="worst_case"
        )
        peaks.append((norm(timeline).max(), norm(window).max(), answer[1].max()))
        print(
            f"| {seq.num_blocks} | {seq.num_trs} | {over_scan * 1e3:.0f} ms "
            f"| {over_window * 1e3:.0f} ms | {shipped * 1e3:.1f} ms |"
        )
    for scan_peak, window_peak, shipped_peak in peaks:
        print(
            f"  peaks: timeline {scan_peak:.4f}, canonical TR {window_peak:.4f}, "
            f"Pulserver {shipped_peak:.4f}"
        )


# --- shared -------------------------------------------------------------


def _style(axis) -> None:
    axis.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=MUTED, labelsize=9)
    axis.yaxis.label.set_color(MUTED)
    axis.xaxis.label.set_color(MUTED)


def _save(fig, stem: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{stem}.png"
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=130, facecolor="white")
    plt.close("all")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only", choices=("all", "scaling", "assembly", "envelope"), default="all"
    )
    args = parser.parse_args()

    if args.only in ("all", "scaling"):
        scaling_report()
    if args.only in ("all", "assembly"):
        print(f"assembly -> {assembly_figure().relative_to(OUT_DIR.parents[2])}")
    if args.only in ("all", "envelope"):
        print(f"envelope -> {envelope_figure().relative_to(OUT_DIR.parents[2])}")


if __name__ == "__main__":
    main()
