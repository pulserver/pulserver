"""Runs of a playout's repetitions that differ only in their phase offsets and phase encodings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._scanner import _area

#: Repetitions below which a run is played block by block: the four plays
#: that give each isochromat's map over one repetition cost about as much.
LEAST = 8

# Most blocks one repetition of a run takes.
_LONGEST = 64

# Largest difference, in rad, between the increases of the phase offsets of
# the pulses, or of the ADC events, of one repetition.
_PHASE = 1e-5

# Largest area a repetition's changes of gradient may leave along an axis by
# one of its pulses, relative to the largest area one of them plays: the single
# precision of the cache's amplitudes.
_EXACT = 1e-6

# The same where samples are played to a tolerance, which also takes as zero an
# area they leave over the repetition up to it: the six significant digits a
# Pulseq file keeps of an amplitude round it by up to 5e-6 of itself, and an
# encoding and its rewinder, each less the first repetition's, are four such
# amplitudes.
_ROUNDED = 2e-5


@dataclass(frozen=True, eq=False)
class Run:
    """Repetitions of the blocks from ``first`` on, ``size`` blocks each.

    Repetition ``n`` plays the first repetition's blocks with every pulse's
    phase offset larger by ``phases[n]`` and every ADC event's by
    ``adc_phases[n]``, in rad, and with gradients that differ from the first
    repetition's by waveforms whose area at the first sample of each of its
    ADC windows is ``areas[n]`` and which hold ``readouts[n]`` through them,
    ``(windows, 3)`` in 1/m and Hz/m, and which leave ``nets[n]`` over it, in
    1/m, all along the physical axes: phase encodings, and readouts turned
    with their prephasers.
    """

    first: int
    size: int
    count: int
    phases: np.ndarray
    adc_phases: np.ndarray
    areas: np.ndarray
    readouts: np.ndarray
    nets: np.ndarray

    @property
    def stop(self) -> int:
        """The block after the run's last."""
        return self.first + self.count * self.size


def runs(
    played: dict,
    turn: np.ndarray | None,
    least: int = LEAST,
    windows: int | None = None,
    rounded: bool = False,
) -> list[Run]:
    """Return the runs of at least ``least`` repetitions among the blocks :func:`pulserver.ir.playout` records.

    A repetition is the fewest consecutive blocks, up to 64, whose events and
    registers the next blocks repeat but for their phase offsets and their
    gradients' amplitudes and waves, and which hold at most ``windows`` ADC
    windows where it is given. From each block, the run of the shortest
    repetition that fits is taken; a block that starts none plays alone. The
    gradients' changes are turned by ``turn`` where a block's gradients are,
    and ``rounded`` takes the areas they leave by a pulse and over a
    repetition as zero up to the precision a Pulseq file keeps of their
    amplitudes rather than the cache's.
    """
    keys = _keys(played)
    periods = range(1, min(_LONGEST, keys.size // least) + 1)
    stops = {period: _disagreeing(keys, period) for period in periods}
    startable = np.zeros((len(periods), keys.size), dtype=bool)
    for period in periods:
        startable[period - 1] = _agreeing(stops[period], keys.size) >= period * (
            least - 1
        )
    starts = np.flatnonzero(startable.any(axis=0))
    found = []
    block = 0
    while (at := int(np.searchsorted(starts, block))) < starts.size:
        block = int(starts[at])
        run = None
        for period in np.flatnonzero(startable[:, block]) + 1:
            if (
                windows is not None
                and played["adc"][block : block + period].sum() > windows
            ):
                continue
            total = (period + _agreeing(stops[period], keys.size, block)) // period
            # A run is read over spans of repetitions that double from the
            # least until one ends inside it.
            span = least
            run = _run(played, block, int(period), span, turn, rounded)
            while run is not None and run.count == span < total:
                span = min(2 * span, total)
                run = _run(played, block, int(period), span, turn, rounded)
            if run is not None and run.count >= least:
                break
            run = None
        if run is None:
            block += 1
            continue
        found.append(run)
        block = run.stop
    return found


def _keys(played: dict) -> np.ndarray:
    """Per block, a number standing for what a repetition of it repeats: its events, and their registers but the phase offsets and the gradients' amplitudes and waves."""
    columns = [
        played["subsequence"],
        played["segment"],
        played["position"],
        played["duration_us"],
        played["rotate"],
        played["rf_amp_hz"],
        played["rf_freq_hz"],
        played["rf_shim"],
        played["adc"],
        played["adc_freq_hz"],
        np.asarray(played["wave"]) >= 0,
    ]
    table = np.column_stack([np.asarray(column, dtype=float) for column in columns])
    return np.unique(table, axis=0, return_inverse=True)[1].ravel()


def _disagreeing(keys: np.ndarray, period: int) -> np.ndarray:
    """Return the blocks whose key differs from the key ``period`` blocks on, or that have none, ascending."""
    same = np.zeros(keys.size, dtype=bool)
    same[: keys.size - period] = keys[period:] == keys[:-period]
    return np.flatnonzero(~same)


def _agreeing(
    stops: np.ndarray, size: int, at: int | np.ndarray | None = None
) -> np.ndarray:
    """Return how many blocks from each of ``at``, every block by default, agree with the block ``period`` on, given where :func:`_disagreeing` stops."""
    at = np.arange(size) if at is None else at
    return stops[np.searchsorted(stops, at)] - at


def _run(
    played: dict,
    first: int,
    size: int,
    count: int,
    turn: np.ndarray | None,
    rounded: bool,
) -> Run | None:
    """Return the longest run of the repetitions of ``size`` blocks from ``first`` on that it can play; None where it plays none."""
    blocks = first + size * np.arange(count)[:, None] + np.arange(size)
    template = blocks[0]
    pulses = np.flatnonzero(
        (played["rf_amp_hz"][template] != 0.0)
        & (np.diff(played["rf_span"][template], axis=1)[:, 0] > 0)
    )
    windows = np.flatnonzero(played["adc"][template].astype(bool))
    if not all(_readable(played, block) for block in template[windows]):
        return None
    phases, pulsed = _increments(played["rf_phase_rad"][blocks[:, pulses]])
    adc_phases, read = _increments(played["adc_phase_rad"][blocks[:, windows]])
    areas, readouts, nets, encoded = _encoding(
        played, blocks, pulses, windows, turn, rounded
    )
    kept = min(pulsed, read, encoded)
    return Run(
        first=first,
        size=size,
        count=kept,
        phases=phases[:kept],
        adc_phases=adc_phases[:kept],
        areas=areas[:kept],
        readouts=readouts[:kept],
        nets=nets[:kept],
    )


def _increments(offsets: np.ndarray) -> tuple[np.ndarray, int]:
    """Return each repetition's increase of ``offsets``, ``(repetitions, events)``, over the first's, and how many repetitions from the first increase every event's alike."""
    if offsets.shape[1] == 0:
        return np.zeros(offsets.shape[0]), offsets.shape[0]
    increase = offsets - offsets[0]
    spread = np.remainder(increase - increase[:, :1] + np.pi, 2.0 * np.pi) - np.pi
    alike = np.all(np.abs(spread) <= _PHASE, axis=1)
    return increase[:, 0], _leading(alike)


def _leading(flags: np.ndarray) -> int:
    """Return how many of ``flags`` hold from the first on."""
    return int(np.argmin(flags)) if not flags.all() else int(flags.size)


def _readable(played: dict, block: int) -> bool:
    """Whether the block's ADC window lies under a gradient held throughout it, on one side of any pulse in the block."""
    pulse, (start, stop) = _spans(played, block)
    if pulse is not None and pulse[1] > start and pulse[0] < stop:
        return False
    return all(
        np.ptp(_through(times, values, start, stop)) == 0.0
        for times, values in _corners(played, block)
    )


def _spans(
    played: dict, block: int
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """Return the block's pulse and ADC window, each as its start and stop in µs from the block's start; None where it plays none."""
    pulse = None
    rf_start, rf_stop = played["rf_span"][block]
    if rf_stop > rf_start and played["rf_amp_hz"][block] != 0.0:
        times = played["rf_time_us"][rf_start:rf_stop]
        pulse = (float(times.min()), float(times.max()))
    window = None
    if played["adc"][block]:
        start = float(played["adc_delay_us"][block])
        duration = 1e-3 * float(played["adc_dwell_ns"][block])
        window = (start, start + duration * int(played["adc_samples"][block]))
    return pulse, window


def _corners(played: dict, block: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return the block's gradient corners per logical axis: times in µs from its start over Hz/m."""
    span = played["gradient_span"][block]
    return [
        (
            played["gradient_time_us"][slice(*span[axis])].astype(float),
            played["gradient_waveform_hz_per_m"][slice(*span[axis])].astype(float),
        )
        for axis in range(3)
    ]


def _through(
    times: np.ndarray, values: np.ndarray, start: float, stop: float
) -> np.ndarray:
    """Return the piecewise-linear waveform's values from ``start`` to ``stop``: at both ends and at its corners between.

    ``values`` may hold several waveforms through the same times,
    ``(..., corners)``.
    """
    if times.size == 0:
        return np.zeros((*values.shape[:-1], 1))
    inside = values[..., (times > start) & (times < stop)]
    ends = _at(times, values, np.array([start, stop]))
    return np.concatenate([ends, inside], axis=-1)


def _at(times: np.ndarray, values: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Return the piecewise-linear waveform's values at each of ``at``, as :func:`numpy.interp` finds them, zero outside its corners; ``(..., at)``."""
    last = times.size - 1
    j = np.searchsorted(times, at, side="right") - 1
    low = np.clip(j, 0, max(last - 1, 0))
    high = np.minimum(low + 1, last)
    width = times[high] - times[low]
    slope = (values[..., high] - values[..., low]) / np.where(width > 0.0, width, 1.0)
    value = slope * (at - times[low]) + values[..., low]
    value = np.where(j == last, values[..., last:], value)
    return np.where((at < times[0]) | (at > times[-1]), 0.0, value)


def _encoding(
    played: dict,
    blocks: np.ndarray,
    pulses: np.ndarray,
    windows: np.ndarray,
    turn: np.ndarray | None,
    rounded: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return how the gradients of the repetitions of ``blocks``, ``(repetitions, size)``, differ from the first repetition's, and how many repetitions from the first it plays.

    Each repetition plays the first repetition's gradients plus, where they
    vary, waveforms zero during their block's pulse and held through their
    block's ADC window: phase encodings, and readouts turned with their
    prephasers (:func:`_change`). The waveforms of a repetition leave no area
    by the start of any of its pulses, ``pulses`` among its blocks, to
    ``_EXACT``, or with ``rounded`` ``_ROUNDED``, of the largest area one of
    its gradients plays; with ``rounded``, an area they leave over it within
    that is taken as zero. Returns their area at the first sample of
    each of its ADC windows, ``windows`` among its blocks, and the gradient
    they hold through each, ``(repetitions, windows, 3)`` in 1/m and Hz/m; the
    area they leave over it, ``(repetitions, 3)`` in 1/m, all along the
    physical axes; and the repetitions from the first that fit.
    """
    template = blocks[0]
    count = blocks.shape[0]
    # Each window's first sample, each pulse's start and the repetition's end,
    # as a block of the repetition and a time into it, in µs.
    at_blocks = np.concatenate(
        [template[windows], template[pulses], [blocks.max() + 1]]
    )
    at_times = np.array(
        [
            *(_first_sample(played, block) for block in template[windows]),
            *(_pulse_start(played, block) for block in template[pulses]),
            0.0,
        ]
    )
    areas = np.zeros((count, windows.size, 3))
    readouts = np.zeros((count, windows.size, 3))
    # The area left by the start of each pulse, and over the repetition.
    left = np.zeros((count, pulses.size + 1, 3))
    largest = np.zeros(count)
    kept = count
    for position, axis in zip(*np.nonzero(_varying(played, blocks)), strict=True):
        lobe = int(template[position])
        times, change, extent, comparable = _change(played, blocks[:, position], axis)
        quiet, held = _fitting(played, lobe, times, change)
        kept = min(kept, comparable, _leading(quiet))
        direction = _direction(played, lobe, axis, turn)
        reached = _reached(lobe, times, change, at_blocks, at_times)
        areas += reached[:, : windows.size, None] * direction
        left += reached[:, windows.size :, None] * direction
        readouts[:, template[windows] == lobe] += held[:, None, None] * direction
        largest = np.maximum(largest, extent)
    within = (_ROUNDED if rounded else _EXACT) * largest
    balanced = np.all(np.abs(left[:, :-1]).max(axis=2) <= within[:, None], axis=1)
    nets = left[:, -1]
    if rounded:
        nets = np.where(
            np.abs(nets).max(axis=1, keepdims=True) <= within[:, None], 0.0, nets
        )
    return areas, readouts, nets, min(kept, _leading(balanced))


def _varying(played: dict, blocks: np.ndarray) -> np.ndarray:
    """Return, per block of the repetition and logical axis, ``(size, 3)``, whether its gradient varies across the repetitions: in amplitude, or, where a block plays a wave, in its wave or the wave's amplitudes too."""
    amplitudes = played["gradient_hz_per_m"][blocks]
    varying = np.any(amplitudes != amplitudes[0], axis=0)
    waves = played["wave"][blocks]
    scaled = played["wave_amp_hz_per_m"][blocks]
    turned = np.any(waves != waves[0], axis=0)[:, None] | np.any(
        scaled != scaled[0], axis=0
    )
    return np.where(np.any(waves >= 0, axis=0)[:, None], varying | turned, varying)


def _change(
    played: dict, column: np.ndarray, axis: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return how the gradient along logical ``axis`` of the repetitions' blocks ``column`` differs from the first repetition's.

    Returns its corners' times, in µs, and each repetition's values less the
    first's, ``(repetitions, corners)`` in Hz/m; the largest area each
    repetition's gradient plays, in 1/m; and how many repetitions from the
    first play corners at those times. A gradient of the IR's amplitude
    registers is its block's in the repetition of largest amplitude, scaled;
    a wave's corners are those each repetition plays, none standing for a
    wave of zero.
    """
    if np.any(played["wave"][column] >= 0):
        return _wave_change(played, column, axis)
    count = column.size
    amplitude = played["gradient_hz_per_m"][column, axis].astype(float)
    strongest = int(np.argmax(np.abs(amplitude)))
    times, values = _corners(played, column[strongest])[axis]
    if times.size == 0:
        return times, np.zeros((count, 0)), np.zeros(count), count
    shape = values / amplitude[strongest]
    lobe = abs(1e-6 * float(_area(times, shape, times[-1:])[0]))
    extent = np.maximum(np.abs(amplitude), abs(amplitude[0])) * lobe
    return times, np.outer(amplitude - amplitude[0], shape), extent, count


def _wave_change(
    played: dict, column: np.ndarray, axis: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """:func:`_change` of a wave, from the corners each repetition plays."""
    count = column.size
    spans = played["gradient_span"][column, axis]
    lengths = spans[:, 1] - spans[:, 0]
    size = int(lengths.max())
    if size == 0:
        return np.zeros(0), np.zeros((count, 0)), np.zeros(count), count
    index = np.minimum(
        spans[:, :1] + np.arange(size), played["gradient_time_us"].size - 1
    )
    times = played["gradient_time_us"][index].astype(float)
    values = played["gradient_waveform_hz_per_m"][index].astype(float)
    reference = times[int(np.argmax(lengths == size))]
    comparable = _leading(
        (lengths == 0) | ((lengths == size) & np.all(times == reference, axis=1))
    )
    values[(lengths == 0) | (np.arange(count) >= comparable)] = 0.0
    extent = np.abs(1e-6 * _area(reference, values, reference)).max(axis=1)
    return reference, values - values[0], np.maximum(extent, extent[0]), comparable


def _fitting(
    played: dict, block: int, times: np.ndarray, change: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return, per repetition, whether ``change``, ``(repetitions, corners)`` through ``times`` in µs, is zero throughout the block's pulse and held through its ADC window, and the value it holds there, in Hz/m."""
    pulse, window = _spans(played, block)
    quiet = np.ones(change.shape[0], dtype=bool)
    held = np.zeros(change.shape[0])
    if pulse is not None:
        quiet &= ~np.any(_through(times, change, *pulse), axis=-1)
    if window is not None:
        through = _through(times, change, *window)
        quiet &= np.all(through == through[:, :1], axis=-1)
        held = through[:, 0]
    return quiet, held


def _reached(
    lobe: int, times: np.ndarray, values: np.ndarray, blocks: np.ndarray, at: np.ndarray
) -> np.ndarray:
    """Return the area, in 1/m, the waveforms ``(..., corners)`` of the repetition's block ``lobe`` have played by each of ``at``, in µs, into the repetition's block of the same index of ``blocks``: ``(..., at)``."""
    ends = np.where(blocks == lobe, at, times[-1] if times.size else 0.0)
    return np.where(blocks < lobe, 0.0, 1e-6 * _area(times, values, ends))


def _first_sample(played: dict, block: int) -> float:
    """Return the time, in µs from the block's start, of its ADC window's first sample."""
    return float(played["adc_delay_us"][block]) + 5e-4 * float(
        played["adc_dwell_ns"][block]
    )


def _pulse_start(played: dict, block: int) -> float:
    """Return the time, in µs from the block's start, of its pulse's first sample."""
    rf_start, rf_stop = played["rf_span"][block]
    return float(played["rf_time_us"][rf_start:rf_stop].min())


def _direction(
    played: dict, block: int, axis: int, turn: np.ndarray | None
) -> np.ndarray:
    """Return the physical direction of the block's logical axis ``axis``."""
    if turn is None or not played["rotate"][block]:
        return np.eye(3)[axis]
    return np.asarray(turn, dtype=float)[:, axis]
