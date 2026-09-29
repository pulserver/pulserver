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

# Largest area a repetition's phase encodings may leave along an axis by one of
# its pulses or over it, relative to the largest area one of them plays: the
# single precision of the cache's amplitudes.
_EXACT = 1e-6

# The same where samples are played to a tolerance: the six significant digits
# a Pulseq file keeps of an amplitude round it by up to 5e-6 of itself, and an
# encoding and its rewinder, each less the first repetition's, are four such
# amplitudes.
_ROUNDED = 2e-5


@dataclass(frozen=True, eq=False)
class Run:
    """Repetitions of the blocks from ``first`` on, ``size`` blocks each.

    Repetition ``n`` plays the first repetition's blocks with every pulse's
    phase offset larger by ``phases[n]`` and every ADC event's by
    ``adc_phases[n]``, in rad, and with phase encodings whose area at the
    first sample of each of its ADC windows exceeds the first repetition's by
    ``areas[n]``, ``(windows, 3)`` in 1/m along the physical axes.
    """

    first: int
    size: int
    count: int
    phases: np.ndarray
    adc_phases: np.ndarray
    areas: np.ndarray

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
    registers the next blocks repeat but for their phase offsets and the
    amplitudes of the gradients the IR marks as varying, and which hold at
    most ``windows`` ADC windows where it is given. From each block, the run
    of the shortest repetition that fits is taken; a block that starts none
    plays alone. The phase encodings are turned by ``turn`` where a block's
    gradients are, and ``rounded`` takes the area they leave as zero up to
    the precision a Pulseq file keeps of their amplitudes rather than the
    cache's.
    """
    net = _ROUNDED if rounded else _EXACT
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
            run = _run(played, block, int(period), span, turn, net)
            while run is not None and run.count == span < total:
                span = min(2 * span, total)
                run = _run(played, block, int(period), span, turn, net)
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
    """Per block, a number standing for what a repetition of it repeats: its events, and their registers but the phase offsets and the gradients' amplitudes."""
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
        played["wave"],
        *np.asarray(played["wave_amp_hz_per_m"]).T,
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
    net: float,
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
    areas, encoded = _encoding(played, blocks, pulses, windows, turn, net)
    kept = min(pulsed, read, encoded)
    return Run(
        first=first,
        size=size,
        count=kept,
        phases=phases[:kept],
        adc_phases=adc_phases[:kept],
        areas=areas[:kept],
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
    start = float(played["adc_delay_us"][block])
    stop = start + 1e-3 * float(played["adc_dwell_ns"][block]) * int(
        played["adc_samples"][block]
    )
    rf_start, rf_stop = played["rf_span"][block]
    if rf_stop > rf_start and played["rf_amp_hz"][block] != 0.0:
        pulse = played["rf_time_us"][rf_start:rf_stop]
        if pulse.max() > start and pulse.min() < stop:
            return False
    return all(
        np.ptp(_through(times, values, start, stop)) == 0.0
        for times, values in _corners(played, block)
    )


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
    """Return the piecewise-linear waveform's values from ``start`` to ``stop``: at both ends and at its corners between."""
    if times.size == 0:
        return np.zeros(1)
    inside = values[(times > start) & (times < stop)]
    ends = np.interp([start, stop], times, values, left=0.0, right=0.0)
    return np.concatenate([ends, inside])


def _encoding(
    played: dict,
    blocks: np.ndarray,
    pulses: np.ndarray,
    windows: np.ndarray,
    turn: np.ndarray | None,
    net: float,
) -> tuple[np.ndarray, int]:
    """Return the phase encoding of the repetitions of ``blocks``, ``(repetitions, size)``, and how many from the first it plays.

    A gradient whose amplitude varies across the repetitions and is zero
    during its block's pulse and ADC window is a phase encoding, its block's
    gradient in the repetition of largest amplitude scaled; a repetition
    plays it at its amplitude less the first repetition's. The phase
    encodings of a repetition leave no area by the start of any of its
    pulses, ``pulses`` among its blocks, nor over it, to ``net`` of the
    largest one of them plays. Any other gradient keeps the first
    repetition's amplitude. Returns each repetition's area at the first
    sample of each of its ADC windows, ``windows`` among its blocks,
    ``(repetitions, windows, 3)`` in 1/m along the physical axes, and the
    repetitions from the first that fit.
    """
    template = blocks[0]
    count = blocks.shape[0]
    amplitudes = played["gradient_hz_per_m"][blocks].astype(float)
    varying = np.any(amplitudes != amplitudes[0], axis=0)
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
    # The area left by the start of each pulse, and over the repetition.
    left = np.zeros((count, pulses.size + 1, 3))
    largest = np.zeros(count)
    kept = count
    for position, axis in zip(*np.nonzero(varying), strict=True):
        amplitude = amplitudes[:, position, axis]
        strongest = int(np.argmax(np.abs(amplitude)))
        times, values = _corners(played, blocks[strongest, position])[axis]
        shape = values / amplitude[strongest]
        lobe = int(template[position])
        if played["wave"][blocks[strongest, position]] >= 0 or not _quiet(
            played, lobe, times, shape
        ):
            kept = min(kept, _leading(amplitude == amplitude[0]))
            continue
        direction = _direction(played, lobe, axis, turn)
        step = (amplitude - amplitude[0])[:, None, None] * direction
        reached = _reached(lobe, times, shape, at_blocks, at_times)
        areas += step * reached[None, : windows.size, None]
        left += step * reached[None, windows.size :, None]
        largest = np.maximum(
            largest, np.maximum(np.abs(amplitude), abs(amplitude[0])) * abs(reached[-1])
        )
    balanced = np.all(np.abs(left).max(axis=2) <= net * largest[:, None], axis=1)
    return areas, min(kept, _leading(balanced))


def _quiet(played: dict, block: int, times: np.ndarray, shape: np.ndarray) -> bool:
    """Whether the waveform is zero throughout the block's pulse and ADC window."""
    spans = []
    rf_start, rf_stop = played["rf_span"][block]
    if rf_stop > rf_start and played["rf_amp_hz"][block] != 0.0:
        pulse = played["rf_time_us"][rf_start:rf_stop]
        spans.append((float(pulse.min()), float(pulse.max())))
    if played["adc"][block]:
        start = float(played["adc_delay_us"][block])
        spans.append(
            (
                start,
                start
                + 1e-3
                * float(played["adc_dwell_ns"][block])
                * int(played["adc_samples"][block]),
            )
        )
    return all(not np.any(_through(times, shape, start, stop)) for start, stop in spans)


def _reached(
    lobe: int, times: np.ndarray, shape: np.ndarray, blocks: np.ndarray, at: np.ndarray
) -> np.ndarray:
    """Return the area, in 1/m, the waveform of the repetition's block ``lobe`` has played by each of ``at``, in µs, into the repetition's block of the same index of ``blocks``."""
    area = 1e-6 * _area(times, shape, np.where(blocks == lobe, at, times[-1]))
    return np.where(blocks < lobe, 0.0, area)


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
