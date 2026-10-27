"""The receive-gain calibration a prescan plays, generated from the chain."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pypulseqpp as pp

#: Thickness a non-selective excitation is taken to excite, in m: a head
#: coil's coverage, so it outranks any slab.
COVERAGE = 0.3
#: Net gradient area under this fraction of the absolute area is balanced.
BALANCE_TOLERANCE = 1e-2
#: Decimals, in m, signals are compared at: candidates equal to rounding
#: tie, and the first in play order is taken.
SIGNAL_DIGITS = 9
#: Plays of the calibration whose readouts set the gain.
FRAMES = 2
#: Rest before each play of the calibration, s, so it reads relaxed
#: magnetization.
REST = 3.0

#: Definition that marks the generated subsequence in the cache.
PRESCAN_ONLY = "PrescanOnly"

_RF, _GX, _GY, _GZ, _ADC = range(5)
_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Span:
    """Blocks ``first`` (0-based, an RF block) up to ``stop`` of file ``file``."""

    file: int
    first: int
    stop: int
    readouts: int = 0
    signal: float = 0.0
    balanced: bool = False


class _File:
    """What the generator reads of one file of the chain, each table once."""

    def __init__(self, sequence: pp.Sequence):
        self.sequence = sequence
        self.tables = sequence.libraries()
        self.blocks = np.asarray(self.tables.blocks)
        self.rf = self.blocks[:, _RF]
        self.reads = self.blocks[:, _ADC] > 0
        self.shims = np.asarray(sequence.block_shims())
        self.flips = np.abs(np.asarray(sequence.rf_flip_angles(), dtype=float))
        self.channels = np.asarray(sequence.rf_channels(), dtype=int)
        self.durations = np.asarray(self.tables.block_durations, dtype=float)
        self._signals: dict[int, np.ndarray] = {}
        self._bandwidths: dict[tuple[int, int, int], float] = {}

    def bandwidth(self, rf_id: int) -> float:
        """Return the bandwidth, Hz, of the RF of ``rf_id``.

        It depends on the magnitude, phase and time shapes alone, so an
        RF-spoiled train, whose every pulse is an RF definition of its own,
        measures it once.
        """
        row = self.tables.rf[rf_id - 1]
        shape = (int(row[1]), int(row[2]), int(row[3]))
        if shape not in self._bandwidths:
            block = int(np.flatnonzero(self.rf == rf_id)[0]) + 1
            rf = self.sequence.get_block(block).rf
            self._bandwidths[shape] = float(pp.calc_rf_bandwidth(rf))
        return self._bandwidths[shape]

    def signal(self, rf_id: int) -> np.ndarray:
        """Return the RF waveform of ``rf_id``, Hz, one row per channel."""
        if rf_id not in self._signals:
            block = int(np.flatnonzero(self.rf == rf_id)[0]) + 1
            signal = np.asarray(self.sequence.get_block(block).rf.signal)
            self._signals[rf_id] = signal.reshape(int(self.channels[rf_id - 1]), -1)
        return self._signals[rf_id]

    def drives(self, block: int, channels: int) -> np.ndarray:
        """Per channel, the largest drive, Hz, the RF of 0-based ``block`` plays at."""
        peaks = np.abs(self.signal(int(self.rf[block]))).max(axis=1)
        shim = int(self.shims[block]) if self.shims.size > block else 0
        if shim > 0:
            pairs = np.asarray(self.tables.rf_shims[shim - 1], dtype=float)
            weights = np.abs(pairs[0::2] + 1j * pairs[1::2])
            return peaks.max() * weights
        if peaks.size == 1:
            return np.full(channels, peaks[0])
        return peaks


def receive_calibration_sequence(
    chain: Sequence[pp.Sequence], system: pp.Opts
) -> pp.Sequence | None:
    """Return the prescan-only subsequence a prescan of the chain plays, or None.

    It opens, in a parallel-transmit chain, with a tour: for every transmit
    channel, the RF block that drives it hardest anywhere in the chain (a
    single waveform with no shim drives every channel), with its blocks up to
    the next RF pulse, its readouts under ``NOPRESCAN``, then a rest until it
    has lasted the shortest interval at which its file plays that pulse.

    Where exactly one file reads out and it repeats with the same RF
    amplitudes, the prescan calibrates on that file from its start after the
    tour, and the subsequence is the tour alone, or None without pTx.
    Otherwise no repetition stands for the scan, and the tour is followed by
    the calibration: the excitation scoring highest on the sine of its flip
    angle, capped at 90 degrees, times the thickness it excites, the RF
    bandwidth over the slice-select gradient, :data:`COVERAGE` without one; a
    parallel-transmit pulse's flip is taken relative to the 1/sqrt(N)
    reference drive. Its blocks up to the next excitation are played
    :data:`FRAMES` times, each after a rest of :data:`REST`, at their own
    amplitudes, every gradient that varies across the file's repetitions at
    zero. Read from relaxed magnetization they carry at least the signal of
    every readout of the scan the excitation outscores: a steady state,
    spoiled or balanced, never exceeds the equilibrium magnetization times the
    sine of its flip. A winner with balanced gradients is logged.

    Every block is one the scan plays, at the scan's amplitude, labels left
    out, so the subsequence drives no channel harder than the scan. It
    carries ``NumGainCalibrationReadouts`` and the :data:`PRESCAN_ONLY`
    definition, which the conversion writes into the cache.

    Parameters
    ----------
    chain : Sequence[pypulseqpp.Sequence]
        The files of the chain, in play order, in the logical frame.
    system : pypulseqpp.Opts
        Limits and rasters of the subsequence.

    Returns
    -------
    pypulseqpp.Sequence or None
    """
    files = [_File(sequence) for sequence in chain]
    channels = max((int(f.channels.max()) for f in files if f.channels.size), default=1)
    if any(f.shims.size and (f.shims > 0).any() for f in files):
        channels = max(
            channels,
            max(len(pairs) // 2 for f in files for pairs in f.tables.rf_shims),
        )
    tour = _tour(files, channels) if channels > 1 else []
    reading = [i for i, f in enumerate(files) if f.reads.any()]
    calibration = None
    if reading and not (len(reading) == 1 and _repeats_alike(files[reading[0]])):
        # Equal signals, to rounding, go to the first in play order.
        calibration = max(
            (span for i in reading for span in _spans(files[i], i, channels)),
            key=lambda span: round(span.signal, SIGNAL_DIGITS),
            default=None,
        )
    if not tour and calibration is None:
        return None
    if calibration is not None and calibration.balanced:
        _log.info(
            "receive calibration: file %d block %d has balanced gradients up to the next excitation",
            calibration.file,
            calibration.first + 1,
        )
    return _build(files, tour, calibration, system)


def _repeats_alike(f: _File) -> bool:
    size, _ = f.sequence.repetition()
    if size <= 0 or size >= f.blocks.shape[0]:
        return False
    for position in range(size):
        ids = f.rf[position::size]
        ids = ids[ids > 0]
        if ids.size and np.ptp(f.flips[ids - 1]) > 1e-6 * max(
            f.flips[ids - 1].max(), 1e-12
        ):
            return False
    return True


def _next_rf(f: _File) -> np.ndarray:
    """Per RF block, the block of the next RF pulse, or the end."""
    pulses = np.flatnonzero(f.rf > 0)
    return np.append(pulses[1:], f.blocks.shape[0])


def _tour(files: list[_File], channels: int) -> list[tuple[_Span, float]]:
    """Per channel the hardest-driving RF block, each once, with the period it plays at."""
    best: list[tuple[float, int, int]] = [(-1.0, -1, -1)] * channels
    for i, f in enumerate(files):
        seen: set[tuple[int, int]] = set()
        for block in np.flatnonzero(f.rf > 0):
            key = (int(f.rf[block]), int(f.shims[block]) if f.shims.size > block else 0)
            if key in seen:
                continue
            seen.add(key)
            for k, drive in enumerate(f.drives(int(block), channels)[:channels]):
                if drive > best[k][0]:
                    best[k] = (float(drive), i, int(block))
    visits = []
    for i, block in sorted({(b[1], b[2]) for b in best if b[1] >= 0}):
        f = files[i]
        stop = int(_next_rf(f)[np.searchsorted(np.flatnonzero(f.rf > 0), block)])
        visits.append((_Span(i, block, stop), _interval(f, block)))
    return visits


def _interval(f: _File, block: int) -> float:
    """Shortest time, s, between two plays of the RF pulse of ``block`` in its file."""
    starts = np.concatenate([[0.0], np.cumsum(f.durations)[:-1]])
    same = np.flatnonzero(f.rf == f.rf[block])
    return float(np.diff(starts[same]).min()) if same.size > 1 else 0.0


def _areas(f: _File, ids: np.ndarray) -> tuple[np.ndarray, bool]:
    """Area of each gradient id, Hz/m s; False where one is not a trapezoid."""
    trapezoids = np.asarray(f.tables.trapezoids, dtype=float)
    index = {int(t): k for k, t in enumerate(np.asarray(f.tables.trapezoid_ids))}
    areas = np.zeros(ids.shape)
    known = True
    for k, gid in np.ndenumerate(ids):
        if gid <= 0:
            continue
        row = index.get(int(gid))
        if row is None:
            known = False
            continue
        amp, rise, flat, fall, _ = trapezoids[row]
        areas[k] = amp * (0.5 * rise + flat + 0.5 * fall)
    return areas, known


def _spans(f: _File, file: int, channels: int):
    excitations = [
        int(i)
        for i in np.flatnonzero(f.rf > 0)
        if f.tables.rf_use[int(f.rf[i]) - 1] in ("excitation", "undefined", "")
    ]
    if not excitations:
        return
    gradients = f.sequence.rf_gradients()
    slice_gradient = {
        int(b) - 1: float(np.linalg.norm(g))
        for b, g in zip(gradients.block, gradients.gradient, strict=True)
    }
    for first, stop in zip(
        excitations, [*excitations[1:], f.blocks.shape[0]], strict=True
    ):
        readouts = int(f.reads[first:stop].sum())
        if readouts == 0:
            continue
        rf_id = int(f.rf[first])
        flip = float(f.flips[rf_id - 1])
        if channels > 1:
            drives = f.drives(first, channels)
            flip *= float(drives.sum() / (drives.max() * math.sqrt(channels)))
        bandwidth = f.bandwidth(rf_id)
        g = slice_gradient.get(first, 0.0)
        thickness = bandwidth / g if g > 0 and bandwidth > 0 else COVERAGE
        signal = math.sin(math.radians(min(flip, 90.0))) * min(thickness, COVERAGE)
        areas, known = _areas(f, f.blocks[first:stop, _GX : _GZ + 1])
        net, total = np.abs(areas.sum(axis=0)), np.abs(areas).sum(axis=0)
        balanced = bool(
            known and total.any() and (net <= BALANCE_TOLERANCE * total).all()
        )
        yield _Span(file, first, stop, readouts, signal, balanced)


def _varying(f: _File, first: int, stop: int) -> np.ndarray:
    """Per block of the span and gradient axis, whether its event varies across repetitions."""
    size, _ = f.sequence.repetition()
    varying = np.zeros((stop - first, 3), dtype=bool)
    if size <= 0 or size >= f.blocks.shape[0]:
        return varying
    for k, index in enumerate(range(first, stop)):
        column = f.blocks[index % size :: size, _GX : _GZ + 1]
        varying[k] = (column != column[0]).any(axis=0)
    return varying


def _events(f: _File, span: _Span, zeroed: bool) -> list[list]:
    """Return the events of each block of the span, labels left out."""
    varying = _varying(f, span.first, span.stop) if zeroed else None
    plays = []
    for k, index in enumerate(range(span.first, span.stop)):
        block = f.sequence.get_block(index + 1)
        events = []
        if block.rf is not None:
            events.append(block.rf)
        for axis, name in enumerate(("gx", "gy", "gz")):
            gradient = getattr(block, name)
            if gradient is not None:
                zero = varying is not None and varying[k, axis]
                events.append(pp.scale_grad(gradient, 0.0) if zero else gradient)
        for name in ("adc", "rf_shim", "rotation"):
            if getattr(block, name, None) is not None:
                events.append(getattr(block, name))
        if not events or block.block_duration > max(
            pp.calc_duration(e) for e in events
        ):
            events.append(pp.make_delay(block.block_duration))
        plays.append(events)
    return plays


def _build(files, tour, calibration, system) -> pp.Sequence:
    sequence = pp.Sequence(system)
    if tour:
        sequence.add_block(
            pp.make_delay(system.block_duration_raster * 25),
            pp.make_label(label="NOPRESCAN", type="SET", value=1),
        )
        for span, period in tour:
            plays = _events(files[span.file], span, zeroed=True)
            for events in plays:
                sequence.add_block(*events)
            played = float(files[span.file].durations[span.first : span.stop].sum())
            if period > played:
                sequence.add_block(pp.make_delay(period - played))
    if calibration is not None:
        plays = _events(files[calibration.file], calibration, zeroed=True)
        for frame in range(FRAMES):
            if frame == 0 and tour:
                sequence.add_block(
                    pp.make_delay(REST),
                    pp.make_label(label="NOPRESCAN", type="SET", value=0),
                )
            else:
                sequence.add_block(pp.make_delay(REST))
            for events in plays:
                sequence.add_block(*events)
        sequence.set_definition(
            "NumGainCalibrationReadouts", FRAMES * calibration.readouts
        )
    sequence.set_definition(PRESCAN_ONLY, 1)
    sequence.set_definition("Name", "prescan_calibration")
    return sequence
