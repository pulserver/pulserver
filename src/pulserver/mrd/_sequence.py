"""Per-readout facts of a Pulseq sequence, read through pypulseqpp.

Each wrapper takes a ``pypulseqpp.Sequence`` and returns arrays in the
sequence's units; nothing here depends on MRD.
"""

from __future__ import annotations

__all__ = ["ReadoutTable", "SequenceDefinitions", "read_chain"]

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

#: Range of an axis over a readout, relative to its widest axis, at or below
#: which the axis is constant.
_CONSTANT = 1e-6

#: Echo tie tolerance, as a fraction of the k step at the nearest sample.
_ECHO_TIE = 1e-2

#: Samples per chunk when readouts are processed in bulk.
_CHUNK_SAMPLES = 1 << 22

#: ADC samples a range of blocks holds before an excitation may start the next.
_RANGE_SAMPLES = 1 << 17

#: Ranges whose k-space a table keeps.
_KEPT_RANGES = 2

#: RF uses whose pulse centre resets k-space to zero, as pypulseqpp integrates it.
_RESETTING = frozenset({"excitation", "undefined"})

#: RF and ADC columns of ``Sequence.block_events``, and the number of columns.
_RF, _ADC, _COLUMNS = 1, 5, 7


def read_chain(path: Path | str, *, verify: bool = False) -> list[tuple[Path, Any]]:
    """Read a sequence file and every file its ``NextSequence`` definitions name, in play order.

    A ``NextSequence`` name is relative to the directory of the file naming it.

    Parameters
    ----------
    path
        The first file of the chain.
    verify
        Refuse a file whose contents do not match the signature it carries. A
        file carrying none is read either way.

    Returns
    -------
    list of (Path, pypulseqpp.Sequence)

    Raises
    ------
    FileNotFoundError
        If a file of the chain does not exist.
    ValueError
        If the chain names a file it has already played.
    """
    import pypulseqpp as pp

    chain: list[tuple[Path, Any]] = []
    played: set[Path] = set()
    current = Path(path)
    while True:
        if not current.is_file():
            raise FileNotFoundError(f"sequence chain file not found: {current}")
        resolved = current.resolve()
        if resolved in played:
            raise ValueError(f"the NextSequence chain returns to {current}")
        played.add(resolved)
        seq = pp.Sequence()
        seq.read(current, verify=verify)
        chain.append((current, seq))
        following = seq.get_definition("NextSequence")
        if following in ("", None):
            return chain
        current = current.parent / str(following)


@dataclass(frozen=True)
class SequenceDefinitions:
    """Definitions describing what a sequence acquires, in Pulseq units.

    Attributes
    ----------
    matrix, navigator_matrix : tuple of int or None
        ``Matrix`` and ``NavMatrix`` as ``(x, y, z)``; ``None`` when undefined.
    fov, navigator_fov : tuple of float or None
        ``FOV`` and ``NavFOV`` as ``(x, y, z)``, in metres; ``None`` when
        undefined.
    tr, te, ti : tuple of float
        ``TR``, ``TE`` and ``TI`` values, in seconds; empty when undefined.
    flip_angle : tuple of float
        ``FlipAngle`` values, in degrees; empty when undefined.
    centre_line, centre_partition : int or None
        ``kSpaceCenterLine`` and ``kSpaceCenterPartition``: the ``LIN`` and
        ``PAR`` counter at which k-space is sampled at its centre; ``None``
        when undefined.
    """

    matrix: tuple[int, int, int] | None
    fov: tuple[float, float, float] | None
    navigator_matrix: tuple[int, int, int] | None
    navigator_fov: tuple[float, float, float] | None
    tr: tuple[float, ...]
    te: tuple[float, ...]
    ti: tuple[float, ...]
    flip_angle: tuple[float, ...]
    centre_line: int | None = None
    centre_partition: int | None = None

    @classmethod
    def from_sequence(cls, seq: Any) -> SequenceDefinitions:
        """Read the definitions of a ``pypulseqpp.Sequence``."""
        return cls(
            matrix=_triple(seq.get_definition("Matrix"), int),
            fov=_triple(seq.get_definition("FOV"), float),
            navigator_matrix=_triple(seq.get_definition("NavMatrix"), int),
            navigator_fov=_triple(seq.get_definition("NavFOV"), float),
            tr=tuple(_numbers(seq.get_definition("TR"))),
            te=tuple(_numbers(seq.get_definition("TE"))),
            ti=tuple(_numbers(seq.get_definition("TI"))),
            flip_angle=tuple(_numbers(seq.get_definition("FlipAngle"))),
            centre_line=_counter(seq.get_definition("kSpaceCenterLine")),
            centre_partition=_counter(seq.get_definition("kSpaceCenterPartition")),
        )


@dataclass(frozen=True, eq=False)
class ReadoutTable:
    """Every ADC readout of one sequence, in play order.

    k-space is integrated over ranges of blocks, each starting at the first
    block or at an excitation without a readout, whose pulse centre resets k:
    a range needs nothing from the blocks before it. Tabulating integrates
    every range once; :meth:`readout_k` integrates a readout's range again
    when it is not among the last few kept. The table holds the sequence for
    this, which must not change once tabulated.

    Attributes
    ----------
    block : ndarray
        ``int64``, 1-based index of the block holding each readout.
    num_samples : ndarray
        ``int32`` samples per readout.
    dwell : ndarray
        ``float64`` dwell time, in seconds.
    labels : dict of str to ndarray
        Every label the sequence writes, with the ``int64`` value in force at
        each readout. Labels the sequence never writes are absent.
    center_sample : ndarray
        ``int32`` echo index: the sample of smallest ``|k|`` over the axes
        that vary across the readout. Samples within 1% of the k step at that
        sample tie, and a tie goes to the later sample, or to the earlier one
        on a readout labelled ``REV``, so reversed lines mirror onto forward
        ones. -1 when no axis varies.
    trajectory_dimensions : ndarray
        ``int8`` axes of :meth:`readout_k` a readout keeps once the trailing
        axes constant across it are dropped; 0 when no axis varies.

    Examples
    --------
    >>> import pypulseqpp as pp
    >>> from pulserver.mrd import ReadoutTable
    >>> seq = pp.Sequence(pp.Opts())
    >>> gx = pp.make_trapezoid("x", flat_area=160.0, flat_time=3.2e-3)
    >>> adc = pp.make_adc(num_samples=32, duration=3.2e-3, delay=gx.rise_time)
    >>> rewinder = pp.make_trapezoid("x", area=-gx.area / 2, duration=1e-3)
    >>> for events in ((rewinder,), (gx, adc), (rewinder,)):
    ...     _ = seq.add_block(*events)
    >>> table = ReadoutTable.from_sequence(seq)
    >>> int(table.center_sample[0]), int(table.trajectory_dimensions[0])
    (16, 1)
    """

    block: np.ndarray
    num_samples: np.ndarray
    dwell: np.ndarray
    labels: dict[str, np.ndarray]
    center_sample: np.ndarray
    trajectory_dimensions: np.ndarray
    _ranges: _Ranges = field(repr=False)

    def __len__(self) -> int:
        return int(self.num_samples.size)

    @classmethod
    def from_sequence(cls, seq: Any) -> ReadoutTable:
        """Tabulate the readouts of a ``pypulseqpp.Sequence``."""
        events = np.array(list(seq.block_events.values()), dtype=np.int64)
        events = events.reshape(-1, _COLUMNS)
        block = np.flatnonzero(events[:, _ADC]) + 1
        count = block.size
        adcs, which = _events_by_id(seq, events[block - 1, _ADC], block, "adc")
        sizes = np.array([int(adc.num_samples) for adc in adcs], dtype=np.int32)
        num_samples = sizes[which]
        dwell = np.array([float(adc.dwell) for adc in adcs])[which]
        labels = (
            {
                name: np.broadcast_to(
                    np.asarray(value, dtype=np.int64), (count,)
                ).copy()
                for name, value in seq.evaluate_labels(evolution="adc").items()
            }
            if count
            else {}
        )

        ranges = _Ranges(
            seq, _range_starts(seq, events, block, num_samples), block, num_samples
        )
        reverse = labels.get("REV", np.zeros(count, dtype=np.int64)) != 0
        center_sample = np.full(count, -1, dtype=np.int32)
        dimensions = np.zeros(count, dtype=np.int8)
        for part in np.unique(ranges.of_readout).tolist():
            rows = np.flatnonzero(ranges.of_readout == part)
            center_sample[rows], dimensions[rows] = _echo_and_dimensions(
                ranges.k(part), ranges.start[rows], num_samples[rows], reverse[rows]
            )
        return cls(
            block=block,
            num_samples=num_samples,
            dwell=dwell,
            labels=labels,
            center_sample=center_sample,
            trajectory_dimensions=dimensions,
            _ranges=ranges,
        )

    def readout_k(self, index: int) -> np.ndarray:
        """Return the k-space position of each sample of one readout, in 1/m.

        ``(3, num_samples)``, absolute, with block rotations applied, as
        ``Sequence.calculate_kspace`` returns it.
        """
        ranges = self._ranges
        k = ranges.k(int(ranges.of_readout[index]))
        start = int(ranges.start[index])
        return k[:, start : start + int(self.num_samples[index])].copy()


class _Ranges:
    """The ranges of blocks a sequence's k-space is integrated over, and the last few integrated."""

    def __init__(
        self,
        seq: Any,
        starts: np.ndarray,
        readout_blocks: np.ndarray,
        num_samples: np.ndarray,
    ) -> None:
        self._sequence = seq
        self._first = starts
        self._last = np.append(starts[1:] - 1, len(seq))
        before = np.concatenate(([0], np.cumsum(num_samples, dtype=np.int64)))
        first_readout = np.append(
            np.searchsorted(readout_blocks, starts), num_samples.size
        )
        self._samples = np.diff(before[first_readout])
        #: Range of each readout, and its first sample's column in that range.
        self.of_readout = np.searchsorted(starts, readout_blocks, side="right") - 1
        self.start = before[:-1] - before[first_readout[self.of_readout]]
        self._kept: OrderedDict[int, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()

    def k(self, part: int) -> np.ndarray:
        """Return the ``(3, samples)`` k of every ADC sample in a range, in 1/m."""
        with self._lock:
            if part in self._kept:
                self._kept.move_to_end(part)
                return self._kept[part]
            first, last = int(self._first[part]), int(self._last[part])
            k = np.asarray(
                self._sequence.calculate_kspace(block_range=(first, last))[0],
                dtype=np.float64,
            ).reshape(3, -1)
            if k.shape[1] != self._samples[part]:
                raise RuntimeError(
                    f"blocks {first} to {last} play {k.shape[1]} ADC samples, "
                    f"their readouts {self._samples[part]}"
                )
            self._kept[part] = k
            while len(self._kept) > _KEPT_RANGES:
                self._kept.popitem(last=False)
            return k


# %% private module subroutines


def _events_by_id(
    seq: Any, ids: np.ndarray, blocks: np.ndarray, kind: str
) -> tuple[list[Any], np.ndarray]:
    """Decode one event per distinct ID, and return where each ID falls among them."""
    _, first, which = np.unique(ids, return_index=True, return_inverse=True)
    decoded = [getattr(seq.get_block(int(blocks[at])), kind) for at in first]
    return decoded, which.reshape(-1)


def _range_starts(
    seq: Any, events: np.ndarray, readout_blocks: np.ndarray, num_samples: np.ndarray
) -> np.ndarray:
    """Return the first block of each k-space range.

    Block 1, then each excitation without a readout that follows at least
    ``_RANGE_SAMPLES`` samples of the range before it. A block holding a
    readout never starts a range, whatever its RF, so no readout precedes a
    pulse centre in the block its range starts at.
    """
    rf_blocks = np.flatnonzero(events[:, _RF]) + 1
    pulses, which = _events_by_id(seq, events[rf_blocks - 1, _RF], rf_blocks, "rf")
    resetting = np.array([pulse.use in _RESETTING for pulse in pulses], dtype=bool)
    candidates = rf_blocks[resetting[which] & (events[rf_blocks - 1, _ADC] == 0)]
    before = np.concatenate(([0], np.cumsum(num_samples, dtype=np.int64)))
    reached = before[np.searchsorted(readout_blocks, candidates)]
    starts, since = [1], 0
    for block, samples in zip(candidates.tolist(), reached.tolist(), strict=True):
        if samples - since >= _RANGE_SAMPLES:
            starts.append(block)
            since = samples
    return np.array(starts, dtype=np.int64)


def _echo_and_dimensions(
    k: np.ndarray,
    sample_offset: np.ndarray,
    num_samples: np.ndarray,
    reverse: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    count = sample_offset.size
    center = np.full(count, -1, dtype=np.int32)
    dimensions = np.zeros(count, dtype=np.int8)
    for n in (int(n) for n in np.unique(num_samples)):
        if n == 0:
            continue
        rows_of_size = np.flatnonzero(num_samples == n)
        chunk = max(1, _CHUNK_SAMPLES // n)
        for start in range(0, rows_of_size.size, chunk):
            rows = rows_of_size[start : start + chunk]
            kk = k[:, sample_offset[rows, None] + np.arange(n)].transpose(1, 0, 2)
            span = np.ptp(kk, axis=2)
            varying = span > _CONSTANT * span.max(axis=1, keepdims=True)
            moving = varying.any(axis=1)
            last_varying = 2 - np.argmax(varying[:, ::-1], axis=1)
            dimensions[rows] = np.where(moving, last_varying + 1, 0)
            if n < 2:
                continue

            swept = kk * varying[:, :, None]
            distance = np.sqrt((swept**2).sum(axis=1))
            nearest = distance.argmin(axis=1)
            index = np.arange(rows.size)
            step = np.maximum(
                _step(swept, index, np.clip(nearest - 1, 0, n - 2)),
                _step(swept, index, np.clip(nearest, 0, n - 2)),
            )
            tied = distance <= (distance[index, nearest] + _ECHO_TIE * step)[:, None]
            earliest = tied.argmax(axis=1)
            latest = n - 1 - tied[:, ::-1].argmax(axis=1)
            chosen = np.where(reverse[rows], earliest, latest)
            center[rows] = np.where(moving, chosen, -1)
    return center, dimensions


def _step(swept: np.ndarray, index: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Length of each readout's k step from sample ``at`` to the next."""
    return np.sqrt(((swept[index, :, at + 1] - swept[index, :, at]) ** 2).sum(axis=1))


def _numbers(value: Any) -> list[float]:
    if value in ("", None):
        return []
    return [float(number) for number in np.atleast_1d(value)]


def _triple(value: Any, kind: type) -> tuple | None:
    numbers = _numbers(value)
    return tuple(kind(number) for number in numbers[:3]) if len(numbers) >= 3 else None


def _counter(value: Any) -> int | None:
    numbers = _numbers(value)
    return round(numbers[0]) if numbers else None
