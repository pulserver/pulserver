"""Per-readout facts of a Pulseq sequence, read through pypulseqpp.

Each wrapper takes a ``pypulseqpp.Sequence`` and returns arrays in the
sequence's units; nothing here depends on MRD.
"""

from __future__ import annotations

__all__ = ["ReadoutTable", "SequenceDefinitions", "read_chain"]

import math
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

#: ADC samples after which a run of readouts ends and the next starts.
_RUN_SAMPLES = 1 << 17

#: Runs whose k-space a table keeps.
_KEPT_RUNS = 2

#: ADC column of ``Sequence.libraries().blocks``.
_ADC = 4


def read_chain(path: Path | str, *, verify: bool = False) -> list[tuple[Path, Any]]:
    """Read a sequence file and every file its ``NextSequence`` definitions name, in play order.

    ``pypulseqpp.io.read_chain``, with the use of every pulse a file leaves
    unlabelled detected by pypulseqpp's rule, so that the cache, the
    enrichment and the virtual scanner take one use for each pulse.

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

    return pp.io.read_chain(path, detect_rf_use=True, verify=verify)


def designed_chain(written: list[tuple[Path, Any]]) -> list[tuple[Path, Any]]:
    """Return a chain as written, each pulse labelled as :func:`read_chain` labels the file.

    The sequences are the ones the files were written from, deduplicated as
    written, so the chain stands for reading the files back. A file is read
    with no system, at 1.5 T and the proton gyromagnetic ratio, and its
    unlabelled pulses are labelled there; the sequences are labelled in place
    at the same field.
    """
    for _, sequence in written:
        sequence.detect_rf_use(B0=1.5, gamma=42.576e6)
    return written


@dataclass(frozen=True)
class SequenceDefinitions:
    """Definitions describing what a sequence acquires, in Pulseq units.

    ``TR`` and ``TE`` a sequence does not define are measured by
    ``Sequence.test_report_dict``: TE from the excitation before the closest
    approach to the k-space centre, TR between the excitations around it. An
    undefined ``FlipAngle`` is every distinct value of
    ``Sequence.rf_flip_angles``, as the report lists them.

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
        """Read the definitions of a ``pypulseqpp.Sequence``, measuring those it lacks.

        Measuring TR or TE runs ``check_timing``, which may record ``TotalDuration`` and
        needs the sequence on a system, as ``pypulseqpp.io.read`` builds one.
        """
        tr = tuple(_numbers(seq.get_definition("TR")))
        te = tuple(_numbers(seq.get_definition("TE")))
        flip_angle = tuple(_numbers(seq.get_definition("FlipAngle")))
        if not (tr and te):
            measured_tr, measured_te = _measured(seq)
            tr = tr or measured_tr
            te = te or measured_te
        if not flip_angle:
            flip_angle = tuple(float(a) for a in np.unique(seq.rf_flip_angles()))
        return cls(
            matrix=_triple(seq.get_definition("Matrix"), int),
            fov=_triple(seq.get_definition("FOV"), float),
            navigator_matrix=_triple(seq.get_definition("NavMatrix"), int),
            navigator_fov=_triple(seq.get_definition("NavFOV"), float),
            tr=tr,
            te=te,
            ti=tuple(_numbers(seq.get_definition("TI"))),
            flip_angle=flip_angle,
            centre_line=_counter(seq.get_definition("kSpaceCenterLine")),
            centre_partition=_counter(seq.get_definition("kSpaceCenterPartition")),
        )


@dataclass(frozen=True, eq=False)
class ReadoutTable:
    """Every ADC readout of one sequence, in play order.

    Tabulating follows k-space through the whole sequence once, to find each
    readout's echo. Where ``Sequence.adc_echoes`` returns each readout's k as
    its block's origin plus the sweep of a block of its kind,
    :meth:`readout_k` adds the two. Otherwise k-space is integrated a run of
    consecutive readouts at a time, by ``Sequence.adc_kspace(readouts=...)``,
    from the last pulse before the run that resets it, when the readout's run
    is not among the last few kept. The table holds the sequence for this,
    which must not change once tabulated.

    Attributes
    ----------
    block : ndarray
        ``int64``, 1-based index of the block holding each readout.
    num_samples : ndarray
        ``int32`` samples per readout.
    dwell : ndarray
        ``float64`` dwell time, in seconds.
    labels : dict of str to ndarray
        Every label the sequence writes, with the ``int32`` value in force at
        each readout. Labels the sequence never writes are absent.
    center_sample : ndarray
        ``int32`` echo index: of the samples ``pypulseqpp.Sequence.adc_echoes``
        finds nearest the centre of k-space over the axes the readout moves
        along, the last, or the first on a readout labelled ``REV``, so
        reversed lines mirror onto forward ones. -1 when no axis moves.
    trajectory_dimensions : ndarray
        ``int8`` axes of :meth:`readout_k` a readout keeps once the trailing
        axes it does not move along are dropped; 0 when no axis moves.

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
    _runs: _Runs = field(repr=False)
    #: One entry per distinct phase modulation shape: the modulation in rad, or None.
    _phase_modulation: tuple[np.ndarray | None, ...] = field(default=(), repr=False)
    #: Which entry of _phase_modulation each readout plays.
    _modulated_by: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64), repr=False
    )
    #: Per readout, k at the start of its block, in 1/m; None where not walked.
    _origin: np.ndarray | None = field(default=None, repr=False)
    _sweep: np.ndarray | None = field(default=None, repr=False)
    _sweeps: tuple[np.ndarray, ...] = field(default=(), repr=False)

    def __len__(self) -> int:
        return int(self.num_samples.size)

    @classmethod
    def from_sequence(cls, seq: Any) -> ReadoutTable:
        """Tabulate the readouts of a ``pypulseqpp.Sequence``."""
        tables = seq.libraries()
        echoes = seq.adc_echoes()
        block = echoes.block.astype(np.int64)
        count = block.size
        num_samples = echoes.num_samples.astype(np.int32)
        adc_rows = np.asarray(tables.adc, dtype=np.float64).reshape(-1, 8)
        adc_row = np.asarray(tables.blocks)[block - 1, _ADC].astype(np.int64) - 1
        dwell = adc_rows[adc_row, 1]
        # A shifted field of view gives every readout an ADC row of its own;
        # the modulation is decoded once per shape the rows name.
        adcs, which = _events_by_shape(seq, adc_rows[:, 7], adc_row, block, "adc")
        labels = _labels_at(seq, block)

        reverse = labels.get("REV", np.zeros(count, dtype=np.int64)) != 0
        moves = echoes.moving.any(axis=1)
        center_sample = np.where(
            moves, np.where(reverse, echoes.echo[:, 0], echoes.echo[:, 1]), -1
        ).astype(np.int32)
        # The last axis moved along, counted from 1.
        moving = echoes.moving
        dimensions = np.where(
            moving[:, 2], 3, np.where(moving[:, 1], 2, moves.astype(np.int8))
        ).astype(np.int8)
        return cls(
            block=block,
            num_samples=num_samples,
            dwell=dwell,
            labels=labels,
            center_sample=center_sample,
            trajectory_dimensions=dimensions,
            _runs=_Runs(seq, num_samples),
            _phase_modulation=tuple(_phase_modulation_of(adc) for adc in adcs),
            _modulated_by=which,
            _origin=getattr(echoes, "origin", None),
            _sweep=getattr(echoes, "sweep", None),
            _sweeps=getattr(echoes, "sweeps", None) or (),
        )

    def readout_phase_modulation(self, index: int) -> np.ndarray | None:
        """Return the phase modulation of one readout's ADC, in rad, or None.

        One value per sample. A readout played under a gradient that holds one
        value throughout carries none: its share of a shifted field of view is
        a phase and a frequency offset, which the receiver applies itself.
        """
        return self._phase_modulation[int(self._modulated_by[index])]

    def readout_block(self, index: int) -> Any:
        """Return the decoded block holding one readout, as ``Sequence.get_block`` does."""
        return self._runs.sequence.get_block(int(self.block[index]))

    def readout_k(self, index: int) -> np.ndarray:
        """Return the k-space position of each sample of one readout, in 1/m.

        ``(3, num_samples)``, absolute, with block rotations applied, as
        ``Sequence.adc_kspace`` returns it.
        """
        if self._origin is not None:
            return self._origin[index][:, None] + self._sweeps[self._sweep[index]]
        runs = self._runs
        run = int(np.searchsorted(runs.first, index, side="right")) - 1
        start = int(runs.before[index] - runs.before[runs.first[run]])
        return runs.k(run)[:, start : start + int(self.num_samples[index])].copy()


class _Runs:
    """Consecutive readouts whose k-space is integrated together, and the last few integrated.

    A run starts at each readout whose first sample, counted over the whole
    sequence, is the first at or past a multiple of ``_RUN_SAMPLES``.
    """

    def __init__(self, seq: Any, num_samples: np.ndarray) -> None:
        self.sequence = seq
        #: Samples before each readout, and after the last.
        self.before = np.concatenate(([0], np.cumsum(num_samples, dtype=np.int64)))
        #: First readout of each run.
        last = int(self.before[-2]) if num_samples.size else -1
        starts = np.arange(0, last + 1, _RUN_SAMPLES)
        self.first = np.unique(np.searchsorted(self.before[:-1], starts))
        self._stop = np.append(self.first[1:], num_samples.size)
        self._kept: OrderedDict[int, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()

    def k(self, run: int) -> np.ndarray:
        """Return the ``(3, samples)`` k of every ADC sample of a run, in 1/m."""
        with self._lock:
            if run in self._kept:
                self._kept.move_to_end(run)
                return self._kept[run]
            readouts = (int(self.first[run]), int(self._stop[run]))
            k = np.asarray(
                self.sequence.adc_kspace(readouts=readouts), dtype=np.float64
            ).reshape(3, -1)
            self._kept[run] = k
            while len(self._kept) > _KEPT_RUNS:
                self._kept.popitem(last=False)
            return k


# %% private module subroutines


def _phase_modulation_of(adc: Any) -> np.ndarray | None:
    """Return an ADC's phase modulation in rad, or None where it has none."""
    values = getattr(adc, "phase_modulation", None)
    if values is None:
        return None
    values = np.asarray(values, dtype=np.float64).ravel()
    return values if values.size else None


def _events_by_shape(
    seq: Any, shape: np.ndarray, row: np.ndarray, blocks: np.ndarray, kind: str
) -> tuple[list[Any], np.ndarray]:
    """Decode one event per distinct shape of the library rows played, and return which each readout plays.

    ``shape`` is per library row, ``row`` the 0-based library row of each
    readout and ``blocks`` its block.
    """
    count = row.size
    first = np.full(shape.size, count, dtype=np.int64)
    np.minimum.at(first, row, np.arange(count))
    used = np.flatnonzero(first < count)
    distinct, inverse = np.unique(shape[used], return_inverse=True)
    start = np.full(distinct.size, count, dtype=np.int64)
    np.minimum.at(start, inverse, first[used])
    decoded = [getattr(seq.get_block(int(blocks[at])), kind) for at in start]
    of_row = np.zeros(shape.size, dtype=np.int64)
    of_row[used] = inverse
    return decoded, of_row[row]


def _labels_at(seq: Any, block: np.ndarray) -> dict[str, np.ndarray]:
    """Return every label the sequence writes, with its value at each readout block.

    pypulseqpp answers a single recorded point with the labels' final values,
    so a sequence with one readout is read at that block of the evolution over
    every block.
    """
    if block.size == 0:
        return {}
    if block.size > 1:
        found = seq.evaluate_labels(evolution="adc")
        return {
            name: np.asarray(value, dtype=np.int32) for name, value in found.items()
        }
    at = int(block[0]) - 1
    return {
        name: np.atleast_1d(np.asarray(value, dtype=np.int32))[
            [at if np.size(value) > 1 else 0]
        ]
        for name, value in seq.evaluate_labels(evolution="blocks").items()
    }


def _measured(seq: Any) -> tuple[tuple[float, ...], ...]:
    """TR and TE as pypulseqpp's report measures them; empty where it cannot.

    A sequence without RF has no TR: the report's fallback to the total
    duration is not one.
    """
    report = seq.test_report_dict()
    tr, te = float(report["TR"]), float(report["TE"])
    plays_rf = int(report["event_count"]["rf"]) > 0
    return (
        (tr,) if plays_rf and math.isfinite(tr) else (),
        (te,) if math.isfinite(te) else (),
    )


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
