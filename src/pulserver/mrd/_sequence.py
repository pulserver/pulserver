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

#: ADC column of ``Sequence.block_events``, and the number of columns.
_ADC, _COLUMNS = 5, 7


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


@dataclass(frozen=True)
class SequenceDefinitions:
    """Definitions describing what a sequence acquires, in Pulseq units.

    ``TR``, ``TE`` and ``FlipAngle`` a sequence does not define are measured
    by ``Sequence.test_report_dict``: TE from the excitation before the
    closest approach to the k-space centre, TR between the excitations around
    it, and every distinct flip angle the sequence plays.

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

        Measuring runs ``check_timing``, which may record ``TotalDuration`` and
        needs the sequence on a system, as ``pypulseqpp.io.read`` builds one.
        """
        tr = tuple(_numbers(seq.get_definition("TR")))
        te = tuple(_numbers(seq.get_definition("TE")))
        flip_angle = tuple(_numbers(seq.get_definition("FlipAngle")))
        if not (tr and te and flip_angle):
            measured_tr, measured_te, measured_flip = _measured(seq)
            tr = tr or measured_tr
            te = te or measured_te
            flip_angle = flip_angle or measured_flip
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

    k-space is integrated a run of consecutive readouts at a time, by
    ``Sequence.adc_kspace(readouts=...)``, from the last pulse before the run
    that resets it. Tabulating integrates the whole sequence once, to find
    each readout's echo; :meth:`readout_k` integrates a readout's run when it
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
        Every label the sequence writes, with the ``int64`` value in force at
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

    def __len__(self) -> int:
        return int(self.num_samples.size)

    @classmethod
    def from_sequence(cls, seq: Any) -> ReadoutTable:
        """Tabulate the readouts of a ``pypulseqpp.Sequence``."""
        events = np.array(list(seq.block_events.values()), dtype=np.int64)
        events = events.reshape(-1, _COLUMNS)
        echoes = seq.adc_echoes()
        block = echoes.block.astype(np.int64)
        count = block.size
        num_samples = echoes.num_samples.astype(np.int32)
        adcs, which = _events_by_id(seq, events[block - 1, _ADC], block, "adc")
        dwell = np.array([float(adc.dwell) for adc in adcs])[which]
        labels = _labels_at(seq, block)

        reverse = labels.get("REV", np.zeros(count, dtype=np.int64)) != 0
        moves = echoes.moving.any(axis=1)
        center_sample = np.where(
            moves, np.where(reverse, echoes.echo[:, 0], echoes.echo[:, 1]), -1
        ).astype(np.int32)
        # The last axis moved along, counted from 1.
        last_moving = 3 - np.argmax(echoes.moving[:, ::-1], axis=1)
        dimensions = np.where(moves, last_moving, 0).astype(np.int8)
        return cls(
            block=block,
            num_samples=num_samples,
            dwell=dwell,
            labels=labels,
            center_sample=center_sample,
            trajectory_dimensions=dimensions,
            _runs=_Runs(seq, num_samples),
        )

    def readout_k(self, index: int) -> np.ndarray:
        """Return the k-space position of each sample of one readout, in 1/m.

        ``(3, num_samples)``, absolute, with block rotations applied, as
        ``Sequence.adc_kspace`` returns it.
        """
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
        self._sequence = seq
        #: Samples before each readout, and after the last.
        self.before = np.concatenate(([0], np.cumsum(num_samples, dtype=np.int64)))
        #: First readout of each run.
        self.first = np.flatnonzero(
            np.diff(self.before[:-1] // _RUN_SAMPLES, prepend=-1)
        )
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
                self._sequence.adc_kspace(readouts=readouts), dtype=np.float64
            ).reshape(3, -1)
            self._kept[run] = k
            while len(self._kept) > _KEPT_RUNS:
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
            name: np.asarray(value, dtype=np.int64).copy()
            for name, value in found.items()
        }
    at = int(block[0]) - 1
    return {
        name: np.atleast_1d(np.asarray(value, dtype=np.int64))[
            [at if np.size(value) > 1 else 0]
        ]
        for name, value in seq.evaluate_labels(evolution="blocks").items()
    }


def _measured(seq: Any) -> tuple[tuple[float, ...], ...]:
    """TR, TE and flip angles as pypulseqpp's report measures them; empty where it cannot.

    A sequence without RF has no TR: the report's fallback to the total
    duration is not one.
    """
    report = seq.test_report_dict()
    tr, te = float(report["TR"]), float(report["TE"])
    plays_rf = int(report["event_count"]["rf"]) > 0
    return (
        (tr,) if plays_rf and math.isfinite(tr) else (),
        (te,) if math.isfinite(te) else (),
        tuple(float(angle) for angle in report["flip_angles_deg"]),
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
