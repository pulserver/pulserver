"""MRD acquisitions, their flags, and the buckets they accumulate into."""

from __future__ import annotations

__all__ = [
    "AcquisitionBucket",
    "AcquisitionBucketStats",
    "AcquisitionFlag",
]

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Flag
from types import SimpleNamespace
from typing import Any

import numpy as np

from ._metadata import has_acquisition_flag


class AcquisitionFlag(Flag):
    """ISMRMRD acquisition flags as bit masks, combinable with ``|``.

    Members are named as the ``ismrmrd.ACQ_*`` constants without the prefix.
    Values are masks, whereas the ``ismrmrd`` constants are 1-based bit
    positions; :attr:`position` and :attr:`flag` give those back. The class does
    not import ``ismrmrd``.

    Examples
    --------
    >>> import pulserver.mrd as mrd
    >>> mrd.AcquisitionFlag.LAST_IN_SLICE.position
    8
    >>> mrd.AcquisitionFlag.IS_NOISE_MEASUREMENT.flag
    'ACQ_IS_NOISE_MEASUREMENT'
    >>> either = mrd.AcquisitionFlag.LAST_IN_SEGMENT | mrd.AcquisitionFlag.LAST_IN_SLICE
    >>> mrd.AcquisitionFlag.LAST_IN_SLICE in either
    True
    """

    FIRST_IN_ENCODE_STEP1 = 1 << 0
    LAST_IN_ENCODE_STEP1 = 1 << 1
    FIRST_IN_ENCODE_STEP2 = 1 << 2
    LAST_IN_ENCODE_STEP2 = 1 << 3
    FIRST_IN_AVERAGE = 1 << 4
    LAST_IN_AVERAGE = 1 << 5
    FIRST_IN_SLICE = 1 << 6
    LAST_IN_SLICE = 1 << 7
    FIRST_IN_CONTRAST = 1 << 8
    LAST_IN_CONTRAST = 1 << 9
    FIRST_IN_PHASE = 1 << 10
    LAST_IN_PHASE = 1 << 11
    FIRST_IN_REPETITION = 1 << 12
    LAST_IN_REPETITION = 1 << 13
    FIRST_IN_SET = 1 << 14
    LAST_IN_SET = 1 << 15
    FIRST_IN_SEGMENT = 1 << 16
    LAST_IN_SEGMENT = 1 << 17
    IS_NOISE_MEASUREMENT = 1 << 18
    IS_PARALLEL_CALIBRATION = 1 << 19
    IS_PARALLEL_CALIBRATION_AND_IMAGING = 1 << 20
    IS_REVERSE = 1 << 21
    IS_NAVIGATION_DATA = 1 << 22
    IS_PHASECORR_DATA = 1 << 23
    LAST_IN_MEASUREMENT = 1 << 24
    IS_HPFEEDBACK_DATA = 1 << 25
    IS_DUMMYSCAN_DATA = 1 << 26
    IS_RTFEEDBACK_DATA = 1 << 27
    IS_SURFACECOILCORRECTIONSCAN_DATA = 1 << 28
    IS_PHASE_STABILIZATION_REFERENCE = 1 << 29
    IS_PHASE_STABILIZATION = 1 << 30
    COMPRESSION1 = 1 << 52
    COMPRESSION2 = 1 << 53
    COMPRESSION3 = 1 << 54
    COMPRESSION4 = 1 << 55
    USER1 = 1 << 56
    USER2 = 1 << 57
    USER3 = 1 << 58
    USER4 = 1 << 59
    USER5 = 1 << 60
    USER6 = 1 << 61
    USER7 = 1 << 62
    USER8 = 1 << 63

    @property
    def flag(self) -> str:
        """The ``ismrmrd`` constant name of a single flag."""
        return f"ACQ_{self.name}"

    @property
    def position(self) -> int:
        """The 1-based bit position of a single flag, as ``ismrmrd`` numbers it."""
        return int(self.value).bit_length()

    @classmethod
    def of(cls, acquisition: Any) -> AcquisitionFlag:
        """Return every flag the acquisition carries."""
        found = cls(0)
        for member in cls:
            if has_acquisition_flag(acquisition, member.flag):
                found |= member
        return found


#: The ``LAST_IN_*`` flags: the boundaries a bucket can end on.
BOUNDARIES = (
    AcquisitionFlag.LAST_IN_ENCODE_STEP1
    | AcquisitionFlag.LAST_IN_ENCODE_STEP2
    | AcquisitionFlag.LAST_IN_AVERAGE
    | AcquisitionFlag.LAST_IN_SLICE
    | AcquisitionFlag.LAST_IN_CONTRAST
    | AcquisitionFlag.LAST_IN_PHASE
    | AcquisitionFlag.LAST_IN_REPETITION
    | AcquisitionFlag.LAST_IN_SET
    | AcquisitionFlag.LAST_IN_SEGMENT
    | AcquisitionFlag.LAST_IN_MEASUREMENT
)


@dataclass(frozen=True)
class AcquisitionBucketStats:
    """Distinct values of each encoding counter in one bucket.

    Field names follow Gadgetron's ``AcquisitionBucketStats``.

    Examples
    --------
    >>> import pulserver.mrd as mrd
    >>> stats = mrd.AcquisitionBucketStats(kspace_encode_step_1=frozenset({0, 1, 2}))
    >>> sorted(stats.kspace_encode_step_1)
    [0, 1, 2]
    """

    kspace_encode_step_1: frozenset[int] = frozenset()
    kspace_encode_step_2: frozenset[int] = frozenset()
    slice: frozenset[int] = frozenset()
    phase: frozenset[int] = frozenset()
    contrast: frozenset[int] = frozenset()
    repetition: frozenset[int] = frozenset()
    set: frozenset[int] = frozenset()
    segment: frozenset[int] = frozenset()
    average: frozenset[int] = frozenset()


@dataclass(frozen=True)
class AcquisitionBucket:
    """Acquisitions accumulated up to a boundary, split as Gadgetron splits them.

    Parameters
    ----------
    data
        Imaging acquisitions.
    datastats
        One :class:`AcquisitionBucketStats` per encoding space in ``data``.
    ref
        Parallel-imaging calibration acquisitions. One flagged as calibration and
        imaging appears in both ``data`` and ``ref``.
    refstats
        One :class:`AcquisitionBucketStats` per encoding space in ``ref``.
    waveforms
        Waveforms received with the stream.
    acquisitions
        Every acquisition in arrival order; the last one closed the bucket.
        Empty derives it as ``data`` followed by the ``ref`` entries not in
        ``data``.

    Examples
    --------
    >>> import numpy as np
    >>> import pulserver.mrd as mrd
    >>> bucket = mrd.AcquisitionBucket.from_arrays(
    ...     np.ones((4, 2, 8), dtype=np.complex64),
    ...     labels={"kspace_encode_step_1": np.arange(4)},
    ... )
    >>> len(bucket.data)
    4
    """

    data: tuple[Any, ...]
    datastats: tuple[AcquisitionBucketStats, ...] = ()
    ref: tuple[Any, ...] = ()
    refstats: tuple[AcquisitionBucketStats, ...] = ()
    waveforms: tuple[Any, ...] = ()
    acquisitions: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if not self.acquisitions:
            extra = tuple(
                acquisition
                for acquisition in self.ref
                if not any(acquisition is item for item in self.data)
            )
            object.__setattr__(self, "acquisitions", self.data + extra)

    @property
    def trigger(self) -> AcquisitionFlag:
        """Boundary flags of the last acquisition; no flags for an empty bucket.

        Classifying flags are excluded, so a trigger equal to one boundary means no
        larger unit ended with it.
        """
        if not self.acquisitions:
            return AcquisitionFlag(0)
        return AcquisitionFlag.of(self.acquisitions[-1]) & BOUNDARIES

    @classmethod
    def from_arrays(
        cls,
        data: Any,
        trajectory: Any | None = None,
        *,
        labels: Mapping[str, Any] | None = None,
        reference: Any | None = None,
        reference_labels: Mapping[str, Any] | None = None,
    ) -> AcquisitionBucket:
        """Build a bucket from arrays, without MRD objects.

        The acquisitions carry no flags and act as their own headers.

        Parameters
        ----------
        data : array_like
            K-space, ``(acquisitions, coils, samples)``.
        trajectory : array_like, default=None
            Per-acquisition trajectories with the same leading axis as ``data``.
        labels : Mapping[str, Any], default=None
            Encoding counters by MRD field name, one value per acquisition; absent
            counters are 0.
        reference : array_like, default=None
            Calibration k-space, ``(acquisitions, coils, samples)``.
        reference_labels : Mapping[str, Any], default=None
            Encoding counters of the ``reference`` acquisitions; absent counters
            are 0.

        Raises
        ------
        ValueError
            If ``data`` has no leading axis, or ``trajectory`` has a different
            number of acquisitions.
        """
        arrays = _split_leading(data)
        trajectories = _split_optional_leading(trajectory, len(arrays))
        label_values = {} if labels is None else dict(labels)
        acquisitions = tuple(
            _ArrayAcquisition(
                array,
                trajectories[index],
                _labels_at(label_values, index),
            )
            for index, array in enumerate(arrays)
        )
        references: tuple[Any, ...] = ()
        if reference is not None:
            reference_values = (
                {} if reference_labels is None else dict(reference_labels)
            )
            references = tuple(
                _ArrayAcquisition(array, None, _labels_at(reference_values, index))
                for index, array in enumerate(_split_leading(reference))
            )
        return cls(data=acquisitions, ref=references)

    def __len__(self) -> int:
        """Return the number of imaging acquisitions."""
        return len(self.data)

    def kspace(self, *, reference: bool = False) -> Any:
        """Stack ``data`` (or ``ref``) as ``(acquisitions, coils, samples)``.

        Torch tensors stack with Torch. Readouts of unequal shape come back as a
        tuple, and no acquisitions as an empty complex array.
        """
        acquisitions = self.ref if reference else self.data
        return _stack_or_tuple(tuple(acquisition.data for acquisition in acquisitions))

    def trajectory(self, *, reference: bool = False) -> Any | None:
        """Stack trajectories as :meth:`kspace` stacks data.

        ``None`` when no acquisition carries one; a tuple holding ``None`` entries
        when only some do.
        """
        acquisitions = self.ref if reference else self.data
        values = tuple(_trajectory(acquisition) for acquisition in acquisitions)
        if not values or all(value is None for value in values):
            return None
        if any(value is None for value in values):
            return values
        return _stack_or_tuple(values)

    def labels(self, name: str, *, reference: bool = False) -> np.ndarray:
        """Return one encoding counter per acquisition, 0 where absent."""
        acquisitions = self.ref if reference else self.data
        return np.asarray(
            [_acquisition_label(acquisition, name) for acquisition in acquisitions]
        )

    @property
    def headers(self) -> tuple[Any, ...]:
        """Native headers of the imaging acquisitions."""
        return tuple(_header(acquisition) for acquisition in self.data)


def _split_leading(value: Any) -> tuple[Any, ...]:
    try:
        return tuple(value[index] for index in range(len(value)))
    except TypeError as error:
        raise ValueError("data must have an acquisition dimension") from error


def _split_optional_leading(value: Any | None, length: int) -> tuple[Any | None, ...]:
    if value is None:
        return (None,) * length
    values = _split_leading(value)
    if len(values) != length:
        raise ValueError("trajectory and data acquisition dimensions must match")
    return values


def _labels_at(labels: Mapping[str, Any], index: int) -> dict[str, int]:
    """Return the counters of acquisition ``index``, 0 for each one not supplied."""
    counters = dict.fromkeys(AcquisitionBucketStats.__dataclass_fields__, 0)
    counters.update({name: int(value[index]) for name, value in labels.items()})
    return counters


def _stack_or_tuple(values: tuple[Any, ...]) -> Any:
    if not values:
        return np.empty((0,), dtype=np.complex64)
    first = values[0]
    if type(first).__module__.startswith("torch"):
        import torch

        try:
            return torch.stack(values)
        except RuntimeError:
            return values
    try:
        return np.stack(values)
    except ValueError:
        return values


def _trajectory(acquisition: Any) -> Any | None:
    value = getattr(acquisition, "traj", None)
    if value is None:
        value = getattr(acquisition, "trajectory", None)
    if value is None or getattr(value, "size", 0) == 0:
        return None
    return value


def _header(acquisition: Any) -> Any:
    get_head = getattr(acquisition, "getHead", None)
    return (
        get_head() if callable(get_head) else getattr(acquisition, "head", acquisition)
    )


def _acquisition_label(acquisition: Any, name: str) -> int:
    index = getattr(acquisition, "idx", None)
    if index is None:
        index = getattr(_header(acquisition), "idx", None)
    if index is not None and hasattr(index, name):
        return int(getattr(index, name))
    return int(getattr(acquisition, name, 0))


class _ArrayAcquisition:
    def __init__(self, data: Any, trajectory: Any | None, labels: Mapping[str, int]):
        self.data = data
        self.traj = trajectory
        self.idx = SimpleNamespace(**labels)
        self.flags = 0

    def getHead(self) -> Any:
        return self

    def is_flag_set(self, flag: int) -> bool:
        return bool(self.flags & (1 << (flag - 1)))
