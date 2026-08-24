"""What a scanner sends, one readout at a time.

An acquisition carries a line of k-space, the counters saying where in the
encoding space it belongs, and flags saying what it is for -- a noise
measurement, a navigator, the last line of a slice. A bucket is what they
accumulate into while a boundary is still open.
"""

from __future__ import annotations

__all__ = [
    "AcquisitionBucket",
    "AcquisitionBucketStats",
    "AcquisitionFlag",
    "acquisition_label",
    "acquisition_labels",
    "has_acquisition_flag",
]

from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from enum import Flag
from typing import Any

import numpy as np

from ._metadata import (
    acquisition_label,
    acquisition_labels,
    has_acquisition_flag,
)


class AcquisitionFlag(Flag):
    """The MRD acquisition flags, as the bits a scanner actually sets.

    Every ``ismrmrd.ACQ_*`` flag, named without the prefix the class already
    supplies, and combinable with ``|`` because that is how an acquisition
    carries them: one line is routinely calibration *and* imaging, or the last
    of its segment *and* the last of its slice.

    Use them wherever a flag is named -- which boundary reconstructs what,
    what a plugin requires or rejects, what a boundary meant::

        super().__init__(
            branches={AcquisitionFlag.LAST_IN_SLICE: "imaging"},
            reject_flags=AcquisitionFlag.IS_NOISE_MEASUREMENT,
        )

    Values are the MRD bit masks. The ``ismrmrd`` constants are 1-based bit
    *positions* rather than masks, which :attr:`position` converts back to, and
    :attr:`flag` gives the constant's name for anything that wants it spelled
    out. Defining them here rather than reading them from ``ismrmrd`` keeps
    this module importable without it; that they agree is a test.

    Examples
    --------
    >>> import pulserver.mrd as mrd
    >>> mrd.AcquisitionFlag.LAST_IN_SLICE.position
    8
    >>> mrd.AcquisitionFlag.IS_NOISE_MEASUREMENT.flag
    'ACQ_IS_NOISE_MEASUREMENT'

    Flags combine, which is how a plugin routes either of two boundaries to
    one branch::

        super().__init__(
            branches={
                AcquisitionFlag.LAST_IN_SEGMENT
                | AcquisitionFlag.LAST_IN_SLICE: "imaging"
            }
        )
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
        """The ``ismrmrd`` constant's name, for anything taking one."""
        return f"ACQ_{self.name}"

    @property
    def position(self) -> int:
        """The 1-based bit position ``ismrmrd`` names this flag by."""
        return int(self.value).bit_length()

    @classmethod
    def of(cls, acquisition: Any) -> AcquisitionFlag:
        """Every flag one acquisition carries."""
        found = cls(0)
        for member in cls:
            if has_acquisition_flag(acquisition, member.flag):
                found |= member
        return found


#: The boundaries a bucket can end on: what a sequence marks when it wants a
#: reconstruction to notice that a unit is complete.
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
    """Encoding labels present in one acquisition bucket.

    The field names match Gadgetron's ``AcquisitionBucketStats``. Each value is
    a frozen set because stats describe a completed bucket.
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
    """Gadgetron-style unit of acquired data presented to a recon app.

    Parameters
    ----------
    data
        Imaging acquisitions.
    datastats
        One :class:`AcquisitionBucketStats` per encoding space in ``data``.
    ref
        Parallel-imaging reference acquisitions. Acquisitions flagged as both
        imaging and calibration appear in both ``data`` and ``ref``, matching
        Gadgetron.
    refstats
        One stats object per encoding space in ``ref``.
    waveforms
        Scanner waveforms associated with this bucket.
    acquisitions
        Every acquisition in the order it arrived, which is the order a
        reconstruction replaying the bucket has to see them in, and whose last
        entry is the one that ended the bucket. ``data`` and ``ref`` are
        classified views over this. Left empty, it is derived from them, and
        reference-only acquisitions then follow the imaging ones.

    Notes
    -----
    Online buckets contain native ``ismrmrd.Acquisition`` and
    ``ismrmrd.Waveform`` objects, preserving the fields familiar to Gadgetron
    users. :meth:`kspace`, :meth:`trajectory`, and :meth:`labels` provide the
    compact array-level path used by most reconstruction algorithms.
    """

    data: tuple[Any, ...]
    datastats: tuple[AcquisitionBucketStats, ...] = ()
    ref: tuple[Any, ...] = ()
    refstats: tuple[AcquisitionBucketStats, ...] = ()
    waveforms: tuple[Any, ...] = ()
    acquisitions: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        """Derive the arrival order, and the boundary, from what was given."""
        if not self.acquisitions:
            extra = tuple(
                acquisition
                for acquisition in self.ref
                if not any(acquisition is item for item in self.data)
            )
            object.__setattr__(self, "acquisitions", self.data + extra)

    @property
    def trigger(self) -> AcquisitionFlag:
        """Which boundaries of the scan the last acquisition closed.

        Only the boundaries: an acquisition's other flags say what it *is*, not
        what it ended, and mixing them in would break the reading that a
        trigger equal to one boundary means nothing larger ended.
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
        """Create an offline bucket without requiring an MRD connection.

        Parameters
        ----------
        data
            K-space shaped ``(acquisitions, coils, samples)``.
        trajectory
            Optional trajectory with the same acquisition dimension.
        labels
            Optional arrays of encoding counters. Missing labels are zero.
        reference
            Optional reference k-space with the same trailing dimensions.
        reference_labels
            Encoding counters of the reference acquisitions. A parallel-imaging
            calibration region is a set of specific phase encodes, so a
            reconstruction that places it on a grid needs these; missing labels
            are zero, as for ``labels``.

        Returns
        -------
        AcquisitionBucket
            Bucket containing acquisition-compatible lightweight objects.
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
        """Stack acquisition data as ``(acquisitions, coils, samples)``.

        Uniform acquisitions are stacked using their native array library when
        possible. Ragged acquisitions remain a tuple rather than being padded.
        """
        acquisitions = self.ref if reference else self.data
        return _stack_or_tuple(tuple(acquisition.data for acquisition in acquisitions))

    def trajectory(self, *, reference: bool = False) -> Any | None:
        """Return stacked trajectories, or ``None`` when they are absent."""
        acquisitions = self.ref if reference else self.data
        values = tuple(_trajectory(acquisition) for acquisition in acquisitions)
        if not values or all(value is None for value in values):
            return None
        if any(value is None for value in values):
            return values
        return _stack_or_tuple(values)

    def labels(self, name: str, *, reference: bool = False) -> np.ndarray:
        """Return one MRD encoding counter for every acquisition."""
        acquisitions = self.ref if reference else self.data
        return np.asarray(
            [_acquisition_label(acquisition, name) for acquisition in acquisitions]
        )

    @property
    def headers(self) -> tuple[Any, ...]:
        """Return native acquisition headers for imaging data."""
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
    """Return one acquisition's counters, every encoding counter present.

    Counters the caller did not supply read as zero rather than being absent,
    so ``acquisition.idx.slice`` answers for an offline acquisition the same
    way it answers for a streamed one.
    """
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
        return bool(self.flags & flag)
