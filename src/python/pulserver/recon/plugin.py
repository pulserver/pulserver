"""Public contract for offline and inline reconstruction plugins.

A reconstruction plugin is a regular Python module containing one
:class:`ReconPlugin` subclass and a module-level ``PLUGIN`` instance.
Pulserver's private MRD runtime drives its lifecycle hooks over an inline
stream; calling the instance replays a ready bucket through the very same
hooks, so a plugin has one behaviour rather than an online one and an offline
one.

The data model keeps Gadgetron's familiar acquisition-bucket vocabulary while
adding array conveniences for users concerned only with reconstruction. MRD
connections, framing, close handling, and output serialization remain private.

The three hooks divide the work the same way in every plugin: :meth:`startup`
lays out the buffers the header describes, :meth:`receive` places each
acquisition and routes the boundaries it closes to a named branch, and
:meth:`recon` holds the reconstruction of each branch over buffers that are
already filled.

Examples
--------
>>> import numpy as np
>>> from types import SimpleNamespace
>>> from pulserver import AcquisitionBucket, ReconPlugin, ReconContext, ReconResult
>>> class RootSumOfSquares(ReconPlugin):
...     def recon(self, branch, context):
...         del branch, context
...         kspace = self.buffers[0].kspace
...         return ReconResult(np.sqrt(np.sum(np.abs(kspace) ** 2, axis=0)))
>>> matrix = SimpleNamespace(matrixSize=SimpleNamespace(x=8, y=4, z=1))
>>> header = SimpleNamespace(
...     encoding=[SimpleNamespace(encodedSpace=matrix, reconSpace=matrix)],
...     acquisitionSystemInformation=SimpleNamespace(receiverChannels=2),
... )
>>> bucket = AcquisitionBucket.from_arrays(
...     np.ones((4, 2, 8), dtype=np.complex64),
...     labels={"kspace_encode_step_1": np.arange(4)},
... )
>>> result = RootSumOfSquares()(bucket, ReconContext.offline(header))
>>> result.data.shape
(4, 8)
"""

from __future__ import annotations

__all__ = [
    "AcquisitionBucket",
    "AcquisitionBucketStats",
    "AcquisitionFlag",
    "EncodingSpace",
    "ExamCache",
    "ReconBuffer",
    "ReconContext",
    "ReconData",
    "ReconPlugin",
    "ReconResult",
    "has_acquisition_flag",
]

import copy
import logging
from abc import ABC, abstractmethod
from enum import Flag
from collections.abc import Callable, Hashable, Iterator, Mapping, MutableMapping
from dataclasses import dataclass, field
from threading import RLock
from types import SimpleNamespace
from typing import Any

import numpy as np

from ._buffers import EncodingSpace, ReconBuffer, ReconData
from ._mrd.metadata import has_acquisition_flag


class AcquisitionFlag(Flag):
    """The MRD acquisition flags, as the bits a scanner actually sets.

    Every ``ismrmrd.ACQ_*`` flag, named without the prefix the class already
    supplies, and combinable with ``|`` because that is how an acquisition
    carries them: one line is routinely calibration *and* imaging, or the last
    of its segment *and* the last of its slice.

    Use them wherever a flag is named -- what ends a bucket, what a plugin
    requires or rejects, what a boundary meant::

        super().__init__(
            split_on=AcquisitionFlag.LAST_IN_SEGMENT | AcquisitionFlag.LAST_IN_SLICE,
            reject_flags=AcquisitionFlag.IS_NOISE_MEASUREMENT,
        )

    Values are the MRD bit masks. The ``ismrmrd`` constants are 1-based bit
    *positions* rather than masks, which :attr:`position` converts back to, and
    :attr:`flag` gives the constant's name for anything that wants it spelled
    out. Defining them here rather than reading them from ``ismrmrd`` keeps
    this module importable without it; that they agree is a test.

    Examples
    --------
    >>> import pulserver.recon as recon
    >>> recon.AcquisitionFlag.LAST_IN_SLICE.position
    8
    >>> recon.AcquisitionFlag.IS_NOISE_MEASUREMENT.flag
    'ACQ_IS_NOISE_MEASUREMENT'

    Flags combine, which is how a plugin says which boundaries it wants
    ``receive`` woken on::

        super().__init__(
            split_on=AcquisitionFlag.LAST_IN_SEGMENT | AcquisitionFlag.LAST_IN_SLICE
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


@dataclass(frozen=True)
class ReconResult:
    """Array-level image for Pulserver to package and emit.

    Native ``ismrmrd.Image`` objects remain valid app outputs for Gadgetron/MRD
    users. ``ReconResult`` avoids header-building boilerplate when the runtime
    can inherit geometry from a reference acquisition.

    Parameters
    ----------
    data
        Image array or tensor.
    reference
        Acquisition index in ``bucket.data`` used for geometry and timing.
    series_index
        Output image-series index.
    image_index
        Explicit image index. ``None`` lets the runtime assign one.
    image_type
        ``"magnitude"``, ``"phase"``, ``"real"``, ``"imaginary"``, or
        ``"complex"``.
    attributes
        Additional MRD image attributes.
    dicom
        Convert the MRD image to DICOM before sending it inline.
    """

    data: Any
    reference: int = 0
    series_index: int = 0
    image_index: int | None = None
    image_type: str = "magnitude"
    attributes: Mapping[str, Any] = field(default_factory=dict)
    dicom: bool = False


class ExamCache(MutableMapping[Hashable, Any]):
    """Thread-safe artifact cache owned by one scanner exam.

    Values remain alive while the corresponding exam generation is leased by
    an active reconstruction. Use :meth:`get_or_create` for expensive
    calibration artifacts; its factory executes at most once per key.

    Parameters
    ----------
    exam_id
        Stable identifier for the exam owning this cache.
    """

    def __init__(self, exam_id: Hashable) -> None:
        self.exam_id = exam_id
        self._values: dict[Hashable, Any] = {}
        self._cleanups: dict[Hashable, Callable[[Any], None] | None] = {}
        self._lock = RLock()
        self._closed = False

    @property
    def closed(self) -> bool:
        """Whether this retired cache has released its values."""
        with self._lock:
            return self._closed

    def __getitem__(self, key: Hashable) -> Any:
        with self._lock:
            self._require_open()
            return self._values[key]

    def __setitem__(self, key: Hashable, value: Any) -> None:
        self.set(key, value)

    def __delitem__(self, key: Hashable) -> None:
        with self._lock:
            self._require_open()
            value = self._values.pop(key)
            cleanup = self._cleanups.pop(key)
        _dispose(value, cleanup)

    def __iter__(self) -> Iterator[Hashable]:
        with self._lock:
            self._require_open()
            return iter(tuple(self._values))

    def __len__(self) -> int:
        with self._lock:
            self._require_open()
            return len(self._values)

    def set(
        self,
        key: Hashable,
        value: Any,
        *,
        cleanup: Callable[[Any], None] | None = None,
    ) -> Any:
        """Store an artifact, disposing any value previously at ``key``.

        Artifact keys should include geometry, coil configuration, trajectory
        or basis identity, and calibration settings as appropriate. Sharing an
        exam does not by itself make sensitivity maps interchangeable.
        """
        previous: tuple[Any, Callable[[Any], None] | None] | None = None
        with self._lock:
            self._require_open()
            if key in self._values:
                previous = (self._values[key], self._cleanups[key])
            self._values[key] = value
            self._cleanups[key] = cleanup
        if previous is not None and previous[0] is not value:
            _dispose(*previous)
        return value

    def get_or_create(
        self,
        key: Hashable,
        factory: Callable[[], Any],
        *,
        cleanup: Callable[[Any], None] | None = None,
    ) -> Any:
        """Return ``key``, constructing it exactly once when absent.

        The factory runs under the cache lock. This serializes duplicate
        calibration requests instead of allocating two large GPU artifacts and
        discarding one.
        """
        with self._lock:
            self._require_open()
            if key not in self._values:
                self._values[key] = factory()
                self._cleanups[key] = cleanup
            return self._values[key]

    def pop(self, key: Hashable, default: Any = ...) -> Any:
        """Remove and return an artifact without disposing it.

        Ownership transfers to the caller. Use ``del cache[key]`` when the
        artifact should be disposed immediately.
        """
        with self._lock:
            self._require_open()
            if key not in self._values:
                if default is ...:
                    raise KeyError(key)
                return default
            self._cleanups.pop(key)
            return self._values.pop(key)

    def clear(self) -> None:
        """Dispose and remove every cached artifact."""
        with self._lock:
            values = tuple(
                (value, self._cleanups[key]) for key, value in self._values.items()
            )
            self._values.clear()
            self._cleanups.clear()
        for value, cleanup in values:
            _dispose(value, cleanup)

    def close(self) -> None:
        """Retire the cache and dispose all artifacts."""
        with self._lock:
            if self._closed:
                return
            values = tuple(
                (value, self._cleanups[key]) for key, value in self._values.items()
            )
            self._values.clear()
            self._cleanups.clear()
            self._closed = True
        for value, cleanup in values:
            _dispose(value, cleanup)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"exam cache {self.exam_id!r} is retired")


@dataclass(frozen=True)
class ReconContext:
    """Scan context passed to :meth:`ReconPlugin.recon`.

    ``header`` and ``config`` intentionally retain the names used by the
    Gadgetron Python connection. ``exam`` is Pulserver's only addition.

    Parameters
    ----------
    header
        Parsed MRD XML header online, or application metadata offline.
    exam
        Exam-scoped artifact cache.
    config
        Configuration payload selecting the application.
    """

    header: Any
    exam: ExamCache
    config: Any = None

    @classmethod
    def offline(
        cls,
        header: Any = None,
        *,
        exam_id: Hashable = "offline",
        config: Any = None,
    ) -> ReconContext:
        """Create a context for direct offline reconstruction."""
        return cls(header=header, exam=ExamCache(exam_id), config=config)

    @property
    def exam_id(self) -> Hashable:
        """Return the identifier of the active exam generation."""
        return self.exam.exam_id


class ReconPlugin(ABC):
    """Base class for Pulserver reconstruction plugins.

    A plugin module creates one configured instance named ``PLUGIN``, and the
    private inline runtime discovers it -- no registration or connection
    callback is required. Like a sequence plugin, a reconstruction plugin is a
    handful of lifecycle hooks the runtime drives over one MRD stream, and the
    division between them is the same in every plugin:

    :meth:`startup`
        Once, when the stream opens, before any acquisition. Lay out the
        buffers the header's encoding spaces describe. *Optional.*
    :meth:`receive`
        For every accepted acquisition, as it arrives. Place it in its buffer,
        and -- reading the acquisition's own flags and counters -- decide
        whether it closed something worth reconstructing and which branch
        reconstructs it. The sorting and the routing both live here, so they
        overlap acquisition dead time instead of waiting for a trigger.
        *Optional.*
    :meth:`recon`
        Whenever :meth:`receive` routes a branch to it, over buffers that are
        already filled. Holds the reconstruction of each branch and nothing
        else. **Required** -- it is the reconstruction.

    Only :meth:`recon` is mandatory. The default :meth:`startup` reads the
    header, the default :meth:`receive` places each acquisition and routes
    ``"imaging"`` at every ``split_on`` boundary, so a plugin with one branch
    overrides :meth:`recon` alone.

    Branches
    --------
    A branch is a name :meth:`receive` chooses and :meth:`recon` switches on --
    ``"calibration"`` and ``"imaging"`` in the scans that have two. A scan
    whose autocalibration block completes long before the slice does is the
    case this exists for::

        def receive(self, acquisition, context):
            self.buffers.add(acquisition)
            if has_acquisition_flag(acquisition, AcquisitionFlag.LAST_IN_SLICE):
                return self.recon("imaging", context)
            if has_acquisition_flag(acquisition, AcquisitionFlag.LAST_IN_SEGMENT):
                return self.recon("calibration", context)
            return None

        def recon(self, branch, context):
            if branch == "calibration":
                self.maps = sensitivities(self.buffers[0])   # no image yet
                return None
            ...                                             # the slice is complete

    Testing the slice before the segment is what makes that read correctly: the
    final acquisition of a slice closes its segment *and* the slice, so the
    calibration branch is reached only where nothing larger ended.

    Whatever :meth:`receive` returns is what the runtime emits, so a branch
    that produces no image simply returns ``None``, exactly as a data sink
    does.

    State
    -----
    Each stream reconstructs through its own instance, which :meth:`spawn`
    produces from the module-level ``PLUGIN`` -- so a buffer allocated in
    :meth:`startup` and filled in :meth:`receive` belongs on ``self``, and two
    concurrent scanner connections cannot see each other's. ``context.exam``
    is the separate, *exam*-scoped cache: it outlives the stream and is how
    successive sequences of one exam share an artifact. A plugin reads and adds
    to it, and never clears it.

    Calling the instance drives the same lifecycle offline over a ready bucket,
    so a plugin needs no second code path for data it did not stream.

    Buffers
    -------
    The header arrives before the data and describes every encoding space the
    scan will produce, one per subsequence, each with the extent of every
    counter that varies in it. Each acquisition then names its space and
    carries its own counters. So sorting needs nothing a plugin has to supply:
    :meth:`startup` lays the spaces out and :meth:`receive` places each
    acquisition as it lands, and :meth:`recon` finds sorted k-space in
    :attr:`buffers` rather than a list to sort.

    ``self.buffers[0].kspace`` is one space's array and ``.axes`` names its
    dimensions; ``.mask`` says where data actually landed, which is how an
    undersampled or partial-echo scan is read, and ``.reference`` which of
    those positions the scanner flagged as parallel-imaging calibration. A
    calibration acquired on its own geometry -- the low-resolution gradient
    echo an EPI scan calibrates from -- is a subsequence, so it is an encoding
    space of its own and reaches ``self.buffers[1]`` without a plugin routing
    it there.

    Parameters
    ----------
    split_on
        The boundary the default :meth:`receive` reconstructs at, as an
        :class:`AcquisitionFlag` (several may be combined with ``|``) or a
        named MRD acquisition flag. ``None`` reconstructs at end of stream.
    require_flags
        Flags every accepted acquisition must contain.
    reject_flags
        Flags that exclude an acquisition, which the runtime then never hands
        to :meth:`receive`.
    buffered
        Sort the acquisitions into :attr:`buffers` as they arrive. Turn it off
        for a plugin that must digest a header too thin to describe its data --
        a generic MRD handler taking streams from elsewhere -- which then
        collects the acquisitions in :meth:`receive` and sorts for itself.

    Attributes
    ----------
    split_on : tuple
        The flags as a tuple, whatever form they were given in. Empty when the
        stream reconstructs once, at its end.
    buffers : ReconData
        Every encoding space of the scan, filled as the acquisitions arrive.
        Laid out by :meth:`startup` from the header; empty until then, and
        empty for a bucket assembled from arrays, which carries no header to
        lay anything out from.
    """

    def __init__(
        self,
        *,
        split_on: int | str | tuple[int | str, ...] | None = "ACQ_LAST_IN_MEASUREMENT",
        require_flags: tuple[int | str, ...] = (),
        reject_flags: tuple[int | str, ...] = (),
        buffered: bool = True,
    ) -> None:
        self.split_on = _flags(split_on)
        self.require_flags = tuple(require_flags)
        self.reject_flags = tuple(reject_flags)
        self.buffered = bool(buffered)
        self.buffers = ReconData()

    def spawn(self) -> ReconPlugin:
        """Return the working instance one stream reconstructs through.

        A shallow copy, so anything expensive the configured plugin holds -- a
        loaded network, a compiled operator -- is shared rather than duplicated,
        while whatever the lifecycle hooks assign stays private to the stream.
        Override to isolate something a shallow copy would still share.
        """
        return copy.copy(self)

    def startup(self, context: ReconContext) -> None:
        """Lay the scan's buffers out, before the first acquisition.

        Runs once when the stream opens. The default reads every encoding space
        the header describes into :attr:`buffers`; nothing is allocated until
        an acquisition names a space, so the spaces this scan does not fill
        cost nothing.

        Override to add what the header cannot say -- a trajectory, a
        precomputed operator -- and call ``super().startup(context)`` to keep
        the buffers.
        """
        if self.buffered:
            self.buffers = ReconData.from_header(context.header)

    def receive(self, acquisition: Any, context: ReconContext) -> Any:
        """Place one acquisition, and reconstruct whatever it completed.

        Runs for every accepted acquisition, as it arrives, so both the sorting
        and the routing overlap acquisition dead time. Placement routes on
        ``encoding_space_ref`` and indexes by the acquisition's own counters,
        which is what the header laid the buffers out for. The routing then
        reads the same acquisition's flags: a boundary it closes selects the
        branch :meth:`recon` runs, and whatever that returns is returned from
        here for the runtime to emit.

        The default places the acquisition and routes ``"imaging"`` at every
        ``split_on`` boundary. Override it for a scan with more than one branch,
        or for work the placement cannot do -- an EPI line that must be phase
        corrected before it belongs anywhere, a running noise estimate.

        Parameters
        ----------
        acquisition
            The acquisition, with its data, counters and flags.
        context
            The scan context, as :meth:`startup` saw it.

        Returns
        -------
        object or None
            What :meth:`recon` produced, or ``None`` when this acquisition
            closed nothing.
        """
        if self.buffered:
            self.buffers.add(acquisition)
        if _closes(acquisition, self.split_on):
            return self.recon("imaging", context)
        return None

    @abstractmethod
    def recon(self, branch: str, context: ReconContext) -> Any:
        """Reconstruct one branch, over buffers :meth:`receive` has filled.

        Holds the reconstruction and nothing else: the acquisitions are already
        sorted, and which branch runs was decided by the hook that called this.

        Parameters
        ----------
        branch
            The branch :meth:`receive` routed -- ``"imaging"`` from the default
            routing, and whatever names a plugin gives its own.
        context
            The scan context, as :meth:`startup` saw it.

        Returns
        -------
        object or None
            A :class:`ReconResult`, a native MRD output, a sequence of either,
            or ``None`` for a branch that produces no image.
        """
        ...

    def run(
        self,
        path: str,
        *,
        group: str = "dataset",
        exam_id: Hashable | None = None,
        config: Any = None,
    ) -> list[Any]:
        """Reconstruct one MRD file, in this process.

        A file holds what the scan sent -- the header, then the acquisitions in
        acquisition order -- which is what a scanner connection delivers. So
        this opens the file and streams it to this plugin directly: same
        :meth:`startup`, same :meth:`receive` per acquisition, same
        :meth:`recon` at each boundary the flags mark. There is no second
        reconstruction path to keep in step, and no socket, port or server
        involved.

        Every plugin inherits this, and none overrides it: what a plugin does
        with a stream is already all in its hooks.

        Parameters
        ----------
        path
            An ISMRMRD HDF5 file.
        group
            The HDF5 group the scan was written under.
        exam_id
            Identifier for the exam-scoped artifact cache. Defaults to the
            path.
        config
            Configuration payload, as an inline connection would carry.

        Returns
        -------
        list
            Everything the reconstruction emitted, in order.

        Examples
        --------
        A plugin module is callable, and this is what it calls::

            from pulserver.app import cartesian2D_recon

            images = cartesian2D_recon("scan.h5")
        """
        from ._mrd.offline import reconstruct_file

        return reconstruct_file(self, path, group=group, exam_id=exam_id, config=config)

    def __call__(self, bucket: AcquisitionBucket, context: ReconContext) -> Any:
        """Reconstruct a ready bucket outside the inline server.

        Replays the bucket's acquisitions through the same lifecycle the
        runtime drives -- :meth:`startup`, then :meth:`receive` per
        acquisition, which routes the branches -- on a fresh :meth:`spawn`. A
        plugin therefore reconstructs assembled data exactly as it
        reconstructs streamed data, and this instance is left untouched.

        The bucket is one boundary, whatever number of them the stream would
        have had, so a plugin that branches several ways answers with the last.
        A bucket whose acquisitions close nothing -- one assembled from arrays,
        which carry no flags -- is reconstructed as ``"imaging"`` at its end.
        """
        plugin = self.spawn()
        plugin.startup(context)
        output = None
        for acquisition in bucket.acquisitions:
            received = plugin.receive(acquisition, context)
            if received is not None:
                output = received
        if output is None and not _closes(_last(bucket.acquisitions), plugin.split_on):
            output = plugin.recon("imaging", context)
        return output


# %% private module subroutines


def _closes(acquisition: Any, split_on: tuple[Any, ...]) -> bool:
    """Whether this acquisition carries one of the boundaries named."""
    if acquisition is None:
        return False
    return any(has_acquisition_flag(acquisition, flag) for flag in split_on)


def _last(acquisitions: tuple[Any, ...]) -> Any | None:
    """The acquisition that ended a bucket, or ``None`` for an empty one."""
    return acquisitions[-1] if acquisitions else None


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


def _flags(value: Any) -> tuple[Any, ...]:
    """Normalise one boundary, several, or none into a tuple of MRD flags.

    An :class:`AcquisitionFlag` may name several at once, and each becomes its
    own flag, so ``split_on=AcquisitionFlag.LAST_IN_SLICE |
    AcquisitionFlag.LAST_IN_SEGMENT`` ends a bucket at either.
    """
    if value is None:
        return ()
    if isinstance(value, AcquisitionFlag):
        return tuple(member.flag for member in AcquisitionFlag if member in value)
    if isinstance(value, (str, int)):
        return (value,)
    return tuple(
        member.flag if isinstance(member, AcquisitionFlag) else member
        for member in value
    )


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


def _dispose(value: Any, cleanup: Callable[[Any], None] | None) -> None:
    try:
        if cleanup is not None:
            cleanup(value)
            return
        close = getattr(value, "close", None)
        if callable(close):
            close()
    except Exception:
        logging.exception("Error disposing exam-cached artifact")
