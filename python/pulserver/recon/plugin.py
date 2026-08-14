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

Examples
--------
>>> import numpy as np
>>> from pulserver import AcquisitionBucket, ReconPlugin, ReconContext, ReconResult
>>> class RootSumOfSquares(ReconPlugin):
...     def recon(self, bucket, context):
...         del context
...         kspace = bucket.kspace()
...         image = np.sqrt(np.sum(np.abs(kspace) ** 2, axis=1))
...         return ReconResult(image)
>>> bucket = AcquisitionBucket.from_arrays(
...     np.ones((4, 2, 8), dtype=np.complex64)
... )
>>> result = RootSumOfSquares()(bucket, ReconContext.offline())
>>> result.data.shape
(4, 8)
"""

from __future__ import annotations

__all__ = [
    "AcquisitionBucket",
    "AcquisitionBucketStats",
    "ExamCache",
    "ReconContext",
    "ReconPlugin",
    "ReconResult",
    "has_acquisition_flag",
]

import copy
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Iterator, Mapping, MutableMapping
from dataclasses import dataclass, field
from threading import RLock
from types import SimpleNamespace
from typing import Any

import numpy as np

from ._mrd.metadata import has_acquisition_flag


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
        """Derive the arrival order from the classified views when unset."""
        if self.acquisitions:
            return
        extra = tuple(
            acquisition
            for acquisition in self.ref
            if not any(acquisition is item for item in self.data)
        )
        object.__setattr__(self, "acquisitions", self.data + extra)

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
            reference_values = {} if reference_labels is None else dict(reference_labels)
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
    handful of lifecycle hooks the runtime drives over one MRD stream:

    :meth:`startup`
        Once, when the stream opens, before any acquisition. Allocate the
        buffers the scan will fill. *Optional.*
    :meth:`receive`
        For every accepted acquisition, as it arrives. Do the per-acquisition
        work -- filtering, sorting, placing -- here, so it overlaps
        acquisition dead time instead of waiting for the trigger. *Optional.*
    :meth:`recon`
        Once per bucket, at each ``split_on`` boundary, on the Gadgetron-style
        :class:`AcquisitionBucket` the runtime has assembled. Produce the
        images. **Required** -- it is the reconstruction.

    Only :meth:`recon` is mandatory; the default hooks do nothing, so a plugin
    that does everything at trigger time overrides :meth:`recon` alone.

    Triggers
    --------
    ``split_on`` may name several flags rather than one, and then
    :meth:`recon` runs at each of them and decides for itself what that
    boundary meant. A scan whose autocalibration block completes long before
    the slice does is the case this exists for: the earlier trigger arrives
    with the calibration data in hand, so :meth:`recon` estimates the coil
    sensitivities and returns ``None``, and the later one reconstructs the
    image. :func:`has_acquisition_flag` on the last acquisition of the bucket
    is what tells them apart.

    Nothing forces the split to mean anything in particular: a bucket that
    produces no image simply returns ``None``, exactly as a data sink does.

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
    Acquisitions the scanner marks as calibration are usually imaging data too,
    and then one buffer holds everything: the autocalibration region is its
    fully sampled centre, and a calibration reads a portion of it. A sequence
    whose calibration is genuinely excluded from the imaging k-space is the
    other arrangement -- allocate two buffers in :meth:`startup`, route each
    acquisition in :meth:`receive` on ``ACQ_IS_PARALLEL_CALIBRATION``, and
    select between them by which trigger fired.

    Parameters
    ----------
    split_on
        Named acquisition flag ending one bucket, or several of them. ``None``
        produces one bucket at end of stream.
    require_flags
        Flags every acquisition entering the bucket must contain.
    reject_flags
        Flags that exclude an acquisition. A rejected acquisition may still
        end a bucket when it contains one of ``split_on``.

    Attributes
    ----------
    split_on : tuple
        The flags as a tuple, whatever form they were given in. Empty when the
        stream is one bucket.
    """

    def __init__(
        self,
        *,
        split_on: int | str | tuple[int | str, ...] | None = "ACQ_LAST_IN_MEASUREMENT",
        require_flags: tuple[int | str, ...] = (),
        reject_flags: tuple[int | str, ...] = (),
    ) -> None:
        self.split_on = _flags(split_on)
        self.require_flags = tuple(require_flags)
        self.reject_flags = tuple(reject_flags)

    def spawn(self) -> ReconPlugin:
        """Return the working instance one stream reconstructs through.

        A shallow copy, so anything expensive the configured plugin holds -- a
        loaded network, a compiled operator -- is shared rather than duplicated,
        while whatever the lifecycle hooks assign stays private to the stream.
        Override to isolate something a shallow copy would still share.
        """
        return copy.copy(self)

    def startup(self, context: ReconContext) -> None:
        """Allocate what the scan will fill, before the first acquisition.

        Runs once when the stream opens. The default does nothing; override to
        size buffers from ``context.header`` and put them on ``self``.
        """
        del context

    def receive(self, acquisition: Any, context: ReconContext) -> None:
        """Fold one acquisition into the reconstruction as it arrives.

        Runs for every accepted acquisition, before its bucket is complete, so
        per-acquisition work -- filtering, sorting, placing -- overlaps
        acquisition dead time rather than waiting for the trigger. The default
        does nothing, leaving :meth:`recon` to do the work at the trigger from
        the assembled bucket.
        """
        del acquisition, context

    @abstractmethod
    def recon(
        self,
        bucket: AcquisitionBucket,
        context: ReconContext,
    ) -> Any:
        """Reconstruct one acquisition bucket at its ``split_on`` boundary.

        Return a :class:`ReconResult`, a native MRD output, a sequence of
        either, or ``None`` for a data sink.
        """
        ...

    def __call__(self, bucket: AcquisitionBucket, context: ReconContext) -> Any:
        """Reconstruct a ready bucket outside the inline server.

        Replays the bucket's acquisitions through the same lifecycle the
        runtime drives -- :meth:`startup`, then :meth:`receive` per
        acquisition, then :meth:`recon` -- on a fresh :meth:`spawn`. A plugin
        therefore reconstructs assembled data exactly as it reconstructs
        streamed data, and this instance is left untouched.

        The bucket is one boundary, whatever number of them the stream would
        have had, so a plugin that splits several ways sees only the last.
        """
        plugin = self.spawn()
        plugin.startup(context)
        for acquisition in bucket.acquisitions:
            plugin.receive(acquisition, context)
        return plugin.recon(bucket, context)


# %% private module subroutines


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
    """Normalise one flag, several, or none into a tuple."""
    if value is None:
        return ()
    if isinstance(value, (str, int)):
        return (value,)
    return tuple(value)


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
