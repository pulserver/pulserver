"""Public contract for offline and inline reconstruction applications.

A reconstruction plugin is a regular Python module containing one
:class:`ReconApp` subclass and a module-level ``PLUGIN`` instance. The same
``recon`` method can be called directly for offline work or driven, alongside
the optional ``startup``/``receive``/``finalize`` lifecycle hooks, by
Pulserver's private MRD runtime for inline reconstruction.

The data model keeps Gadgetron's familiar acquisition-bucket vocabulary while
adding array conveniences for users concerned only with reconstruction. MRD
connections, framing, close handling, and output serialization remain private.

Examples
--------
>>> import numpy as np
>>> from pulserver import AcquisitionBucket, ReconApp, ReconContext, ReconResult
>>> class RootSumOfSquares(ReconApp):
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
    "ReconApp",
    "ReconContext",
    "ReconResult",
]

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Iterator, Mapping, MutableMapping
from dataclasses import dataclass, field
from threading import RLock
from types import SimpleNamespace
from typing import Any

import numpy as np


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
    """Scan context passed to :meth:`ReconApp.recon`.

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


class ReconApp(ABC):
    """Base class for Pulserver reconstruction plugins.

    A plugin module creates one configured instance named ``PLUGIN``, and the
    private inline runtime discovers it -- no registration or connection
    callback is required. Like a sequence plugin, an app is a handful of
    lifecycle hooks the runtime drives over one MRD stream:

    :meth:`startup`
        Once, when the stream opens, before any acquisition. Prepare
        exam-scoped state. *Optional.*
    :meth:`receive`
        For every accepted acquisition, as it arrives. Do the per-acquisition
        work -- filtering, sorting, gridding -- here, so it overlaps
        acquisition dead time instead of waiting for the trigger. *Optional.*
    :meth:`recon`
        Once per bucket, at the ``split_on`` boundary, on the Gadgetron-style
        :class:`AcquisitionBucket` the runtime has assembled. Produce the
        images. **Required** -- it is the reconstruction.
    :meth:`finalize`
        Once, when the stream closes, after the last bucket. Emit any trailing
        output and release exam-scoped state. *Optional.*

    Only :meth:`recon` is mandatory; the default hooks do nothing, so an app
    that grids at trigger time overrides :meth:`recon` alone. Calling the
    instance runs :meth:`recon` directly for offline work on a ready bucket.

    ``ReconApp`` instances may serve concurrent scanner connections. Keep their
    attributes immutable after construction and store mutable, exam-specific
    state in ``context.exam``, which is where the streaming hooks hand work to
    one another.

    Parameters
    ----------
    split_on
        Named acquisition flag ending one bucket. ``None`` produces one bucket
        at end of stream.
    require_flags
        Flags every acquisition entering the bucket must contain.
    reject_flags
        Flags that exclude an acquisition. A rejected acquisition may still
        end a bucket when it contains ``split_on``.
    """

    def __init__(
        self,
        *,
        split_on: int | str | None = "ACQ_LAST_IN_MEASUREMENT",
        require_flags: tuple[int | str, ...] = (),
        reject_flags: tuple[int | str, ...] = (),
    ) -> None:
        self.split_on = split_on
        self.require_flags = tuple(require_flags)
        self.reject_flags = tuple(reject_flags)

    def startup(self, context: ReconContext) -> None:
        """Prepare exam-scoped state before the first acquisition arrives.

        Runs once when the stream opens. The default does nothing; override to
        allocate buffers or load a reusable calibration into ``context.exam``.
        """
        del context

    def receive(self, acquisition: Any, context: ReconContext) -> None:
        """Fold one acquisition into the reconstruction as it arrives.

        Runs for every accepted acquisition, before its bucket is complete, so
        per-acquisition work -- filtering, sorting, gridding -- overlaps
        acquisition dead time rather than waiting for the trigger. State
        belongs in ``context.exam``. The default does nothing, leaving
        :meth:`recon` to do the work at the trigger from the assembled bucket.
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

    def finalize(self, context: ReconContext) -> Any:
        """Emit any trailing output once the stream has closed.

        Runs once after the last bucket. The default returns ``None``; override
        to flush an aggregate result or release exam-scoped state. The return
        value is emitted exactly as :meth:`recon`'s is.
        """
        del context
        return None

    def __call__(self, bucket: AcquisitionBucket, context: ReconContext) -> Any:
        """Run :meth:`recon` directly outside the inline server, on a ready bucket."""
        return self.recon(bucket, context)


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


def _labels_at(labels: Mapping[str, Any], index: int) -> dict[str, int]:
    return {name: int(value[index]) for name, value in labels.items()}


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
