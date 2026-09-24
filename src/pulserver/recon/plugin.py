"""Reconstruction plugin contract.

A plugin module defines a :class:`ReconPlugin` subclass and a module-level
``PLUGIN`` instance. The same hooks run over a live MRD stream, over an MRD
file (:meth:`ReconPlugin.run`) and over an assembled bucket (calling the
instance).

Examples
--------
>>> import numpy as np
>>> from types import SimpleNamespace
>>> from pulserver.recon import ReconContext, ReconPlugin, ReconResult
>>> from pulserver.mrd import AcquisitionBucket
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
    "ExamCache",
    "Gadget",
    "ReconBuffer",
    "ReconContext",
    "ReconData",
    "ReconPlugin",
    "ReconResult",
]

import contextlib
import copy
import hashlib
import logging
import os
import pickle
import threading
from abc import ABC, abstractmethod
from collections.abc import (
    Callable,
    Hashable,
    Iterator,
    Mapping,
    MutableMapping,
    Sequence,
)
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np

from ..mrd._acquisitions import AcquisitionBucket, AcquisitionFlag
from ..mrd._metadata import has_acquisition_flag
from ._buffers import ReconBuffer, ReconData


@dataclass(frozen=True)
class ReconResult:
    """Image array for the runtime to package as an MRD image.

    Geometry and timing come from a reference acquisition, so a plugin builds no
    image header. Plugins may return ``ismrmrd.Image`` objects instead.

    Parameters
    ----------
    data
        NumPy array or Torch tensor, on any device.
    reference
        Index into the imaging acquisitions of the unit being emitted; negative
        values count from the end, ``-1`` being the acquisition that closed it.
    series_index
        MRD ``image_series_index``.
    image_index
        MRD ``image_index``; ``None`` numbers images consecutively.
    image_type
        ``"magnitude"``, ``"phase"``, ``"real"``, ``"imaginary"`` or
        ``"complex"``.
    attributes
        MRD meta attributes, merged over the runtime's defaults.
    dicom
        Convert the image to DICOM before sending it.

    Examples
    --------
    >>> import numpy as np
    >>> import pulserver.recon as recon
    >>> result = recon.ReconResult(np.zeros((4, 4)), series_index=2)
    >>> result.data.shape, result.series_index, result.image_type
    ((4, 4), 2, 'magnitude')
    """

    data: Any
    reference: int = 0
    series_index: int = 0
    image_index: int | None = None
    image_type: str = "magnitude"
    attributes: Mapping[str, Any] = field(default_factory=dict)
    dicom: bool = False


class ExamCache(MutableMapping[Hashable, Any]):
    """Thread-safe store of artifacts shared by the series of one exam.

    A value is disposed when replaced by another object, deleted, cleared, or
    when the cache closes: through the ``cleanup`` stored with it, else its
    ``close()`` method when it has one. Disposal errors are logged, not raised.
    Reads and writes after :meth:`close` raise ``RuntimeError``.

    With a ``directory``, every value stored is also written there with
    pickle, and a key this cache does not hold is read from there, so the
    caches of one exam in different processes share their values. A value
    read back is a copy, without the ``cleanup`` stored with it; a key or value
    that cannot be pickled is held by this cache alone. Closing leaves the
    directory as it is.

    Parameters
    ----------
    exam_id
        Identifier of the owning exam.
    directory
        Directory the exam's caches share; ``None`` holds values in memory
        only. Its files are unpickled, so only the processes sharing the exam
        may write to it: the proxy creates it under a directory only its own
        user can open.

    Examples
    --------
    >>> import pulserver.recon as recon
    >>> cache = recon.ExamCache("exam-1")
    >>> cache["coil_maps"] = "maps"
    >>> "coil_maps" in cache, len(cache)
    (True, 1)
    """

    def __init__(self, exam_id: Hashable, directory: Path | str | None = None) -> None:
        self.exam_id = exam_id
        self.directory = None if directory is None else Path(directory)
        self._values: dict[Hashable, Any] = {}
        self._cleanups: dict[Hashable, Callable[[Any], None] | None] = {}
        self._lock = RLock()
        self._closed = False

    @property
    def closed(self) -> bool:
        """Whether :meth:`close` has run."""
        with self._lock:
            return self._closed

    def __getitem__(self, key: Hashable) -> Any:
        with self._lock:
            self._require_open()
            if key not in self._values:
                self._values[key] = self._read(key)
                self._cleanups[key] = None
            return self._values[key]

    def __setitem__(self, key: Hashable, value: Any) -> None:
        self.set(key, value)

    def __contains__(self, key: object) -> bool:
        with self._lock:
            self._require_open()
            return key in self._values or self._written(key)

    def __delitem__(self, key: Hashable) -> None:
        with self._lock:
            self._require_open()
            written = self._forget(key)
            if key not in self._values:
                if not written:
                    raise KeyError(key)
                return
            value = self._values.pop(key)
            cleanup = self._cleanups.pop(key)
        _dispose(value, cleanup)

    def __iter__(self) -> Iterator[Hashable]:
        with self._lock:
            self._require_open()
            return iter(tuple(dict.fromkeys((*self._values, *self._written_keys()))))

    def __len__(self) -> int:
        with self._lock:
            self._require_open()
            return len(dict.fromkeys((*self._values, *self._written_keys())))

    def set(
        self,
        key: Hashable,
        value: Any,
        *,
        cleanup: Callable[[Any], None] | None = None,
    ) -> Any:
        """Store an artifact and return it, disposing the value it replaces.

        A key should identify everything the artifact depends on -- geometry, coil
        configuration, trajectory, calibration settings: sharing an exam does not
        make sensitivity maps interchangeable.

        Parameters
        ----------
        cleanup
            Called with the value when it is disposed, instead of its ``close()``.
        """
        previous: tuple[Any, Callable[[Any], None] | None] | None = None
        with self._lock:
            self._require_open()
            if key in self._values:
                previous = (self._values[key], self._cleanups[key])
            self._values[key] = value
            self._cleanups[key] = cleanup
            self._write(key, value)
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
        """Return the artifact at ``key``, calling ``factory`` to create it when absent.

        The factory runs under the cache lock: concurrent callers create a key once,
        and every other access waits until the factory returns.
        """
        with self._lock:
            self._require_open()
            if key not in self._values and self._written(key):
                self._values[key] = self._read(key)
                self._cleanups[key] = None
            elif key not in self._values:
                self._values[key] = factory()
                self._cleanups[key] = cleanup
                self._write(key, self._values[key])
            return self._values[key]

    def pop(self, key: Hashable, default: Any = ...) -> Any:
        """Remove and return an artifact without disposing it.

        Ownership transfers to the caller. Use ``del cache[key]`` when the
        artifact should be disposed immediately.
        """
        with self._lock:
            self._require_open()
            if key not in self._values and not self._written(key):
                if default is ...:
                    raise KeyError(key)
                return default
            value = self._values.pop(key) if key in self._values else self._read(key)
            self._cleanups.pop(key, None)
            self._forget(key)
            return value

    def clear(self) -> None:
        """Dispose and remove every cached artifact, the directory's included."""
        with self._lock:
            values = tuple(
                (value, self._cleanups[key]) for key, value in self._values.items()
            )
            self._values.clear()
            self._cleanups.clear()
            if not self._closed:
                for key in self._written_keys():
                    self._forget(key)
        for value, cleanup in values:
            _dispose(value, cleanup)

    def close(self) -> None:
        """Dispose every artifact and retire the cache; repeated calls do nothing."""
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

    # A key is written as ``<digest>.value`` and then ``<digest>.key``, so a
    # key file names a value that is complete.

    def _entry(self, key: object) -> Path | None:
        """Return the path of the key's files without their suffix, or ``None``."""
        if self.directory is None:
            return None
        try:
            digest = hashlib.sha256(pickle.dumps(key, pickle.HIGHEST_PROTOCOL))
        except Exception:
            return None
        return self.directory / digest.hexdigest()

    def _written(self, key: object) -> bool:
        entry = self._entry(key)
        return entry is not None and entry.with_suffix(".key").is_file()

    def _written_keys(self) -> list[Hashable]:
        if self.directory is None or not self.directory.is_dir():
            return []
        keys = []
        for path in sorted(self.directory.glob("*.key")):
            with contextlib.suppress(OSError, EOFError, pickle.UnpicklingError):
                # Only the processes sharing the exam write here; see ``directory``.
                keys.append(pickle.loads(path.read_bytes()))  # noqa: S301  # nosec B301
        return keys

    def _read(self, key: Hashable) -> Any:
        entry = self._entry(key)
        if entry is None or not entry.with_suffix(".key").is_file():
            raise KeyError(key)
        try:
            payload = entry.with_suffix(".value").read_bytes()
        except FileNotFoundError:
            raise KeyError(key) from None
        return pickle.loads(payload)  # noqa: S301  # nosec B301

    def _write(self, key: Hashable, value: Any) -> None:
        if self.directory is None:
            return
        entry = self._entry(key)
        try:
            if entry is None:
                raise TypeError("the key cannot be pickled")
            payload = pickle.dumps(value, pickle.HIGHEST_PROTOCOL)
        except Exception:
            logging.warning(
                "exam cache: %r cannot be pickled and stays with this series", key
            )
            self._forget(key)
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        _replace(entry.with_suffix(".value"), payload)
        _replace(entry.with_suffix(".key"), pickle.dumps(key, pickle.HIGHEST_PROTOCOL))

    def _forget(self, key: object) -> bool:
        """Delete the key's files; return whether it was written."""
        entry = self._entry(key)
        if entry is None:
            return False
        written = entry.with_suffix(".key").is_file()
        for suffix in (".key", ".value"):
            with contextlib.suppress(FileNotFoundError):
                entry.with_suffix(suffix).unlink()
        return written


@dataclass(frozen=True)
class ReconContext:
    """Scan context passed to every hook of a plugin.

    Parameters
    ----------
    header
        Parsed MRD XML header; offline, ``None`` or any header-like object.
    exam
        Artifact cache shared by the series of the exam.
    config
        Configuration payload the client sent with the stream.
    device
        The GPU the proxy gave this series, as a torch device such as
        ``"cuda:0"``; ``None`` on the host, and offline.

    Examples
    --------
    >>> from types import SimpleNamespace
    >>> import pulserver.recon as recon
    >>> matrix = SimpleNamespace(matrixSize=SimpleNamespace(x=8, y=4, z=1))
    >>> header = SimpleNamespace(
    ...     encoding=[SimpleNamespace(encodedSpace=matrix, reconSpace=matrix)],
    ...     acquisitionSystemInformation=SimpleNamespace(receiverChannels=2),
    ... )
    >>> context = recon.ReconContext.offline(header)
    >>> isinstance(context.exam, recon.ExamCache)
    True
    """

    header: Any
    exam: ExamCache
    config: Any = None
    device: str | None = None

    @classmethod
    def offline(
        cls,
        header: Any = None,
        *,
        exam_id: Hashable = "offline",
        config: Any = None,
    ) -> ReconContext:
        """Create a context with a new :class:`ExamCache`."""
        return cls(header=header, exam=ExamCache(exam_id), config=config)

    @property
    def exam_id(self) -> Hashable:
        """Identifier of :attr:`exam`."""
        return self.exam.exam_id


class Gadget(ABC):
    """One per-acquisition step of a plugin's ``chain``.

    A gadget keeps what it learns from earlier acquisitions of its stream -- a
    noise covariance, a coil basis -- on ``self``. :meth:`ReconPlugin.spawn`
    copies the chain, so each stream has its own.

    Attributes
    ----------
    context : ReconContext
        Set by :meth:`startup`.

    Examples
    --------
    >>> import pulserver.mrd as mrd
    >>> import pulserver.recon as recon
    >>> class DropNavigators(recon.Gadget):
    ...     def __call__(self, acquisition, data):
    ...         if mrd.has_acquisition_flag(acquisition, "ACQ_IS_NAVIGATION_DATA"):
    ...             return None
    ...         return data
    >>> isinstance(DropNavigators(), recon.Gadget)
    True
    """

    def startup(self, context: Any) -> None:
        """Prepare for one stream; the default stores ``context``, so overrides call it."""
        self.context = context

    @abstractmethod
    def __call__(self, acquisition: Any, data: Any) -> Any:
        """Return the readout as the next step should see it.

        Parameters
        ----------
        acquisition
            The acquisition, for its flags and counters.
        data
            ``(coils, samples)``: the acquisition's data, or the previous step's
            output.

        Returns
        -------
        ndarray or None
            The readout, or ``None`` to consume the acquisition.
        """


class ReconPlugin(ABC):
    """Base class for reconstruction plugins.

    The runtime drives three hooks over one MRD stream: :meth:`startup` once,
    before any acquisition; :meth:`receive` for each accepted acquisition, as it
    arrives; and :meth:`recon` for each branch :meth:`receive` routes. Only
    :meth:`recon` is abstract. The default :meth:`receive` runs the ``chain``,
    places the readout in :attr:`buffers` and routes by ``branches``, so a
    plugin usually declares those two and writes only :meth:`recon`. When a
    stream ends on acquisitions that closed no branch, :meth:`recon` runs once
    more with ``"imaging"``.

    Each stream runs on its own :meth:`spawn` of the module-level ``PLUGIN``, so
    state set in the hooks belongs to one stream. ``context.exam`` is shared
    across the series of an exam; a plugin adds to it and never clears it.

    Parameters
    ----------
    chain
        :class:`Gadget` steps applied to every readout on arrival, in order.
    branches
        ``{AcquisitionFlag: name}``, tried in order: the first flag the
        acquisition carries names the branch. List larger units first, since the
        last acquisition of a slice also closes its segment. Flags combined with
        ``|`` route either one to the branch. The default routes
        ``LAST_IN_MEASUREMENT`` to ``"imaging"``.
    require_flags
        Flags an acquisition must all carry to be accepted. A combined
        :class:`AcquisitionFlag` counts as its members.
    reject_flags
        Flags any one of which excludes an acquisition. The runtime never passes
        excluded acquisitions to :meth:`receive`.
    buffered
        Place acquisitions into :attr:`buffers`. Disable for streams whose header
        does not describe their encoding spaces; the plugin then collects
        acquisitions itself.

    Attributes
    ----------
    buffers : ReconData
        Every encoding space of the scan, laid out by :meth:`startup`.
    acquisition : object
        The last acquisition the chain passed -- the one that closed the branch
        :meth:`recon` is running -- or ``None``.

    Examples
    --------
    >>> import numpy as np
    >>> import pulserver.mrd as mrd
    >>> import pulserver.recon as recon
    >>> class DropNoise(recon.Gadget):
    ...     def __call__(self, acquisition, data):
    ...         noise = mrd.has_acquisition_flag(acquisition, "ACQ_IS_NOISE_MEASUREMENT")
    ...         return None if noise else data
    >>> class RootSumOfSquares(recon.ReconPlugin):
    ...     def __init__(self):
    ...         super().__init__(
    ...             chain=[DropNoise()],
    ...             branches={mrd.AcquisitionFlag.LAST_IN_SLICE: "imaging"},
    ...         )
    ...     def recon(self, branch, context):
    ...         kspace = self.buffers[0].kspace
    ...         return recon.ReconResult(np.sqrt(np.sum(np.abs(kspace) ** 2, axis=0)))
    >>> RootSumOfSquares().branches[mrd.AcquisitionFlag.LAST_IN_SLICE]
    'imaging'
    """

    def __init__(
        self,
        *,
        chain: Sequence[Any] = (),
        branches: Mapping[Any, str] | None = None,
        require_flags: tuple[int | str, ...] = (),
        reject_flags: tuple[int | str, ...] = (),
        buffered: bool = True,
    ) -> None:
        self.chain = tuple(chain)
        self.branches = dict(
            {AcquisitionFlag.LAST_IN_MEASUREMENT: "imaging"}
            if branches is None
            else branches
        )
        self.require_flags = _flag_members(require_flags)
        self.reject_flags = _flag_members(reject_flags)
        self.buffered = bool(buffered)
        self.buffers = ReconData()
        self.acquisition: Any = None

    def spawn(self) -> ReconPlugin:
        """Return the instance one stream runs on.

        A shallow copy holding its own copy of each gadget, so resources the
        configured plugin holds -- a loaded network, a compiled operator -- are
        shared. Override to isolate anything else a shallow copy would share.
        """
        plugin = copy.copy(self)
        plugin.chain = tuple(copy.copy(gadget) for gadget in self.chain)
        return plugin

    def startup(self, context: ReconContext) -> None:
        """Start every gadget, then lay out :attr:`buffers` from the header.

        No buffer is allocated until an acquisition names its space. Overrides call
        ``super().startup(context)``.
        """
        for gadget in self.chain:
            gadget.startup(context)
        if self.buffered:
            self.buffers = ReconData.from_header(context.header)

    def process(self, acquisition: Any, data: Any = None) -> Any:
        """Run the chain over one readout and return what it left.

        Parameters
        ----------
        acquisition
            The acquisition, for its flags and counters.
        data
            The readout to start from; ``None`` takes ``acquisition.data``.

        Returns
        -------
        ndarray or None
            The corrected readout, or ``None`` when a step consumed it.
        """
        if data is None:
            data = np.asarray(acquisition.data)
        for gadget in self.chain:
            data = gadget(acquisition, data)
            if data is None:
                return None
        return data

    def gadget(self, kind: type) -> Any:
        """Return this stream's first gadget of type ``kind``.

        Hooks reach gadgets through this rather than through ``PLUGIN``, whose chain
        :meth:`spawn` copied.

        Raises
        ------
        LookupError
            If the chain holds no gadget of that type.
        """
        for gadget in self.chain:
            if isinstance(gadget, kind):
                return gadget
        raise LookupError(f"this plugin's chain has no {kind.__name__}")

    def branch_for(self, acquisition: Any) -> str | None:
        """Return the branch the acquisition closes, or ``None``; see ``branches``."""
        if acquisition is None:
            return None
        for flag, branch in self.branches.items():
            if _closes(acquisition, flag):
                return branch
        return None

    def receive(self, acquisition: Any, context: ReconContext) -> Any:
        """Run the chain on one acquisition, place it, and run the branch it closes.

        Placement follows ``encoding_space_ref`` and the acquisition's counters.
        Override for placement the default cannot express, calling :meth:`process`
        so the chain still runs.

        Returns
        -------
        object or None
            What :meth:`recon` returned, or ``None`` when the chain consumed the
            acquisition or it closed no branch.
        """
        data = self.process(acquisition)
        if data is None:
            return None
        self.acquisition = acquisition
        if self.buffered:
            self.buffers.add(acquisition, data)
        branch = self.branch_for(acquisition)
        return None if branch is None else self.recon(branch, context)

    @abstractmethod
    def recon(self, branch: str, context: ReconContext) -> Any:
        """Reconstruct one branch from the filled buffers.

        Parameters
        ----------
        branch
            The branch name :meth:`receive` routed.
        context
            The scan context.

        Returns
        -------
        object or None
            A :class:`ReconResult`, an ``ismrmrd`` output, an array (a magnitude
            :class:`ReconResult`), a sequence of these, or ``None``.
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
        """Reconstruct one ISMRMRD HDF5 file in this process, through the same hooks.

        The file's waveforms are delivered before its acquisitions.

        Parameters
        ----------
        path
            ISMRMRD HDF5 file.
        group
            HDF5 group holding the scan.
        exam_id
            Identifier of the exam cache; ``path`` when not given.
        config
            Configuration payload, as a stream would carry it.

        Returns
        -------
        list
            Everything emitted, in order: MRD images, named DICOM datasets for
            results with ``dicom=True``, and any other output the plugin returned.

        Raises
        ------
        FileNotFoundError
            If there is no such file.
        ValueError
            If the file has no MRD XML header.
        """
        from ._runtime.offline import reconstruct_file

        return reconstruct_file(self, path, group=group, exam_id=exam_id, config=config)

    def __call__(self, bucket: AcquisitionBucket, context: ReconContext) -> Any:
        """Reconstruct an assembled bucket on a new :meth:`spawn`.

        Runs :meth:`startup`, then :meth:`receive` for every acquisition in arrival
        order, and returns the last output that was not ``None``. When there was
        none and the last acquisition closes no branch, returns :meth:`recon` with
        ``"imaging"``. ``require_flags`` and ``reject_flags`` are not applied.
        """
        plugin = self.spawn()
        plugin.startup(context)
        output = None
        for acquisition in bucket.acquisitions:
            received = plugin.receive(acquisition, context)
            if received is not None:
                output = received
        if output is None and plugin.branch_for(_last(bucket.acquisitions)) is None:
            output = plugin.recon("imaging", context)
        return output


# %% private module subroutines


def _closes(acquisition: Any, flag: Any) -> bool:
    """Whether this acquisition carries the boundary named.

    A combined :class:`AcquisitionFlag` names several at once, and carrying any
    of them closes the branch it was mapped to.
    """
    if isinstance(flag, AcquisitionFlag):
        return any(
            has_acquisition_flag(acquisition, member.flag)
            for member in AcquisitionFlag
            if member in flag
        )
    return has_acquisition_flag(acquisition, flag)


def _flag_members(flags: Any) -> tuple[Any, ...]:
    """Return flags as a tuple, a combined :class:`AcquisitionFlag` split into its members.

    The members are read from the class because iterating a combined ``Flag``
    needs Python 3.11.
    """
    if isinstance(flags, AcquisitionFlag):
        return tuple(member for member in AcquisitionFlag if member in flags)
    return tuple(flags)


def _last(acquisitions: tuple[Any, ...]) -> Any | None:
    """Return the acquisition that ended a bucket, or ``None`` for an empty one."""
    return acquisitions[-1] if acquisitions else None


def _replace(path: Path, payload: bytes) -> None:
    """Write a file whole, so a reader in another process never sees part of it."""
    partial = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}")
    partial.write_bytes(payload)
    partial.replace(path)


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
