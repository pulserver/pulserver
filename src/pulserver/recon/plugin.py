"""Reconstruction plugin contract.

A plugin module defines a :class:`ReconPlugin` subclass and a module-level
``PLUGIN`` instance. The same hooks run over a live MRD stream, over an MRD
file (:meth:`ReconPlugin.run`) and over an assembled bucket (calling the
instance). The plugin writes :meth:`ReconPlugin.recon`, which is given each
reconstruction unit as it closes.

Examples
--------
>>> import numpy as np
>>> from types import SimpleNamespace
>>> from pulserver.recon import ReconContext, ReconPlugin, ReconResult
>>> from pulserver.mrd import AcquisitionBucket
>>> class RootSumOfSquares(ReconPlugin):
...     def recon(self, context, branch, data):
...         del context, branch
...         return ReconResult(np.sqrt(np.sum(np.abs(data.data.kspace) ** 2, axis=0)))
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
import warnings
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
from typing import Any, final

import numpy as np

from ..mrd._acquisitions import AcquisitionBucket, AcquisitionFlag
from ..mrd._header import LOOP_COUNTERS, EncodingSpace
from ..mrd._metadata import has_acquisition_flag
from ._buffers import ReconBuffer, ReconData, ReconUnit
from ._units import UnitKey, _Closure, _FlagClosure, unit_key


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


#: What a calibration scan leaves in the exam cache for the series that follow.
#:
#: A map is measured once and read by every later series of the exam, so the
#: two have to mean the same thing by it. These are the names they agree on;
#: a plugin storing a map under one of its own is storing it for itself alone.
#:
#: ``B0_MAP`` is off-resonance in Hz, ``B1_MAP`` the transmit field as a
#: fraction of what was asked for, and ``COIL_SENSITIVITIES`` the receive
#: sensitivity of each coil. Each is stored in the frame it was measured in;
#: a series at another prescription resamples it.
B0_MAP = "b0_map"
B1_MAP = "b1_map"
COIL_SENSITIVITIES = "coil_sensitivities"

#: The names a :class:`ReconContext` reads and writes as attributes of itself.
#: Each is its own key in the exam cache, so a hook reaching one by attribute
#: and a hook reaching it through :attr:`ReconContext.exam` reach the same map.
EXAM_ARTIFACTS = (B0_MAP, B1_MAP, COIL_SENSITIVITIES)


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


@dataclass
class ReconContext:
    """Scan context passed to every hook of a plugin.

    The maps the series of an exam share are attributes of the context: a hook
    reads :attr:`b0_map`, :attr:`b1_map` and :attr:`coil_sensitivities` from it
    and assigns to them what it measures, and a later series of the exam reads
    what was assigned, whichever proxy reconstructs it. Each is ``None`` until
    some series of the exam has measured it.

    Those three names are the whole vocabulary, so a misspelt one raises
    instead of being stored where nothing looks for it. A plugin carrying an
    artifact of its own puts it in :attr:`exam` under a key it chooses.

    The scan context itself does not change once built: assigning to
    :attr:`header`, :attr:`exam`, :attr:`config` or :attr:`device` raises.

    Parameters
    ----------
    header
        Parsed MRD XML header; offline, ``None`` or any header-like object.
    exam
        Artifact cache shared by the series of the exam, for artifacts the
        three names do not cover.
    config
        Configuration payload the client sent with the stream.
    device
        The GPU the proxy gave this series, as a torch device such as
        ``"cuda:0"``; ``None`` on the host, and offline.

    Examples
    --------
    >>> import pulserver.recon as recon
    >>> context = recon.ReconContext.offline()
    >>> context.b1_map is None
    True
    >>> context.b1_map = [1.0, 0.9]
    >>> context.b1_map
    [1.0, 0.9]
    >>> context.exam[recon.B1_MAP]
    [1.0, 0.9]
    >>> context.b1_mpa = [1.0]
    Traceback (most recent call last):
    AttributeError: ReconContext has no attribute 'b1_mpa'
    """

    header: Any
    exam: ExamCache
    config: Any = None
    device: str | None = None

    def __getattr__(self, name: str) -> Any:
        # Reached only where normal lookup failed, so never for a field.
        if name in EXAM_ARTIFACTS:
            exam = self.__dict__.get("exam")
            return None if exam is None else exam.get(name)
        raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in EXAM_ARTIFACTS:
            self.exam[name] = value
            return
        if name in self.__dataclass_fields__ and name not in self.__dict__:
            object.__setattr__(self, name, value)
            return
        if name in self.__dataclass_fields__:
            raise AttributeError(
                f"{type(self).__name__}.{name} is the scan's, not a hook's"
            )
        raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")

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
    """One per-acquisition step of a plugin's ``gadgets``.

    A gadget keeps what it learns from earlier acquisitions of its stream -- a
    noise covariance, a coil basis -- on ``self``. :meth:`ReconPlugin.spawn`
    copies the gadgets, so each stream has its own.

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

    The runtime drives one MRD stream through three hooks: :meth:`startup`
    once, before any acquisition; :meth:`recon` for each reconstruction unit as
    it closes; and :meth:`finish` once, after the last unit. Only :meth:`recon`
    is abstract; the framework method :meth:`receive` is not overridden.

    A *unit* is the set of readouts reconstructed together: those of one
    branch and encoding space that share every image counter (slice, contrast,
    phase, repetition, set, average) not listed in ``axes``. The counters in
    ``axes`` are axes of the unit's k-space instead; ``segment`` and the user
    counters separate no units unless ``segment`` is listed. Each accepted
    acquisition runs through the ``gadgets``, then joins the unit of the branch
    :meth:`branch_for` names, which places it by its flags (see
    :class:`ReconData`). A unit closes when the flag ``triggers`` names for its
    branch has arrived for every combination of the counters in ``axes``. It
    leaves the plugin before :meth:`recon` runs, so its buffers are freed once
    :meth:`recon` returns; the memory a stream holds is bounded by the units
    open at once. Units still open at ``LAST_IN_MEASUREMENT`` or at the end of
    the stream are reconstructed then, in the order they opened and under their
    own branches.

    Each stream runs on its own :meth:`spawn` of the module-level ``PLUGIN``, so
    state set in the hooks belongs to one stream. ``context.exam`` is shared
    across the series of an exam; a plugin adds to it and never clears it.

    Parameters
    ----------
    gadgets
        :class:`Gadget` steps applied to every accepted readout on arrival, in
        order, before it is placed.
    triggers
        ``{branch: flag}``: the :class:`~pulserver.mrd.AcquisitionFlag` that
        closes a unit of the branch, combined with ``|`` for either of several.
        The first branch declared receives the readouts :meth:`branch_for`
        does not route elsewhere. The default is ``{"imaging":
        LAST_IN_MEASUREMENT}``.
    axes
        Counters of a unit placed along axes of its k-space and waited for by
        the closing flag, instead of separating units: any of ``repetition``,
        ``phase``, ``slice``, ``contrast``, ``set``, ``average`` and
        ``segment``. Their extents are the header's encoding limits.
    require_flags
        Flags an acquisition must all carry to be accepted. A combined
        :class:`AcquisitionFlag` counts as its members.
    reject_flags
        Flags any one of which excludes an acquisition.
    buffered
        Place readouts in :attr:`ReconData.data` and :attr:`ReconData.ref`.
        Disable for streams whose header does not describe their encoding
        spaces; the plugin then reads :attr:`ReconData.acquisitions`, and a
        unit closes at the first flag, whatever its ``axes``.
    chain
        Deprecated alias of ``gadgets``; warns.
    branches
        Deprecated alias of ``triggers``, mapping ``{flag: branch}``; warns.

    Attributes
    ----------
    gadgets : tuple of Gadget
    triggers : dict
    axes : tuple of str

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
    ...             gadgets=[DropNoise()],
    ...             triggers={"imaging": mrd.AcquisitionFlag.LAST_IN_SLICE},
    ...             axes=("average",),
    ...         )
    ...     def recon(self, context, branch, data):
    ...         kspace = data.data.kspace
    ...         return recon.ReconResult(np.sqrt(np.sum(np.abs(kspace) ** 2, axis=0)))
    >>> RootSumOfSquares().triggers["imaging"]
    <AcquisitionFlag.LAST_IN_SLICE: 128>
    """

    def __init__(
        self,
        *,
        gadgets: Sequence[Gadget] = (),
        triggers: Mapping[str, Any] | None = None,
        axes: Sequence[str] = (),
        require_flags: tuple[int | str, ...] | AcquisitionFlag = (),
        reject_flags: tuple[int | str, ...] | AcquisitionFlag = (),
        buffered: bool = True,
        chain: Sequence[Gadget] | None = None,
        branches: Mapping[Any, str] | None = None,
    ) -> None:
        if chain is not None:
            warnings.warn(
                "ReconPlugin(chain=...) is deprecated; pass gadgets=",
                DeprecationWarning,
                stacklevel=2,
            )
            if gadgets:
                raise TypeError("pass gadgets, not both gadgets and chain")
            gadgets = chain
        if branches is not None:
            warnings.warn(
                "ReconPlugin(branches={flag: name}) is deprecated; "
                "pass triggers={name: flag}",
                DeprecationWarning,
                stacklevel=2,
            )
            if triggers is not None:
                raise TypeError("pass triggers, not both triggers and branches")
            triggers = _triggers_of(branches)
        unknown = [name for name in axes if name not in _AXES]
        if unknown:
            raise ValueError(
                f"axes {unknown} are not counters a unit can be laid out along; "
                f"they are {list(_AXES)}"
            )
        self.gadgets = tuple(gadgets)
        self.triggers = dict(
            triggers or {"imaging": AcquisitionFlag.LAST_IN_MEASUREMENT}
        )
        self.axes = tuple(axes)
        self.require_flags = _flag_members(require_flags)
        self.reject_flags = _flag_members(reject_flags)
        self.buffered = bool(buffered)
        self._reset()

    def spawn(self) -> ReconPlugin:
        """Return the instance one stream runs on.

        A shallow copy holding its own copy of each gadget, so resources the
        configured plugin holds -- a loaded network, a compiled operator -- are
        shared. Override to isolate anything else a shallow copy would share.
        """
        plugin = copy.copy(self)
        plugin.gadgets = tuple(copy.copy(gadget) for gadget in self.gadgets)
        return plugin

    def startup(self, context: ReconContext) -> None:
        """Start every gadget and lay out the units' encoding spaces from the header.

        Nothing is allocated until a readout is placed. Overrides call
        ``super().startup(context)``.
        """
        self._reset()
        for gadget in self.gadgets:
            gadget.startup(context)
        if self.buffered:
            self._spaces = {
                space.index: space
                for space in EncodingSpace.all_from_header(context.header, self.axes)
            }

    def gadget(self, kind: type) -> Any:
        """Return this stream's first gadget of type ``kind``.

        Hooks reach gadgets through this rather than through ``PLUGIN``, whose gadgets
        :meth:`spawn` copied.

        Raises
        ------
        LookupError
            If the plugin holds no gadget of that type.
        """
        for gadget in self.gadgets:
            if isinstance(gadget, kind):
                return gadget
        raise LookupError(f"this plugin has no {kind.__name__}")

    def branch_for(self, acquisition: Any) -> str | None:
        """Return the branch an acquisition belongs to, or ``None`` to drop it.

        Noise measurements belong to none. Navigator readouts belong to
        ``"navigator"`` when ``triggers`` declares it, else to none. Anything
        else belongs to the first branch ``triggers`` declares.
        """
        if has_acquisition_flag(acquisition, "ACQ_IS_NOISE_MEASUREMENT"):
            return None
        if has_acquisition_flag(acquisition, "ACQ_IS_NAVIGATION_DATA"):
            return "navigator" if "navigator" in self.triggers else None
        return next(iter(self.triggers))

    @final
    def receive(
        self, acquisition: Any, context: ReconContext
    ) -> list[tuple[ReconData | None, Any]]:
        """Take in one acquisition and reconstruct the units it completes.

        In order: the acquisition must pass ``require_flags`` and
        ``reject_flags``; the ``gadgets`` run, any of which may consume it;
        :meth:`branch_for` names its branch; it is added to its unit; the
        units that close are removed from the plugin and reconstructed by
        :meth:`recon`, in the order they opened. An acquisition carrying
        ``LAST_IN_MEASUREMENT`` closes every open unit instead, as :meth:`flush`
        does, whether or not it has samples: the stream's end marker has none.

        Returns
        -------
        list
            ``(data, output)`` for each unit :meth:`recon` returned something
            for, ``data`` being what it was given.
        """
        emitted: list[tuple[ReconData | None, Any]] = []
        if self._accepts(acquisition):
            readout = self._process(acquisition)
            branch = None if readout is None else self.branch_for(acquisition)
            if branch is not None:
                self._unit(branch, acquisition).add_acquisition(acquisition, readout)
            if has_acquisition_flag(acquisition, "ACQ_LAST_IN_MEASUREMENT"):
                emitted = self.flush(context)
            elif branch is not None:
                closed = self._closure.closed(self._units, acquisition, branch)
                emitted = self._reconstruct(closed, context)
        return emitted

    def receive_waveform(self, waveform: Any) -> None:
        """Hold a waveform for the next units that close; see :attr:`ReconData.waveforms`."""
        self._waveforms.append(waveform)

    def flush(self, context: ReconContext) -> list[tuple[ReconData | None, Any]]:
        """Reconstruct every unit still open under its own branch, then call :meth:`finish`.

        :meth:`receive` calls it at ``LAST_IN_MEASUREMENT`` and the runtime at
        the end of the stream; :meth:`finish` runs the first time only.

        Returns
        -------
        list
            As :meth:`receive`; the output of :meth:`finish`, if any, comes last
            and with ``None`` for ``data``.
        """
        emitted = self._reconstruct(list(self._units), context)
        if not self._finished:
            self._finished = True
            output = self.finish(context)
            if output is not None:
                emitted.append((None, output))
        return emitted

    @abstractmethod
    def recon(self, context: ReconContext, branch: str, data: ReconData) -> Any:
        """Reconstruct one unit.

        Parameters
        ----------
        context
            The scan context.
        branch
            The unit's branch, ``data.branch``.
        data
            The unit's readouts. Nothing else keeps them, so a plugin that
            wants them later keeps the parts it needs.

        Returns
        -------
        object or None
            A :class:`ReconResult`, an ``ismrmrd`` output, an array (a magnitude
            :class:`ReconResult`), a sequence of these, or ``None``.
        """
        ...

    def finish(self, context: ReconContext) -> Any:
        """Run once, after the last unit has been reconstructed.

        A :class:`ReconResult` takes its geometry from an acquisition of its
        unit, and this has none, so it returns ``ismrmrd`` outputs or ``None``.
        The default returns ``None``.
        """
        del context
        return None

    def run(
        self,
        path: str,
        *,
        group: str = "dataset",
        exam_id: Hashable | None = None,
        config: Any = None,
        store: str | None = None,
    ) -> list[Any]:
        """Reconstruct one ISMRMRD HDF5 file in this process, through the same hooks.

        The file's waveforms are delivered before its acquisitions. With
        ``store``, the file is a series as the scanner sends it, such as
        :func:`pulserver.virtual.record` writes, and is enriched from its design
        in that store as the proxy enriches it.

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
        store
            Directory of designs, as the design calls write it.

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

        return reconstruct_file(
            self, path, group=group, exam_id=exam_id, config=config, store=store
        )

    def __call__(self, bucket: AcquisitionBucket, context: ReconContext) -> Any:
        """Reconstruct an assembled bucket on a new :meth:`spawn`.

        Runs :meth:`startup`, delivers the bucket's waveforms, then runs
        :meth:`receive` for every acquisition in arrival order and :meth:`flush`,
        and returns the last output that was not ``None``.
        """
        plugin = self.spawn()
        plugin.startup(context)
        for waveform in bucket.waveforms:
            plugin.receive_waveform(waveform)
        output = None
        for acquisition in bucket.acquisitions:
            for _, result in plugin.receive(acquisition, context):
                output = result
        for _, result in plugin.flush(context):
            output = result
        return output

    def _reset(self) -> None:
        """Begin a stream with no unit open."""
        self._spaces: dict[int, EncodingSpace] = {}
        self._closure: _Closure = _FlagClosure(self.triggers, self.axes)
        self._units: dict[UnitKey, ReconUnit] = {}
        self._waveforms: list[Any] = []
        self._finished = False

    def _accepts(self, acquisition: Any) -> bool:
        return all(
            has_acquisition_flag(acquisition, flag) for flag in self.require_flags
        ) and not any(
            has_acquisition_flag(acquisition, flag) for flag in self.reject_flags
        )

    def _process(self, acquisition: Any) -> Any:
        """Run the gadgets over one readout; ``None`` when one consumed it or it has no samples."""
        data = np.asarray(acquisition.data)
        if data.size == 0:
            return None
        for gadget in self.gadgets:
            data = gadget(acquisition, data)
            if data is None:
                return None
        return data

    def _unit(self, branch: str, acquisition: Any) -> ReconUnit:
        key = unit_key(branch, acquisition, self.axes)
        if key not in self._units:
            self._units[key] = ReconUnit(key, self._spaces, buffered=self.buffered)
        return self._units[key]

    def _reconstruct(
        self, keys: list[UnitKey], context: ReconContext
    ) -> list[tuple[ReconData | None, Any]]:
        if not keys:
            return []
        if len(keys) > 1:
            closing = set(keys)
            keys = [key for key in self._units if key in closing]
        waveforms = tuple(self._waveforms)
        self._waveforms.clear()
        emitted: list[tuple[ReconData | None, Any]] = []
        for key in keys:
            data = self._units.pop(key).data
            data.waveforms = waveforms
            output = self.recon(context, data.branch, data)
            if output is not None:
                emitted.append((data, output))
        return emitted


#: Counters a unit can be laid out along, as ``axes`` names them.
_AXES = (*LOOP_COUNTERS, "segment")


# %% private module subroutines


def _flag_members(flags: Any) -> tuple[Any, ...]:
    """Return flags as a tuple, a combined :class:`AcquisitionFlag` split into its members.

    The members are read from the class because iterating a combined ``Flag``
    needs Python 3.11.
    """
    if isinstance(flags, AcquisitionFlag):
        return tuple(member for member in AcquisitionFlag if member in flags)
    return tuple(flags)


def _triggers_of(branches: Mapping[Any, str]) -> dict[str, Any]:
    """Return ``{branch: flag}`` for the deprecated ``{flag: branch}``."""
    triggers: dict[str, Any] = {}
    for flag, name in branches.items():
        if name in triggers:
            raise ValueError(
                f"branches maps two flags to {name!r}; declare "
                f"triggers={{{name!r}: first | second}} instead"
            )
        triggers[name] = flag
    return triggers


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
