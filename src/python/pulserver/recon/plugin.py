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
    "ReconBuffer",
    "ReconContext",
    "ReconData",
    "ReconPlugin",
    "ReconResult",
]

import copy
import logging
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
from threading import RLock
from typing import Any

import numpy as np

from ..mrd._acquisitions import (
    AcquisitionBucket,
    AcquisitionFlag,
    has_acquisition_flag,
)
from ._buffers import ReconBuffer, ReconData


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
    """Thread-safe artifact cache owned by one scanner exam.

    Values remain alive while the corresponding exam generation is leased by
    an active reconstruction. Use :meth:`get_or_create` for expensive
    calibration artifacts; its factory executes at most once per key.

    Parameters
    ----------
    exam_id
        Stable identifier for the exam owning this cache.

    Examples
    --------
    >>> import pulserver.recon as recon
    >>> cache = recon.ExamCache("exam-1")
    >>> cache["coil_maps"] = "maps"
    >>> "coil_maps" in cache, len(cache)
    (True, 1)
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

    Only :meth:`recon` is mandatory. What the other two do is *declared* rather
    than written: a plugin lists the per-acquisition steps its readouts go
    through as its ``chain``, and the boundaries worth reconstructing at as its
    ``branches``, and the default :meth:`receive` does the rest.

    The chain
    ---------
    A :class:`~pulserver.recon.Gadget` is one per-acquisition step -- noise
    adjustment, coil compression, the EPI corrections -- run in order as each
    readout lands, before it is placed. A step that returns ``None`` consumes
    the acquisition, which is how a noise scan and a navigator line reach no
    buffer at all::

        super().__init__(chain=[NoiseAdjust(), EpiPhaseCorrection(order=1)])

    Branches
    --------
    A branch is a name :meth:`recon` switches on -- ``"calibration"`` and
    ``"imaging"`` in the scans that have two. ``branches`` maps the flag that
    closes each one to its name, and the **order is the priority**::

        super().__init__(
            chain=[NoiseAdjust()],
            branches={
                AcquisitionFlag.LAST_IN_SLICE: "imaging",
                AcquisitionFlag.LAST_IN_SEGMENT: "calibration",
            },
        )

        def recon(self, branch, context):
            if branch == "calibration":
                self.maps = sensitivities(self.buffers[0])   # no image yet
                return None
            ...                                             # the slice is complete

    Listing the slice first is what makes that read correctly: the final
    acquisition of a slice closes its trailing segment *and* the slice, so the
    calibration branch is reached only where nothing larger ended. An empty
    mapping reconstructs once, at the end of the stream.

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
    chain
        The :class:`~pulserver.recon.Gadget` steps every readout goes through
        on arrival, in order.
    branches
        Which boundary reconstructs which branch, as
        ``{AcquisitionFlag: name}``, tried in order. Several flags may be
        combined with ``|`` to route either of them to one branch. The default
        reconstructs ``"imaging"`` at the end of the measurement; an empty
        mapping reconstructs once, at the end of the stream.
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
    chain : tuple
        This stream's gadgets. :meth:`spawn` gives each stream its own, so what
        one learns from its acquisitions is never another's.
    branches : dict
        The boundary-to-branch mapping, in priority order.
    acquisition : object
        The last acquisition :meth:`receive` accepted, which is the one that
        closed the branch :meth:`recon` is running -- so a reconstruction that
        runs per slice reads its index from ``self.acquisition.idx.slice``
        rather than tracking it. ``None`` before the first one arrives.
    buffers : ReconData
        Every encoding space of the scan, filled as the acquisitions arrive.
        Laid out by :meth:`startup` from the header; empty until then, and
        empty for a bucket assembled from arrays, which carries no header to
        lay anything out from.
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
        self.require_flags = tuple(require_flags)
        self.reject_flags = tuple(reject_flags)
        self.buffered = bool(buffered)
        self.buffers = ReconData()
        self.acquisition: Any = None

    def spawn(self) -> ReconPlugin:
        """Return the working instance one stream reconstructs through.

        A shallow copy, so anything expensive the configured plugin holds -- a
        loaded network, a compiled operator -- is shared rather than duplicated,
        while whatever the lifecycle hooks assign stays private to the stream.
        The gadgets are copied too, because what one learns from its
        acquisitions -- a noise covariance, a phase fit -- belongs to the scan
        it learned it from. Override to isolate something a shallow copy would
        still share.
        """
        plugin = copy.copy(self)
        plugin.chain = tuple(copy.copy(gadget) for gadget in self.chain)
        return plugin

    def startup(self, context: ReconContext) -> None:
        """Lay the scan's buffers out, before the first acquisition.

        Runs once when the stream opens. The default reads every encoding space
        the header describes into :attr:`buffers`; nothing is allocated until
        an acquisition names a space, so the spaces this scan does not fill
        cost nothing.

        Override to add what the header cannot say -- a trajectory, a
        precomputed operator -- and call ``super().startup(context)`` to keep
        the buffers and the chain.
        """
        for gadget in self.chain:
            gadget.startup(context)
        if self.buffered:
            self.buffers = ReconData.from_header(context.header)

    def process(self, acquisition: Any, data: Any = None) -> Any:
        """Run the chain over one readout, and return what it left.

        Each :class:`~pulserver.recon.Gadget` is handed the acquisition and the
        readout the step before it produced. Exposed so a plugin writing its
        own :meth:`receive` still puts every readout through the same steps.

        Parameters
        ----------
        acquisition
            The acquisition, for its flags and counters.
        data
            The readout to start from. ``None`` takes the acquisition's own.

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
        """This stream's gadget of one kind.

        :meth:`spawn` gives every stream its own gadgets, so a hook reaching
        one -- a calibration branch handing its filled buffer to the coil
        compression to learn from -- has to ask for it rather than hold the
        configured plugin's.

        Raises
        ------
        LookupError
            If the chain holds no gadget of that kind.
        """
        for gadget in self.chain:
            if isinstance(gadget, kind):
                return gadget
        raise LookupError(f"this plugin's chain has no {kind.__name__}")

    def branch_for(self, acquisition: Any) -> str | None:
        """The branch this acquisition closes, or ``None``.

        The first entry of ``branches`` whose flag the acquisition carries, so
        the order they were declared in is their priority: a slice listed ahead
        of a segment is what makes "a segment that closed nothing larger"
        the calibration block.
        """
        if acquisition is None:
            return None
        for flag, branch in self.branches.items():
            if _closes(acquisition, flag):
                return branch
        return None

    def receive(self, acquisition: Any, context: ReconContext) -> Any:
        """Correct one acquisition, place it, and reconstruct what it completed.

        Runs for every accepted acquisition, as it arrives, so the chain, the
        sorting and the routing all overlap acquisition dead time. The chain
        runs first, and a step that consumes the readout ends it here.
        Placement then routes on ``encoding_space_ref`` and indexes by the
        acquisition's own counters, which is what the header laid the buffers
        out for. The routing reads the same acquisition's flags: the first
        boundary of ``branches`` it closes selects what :meth:`recon` runs, and
        whatever that returns is returned from here for the runtime to emit.

        Overriding it is for placement the declaration cannot express -- an EPI
        prescan whose lines belong in a different buffer from the imaging ones.
        Call :meth:`process` from any override, so the chain still runs.

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
        from ._server.offline import reconstruct_file

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


def _last(acquisitions: tuple[Any, ...]) -> Any | None:
    """The acquisition that ended a bucket, or ``None`` for an empty one."""
    return acquisitions[-1] if acquisitions else None


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
