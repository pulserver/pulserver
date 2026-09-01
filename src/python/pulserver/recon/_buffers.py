"""Placing acquisitions into the buffers a header describes.

Gadgetron's chain sorts a bucket before anything reconstructs it:
``BucketToBuffer`` turns the acquisitions it accumulated into one buffer per
encoding space, sized from the header's ``encodingLimits`` and indexed by the
counters each acquisition carries. This is that step.

The XML header arrives before any data, and it describes **every** encoding
space the scan will produce -- one per subsequence. Each acquisition then says
which one it belongs to through ``encoding_space_ref``. So the layout is known
before the first line arrives, and every line can be placed the moment it does,
rather than accumulated in a list and sorted at the trigger.

Axis order is coils first, readout last, matching the rest of the package
rather than Gadgetron's ``[E0,E1,E2,CHA,N,S,LOC]``. What is borrowed is the
shape of the idea, not its index order: a buffer that disagreed with the arrays
around it would cost a transpose at every boundary.
An axis exists where the header says something varies, and nowhere else: a
two-dimensional scan has no partition axis and a single-slice one no slice
axis, so what comes out is the array a reconstruction would have built by
hand. Coils and readout are always axes -- combining and sampling are what
every reconstruction does. :attr:`ReconBuffer.axes` names them all, so a plugin
reads the layout off the buffer rather than knowing it by convention.
"""

from __future__ import annotations

__all__ = ["ReconBuffer", "ReconData"]

from collections.abc import Iterator, Mapping
from typing import Any, ClassVar

import numpy as np

from ..mrd._header import LOOP_COUNTERS, EncodingSpace
from ..mrd._metadata import acquisition_label, has_acquisition_flag

#: Counters selecting which *unit* an acquisition belongs to, ordered as they
#: nest from outermost in. ``segment`` is deliberately absent: it names a piece
#: of one readout train, not a separate image, and splitting on it would take
#: an EPI shot apart.
_AXIS_NAMES = frozenset((*LOOP_COUNTERS, "partition", "phase_encode"))


class ReconBuffer:
    """One encoding space's k-space, filled one acquisition at a time.

    Laid out from the header, so an acquisition is placed the moment it is
    received rather than held for a sort at the trigger. :attr:`mask` records
    what was actually sampled, which is what tells a reconstruction whether a
    phase encode was skipped or a readout truncated.

    A readout shorter than the buffer is right-aligned against it: truncating
    the samples before the echo is what a partial echo does, so the acquired
    window ends where a full one would.

    Parameters
    ----------
    space
        The encoding space this buffers.
    coils, readout
        Channels and samples to allocate. Both are facts the data carries
        rather than the header, so the first arrival settles them. They settle
        differently: a readout narrower than the encoded matrix is a partial
        echo and still needs the full width to be right-aligned in, so the
        readout only ever widens, while the channel count is simply what
        arrived -- a plugin that compresses the array before placing a readout
        places fewer channels than the header declares, and the buffer holds
        what it was given. Where the acquisitions go is another matter: that
        the header declares, and a contradiction is an error.
    dtype
        Complex dtype of :attr:`kspace`.

    Attributes
    ----------
    kspace : ndarray
        The buffer, shaped as :attr:`axes` names.
    mask : ndarray
        Boolean, :attr:`kspace` without its coil axis: where data landed. What
        a reconstruction reads to know a phase encode was skipped or a readout
        truncated.
    reference : ndarray
        Boolean, the same shape: which of those positions the scanner flagged
        as parallel-imaging calibration.
    trajectory : ndarray or None
        Where each sample was taken, ``(dimensions, ...)`` over the same axes
        as :attr:`kspace`, or ``None`` for a scan whose acquisitions carry no
        trajectory. Allocated on the first one that does, so a Cartesian scan
        never holds one, and as wide as the widest acquisition placed so far:
        an axis an acquisition does not traverse is left off what it carries
        and reads back as the zero it was.
    center_sample : int or None
        Index of the echo along the readout axis, from the acquisitions
        themselves, or ``None`` until the first one arrives.
    sample_time : float or None
        Dwell time (s), likewise from the acquisitions.

    Examples
    --------
    A plugin receives its buffers already laid out, from what the header
    says. Stating a layout directly is how a test asks for one without a scan
    file -- four channels, two slices, a 32-line readout of 64 samples:

    >>> import pulserver.recon as recon
    >>> import pulserver.mrd as mrd
    >>> space = mrd.EncodingSpace(
    ...     index=0, coils=4, readout=64, phase_encodes=32, partitions=1,
    ...     loops=("slice",), loop_sizes=(2,), recon_matrix=(32, 32),
    ... )
    >>> buffer = recon.ReconBuffer(space)

    The layout is the encoding space's, so a plugin knows the shape before the
    first line arrives:

    >>> buffer.kspace.shape
    (4, 2, 32, 64)
    >>> buffer.extents
    {'coil': 4, 'slice': 2, 'phase_encode': 32, 'readout': 64}
    >>> buffer.image_shape
    (32, 32)
    """

    def __init__(
        self,
        space: EncodingSpace,
        *,
        coils: int | None = None,
        readout: int | None = None,
        dtype: Any = np.complex64,
    ) -> None:
        self.space = space
        self.coils = int(coils) if coils else space.coils
        self.readout = max(space.readout, readout or 0)
        shape = (self.coils, *space.shape[1:-1], self.readout)
        self.kspace = np.zeros(shape, dtype=dtype)
        self.mask = np.zeros(shape[1:], dtype=bool)
        self.reference = np.zeros(shape[1:], dtype=bool)
        self.trajectory: Any | None = None
        self.center_sample: int | None = None
        self.sample_time: float | None = None
        self.headers: list[Any] = []

    @property
    def axes(self) -> tuple[str, ...]:
        """Name of every axis of :attr:`kspace`, in order."""
        return self.space.axes

    @property
    def extents(self) -> dict[str, int]:
        """How far each axis of :attr:`kspace` runs, by the name it goes under.

        What a plugin loops over: ``extents.get("slice", 1)`` answers for a scan
        whether or not it varies the slice, so a reconstruction is written once
        and a single-slice scan is that loop run once.
        """
        return dict(zip(self.axes, self.kspace.shape, strict=True))

    @property
    def image_shape(self) -> tuple[int, ...]:
        """The matrix the header asks the images to be cropped to.

        The reconstructed space, so the readout oversampling the scanner
        digitises and any phase field-of-view oversampling are off it.
        """
        return self.space.recon_matrix

    #: The MRD counter each placement axis is read from.
    _COUNTERS: ClassVar[dict[str, str]] = {
        "partition": "kspace_encode_step_2",
        "phase_encode": "kspace_encode_step_1",
    }

    def position(self, acquisition: Any) -> tuple[int, ...]:
        """Index along each axis of :attr:`kspace` this acquisition fills.

        Raises
        ------
        ValueError
            If a counter runs past what the header laid this space out for.
        """
        where: list[int] = []
        for name, extent in self.space.extents:
            counter = self._COUNTERS.get(name, name)
            index = int(acquisition_label(acquisition, counter, 0) or 0)
            if not 0 <= index < extent:
                raise ValueError(
                    f"acquisition has {counter}={index}, past the {extent} "
                    f"{name} positions encoding space {self.space.index} was "
                    f"laid out for"
                )
            if extent > 1:
                where.append(index)
        return tuple(where)

    def add(self, acquisition: Any, data: Any = None) -> None:
        """Place one acquisition where its counters say it belongs.

        Parameters
        ----------
        acquisition
            The acquisition, whose counters and flags say where it goes.
        data
            ``(coils, samples)`` to place instead of ``acquisition.data``, for
            a readout a plugin corrected on arrival -- the reversed line of an
            EPI train, flipped and phase corrected in
            :meth:`~pulserver.ReconPlugin.receive`. ``None`` places what the
            acquisition carries.

        Raises
        ------
        ValueError
            If the acquisition does not fit the space it is placed in.
        """
        data = np.asarray(acquisition.data if data is None else data)
        if data.ndim != 2:
            raise ValueError(
                f"acquisition data must be (coils, samples), got shape {data.shape}"
            )
        coils, samples = data.shape
        if samples > self.readout or coils > self.coils:
            raise ValueError(
                f"acquisition is {coils} x {samples} but encoding space "
                f"{self.space.index} was laid out for "
                f"{self.coils} x {self.readout}"
            )

        where = self.position(acquisition)

        # Right-aligned, which is where a partial echo's acquired window ends.
        offset = self.readout - samples
        readout = slice(offset, self.readout)
        self.kspace[(slice(0, coils), *where, readout)] = data
        self.mask[(*where, readout)] = True
        if has_acquisition_flag(
            acquisition, "ACQ_IS_PARALLEL_CALIBRATION"
        ) or has_acquisition_flag(
            acquisition, "ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING"
        ):
            self.reference[(*where, readout)] = True
        self._place_trajectory(acquisition, where, readout)
        if self.center_sample is None:
            center = acquisition_label(acquisition, "center_sample", None)
            if center is not None:
                self.center_sample = int(center) + offset
        if self.sample_time is None:
            dwell = acquisition_label(acquisition, "sample_time_us", None)
            if dwell:
                self.sample_time = float(dwell) * 1e-6
        self.headers.append(acquisition)

    def _place_trajectory(self, acquisition: Any, where: tuple, readout: slice) -> None:
        """Store where this acquisition's samples were taken, if it says.

        MRD gives a trajectory as ``(samples, dimensions)``; it is transposed
        here so it is indexed like :attr:`kspace`, dimension first.

        An acquisition states its own dimensionality, and a trailing axis it
        does not traverse is left off what it carries -- the centre partition
        of a Cartesian slab traverses no kz, and says so with two dimensions
        where its neighbours say three. So the rows an acquisition carries are
        placed and the rest stay zero, which is where that axis was.
        """
        traj = getattr(acquisition, "traj", None)
        if traj is None:
            return
        traj = np.asarray(traj)
        if traj.size == 0:
            return
        dimensions = int(traj.shape[-1])
        if self.trajectory is None:
            self.trajectory = np.zeros(
                (dimensions, *self.kspace.shape[1:]), dtype=traj.dtype
            )
        elif dimensions > self.trajectory.shape[0]:
            widened = np.zeros(
                (dimensions, *self.trajectory.shape[1:]),
                dtype=self.trajectory.dtype,
            )
            widened[: self.trajectory.shape[0]] = self.trajectory
            self.trajectory = widened
        self.trajectory[(slice(0, dimensions), *where, readout)] = traj.T

    def select(self, **where: int) -> tuple[Any, Any]:
        """The ``(kspace, mask)`` at one position along the named axes.

        A name this buffer has no axis for is accepted at index 0 and refused
        beyond it, because position 0 of an axis that does not vary is the only
        position there is. So a plugin asks for ``select(slice=index)`` without
        first checking whether the scan has more than one slice, and a request
        that genuinely falls off the end is still an error.

        Parameters
        ----------
        **where
            One index per axis to fix, by the name :attr:`axes` gives it. Axes
            left unnamed come back whole.

        Returns
        -------
        tuple
            The k-space at that position and its sampling mask, both without
            the axes that were fixed.

        Raises
        ------
        KeyError
            If a name is not an MRD encoding axis at all.
        IndexError
            If an axis this space laid out flat is asked for beyond position 0.
        """
        picks = self._picks(where)
        return self.kspace[(slice(None), *picks)], self.mask[picks]

    @property
    def readout_time(self) -> Any:
        """When each sample of a readout was taken, relative to the echo (s).

        Off-resonance is a phase that accrues along the readout, so what a
        correction needs is the clock the samples ran on, not their position
        in k.

        Returns
        -------
        numpy.ndarray or None
            ``(readout,)`` seconds, negative before the echo, or ``None`` when
            the acquisitions state no dwell or no echo position.
        """
        if self.sample_time is None or self.center_sample is None:
            return None
        return (np.arange(self.readout) - self.center_sample) * self.sample_time

    def points(self, **where: int) -> Any:
        """Where the samples at one position were taken, or ``None``.

        The trajectory counterpart of :meth:`select`, indexed the same way, so
        a non-Cartesian reconstruction pairs measurement with position without
        knowing which axes this scan happens to have.

        Returns
        -------
        ndarray or None
            ``(dimensions, ...)`` over the axes that were not fixed, or
            ``None`` for a scan whose acquisitions carry no trajectory.
        """
        if self.trajectory is None:
            return None
        return self.trajectory[(slice(None), *self._picks(where))]

    def _picks(self, where: Mapping[str, int]) -> tuple[Any, ...]:
        """Index each placement axis, checking the ones this space laid flat."""
        placement = self.axes[1:-1]
        for name, index in where.items():
            if name in placement:
                continue
            if name not in _AXIS_NAMES:
                raise KeyError(
                    f"{name!r} is not an encoding axis; encoding space "
                    f"{self.space.index} has {list(placement)}"
                )
            if index != 0:
                raise IndexError(
                    f"{name}={index} but encoding space {self.space.index} "
                    f"has only one {name}"
                )
        return tuple(where.get(name, slice(None)) for name in placement)

    def __repr__(self) -> str:
        named = ", ".join(
            f"{n}={s}" for n, s in zip(self.axes, self.kspace.shape, strict=True)
        )
        return f"ReconBuffer(encoding={self.space.index}, {named})"


class ReconData(Mapping):
    """Every encoding space of a scan, filled as the acquisitions arrive.

    The buffered counterpart of :class:`AcquisitionBucket`: the bucket is what
    arrived, this is where it goes. It maps an encoding-space index to its
    :class:`ReconBuffer`.

    One buffer per space, not the imaging/reference pair Gadgetron splits a
    bucket into. That split exists because a calibration can be acquired on a
    different grid from the image, and here it cannot: a subsequence is an
    encoding space, so a calibration with its own geometry -- the low-resolution
    gradient echo an EPI scan calibrates from -- is already a separate space
    with its own buffer. Within one space the calibration lines are on the
    imaging grid by construction, and which of them the scanner flagged is
    :attr:`ReconBuffer.reference`.

    A buffer is allocated the first time an acquisition names its space, so a
    scan never pays for a space it does not fill, and the first arrival can
    widen the readout past what the encoded matrix declared.

    Parameters
    ----------
    spaces
        The encoding spaces the header describes.
    dtype
        Complex dtype of the k-space arrays.

    Attributes
    ----------
    spaces : dict
        What the header described, by encoding-space index.
    data : dict
        The buffers allocated so far, by encoding-space index.

    Examples
    --------
    :meth:`from_header` is what a stream uses. Stating the spaces directly is
    the offline spelling of the same thing:

    >>> import pulserver.recon as recon
    >>> import pulserver.mrd as mrd
    >>> space = mrd.EncodingSpace(
    ...     index=0, coils=4, readout=64, phase_encodes=32, partitions=1,
    ...     loops=("slice",), loop_sizes=(2,), recon_matrix=(32, 32),
    ... )
    >>> data = recon.ReconData([space])

    A buffer costs its memory only once something reaches it, so a scan whose
    prescan never runs never allocates its grid:

    >>> len(data)
    0
    >>> data[0].kspace.shape
    (4, 2, 32, 64)
    >>> len(data)
    1
    """

    def __init__(self, spaces: Any = (), *, dtype: Any = np.complex64) -> None:
        self.spaces = {space.index: space for space in spaces}
        self.dtype = dtype
        self.data: dict[int, ReconBuffer] = {}

    @classmethod
    def from_header(cls, header: Any, *, dtype: Any = np.complex64) -> ReconData:
        """Lay out every encoding space the header describes.

        A header describing none -- an offline bucket assembled from arrays --
        gives an empty container, which accepts acquisitions and buffers
        nothing, because nothing said where they go.
        """
        return cls(EncodingSpace.all_from_header(header), dtype=dtype)

    def buffer(
        self,
        encoding: int = 0,
        *,
        coils: int | None = None,
        readout: int | None = None,
    ) -> ReconBuffer:
        """The buffer for one encoding space, allocating it on first use.

        Raises
        ------
        KeyError
            If the header described no such encoding space.
        """
        if encoding not in self.spaces:
            raise KeyError(
                f"the header describes no encoding space {encoding}; "
                f"it has {sorted(self.spaces)}"
            )
        if encoding not in self.data:
            self.data[encoding] = ReconBuffer(
                self.spaces[encoding], coils=coils, readout=readout, dtype=self.dtype
            )
        return self.data[encoding]

    def add(self, acquisition: Any, data: Any = None) -> None:
        """Place one acquisition in the buffer its header names.

        Routed by ``encoding_space_ref``, which is how a scan of several
        subsequences sorts itself with nothing declared per plugin. ``data``
        replaces what the acquisition carries; see :meth:`ReconBuffer.add`.
        """
        if not self.spaces:
            return
        encoding = int(acquisition_label(acquisition, "encoding_space_ref", 0) or 0)
        coils, samples = np.shape(acquisition.data if data is None else data)[-2:]
        self.buffer(encoding, coils=coils, readout=samples).add(acquisition, data)

    def __getitem__(self, encoding: int) -> ReconBuffer:
        return self.buffer(encoding)

    def __iter__(self) -> Iterator[int]:
        return iter(sorted(self.data))

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        filled = ", ".join(repr(self.data[key]) for key in sorted(self.data))
        return f"ReconData({filled})"
