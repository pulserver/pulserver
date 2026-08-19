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

__all__ = ["EncodingSpace", "ReconBuffer", "ReconData"]

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from ._mrd.metadata import acquisition_label, has_acquisition_flag

#: Counters selecting which *unit* an acquisition belongs to, ordered as they
#: nest from outermost in. ``segment`` is deliberately absent: it names a piece
#: of one readout train, not a separate image, and splitting on it would take
#: an EPI shot apart.
LOOP_COUNTERS = (
    "repetition",
    "phase",
    "slice",
    "contrast",
    "set",
    "average",
)


#: Every axis an acquisition can be placed along, which is the vocabulary
#: :meth:`ReconBuffer.select` accepts whether or not this scan varies it.
_AXIS_NAMES = frozenset((*LOOP_COUNTERS, "partition", "phase_encode"))


def _is_cartesian(encoding: Any) -> bool:
    """Whether this encoding samples a grid rather than a trajectory.

    A header that does not say is read as Cartesian, which is what an MRD
    header omitting the field has always meant.
    """
    trajectory = getattr(encoding, "trajectory", None)
    if trajectory is None:
        return True
    name = getattr(trajectory, "name", None) or str(trajectory)
    return name.rsplit(".", 1)[-1].upper() == "CARTESIAN"


def _limit(limits: Any, name: str) -> int:
    """Extent of one encoding limit, or 0 when the header does not state it."""
    entry = getattr(limits, name, None) if limits is not None else None
    maximum = getattr(entry, "maximum", None) if entry is not None else None
    return 0 if maximum is None else int(maximum) + 1


@dataclass(frozen=True)
class EncodingSpace:
    """What one encoding space of the header says its buffer must hold.

    Parameters
    ----------
    index
        Position in the header's encoding list, which is what an acquisition's
        ``encoding_space_ref`` names.
    coils
        Receive channels.
    readout
        Samples per readout.
    phase_encodes, partitions
        Extent along ``kspace_encode_step_1`` and ``kspace_encode_step_2``.
        Which of the header's two statements answers this is what the encoding's
        ``trajectory`` decides: a Cartesian space is a grid, and an undersampled
        one still needs every column of it, so the encoded matrix wins over a
        limit that counts only the lines acquired. A non-Cartesian space has no
        grid -- its ``kspace_encode_step_1`` counts views, which bear no
        relation to the image matrix -- so the limit wins.
    loops
        Names of the counters that vary in this space, outermost first.
    loop_sizes
        Extent of each of those counters.
    recon_matrix
        The image matrix the header asks for, ``(n_y, n_x)`` for a plane or
        ``(n_z, n_y, n_x)`` for a volume -- which of the two is the header's
        answer, not the buffer's: a stack of spokes has no partition axis and
        still reconstructs a volume.
    """

    index: int
    coils: int
    readout: int
    phase_encodes: int
    partitions: int
    loops: tuple[str, ...]
    loop_sizes: tuple[int, ...]
    recon_matrix: tuple[int, ...]

    @classmethod
    def from_header(cls, header: Any, index: int = 0) -> EncodingSpace:
        """Read encoding space ``index`` out of a parsed MRD header.

        Raises
        ------
        IndexError
            If the header describes no such encoding space.
        """
        encodings = getattr(header, "encoding", None) or ()
        encoding = encodings[index]
        encoded = encoding.encodedSpace.matrixSize
        limits = getattr(encoding, "encodingLimits", None)

        loops: list[str] = []
        sizes: list[int] = []
        for name in LOOP_COUNTERS:
            extent = _limit(limits, name)
            if extent > 1:
                loops.append(name)
                sizes.append(extent)

        system = getattr(header, "acquisitionSystemInformation", None)
        coils = int(getattr(system, "receiverChannels", 1) or 1)

        matrix = encoded if getattr(encoding, "reconSpace", None) is None else None
        matrix = matrix or getattr(encoding.reconSpace, "matrixSize", None) or encoded
        recon_matrix = (int(matrix.z), int(matrix.y), int(matrix.x))
        if recon_matrix[0] == 1:
            recon_matrix = recon_matrix[1:]

        views = _limit(limits, "kspace_encoding_step_1")
        partitions = _limit(limits, "kspace_encoding_step_2")
        gridded = _is_cartesian(encoding)

        return cls(
            index=index,
            coils=coils,
            readout=int(encoded.x),
            phase_encodes=max(views, int(encoded.y))
            if gridded
            else views or int(encoded.y),
            # A stack is Cartesian along z whatever it does in plane.
            partitions=max(partitions, int(encoded.z)),
            loops=tuple(loops),
            loop_sizes=tuple(sizes),
            recon_matrix=recon_matrix,
        )

    @classmethod
    def all_from_header(cls, header: Any) -> tuple[EncodingSpace, ...]:
        """Read every encoding space the header describes.

        One per subsequence, so this is the whole scan's layout, known before a
        single acquisition arrives.
        """
        encodings = getattr(header, "encoding", None) or ()
        return tuple(cls.from_header(header, index) for index in range(len(encodings)))

    @property
    def extents(self) -> tuple[tuple[str, int], ...]:
        """Every axis an acquisition is placed along, named, outermost first.

        Coils and readout are not among them: an acquisition spans those
        rather than being placed along them.
        """
        return (
            *zip(self.loops, self.loop_sizes, strict=True),
            ("partition", self.partitions),
            ("phase_encode", self.phase_encodes),
        )

    @property
    def axes(self) -> tuple[str, ...]:
        """Name of every axis of :attr:`ReconBuffer.kspace`, in order.

        The ones that vary. A scan with one partition has no partition axis,
        which is what makes a two-dimensional buffer two-dimensional.
        """
        varying = tuple(name for name, extent in self.extents if extent > 1)
        return ("coil", *varying, "readout")

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape a buffer of this space has, before any widening."""
        varying = tuple(extent for _, extent in self.extents if extent > 1)
        return (self.coils, *varying, self.readout)


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
        Channels and samples to allocate, widening the space's declaration.
        Both are facts the data carries -- readout oversampling makes an
        acquisition wider than the encoded matrix says, and a header need not
        state its channel count at all -- so the first arrival widens the
        buffer rather than being refused. Where the acquisitions go is another
        matter: that the header declares, and a contradiction is an error.
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
        never holds one.
    center_sample : int or None
        Index of the echo along the readout axis, from the acquisitions
        themselves, or ``None`` until the first one arrives.
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
        self.coils = max(space.coils, coils or 0)
        self.readout = max(space.readout, readout or 0)
        shape = (self.coils, *space.shape[1:-1], self.readout)
        self.kspace = np.zeros(shape, dtype=dtype)
        self.mask = np.zeros(shape[1:], dtype=bool)
        self.reference = np.zeros(shape[1:], dtype=bool)
        self.trajectory: Any | None = None
        self.center_sample: int | None = None
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
        self.headers.append(acquisition)

    def _place_trajectory(self, acquisition: Any, where: tuple, readout: slice) -> None:
        """Store where this acquisition's samples were taken, if it says.

        MRD gives a trajectory as ``(samples, dimensions)``; it is transposed
        here so it is indexed like :attr:`kspace`, dimension first.
        """
        traj = getattr(acquisition, "traj", None)
        if traj is None:
            return
        traj = np.asarray(traj)
        if traj.size == 0:
            return
        if self.trajectory is None:
            self.trajectory = np.zeros(
                (traj.shape[-1], *self.kspace.shape[1:]), dtype=traj.dtype
            )
        self.trajectory[(slice(None), *where, readout)] = traj.T

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
