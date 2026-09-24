"""K-space buffers laid out from the MRD header and filled one acquisition at a time.

One buffer per encoding space, sized from the header before data arrives and
indexed by each acquisition's counters. Axes run coil first and readout last,
with a placement axis only where the header says a counter varies.
"""

from __future__ import annotations

__all__ = ["ReconBuffer", "ReconData"]

from collections.abc import Iterator, Mapping
from typing import Any, ClassVar

import numpy as np

from ..mrd._header import LOOP_COUNTERS, EncodingSpace
from ..mrd._metadata import acquisition_label, has_acquisition_flag

#: Every name :meth:`ReconBuffer.select` accepts, whether or not a space varies it.
_AXIS_NAMES = frozenset((*LOOP_COUNTERS, "partition", "phase_encode"))


class ReconBuffer:
    """K-space of one encoding space, filled one acquisition at a time.

    A readout shorter than the buffer is right-aligned, where a partial echo's
    samples belong.

    Parameters
    ----------
    space
        Encoding space to lay out.
    coils
        Channels to allocate; ``space.coils`` when not given. Fewer than the
        header declares is valid, for data compressed before placement.
    readout
        Samples to allocate; never fewer than ``space.readout``.
    dtype
        Complex dtype of :attr:`kspace`.

    Attributes
    ----------
    kspace : ndarray
        Shaped as :attr:`axes` names.
    mask : ndarray
        Boolean, :attr:`kspace` without the coil axis: samples that were placed.
    reference : ndarray
        Boolean, shaped as :attr:`mask`: samples flagged as parallel-imaging
        calibration.
    trajectory : ndarray or None
        ``(dimensions, ...)`` over the axes of :attr:`mask`, in the units the
        acquisitions carry. ``None`` until an acquisition carries a trajectory;
        widened to the most dimensions any acquisition carried, missing trailing
        dimensions reading 0.
    center_sample : int or None
        Echo index along the readout axis, after right alignment, from the first
        acquisition that states it.
    sample_time : float or None
        Dwell time in seconds, from the first acquisition that states it.
    headers : list
        Placed acquisitions, in placement order.

    Examples
    --------
    >>> import pulserver.recon as recon
    >>> import pulserver.mrd as mrd
    >>> space = mrd.EncodingSpace(
    ...     index=0, coils=4, readout=64, phase_encodes=32, partitions=1,
    ...     loops=("slice",), loop_sizes=(2,), recon_matrix=(32, 32),
    ... )
    >>> buffer = recon.ReconBuffer(space)
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
        """Extent of each axis of :attr:`kspace`, by name.

        Axes that do not vary are absent, so ``extents.get("slice", 1)`` answers for
        any scan.
        """
        return dict(zip(self.axes, self.kspace.shape, strict=True))

    @property
    def image_shape(self) -> tuple[int, ...]:
        """Image matrix the header prescribes; see :attr:`EncodingSpace.recon_matrix`."""
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

        Also records its trajectory, and ``center_sample`` and dwell when not yet
        known.

        Parameters
        ----------
        acquisition
            The acquisition, for its counters, flags and data.
        data
            ``(coils, samples)`` to place instead of ``acquisition.data``, for a
            readout corrected before placement.

        Raises
        ------
        ValueError
            If the data is not two-dimensional, is larger than the buffer, or a
            counter is outside the laid-out extent.
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
        """Store the acquisition's trajectory, if it carries one.

        MRD trajectories are ``(samples, dimensions)`` and are stored transposed.
        An acquisition may carry fewer trailing dimensions than its neighbours --
        the centre partition of a slab traverses no kz -- and the rows it omits stay
        0.
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
        """Return the ``(kspace, mask)`` at one position along named axes.

        An axis this space does not vary is accepted at index 0, its only position,
        so ``select(slice=i)`` works whether or not the scan has slices.

        Parameters
        ----------
        **where
            Index per axis, by the names in :attr:`axes`; unnamed axes are kept
            whole.

        Returns
        -------
        tuple
            K-space and sampling mask without the fixed axes.

        Raises
        ------
        KeyError
            If a name is not an encoding axis.
        IndexError
            If an axis this space does not vary is asked for beyond index 0.
        """
        picks = self._picks(where)
        return self.kspace[(slice(None), *picks)], self.mask[picks]

    @property
    def readout_time(self) -> Any:
        """Sample times relative to the echo, ``(readout,)`` in seconds.

        ``None`` until the acquisitions state both a dwell and an echo position.
        """
        if self.sample_time is None or self.center_sample is None:
            return None
        return (np.arange(self.readout) - self.center_sample) * self.sample_time

    def grid_trajectory(self) -> Any:
        """Return the trajectory in grid units, laid out as ``bartorch.linop.NUFFT`` takes it.

        ``(*placement, readout, 3)``: the placement axes of :attr:`mask`, the
        samples, then kx, ky and kz, each k in 1/m, as the proxy's enrichment
        writes it, times the reconstructed field of view along its axis, so
        that an ``N``-point image matrix spans ``[-N/2, N/2)``. Axes the
        acquisitions do not carry are 0. The phase-encoding axis is the last
        placement axis, so this is ``(*encoding, shots, samples, 3)`` against
        :attr:`kspace` as ``(coils, *encoding, shots, samples)``.

        Returns
        -------
        ndarray or None
            ``float32``; ``None`` when no acquisition carried a trajectory.

        Raises
        ------
        ValueError
            If the header states no field of view along an axis the trajectory
            carries.
        """
        if self.trajectory is None:
            return None
        dimensions = self.trajectory.shape[0]
        fov = self.space.recon_fov or ()
        if len(fov) < dimensions:
            raise ValueError(
                f"a trajectory of {dimensions} axes needs the field of view along "
                f"each, and encoding space {self.space.index} states "
                f"{len(fov)}"
            )
        grid = np.zeros((*self.trajectory.shape[1:], 3), dtype=np.float32)
        grid[..., :dimensions] = (
            np.moveaxis(self.trajectory, 0, -1) * fov[::-1][:dimensions]
        )
        return grid

    def points(self, **where: int) -> Any:
        """Return the trajectory at one position, indexed as :meth:`select` indexes.

        Returns
        -------
        ndarray or None
            ``(dimensions, ...)`` over the axes not fixed, or ``None`` when no
            acquisition carried a trajectory.
        """
        if self.trajectory is None:
            return None
        return self.trajectory[(slice(None), *self._picks(where))]

    def _picks(self, where: Mapping[str, int]) -> tuple[Any, ...]:
        """Index the placement axes, accepting only index 0 for an axis this space does not vary."""
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
    """Every encoding space of a scan, mapping space index to :class:`ReconBuffer`.

    A buffer is allocated when first indexed or when an acquisition first names
    its space, sized in the latter case by that acquisition's coils and samples.
    ``len`` and iteration cover allocated buffers only. Calibration acquired on
    its own geometry is a separate subsequence and so a separate space;
    calibration within a space is marked in :attr:`ReconBuffer.reference`.

    Parameters
    ----------
    spaces
        Encoding spaces of the scan.
    dtype
        Complex dtype of the k-space arrays.

    Attributes
    ----------
    spaces : dict
        Encoding spaces by index.
    data : dict
        Allocated buffers by index.

    Examples
    --------
    >>> import pulserver.recon as recon
    >>> import pulserver.mrd as mrd
    >>> space = mrd.EncodingSpace(
    ...     index=0, coils=4, readout=64, phase_encodes=32, partitions=1,
    ...     loops=("slice",), loop_sizes=(2,), recon_matrix=(32, 32),
    ... )
    >>> data = recon.ReconData([space])
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
        """Lay out every encoding space of a parsed MRD header.

        A header describing none gives a container whose :meth:`add` places nothing.
        """
        return cls(EncodingSpace.all_from_header(header), dtype=dtype)

    def buffer(
        self,
        encoding: int = 0,
        *,
        coils: int | None = None,
        readout: int | None = None,
    ) -> ReconBuffer:
        """Return the buffer of one encoding space, allocating it on first use.

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
        """Place one acquisition in the space its ``encoding_space_ref`` names.

        See :meth:`ReconBuffer.add`.
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
