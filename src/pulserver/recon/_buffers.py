"""K-space buffers of a reconstruction unit, laid out from the MRD header and filled one acquisition at a time.

A unit holds one buffer for its imaging readouts and one for its
parallel-imaging calibration readouts. Axes run coil first and readout last,
with a placement axis only where the header says a counter varies. Readouts
of a Cartesian space are placed by their echo along the readout and by the
offset of their lines from the k-space centre along the encoded axes, as
Gadgetron's acquisition bucket places them.
"""

from __future__ import annotations

__all__ = ["ReconBuffer", "ReconData"]

import math
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..mrd._header import LOOP_COUNTERS, EncodingSpace
from ..mrd._metadata import acquisition_label, has_acquisition_flag

#: Every name :meth:`ReconBuffer.select` accepts, whether or not a space varies it.
_AXIS_NAMES = frozenset((*LOOP_COUNTERS, "segment", "partition", "phase_encode"))

#: The MRD counter each placement axis is read from.
_COUNTERS = {
    "partition": "kspace_encode_step_2",
    "phase_encode": "kspace_encode_step_1",
}

#: The placement axes a buffer can be cropped along.
_CROPPABLE = ("partition", "phase_encode")


def echo_centre(acquisition: Any, samples: int) -> int:
    """Return the index of the echo in a readout of ``samples`` samples.

    The acquisition's ``center_sample``; ``samples // 2`` when it states none,
    which is the echo of a centred full echo.
    """
    centre = acquisition_label(acquisition, "center_sample", None)
    return samples // 2 if centre is None else int(centre)


def discards(acquisition: Any) -> tuple[int, int]:
    """Return the ``(discard_pre, discard_post)`` sample counts of an acquisition, 0 where it states none."""
    pre = int(acquisition_label(acquisition, "discard_pre", 0) or 0)
    post = int(acquisition_label(acquisition, "discard_post", 0) or 0)
    return pre, post


def _as_readout(data: Any) -> np.ndarray:
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(
            f"acquisition data must be (coils, samples), got shape {data.shape}"
        )
    return data


def _grid_position(space: EncodingSpace, acquisition: Any) -> tuple[int, ...]:
    """Return the index of an acquisition on each varying placement axis of ``space``'s grid.

    Raises
    ------
    ValueError
        If a counter, shifted by the offset of its axis, lies outside the grid
        the header laid this space out for.
    """
    where: list[int] = []
    for name, extent in space.extents:
        counter = _COUNTERS.get(name, name)
        index = int(acquisition_label(acquisition, counter, 0) or 0)
        shift = space.offset(name)
        if not 0 <= index + shift < extent:
            moved = (
                f" ({index + shift} once placed about the limits' centre)"
                if shift
                else ""
            )
            raise ValueError(
                f"acquisition has {counter}={index}{moved}, outside the {extent} "
                f"{name} positions encoding space {space.index} was laid out for"
            )
        if extent > 1:
            where.append(index + shift)
    return tuple(where)


def _width(space: EncodingSpace, acquisition: Any, samples: int) -> int:
    """Return the readout samples of a buffer of ``space`` whose first readout is ``acquisition``.

    The readout's own samples when it is a centred full echo, which is how a
    readout completed by :class:`~pulserver.recon.AsymmetricEcho` arrives, or
    when the space is not Cartesian; otherwise the encoded matrix, in which a
    partial echo is placed by its echo.
    """
    if not space.cartesian:
        return max(space.readout, samples)
    centred = echo_centre(acquisition, samples) == samples // 2
    return samples if centred else space.readout


class ReconBuffer:
    """K-space of one encoding space, filled one acquisition at a time.

    Along the readout of a Cartesian space, the samples of an acquisition from
    ``discard_pre`` to ``number_of_samples - discard_post`` are placed so that
    its echo, ``center_sample``, lies at sample ``samples // 2`` of the buffer.
    Along an encoded axis of a Cartesian space, a counter is placed at
    ``counter - center + extent // 2``, ``center`` being the counter of the
    k-space centre the header's limits state, so the centre line lies at
    ``extent // 2`` whatever the counters start at. A non-Cartesian space
    places a readout right-aligned and a counter as it is. The axes are those
    of ``space``: its counters that vary and were asked for, then the encoded
    axes.

    Parameters
    ----------
    space
        Encoding space to lay out.
    coils
        Channels to allocate; ``space.coils`` when not given. Fewer than the
        header declares is valid, for data compressed before placement.
    samples
        Samples to allocate along the readout; ``space.readout`` when not given.
    crop
        ``{axis: (first, stop)}`` for the ``phase_encode`` and ``partition``
        axes: the positions of the space's grid the buffer holds, the whole axis
        where an axis is not named.
    dtype
        Complex dtype of :attr:`kspace`.

    Attributes
    ----------
    kspace : ndarray
        Shaped as :attr:`axes` names.
    mask : ndarray
        Boolean, :attr:`kspace` without the coil axis: samples that were placed.
    origin : dict of str to int
        Position on the space's grid of the first entry of each placement axis
        of :attr:`axes`; 0 along an axis that is not cropped.
    readout : tuple of int or None
        First and last sample of the readout axis placed, inclusive; ``None``
        until an acquisition is placed. The samples outside it are zero fill.
    trajectory : ndarray or None
        ``(dimensions, ...)`` over the axes of :attr:`mask`, in the units the
        acquisitions carry. ``None`` until an acquisition carries a trajectory;
        widened to the most dimensions any acquisition carried, missing trailing
        dimensions reading 0.
    center_sample : int or None
        Echo index along the readout axis. ``samples // 2`` in a Cartesian
        space, where every echo is placed there; in a non-Cartesian space that
        of the first acquisition that states it, after alignment.
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
        samples: int | None = None,
        crop: Mapping[str, tuple[int, int]] | None = None,
        dtype: Any = np.complex64,
    ) -> None:
        self.space = space
        self.coils = int(coils) if coils else space.coils
        crop = dict(crop or {})
        unknown = sorted(set(crop) - set(_CROPPABLE))
        if unknown:
            raise ValueError(f"only {list(_CROPPABLE)} can be cropped, not {unknown}")
        self.origin: dict[str, int] = {}
        extents: list[int] = []
        for name, extent in space.extents:
            if extent <= 1:
                continue
            first, stop = crop.get(name, (0, extent))
            if not 0 <= first < stop <= extent:
                raise ValueError(
                    f"cannot crop the {extent} {name} positions of encoding space "
                    f"{space.index} to {first}..{stop}"
                )
            self.origin[name] = first
            extents.append(stop - first)
        shape = (self.coils, *extents, int(samples) if samples else space.readout)
        self.kspace = np.zeros(shape, dtype=dtype)
        self.mask = np.zeros(shape[1:], dtype=bool)
        self.readout: tuple[int, int] | None = None
        self.trajectory: Any | None = None
        self.center_sample: int | None = None
        self.sample_time: float | None = None
        self.headers: list[Any] = []
        self._overwrote = False

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

    def position(self, acquisition: Any) -> tuple[int, ...]:
        """Index along each placement axis of :attr:`kspace` this acquisition fills.

        Raises
        ------
        ValueError
            If a counter, placed about the limits' centre, lies outside what the
            header laid this space out for, or outside the part of the grid a
            cropped buffer holds.
        """
        grid = _grid_position(self.space, acquisition)
        where = tuple(
            index - self.origin[name]
            for index, name in zip(grid, self.axes[1:-1], strict=True)
        )
        for index, name, size in zip(
            where, self.axes[1:-1], self.kspace.shape[1:-1], strict=True
        ):
            if not 0 <= index < size:
                raise ValueError(
                    f"acquisition lies at {name} position {index + self.origin[name]}, "
                    f"outside the {size} from {self.origin[name]} that this buffer "
                    f"of encoding space {self.space.index} holds"
                )
        return where

    def add(self, acquisition: Any, data: Any = None) -> None:
        """Place one acquisition where its counters and echo say it belongs.

        Also records its trajectory, and ``center_sample`` and dwell when not yet
        known. A readout placed where one already is replaces it, with a warning
        the first time: the sequence labels neither apart.

        Parameters
        ----------
        acquisition
            The acquisition, for its counters, ``center_sample``, discards and
            flags, and its data. An acquisition that states no ``center_sample``
            is a centred full echo.
        data
            ``(coils, samples)`` to place instead of ``acquisition.data``: a
            readout corrected before placement, whose ``center_sample`` and
            discards the acquisition states.

        Raises
        ------
        ValueError
            If the data is not two-dimensional, the readout does not fit the
            buffer along the readout, or a counter is outside the laid-out extent.
        """
        data = _as_readout(acquisition.data if data is None else data)
        coils, samples = data.shape
        width = self.kspace.shape[-1]
        if coils > self.coils:
            raise ValueError(
                f"acquisition has {coils} channels, more than the {self.coils} "
                f"encoding space {self.space.index} was laid out for"
            )

        where = self.position(acquisition)

        if self.space.cartesian:
            centre = echo_centre(acquisition, samples)
            skip, after = discards(acquisition)
            count = samples - skip - after
            offset = width // 2 - (centre - skip)
            detail = f"with its echo at {centre} and {skip} + {after} samples discarded"
        else:
            skip, count, offset = 0, samples, width - samples
            detail = ""
        if count < 1 or offset < 0 or offset + count > width:
            raise ValueError(
                f"a readout of {samples} samples {detail} does not fit the {width} "
                f"samples of encoding space {self.space.index}: its {count} placed "
                f"samples would occupy {offset} to {offset + count - 1}"
            )
        region = slice(offset, offset + count)
        if not self._overwrote and self.mask[(*where, region)].any():
            self._overwrote = True
            warnings.warn(
                f"a readout replaced one already placed at {dict(zip(self.axes[1:-1], where, strict=False))} "
                f"of encoding space {self.space.index}: readouts that share their "
                "encoding counters overwrite each other; label each with its line, "
                "partition, slice, average or repetition (pypulseqpp.make_label)",
                stacklevel=2,
            )
        self.kspace[(slice(0, coils), *where, region)] = data[:, skip : skip + count]
        self.mask[(*where, region)] = True
        self._place_trajectory(acquisition, where, region, skip, samples)
        self.readout = (
            (region.start, region.stop - 1)
            if self.readout is None
            else (
                min(self.readout[0], region.start),
                max(self.readout[1], region.stop - 1),
            )
        )
        if self.center_sample is None:
            if self.space.cartesian:
                self.center_sample = width // 2
            else:
                center = acquisition_label(acquisition, "center_sample", None)
                if center is not None:
                    self.center_sample = int(center) + offset
        if self.sample_time is None:
            dwell = acquisition_label(acquisition, "sample_time_us", None)
            if dwell:
                self.sample_time = float(dwell) * 1e-6
        self.headers.append(acquisition)

    def _place_trajectory(
        self, acquisition: Any, where: tuple, region: slice, skip: int, samples: int
    ) -> None:
        """Store the acquisition's trajectory, if it carries one.

        MRD trajectories are ``(samples, dimensions)`` and are stored transposed.
        A trajectory with a row for each of the readout's ``samples`` is cut to
        the samples placed; one with a row for each sample placed is stored as it
        is. A trajectory with any other count, as after a gadget resampled the
        readout, is not stored. An acquisition may carry fewer trailing
        dimensions than its neighbours -- the centre partition of a slab
        traverses no kz -- and the rows it omits stay 0.
        """
        traj = getattr(acquisition, "traj", None)
        if traj is None:
            return
        traj = np.asarray(traj)
        if traj.size == 0:
            return
        count = region.stop - region.start
        if traj.shape[0] == samples:
            traj = traj[skip : skip + count]
        elif traj.shape[0] != count:
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
        self.trajectory[(slice(0, dimensions), *where, region)] = traj.T

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
    def reference(self) -> Any:
        """The placed acquisition nearest the k-space centre, ``None`` while none is placed.

        Nearness is the Euclidean distance, in positions on the grid of the
        space, from ``extent // 2``, where the buffer places the k-space centre,
        along ``phase_encode`` and, where the space has more than one partition,
        ``partition``; the first placed of acquisitions at the same distance is
        returned. A :class:`~pulserver.recon.ReconResult` takes the geometry of
        the image from this acquisition unless it names another.
        """
        return nearest_to_centre(self.space, self.headers)

    @property
    def readout_time(self) -> Any:
        """Sample times relative to the echo, ``(readout,)`` in seconds.

        ``None`` until the acquisitions state both a dwell and an echo position.
        """
        if self.sample_time is None or self.center_sample is None:
            return None
        samples = self.kspace.shape[-1]
        return (np.arange(samples) - self.center_sample) * self.sample_time

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


@dataclass(eq=False)
class ReconData:
    """The readouts of one reconstruction unit, as :meth:`ReconPlugin.recon` receives them.

    A unit is the set of readouts of one branch and encoding space that share
    every image counter not listed in the plugin's ``axes``; the plugin
    documents how a unit is closed. A unit is allocated on its first readout
    and released once :meth:`ReconPlugin.recon` has returned.

    Attributes
    ----------
    branch : str
        The branch the unit belongs to.
    data : ReconBuffer or None
        K-space of the imaging readouts, with the unit's ``axes`` along the
        header's encoded axes. ``None`` when the unit placed none: the plugin
        is not ``buffered``, the header describes no encoding space, or every
        readout was calibration only.
    ref : ReconBuffer or None
        K-space of the parallel-imaging calibration readouts, laid out as
        ``data`` but cropped along the phase-encode and partition axes to the
        lines the readouts cover, whose position on the grid is its
        :attr:`~ReconBuffer.origin`; ``None`` when the unit has none. A readout
        flagged as calibration and imaging is in both buffers; one flagged as
        calibration only is in ``ref`` alone; a phase-correction readout is in
        neither.
    counters : dict of str to int
        The unit's image counters not listed in ``axes``, by MRD name.
    waveforms : tuple
        The waveforms received since the previous close; units that close
        together share them.
    acquisitions : list
        Every readout the unit received, as received and in arrival order,
        whichever buffer it was placed in. All a plugin that is not
        ``buffered`` is given.

    Examples
    --------
    >>> import pulserver.recon as recon
    >>> data = recon.ReconData("imaging", counters={"slice": 2})
    >>> data.counters["slice"], data.data is None
    (2, True)
    """

    branch: str
    data: ReconBuffer | None = None
    ref: ReconBuffer | None = None
    counters: dict[str, int] = field(default_factory=dict)
    waveforms: tuple[Any, ...] = ()
    acquisitions: list[Any] = field(default_factory=list, repr=False)


class ReconUnit:
    """The readouts of one unit as they arrive, placed into the buffers of :attr:`data`.

    Imaging readouts are placed as they arrive, into a buffer allocated at the
    first of them. Calibration readouts are held until :meth:`close`, which lays
    them out over the lines they cover.

    Parameters
    ----------
    key
        ``(branch, encoding space, ((counter, value), ...))``, as
        :func:`pulserver.recon._units.unit_key` states it.
    spaces
        The header's encoding spaces by index, laid out along the plugin's
        ``axes``. Empty when the header describes none, so nothing is placed.
    buffered
        Place readouts. ``False`` only records them in :attr:`ReconData.acquisitions`.
    dtype
        Complex dtype of the k-space arrays.
    merged
        Positions of the plugin's ``merge`` counters by encoding space index:
        the combinations of their values that the closing flag has to arrive
        at, in addition to those along the axes. 1 where a space is absent.

    Attributes
    ----------
    data : ReconData
        What the unit has collected, which is what the plugin receives once
        the unit is closed.
    """

    def __init__(
        self,
        key: tuple[str, int, tuple[tuple[str, int], ...]],
        spaces: Mapping[int, EncodingSpace],
        *,
        buffered: bool = True,
        dtype: Any = np.complex64,
        merged: Mapping[int, int] | None = None,
    ) -> None:
        branch, self.space_index, counters = key
        self.key = key
        self.spaces = spaces
        self.buffered = buffered
        self.dtype = dtype
        self.merged = dict(merged or {})
        self.data = ReconData(branch, counters=dict(counters))
        self._reference: list[tuple[Any, np.ndarray, tuple[int, ...]]] = []

    @property
    def combinations(self) -> int:
        """Number of positions along the unit's axes and merged counters, 1 where its space is not described."""
        space = self.spaces.get(self.space_index)
        if space is None:
            return 1
        return math.prod(space.loop_sizes) * self.merged.get(self.space_index, 1)

    def add_acquisition(
        self, acquisition: Any, readout: Any, placed: Any = None
    ) -> None:
        """Record one acquisition and place ``readout`` in the buffers its flags select.

        The first readout the unit places in a buffer sizes it: its channels,
        and along the readout its own samples when it is a centred full echo,
        the encoded matrix's readout otherwise. A calibration readout is held
        until :meth:`close`.

        Parameters
        ----------
        acquisition
            The acquisition as received, for its flags, which select the
            buffers, and for :attr:`ReconData.acquisitions`.
        readout
            ``(coils, samples)`` to place: the acquisition's data as the
            gadgets left it.
        placed
            The acquisition as the gadgets left it, whose ``center_sample`` and
            discards describe ``readout`` and which the buffers keep as their
            header; ``acquisition`` when not given.

        Raises
        ------
        KeyError
            If the header describes encoding spaces and not the one this unit
            is in.
        ValueError
            As :meth:`ReconBuffer.add`.
        """
        self.data.acquisitions.append(acquisition)
        if not (self.buffered and self.spaces):
            return
        readout = _as_readout(readout)
        placed = acquisition if placed is None else placed
        in_data, in_ref = readout_roles(acquisition)
        if in_data:
            self.data.data = self._place(self.data.data, placed, readout)
        if in_ref:
            self._reference.append(
                (placed, readout, _grid_position(self._space(), placed))
            )

    def close(self) -> ReconData:
        """Lay out the calibration readouts and return what the unit collected.

        The calibration buffer covers the lines its readouts do along the
        phase-encode and partition axes, and every position of the other axes.
        """
        if self._reference:
            self.data.ref = self._assemble(self._reference)
            self._reference = []
        return self.data

    def _space(self) -> EncodingSpace:
        if self.space_index not in self.spaces:
            raise KeyError(
                f"the header describes no encoding space {self.space_index}; "
                f"it has {sorted(self.spaces)}"
            )
        return self.spaces[self.space_index]

    def _place(
        self, buffer: ReconBuffer | None, placed: Any, readout: np.ndarray
    ) -> ReconBuffer:
        if buffer is None:
            space = self._space()
            coils, samples = readout.shape
            buffer = ReconBuffer(
                space,
                coils=coils,
                samples=_width(space, placed, samples),
                dtype=self.dtype,
            )
        buffer.add(placed, readout)
        return buffer

    def _assemble(
        self, readouts: list[tuple[Any, np.ndarray, tuple[int, ...]]]
    ) -> ReconBuffer:
        space = self._space()
        placed, readout, _ = readouts[0]
        coils, samples = readout.shape
        varying = [name for name, extent in space.extents if extent > 1]
        grid = np.array([position for _, _, position in readouts], dtype=np.int64)
        crop = {
            name: (int(grid[:, column].min()), int(grid[:, column].max()) + 1)
            for column, name in enumerate(varying)
            if name in _CROPPABLE
        }
        buffer = ReconBuffer(
            space,
            coils=coils,
            samples=_width(space, placed, samples),
            crop=crop,
            dtype=self.dtype,
        )
        for placed, readout, _ in readouts:
            buffer.add(placed, readout)
        return buffer


def nearest_to_centre(space: EncodingSpace, acquisitions: Iterable[Any]) -> Any:
    """Return the acquisition placed nearest the k-space centre of ``space``, ``None`` when there are none.

    Distance is Euclidean over the positions on the grid along ``phase_encode``
    and, where ``space`` has more than one partition, ``partition``, from
    ``extent // 2``, where a buffer of ``space`` places the centre. The earliest
    of equally near acquisitions is returned.
    """
    axes = [
        (name, extent)
        for name, extent in space.extents
        if name in _COUNTERS and extent > 1
    ]

    def distance(acquisition: Any) -> int:
        return sum(
            (
                int(acquisition_label(acquisition, _COUNTERS[name], 0) or 0)
                + space.offset(name)
                - extent // 2
            )
            ** 2
            for name, extent in axes
        )

    return min(acquisitions, key=distance, default=None)


def readout_roles(acquisition: Any) -> tuple[bool, bool]:
    """Return whether an acquisition belongs to a unit's imaging data and to its calibration data.

    A readout flagged as calibration only is calibration; one flagged as
    calibration and imaging is both; a phase-correction readout is neither
    unless it is also calibration; any other readout is imaging.
    """
    calibration = has_acquisition_flag(acquisition, "ACQ_IS_PARALLEL_CALIBRATION")
    both = has_acquisition_flag(acquisition, "ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING")
    phase_correction = has_acquisition_flag(acquisition, "ACQ_IS_PHASECORR_DATA")
    return both or not (calibration or phase_correction), calibration or both
