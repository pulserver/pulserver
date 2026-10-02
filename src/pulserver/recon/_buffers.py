"""K-space buffers of a reconstruction unit, laid out from the MRD header and filled one acquisition at a time.

A unit holds one buffer for its imaging readouts and one for its
parallel-imaging calibration readouts, each allocated when its first readout
arrives and indexed by that readout's counters. Axes run coil first and
readout last, with a placement axis only where the header says a counter
varies.
"""

from __future__ import annotations

__all__ = ["ReconBuffer", "ReconData"]

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from ..mrd._header import LOOP_COUNTERS, EncodingSpace
from ..mrd._metadata import acquisition_label, has_acquisition_flag

#: Every name :meth:`ReconBuffer.select` accepts, whether or not a space varies it.
_AXIS_NAMES = frozenset((*LOOP_COUNTERS, "segment", "partition", "phase_encode"))


class ReconBuffer:
    """K-space of one encoding space, filled one acquisition at a time.

    A readout shorter than the buffer is right-aligned, where a partial echo's
    samples belong. The axes are those of ``space``: its counters that vary
    and were asked for, then the encoded axes.

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
        known. A readout placed where one already is replaces it, with a warning
        the first time: the sequence labels neither apart.

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
        if not self._overwrote and self.mask[(*where, readout)].any():
            self._overwrote = True
            warnings.warn(
                f"a readout replaced one already placed at {dict(zip(self.axes[1:-1], where, strict=False))} "
                f"of encoding space {self.space.index}: readouts that share their "
                "encoding counters overwrite each other; label each with its line, "
                "partition, slice, average or repetition (pypulseqpp.make_label)",
                stacklevel=2,
            )
        self.kspace[(slice(0, coils), *where, readout)] = data
        self.mask[(*where, readout)] = True
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
        0. A trajectory whose samples are not those placed, as after a gadget
        resampled the readout, is not stored.
        """
        traj = getattr(acquisition, "traj", None)
        if traj is None:
            return
        traj = np.asarray(traj)
        if traj.size == 0 or traj.shape[0] != readout.stop - readout.start:
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
        ``data``; ``None`` when the unit has none. A readout flagged as
        calibration and imaging is in both buffers; one flagged as
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

    Attributes
    ----------
    data : ReconData
        What the unit has collected, which is what the plugin receives.
    """

    def __init__(
        self,
        key: tuple[str, int, tuple[tuple[str, int], ...]],
        spaces: Mapping[int, EncodingSpace],
        *,
        buffered: bool = True,
        dtype: Any = np.complex64,
    ) -> None:
        branch, self.space_index, counters = key
        self.key = key
        self.spaces = spaces
        self.buffered = buffered
        self.dtype = dtype
        self.data = ReconData(branch, counters=dict(counters))

    @property
    def combinations(self) -> int:
        """Number of positions along the unit's axes, 1 where its space is not described."""
        space = self.spaces.get(self.space_index)
        return 1 if space is None else math.prod(space.loop_sizes)

    def add_acquisition(self, acquisition: Any, readout: Any) -> None:
        """Record one acquisition and place ``readout`` in the buffers its flags select.

        A buffer is allocated at this readout's coils and samples when it is
        the first the unit places in it.

        Parameters
        ----------
        acquisition
            The acquisition, for its counters and flags.
        readout
            ``(coils, samples)`` to place: the acquisition's data as the
            gadgets left it.

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
        in_data, in_ref = readout_roles(acquisition)
        if in_data:
            self.data.data = self._place(self.data.data, acquisition, readout)
        if in_ref:
            self.data.ref = self._place(self.data.ref, acquisition, readout)

    def _place(
        self, buffer: ReconBuffer | None, acquisition: Any, readout: Any
    ) -> ReconBuffer:
        if buffer is None:
            if self.space_index not in self.spaces:
                raise KeyError(
                    f"the header describes no encoding space {self.space_index}; "
                    f"it has {sorted(self.spaces)}"
                )
            coils, samples = np.shape(readout)[-2:]
            buffer = ReconBuffer(
                self.spaces[self.space_index],
                coils=coils,
                readout=samples,
                dtype=self.dtype,
            )
        buffer.add(acquisition, readout)
        return buffer


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
