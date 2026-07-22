"""Slice/SMS acquisition grouping and frequency-offset helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from . import ordering


@dataclass(frozen=True)
class SliceGroup:
    """One excitation's worth of slices, with their physical positions.

    For conventional imaging a group holds a single slice; under simultaneous
    multi-slice (SMS) it holds one slice per band, all excited together. The
    group keeps its *logical* acquisition index separate from the *physical*
    slice indices, so a sequence can label acquisitions by position while
    looping in interleaved or centre-out order.

    Attributes
    ----------
    group_index : int
        Logical index of this group within the acquisition order.
    slice_indices : tuple of int
        Physical slice indices excited together (one per SMS band).
    positions_m : numpy.ndarray
        Read-only slice positions (m) relative to isocentre, one per index.

    Examples
    --------
    >>> from pulserver.pypulseq import make_slice_sampling
    >>> group = make_slice_sampling(6, 3e-3, sms_factor=2)[0]
    >>> group.slice_indices
    (0, 3)
    >>> group.positions_m
    array([-0.0075,  0.0015])
    """

    group_index: int
    slice_indices: tuple[int, ...]
    positions_m: np.ndarray

    def __post_init__(self):
        positions = np.array(self.positions_m, dtype=float, copy=True)
        if positions.ndim != 1 or len(positions) != len(self.slice_indices) or not np.all(np.isfinite(positions)):
            raise ValueError("positions_m must be a finite 1D value per slice")
        positions.setflags(write=False)
        object.__setattr__(self, "positions_m", positions)

    def frequency_offsets_hz(self, gradient_hz_per_m: float) -> np.ndarray:
        """Return the RF frequency offset (Hz) that selects each slice.

        Under a selection gradient of ``gradient_hz_per_m``, position ``z``
        resonates at ``z * gradient_hz_per_m``. Set the result as the RF
        event's ``freq_offset`` — one value for conventional imaging, one per
        band for an SMS multiband pulse.

        Parameters
        ----------
        gradient_hz_per_m : float
            Slice-selection gradient amplitude in Pulseq units (Hz/m).

        Returns
        -------
        numpy.ndarray
            Frequency offsets (Hz), one per slice in the group.

        Examples
        --------
        >>> from pulserver.pypulseq import make_slice_sampling
        >>> group = make_slice_sampling(6, 3e-3, sms_factor=2)[0]
        >>> group.frequency_offsets_hz(42_000.0)
        array([-315.,   63.])
        """
        gradient_hz_per_m = float(gradient_hz_per_m)
        if not np.isfinite(gradient_hz_per_m):
            raise ValueError("gradient_hz_per_m must be finite")
        return self.positions_m * gradient_hz_per_m


@dataclass(frozen=True)
class SliceSampling(Sequence[SliceGroup]):
    """Physical slice/SMS selection support and its independent schedule.

    ``mask`` has one row per excitation in *acquisition order* and one column
    per physical slice. A conventional schedule has one true value per row;
    SMS has ``sms_factor`` true values per row. Every physical slice is
    covered exactly once: SMS changes simultaneous banding, not the phase
    encoding mask or the physical slice positions.
    """

    positions_m: np.ndarray
    groups: tuple[SliceGroup, ...]
    sms_factor: int

    def __post_init__(self):
        positions = np.array(self.positions_m, dtype=float, copy=True)
        groups = tuple(self.groups)
        sms_factor = int(self.sms_factor)
        if positions.ndim != 1 or not len(positions) or not np.all(np.isfinite(positions)):
            raise ValueError("positions_m must be a non-empty finite 1D array")
        if sms_factor < 1 or len(positions) % sms_factor or len(groups) != len(positions) // sms_factor:
            raise ValueError("groups and sms_factor do not match positions_m")
        indices = tuple(index for group in groups for index in group.slice_indices)
        if sorted(indices) != list(range(len(positions))):
            raise ValueError("groups must cover every physical slice exactly once")
        if any(not np.array_equal(group.positions_m, positions[list(group.slice_indices)]) for group in groups):
            raise ValueError("group positions_m must match the physical slice positions")
        positions.setflags(write=False)
        object.__setattr__(self, "positions_m", positions)
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "sms_factor", sms_factor)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        return self.groups[index]

    @property
    def mask(self) -> np.ndarray:
        """Boolean physical-slice selection mask, one row per excitation."""
        mask = np.zeros((len(self), len(self.positions_m)), dtype=bool)
        for row, group in enumerate(self.groups):
            mask[row, group.slice_indices] = True
        return mask

    def frequency_table_hz(self, gradient_hz_per_m: float) -> np.ndarray:
        """Return RF offsets (Hz) for every scheduled conventional/SMS shot."""
        return np.stack([group.frequency_offsets_hz(gradient_hz_per_m) for group in self.groups])


def make_slice_sampling(num_slices, spacing_m, *, order="interleaved", sms_factor=1):
    """Build the independent physical slice/SMS selection schedule.

    Slices are laid out symmetrically about isocentre and grouped into
    ``num_slices // sms_factor`` excitations. Bands of an SMS group are taken
    one group-count apart, so they are maximally separated in space. The
    returned tuple is already in acquisition order — iterate it directly.

    ``"interleaved"`` (the default) acquires even slices then odd ones, leaving
    the largest possible gap between consecutive excitations and so minimising
    slice cross-talk from imperfect profiles.

    Parameters
    ----------
    num_slices : int
        Total number of slices; must be divisible by ``sms_factor``.
    spacing_m : float
        Centre-to-centre slice spacing (m); sign flips the position axis.
    order : {'interleaved', 'sequential', 'reverse', 'center_out', 'outside_in'}, optional
        Order in which the *groups* are acquired.
    sms_factor : int, optional
        Simultaneously excited bands per group (1 = conventional).

    Returns
    -------
    SliceSampling
        Physical slice positions, one selection-mask row per excitation, and
        one :class:`SliceGroup` per row in acquisition order.

    Examples
    --------
    >>> from pulserver.pypulseq import make_slice_sampling
    >>> schedule = make_slice_sampling(6, 3e-3)
    >>> [group.group_index for group in schedule]
    [0, 2, 4, 1, 3, 5]
    >>> sms = make_slice_sampling(6, 3e-3, sms_factor=2)
    >>> sms.mask.astype(int)
    array([[1, 0, 0, 1, 0, 0],
           [0, 1, 0, 0, 1, 0],
           [0, 0, 1, 0, 0, 1]])
    (3, (0, 3))

    Drive the slice loop and set the excitation frequency per group::

        for group in make_slice_sampling(32, 5e-3):
            excitation(seq, freq_offset_hz=group.frequency_offsets_hz(gz.amplitude)[0])

    The slice selection mask is deliberately distinct from a Cartesian
    ``(ky, kz)`` mask and its echo-train ordering. ``spacing_m`` fixes the
    physical positions, and the selection-gradient amplitude later converts
    those positions to the RF frequency offsets.

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       import pulserver.pypulseq as pp
       schedule = pp.make_slice_sampling(12, 4e-3, sms_factor=3)
       fig, (mask, offsets) = plt.subplots(1, 2, figsize=(8, 3.2))
       mask.imshow(schedule.mask, cmap="gray", origin="lower", aspect="auto")
       mask.set(xlabel="physical slice", ylabel="excitation", title="SMS selection mask")
       offsets.plot(schedule.frequency_table_hz(1_000).T, marker="o")
       offsets.set(xlabel="excitation", ylabel="RF offset [Hz]", title="offsets from spacing")
       fig.tight_layout()

    See Also
    --------
    SliceGroup.frequency_offsets_hz : offsets that select a group's slices.
    make_outer_product : combine the slice loop with other outer dimensions.
    """
    num_slices, sms_factor = int(num_slices), int(sms_factor)
    spacing_m = float(spacing_m)
    if num_slices <= 0 or sms_factor <= 0 or num_slices % sms_factor:
        raise ValueError("num_slices must be positive and divisible by sms_factor")
    if not np.isfinite(spacing_m) or spacing_m == 0:
        raise ValueError("spacing_m must be finite and nonzero")
    n_groups = num_slices // sms_factor
    functions = {
        "sequential": ordering.sequential,
        "reverse": ordering.reverse,
        "interleaved": ordering.interleaved,
        "center_out": ordering.center_out,
        "outside_in": ordering.outside_in,
    }
    if order not in functions:
        raise ValueError(f"unknown slice order {order!r}")
    positions = (np.arange(num_slices) - (num_slices - 1) / 2.0) * spacing_m
    groups = []
    for group_index in functions[order](n_groups):
        members = tuple(int(group_index + band * n_groups) for band in range(sms_factor))
        groups.append(SliceGroup(int(group_index), members, positions[list(members)]))
    return SliceSampling(positions, tuple(groups), sms_factor)
