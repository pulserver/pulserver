"""Slice/SMS acquisition grouping and frequency-offset helpers."""

from __future__ import annotations

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
    >>> from pulserver.pypulseq import make_slice_groups
    >>> group = make_slice_groups(6, 3e-3, sms_factor=2)[0]
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
        >>> from pulserver.pypulseq import make_slice_groups
        >>> group = make_slice_groups(6, 3e-3, sms_factor=2)[0]
        >>> group.frequency_offsets_hz(42_000.0)
        array([-315.,   63.])
        """
        gradient_hz_per_m = float(gradient_hz_per_m)
        if not np.isfinite(gradient_hz_per_m):
            raise ValueError("gradient_hz_per_m must be finite")
        return self.positions_m * gradient_hz_per_m


def make_slice_groups(num_slices, spacing_m, *, order="interleaved", sms_factor=1):
    """Plan the slice loop: acquisition order, SMS banding, and positions.

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
    tuple of SliceGroup
        One entry per excitation, in acquisition order.

    Examples
    --------
    >>> from pulserver.pypulseq import make_slice_groups
    >>> groups = make_slice_groups(6, 3e-3)
    >>> [group.group_index for group in groups]
    [0, 2, 4, 1, 3, 5]
    >>> sms = make_slice_groups(6, 3e-3, sms_factor=2)
    >>> len(sms), sms[0].slice_indices
    (3, (0, 3))

    Drive the slice loop and set the excitation frequency per group::

        for group in make_slice_groups(32, 5e-3):
            excitation(seq, freq_offset_hz=group.frequency_offsets_hz(gz.amplitude)[0])

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
    return tuple(groups)
