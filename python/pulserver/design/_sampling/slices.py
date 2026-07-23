"""The slice/SMS scan loop."""

from __future__ import annotations

import numpy as np

from ._axes import EncodingAxis
from ._scanloop import ScanLoop
from .ordering import calc_traversal_order


def make_slice_loop(num_slices, spacing_m, *, order="interleaved", sms_factor=1):
    """Build the physical slice/SMS loop of a 2D acquisition.

    Slices are laid out symmetrically about isocentre and grouped into
    ``num_slices // sms_factor`` excitations — one shot each. Bands of an SMS
    shot are taken one group-count apart, so they are maximally separated in
    space. The result is already in acquisition order.

    Positions are in **metres**, so :meth:`~pulserver.ScanLoop.to_frequencies`
    converts them to the RF offsets that select them once the excitation's
    selection gradient exists. The *label* of a slice is its position index,
    ``loop.shots[i]``, not the position itself.

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
        Order in which the excitations are played.
    sms_factor : int, optional
        Simultaneously excited bands per shot (1 = conventional).

    Returns
    -------
    ScanLoop
        ``positions`` of physical slice positions (m) in slice order, and one
        shot per excitation in acquisition order.

    Raises
    ------
    ValueError
        If ``num_slices`` is not a positive multiple of ``sms_factor``,
        ``spacing_m`` is zero or non-finite, or ``order`` is unknown.

    Examples
    --------
    >>> from pulserver.design import make_slice_loop
    >>> loop = make_slice_loop(6, 3e-3)
    >>> [int(shot[0]) for shot in loop.shots]
    [0, 2, 4, 1, 3, 5]

    An SMS loop excites several slices per shot; the positions show how far
    apart the bands are:

    >>> sms = make_slice_loop(6, 3e-3, sms_factor=2)
    >>> sms.shot_lengths
    (2, 2, 2)
    >>> sms[0].ravel().round(4).tolist()
    [-0.0075, 0.0015]

    Drive the slice loop and set the excitation frequency per shot::

        offsets = slices.to_frequencies(excitation.gradients[0].amplitude)
        for shot in slices.shots:
            excitation.set_state(freq_offset_hz=float(offsets[shot[0]]))
            excitation.set_labels(pp.make_label(type="SET", label="SLC", value=int(shot[0])))
            excitation.add_to(seq)

    Pass the whole row instead of ``[0]`` for an SMS multiband pulse.

    The slice loop is deliberately independent of the k-space loop: nest them
    in whichever order the sequence needs. ``spacing_m`` fixes the physical
    positions, and the selection-gradient amplitude converts those positions
    to the RF frequency offsets.

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       import numpy as np
       import pulserver.design as design
       import pulserver.pypulseq as pp
       loop = design.make_slice_loop(12, 4e-3, sms_factor=3)
       membership = np.zeros((len(loop), loop.n_positions), dtype=float)
       for row, shot in enumerate(loop.shots):
           membership[row, shot] = 1.0
       fig, (mask, offsets) = plt.subplots(1, 2, figsize=(8, 3.2))
       mask.imshow(membership, cmap="gray", origin="lower", aspect="auto")
       mask.set(xlabel="physical slice", ylabel="excitation", title="SMS selection mask")
       table = loop.to_frequencies(1_000.0)
       offsets.plot([table[shot] for shot in loop.shots], marker="o")
       offsets.set(xlabel="excitation", ylabel="RF offset [Hz]", title="offsets from spacing")
       fig.tight_layout()

    See Also
    --------
    pulserver.ScanLoop.to_frequencies : offsets that select each position.
    pulserver.design._lowlevel.calc_traversal_order : the underlying 1D orderings.
    """
    num_slices, sms_factor = int(num_slices), int(sms_factor)
    spacing_m = float(spacing_m)
    if num_slices <= 0 or sms_factor <= 0 or num_slices % sms_factor:
        raise ValueError("num_slices must be positive and divisible by sms_factor")
    if not np.isfinite(spacing_m) or spacing_m == 0:
        raise ValueError("spacing_m must be finite and nonzero")
    if order == "random":
        raise ValueError("slice order must be interleaved, sequential, reverse, center_out, or outside_in")

    n_groups = num_slices // sms_factor
    positions = (np.arange(num_slices) - (num_slices - 1) / 2.0) * spacing_m
    shots = tuple(
        np.array([group + band * n_groups for band in range(sms_factor)], dtype=np.intp)
        for group in calc_traversal_order(n_groups, order)
    )
    return ScanLoop(
        positions[:, None],
        shots,
        None,
        (EncodingAxis("SLC", kind="position", size=num_slices),),
    )
