"""The loop over any counter that is not k-space and not slices."""

from __future__ import annotations

import numpy as np

from ._axes import EncodingAxis
from ._scanloop import ScanLoop
from .ordering import calc_traversal_order

__all__ = ["make_counter_loop"]


def make_counter_loop(values, *, label, order="sequential", kind=None, size=None) -> ScanLoop:
    """Build the loop over a frame, contrast, average, or any other counter.

    k-space and slices are not special. An encoding position is a set of
    frequency offsets, gradient scalings and rotations, and the *counter* that
    labels it is free: a dynamic series varies ``REP``, a multi-contrast scan
    varies ``SET`` or ``PHS``, a repeated acquisition varies ``AVG``. This
    factory builds those loops with the same object the k-space and slice
    factories return, so they nest with each other and label themselves the
    same way.

    Two forms, chosen by what ``values`` is:

    - **A count.** ``make_counter_loop(40, label="REP")`` — a bare counter.
      Positions *are* the indices, one shot each.
    - **A table of values.** ``make_counter_loop([0.1, 0.3, 1.0], label="SET")``
      — an inversion-time list, a b-value list, a saturation-offset list. The
      positions carry the physical numbers the sequence needs; the counter is
      still the index, because that is what the reconstruction sorts by.

    The loop does not know what its values *drive*. Converting an inversion
    time into a delay, or a saturation offset into ``freq_offset_hz``, is the
    sequence's business — which is exactly why one factory covers every
    contrast dimension instead of one factory per physics.

    Parameters
    ----------
    values : int or array_like
        Number of positions, or the per-position values themselves.
    label : str
        Counter to emit — any of :data:`~pulserver.pypulseq.COUNTER_LABELS`,
        or a custom string. ``REP`` for frames, ``SET`` or ``PHS`` for
        contrasts, ``AVG`` for averages.
    order : {'sequential', 'interleaved', 'reverse', 'center_out', 'outside_in'}, optional
        Order the positions are visited in; the same orderings
        :func:`~pulserver.design._lowlevel.calc_traversal_order` supplies. The
        counter follows the *position*, so reordering never renumbers the
        data.
    kind : str, optional
        Position semantics; defaults to ``'index'`` for a count and
        ``'value'`` for a table. Pass ``'position'`` for metres, so
        :meth:`~pulserver.ScanLoop.to_frequencies` applies.
    size : int, optional
        Full extent of the counter, when this loop covers only part of it
        (an interrupted or resumed dynamic series). Defaults to the number of
        positions.

    Returns
    -------
    ScanLoop
        One shot per position, in acquisition order, over a single axis.

    Raises
    ------
    ValueError
        If ``values`` is neither a positive count nor a 1D table, or ``order``
        is unknown.

    Examples
    --------
    The frame loop of a dynamic acquisition:

    >>> import pulserver.design as design
    >>> import pulserver.pypulseq as pp
    >>> frames = design.make_counter_loop(40, label="REP")
    >>> len(frames), frames.label_limits()
    (40, {'REP': (0, 39)})

    A contrast loop that carries the inversion times it schedules:

    >>> import pulserver.design as design
    >>> ti = design.make_counter_loop([0.1, 0.3, 1.0, 2.5], label="SET")
    >>> float(ti[2][0, 0]), [(e.label, e.value) for e in ti.labels(2)]
    (1.0, [('SET', 2)])

    Nest it like any other loop — the counter is emitted once per shot, and
    the readout keeps emitting its own ``LIN``/``PAR``/``ECO``::

        for f in range(len(frames)):
            for c in range(len(contrasts)):
                inversion.set_state(...).set_labels(*frames.labels(f), *contrasts.labels(c))
                inversion.add_to(seq)
                for shot in sampling:
                    readout.set_state(lin_idx=int(shot[0, 0])).add_to(seq)

    See Also
    --------
    pulserver.design.make_slice_loop : the same object for the slice axis.
    pulserver.design._lowlevel.calc_traversal_order : the underlying 1D orderings.
    pulserver.ScanLoop.labels : what makes the counter reach the recon.
    """
    if np.ndim(values) == 0:
        count = int(values)
        if count <= 0:
            raise ValueError("values must be a positive count or a 1D table")
        positions = np.arange(count, dtype=np.intp)[:, None]
        kind = "index" if kind is None else kind
    else:
        positions = np.asarray(values, dtype=float)
        if positions.ndim != 1 or not positions.size:
            raise ValueError("values must be a positive count or a 1D table")
        count = int(positions.size)
        positions = positions[:, None]
        kind = "value" if kind is None else kind

    shots = tuple(np.array([index], dtype=np.intp) for index in calc_traversal_order(count, order))
    return ScanLoop(
        positions,
        shots,
        None,
        (EncodingAxis(label, kind=kind, size=count if size is None else int(size)),),
    )
