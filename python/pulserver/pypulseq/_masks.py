"""Cartesian undersampling masks and view orderings.

Plain arrays: a boolean mask over the phase-encode grid, or a list of shots
each holding the view indices it acquires in order. Turning either into a scan
loop belongs to :mod:`pulserver.design`.

References
----------
Echo-train reordering follows Buonincontri et al., "Doubling the repetition
time without paying the price: 3D TSE with individually parameterized echo
trains", ISMRM abstract 566-05-007 (Fig. 2). Random shuffling follows Tamir et
al., "T2 Shuffling", Magn Reson Med 2017;77:180-195. The Poisson-disc kernel is
ported from SigPy's ``sigpy.mri.samp.poisson`` (BSD 3-Clause).
"""

from __future__ import annotations

__all__ = [
    "calc_calibration_lines",
    "calc_sampled_lines",
    "calc_sampled_pairs",
    "make_caipirinha_mask",
    "make_centric_order",
    "make_linear_order",
    "make_poisson_disc_mask",
    "make_radial_adaptive_order",
    "make_radial_order",
    "make_random_mask",
    "make_shuffling_order",
]

import numpy as np

from ._ordering import calc_chunk_indices


def calc_sampled_lines(
    n: int,
    r: int,
    acs_lines: int,
    *,
    order: str = "ascending",
    partial_fourier: float = 1.0,
) -> list[int]:
    """Return the sampled view indices for uniform undersampling + ACS block.

    Every ``r``-th view is sampled, plus a centered block of ``acs_lines``
    autocalibration views.

    Parameters
    ----------
    n : int
        Total number of views.
    r : int
        Acceleration factor (view ``i`` is sampled when ``i % r == 0``).
    acs_lines : int
        Number of fully sampled center views.
    order : str, optional
        ``'ascending'`` traverses k-space from one edge to the other.
        ``'calibration_first'`` puts the autocalibration block ahead of
        everything else, so a reconstruction can estimate coil sensitivities
        from it while the remaining views are still being acquired. It acquires
        the centre of k-space before the magnetisation has reached steady
        state, so a sequence using it wants dummy repetitions first. Default is
        ``'ascending'``.
    partial_fourier : float, optional
        Fraction of the phase-encode extent acquired, in ``(0.5, 1]``. The
        views dropped are the leading ones, so the centre stays in and the
        conjugate symmetry of k-space covers what is missing. It applies to
        the calibration block as well: a view that is not played is not in the
        list, whatever else would have asked for it. Default is 1.0.

    Returns
    -------
    list of int
        Sampled view indices, in acquisition order.

    Raises
    ------
    ValueError
        If ``order`` is neither ``'ascending'`` nor ``'calibration_first'``, or
        ``partial_fourier`` is outside ``(0.5, 1]``.

    Examples
    --------
    >>> import pulserver.pypulseq as pp
    >>> pp.calc_sampled_lines(8, 2, 0)
    [0, 2, 4, 6]

    The calibration block first, then what is left of the ascending traversal:

    >>> pp.calc_sampled_lines(8, 2, 4, order='calibration_first')
    [2, 3, 4, 5, 0, 6]

    Three quarters of the extent, counted from the far edge:

    >>> pp.calc_sampled_lines(8, 1, 0, partial_fourier=0.75)
    [2, 3, 4, 5, 6, 7]
    """
    if order not in ("ascending", "calibration_first"):
        raise ValueError("order must be 'ascending' or 'calibration_first'")

    first = n - round(_checked_partial_fourier(partial_fourier) * n)
    sampled = {i for i in range(first, n) if (i % r) == 0}
    calibration = calc_calibration_lines(n, acs_lines, partial_fourier=partial_fourier)
    sampled.update(calibration)
    if order == "ascending":
        return sorted(sampled)
    return calibration + sorted(sampled.difference(calibration))


def calc_calibration_lines(
    n: int,
    acs_lines: int,
    *,
    partial_fourier: float = 1.0,
) -> list[int]:
    """Return the autocalibration view indices for a phase-encode axis.

    The fully sampled block at the centre of k-space that
    :func:`calc_sampled_lines` acquires whatever the acceleration is -- the
    subset a reconstruction calibrates its coil sensitivities from. It is the
    single source of truth for *which* views are autocalibration views, so the
    count a 2D sequence needs and the rectangle a 3D sequence carves both read
    the same window rather than each re-deriving it.

    Parameters
    ----------
    n : int
        Total number of views.
    acs_lines : int
        Number of fully sampled centre views. Zero (or fewer) means no block.
    partial_fourier : float, optional
        Fraction of the phase-encode extent acquired, in ``(0.5, 1]``. A
        calibration view the partial-Fourier truncation drops is not returned,
        matching :func:`calc_sampled_lines`. Default is 1.0.

    Returns
    -------
    list of int
        The autocalibration view indices, ascending.

    Raises
    ------
    ValueError
        If ``partial_fourier`` is outside ``(0.5, 1]``.

    Examples
    --------
    >>> import pulserver.pypulseq as pp
    >>> pp.calc_calibration_lines(8, 4)
    [2, 3, 4, 5]
    """
    first = n - round(_checked_partial_fourier(partial_fourier) * n)
    if acs_lines <= 0:
        return []
    center = n // 2
    start = max(0, center - acs_lines // 2)
    stop = min(n, start + acs_lines)
    return [i for i in range(start, stop) if i >= first]


def calc_sampled_pairs(
    shape: tuple[int, int],
    acceleration: tuple[int, int],
    calibration: tuple[int, int],
    *,
    partial_fourier: tuple[float, float] = (1.0, 1.0),
    caipi_shift: int = 0,
    order: str = "calibration_first",
) -> tuple[list[tuple[int, int]], int]:
    """Return the ``(line, partition)`` pairs a 2D phase-encode grid samples.

    The 3D counterpart of :func:`calc_sampled_lines`: a CAIPIRINHA lattice
    (:func:`make_caipirinha_mask`) on the two phase-encode axes, a fully
    sampled autocalibration *rectangle* at the centre of k-space, and partial
    Fourier on either axis. ``caipi_shift`` is the per-``ky``-block shift the
    lattice applies along kz; ``0`` degenerates to a plain ``r_y x r_z``
    grid.

    ``'calibration_first'`` leads the traversal with the rectangle -- every
    sampled pair whose line *and* partition are both autocalibration views --
    partitions outer and lines inner, then the remaining pairs in the same
    nesting. This is what lets a reconstruction calibrate the moment the
    rectangle is complete rather than at the end of the scan. The rectangle is
    carved out explicitly because a product of two per-axis calibration-first
    orders would interleave autocalibration and non-autocalibration pairs.

    Parameters
    ----------
    shape : tuple of int
        ``(n_y, n_z)``, the phase-encode and partition-encode counts.
    acceleration : tuple of int
        ``(r_y, r_z)``, the uniform undersampling factor on each axis.
    calibration : tuple of int
        ``(acs_y, acs_z)``, the autocalibration extent on each axis, in views.
    partial_fourier : tuple of float, optional
        ``(pf_y, pf_z)``, the acquired fraction of each axis in ``(0.5, 1]``.
        Default is ``(1.0, 1.0)``.
    caipi_shift : int, optional
        CAIPIRINHA shift along kz per sampled-ky block, ``0 <= caipi_shift <
        r_z``. ``0`` (the default) is a regular lattice; a non-zero shift
        spreads the aliasing into both phase-encode directions. Default is 0.
    order : str, optional
        ``'calibration_first'`` (the default) leads with the rectangle;
        ``'ascending'`` traverses the whole grid partitions-outer,
        lines-inner without pulling the rectangle forward.

    Returns
    -------
    pairs : list of tuple of int
        ``(line, partition)`` in acquisition order.
    n_calibration : int
        How many leading pairs make up the autocalibration rectangle. With
        ``order='ascending'`` this still reports the rectangle's size, though
        those pairs are not contiguous at the front.

    Raises
    ------
    ValueError
        If ``order`` is not one of the two recognised values, or a
        ``partial_fourier`` entry is outside ``(0.5, 1]``.

    Examples
    --------
    >>> import pulserver.pypulseq as pp
    >>> pairs, n_cal = pp.calc_sampled_pairs((4, 4), (2, 2), (2, 2))
    >>> n_cal
    4
    >>> pairs[:n_cal]
    [(1, 1), (2, 1), (1, 2), (2, 2)]
    """
    if order not in ("ascending", "calibration_first"):
        raise ValueError("order must be 'ascending' or 'calibration_first'")
    n_y, n_z = shape
    r_y, r_z = acceleration
    acs_y, acs_z = calibration
    pf_y, pf_z = partial_fourier

    first_y = n_y - round(_checked_partial_fourier(pf_y) * n_y)
    first_z = n_z - round(_checked_partial_fourier(pf_z) * n_z)

    # The CAIPI lattice within the partial-Fourier region, plus the fully
    # sampled autocalibration rectangle.
    sampled = make_caipirinha_mask((n_y, n_z), r_y, r_z, delta=caipi_shift)
    sampled[:first_y, :] = False
    sampled[:, :first_z] = False
    acs_lines = calc_calibration_lines(n_y, acs_y, partial_fourier=pf_y)
    acs_partitions = calc_calibration_lines(n_z, acs_z, partial_fourier=pf_z)
    if acs_lines and acs_partitions:
        sampled[np.ix_(acs_lines, acs_partitions)] = True

    acs_line_set = set(acs_lines)
    acs_partition_set = set(acs_partitions)

    def calibrates(line: int, partition: int) -> bool:
        return line in acs_line_set and partition in acs_partition_set

    pairs = [
        (int(line), int(partition))
        for partition in range(n_z)
        for line in range(n_y)
        if sampled[line, partition]
    ]
    rectangle = [pair for pair in pairs if calibrates(*pair)]
    if order == "ascending":
        return pairs, len(rectangle)
    body = [pair for pair in pairs if not calibrates(*pair)]
    return rectangle + body, len(rectangle)


def _checked_partial_fourier(partial_fourier: float) -> float:
    """Return ``partial_fourier`` if it is a valid fraction, else raise."""
    if not 0.5 < partial_fourier <= 1.0:
        raise ValueError("partial_fourier must be in (0.5, 1]")
    return partial_fourier


def _as_coords(coords) -> np.ndarray:
    """Normalize a coordinate argument to an ``(N, 2)`` float array."""
    raw = np.asarray(coords)
    if raw.dtype == bool and raw.ndim in (1, 2):
        arr = np.argwhere(raw).astype(float)
    else:
        arr = np.asarray(coords, dtype=float)
    if arr.ndim == 1:
        arr = np.column_stack([arr, np.zeros_like(arr)])
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("coords must be shape (N,) or (N, 2)")
    return arr

def _split_into_shots(order: list[int], etl: int) -> list[list[int]]:
    return calc_chunk_indices(order, etl)


def _polar(pts: np.ndarray, center) -> tuple[np.ndarray, np.ndarray]:
    """Return per-point radius and angle about ``center`` (mean when ``None``)."""
    origin = pts.mean(axis=0) if center is None else np.asarray(center, dtype=float)
    rel = pts - origin
    return np.hypot(rel[:, 0], rel[:, 1]), np.arctan2(rel[:, 1], rel[:, 0])


def _finish_trains(
    trains: list[list[int | None]], etl: int, pad: bool
) -> list[list[int | None]]:
    """Pad each train to ``etl`` with ``None`` (``pad``), or drop the gaps."""
    if pad:
        return [train + [None] * (etl - len(train)) for train in trains]
    return [[index for index in train if index is not None] for train in trains]


def _deal_echo_major(
    order: list[int],
    secondary: np.ndarray,
    n_trains: int,
    etl: int,
    radius: np.ndarray,
    *,
    center_echo: int | None,
    adaptive: bool,
) -> list[list[int | None]]:
    """Deal a ranked index list into echo-major echo trains.

    ``order`` is the primary ranking. It is cut into ``etl`` groups of up to
    ``n_trains`` views; group ``g`` becomes one echo, and within a group the
    views are dealt across trains in ``secondary`` order so successive echoes
    of one train stay neighbours. ``center_echo`` places the group holding the
    k-space centre at that echo -- rolled for a monotone ranking, folded
    (``adaptive``) so the radius grows away from the target echo in both
    directions. This is the shared machinery behind the Cartesian echo-train
    orderings (566-05-007, Fig. 2).
    """
    groups = [order[g * n_trains : (g + 1) * n_trains] for g in range(etl)]
    echo_of_group = list(range(etl))
    if center_echo is not None:
        if adaptive:
            # Innermost group (rank 0) at the target echo, then outward.
            echo_of_group = sorted(range(etl), key=lambda echo: abs(echo - center_echo))
        else:
            centre = int(np.argmin(radius))
            holds_centre = next(
                index for index, group in enumerate(groups) if centre in group
            )
            roll = (holds_centre - center_echo) % etl
            groups = groups[roll:] + groups[:roll]

    trains: list[list[int | None]] = [[None] * etl for _ in range(n_trains)]
    for rank, group in enumerate(groups):
        echo = echo_of_group[rank]
        for train, index in enumerate(sorted(group, key=lambda i: secondary[i])):
            trains[train][echo] = index
    return trains

def make_linear_order(
    coords,
    train_length: int,
    *,
    center=None,
    center_echo: int | None = None,
    pad: bool = False,
) -> list[list[int | None]]:
    """Linear (raster) train ordering over a Cartesian point set.

    Views are ranked in raster order and dealt echo-major into trains: each
    echo is a raster band, and successive echoes of one train stay neighbours.
    This is scheme A ("linear reordering") of ISMRM abstract 566-05-007,
    Fig. 2.

    Parameters
    ----------
    coords : int or array_like
        Number of sequential views, or phase-encode locations with shape
        ``(N,)`` (ky only) or ``(N, 2)`` (ky, kz).
    train_length : int
        Echo-train or segment length.
    center : tuple of float or None, optional
        The k-space centre the target echo is placed on. ``None`` uses the
        centroid of ``coords``; a caller with a fixed grid passes its centre.
    center_echo : int or None, optional
        Echo the k-space centre is acquired at. ``None`` leaves the raster
        bands in order (centre wherever it falls); an integer rolls the bands
        so the one holding the centre plays at that echo -- the effective-TE
        control an echo train wants.
    pad : bool, optional
        When True every train is padded to ``train_length`` with ``None`` so
        the echo index is the position in the train; when False (the default)
        the gaps are dropped and each train holds only its real views.

    Returns
    -------
    list of list of int
        Shots of view indices (indices into ``coords``); echo index is the
        position within the shot.

    Examples
    --------
    >>> import pulserver.pypulseq as pp
    >>> shots = pp.make_linear_order([[0, 0], [1, 0], [0, 1], [1, 1]], 2)
    >>> sorted(i for shot in shots for i in shot)
    [0, 1, 2, 3]

    .. plot::
       :include-source: false

       import numpy as np
       import pulserver.pypulseq as pp
       from _figures import order_figure
       ky, kz = np.meshgrid(np.arange(-16, 16), np.arange(-16, 16))
       coords = np.column_stack([ky.ravel(), kz.ravel()])
       order_figure([("linear", pp.make_linear_order(coords, 32))], coords)
    """
    if np.asarray(coords).ndim == 0:
        count = int(coords)
        if count < 0:
            raise ValueError("coords must be nonnegative when given as a count")
        return _finish_trains(
            _split_into_shots(list(range(count)), train_length), train_length, pad
        )
    pts = _as_coords(coords)
    n = len(pts)
    if n == 0:
        return []
    n_trains = int(np.ceil(n / train_length))
    radius, _ = _polar(pts, center)
    order = sorted(range(n), key=lambda i: (pts[i, 1], pts[i, 0]))
    trains = _deal_echo_major(
        order,
        pts[:, 0],
        n_trains,
        train_length,
        radius,
        center_echo=center_echo,
        adaptive=False,
    )
    return _finish_trains(trains, train_length, pad)

def make_centric_order(
    coords,
    train_length: int,
    *,
    center=None,
    center_echo: int | None = None,
    pad: bool = False,
) -> list[list[int | None]]:
    """Globally center-out Cartesian ordering, dealt into trains.

    This is the conventional centric segmented-GRE/MPRAGE ordering: sampled
    locations are ranked by distance from the encoded k-space centre, so the
    earliest echo of every train clusters near the centre. Unlike
    :func:`make_radial_adaptive_order` the echo bands stay in radius order
    unless ``center_echo`` rolls them to a later effective TE.

    Parameters
    ----------
    coords : array_like
        Phase-encode locations, shape ``(N,)`` or ``(N, 2)``.
    train_length : int
        Echo-train or segment length; the number of shots is
        ``ceil(N / train_length)``.
    center : tuple of float or None, optional
        k-space centre; ``None`` uses the centroid of ``coords``.
    center_echo : int or None, optional
        Echo the k-space centre is acquired at; ``None`` keeps it at the first
        echo.
    pad : bool, optional
        Pad each train to ``train_length`` with ``None`` (see
        :func:`make_linear_order`).

    Returns
    -------
    list of list of int
        Shots of view indices, in a single global center-out order.

    Examples
    --------
    >>> import numpy as np
    >>> import pulserver.pypulseq as pp
    >>> ky, kz = np.meshgrid(np.arange(-2, 3), np.arange(-2, 3))
    >>> coords = np.column_stack([ky.ravel(), kz.ravel()])
    >>> shots = pp.make_centric_order(coords, 5)
    >>> all(len(s) <= 5 for s in shots)
    True

    Echo index (colour) is global distance-from-centre rank, so the earliest
    echoes of *every* shot cluster near the centre rather than each shot
    starting its own center-out sweep (contrast :func:`make_radial_order`):

    .. plot::
       :include-source: false

       import numpy as np
       import pulserver.pypulseq as pp
       from _figures import order_figure
       ky, kz = np.meshgrid(np.arange(-16, 16), np.arange(-16, 16))
       coords = np.column_stack([ky.ravel(), kz.ravel()])
       order_figure([("centric", pp.make_centric_order(coords, 32))], coords)

    See Also
    --------
    make_radial_order, make_linear_order, make_radial_adaptive_order
    """
    pts = _as_coords(coords)
    n = len(pts)
    if n == 0:
        return []
    n_trains = int(np.ceil(n / train_length))
    radius, angle = _polar(pts, center)
    order = sorted(range(n), key=lambda index: (radius[index], angle[index]))
    trains = _deal_echo_major(
        order,
        angle,
        n_trains,
        train_length,
        radius,
        center_echo=center_echo,
        adaptive=False,
    )
    return _finish_trains(trains, train_length, pad)

def make_radial_order(
    coords,
    train_length: int,
    *,
    center=None,
    pad: bool = False,
) -> list[list[int | None]]:
    """Center-out radial (wedge) echo-train ordering.

    k-space is partitioned into angular wedges (one per shot). Within each
    wedge the views are ordered by radial distance from the k-space center,
    so every echo train samples the center first and the periphery last —
    scheme B ("radial wedge reordering") of 566-05-007, Fig. 2, and the
    conventional proton-density 3D FSE center-out ordering (Busse et al.).
    The centre of k-space is therefore always at the first echo.

    Parameters
    ----------
    coords : array_like
        Phase-encode locations, shape ``(N,)`` or ``(N, 2)``.
    train_length : int
        Echo-train or segment length; the number of wedges is
        ``ceil(N / train_length)``.
    center : tuple of float or None, optional
        k-space centre; ``None`` uses the centroid of ``coords``.
    pad : bool, optional
        Pad each train to ``train_length`` with ``None`` (see
        :func:`make_linear_order`).

    Returns
    -------
    list of list of int
        Shots of view indices, each ordered center-out.

    Examples
    --------
    >>> import numpy as np
    >>> import pulserver.pypulseq as pp
    >>> ky, kz = np.meshgrid(np.arange(-2, 3), np.arange(-2, 3))
    >>> coords = np.column_stack([ky.ravel(), kz.ravel()])
    >>> shots = pp.make_radial_order(coords, 5)
    >>> all(len(s) <= 5 for s in shots)
    True

    Echo index (colour) across (ky, kz) determines the T2 weighting of each
    region of k-space:

    .. plot::
       :include-source: false

       import numpy as np
       import pulserver.pypulseq as pp
       from _figures import order_figure
       ky, kz = np.meshgrid(np.arange(-16, 16), np.arange(-16, 16))
       coords = np.column_stack([ky.ravel(), kz.ravel()])
       order_figure([("radial", pp.make_radial_order(coords, 32))], coords)

    See Also
    --------
    make_linear_order, make_radial_adaptive_order, make_shuffling_order
    """
    pts = _as_coords(coords)
    n = len(pts)
    if n == 0:
        return []
    radius, angle = _polar(pts, center)
    n_shots = int(np.ceil(n / train_length))
    # Angular wedges of equal view count keep every echo train the same length.
    by_angle = sorted(range(n), key=lambda i: angle[i])
    shots: list[list[int | None]] = []
    for s in range(n_shots):
        wedge = by_angle[s * train_length : (s + 1) * train_length]
        wedge.sort(key=lambda i: radius[i])
        shots.append(wedge)
    return _finish_trains(shots, train_length, pad)

def make_radial_adaptive_order(
    coords,
    train_length: int,
    *,
    center=None,
    center_echo: int | None = None,
    pad: bool = False,
) -> list[list[int | None]]:
    """Adaptive radial echo-train ordering (individually parameterized trains).

    Views are ranked by radius and dealt echo-major into trains, but the echo
    a radius band plays at *folds* around ``center_echo``: the innermost band
    lands on the target echo and the radius grows away from it in both echo
    directions. This is scheme C ("modified radial / adaptive reordering") of
    566-05-007, Fig. 2C-D, which enforces a UI-defined target TE at the centre
    of k-space without a central-k-space discontinuity, and within a band the
    views are ordered angularly so successive echoes stay neighbours.

    Parameters
    ----------
    coords : array_like
        Phase-encode locations, shape ``(N,)`` or ``(N, 2)``.
    train_length : int
        Echo-train or segment length; the number of shots is
        ``ceil(N / train_length)``.
    center : tuple of float or None, optional
        k-space centre; ``None`` uses the centroid of ``coords``.
    center_echo : int or None, optional
        Echo the k-space centre is acquired at; ``None`` keeps the innermost
        band at the first echo (equivalent to a plain radial ordering).
    pad : bool, optional
        Pad each train to ``train_length`` with ``None`` (see
        :func:`make_linear_order`).

    Returns
    -------
    list of list of int
        Shots of view indices; the echo order runs outward from the target
        echo in both directions.

    Examples
    --------
    >>> import numpy as np
    >>> import pulserver.pypulseq as pp
    >>> ky, kz = np.meshgrid(np.arange(-3, 4), np.arange(-3, 4))
    >>> coords = np.column_stack([ky.ravel(), kz.ravel()])
    >>> shots = pp.make_radial_adaptive_order(coords, 7)
    >>> len(shots)
    7
    >>> sum(len(shot) for shot in shots) == len(coords)
    True

    .. plot::
       :include-source: false

       import numpy as np
       import pulserver.pypulseq as pp
       from _figures import order_figure
       ky, kz = np.meshgrid(np.arange(-16, 16), np.arange(-16, 16))
       coords = np.column_stack([ky.ravel(), kz.ravel()])
       order_figure([("radial adaptive", pp.make_radial_adaptive_order(coords, 32))], coords)
    """
    pts = _as_coords(coords)
    n = len(pts)
    if n == 0:
        return []
    n_trains = int(np.ceil(n / train_length))
    radius, angle = _polar(pts, center)
    order = sorted(range(n), key=lambda index: (radius[index], angle[index]))
    trains = _deal_echo_major(
        order,
        angle,
        n_trains,
        train_length,
        radius,
        center_echo=0 if center_echo is None else center_echo,
        adaptive=True,
    )
    return _finish_trains(trains, train_length, pad)

def make_shuffling_order(
    coords,
    train_length: int,
    *,
    seed: int | None = None,
    cluster: bool = True,
    pad: bool = False,
) -> list[list[int | None]]:
    """Randomly shuffled echo-train ordering (T2 Shuffling).

    Phase encodes are randomly assigned to echo positions so that k-t space
    is sampled incoherently (Tamir et al., "T2 Shuffling"). To limit
    gradient switching within a train, nearby phase encodes are grouped into
    the same echo train (``cluster=True``) before the echo order within each
    train is randomized — mirroring the paper's mitigation of eddy-current
    effects by "assigning nearby phase encodes to the same echo train".

    Parameters
    ----------
    coords : array_like
        Phase-encode locations, shape ``(N,)`` or ``(N, 2)``.
    train_length : int
        Echo train length.
    seed : int or None, optional
        Seed for reproducible shuffling.
    cluster : bool, optional
        When True, group spatially nearby views into the same train before
        randomizing echo order; when False, assign views to trains at random.
    pad : bool, optional
        Pad each train to ``train_length`` with ``None`` (see
        :func:`make_linear_order`).

    Returns
    -------
    list of list of int
        Shots of view indices in randomized echo order.

    Examples
    --------
    >>> import numpy as np
    >>> import pulserver.pypulseq as pp
    >>> ky, kz = np.meshgrid(np.arange(8), np.arange(8))
    >>> coords = np.column_stack([ky.ravel(), kz.ravel()])
    >>> a = pp.make_shuffling_order(coords, 8, seed=0)
    >>> b = pp.make_shuffling_order(coords, 8, seed=0)
    >>> a == b
    True

    .. plot::
       :include-source: false

       import numpy as np
       import pulserver.pypulseq as pp
       from _figures import order_figure
       ky, kz = np.meshgrid(np.arange(-16, 16), np.arange(-16, 16))
       coords = np.column_stack([ky.ravel(), kz.ravel()])
       order_figure([("T2 shuffling", pp.make_shuffling_order(coords, 32, seed=0))], coords)
    """
    pts = _as_coords(coords)
    n = len(pts)
    if n == 0:
        return []
    rng = np.random.default_rng(seed)
    n_shots = int(np.ceil(n / train_length))

    # Grid-strip clustering sorts by kz then ky so each train covers a
    # spatially compact region; the alternative starts from a random order.
    base_order = sorted(range(n), key=lambda i: (pts[i, 1], pts[i, 0])) if cluster else list(rng.permutation(n))

    shots: list[list[int | None]] = []
    for s in range(n_shots):
        train = base_order[s * train_length : (s + 1) * train_length]
        train = list(rng.permutation(train))
        shots.append(train)
    return _finish_trains(shots, train_length, pad)

def make_random_mask(
    shape: tuple[int, int],
    accel: float,
    *,
    calib: tuple[int, int] = (0, 0),
    seed: int | None = None,
) -> np.ndarray:
    """Generate a uniform-random undersampling mask with a calibration region.

    Exactly ``round(N / accel)`` locations are sampled (including the fully
    sampled centered calibration block), drawn uniformly at random.

    Parameters
    ----------
    shape : tuple of int
        Mask shape ``(ny, nz)``.
    accel : float
        Target acceleration factor (> 1).
    calib : tuple of int, optional
        Fully sampled centered calibration shape.
    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    numpy.ndarray
        Boolean mask of ``shape``.

    Examples
    --------
    >>> import pulserver.pypulseq as pp
    >>> mask = pp.make_random_mask((32, 32), 4.0, calib=(8, 8), seed=0)
    >>> mask.shape
    (32, 32)

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       import pulserver.pypulseq as pp
       fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
       masks = [
           ("random, R=4", pp.make_random_mask((64, 64), 4.0, calib=(12, 12), seed=0)),
           ("poisson-disc, R=4", pp.make_poisson_disc_mask((64, 64), 4.0, calib=(12, 12), seed=1)),
           ("CAIPI 2x2, delta=1", pp.make_caipirinha_mask((64, 64), 2, 2, delta=1)),
       ]
       for ax, (title, mask) in zip(axes, masks):
           ax.imshow(mask.T, cmap="gray", origin="lower", interpolation="nearest")
           ax.set_title(title, fontsize=9)
           ax.set_xlabel("ky"); ax.set_ylabel("kz")
       fig.tight_layout()

    See Also
    --------
    make_poisson_disc_mask : incoherent but locally uniform alternative.
    make_caipirinha_mask : deterministic lattice for parallel imaging.
    ScanLoop.from_mask : turn a mask into shots.
    """
    if accel <= 1:
        raise ValueError(f"accel must be greater than 1, got {accel}")
    ny, nz = shape
    rng = np.random.default_rng(seed)
    mask = np.zeros(shape, dtype=bool)
    y0 = ny // 2 - calib[0] // 2
    z0 = nz // 2 - calib[1] // 2
    mask[y0 : y0 + calib[0], z0 : z0 + calib[1]] = True

    n_target = round(ny * nz / accel)
    n_extra = max(0, n_target - int(mask.sum()))
    free = np.flatnonzero(~mask.ravel())
    chosen = rng.choice(free, size=min(n_extra, free.size), replace=False)
    flat = mask.ravel()
    flat[chosen] = True
    return flat.reshape(shape)

def make_caipirinha_mask(
    shape: tuple[int, int],
    ry: int,
    rz: int,
    *,
    delta: int = 1,
) -> np.ndarray:
    """Generate a 2D CAIPIRINHA lattice undersampling mask.

    Standard CAIPI shift pattern: row ``ky`` is sampled on the ``rz``-grid
    with a per-``ky``-block shift of ``delta`` along kz, spreading aliasing
    in both phase-encode directions (Breuer et al., MRM 2006).

    Parameters
    ----------
    shape : tuple of int
        Mask shape ``(ny, nz)``.
    ry : int
        Acceleration along the first axis.
    rz : int
        Acceleration along the second axis.
    delta : int, optional
        CAIPI shift applied per sampled-ky step (``0 <= delta < rz``;
        ``delta=0`` degenerates to a regular ``ry x rz`` lattice).

    Returns
    -------
    numpy.ndarray
        Boolean mask of ``shape`` with exact acceleration ``ry * rz``.

    Examples
    --------
    >>> import pulserver.pypulseq as pp
    >>> mask = pp.make_caipirinha_mask((8, 8), 2, 2, delta=1)
    >>> int(mask.sum())
    16

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       import pulserver.pypulseq as pp
       fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
       for ax, delta in zip(axes, (0, 1, 2)):
           ax.imshow(pp.make_caipirinha_mask((32, 16), 2, 3, delta=delta).T,
                     cmap="gray", origin="lower", interpolation="nearest")
           ax.set_title(f"ry=2, rz=3, delta={delta}", fontsize=9)
           ax.set_xlabel("ky"); ax.set_ylabel("kz")
       fig.tight_layout()

    References
    ----------
    Breuer et al., CAIPIRINHA, DOI ``10.1002/mrm.20787``.

    See Also
    --------
    make_skipped_caipi_order : segmented EPI ordering over this lattice.
    """
    if ry < 1 or rz < 1:
        raise ValueError("ry and rz must be >= 1")
    ny, nz = shape
    ky = np.arange(ny)[:, None]
    kz = np.arange(nz)[None, :]
    shift = (ky // ry) * delta
    return (ky % ry == 0) & ((kz - shift) % rz == 0)

def make_poisson_disc_mask(
    shape: tuple[int, int],
    accel: float,
    *,
    calib: tuple[int, int] = (0, 0),
    seed: int = 0,
    max_attempts: int = 30,
    tol: float = 0.1,
    crop_corner: bool = True,
) -> np.ndarray:
    """Generate a variable-density Poisson-disc undersampling mask.

    Ported from ``refcode/sigpy`` (``sigpy.mri.samp.poisson``): sampling
    density falls off as ``1 / (1 + s|r|)`` with the slope ``s`` found by
    binary search so the realized acceleration matches ``accel`` within
    ``tol``; points are placed with Bridson dart throwing.

    Parameters
    ----------
    shape : tuple of int
        Mask shape ``(ny, nz)``.
    accel : float
        Target acceleration factor (> 1).
    calib : tuple of int, optional
        Fully sampled centered calibration shape.
    seed : int, optional
        Random seed.
    max_attempts : int, optional
        Bridson candidate attempts per active point.
    tol : float, optional
        Allowed deviation of the realized acceleration.
    crop_corner : bool, optional
        Restrict sampling to the inscribed k-space ellipse.

    Returns
    -------
    numpy.ndarray
        Boolean mask of ``shape``.

    Examples
    --------
    >>> import pulserver.pypulseq as pp
    >>> mask = pp.make_poisson_disc_mask((48, 48), 4.0, calib=(8, 8), seed=1)
    >>> mask.shape
    (48, 48)

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       import pulserver.pypulseq as pp
       fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
       for ax, accel in zip(axes, (2.0, 4.0, 8.0)):
           ax.imshow(pp.make_poisson_disc_mask((64, 64), accel, calib=(12, 12), seed=1).T,
                     cmap="gray", origin="lower", interpolation="nearest")
           ax.set_title(f"R={accel:g}", fontsize=9)
           ax.set_xlabel("ky"); ax.set_ylabel("kz")
       fig.tight_layout()

    References
    ----------
    SigPy ``sigpy.mri.samp.poisson`` (BSD 3-Clause); Bridson, SIGGRAPH 2007.
    """
    if accel <= 1:
        raise ValueError(f"accel must be greater than 1, got {accel}")
    from pulserver._ext import _sampling_wrapper

    return _sampling_wrapper.poisson_disc_mask(shape, accel, calib, seed, max_attempts, tol, crop_corner)
