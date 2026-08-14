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
    "calc_sampled_lines",
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
    n: int, r: int, acs_lines: int, *, order: str = "ascending"
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

    Returns
    -------
    list of int
        Sampled view indices, in acquisition order.

    Raises
    ------
    ValueError
        If ``order`` is neither ``'ascending'`` nor ``'calibration_first'``.

    Examples
    --------
    >>> import pulserver.pypulseq as pp
    >>> pp.calc_sampled_lines(8, 2, 0)
    [0, 2, 4, 6]

    The calibration block first, then what is left of the ascending traversal:

    >>> pp.calc_sampled_lines(8, 2, 4, order='calibration_first')
    [2, 3, 4, 5, 0, 6]
    """
    if order not in ("ascending", "calibration_first"):
        raise ValueError("order must be 'ascending' or 'calibration_first'")

    sampled = {i for i in range(n) if (i % r) == 0}
    calibration: list[int] = []
    if acs_lines > 0:
        center = n // 2
        start = max(0, center - acs_lines // 2)
        stop = min(n, start + acs_lines)
        calibration = list(range(start, stop))
        sampled.update(calibration)
    if order == "ascending":
        return sorted(sampled)
    return calibration + sorted(sampled.difference(calibration))


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

def make_linear_order(coords, train_length: int) -> list[list[int]]:
    """Linear (raster) train ordering over a Cartesian point set.

    Views are sorted in raster order (kz major, ky minor) and chunked into
    echo trains. This is scheme A ("linear reordering") of ISMRM abstract
    566-05-007, Fig. 2.

    Parameters
    ----------
    coords : int or array_like
        Number of sequential views, or phase-encode locations with shape
        ``(N,)`` (ky only) or ``(N, 2)`` (ky, kz).
    train_length : int
        Echo-train or segment length.

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
        return _split_into_shots(list(range(count)), train_length)
    pts = _as_coords(coords)
    order = sorted(range(len(pts)), key=lambda i: (pts[i, 1], pts[i, 0]))
    return _split_into_shots(order, train_length)

def make_centric_order(coords, train_length: int) -> list[list[int]]:
    """Globally center-out Cartesian ordering, chunked into trains.

    This is the conventional centric segmented-GRE/MPRAGE ordering: sampled
    locations are sorted by distance from the encoded k-space centre before
    being split into segments.  Unlike :func:`make_radial_order`, it does not
    form angular wedges whose individual trains each start near the centre.

    Parameters
    ----------
    coords : array_like
        Phase-encode locations, shape ``(N,)`` or ``(N, 2)``.
    train_length : int
        Echo-train or segment length; the number of shots is
        ``ceil(N / train_length)``.

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
    if not len(pts):
        return []
    center = pts.mean(axis=0)
    relative = pts - center
    radius = np.hypot(relative[:, 0], relative[:, 1])
    angle = np.arctan2(relative[:, 1], relative[:, 0])
    order = sorted(range(len(pts)), key=lambda index: (radius[index], angle[index]))
    return _split_into_shots(order, train_length)

def make_radial_order(coords, train_length: int) -> list[list[int]]:
    """Center-out radial (wedge) echo-train ordering.

    k-space is partitioned into angular wedges (one per shot). Within each
    wedge the views are ordered by radial distance from the k-space center,
    so every echo train samples the center first and the periphery last —
    scheme B ("radial wedge reordering") of 566-05-007, Fig. 2, and the
    conventional proton-density 3D FSE center-out ordering (Busse et al.).

    Parameters
    ----------
    coords : array_like
        Phase-encode locations, shape ``(N,)`` or ``(N, 2)``.
    train_length : int
        Echo-train or segment length; the number of wedges is
        ``ceil(N / train_length)``.

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
    center = pts.mean(axis=0)
    rel = pts - center
    radius = np.hypot(rel[:, 0], rel[:, 1])
    angle = np.arctan2(rel[:, 1], rel[:, 0])
    n_shots = int(np.ceil(n / train_length))
    # Angular wedges of equal view count keep every echo train the same length.
    by_angle = sorted(range(n), key=lambda i: angle[i])
    shots: list[list[int]] = []
    for s in range(n_shots):
        wedge = by_angle[s * train_length : (s + 1) * train_length]
        wedge.sort(key=lambda i: radius[i])
        shots.append(wedge)
    return shots

def make_radial_adaptive_order(coords, train_length: int, *, n_sections: int = 3) -> list[list[int]]:
    """Adaptive radial echo-train ordering (individually parameterized trains).

    k-space is divided into ``n_sections`` concentric radial sections; the
    echo index of each view is set by its section (center sections get early
    echoes to enforce the UI-defined target TE at the k-space center), and
    within a section views are sorted angularly. This is scheme C ("modified
    radial / adaptive reordering") of 566-05-007, Fig. 2C-D, which supports
    per-shot parameter variation without central-k-space discontinuities.

    Parameters
    ----------
    coords : array_like
        Phase-encode locations, shape ``(N,)`` or ``(N, 2)``.
    train_length : int
        Echo-train or segment length; the number of shots is
        ``ceil(N / train_length)``.
    n_sections : int, optional
        Number of radial sections.

    Returns
    -------
    list of list of int
        Shots of view indices; within each shot the echo order runs from the
        innermost section outward.

    Examples
    --------
    >>> import numpy as np
    >>> import pulserver.pypulseq as pp
    >>> ky, kz = np.meshgrid(np.arange(-3, 4), np.arange(-3, 4))
    >>> coords = np.column_stack([ky.ravel(), kz.ravel()])
    >>> shots = pp.make_radial_adaptive_order(coords, 7, n_sections=3)
    >>> len(shots)
    7
    >>> sum(len(shot) for shot in shots) == len(coords)
    True

    Train lengths vary by a view or two: sections are dealt round-robin across
    shots, so a section whose size is not a multiple of the shot count leaves
    some trains one view longer.

    >>> sorted({len(shot) for shot in shots})
    [6, 7, 9]

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
    n_sections = max(1, int(n_sections))
    center = pts.mean(axis=0)
    rel = pts - center
    radius = np.hypot(rel[:, 0], rel[:, 1])
    angle = np.arctan2(rel[:, 1], rel[:, 0])
    n_shots = int(np.ceil(n / train_length))

    # Assign each view to a radial section by equal-count quantile so the
    # innermost views reliably land in the earliest echoes.
    radial_rank = np.argsort(np.argsort(radius))
    section = (radial_rank * n_sections // max(1, n)).astype(int)
    section = np.clip(section, 0, n_sections - 1)

    # Within each section, sort angularly and deal views round-robin across
    # shots so every shot draws one angular ray per section (center outward).
    shots: list[list[int]] = [[] for _ in range(n_shots)]
    for sec in range(n_sections):
        members = [i for i in range(n) if section[i] == sec]
        members.sort(key=lambda i: angle[i])
        for j, idx in enumerate(members):
            shots[j % n_shots].append(idx)
    return [s for s in shots if s]

def make_shuffling_order(
    coords, train_length: int, *, seed: int | None = None, cluster: bool = True
) -> list[list[int]]:
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

    shots: list[list[int]] = []
    for s in range(n_shots):
        train = base_order[s * train_length : (s + 1) * train_length]
        train = list(rng.permutation(train))
        shots.append(train)
    return shots

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
