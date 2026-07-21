"""Cartesian k-space masks and absolute view-ordering helpers.

The echo-train / segment view-ordering schemes (``linear``, ``radial``,
``radial_adaptive``, ``shuffling``) apply to any acquisition with an outer
loop and an inner echo train / segment — 2D and 3D FSE as well as
MPRAGE-style segmented GRE. They operate on a set of phase-encode locations
in the (ky, kz) plane (kz all-zero for a 2D single-partition acquisition)
and return a list of shots, each shot an ordered list of view indices (the
echo/segment index is the position within the shot).

References
----------
FSE linear / radial / radial-adaptive reordering follow Buonincontri et al.,
"Doubling the repetition time without paying the price: 3D TSE with
individually parameterized echo trains", ISMRM abstract 566-05-007 (Fig. 2),
in ``refcode/Abstract 566-05-007.pdf``. Random shuffling follows Tamir et
al., "T2 Shuffling", Magn Reson Med 2017;77:180-195, in
``refcode/nihms804984.pdf``.
"""

from __future__ import annotations

import numpy as np

from ._pattern import SamplingPattern


def chunk_indices(indices: list[int], size: int) -> list[list[int]]:
    """Split ``indices`` into consecutive chunks of at most ``size`` entries.

    This is the inner-loop segmentation used by segmented (MPRAGE/FSE-style)
    acquisitions: each chunk is one shot / echo train.

    Parameters
    ----------
    indices : list of int
        View indices to segment, in acquisition order.
    size : int
        Maximum entries per chunk (clamped to at least 1).

    Returns
    -------
    list of list of int
        Consecutive chunks covering ``indices`` in order.

    Examples
    --------
    >>> import pulserver
    >>> pulserver.chunk_indices([0, 1, 2, 3, 4], 2)
    [[0, 1], [2, 3], [4]]
    """
    size = max(1, int(size))
    return [indices[i : i + size] for i in range(0, len(indices), size)]


def sampled_lines(n: int, r: int, acs_lines: int) -> list[int]:
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

    Returns
    -------
    list of int
        Sorted sampled view indices.

    Examples
    --------
    >>> import pulserver
    >>> pulserver.sampled_lines(8, 2, 0)
    [0, 2, 4, 6]
    """
    sampled = {i for i in range(n) if (i % r) == 0}
    if acs_lines > 0:
        center = n // 2
        start = max(0, center - acs_lines // 2)
        stop = min(n, start + acs_lines)
        for i in range(start, stop):
            sampled.add(i)
    return sorted(sampled)


def linear_order(n: int, etl: int) -> list[list[int]]:
    """Split ``n`` views into consecutive echo trains of length ``etl``.

    The simplest reordering: view ``i`` goes to shot ``i // etl``, echo
    ``i % etl``. This is the sequential ky ordering used by the current
    FSE/MPRAGE plugins.

    Parameters
    ----------
    n : int
        Number of views.
    etl : int
        Echo train length / segment length.

    Returns
    -------
    list of list of int
        Shots, each an ordered list of view indices.

    Examples
    --------
    >>> import pulserver
    >>> pulserver.linear_order(5, 2)
    [[0, 1], [2, 3], [4]]
    """
    return chunk_indices(list(range(n)), etl)


def outer_inner_order(outer_indices: list[int], inner_len: int) -> list[list[tuple[int, int]]]:
    """Nest an inner echo train inside an outer loop.

    Returns one entry per ``(outer, inner_chunk)`` pair: each shot is a list
    of ``(outer_index, inner_index)`` tuples. Used by acquisitions that fix
    an outer coordinate (slice or partition) and sweep an inner echo train
    of length ``inner_len`` over ``range(inner_len)``.

    Parameters
    ----------
    outer_indices : list of int
        Outer-loop indices (e.g. sampled partitions).
    inner_len : int
        Length of the inner echo train / segment.

    Returns
    -------
    list of list of tuple of int
        Shots of ``(outer_index, inner_index)`` pairs.

    Examples
    --------
    >>> import pulserver
    >>> pulserver.outer_inner_order([0, 1], 2)
    [[(0, 0), (0, 1)], [(1, 0), (1, 1)]]
    """
    return [[(outer, inner) for inner in range(inner_len)] for outer in outer_indices]


def _as_coords(coords) -> np.ndarray:
    """Normalize a coordinate argument to an ``(N, 2)`` float array."""
    arr = np.asarray(coords, dtype=float)
    if arr.ndim == 1:
        arr = np.column_stack([arr, np.zeros_like(arr)])
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("coords must be shape (N,) or (N, 2)")
    return arr


def _split_into_shots(order: list[int], etl: int) -> list[list[int]]:
    return chunk_indices(order, etl)


def fse_linear_order(coords, etl: int) -> list[list[int]]:
    """Linear (raster) echo-train ordering over a (ky, kz) point set.

    Views are sorted in raster order (kz major, ky minor) and chunked into
    echo trains. This is scheme A ("linear reordering") of ISMRM abstract
    566-05-007, Fig. 2.

    Parameters
    ----------
    coords : array_like
        Phase-encode locations, shape ``(N,)`` (ky only) or ``(N, 2)``
        (ky, kz).
    etl : int
        Echo train length.

    Returns
    -------
    list of list of int
        Shots of view indices (indices into ``coords``); echo index is the
        position within the shot.

    Examples
    --------
    >>> import pulserver
    >>> shots = pulserver.fse_linear_order([[0, 0], [1, 0], [0, 1], [1, 1]], 2)
    >>> sorted(i for shot in shots for i in shot)
    [0, 1, 2, 3]
    """
    pts = _as_coords(coords)
    order = sorted(range(len(pts)), key=lambda i: (pts[i, 1], pts[i, 0]))
    return _split_into_shots(order, etl)


def fse_radial_order(coords, etl: int) -> list[list[int]]:
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
    etl : int
        Echo train length; the number of wedges is ``ceil(N / etl)``.

    Returns
    -------
    list of list of int
        Shots of view indices, each ordered center-out.

    Examples
    --------
    >>> import numpy as np
    >>> import pulserver
    >>> ky, kz = np.meshgrid(np.arange(-2, 3), np.arange(-2, 3))
    >>> coords = np.column_stack([ky.ravel(), kz.ravel()])
    >>> shots = pulserver.fse_radial_order(coords, 5)
    >>> all(len(s) <= 5 for s in shots)
    True
    """
    pts = _as_coords(coords)
    n = len(pts)
    if n == 0:
        return []
    center = pts.mean(axis=0)
    rel = pts - center
    radius = np.hypot(rel[:, 0], rel[:, 1])
    angle = np.arctan2(rel[:, 1], rel[:, 0])
    n_shots = int(np.ceil(n / etl))
    # Angular wedges of equal view count keep every echo train the same length.
    by_angle = sorted(range(n), key=lambda i: angle[i])
    shots: list[list[int]] = []
    for s in range(n_shots):
        wedge = by_angle[s * etl : (s + 1) * etl]
        wedge.sort(key=lambda i: radius[i])
        shots.append(wedge)
    return shots


def fse_radial_adaptive_order(coords, etl: int, *, n_sections: int = 3) -> list[list[int]]:
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
    etl : int
        Echo train length; the number of shots is ``ceil(N / etl)``.
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
    >>> import pulserver
    >>> ky, kz = np.meshgrid(np.arange(-3, 4), np.arange(-3, 4))
    >>> coords = np.column_stack([ky.ravel(), kz.ravel()])
    >>> shots = pulserver.fse_radial_adaptive_order(coords, 7, n_sections=3)
    >>> all(len(s) <= 7 for s in shots)
    True
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
    n_shots = int(np.ceil(n / etl))

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


def fse_shuffling_order(coords, etl: int, *, seed: int | None = None, cluster: bool = True) -> list[list[int]]:
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
    etl : int
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
    >>> import pulserver
    >>> ky, kz = np.meshgrid(np.arange(8), np.arange(8))
    >>> coords = np.column_stack([ky.ravel(), kz.ravel()])
    >>> a = pulserver.fse_shuffling_order(coords, 8, seed=0)
    >>> b = pulserver.fse_shuffling_order(coords, 8, seed=0)
    >>> a == b
    True
    """
    pts = _as_coords(coords)
    n = len(pts)
    if n == 0:
        return []
    rng = np.random.default_rng(seed)
    n_shots = int(np.ceil(n / etl))

    # Grid-strip clustering sorts by kz then ky so each train covers a
    # spatially compact region; the alternative starts from a random order.
    base_order = sorted(range(n), key=lambda i: (pts[i, 1], pts[i, 0])) if cluster else list(rng.permutation(n))

    shots: list[list[int]] = []
    for s in range(n_shots):
        train = base_order[s * etl : (s + 1) * etl]
        train = list(rng.permutation(train))
        shots.append(train)
    return shots


def random_mask(
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
    >>> import pulserver
    >>> mask = pulserver.random_mask((32, 32), 4.0, calib=(8, 8), seed=0)
    >>> mask.shape
    (32, 32)
    """
    if accel <= 1:
        raise ValueError(f"accel must be greater than 1, got {accel}")
    ny, nz = shape
    rng = np.random.default_rng(seed)
    mask = np.zeros(shape, dtype=bool)
    y0 = ny // 2 - calib[0] // 2
    z0 = nz // 2 - calib[1] // 2
    mask[y0 : y0 + calib[0], z0 : z0 + calib[1]] = True

    n_target = int(round(ny * nz / accel))
    n_extra = max(0, n_target - int(mask.sum()))
    free = np.flatnonzero(~mask.ravel())
    chosen = rng.choice(free, size=min(n_extra, free.size), replace=False)
    flat = mask.ravel()
    flat[chosen] = True
    return flat.reshape(shape)


def caipirinha_mask(
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
    >>> import pulserver
    >>> mask = pulserver.caipirinha_mask((8, 8), 2, 2, delta=1)
    >>> int(mask.sum())
    16
    """
    if ry < 1 or rz < 1:
        raise ValueError("ry and rz must be >= 1")
    ny, nz = shape
    ky = np.arange(ny)[:, None]
    kz = np.arange(nz)[None, :]
    shift = (ky // ry) * delta
    return (ky % ry == 0) & ((kz - shift) % rz == 0)


def from_mask(
    mask: np.ndarray,
    *,
    train_length: int = 1,
    ordering: str = "linear",
    seed: int = 0,
    n_sections: int = 3,
) -> SamplingPattern:
    """Build absolute Cartesian trains while retaining mask/order separately."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim not in (1, 2):
        raise ValueError("mask must be one- or two-dimensional")
    train_length = int(train_length)
    if train_length <= 0:
        raise ValueError("train_length must be positive")
    support = np.argwhere(mask)
    if not len(support):
        return SamplingPattern(support, (), mask)
    coords = support[:, 0] if mask.ndim == 1 else support
    if ordering == "linear":
        shots = fse_linear_order(coords, train_length)
    elif ordering == "radial":
        shots = fse_radial_order(coords, train_length)
    elif ordering == "radial_adaptive":
        shots = fse_radial_adaptive_order(coords, train_length, n_sections=n_sections)
    elif ordering == "shuffling":
        shots = fse_shuffling_order(coords, train_length, seed=seed)
    else:
        raise ValueError("ordering must be linear, radial, radial_adaptive, or shuffling")
    indices = tuple(np.asarray(shot, dtype=np.intp) for shot in shots)
    flattened = np.concatenate(indices) if indices else np.empty(0, dtype=np.intp)
    if len(flattened) != len(support) or len(np.unique(flattened)) != len(support):
        raise RuntimeError("Cartesian ordering did not cover the mask exactly once")
    return SamplingPattern(support, indices, mask)


def poisson_disc_mask(
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
    >>> import pulserver
    >>> mask = pulserver.poisson_disc_mask((48, 48), 4.0, calib=(8, 8), seed=1)
    >>> mask.shape
    (48, 48)
    """
    if accel <= 1:
        raise ValueError(f"accel must be greater than 1, got {accel}")
    from pulserver._ext import _sampling_wrapper

    return _sampling_wrapper.poisson_disc_mask(shape, accel, calib, seed, max_attempts, tol, crop_corner)


def golden_angles(n: int) -> np.ndarray:
    """Return ``n`` MRI golden-angle spoke rotations (111.246° increment).

    Thin wrapper over :func:`pulserver.pypulseq.arbgrad.shot_angles` with the
    ``"mri-golden"`` tilt (Winkelmann et al., IEEE TMI 2007).

    Parameters
    ----------
    n : int
        Number of shots.

    Returns
    -------
    numpy.ndarray
        Angles in radians, shape ``(n,)``.

    Examples
    --------
    >>> import numpy as np
    >>> import pulserver
    >>> angles = pulserver.golden_angles(4)
    >>> round(float(np.rad2deg(angles[1] - angles[0])), 3)
    111.246
    """
    from .. import _arbgrad as arbgrad

    return arbgrad.shot_angles(n, mode="mri-golden")


def uniform_angles(n: int) -> np.ndarray:
    """Return ``n`` uniformly spaced spoke rotations over the full circle.

    Thin wrapper over :func:`pulserver.pypulseq.arbgrad.shot_angles` with the
    ``"uniform"`` tilt (increment ``2*pi/n``).

    Parameters
    ----------
    n : int
        Number of shots.

    Returns
    -------
    numpy.ndarray
        Angles in radians, shape ``(n,)``.

    Examples
    --------
    >>> import pulserver
    >>> len(pulserver.uniform_angles(8))
    8
    """
    from .. import _arbgrad as arbgrad

    return arbgrad.shot_angles(n, mode="uniform")
