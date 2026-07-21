"""Absolute-coordinate and relative-shift sampling plans for EPI-like trains."""

from __future__ import annotations

import numpy as np

from ._pattern import SamplingPattern
from .cartesian import caipirinha_mask


def _integers(value, name):
    raw = np.asarray(value)
    if not np.issubdtype(raw.dtype, np.number) or not np.all(np.isfinite(raw)):
        raise ValueError(f"{name} must contain finite integers")
    rounded = np.rint(raw)
    if not np.all(raw == rounded):
        raise ValueError(f"{name} must contain integers")
    return rounded.astype(np.intp)


def _from_shots(shots, shape):
    shape = tuple(int(n) for n in shape)
    if not shape or any(n <= 0 for n in shape):
        raise ValueError("shape entries must be positive")
    lookup = {}
    support = []
    order = []
    mask = np.zeros(shape, dtype=bool)
    for shot in shots:
        indices = []
        for point in shot:
            key = tuple(int(v) for v in point)
            if len(key) != len(shape) or any(v < 0 or v >= shape[d] for d, v in enumerate(key)):
                raise IndexError(f"coordinate {key} is outside shape {shape}")
            if key not in lookup:
                lookup[key] = len(support)
                support.append(key)
                mask[key] = True
            indices.append(lookup[key])
        order.append(np.asarray(indices, dtype=np.intp))
    support_array = np.asarray(support, dtype=np.intp).reshape(-1, len(shape))
    return SamplingPattern(support_array, tuple(order), mask)


def from_relative_shifts(starts, shifts, *, shape) -> SamplingPattern:
    """Build a sampling plan from per-shot start points and blip increments.

    The natural description of any blipped train: a shot starts somewhere in
    (ky, kz) and each subsequent echo is reached by a *relative* step. Both
    views the sequence needs come out of the result — ``pattern[shot]`` gives
    the absolute coordinates to label, ``pattern.increments(shot)`` the blip
    areas to play.

    ``shifts`` may be one shared list of increments, or one list per shot when
    trains differ (EPTI, zigzag).

    Parameters
    ----------
    starts : array_like
        Shape ``(n_shots, D)`` integer start coordinates.
    shifts : array_like
        Shape ``(inner, D)`` shared by all shots, or ``(n_shots, inner, D)``.
        Entry 0 is normally all-zero so the first echo sits on ``starts``.
    shape : tuple of int
        Cartesian grid extent; every resulting coordinate must fall inside it.

    Returns
    -------
    SamplingPattern
        Deduplicated ``support`` of absolute coordinates, one order entry per
        shot, plus the implied Cartesian ``mask``.

    Examples
    --------
    >>> from pulserver.pypulseq import from_relative_shifts
    >>> plan = from_relative_shifts(
    ...     starts=[[0, 0], [1, 0]],
    ...     shifts=[[0, 0], [2, 1], [4, 2]],
    ...     shape=(8, 4),
    ... )
    >>> plan[0]
    array([[0, 0],
           [2, 1],
           [4, 2]])
    >>> plan.increments(0)
    array([[2, 1],
           [2, 1]])

    See Also
    --------
    skipped_caipi : segmented blipped-CAIPI plan built on this representation.
    """
    starts = _integers(starts, "starts")
    shifts = _integers(shifts, "shifts")
    if starts.ndim != 2:
        raise ValueError("starts must have shape (n_shots, dimensions)")
    n_shots, ndim = starts.shape
    if len(shape) != ndim:
        raise ValueError("shape dimensionality must match starts")
    if shifts.ndim == 2:
        if shifts.shape[1] != ndim:
            raise ValueError("shifts dimensionality must match starts")
        shifts = np.broadcast_to(shifts, (n_shots, *shifts.shape))
    elif shifts.ndim != 3 or shifts.shape[0] != n_shots or shifts.shape[2] != ndim:
        raise ValueError("shifts must have shape (inner, D) or (n_shots, inner, D)")
    return _from_shots(starts[:, None, :] + shifts, shape)


def interleaved(shape, *, acceleration=1, num_shots=1, axis=0, reverse_alternate=False):
    shape = tuple(int(n) for n in shape)
    if not shape or any(n <= 0 for n in shape):
        raise ValueError("shape entries must be positive")
    ndim = len(shape)
    axis = int(axis)
    if axis < 0:
        axis += ndim
    if not 0 <= axis < ndim:
        raise ValueError("axis is out of range")
    if np.ndim(acceleration) == 0:
        accel = np.ones(ndim, dtype=np.intp)
        accel[axis] = int(acceleration)
    else:
        accel = _integers(acceleration, "acceleration")
        if accel.shape != (ndim,):
            raise ValueError("acceleration must be scalar or have one value per dimension")
    if np.any(accel <= 0):
        raise ValueError("acceleration must be positive")
    num_shots = int(num_shots)
    if num_shots <= 0:
        raise ValueError("num_shots must be positive")
    coords = np.argwhere(np.ones(shape, dtype=bool))[::1]
    coords = coords[np.all(coords % accel == 0, axis=1)]
    shots = []
    slow_axes = [d for d in range(ndim) if d != axis]
    keys = tuple([coords[:, axis], *[coords[:, d] for d in reversed(slow_axes)]])
    coords = coords[np.lexsort(keys)]
    groups = (coords[:, axis] // accel[axis]) % num_shots
    for shot_idx in range(num_shots):
        shot = coords[groups == shot_idx]
        if reverse_alternate and shot_idx % 2:
            shot = shot[::-1]
        shots.append(shot)
    return _from_shots(shots, shape)


def skipped_caipi(shape, *, acceleration, caipi_shift, segments):
    """Build a segmented skipped-CAIPI 3D-EPI sampling plan.

    Combines in-plane and through-plane acceleration with a CAIPI shift that
    advances ``caipi_shift`` partitions per acquired line, and splits the
    result into ``segments`` interleaved shots so the echo train (and hence T2*
    blurring and distortion) shortens without changing the sampled lattice.

    The construction is verified against
    :func:`~pulserver.pypulseq.caipirinha_mask` before returning: the shots
    cover that lattice exactly, once each.

    Parameters
    ----------
    shape : tuple of int
        ``(ny, nz)`` phase-encode and partition matrix. ``ny`` must be
        divisible by ``segments * ry`` and ``nz`` by ``rz``.
    acceleration : tuple of int
        ``(ry, rz)`` in-plane and through-plane acceleration.
    caipi_shift : int
        Partition shift applied per acquired phase-encode line.
    segments : int
        Number of interleaved shots the train is split into.

    Returns
    -------
    SamplingPattern
        Absolute ``(ky, kz)`` coordinates per shot, plus the Cartesian
        ``mask``.

    Raises
    ------
    RuntimeError
        If the generated shots do not exactly tile the equivalent CAIPI mask.

    Examples
    --------
    >>> from pulserver.pypulseq import skipped_caipi
    >>> plan = skipped_caipi((32, 8), acceleration=(2, 2), caipi_shift=1, segments=2)
    >>> plan.n_shots, plan.n_samples
    (8, 64)
    >>> plan[0]
    array([[ 0,  0],
           [ 4,  2],
           [ 8,  4],
           [12,  6],
           [16,  0],
           [20,  2],
           [24,  4],
           [28,  6]])

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       from pulserver.pypulseq import skipped_caipi

       plan = skipped_caipi((32, 8), acceleration=(2, 2), caipi_shift=1, segments=2)
       fig, ax = plt.subplots(figsize=(6, 2.6))
       for shot in range(plan.n_shots):
           c = plan[shot]
           ax.plot(c[:, 0], c[:, 1], marker="o", ms=4, lw=0.8)
       ax.set_xlabel("ky"); ax.set_ylabel("kz")
       ax.set_title("skipped-CAIPI (32, 8), R=(2, 2), shift 1, 2 segments")
       fig.tight_layout()

    References
    ----------
    Stirnberg et al., segmented skipped-CAIPI, DOI ``10.1002/mrm.28486``.
    """
    ny, nz = (int(v) for v in shape)
    ry, rz = (int(v) for v in acceleration)
    segments = int(segments)
    caipi_shift = int(caipi_shift)
    if min(ny, nz, ry, rz, segments) <= 0:
        raise ValueError("shape, acceleration, and segments must be positive")
    if ny % (segments * ry) or nz % rz:
        raise ValueError("shape must be divisible by segmented acceleration")
    shots = []
    for residue in range(segments):
        for family in range(nz // rz):
            ky = residue * ry + np.arange(ny // (segments * ry)) * segments * ry
            kz = (family * rz + (ky // ry) * caipi_shift) % nz
            shots.append(np.column_stack((ky, kz)))
    pattern = _from_shots(shots, (ny, nz))
    expected = caipirinha_mask((ny, nz), ry, rz, delta=caipi_shift)
    if not np.array_equal(pattern.mask, expected) or pattern.n_samples != int(expected.sum()):
        raise RuntimeError("skipped-CAIPI construction did not cover its CAIPI mask exactly")
    return pattern
