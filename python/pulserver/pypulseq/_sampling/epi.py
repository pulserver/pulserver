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
