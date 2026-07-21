"""Non-Cartesian tilt support and acquisition ordering."""

from __future__ import annotations

import math

import numpy as np

from ._pattern import SamplingPattern

_PHI = (1.0 + math.sqrt(5.0)) / 2.0


def _segments(indices, length):
    return tuple(np.asarray(indices[i : i + length], dtype=np.intp) for i in range(0, len(indices), length))


def _generalized_fibonacci(order, index):
    if order == 1:
        return 1
    previous, current = 1, int(index)
    for _ in range(2, order):
        previous, current = current, previous + current
    return current


def radial_2d(
    n_spokes,
    *,
    scheme="linear",
    period=np.pi,
    increment=None,
    tiny_index=1,
    approximation_order=13,
    segment_length=1,
):
    n_spokes, segment_length = int(n_spokes), int(segment_length)
    period = float(period)
    if n_spokes < 0 or segment_length <= 0 or not np.isfinite(period) or period <= 0:
        raise ValueError("n_spokes must be nonnegative and period/segment_length positive")
    if scheme == "raga":
        tiny_index, approximation_order = int(tiny_index), int(approximation_order)
        if tiny_index < 1 or approximation_order < 2:
            raise ValueError("tiny_index must be >= 1 and approximation_order >= 2")
        support_size = _generalized_fibonacci(approximation_order, tiny_index)
        step = _generalized_fibonacci(approximation_order - 1, 1)
        support = (np.arange(support_size) * period / support_size)[:, None]
        chronological = (np.arange(n_spokes, dtype=np.intp) * step) % support_size
    else:
        if scheme == "linear":
            step = (
                period / n_spokes
                if increment is None and n_spokes
                else (0.0 if increment is None else float(increment))
            )
        elif scheme == "golden":
            step = period / _PHI
        elif scheme == "tiny_golden":
            tiny_index = int(tiny_index)
            if tiny_index < 1:
                raise ValueError("tiny_index must be >= 1")
            step = period / (_PHI + tiny_index - 1)
        else:
            raise ValueError("scheme must be linear, golden, tiny_golden, or raga")
        if not np.isfinite(step):
            raise ValueError("increment must be finite")
        support = np.mod(np.arange(n_spokes) * step, period)[:, None]
        chronological = np.arange(n_spokes, dtype=np.intp)
    return SamplingPattern(support, _segments(chronological, segment_length))


def golden_angles(n: int) -> np.ndarray:
    # Legacy API uses the full-circle, monotonically accumulated arbgrad
    # convention.  ``radial_2d`` remains the explicit half/full-period API.
    from .cartesian import golden_angles as _legacy_golden_angles

    return _legacy_golden_angles(n)


def uniform_angles(n: int) -> np.ndarray:
    from .cartesian import uniform_angles as _legacy_uniform_angles

    return _legacy_uniform_angles(n)


def _directions(z, azimuth):
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack((radius * np.cos(azimuth), radius * np.sin(azimuth), z))


def golden_means_3d(n_spokes, *, segment_length=1):
    n_spokes, segment_length = int(n_spokes), int(segment_length)
    if n_spokes < 0 or segment_length <= 0:
        raise ValueError("n_spokes must be nonnegative and segment_length positive")
    m = np.arange(n_spokes, dtype=float)
    z = 2.0 * np.mod(m * 0.465571231876768, 1.0) - 1.0
    azimuth = 2.0 * np.pi * np.mod(m * 0.682327803828019, 1.0)
    support = _directions(z, azimuth)
    return SamplingPattern(support, _segments(np.arange(n_spokes, dtype=np.intp), segment_length))


def _is_fibonacci(value):
    return any(
        int(math.isqrt(candidate)) ** 2 == candidate for candidate in (5 * value * value + 4, 5 * value * value - 4)
    )


def spiral_phyllotaxis(n_spokes, n_interleaves, *, require_fibonacci=True):
    n_spokes, n_interleaves = int(n_spokes), int(n_interleaves)
    if n_spokes <= 0 or n_interleaves <= 0 or n_spokes % n_interleaves:
        raise ValueError("positive n_spokes must be divisible by positive n_interleaves")
    if require_fibonacci and not _is_fibonacci(n_interleaves):
        raise ValueError("n_interleaves must be a Fibonacci number")
    m = np.arange(n_spokes, dtype=float)
    z = np.ones(1) if n_spokes == 1 else 1.0 - 2.0 * m / (n_spokes - 1)
    support = _directions(z, m * np.pi * (3.0 - math.sqrt(5.0)))
    order = tuple(np.arange(j, n_spokes, n_interleaves, dtype=np.intp) for j in range(n_interleaves))
    return SamplingPattern(support, order)


def directions_to_rotations(directions, *, reference=(1.0, 0.0, 0.0)):
    directions = np.asarray(directions, dtype=float)
    if directions.ndim == 1:
        directions = directions[None, :]
    reference = np.asarray(reference, dtype=float)
    if directions.ndim != 2 or directions.shape[1] != 3 or reference.shape != (3,):
        raise ValueError("directions must have shape (3,) or (N, 3), and reference shape (3,)")
    if not np.all(np.isfinite(directions)) or not np.all(np.isfinite(reference)):
        raise ValueError("directions and reference must be finite")
    ref_norm = np.linalg.norm(reference)
    norms = np.linalg.norm(directions, axis=1)
    if ref_norm == 0 or np.any(norms == 0):
        raise ValueError("directions and reference must be nonzero")
    ref = reference / ref_norm
    result = np.empty((len(directions), 3, 3), dtype=float)
    identity = np.eye(3)
    for idx, direction in enumerate(directions / norms[:, None]):
        cross = np.cross(ref, direction)
        sine = np.linalg.norm(cross)
        cosine = float(np.dot(ref, direction))
        if sine > 1e-12:
            axis = cross / sine
            skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            result[idx] = identity + sine * skew + (1.0 - cosine) * (skew @ skew)
        elif cosine > 0:
            result[idx] = identity
        else:
            basis = identity[np.argmin(np.abs(identity @ ref))]
            axis = np.cross(ref, basis)
            axis /= np.linalg.norm(axis)
            result[idx] = 2.0 * np.outer(axis, axis) - identity
    return result
