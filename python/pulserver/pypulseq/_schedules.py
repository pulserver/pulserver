"""RF phase and refocusing-flip schedules."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def make_rf_spoiling_schedule(
    length: int,
    *,
    increment: float = np.deg2rad(117.0),
    initial_phase: float = 0.0,
    initial_increment: float = 0.0,
) -> np.ndarray:
    """Return the standard quadratic RF-spoiling phase schedule in radians."""
    if length < 0:
        raise ValueError("length must be >= 0")
    phases = np.empty(length, dtype=float)
    phase = float(initial_phase)
    phase_increment = float(initial_increment)
    for index in range(length):
        phases[index] = phase % (2.0 * np.pi)
        phase = (phase + phase_increment) % (2.0 * np.pi)
        phase_increment = (phase_increment + increment) % (2.0 * np.pi)
    return phases


def make_phase_cycling_schedule(
    length: int,
    phases: Sequence[float] = (0.0, np.pi),
) -> np.ndarray:
    """Repeat an arbitrary phase cycle to ``length`` entries, in radians."""
    if length < 0:
        raise ValueError("length must be >= 0")
    cycle = np.asarray(phases, dtype=float)
    if cycle.ndim != 1 or cycle.size == 0 or not np.all(np.isfinite(cycle)):
        raise ValueError("phases must be a non-empty one-dimensional finite sequence")
    return np.resize(cycle, length) % (2.0 * np.pi)


def make_traps_schedule(
    length: int,
    target_flip_angle: float,
    *,
    variable: bool = True,
) -> np.ndarray:
    """Return an Alsop-style TRAPS refocusing schedule in radians."""
    if length < 1:
        raise ValueError("length must be >= 1")
    if not np.isfinite(target_flip_angle) or target_flip_angle <= 0:
        raise ValueError("target_flip_angle must be a positive finite angle")
    if not variable or length == 1:
        return np.full(length, target_flip_angle, dtype=float)

    first = (
        np.pi / 2.0
        + target_flip_angle / 2.0
        + 0.4 * ((2.0 - 1.0) / 2.0) ** 2.0 * (np.pi / 2.0 - target_flip_angle / 2.0)
    )
    delta = first - target_flip_angle
    echo = np.arange(2, length + 1, dtype=float)
    return np.concatenate(([first], target_flip_angle + delta / (2.0 ** (echo - 0.5))))
