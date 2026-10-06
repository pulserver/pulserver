"""A phantom sampled for the Fourier engine."""

from __future__ import annotations

__all__ = ["Tissue", "transmitted"]

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, eq=False)
class Tissue:
    """A phantom sampled for the Fourier engine: what each cube of it holds.

    Attributes
    ----------
    positions
        ``(n, 3)`` each entry's position, in m, along the physical axes.
    density
        ``(n,)`` its proton density: its equilibrium magnetization, per entry.
    t1, t2, t2_prime
        ``(n,)`` its relaxation times, in s.
    frequency
        ``(n,)`` its precession frequency at rest, in Hz from the reference
        frequency: field inhomogeneity and chemical shift together.
    spacing
        Width of an entry's cube, in m.
    axes
        ``(m, 3)`` the physical axes along which an entry spans ``spacing``.
    transmit
        ``(n,)`` the complex transmit field an entry sees, relative to a
        pulse's nominal amplitude; None for one uniform channel.
    receive
        Each coil's receive sensitivity at ``(p, 3)`` physical points, ``(p,
        coils)``; None for one coil of unit sensitivity.
    coils
        Receive channels.
    """

    positions: np.ndarray
    density: np.ndarray
    t1: np.ndarray
    t2: np.ndarray
    t2_prime: np.ndarray
    frequency: np.ndarray
    spacing: float
    axes: np.ndarray
    transmit: np.ndarray | None
    receive: Callable[[np.ndarray], np.ndarray | None]
    coils: int


def transmitted(coil, positions: np.ndarray) -> np.ndarray | None:
    """Return the transmit field a coil's channels play a single-channel pulse with at ``(n, 3)`` physical points, ``(n,)`` relative to its nominal amplitude; None for one uniform channel.

    Each channel plays the pulse times its weight in the coil's default shim.
    """
    sensitivities = coil.transmit(positions)
    if sensitivities is None:
        return None
    sensitivities = np.asarray(sensitivities).reshape(len(positions), -1)
    shim = coil.default_shim
    return sensitivities[:, 0] if shim is None else sensitivities @ shim
