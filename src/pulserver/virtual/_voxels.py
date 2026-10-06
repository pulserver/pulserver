"""Isochromats spread over their voxel and over the Lorentzian line of their T2'."""

from __future__ import annotations

import math

import numpy as np

#: The shapes a voxel's isochromats are spread over.
VOXELS = ("point", "box", "jittered")

#: Half widths of a T2' line from its centre at which it is cut: an isochromat
#: precesses at most this many times ``1 / (2 pi T2')`` from its voxel.
CUT = 32.0


def stencil(spins: int, voxel: str, spacing: float, axes: np.ndarray) -> tuple:
    """Return the offsets of a voxel's isochromats from its centre, ``(spins, 3)`` in m, and the order they take the strata of its line in.

    ``"point"`` places every isochromat at the centre; ``"box"`` places them
    at the centres of a grid of ``spins**(1/d)`` cells along each of the
    ``(d, 3)`` unit ``axes`` of a cube ``spacing`` wide, which take the strata
    of the line in the order of the golden ratio's multiples. ``"jittered"``
    has the box's cells and order; :func:`jitter` moves each isochromat within
    its cell.

    Raises
    ------
    ValueError
        If ``spins`` is below one, ``voxel`` is not a shape of
        :data:`VOXELS`, or a box's spins are not a whole number of cells along
        each axis.
    """
    if spins < 1:
        raise ValueError(f"a voxel holds one isochromat or more, not {spins}")
    if voxel not in VOXELS:
        raise ValueError(f"a voxel is one of {VOXELS}, not {voxel!r}")
    if voxel == "point":
        return np.zeros((spins, 3)), np.arange(spins)
    axes = np.atleast_2d(np.asarray(axes, dtype=float))
    cells = round(spins ** (1.0 / len(axes)))
    if cells ** len(axes) != spins:
        raise ValueError(
            f"a {voxel} of {len(axes)} axes holds a whole number of cells along "
            f"each, not {spins} isochromats"
        )
    centres = (np.arange(cells) + 0.5) / cells - 0.5
    grid = np.stack(np.meshgrid(*[centres] * len(axes), indexing="ij"), axis=-1)
    offsets = spacing * grid.reshape(spins, len(axes)) @ axes
    order = np.argsort(np.modf(np.arange(spins) * (math.sqrt(5.0) - 1.0) / 2.0)[0])
    return offsets, order


def spins_for(spins: int | None, voxel: str, axes: np.ndarray) -> int:
    """Return ``spins``, or without it one for a ``"point"`` voxel and two cells along each of ``axes`` for the others."""
    if spins is not None:
        return spins
    return 1 if voxel == "point" else 2 ** len(np.atleast_2d(axes))


def jitter(
    voxels: int,
    spins: int,
    spacing: float,
    axes: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a displacement for each spread isochromat, ``(voxels * spins, 3)`` in m, uniform over its cell of a ``"jittered"`` voxel.

    Each isochromat is drawn on its own, so no two voxels share their
    isochromats' positions and none lies on a lattice.
    """
    axes = np.atleast_2d(np.asarray(axes, dtype=float))
    cells = round(spins ** (1.0 / len(axes)))
    within = rng.uniform(-0.5, 0.5, (voxels * spins, len(axes)))
    return (spacing / cells) * within @ axes


def spread(values: np.ndarray, spins: int) -> np.ndarray:
    """Repeat each voxel's ``values`` for each of its ``spins`` isochromats, a voxel's together."""
    return np.repeat(values, spins, axis=0)


def frequencies(
    t2_prime: np.ndarray, order: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Return each spread isochromat's frequency from its voxel's, in Hz, from its voxel's T2', in s.

    A voxel's isochromats take one point each of the strata of equal
    probability of the Lorentzian line of half width ``1 / (2 pi T2')``, cut
    at :data:`CUT` half widths, the strata shifted by one uniform draw per
    voxel, so that the mean over voxels of their signal decays as
    ``exp(-|t| / T2')``, the cut aside. One isochromat, or an infinite T2',
    takes the voxel's frequency.
    """
    spins = len(order)
    if spins == 1:
        return np.zeros(len(t2_prime))
    shift = rng.random((len(t2_prime), 1))
    strata = (order[None, :] + shift) / spins
    line = np.tan(math.atan(CUT) * (2.0 * strata - 1.0)) / (2.0 * math.pi)
    rate = np.where(np.isfinite(t2_prime), 1.0 / t2_prime, 0.0)
    return (rate[:, None] * line).ravel()
