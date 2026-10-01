"""Isochromats that move with their subject and diffuse during a scan."""

from __future__ import annotations

__all__ = ["RigidMotion"]

import itertools
import math
from collections.abc import Callable, Sequence

import numpy as np

# Gauss-Legendre points and weights on [0, 1]: three points integrate a
# polynomial of degree five exactly.
_NODES = 0.5 + 0.5 * np.array([-math.sqrt(0.6), 0.0, math.sqrt(0.6)])
_WEIGHTS = np.array([5.0, 8.0, 5.0]) / 18.0

#: Longest stretch, in s, over which a block's gradient times a displacement
#: is integrated by one set of points.
_LONGEST = 0.5e-3


class RigidMotion:
    """A subject moving as a rigid body, as :class:`~pulserver.virtual.Isochromats` takes a ``motion``.

    At time ``t``, in s on the isochromats' clock, the point ``p`` of the
    subject at rest lies at ``R(t) (p - centre) + centre + offset(t)``, every
    vector along the axes of the positions, in m.

    Parameters
    ----------
    rotation : callable, optional
        ``R(t)``, a ``(3, 3)`` orthonormal matrix; the identity without one.
    offset : callable, optional
        ``offset(t)``, a translation of ``(3,)`` metres; none without one.
    centre : array_like, default=(0, 0, 0)
        The point ``rotation`` turns about, in m.

    Examples
    --------
    A nod of 2 degrees about x, once every 4 s, and a drift of 1 mm/min
    along z:

    >>> import numpy as np
    >>> from pulserver.virtual import RigidMotion
    >>> def nod(t):
    ...     angle = np.radians(2.0) * np.sin(2.0 * np.pi * t / 4.0)
    ...     c, s = np.cos(angle), np.sin(angle)
    ...     return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    >>> motion = RigidMotion(nod, lambda t: [0.0, 0.0, 1e-3 * t / 60.0])
    >>> motion(1.0, np.array([[0.0, 0.1, 0.0]])).round(5)
    array([[0.     , 0.09994, 0.00351]])
    """

    def __init__(
        self,
        rotation: Callable[[float], np.ndarray] | None = None,
        offset: Callable[[float], Sequence[float]] | None = None,
        centre: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.rotation = rotation
        self.offset = offset
        self.centre = np.asarray(centre, dtype=float)

    def transform(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(R, shift)``, with which the point ``p`` at rest lies at ``R @ p + shift`` at ``t``."""
        turn = np.eye(3)
        if self.rotation is not None:
            turn = np.asarray(self.rotation(t), dtype=float)
            if turn.shape != (3, 3) or not np.allclose(
                turn @ turn.T, np.eye(3), atol=1e-9
            ):
                raise ValueError(f"the rotation at {t} s is not a (3, 3) rotation")
        shift = self.centre - turn @ self.centre
        if self.offset is not None:
            shift = shift + np.asarray(self.offset(t), dtype=float)
        return turn, shift

    def __call__(self, t: float, positions: np.ndarray) -> np.ndarray:
        """Return where the ``(n, 3)`` points at rest lie at ``t``."""
        turn, shift = self.transform(t)
        return positions @ turn.T + shift


class _Stretches:
    """A block's gradient along the positions' axes from ``start`` to ``end``, at three points on each stretch on which it is linear.

    ``times`` and ``weights`` are ``(k, 3)``: the points, in s from the
    block's start, and their quadrature weights. ``gradient`` is the gradient
    at them, ``(k, 3, 3)`` in Hz/m, and ``remaining`` the area it plays from
    each to ``end``, in 1/m; ``area`` is the whole area from ``start``.
    """

    def __init__(
        self,
        gradients: Sequence[np.ndarray | None],
        rotation: np.ndarray | None,
        start: float,
        end: float,
    ) -> None:
        corners = [start, end]
        for axis in gradients:
            if axis is not None:
                corners.extend(t for t in axis[0] if start < t < end)
        edges = []
        for a, b in itertools.pairwise(sorted(set(corners))):
            pieces = max(1, math.ceil((b - a) / _LONGEST))
            edges.extend(a + (b - a) * np.arange(pieces) / pieces)
        edges = np.append(np.asarray(edges, dtype=float), end)
        a, b = edges[:-1, None], edges[1:, None]
        self.times = a + (b - a) * _NODES
        self.weights = (b - a) * _WEIGHTS

        def along(t: np.ndarray) -> np.ndarray:
            g = np.stack(
                [
                    np.zeros_like(t)
                    if axis is None
                    else np.interp(t, axis[0], axis[1], left=0.0, right=0.0)
                    for axis in gradients
                ],
                axis=-1,
            )
            return g if rotation is None else g @ np.asarray(rotation).T

        self.gradient = along(self.times)
        # The gradient is linear on a stretch: its ends from two inner points.
        middle = along(0.5 * (a + b)[:, 0])
        slope = (along(a[:, 0] + 0.25 * (b - a)[:, 0]) - middle) / (-0.25 * (b - a))
        left = middle - 0.5 * (b - a) * slope
        right = middle + 0.5 * (b - a) * slope
        played = np.concatenate(
            [np.zeros((1, 3)), np.cumsum(0.5 * (b - a) * (left + right), axis=0)]
        )
        self.area = played[-1]
        x = _NODES[None, :, None]
        width = (b - a)[:, :, None]
        before = played[:-1, None, :] + width * (
            x * left[:, None, :] + 0.5 * x**2 * (right - left)[:, None, :]
        )
        self.remaining = self.area - before


def motion_phase(
    stretches: _Stretches,
    motion,
    rest: np.ndarray,
    began: float,
    placed: np.ndarray,
) -> np.ndarray:
    """Return the phase, in rad, each isochromat accrues from ``motion`` over ``stretches`` beyond what it accrues where ``placed``.

    ``began`` is the block's start on the isochromats' clock.
    """
    if isinstance(motion, RigidMotion):
        turn, shift = motion.transform(began)
        weight = np.zeros(3)
        offset = 0.0
        for times, weights, gradient in zip(
            stretches.times.ravel(),
            stretches.weights.ravel(),
            stretches.gradient.reshape(-1, 3),
            strict=True,
        ):
            moved, moved_shift = motion.transform(began + times)
            weight += weights * (gradient @ (moved - turn))
            offset += weights * (gradient @ (moved_shift - shift))
        return 2.0 * math.pi * (rest @ weight + offset)
    phase = np.zeros(len(rest))
    for times, weights, gradient in zip(
        stretches.times.ravel(),
        stretches.weights.ravel(),
        stretches.gradient.reshape(-1, 3),
        strict=True,
    ):
        if np.any(gradient):
            moved = np.asarray(motion(began + times, rest), dtype=float)
            phase += weights * ((moved - placed) @ gradient)
    return 2.0 * math.pi * phase


def diffusion_phase(
    stretches: _Stretches | None,
    walked: float,
    window: float,
    diffusion: np.ndarray,
    walk: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return the phase, in rad, each isochromat accrues from its Brownian walk, and advance the walk.

    ``walk`` holds each isochromat's ``(n, 3)`` displacement, in m, from its
    position at rest, as it stands where the block starts. The walk takes
    ``walked`` s, then ``window`` s over ``stretches``, the part of the block
    after its RF pulse, whose gradient turns it. On that part, each axis's
    displacement and the phase it accrues are drawn from their joint normal
    distribution: with ``K(t)`` the area left from ``t`` to the window's
    end, the phase is ``2 pi`` times the integral of ``K`` against the walk's
    increments, of variance ``2 D`` times the integral of ``K**2``, and
    covariance ``2 D`` times the integral of ``K`` with the displacement.
    """
    n = len(walk)
    scale = np.sqrt(2.0 * diffusion)[:, None]
    phase = np.zeros(n)
    if walked > 0.0:
        before = scale * math.sqrt(walked) * rng.standard_normal((n, 3))
        walk += before
    if window <= 0.0:
        return phase
    first = rng.standard_normal((n, 3))
    if stretches is not None:
        remaining = stretches.remaining.reshape(-1, 3)
        weights = stretches.weights.reshape(-1, 1)
        along = (weights * remaining).sum(axis=0)
        square = (weights * remaining**2).sum(axis=0)
        rest = np.sqrt(np.maximum(square - along**2 / window, 0.0))
        second = rng.standard_normal((n, 3))
        phase += (
            2.0
            * math.pi
            * (
                walk @ stretches.area
                + (scale * (first * (along / math.sqrt(window)) + second * rest)).sum(
                    axis=1
                )
            )
        )
    walk += scale * math.sqrt(window) * first
    return phase
