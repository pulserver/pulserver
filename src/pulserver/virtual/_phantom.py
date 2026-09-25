"""An analytic phantom: ellipses of uniform magnetization, received by coils of known sensitivity."""

from __future__ import annotations

__all__ = ["Ellipse", "Phantom"]

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.special import j1


@dataclass(frozen=True)
class Ellipse:
    """An ellipse of uniform magnetization in a plane of constant z.

    Lengths are in metres along the axes of the phantom it belongs to;
    ``angle`` turns the first semi-axis from x towards y, in radians. The
    ellipse is infinitely thin, so its signal depends on the k-space location
    along z only through the phase of its plane.
    """

    centre: tuple[float, float, float]
    semi_axes: tuple[float, float]
    angle: float = 0.0
    intensity: complex = 1.0

    def spectrum(self, k: np.ndarray) -> np.ndarray:
        """Return its magnetization's Fourier transform at ``(3, n)`` k-space locations in 1/m.

        The transform of magnetization m is the integral of m(r) exp(-2 pi i k.r),
        with k and r along the phantom's axes.
        """
        k = np.asarray(k, dtype=float)
        a, b = self.semi_axes
        cos, sin = math.cos(self.angle), math.sin(self.angle)
        along = cos * k[0] + sin * k[1]
        across = -sin * k[0] + cos * k[1]
        q = 2.0 * math.pi * np.hypot(a * along, b * across)
        safe = np.where(q > 1e-12, q, 1.0)
        profile = np.where(q > 1e-12, 2.0 * j1(safe) / safe, 1.0)
        shift = np.exp(-2j * math.pi * (np.asarray(self.centre) @ k))
        return self.intensity * math.pi * a * b * profile * shift


class Phantom:
    """Ellipses whose signals add, received by one coil or several.

    The ellipses and the sensitivities are stated along the phantom's own
    axes, which ``rotation`` and ``position`` place in the physical frame:
    the phantom's point r lies at ``rotation @ r + position``, and its coils
    move with it. With ``coils`` above one, coil c's sensitivity is
    ``1 + depth cos(2 pi q_c.r)``, where ``q_c`` has magnitude
    ``1 / period`` and points at ``2 pi c / coils`` from x towards y, so each
    coil's k-space is the object's, and two copies shifted by ``q_c``: every
    coil is analytic.

    Parameters
    ----------
    ellipses
        The object.
    coils
        Receive channels.
    period
        Spatial period of the sensitivities, in metres.
    depth
        Their modulation depth.
    rotation
        ``(3, 3)`` orthonormal matrix from the phantom's axes to the physical
        ones, a reflection included; the identity by default.
    position
        Physical location of the phantom's origin, in metres.
    """

    def __init__(
        self,
        ellipses: Sequence[Ellipse],
        *,
        coils: int = 1,
        period: float = 0.5,
        depth: float = 0.5,
        rotation: np.ndarray | None = None,
        position: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> None:
        if coils < 1:
            raise ValueError(f"a phantom is received by one coil or more, not {coils}")
        self.ellipses = tuple(ellipses)
        self.coils = coils
        angles = 2.0 * math.pi * np.arange(coils) / coils
        self._waves = (
            np.stack([np.cos(angles), np.sin(angles), np.zeros(coils)]) / period
        )
        self._depth = depth if coils > 1 else 0.0
        self._rotation = np.eye(3) if rotation is None else np.asarray(rotation, float)
        self._position = np.asarray(position, dtype=float)

    def kspace(self, k: np.ndarray) -> np.ndarray:
        """Return each coil's signal at ``(3, n)`` physical k-space locations in 1/m: ``(coils, n)``."""
        k = np.asarray(k, dtype=float)
        own = self._rotation.T @ k
        signal = np.empty((self.coils, k.shape[1]), dtype=complex)
        for c in range(self.coils):
            wave = self._waves[:, c : c + 1]
            signal[c] = self._spectrum(own)
            if self._depth:
                signal[c] += (
                    0.5
                    * self._depth
                    * (self._spectrum(own - wave) + self._spectrum(own + wave))
                )
        return signal * np.exp(-2j * math.pi * (self._position @ k))

    def _spectrum(self, k: np.ndarray) -> np.ndarray:
        total = np.zeros(k.shape[1], dtype=complex)
        for ellipse in self.ellipses:
            total += ellipse.spectrum(k)
        return total
