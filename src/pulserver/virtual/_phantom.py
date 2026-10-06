"""An analytic phantom: ellipses of uniform magnetization, received by coils of known sensitivity."""

from __future__ import annotations

__all__ = ["Ellipse", "Phantom"]

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pypulseqpp as pp
from scipy.special import j1

from ._coils import Coil

if TYPE_CHECKING:
    from ._tissue import Tissue


@dataclass(frozen=True)
class Ellipse:
    """An ellipse of uniform magnetization in a plane of constant z.

    Lengths are in metres along the axes of the phantom it belongs to;
    ``angle`` turns the first semi-axis from x towards y, in radians. The
    ellipse is infinitely thin, so its signal depends on the k-space location
    along z only through the phase of its plane. ``shift_ppm`` is the chemical
    shift of its spins from water, in ppm: -3.45 for the main fat resonance.
    ``t1``, ``t2`` and ``t2_prime``, in s, act in :meth:`Phantom.tissue`
    alone; ``diffusion`` is the isotropic diffusion coefficient in m²/s.
    """

    centre: tuple[float, float, float]
    semi_axes: tuple[float, float]
    angle: float = 0.0
    intensity: complex = 1.0
    shift_ppm: float = 0.0
    t1: float = math.inf
    t2: float = math.inf
    t2_prime: float = math.inf
    diffusion: float = 0.0

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

    def tissue(
        self,
        spacing: float,
        *,
        field_t: float | None = None,
        off_resonance_hz: float = 0.0,
        region: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        coil: Coil | None = None,
    ) -> Tissue:
        """Return the phantom sampled for the Fourier engine.

        Each point spans a square ``spacing`` wide in the ellipse's plane.
        ``region`` keeps the points it answers True for, from their ``(n, 3)``
        physical positions, in m, and their frequencies, in Hz.

        Raises
        ------
        ValueError
            If an ellipse has a complex intensity, the phantom has a chemical
            shift and ``field_t`` is not given, or it has coils of its own and
            ``coil`` is given.
        """
        from ._tissue import Tissue, transmitted

        if coil is not None and self.coils > 1:
            raise ValueError(
                "a phantom received by coils of its own is not scanned with a coil"
            )
        _, positions, density, t1, t2, frequency, t2_prime, _ = self._sampled(
            spacing, field_t, off_resonance_hz, region
        )
        if coil is None:
            receive = lambda points: self._received(  # noqa: E731
                (np.asarray(points) - self._position) @ self._rotation
            )
        else:
            receive = coil.receive
        return Tissue(
            positions=positions,
            density=density,
            t1=t1,
            t2=t2,
            t2_prime=t2_prime,
            frequency=frequency,
            spacing=spacing,
            axes=self._rotation[:, :2].T,
            transmit=None if coil is None else transmitted(coil, positions),
            receive=receive,
            coils=self.coils if coil is None else coil.receive_channels,
        )

    def _sampled(
        self,
        spacing: float,
        field_t: float | None,
        off_resonance_hz: float,
        region: Callable[[np.ndarray, np.ndarray], np.ndarray] | None,
    ) -> tuple[np.ndarray, ...]:
        """Return the points along the phantom's axes, the physical positions, proton densities, T1, T2, frequencies, T2' and diffusion coefficients of its voxels."""
        if field_t is None and any(self.shifts_ppm):
            raise ValueError("a phantom with a chemical shift is scanned at a field_t")
        per_ppm = 0.0 if field_t is None else 1e-6 * pp.Opts().gamma * field_t
        points = [_sampled(ellipse, spacing) for ellipse in self.ellipses]
        own = np.concatenate(points)
        tissues = [
            (
                np.real(e.intensity) * spacing**2,
                e.t1,
                e.t2,
                per_ppm * e.shift_ppm + off_resonance_hz,
                e.t2_prime,
                e.diffusion,
            )
            for e in self.ellipses
        ]
        columns = np.repeat(
            np.array(tissues, dtype=float).reshape(-1, 6),
            [len(p) for p in points],
            axis=0,
        ).T
        positions = own @ self._rotation.T + self._position
        if region is not None:
            kept = region(positions, columns[3])
            own, positions, columns = own[kept], positions[kept], columns[:, kept]
        return (own, positions, *columns)

    def _received(self, points: np.ndarray) -> np.ndarray | None:
        """Return each coil's sensitivity at ``(n, 3)`` points along the phantom's axes; None for one coil."""
        if self.coils == 1:
            return None
        return 1.0 + self._depth * np.cos(2.0 * math.pi * (points @ self._waves))

    @property
    def shifts_ppm(self) -> tuple[float, ...]:
        """The chemical shifts of its ellipses, each once, in ascending order."""
        return tuple(sorted({ellipse.shift_ppm for ellipse in self.ellipses}))

    def kspace(self, k: np.ndarray, shift_ppm: float | None = None) -> np.ndarray:
        """Return each coil's signal at ``(3, n)`` physical k-space locations in 1/m: ``(coils, n)``.

        With ``shift_ppm``, of the ellipses of that chemical shift alone.
        """
        k = np.asarray(k, dtype=float)
        ellipses = [
            e for e in self.ellipses if shift_ppm is None or e.shift_ppm == shift_ppm
        ]
        own = self._rotation.T @ k
        signal = np.empty((self.coils, k.shape[1]), dtype=complex)
        for c in range(self.coils):
            wave = self._waves[:, c : c + 1]
            signal[c] = _spectrum(ellipses, own)
            if self._depth:
                signal[c] += (
                    0.5
                    * self._depth
                    * (
                        _spectrum(ellipses, own - wave)
                        + _spectrum(ellipses, own + wave)
                    )
                )
        return signal * np.exp(-2j * math.pi * (self._position @ k))

    def proton_density(
        self,
        points: np.ndarray,
        *,
        normal: np.ndarray,  # noqa: ARG002 -- the discs need no averaging across the slab
        thickness: float,
    ) -> np.ndarray:
        """Return the magnitude of the intensity of a slab ``thickness`` thick at each of ``(n, 3)`` physical points, in metres.

        Each ellipse stands for a disc ``thickness`` thick about its plane, so a
        slab through it at any orientation shows its section, and overlapping
        ellipses add.
        """
        own = (np.asarray(points, dtype=float) - self._position) @ self._rotation
        total = np.zeros(len(own), dtype=complex)
        for ellipse in self.ellipses:
            cos, sin = math.cos(ellipse.angle), math.sin(ellipse.angle)
            dx = own[:, 0] - ellipse.centre[0]
            dy = own[:, 1] - ellipse.centre[1]
            along = (cos * dx + sin * dy) / ellipse.semi_axes[0]
            across = (-sin * dx + cos * dy) / ellipse.semi_axes[1]
            inside = (along**2 + across**2 <= 1.0) & (
                np.abs(own[:, 2] - ellipse.centre[2]) <= 0.5 * thickness
            )
            total[inside] += ellipse.intensity
        return np.abs(total)


def _sampled(ellipse: Ellipse, spacing: float) -> np.ndarray:
    """Return the points of the grid of ``spacing`` inside ``ellipse``, ``(n, 3)`` along the phantom's axes."""
    if np.imag(ellipse.intensity) != 0.0:
        raise ValueError(
            "a sampled phantom carries a real proton density, not a complex intensity"
        )
    reach = max(ellipse.semi_axes) + spacing
    lo = np.floor((np.asarray(ellipse.centre[:2]) - reach) / spacing)
    hi = np.ceil((np.asarray(ellipse.centre[:2]) + reach) / spacing)
    x, y = np.meshgrid(
        np.arange(lo[0], hi[0] + 1) * spacing,
        np.arange(lo[1], hi[1] + 1) * spacing,
        indexing="ij",
    )
    cos, sin = math.cos(ellipse.angle), math.sin(ellipse.angle)
    dx, dy = x.ravel() - ellipse.centre[0], y.ravel() - ellipse.centre[1]
    along = (cos * dx + sin * dy) / ellipse.semi_axes[0]
    across = (-sin * dx + cos * dy) / ellipse.semi_axes[1]
    inside = along**2 + across**2 <= 1.0
    return np.column_stack(
        [x.ravel()[inside], y.ravel()[inside], np.full(inside.sum(), ellipse.centre[2])]
    )


def _spectrum(ellipses: Sequence[Ellipse], k: np.ndarray) -> np.ndarray:
    total = np.zeros(k.shape[1], dtype=complex)
    for ellipse in ellipses:
        total += ellipse.spectrum(k)
    return total
