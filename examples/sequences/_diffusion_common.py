"""Shared Stejskal-Tanner diffusion-weighting gradient module.

A monopolar diffusion gradient pair flanking the spin-echo refocusing
pulse: one lobe of duration ``delta`` right before the 180, one right
after. Gradient separation ``Delta`` (lobe start-to-start) is approximated
as ``TE / 2`` (i.e. the diffusion gradients occupy the existing TE/2
windows either side of the 180, ignoring the much-smaller RF pulse
durations) — a deliberate simplification for this research/simulation
build, not a full scanner-grade timing optimizer. ``delta`` is set to 60%
of ``Delta`` so both lobes fit inside their window with margin.

``b`` (``FloatKey.DIFFUSION_BVALUES``, s/mm^2) and the direction count
(``IntKey.DIFFUSION_DIRECTIONS``) are both real native GE variables — no
``opuser`` needed. ``b <= 0`` means diffusion weighting is disabled (a
plain b0 acquisition), the standard convention, so no separate enable flag
is required either.

Loaded via file-path (``importlib.util.spec_from_file_location``), same
convention as ``_gre_common.py``/``_prep_common.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pypulseq as pp

DELTA_FRACTION_OF_SEPARATION = 0.6

# Direction tables for common counts; any other count cycles through x/y/z.
_DIRECTION_TABLES: dict[int, list[tuple[float, float, float]]] = {
    1: [(0.0, 0.0, 1.0)],
    3: [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
    6: [
        (1.0, 1.0, 0.0), (1.0, -1.0, 0.0),
        (1.0, 0.0, 1.0), (1.0, 0.0, -1.0),
        (0.0, 1.0, 1.0), (0.0, 1.0, -1.0),
    ],
}


def diffusion_directions(n_directions: int) -> list[np.ndarray]:
    """Return ``n_directions`` unit vectors (x, y, z)."""
    n_directions = max(1, int(n_directions))
    raw = _DIRECTION_TABLES.get(n_directions)
    if raw is None:
        axes = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        raw = [axes[i % 3] for i in range(n_directions)]
    return [np.asarray(v, dtype=float) / np.linalg.norm(v) for v in raw]


def diffusion_timing(te_s: float) -> tuple[float, float]:
    """Return ``(delta_s, separation_s)`` for the gradient pair given TE."""
    separation_s = te_s / 2.0
    delta_s = DELTA_FRACTION_OF_SEPARATION * separation_s
    return delta_s, separation_s


def required_gradient_t_per_m(b_s_mm2: float, delta_s: float, separation_s: float, gamma_hz_per_t: float) -> float:
    """Solve the Stejskal-Tanner equation for the gradient amplitude (T/m)
    needed to reach ``b_s_mm2`` with the given lobe timing."""
    if b_s_mm2 <= 0.0:
        return 0.0
    b_s_per_m2 = b_s_mm2 * 1e6
    gamma_rad_s_t = 2.0 * math.pi * gamma_hz_per_t
    denom = (gamma_rad_s_t**2) * (delta_s**2) * (separation_s - delta_s / 3.0)
    if denom <= 0.0:
        return float("inf")
    return math.sqrt(b_s_per_m2 / denom)


def build_diffusion_gradients(opts, direction: np.ndarray, delta_s: float, grad_t_per_m: float):
    """Build one trapezoid per non-zero axis of ``direction`` (a single lobe
    — callers play this twice, before and after the 180). Returns ``[]`` if
    ``grad_t_per_m`` is 0 (b=0)."""
    if grad_t_per_m <= 0.0:
        return []
    area_per_axis = grad_t_per_m * opts.gamma * delta_s
    trapezoids = []
    for axis, component in zip(("x", "y", "z"), direction):
        if abs(component) < 1e-9:
            continue
        trapezoids.append(
            pp.make_trapezoid(channel=axis, area=area_per_axis * float(component), duration=delta_s, system=opts)
        )
    return trapezoids
