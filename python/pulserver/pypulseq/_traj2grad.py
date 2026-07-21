"""System-aware trajectory-to-gradient design."""

from __future__ import annotations

import numpy as np

from . import _arbgrad as arbgrad

__all__ = ["traj2grad"]


def traj2grad(
    trajectory: np.ndarray,
    system,
    *,
    oversampling: int = 8,
    start_at_zero: bool = True,
    end_at_zero: bool = True,
) -> np.ndarray:
    """Generate an MRArbGrad-limited gradient from a sampled k-space path.

    ``trajectory`` has shape ``(n, 2)`` or ``(n, 3)`` and units cycles/m.
    The returned ``(n_grad, 3)`` array is in Pulseq gradient units (Hz/m),
    sampled on ``system.grad_raster_time``. Gradient and slew limits are the
    vector limits from ``system``.

    This intentionally differs from pypulseq's finite-difference
    ``traj_to_grad``: input samples specify path geometry only; MRArbGrad
    chooses a time parameterization satisfying ``gmax`` and ``smax``.
    """
    return arbgrad.traj2grad(
        trajectory,
        max_slew=system.max_slew,
        max_grad=system.max_grad,
        dt=system.grad_raster_time,
        oversampling=oversampling,
        start_at_zero=start_at_zero,
        end_at_zero=end_at_zero,
    )
