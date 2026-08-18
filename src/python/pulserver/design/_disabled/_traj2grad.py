"""Sample-major adapter over :func:`pulserver.pypulseq.traj_to_grad`.

The readout builders hold trajectories as ``(n, ndim)``; Pulseq's convention,
which the public function follows, is time last.
"""

from __future__ import annotations

__all__ = ["traj2grad"]

import numpy as np

from ..pypulseq._traj_to_grad import traj_to_grad


def traj2grad(
    trajectory: np.ndarray,
    system,
    *,
    oversampling: int = 8,
    start_at_zero: bool = True,
    end_at_zero: bool = True,
) -> np.ndarray:
    """The gradient tracing ``trajectory``, shaped ``(n_grad, 3)``.

    Parameters
    ----------
    trajectory : numpy.ndarray
        K-space path, ``(n, 2)`` or ``(n, 3)``, in 1/m. Geometry only.
    system : pypulseq.Opts
        System limits.
    oversampling : int, optional
        Path-resampling factor the solver works at.
    start_at_zero, end_at_zero : bool, optional
        Ramp up from and back down to zero amplitude.

    Returns
    -------
    numpy.ndarray
        Gradient of shape ``(n_grad, 3)`` in Hz/m, on the gradient raster.
    """
    trajectory = np.atleast_2d(np.asarray(trajectory, dtype=float))
    gradient, _ = traj_to_grad(
        trajectory.T,
        system=system,
        oversampling=oversampling,
        start_at_zero=start_at_zero,
        end_at_zero=end_at_zero,
    )
    gradient = np.atleast_2d(gradient).T
    if gradient.shape[1] < 3:
        gradient = np.hstack([gradient, np.zeros((gradient.shape[0], 3 - gradient.shape[1]))])
    return gradient
