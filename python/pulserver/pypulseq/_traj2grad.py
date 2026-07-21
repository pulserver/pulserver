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
    """Design a hardware-feasible gradient from a k-space path you specify.

    Give it the *shape* of the trajectory you want; it returns a gradient
    waveform that traces exactly that path as fast as the scanner allows.

    This is deliberately not pypulseq's ``traj_to_grad``, which differentiates
    the samples you supply and so inherits — and can violate — whatever
    timing you happened to sample them with. Here the input samples carry
    **geometry only**: MRArbGrad re-parameterises the path in time, slowing
    down through high-curvature regions, so that ``system.max_grad`` and
    ``system.max_slew`` hold everywhere by construction. Resample the same
    path more finely and the gradient does not change; sample it too coarsely
    and the *path* changes, which is the only thing that should matter.

    Parameters
    ----------
    trajectory : numpy.ndarray
        K-space path, shape ``(n, 2)`` or ``(n, 3)``, in cycles/m. Only its
        geometry is used, not its sample spacing.
    system : pypulseq.Opts
        System limits; supplies ``max_grad``, ``max_slew`` and
        ``grad_raster_time``.
    oversampling : int, optional
        Internal path-resampling factor used by the solver.
    start_at_zero : bool, optional
        Force the waveform to ramp up from zero amplitude.
    end_at_zero : bool, optional
        Force the waveform to ramp back down to zero amplitude.

    Returns
    -------
    numpy.ndarray
        Gradient of shape ``(n_grad, 3)`` in Pulseq units (Hz/m), on
        ``system.grad_raster_time``. ``n_grad`` is set by the solver, not by
        ``len(trajectory)``.

    Examples
    --------
    Turn an analytic spiral into a playable gradient:

    >>> import numpy as np
    >>> import pulserver.pypulseq as pp
    >>> theta = np.linspace(0, 8 * np.pi, 2000)
    >>> radius = np.linspace(0, 250.0, 2000)
    >>> traj = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=-1)
    >>> system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    >>> gradient = pp.traj2grad(traj, system)
    >>> gradient.shape[1]
    3

    Then hand it to pypulseq per axis::

        gx = pp.make_arbitrary_grad("x", gradient[:, 0], system=system)

    .. plot::
       :include-source: false

       import numpy as np
       import matplotlib.pyplot as plt
       import pulserver.pypulseq as pp

       theta = np.linspace(0, 8 * np.pi, 2000)
       radius = np.linspace(0, 250.0, 2000)
       traj = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=-1)
       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       g = pp.traj2grad(traj, system)
       t = np.arange(len(g)) * system.grad_raster_time * 1e3

       fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.6))
       a.plot(traj[:, 0], traj[:, 1], lw=1)
       a.set_aspect("equal"); a.set_xlabel("kx [1/m]"); a.set_ylabel("ky [1/m]")
       a.set_title("requested path")
       b.plot(t, g[:, 0], label="Gx"); b.plot(t, g[:, 1], label="Gy")
       b.set_xlabel("t [ms]"); b.set_ylabel("G [Hz/m]")
       b.set_title("designed gradient"); b.legend(fontsize=8)
       fig.tight_layout()

    See Also
    --------
    pulserver.pypulseq.arbgrad : analytic spiral/rosette base waveforms.
    make_arbitrary_grad : wrap one axis of the result as a Pulseq event.
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
