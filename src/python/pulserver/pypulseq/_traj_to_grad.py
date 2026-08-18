"""Turning a k-space path into a gradient that traces it."""

from __future__ import annotations

__all__ = ["traj_to_grad"]

import numpy as np

from ._opts import default_system


def traj_to_grad(
    k: np.ndarray,
    raster_time: float | None = None,
    *,
    time_optimal: bool = True,
    system=None,
    oversampling: int = 8,
    start_at_zero: bool = True,
    end_at_zero: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Gradient and slew rate for the k-space trajectory ``k``.

    Two readings of "convert a trajectory", chosen with ``time_optimal``.

    Under the default, ``k`` carries **geometry only**: the path is
    re-parameterised in time by MRArbGrad, slowing through high-curvature
    regions so that ``max_grad`` and ``max_slew`` hold everywhere by
    construction. Resample the same path more finely and the gradient does not
    change; sample it too coarsely and the *path* changes, which is the only
    thing that should. The returned length is the solver's, not ``k``'s.

    With ``time_optimal=False`` the samples are taken as already sitting on the
    raster and are differentiated, giving ``len(k) - 1`` gradient samples and
    whatever slew that spacing implies -- including one past the limit.

    Parameters
    ----------
    k : numpy.ndarray
        K-space trajectory, **time last**: shape ``(n,)``, ``(2, n)`` or
        ``(3, n)``, in 1/m.
    raster_time : float, optional
        Gradient raster (s). Defaults to the system's.
    time_optimal : bool, optional
        Re-parameterise the path within the limits rather than differentiate it.
    system : pypulseq.Opts, optional
        System limits; supplies ``max_grad``, ``max_slew`` and the raster.
    oversampling : int, optional
        Path-resampling factor the solver works at.
    start_at_zero, end_at_zero : bool, optional
        Ramp the waveform up from and back down to zero amplitude.

    Returns
    -------
    g : numpy.ndarray
        Gradient waveform (Hz/m), same axis convention as ``k``.
    slew : numpy.ndarray
        Slew rate (Hz/m/s), one value per gradient sample.

    Examples
    --------
    An analytic spiral becomes a playable gradient:

    >>> import numpy as np
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    >>> theta = np.linspace(0, 8 * np.pi, 2000)
    >>> radius = np.linspace(0, 250.0, 2000)
    >>> k = np.stack([radius * np.cos(theta), radius * np.sin(theta)])
    >>> g, slew = pp.traj_to_grad(k, system=system)
    >>> bool(np.abs(np.linalg.norm(g, axis=0)).max() <= system.max_grad * 1.001)
    True

    Then hand it to a factory per axis::

        gx = pp.make_arbitrary_grad("x", g[0], system=system)

    Differentiating instead, for a path already on the raster:

    >>> g, slew = pp.traj_to_grad(k, system=system, time_optimal=False)
    >>> g.shape[1] == k.shape[1] - 1
    True

    See Also
    --------
    make_arbitrary_grad : wrap one axis of the result as an event.
    """
    k = np.asarray(k, dtype=float)
    system = default_system(system)
    if raster_time is None:
        raster_time = system.grad_raster_time

    if not time_optimal:
        return _differentiate(k, raster_time)

    from . import _arbgrad

    flat = k[None, :] if k.ndim == 1 else k
    # The solver wants a plane at minimum; a single axis rides in one of a pair.
    solved = flat if flat.shape[0] >= 2 else np.vstack([flat, np.zeros_like(flat)])
    gradient = _arbgrad.traj2grad(
        solved.T,
        max_slew=system.max_slew,
        max_grad=system.max_grad,
        dt=raster_time,
        oversampling=oversampling,
        start_at_zero=start_at_zero,
        end_at_zero=end_at_zero,
    )
    # The solver always works in three dimensions; give back the ones asked for.
    g = gradient.T[: flat.shape[0]]
    slew = np.diff(g, axis=-1, prepend=g[..., :1]) / raster_time
    return (g[0], slew[0]) if k.ndim == 1 else (g, slew)


def _differentiate(k: np.ndarray, raster_time: float) -> tuple[np.ndarray, np.ndarray]:
    """Finite difference of an already-rastered trajectory.

    The gradient falls between k-space points and the slew between gradient
    points, so the slew is averaged back onto the gradient samples.
    """
    g = (k[..., 1:] - k[..., :-1]) / raster_time
    between = (g[..., 1:] - g[..., :-1]) / raster_time

    slew = np.zeros((*between.shape[:-1], between.shape[-1] + 1))
    slew[..., 0] = between[..., 0]
    slew[..., 1:-1] = 0.5 * (between[..., :-1] + between[..., 1:])
    slew[..., -1] = between[..., -1]
    return g, slew
