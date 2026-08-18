"""Gradient factories stated in imaging terms rather than in area.

Each is a thin wrapper over one of PyPulseq's own factories, taking the
quantity a sequence actually knows -- a resolution, a field of view, cycles of
dephasing across a voxel -- and doing the one division that turns it into an
area.
"""

from __future__ import annotations

__all__ = [
    "concatenate_gradients",
    "make_crusher",
    "make_phase_blip",
    "make_phase_encoding",
]

from typing import Any

from . import _events
from ._opts import default_system


def concatenate_gradients(*grads: Any, system=None):
    """Lay gradients on one channel end to end and sum them into one event.

    Each gradient after the first is delayed to begin where the previous one
    ends, so the result plays them back to back on the shared channel. This is
    what folds a slice rephaser onto its selection gradient -- exactly what a
    slab excitation does internally -- so a spectral-spatial or multiband pulse,
    whose rephaser is a separate event, can hand a 3D or SMS readout one merged
    ``z`` lobe rather than a second gradient the readout has no block for. The
    inputs are not modified; ``None`` entries are dropped.

    Parameters
    ----------
    *grads : GradEvent or TrapEvent or None
        Gradients on the **same** channel, in play order. ``None`` is skipped.
    system : pypulseq.Opts, optional
        System limits.

    Returns
    -------
    GradEvent
        The gradients concatenated into a single event, or the sole gradient
        unchanged when only one is given.

    Raises
    ------
    ValueError
        If no gradient is given, or they are not all on one channel.

    Examples
    --------
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts()
    >>> select = pp.make_trapezoid("z", area=200.0, duration=2e-3, system=system)
    >>> rephase = pp.make_trapezoid("z", area=-100.0, duration=1e-3, system=system)
    >>> merged = pp.concatenate_gradients(select, rephase, system=system)
    >>> round(pp.calc_duration(merged), 6) == round(
    ...     pp.calc_duration(select) + pp.calc_duration(rephase), 6
    ... )
    True

    The rephaser starts where the selection lobe ends, so the inputs are left
    as they were:

    >>> float(rephase.delay)
    0.0
    """
    from . import add_gradients, calc_duration, scale_grad

    events = [grad for grad in grads if grad is not None]
    if not events:
        raise ValueError("concatenate_gradients needs at least one gradient")
    channels = {grad.channel for grad in events}
    if len(channels) != 1:
        raise ValueError(
            f"all gradients must be on one channel, got {sorted(channels)}"
        )
    if len(events) == 1:
        return events[0]

    system = default_system(system)
    placed = [events[0]]
    offset = calc_duration(events[0])
    for grad in events[1:]:
        # scale_grad(1.0) is a copy; the input keeps its own delay.
        shifted = scale_grad(grad, 1.0, system=system)
        shifted.delay = float(shifted.delay) + offset
        placed.append(shifted)
        offset += calc_duration(grad)
    return add_gradients(placed, system=system)


def make_phase_encoding(
    channel: str, resolution: float, system=None, duration: float | None = None
):
    """The full-amplitude phase-encode trapezoid for a target resolution.

    The largest phase-encode step an acquisition needs is set by resolution
    alone: index ``i`` of ``n`` encodes area ``(i - n/2) / fov``, so the extreme
    is ``(n/2) / fov``, and since ``resolution = fov / n`` that is
    ``1 / (2 * resolution)``. Field of view and matrix size cancel, so neither
    is asked for.

    This builds **one** gradient. Which views are played, and in what order, is
    a scan loop's business; scale this template per view.

    Parameters
    ----------
    channel : str
        Gradient channel (``"x"``, ``"y"`` or ``"z"``).
    resolution : float
        Target resolution along ``channel`` (m) -- ``fov / matrix`` for an
        in-plane encode, the partition spacing for a 3D slab encode.
    system : pypulseq.Opts, optional
        System limits.
    duration : float, optional
        Force a duration (s); the default is the shortest feasible.

    Returns
    -------
    TrapEvent
        Trapezoid of area ``1 / (2 * resolution)``, to be
        :func:`~pypulseq.scale_grad`-ed per view.

    Raises
    ------
    ValueError
        If ``resolution`` is not positive.

    Examples
    --------
    >>> import pulserver.pypulseq as pp
    >>> gy = pp.make_phase_encoding("y", 0.22 / 64, system=pp.Opts())
    >>> round(gy.area, 3)
    145.455

    The same call encodes a 3D slab's partitions, the resolution along ``z``
    being the partition spacing:

    >>> round(pp.make_phase_encoding("z", 1e-3, system=pp.Opts()).area, 1)
    500.0

    See Also
    --------
    make_phase_blip : the small step played between echoes of a train.
    """
    resolution = float(resolution)
    if not resolution > 0.0:
        raise ValueError(f"resolution must be > 0, got {resolution}")
    return _trapezoid(channel, 1.0 / (2.0 * resolution), system, duration)


def make_phase_blip(
    channel: str,
    fov: float,
    steps: float = 1.0,
    system=None,
    duration: float | None = None,
):
    """The short gradient played between echoes of a train, in k-space cells.

    One cell is ``1 / fov``, so ``steps=1`` moves one line and ``steps=R``
    implements an acceleration of ``R`` with no further arithmetic. Negative
    ``steps`` blip backwards.

    Parameters
    ----------
    channel : str
        Gradient channel (``"x"``, ``"y"`` or ``"z"``).
    fov : float
        Field of view along ``channel`` (m).
    steps : float, optional
        Cells to traverse; non-zero, may be negative.
    system : pypulseq.Opts, optional
        System limits.
    duration : float, optional
        Force a duration (s); the default is the shortest feasible.

    Returns
    -------
    TrapEvent
        Trapezoid of area ``steps / fov``.

    Raises
    ------
    ValueError
        If ``fov`` is not positive or ``steps`` is zero.

    Examples
    --------
    >>> import pulserver.pypulseq as pp
    >>> round(pp.make_phase_blip("y", 0.24, steps=2, system=pp.Opts()).area, 3)
    8.333

    See Also
    --------
    make_phase_encoding : the full-area encode played once per shot.
    """
    fov = float(fov)
    if not fov > 0.0:
        raise ValueError(f"fov must be > 0, got {fov}")
    if steps == 0:
        raise ValueError("steps must be non-zero")
    return _trapezoid(channel, float(steps) / fov, system, duration)


def make_crusher(
    dephasing_cycles: float,
    voxel_size: float,
    channel: str,
    grad_start: float = 0.0,
    grad_end: float = 0.0,
    convert_to_arbitrary: bool = False,
    system=None,
):
    """A crusher stated as cycles of dephasing across a voxel.

    :func:`~pypulseq.make_extended_trapezoid_area` with its ``area`` replaced by
    the two numbers a sequence knows: ``dephasing_cycles / voxel_size`` is the
    area that winds that many cycles of phase across ``voxel_size`` metres along
    ``channel``. Same argument order otherwise, and the same return.

    ``grad_start`` and ``grad_end`` let the crusher ride a gradient that is
    already running -- the slice-select lobe around a refocusing pulse -- rather
    than requiring it to fall to zero first.

    Parameters
    ----------
    dephasing_cycles : float
        Cycles of phase to accrue across ``voxel_size``.
    voxel_size : float
        Length over which the dephasing is counted (m).
    channel : str
        Gradient channel (``"x"``, ``"y"`` or ``"z"``).
    grad_start, grad_end : float, optional
        Amplitudes to begin and end at (Hz/m).
    convert_to_arbitrary : bool, optional
        Return the gradient as an arbitrary waveform.
    system : pypulseq.Opts, optional
        System limits.

    Returns
    -------
    grad : GradEvent
        The crusher.
    times : numpy.ndarray
        Its vertex times (s).
    amplitudes : numpy.ndarray
        Its vertex amplitudes (Hz/m).

    Raises
    ------
    ValueError
        If ``voxel_size`` or ``dephasing_cycles`` is not positive.

    Examples
    --------
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts(max_grad=30, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    >>> crusher, times, amplitudes = pp.make_crusher(4.0, 5e-3, "z", system=system)
    >>> round(float(amplitudes[0])), round(float(amplitudes[-1]))
    (0, 0)

    Four cycles across a 5 mm voxel is an area of 800 per metre:

    >>> import numpy as np
    >>> round(float(np.trapezoid(amplitudes, times)), 6)
    800.0

    See Also
    --------
    make_hexagon_gradient_area : the same area without holding a plateau.
    """
    dephasing_cycles = float(dephasing_cycles)
    voxel_size = float(voxel_size)
    if not voxel_size > 0.0:
        raise ValueError(f"voxel_size must be > 0, got {voxel_size}")
    if not dephasing_cycles > 0.0:
        raise ValueError(f"dephasing_cycles must be > 0, got {dephasing_cycles}")

    from pypulseq.make_extended_trapezoid_area import make_extended_trapezoid_area

    grad, times, amplitudes = make_extended_trapezoid_area(
        area=dephasing_cycles / voxel_size,
        channel=channel,
        grad_start=grad_start,
        grad_end=grad_end,
        convert_to_arbitrary=convert_to_arbitrary,
        system=default_system(system),
    )
    return _events.convert(grad), times, amplitudes


def _trapezoid(channel: str, area: float, system, duration: float | None):
    """One trapezoid of the given area, with the duration left free unless set."""
    arguments = {"channel": channel, "area": area, "system": default_system(system)}
    if duration is not None:
        arguments["duration"] = duration
    return _events.make_trapezoid(**arguments)
