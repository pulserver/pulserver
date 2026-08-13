"""The few things every readout module would otherwise write out twice."""

from __future__ import annotations

__all__ = ["AXES", "as_tuple", "bridge", "solve_delay"]

from typing import Any

from ... import pypulseq as pp

AXES = ("x", "y", "z")


def as_tuple(value: Any, length: int, name: str, cast=float) -> tuple:
    """Broadcast a scalar to ``length``, or check a sequence already is that long."""
    if isinstance(value, int | float):
        return (cast(value),) * length
    values = tuple(cast(item) for item in value)
    if len(values) != length:
        raise ValueError(f"{name} must be a scalar or {length} values, got {len(values)}")
    return values


def bridge(system: pp.Opts, channel: str, area: float, grad_start: float, grad_end: float):
    """The shortest ``grad_start -> ... -> grad_end`` waveform achieving ``area``.

    A spoiler that rides straight off the readout lobe instead of waiting for
    it to fall to zero, which is what keeps a short-TR steady-state sequence
    short. :func:`pypulseq.make_extended_trapezoid_area` searches for the
    slew-safe solution directly, so it stays feasible where a fixed-ramp
    trapezoid would not: the endpoints and the solved plateau may have
    opposite signs and a combined swing approaching twice ``max_grad``, which
    is exactly the readout-against-spoiler case.

    Returns
    -------
    GradEvent
        Left-aligned (``delay`` is zero); shift it by assigning ``delay``.
    """
    grad, _, _ = pp.make_extended_trapezoid_area(
        area=area, channel=channel, grad_start=grad_start, grad_end=grad_end, system=system
    )
    return grad


def solve_delay(requested: float | None, minimum: float, name: str, system: pp.Opts) -> float:
    """The wait that turns ``minimum`` into ``requested``, rounded onto the raster.

    Parameters
    ----------
    requested : float or None
        Target time (s). ``None`` means "as short as possible", which is no
        wait at all.
    minimum : float
        What the module achieves with no wait (s).
    name : str
        What to call the time in the error, e.g. ``"TE"``.
    system : pypulseq.Opts
        System limits, read for the block duration raster.

    Returns
    -------
    float
        Delay to insert (s); zero when ``requested`` is ``None``.

    Raises
    ------
    ValueError
        If ``requested`` is shorter than ``minimum``.
    """
    if requested is None:
        return 0.0
    delay = float(requested) - float(minimum)
    if delay < -1e-12:
        raise ValueError(
            f"the requested {name} of {float(requested) * 1e3:.3f} ms is shorter than the "
            f"{minimum * 1e3:.3f} ms this readout can achieve"
        )
    return pp.round_to_raster(max(delay, 0.0), system.block_duration_raster)
