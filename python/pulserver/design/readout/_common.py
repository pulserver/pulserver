"""The few things every readout module would otherwise write out twice."""

from __future__ import annotations

__all__ = [
    "AXES",
    "as_tuple",
    "bridge",
    "left_align_rephaser",
    "present",
    "solve_delay",
    "solve_rephasing",
]

from typing import Any

from ... import pypulseq as pp

AXES = ("x", "y", "z")


def present(event: Any) -> tuple:
    """``(event,)`` when there is one, so it can be splatted into a block."""
    return () if event is None else (event,)


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


def left_align_rephaser(gz_reph: Any, occupied: tuple[str, ...], owner: str):
    """A slice rephaser placed at the head of its block, or ``None``.

    Left-aligned because a rephaser has to run straight off the selection lobe:
    anything between the two is time the slice spends dephasing for nothing.

    Parameters
    ----------
    gz_reph : GradEvent or None
        The rephaser to place.
    occupied : tuple of str
        Channels the block it would join already plays a gradient on.
    owner : str
        Class name, for the error.

    Returns
    -------
    GradEvent or None
        A new event with zero delay; the caller's is left untouched.

    Raises
    ------
    ValueError
        If the block already plays a gradient on the rephaser's channel.
    """
    if gz_reph is None:
        return None
    if gz_reph.channel in occupied:
        raise ValueError(
            f"{owner} already plays a gradient on {gz_reph.channel} in the block the slice "
            f"rephaser would go in; excite with is_slab=True so the rephaser is carried by "
            f"the selection gradient itself"
        )
    return pp.align(left=[gz_reph])[0]


def solve_rephasing(
    te: float | None,
    te_base: float,
    pre_span: float,
    reph_span: float,
    system: pp.Opts,
) -> tuple[float, float, float]:
    """Size the two blocks between the pulse and the acquisition.

    The rephaser goes in the first block after the pulse -- the TE wait when
    there is one, the prewinder block otherwise -- so that nothing separates it
    from the selection lobe.

    Parameters
    ----------
    te : float or None
        Requested echo time (s). ``None`` is as short as possible.
    te_base : float
        The part of the echo time neither block accounts for (s): the tail of
        the pulse block plus the acquisition's own lead-in.
    pre_span : float
        Prewinder block duration with no rephaser in it (s).
    reph_span : float
        Rephaser duration (s), zero when there is none.
    system : pypulseq.Opts
        System limits, read for the block duration raster.

    Returns
    -------
    wait : float
        TE wait block duration (s); zero when there is no wait block, which is
        also what says the rephaser belongs in the prewinder block.
    pre : float
        Prewinder block duration (s).
    echo_time : float
        Achieved echo time (s).

    Raises
    ------
    ValueError
        If ``te`` is shorter than the layout can achieve.
    """
    raster = system.block_duration_raster
    pre_span = pp.ceil_to_raster(pre_span, raster)
    reph_span = pp.ceil_to_raster(reph_span, raster)

    merged = max(pre_span, reph_span)
    te_min = te_base + merged
    delay = solve_delay(te, te_min, "TE", system)

    wait = delay + merged - pre_span
    if delay and wait >= reph_span:
        return wait, pre_span, te_min + delay

    # The wait is too short to hold the rephaser, so moving it there would push
    # the echo past the TE that was asked for. It stays where it already fits,
    # and the prewinder block absorbs the wait by starting later.
    return 0.0, merged + delay, te_min + delay
