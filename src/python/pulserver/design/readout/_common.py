"""The few things every readout module would otherwise write out twice."""

from __future__ import annotations

__all__ = [
    "AXES",
    "WAVE_MODES",
    "as_tuple",
    "bridge",
    "left_align_rephaser",
    "present",
    "solve_delay",
    "solve_rephasing",
    "wave_gradients",
]

from typing import Any

import numpy as np

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
        raise ValueError(
            f"{name} must be a scalar or {length} values, got {len(values)}"
        )
    return values


def bridge(
    system: pp.Opts, channel: str, area: float, grad_start: float, grad_end: float
):
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
        area=area,
        channel=channel,
        grad_start=grad_start,
        grad_end=grad_end,
        system=system,
    )
    return grad


def solve_delay(
    requested: float | None, minimum: float, name: str, system: pp.Opts
) -> float:
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


#: What each wave mode drives: the phase axis, the partition axis, or both.
WAVE_MODES = {
    "phase": ("y",),
    "partition": ("z",),
    "both": ("y", "z"),
}


def wave_gradients(
    system: pp.Opts,
    *,
    flat_time: float,
    delay: float,
    cycles: int,
    amplitude: float,
    mode: str = "both",
) -> dict:
    """The corkscrew gradients a wave-encoded readout plays under its lobe.

    A sinusoid on the phase axis and a cosinusoid on the partition axis, a
    quarter period apart, turn the readout into a corkscrew: every voxel is
    smeared along it, the further from the centre the further, so the aliasing
    parallel imaging has to separate is spread out with it.

    Both events are **self-balanced** -- they enter and leave at zero and their
    net area is exactly zero. That is what lets a scan switch the wave off for
    one readout by scaling the event to zero: no rewinder anywhere has to know
    whether it was played, so the readout's own encoding is untouched either
    way. A cosinusoid does not start at zero on its own, so both axes are
    brought in and out over a quarter period at each end, and whatever area
    that leaves is taken back over the same envelope.

    Parameters
    ----------
    system
        The limits the gradients are built against.
    flat_time
        The readout lobe's flat top, where the samples are. The corkscrew
        lives entirely inside it.
    delay
        Where the flat top starts within the block, which is the readout
        lobe's rise time.
    cycles
        Periods of the sinusoid across the flat top.
    amplitude
        Requested peak, in T/m. **A ceiling, not a prescription**: a sinusoid
        of angular frequency ``w`` slews at ``amplitude * w``, so a fast
        corkscrew is bounded by the slew rate rather than by what was asked
        for, and what gets built is the lower of the two.
    mode
        Which axes to drive: ``"phase"``, ``"partition"`` or ``"both"``.

    Returns
    -------
    dict
        The events for the axes the mode drives, by channel, and
        ``"amplitude"``: the peak that survived the slew limit, in T/m.

    Raises
    ------
    ValueError
        If ``mode`` is not one of the three, if ``cycles`` is not positive, or
        if the flat top is too short to hold the cycles asked of it.
    """
    if mode not in WAVE_MODES:
        raise ValueError(f"wave mode must be one of {tuple(WAVE_MODES)}, got {mode!r}")
    cycles = int(cycles)
    if cycles < 1:
        raise ValueError("wave cycles must be at least one")
    if amplitude <= 0:
        raise ValueError("wave amplitude must be positive")

    raster = system.grad_raster_time
    n_flat = round(flat_time / raster)
    # A quarter period is what each end is brought in over, so the corkscrew
    # needs whole periods and enough raster to shape their edges.
    n_edge = n_flat // (4 * cycles)
    if n_edge < 1:
        raise ValueError(
            f"{cycles} wave cycles need at least {4 * cycles} gradient raster periods "
            f"across the readout's flat top, which holds {n_flat}"
        )

    centres = (np.arange(n_flat) + 0.5) * raster
    envelope = np.ones(n_flat)
    edge = 0.5 * (1.0 - np.cos(np.pi * (np.arange(n_edge) + 0.5) / n_edge))
    envelope[:n_edge], envelope[-n_edge:] = edge, edge[::-1]

    rate = 2 * np.pi * cycles / (n_flat * raster)
    shapes = {
        channel: _balanced(
            (np.sin if channel == "y" else np.cos)(rate * centres), envelope
        )
        for channel in WAVE_MODES[mode]
    }

    # The waveform is linear in its amplitude, so the slew it costs per unit of
    # amplitude is a property of the shape and the amplitude that fits follows
    # from it. Measuring beats bounding here: a sinusoid and the envelope that
    # brings it in are steepest at different moments, and adding their worst
    # cases would give away amplitude neither of them takes. The samples sit at
    # raster centres, so the steps into and out of zero cross half a raster and
    # cost twice what the interior ones do.
    steepest = max(
        float(
            np.abs(
                np.concatenate([[2.0 * shape[0]], np.diff(shape), [-2.0 * shape[-1]]])
            ).max()
        )
        / raster
        for shape in shapes.values()
    )
    peak = min(
        float(amplitude) * system.gamma, system.max_slew / steepest, system.max_grad
    )

    events = {
        channel: pp.make_arbitrary_grad(
            channel=channel,
            waveform=peak * shape,
            first=0.0,
            last=0.0,
            delay=delay,
            system=system,
        )
        for channel, shape in shapes.items()
    }
    return {**events, "amplitude": peak / system.gamma}


def _area(waveform: np.ndarray) -> float:
    """The area under an arbitrary gradient entered and left at zero, per raster.

    Its samples sit at raster centres and it reaches zero over half a raster at
    each end, so k follows the straight lines between those points and the
    first and last samples are worth a quarter of what an interior one is.
    Summing the samples instead is a different waveform's area, and one the
    sequence's own k-space would not agree with.
    """
    return float(waveform.sum() - 0.25 * (waveform[0] + waveform[-1]))


def _balanced(shape: np.ndarray, envelope: np.ndarray) -> np.ndarray:
    """``shape`` under ``envelope``, offset until its net area is exactly zero.

    The offset rides the envelope too, so correcting the area cannot put the
    waveform's ends anywhere but zero. Everything here is affine in the
    offset, which makes finding it one division rather than a search.
    """
    at_zero = _area(shape * envelope)
    per_unit = _area(envelope)
    return (shape - at_zero / per_unit) * envelope
