"""Arithmetic every readout module does before it lays out a block."""

from __future__ import annotations

__all__ = [
    "AXES",
    "DEFAULT_BANDWIDTH_HZ",
    "ReadoutSampling",
    "as_tuple",
    "bridge",
    "readout_sampling",
    "solve_delay",
]

from dataclasses import dataclass
from typing import Any

from ... import pypulseq as pp

#: Receiver bandwidth (Hz) a readout designs at unless told another.
DEFAULT_BANDWIDTH_HZ = 250e3

#: Fraction of ``max_grad`` a readout lobe is allowed to reach, leaving the
#: rest for the phase encodes riding alongside it.
_READOUT_GRAD_MARGIN = 0.8

AXES = ("x", "y", "z")


def as_tuple(value: Any, length: int, name: str, cast=float) -> tuple:
    """Broadcast a scalar to ``length``, or check a sequence already is that long."""
    if isinstance(value, int | float):
        return (cast(value),) * length
    values = tuple(cast(item) for item in value)
    if len(values) != length:
        raise ValueError(f"{name} must be a scalar or {length} values, got {len(values)}")
    return values


@dataclass(frozen=True)
class ReadoutSampling:
    """How one readout line is sampled, and where its echo falls.

    Attributes
    ----------
    num_samples : int
        ADC samples acquired.
    num_pre, num_post : int
        Samples before and after the echo. Partial echo shortens ``num_pre``.
    delta_k : float
        K-space step between samples (1/m).
    k_width : float
        Total k-space traversed by the flat top (1/m).
    dwell : float
        ADC dwell (s). Report ``1 / dwell`` as the achieved bandwidth.
    duration : float
        Flat-top duration (s), a whole number of gradient rasters.
    """

    num_samples: int
    num_pre: int
    num_post: int
    delta_k: float
    k_width: float
    dwell: float
    duration: float

    @property
    def echo_offset(self) -> float:
        """Time from the start of the flat top to the k = 0 crossing (s)."""
        return self.num_pre * self.dwell


def readout_sampling(
    system: pp.Opts,
    matrix_x: int,
    fov_x_m: float,
    *,
    oversampling: float = 1.0,
    partial_echo: float = 1.0,
    bandwidth_hz: float = DEFAULT_BANDWIDTH_HZ,
) -> ReadoutSampling:
    """Sample counts, a k-space step and a dwell legal on both time rasters.

    Oversampling densifies the sampling: ``delta_k`` shrinks and the sampled
    field of view grows, while the k-space width -- and so the resolution --
    is fixed by ``matrix_x`` and ``fov_x_m`` alone. Partial echo truncates the
    samples *before* the echo, which moves the echo earlier in the ADC and
    shortens the prephaser rather than the readout gradient.

    The dwell comes from :func:`pulserver.pypulseq.calc_adc_timing`, which
    solves the ADC and gradient rasters together, so the bandwidth achieved is
    generally not the one requested.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits: the two rasters, and ``max_grad`` for the shortest
        readout that fits.
    matrix_x : int
        Readout matrix size.
    fov_x_m : float
        Field of view along the readout (m).
    oversampling : float, optional
        Readout oversampling factor (>= 1).
    partial_echo : float, optional
        Fraction of the full echo acquired, in ``(0.5, 1]``.
    bandwidth_hz : float, optional
        Requested receiver bandwidth (Hz).

    Returns
    -------
    ReadoutSampling

    Raises
    ------
    ValueError
        If ``matrix_x`` is below 2, ``fov_x_m`` or ``bandwidth_hz`` is not
        positive, ``oversampling`` is below 1, or ``partial_echo`` is outside
        ``(0.5, 1]``.
    """
    if int(matrix_x) < 2:
        raise ValueError("the readout matrix must be at least 2")
    if fov_x_m <= 0:
        raise ValueError("fov_x_m must be positive")
    if oversampling < 1.0:
        raise ValueError("oversampling must be >= 1")
    if not 0.5 < partial_echo <= 1.0:
        raise ValueError("partial_echo must be in (0.5, 1]")
    if bandwidth_hz <= 0:
        raise ValueError("bandwidth_hz must be positive")

    delta_k = 1.0 / (oversampling * fov_x_m)
    num_full = round(oversampling * int(matrix_x))
    num_post = num_full // 2
    num_samples = max(num_post + 1, round(partial_echo * num_full))
    num_pre = num_samples - num_post
    k_width = num_samples * delta_k

    dwell, duration = pp.calc_adc_timing(
        num_samples,
        1.0 / bandwidth_hz,
        grad_raster_time=system.grad_raster_time,
        adc_raster_time=system.adc_raster_time,
        min_readout_duration=k_width / (_READOUT_GRAD_MARGIN * system.max_grad),
    )
    return ReadoutSampling(
        num_samples=num_samples,
        num_pre=num_pre,
        num_post=num_post,
        delta_k=delta_k,
        k_width=k_width,
        dwell=dwell,
        duration=duration,
    )


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
