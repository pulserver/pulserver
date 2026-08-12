"""Making an ADC and its readout gradient land on both time rasters at once."""

from __future__ import annotations

__all__ = ["calc_adc_timing", "ceil_to_raster", "quantize_readout_timing", "round_to_raster"]

import math

import numpy as np

#: How far the dwell may be widened past the requested one while looking for a
#: readout duration that lands on the gradient raster. Large enough that no
#: realistic (samples, raster) pair runs out of room before finding one.
MAX_RASTER_SEARCH_STEPS = 50_000


def calc_adc_timing(
    num_samples: int,
    target_dwell: float,
    *,
    grad_raster_time: float,
    adc_raster_time: float,
    min_readout_duration: float = 0.0,
) -> tuple[float, float]:
    """Return an ADC dwell and readout duration legal on both time rasters.

    A requested receiver bandwidth almost never lands on a legal dwell: the ADC
    is quantized to ``adc_raster_time`` while the readout gradient must end on
    ``grad_raster_time``. This solves both at once, returning the achievable
    dwell closest to the request and the duration ``num_samples * dwell``, which
    is by construction a whole number of gradient rasters -- so the ADC can be
    laid exactly over the flat top.

    Report the returned dwell as the actual receiver bandwidth, not the
    requested one.

    Parameters
    ----------
    num_samples : int
        Number of ADC samples (>= 1).
    target_dwell : float
        Requested dwell time (s), i.e. ``1 / bandwidth``.
    grad_raster_time : float
        Gradient raster (s), e.g. ``system.grad_raster_time``.
    adc_raster_time : float
        ADC raster (s), e.g. ``system.adc_raster_time``.
    min_readout_duration : float, optional
        Lower bound on the returned duration (s), e.g. to fit a flat top.

    Returns
    -------
    dwell : float
        Feasible dwell time (s), a multiple of ``adc_raster_time``.
    duration : float
        ``num_samples * dwell`` (s), a multiple of ``grad_raster_time``.

    Raises
    ------
    ValueError
        If ``num_samples`` is below one, or any time is not positive.

    Examples
    --------
    >>> import pulserver.pypulseq as pp
    >>> dwell, duration = pp.calc_adc_timing(
    ...     96, 3.7e-6, grad_raster_time=10e-6, adc_raster_time=100e-9
    ... )
    >>> round(dwell * 1e6, 4), round(duration * 1e6, 4)
    (5.0, 480.0)
    >>> round(duration / 10e-6, 6)
    48.0

    See Also
    --------
    make_adc : build the ADC event from the returned dwell.
    calc_adc_segments : split a long ADC into equal segments.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")
    if target_dwell <= 0 or grad_raster_time <= 0 or adc_raster_time <= 0:
        raise ValueError("target_dwell and raster times must be > 0")
    if min_readout_duration < 0:
        raise ValueError("min_readout_duration must be >= 0")
    return quantize_readout_timing(
        num_samples,
        1.0 / target_dwell,
        grad_raster_time,
        adc_raster_time,
        min_readout_duration,
    )


def quantize_readout_timing(
    nx_ro: int,
    target_bw_hz_px: float,
    grad_raster_s: float,
    adc_raster_s: float,
    min_flat_time_s: float,
) -> tuple[float, float]:
    """The same search, keyed on bandwidth per pixel rather than on dwell.

    Walks upward from the smallest ADC-raster-multiple dwell that meets
    ``min_flat_time_s``, and keeps the one whose ``nx_ro * dwell`` lands on the
    gradient raster closest to the requested bandwidth.

    Parameters
    ----------
    nx_ro : int
        Number of readout samples.
    target_bw_hz_px : float
        Requested receiver bandwidth per pixel (Hz).
    grad_raster_s, adc_raster_s : float
        Gradient and ADC rasters (s).
    min_flat_time_s : float
        Lower bound on the flat time (s), e.g. from the max-gradient limit.

    Returns
    -------
    dwell_s : float
        ADC dwell time (s), a multiple of ``adc_raster_s``.
    flat_time_s : float
        ``nx_ro * dwell_s`` (s), a multiple of ``grad_raster_s``.

    Examples
    --------
    >>> from pulserver.pypulseq._timing import quantize_readout_timing
    >>> dwell, flat = quantize_readout_timing(256, 62_500.0, 1e-5, 1e-7, 0.0)
    >>> abs(flat / 1e-5 - round(flat / 1e-5)) < 1e-9
    True
    """
    target_dwell = 1.0 / target_bw_hz_px
    m_target = max(1, round(target_dwell / adc_raster_s))

    m_min = max(m_target, int(np.ceil(min_flat_time_s / (nx_ro * adc_raster_s))))

    best_m = None
    best_cost = None

    for step in range(MAX_RASTER_SEARCH_STEPS):
        m = m_min + step
        flat = nx_ro * m * adc_raster_s
        ratio = flat / grad_raster_s
        if abs(ratio - round(ratio)) > 1e-9:
            continue
        cost = abs((m * adc_raster_s) - target_dwell)
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_m = m
            if cost == 0.0:
                break

    if best_m is None:
        best_m = m_min

    dwell = best_m * adc_raster_s
    return dwell, nx_ro * dwell


def round_to_raster(value_s: float, raster_s: float = 1e-5) -> float:
    """Round ``value_s`` to the nearest multiple of ``raster_s``.

    An RF pulse's center of mass -- an adiabatic hypsec pulse's weighted center,
    say -- generally does not fall on a raster boundary, so a delay computed
    from it has to be re-aligned before it can be a block duration.

    Parameters
    ----------
    value_s : float
        Time to round (s).
    raster_s : float, optional
        Raster period (s).

    Returns
    -------
    float
        ``value_s`` rounded to the nearest raster multiple.

    Examples
    --------
    >>> import pulserver.pypulseq as pp
    >>> round(pp.round_to_raster(1.23456e-3), 12)
    0.00123
    """
    return round(value_s / raster_s) * raster_s


def ceil_to_raster(value_s: float, raster_s: float) -> float:
    """Round ``value_s`` up to the next multiple of ``raster_s``.

    A small tolerance keeps a value already on the raster from being pushed up
    a full step by floating-point noise.

    Parameters
    ----------
    value_s : float
        Time to round (s).
    raster_s : float
        Raster period (s).

    Returns
    -------
    float
        Smallest raster multiple ``>= value_s`` (up to tolerance).

    Examples
    --------
    >>> import pulserver.pypulseq as pp
    >>> pp.ceil_to_raster(1.01e-5, 1e-5)
    2e-05
    >>> pp.ceil_to_raster(2e-5, 1e-5)
    2e-05
    """
    return math.ceil(value_s / raster_s - 1e-10) * raster_s
