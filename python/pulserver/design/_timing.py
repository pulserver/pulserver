"""Public ADC/readout timing calculations."""

from __future__ import annotations

from ._system import quantize_readout_timing


def calc_adc_timing(
    num_samples: int,
    target_dwell: float,
    *,
    grad_raster_time: float,
    adc_raster_time: float,
    min_readout_duration: float = 0.0,
) -> tuple[float, float]:
    """Return an ADC dwell and readout duration legal on both time rasters.

    A requested receiver bandwidth almost never lands on a legal dwell: the
    ADC is quantized to ``adc_raster_time`` (tens of nanoseconds) while the
    readout gradient must end on ``grad_raster_time`` (microseconds). This
    solves both constraints at once, returning the achievable dwell closest to
    the request and the readout duration ``num_samples * dwell`` — which is by
    construction a whole number of gradient rasters.

    Use the returned dwell, not the requested one, when reporting the actual
    receiver bandwidth.

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

    Examples
    --------
    >>> from pulserver.design import calc_adc_timing
    >>> dwell, duration = calc_adc_timing(
    ...     96, 3.7e-6, grad_raster_time=10e-6, adc_raster_time=100e-9
    ... )
    >>> round(dwell * 1e6, 4), round(duration * 1e6, 4)
    (5.0, 480.0)
    >>> round(duration / 10e-6, 6)
    48.0

    See Also
    --------
    make_adc : create the ADC event from the returned dwell.
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
