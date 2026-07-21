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
    """Return feasible ADC dwell and readout duration on both time rasters."""
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
