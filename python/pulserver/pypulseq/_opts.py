"""Vendor-neutral Pulseq system defaults."""

from __future__ import annotations

import pypulseq as _pp


class Opts(_pp.Opts):
    """PyPulseq system limits with conservative vendor-neutral rasters.

    The hardware amplitude and slew defaults remain PyPulseq's.  Timing
    defaults deliberately use the coarsest common GE/Siemens values so a
    sequence designed without explicit scanner limits is not silently tied to
    either vendor.
    """

    def __init__(
        self,
        adc_dead_time: float | None = 0.0,
        adc_raster_time: float | None = 2e-6,
        block_duration_raster: float | None = 10e-6,
        gamma: float | None = None,
        grad_raster_time: float | None = 10e-6,
        grad_unit: str = "Hz/m",
        max_grad: float | None = None,
        max_slew: float | None = None,
        rf_dead_time: float | None = 0.0,
        rf_raster_time: float | None = 2e-6,
        rf_ringdown_time: float | None = 0.0,
        adc_samples_limit: int | None = None,
        adc_samples_divisor: int | None = None,
        rise_time: float | None = None,
        slew_unit: str = "Hz/m/s",
        B0: float | None = None,
    ) -> None:
        super().__init__(
            adc_dead_time=adc_dead_time,
            adc_raster_time=adc_raster_time,
            block_duration_raster=block_duration_raster,
            gamma=gamma,
            grad_raster_time=grad_raster_time,
            grad_unit=grad_unit,
            max_grad=max_grad,
            max_slew=max_slew,
            rf_dead_time=rf_dead_time,
            rf_raster_time=rf_raster_time,
            rf_ringdown_time=rf_ringdown_time,
            adc_samples_limit=adc_samples_limit,
            adc_samples_divisor=adc_samples_divisor,
            rise_time=rise_time,
            slew_unit=slew_unit,
            B0=B0,
        )

    def set_as_default(self) -> None:
        """Use this object as the default for Pulserver RF factories."""
        type(self).default = self

    @classmethod
    def reset_default(cls) -> None:
        """Restore Pulserver's vendor-neutral defaults."""
        cls.default = cls()


Opts.default = Opts()


def default_system(system: _pp.Opts | None) -> _pp.Opts:
    """Return ``system`` or Pulserver's current default limits."""
    return Opts.default if system is None else system
