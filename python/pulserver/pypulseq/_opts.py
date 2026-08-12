"""Vendor-neutral Pulseq system defaults."""

from __future__ import annotations

import pypulseq as _pp


class Opts(_pp.Opts):
    """PyPulseq system limits with vendor-neutral raster defaults.

    The constructor signature is upstream's, argument for argument, so an
    :class:`Opts` built by a PyPulseq script behaves identically here. Only
    the *defaults* differ, and only for the rasters and the dead/ringdown
    times.

    The raster defaults are the common multiples of the two vendors' hardware
    rasters -- 20 us for gradients and block durations (GE 4 us, Siemens
    10 us), 2 us for RF and ADC (Siemens 1 us and 100 ns, GE 2 us) -- so a
    sequence designed without explicit scanner limits plays on either without
    re-rastering. Dead and ringdown times default to zero because they are
    guidelines rather than hardware constants; set them from the system you
    are targeting when it matters.

    Amplitude, slew, gamma and B0 defaults remain PyPulseq's.

    Notes
    -----
    Coarser gradient rasters quantise ramps more coarsely, so a short blip
    designed at 20 us may come out longer than one designed at 10 us. Pass
    ``grad_raster_time`` explicitly when targeting one vendor.
    """

    def __init__(
        self,
        adc_dead_time: float | None = 0.0,
        adc_raster_time: float | None = 2e-6,
        block_duration_raster: float | None = 20e-6,
        gamma: float | None = None,
        grad_raster_time: float | None = 20e-6,
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
        # Every raster is passed explicitly rather than left None: upstream
        # fills a None from ``pypulseq.Opts.default``, which is the Siemens
        # table, not this class's.
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
        """Use this object as the default for Pulserver's factories."""
        type(self).default = self

    @classmethod
    def reset_default(cls) -> None:
        """Restore Pulserver's vendor-neutral defaults."""
        cls.default = cls()


Opts.default = Opts()


def default_system(system: _pp.Opts | None) -> _pp.Opts:
    """Return ``system`` or Pulserver's current default limits."""
    return Opts.default if system is None else system
