"""SLR, hard, frequency-selective, and slice-selective RF pulse factories."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pypulseq as pp

from ._base import RfPulse
from ._slr import design_slr

PulseType = Literal["st", "ex", "se", "inv", "sat"]
FilterType = Literal["ls", "pm", "min", "max", "ms"]

DEFAULT_DURATION = 4e-3
DEFAULT_TIME_BANDWIDTH_PRODUCT = 4.0
DEFAULT_SLICE_DURATION = 2e-3
RF_SPOILING_INCREMENT_DEG = 117.0


def _system_or_default(system: pp.Opts | None) -> pp.Opts:
    return pp.Opts.default if system is None else system


def _slr_sample_count(duration: float, dwell: float) -> int:
    if duration <= 0:
        raise ValueError("duration must be > 0")
    if dwell <= 0:
        raise ValueError("dwell must be > 0")
    count = max(8, int(round(duration / dwell)))
    return count + count % 2


def make_hard_pulse(flip_angle: float, *, system: pp.Opts | None = None, **kwargs) -> RfPulse:
    """Wrap :func:`pypulseq.make_block_pulse` in the RF module protocol."""
    system = _system_or_default(system)
    event = pp.make_block_pulse(flip_angle, system=system, **kwargs)
    return RfPulse(system, event)


def make_adiabatic_pulse(
    pulse_type: str,
    *,
    system: pp.Opts | None = None,
    **kwargs,
) -> RfPulse:
    """Wrap :func:`pypulseq.make_adiabatic_pulse` in the RF module protocol."""
    system = _system_or_default(system)
    result = pp.make_adiabatic_pulse(pulse_type=pulse_type, system=system, **kwargs)
    if isinstance(result, tuple):
        event, gradient, rephaser = result
        return RfPulse(system, event, (gradient,), (rephaser,))
    return RfPulse(system, result)


def make_sigpy_pulse(
    flip_angle: float,
    *,
    duration: float = DEFAULT_DURATION,
    delay: float = 0.0,
    dwell: float = 0.0,
    freq_offset: float = 0.0,
    phase_offset: float = 0.0,
    center_pos: float = 0.5,
    slice_thickness: float = 0.0,
    return_gz: bool = False,
    time_bw_product: float = DEFAULT_TIME_BANDWIDTH_PRODUCT,
    pulse_type: PulseType = "st",
    filter_type: FilterType = "ls",
    passband_ripple: float = 0.01,
    stopband_ripple: float = 0.01,
    cancel_alpha_phase: bool = False,
    max_grad: float = 0.0,
    max_slew: float = 0.0,
    system: pp.Opts | None = None,
    use: str = "undefined",
    freq_ppm: float = 0.0,
    phase_ppm: float = 0.0,
):
    """Create an SLR pulse without requiring SigPy.

    The interface follows pypulseq's former SigPy pulse factory. The result is
    an :class:`RfPulse`; when ``return_gz`` is true its ``gradients`` and
    ``rephasers`` contain the slice-selection pair.
    """
    system = _system_or_default(system)
    dwell = system.rf_raster_time if dwell == 0 else dwell
    if not 0.0 <= center_pos <= 1.0:
        raise ValueError("center_pos must lie in [0, 1]")
    if return_gz and slice_thickness <= 0:
        raise ValueError("slice_thickness must be > 0 when return_gz=True")

    n = _slr_sample_count(duration, dwell)
    waveform = design_slr(
        n,
        time_bw_product,
        pulse_type=pulse_type,
        filter_type=filter_type,
        passband_ripple=passband_ripple,
        stopband_ripple=stopband_ripple,
        cancel_alpha_phase=cancel_alpha_phase,
    )
    actual_duration = n * dwell
    result = pp.make_arbitrary_rf(
        signal=waveform,
        flip_angle=flip_angle,
        delay=delay,
        dwell=dwell,
        freq_offset=freq_offset,
        phase_offset=phase_offset,
        return_gz=return_gz,
        slice_thickness=slice_thickness,
        bandwidth=time_bw_product / actual_duration,
        time_bw_product=time_bw_product,
        max_grad=max_grad,
        max_slew=max_slew,
        system=system,
        use=use,
        freq_ppm=freq_ppm,
        phase_ppm=phase_ppm,
        center=center_pos * actual_duration,
    )
    if not return_gz:
        return RfPulse(system, result)

    rf, gz = result
    flat_area = gz.amplitude * gz.flat_time
    ramp_area = gz.area - flat_area
    rephase_area = -flat_area * (1.0 - center_pos) - 0.5 * ramp_area
    gz_reph = pp.make_trapezoid(channel="z", area=rephase_area, system=system)
    return RfPulse(system, rf, (gz,), (gz_reph,))


# The user-facing SLR name is intentionally the same implementation object.
make_slr_pulse = make_sigpy_pulse


def make_frequency_selective_pulse(
    flip_angle: float,
    bandwidth: float,
    *,
    freq_offset: float = 0.0,
    duration: float | None = None,
    time_bw_product: float = DEFAULT_TIME_BANDWIDTH_PRODUCT,
    pulse_type: PulseType = "st",
    filter_type: FilterType = "ls",
    system: pp.Opts | None = None,
    use: str = "excitation",
    **kwargs,
):
    """Create a non-spatial SLR pulse centered at ``freq_offset``.

    ``bandwidth`` and ``time_bw_product`` determine the duration when it is
    omitted, leaving the API in the usual MR design terms.
    """
    if bandwidth <= 0:
        raise ValueError("bandwidth must be > 0")
    if duration is None:
        duration = time_bw_product / bandwidth
    elif not math.isclose(time_bw_product / duration, bandwidth, rel_tol=0.02):
        raise ValueError("bandwidth, duration, and time_bw_product are inconsistent")
    resolved_system = _system_or_default(system)
    # Spectral pulses are typically long; 10 us retains ample spectral
    # resolution without solving unnecessarily large FIR systems at 1 us.
    kwargs.setdefault("dwell", max(10e-6, resolved_system.rf_raster_time))
    return make_slr_pulse(
        flip_angle,
        duration=duration,
        freq_offset=freq_offset,
        time_bw_product=time_bw_product,
        pulse_type=pulse_type,
        filter_type=filter_type,
        system=resolved_system,
        use=use,
        **kwargs,
    )


def make_slice_selective_pulse(
    flip_angle: float,
    slice_thickness: float,
    *,
    duration: float = DEFAULT_SLICE_DURATION,
    time_bw_product: float = DEFAULT_TIME_BANDWIDTH_PRODUCT,
    system: pp.Opts | None = None,
    use: str = "excitation",
    **kwargs,
):
    """Create a slice/slab-selective SLR module with selection and rephasing gradients."""
    return make_slr_pulse(
        flip_angle,
        duration=duration,
        slice_thickness=slice_thickness,
        return_gz=True,
        time_bw_product=time_bw_product,
        pulse_type="ex",
        system=system,
        use=use,
        **kwargs,
    )


def make_refocusing_pulse(
    *,
    slice_thickness: float | None = None,
    duration: float = 3e-3,
    time_bw_product: float = DEFAULT_TIME_BANDWIDTH_PRODUCT,
    phase_offset: float = np.pi / 2.0,
    system: pp.Opts | None = None,
    **kwargs,
):
    """Create a nominal pi SLR refocusing pulse.

    With ``slice_thickness=None`` the module is broad-band and nonselective.
    Otherwise its selection and rephasing events are exposed through
    ``gradients``/``rephasers``. FSE code may use ``module.rf`` and
    ``module.gradients[0]``, then rescale the RF envelope per echo.
    """
    return make_slr_pulse(
        np.pi,
        duration=duration,
        phase_offset=phase_offset,
        slice_thickness=0.0 if slice_thickness is None else slice_thickness,
        return_gz=slice_thickness is not None,
        time_bw_product=time_bw_product,
        pulse_type="se",
        system=system,
        use="refocusing",
        **kwargs,
    )


def make_inversion_pulse(
    *,
    adiabatic: bool = True,
    slice_thickness: float | None = None,
    duration: float = 10.24e-3,
    time_bw_product: float = 6.0,
    system: pp.Opts | None = None,
    **kwargs,
):
    """Create a nominal pi inversion pulse, optionally slice/slab selective."""
    system = _system_or_default(system)
    if adiabatic:
        result = pp.make_adiabatic_pulse(
            pulse_type="hypsec",
            duration=duration,
            dwell=kwargs.pop("dwell", 10e-6),
            slice_thickness=0.0 if slice_thickness is None else slice_thickness,
            return_gz=slice_thickness is not None,
            system=system,
            use="inversion",
            **kwargs,
        )
        if isinstance(result, tuple):
            event, gradient, rephaser = result
            return RfPulse(system, event, (gradient,), (rephaser,))
        return RfPulse(system, result)
    return make_slr_pulse(
        np.pi,
        duration=duration,
        slice_thickness=0.0 if slice_thickness is None else slice_thickness,
        return_gz=slice_thickness is not None,
        time_bw_product=time_bw_product,
        pulse_type="inv",
        system=system,
        use="inversion",
        **kwargs,
    )
