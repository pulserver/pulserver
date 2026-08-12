"""RF pulse factories, each returning events rather than a module.

They follow :func:`pypulseq.make_sinc_pulse`'s shape: the pulse alone by
default, and ``(rf, gz, gz_reph)`` under ``return_gz=True`` where a selection
gradient makes sense.
"""

from __future__ import annotations

__all__ = ["make_sigpy_pulse", "make_slr_pulse", "make_sms_pulse", "make_spsp_pulse"]

from typing import Literal, Sequence

import numpy as np

from . import _events
from ._opts import default_system
from ._slr import design_slr

PulseType = Literal["st", "ex", "se", "inv", "sat"]
FilterType = Literal["ls", "pm", "min", "max", "ms"]

DEFAULT_DURATION = 4e-3
DEFAULT_TIME_BANDWIDTH_PRODUCT = 4.0


def _slr_sample_count(duration: float, dwell: float) -> int:
    if duration <= 0:
        raise ValueError("duration must be > 0")
    if dwell <= 0:
        raise ValueError("dwell must be > 0")
    count = round(duration / dwell)
    if count < 4:
        raise ValueError("duration must span at least four RF samples")
    return count


def make_slr_pulse(
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
    system=None,
    use: str = "undefined",
    freq_ppm: float = 0.0,
    phase_ppm: float = 0.0,
):
    """An SLR-designed pulse, with the FIR filter solved in-package.

    Shinnar-Le Roux turns a flip-angle profile into a filter design problem,
    which gives a far sharper slice at a given time-bandwidth product than a
    windowed sinc, and lets the passband and stopband ripple be specified
    rather than discovered. SciPy solves the filter, so SigPy is not needed.

    Parameters
    ----------
    flip_angle : float
        Nominal flip angle (rad).
    duration : float, optional
        Pulse duration (s).
    delay : float, optional
        Delay before the pulse (s).
    dwell : float, optional
        RF sample spacing (s); ``0`` uses ``system.rf_raster_time``.
    freq_offset, phase_offset : float, optional
        Frequency (Hz) and phase (rad) offsets.
    center_pos : float, optional
        Where the effective centre sits in the pulse, in ``[0, 1]``.
    slice_thickness : float, optional
        Slice thickness (m); required when ``return_gz``.
    return_gz : bool, optional
        Also return the selection gradient and its rephaser.
    time_bw_product : float, optional
        Time-bandwidth product.
    pulse_type : {'st', 'ex', 'se', 'inv', 'sat'}, optional
        Small-tip, excitation, spin-echo, inversion or saturation.
    filter_type : {'ls', 'pm', 'min', 'max', 'ms'}, optional
        FIR design method.
    passband_ripple, stopband_ripple : float, optional
        Ripple allowed in each band.
    cancel_alpha_phase : bool, optional
        Remove the SLR alpha polynomial's phase.
    max_grad, max_slew : float, optional
        Override the system limits for the selection gradient.
    system : pypulseq.Opts, optional
        System limits.
    use : str, optional
        Pulseq ``use`` tag.
    freq_ppm, phase_ppm : float, optional
        Field-strength-relative offsets.

    Returns
    -------
    RfEvent, or tuple
        The pulse, or ``(rf, gz, gz_reph)`` when ``return_gz``.

    Raises
    ------
    ValueError
        If ``center_pos`` is outside ``[0, 1]``, or ``return_gz`` is asked for
        without a positive ``slice_thickness``.

    Examples
    --------
    >>> import numpy as np
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    >>> rf, gz, gz_reph = pp.make_slr_pulse(
    ...     np.deg2rad(90), slice_thickness=5e-3, return_gz=True, system=system
    ... )
    >>> rf.type, gz.channel
    ('rf', 'z')

    See Also
    --------
    make_sinc_pulse : the windowed-sinc alternative.
    sim_rf : simulate the profile the design actually produces.
    """
    system = default_system(system)
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
    result = _events.make_arbitrary_rf(
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
        return result

    rf, gz = result
    flat_area = gz.amplitude * gz.flat_time
    ramp_area = gz.area - flat_area
    rephase_area = -flat_area * (1.0 - center_pos) - 0.5 * ramp_area
    return rf, gz, _events.make_trapezoid(channel="z", area=rephase_area, system=system)


#: The name PyPulseq's SigPy-backed factory went by. Same design, no SigPy.
make_sigpy_pulse = make_slr_pulse


def make_sms_pulse(
    rf,
    num_bands: int,
    band_offset: float,
    *,
    sideband_power: float | Sequence[float] = 1.0,
    phases: str | Sequence[float] | None = None,
):
    """Modulate one RF pulse into equispaced spectral bands.

    Multiplies the envelope by a sum of complex exponentials, so one designed
    pulse excites several slices at once. ``sideband_power`` is power relative
    to the on-resonance band rather than an amplitude multiplier, so the
    modulation uses its square root; a scalar applies to every off-resonance
    band, and one value per band gives asymmetric saturation.

    The offset list always contains 0 Hz. For an even band count the otherwise
    unpaired band goes on the positive-frequency side.

    Parameters
    ----------
    rf : RfEvent or SimpleNamespace
        Base pulse to modulate.
    num_bands : int
        Number of bands, counting the on-resonance one.
    band_offset : float
        Spacing between adjacent bands (Hz).
    sideband_power : float or sequence of float, optional
        Power of each off-resonance band relative to the on-resonance band.
    phases : {'quadratic'} or sequence of float, optional
        Per-band phase (rad). ``'quadratic'`` is the MATLAB reference schedule;
        ``None`` leaves every band in phase.

    Returns
    -------
    rf : RfEvent
        The modulated pulse.
    offsets : numpy.ndarray
        Band offsets applied (Hz).
    weights : numpy.ndarray
        Complex weight applied to each band.

    Examples
    --------
    >>> import numpy as np
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    >>> base = pp.make_sinc_pulse(flip_angle=np.deg2rad(30), duration=2e-3, system=system)
    >>> rf, offsets, weights = pp.make_sms_pulse(base, 3, 1000.0)
    >>> offsets.tolist()
    [-1000.0, 0.0, 1000.0]

    See Also
    --------
    sim_rf : check where the bands actually land.
    """
    offsets = _band_offsets(num_bands, band_offset)
    weights = _band_weights(len(offsets), sideband_power, phases)

    source = _events.as_namespace(rf)
    time = np.asarray(source.t, dtype=float)
    modulation = np.sum(
        weights[:, None] * np.exp(2j * np.pi * offsets[:, None] * time[None, :]), axis=0
    )

    banded = _events.convert(source)
    banded.signal = np.asarray(source.signal) * modulation
    return banded, offsets, weights


def _band_offsets(num_bands: int, band_offset: float) -> np.ndarray:
    num_bands = int(num_bands)
    if num_bands < 1:
        raise ValueError("num_bands must be >= 1")
    if band_offset <= 0 and num_bands > 1:
        raise ValueError("band_offset must be > 0 when num_bands > 1")
    # An even count cannot be both symmetric and include zero; keep the nearest
    # symmetric pairs and put the unmatched band on the positive side.
    indices = np.arange(-((num_bands - 1) // 2), num_bands // 2 + 1)
    return indices.astype(float) * float(band_offset)


def _band_weights(num_bands: int, sideband_power, phases) -> np.ndarray:
    if np.isscalar(sideband_power):
        powers = np.full(num_bands, float(sideband_power))
        powers[(num_bands - 1) // 2] = 1.0
    else:
        powers = np.asarray(sideband_power, dtype=float)
        if powers.shape != (num_bands,):
            raise ValueError("sideband_power must be a scalar or one value per band")
    if np.any(~np.isfinite(powers)) or np.any(powers < 0):
        raise ValueError("band powers must be finite and >= 0")
    if phases is None:
        phase = np.zeros(num_bands)
    elif isinstance(phases, str) and phases == "quadratic":
        position = np.arange(num_bands) - (num_bands - 1) / 2.0
        phase = (3.4 / num_bands) * position**2
    elif isinstance(phases, str):
        raise ValueError("phases must be 'quadratic' or one value per band")
    else:
        phase = np.asarray(phases, dtype=float)
        if phase.shape != (num_bands,):
            raise ValueError("phases must be 'quadratic' or one value per band")
    return np.sqrt(powers) * np.exp(1j * phase)


def make_spsp_pulse(
    flip_angle: float,
    slice_thickness: float,
    spectral_bandwidth: float,
    *,
    freq_offset: float = 0.0,
    spatial_time_bandwidth_product: float = 4.0,
    spectral_time_bandwidth_product: float = 3.0,
    n_subpulses: int = 10,
    system=None,
    use: str = "excitation",
):
    """A spectral-spatial pulse on an alternating slice gradient.

    Selects a slice *and* a spectral band at once: a train of short
    slice-selective subpulses rides an alternating gradient, and the subpulse
    envelope, sampled at the subpulse repetition rate, sets the spectral
    profile. Water-selective excitation therefore costs no separate
    fat-saturation module and no extra TR time.

    Both envelopes are SLR designs. ``n_subpulses`` is rounded up to an even
    count so the alternating gradient ends balanced.

    Parameters
    ----------
    flip_angle : float
        Nominal flip angle (rad).
    slice_thickness : float
        Slice thickness (m).
    spectral_bandwidth : float
        Spectral passband (Hz).
    freq_offset : float, optional
        Centre of the spectral passband (Hz).
    spatial_time_bandwidth_product : float, optional
        Time-bandwidth product of each spatial subpulse.
    spectral_time_bandwidth_product : float, optional
        Time-bandwidth product of the spectral envelope; sets the total
        duration as ``spectral_time_bandwidth_product / spectral_bandwidth``.
    n_subpulses : int, optional
        Number of subpulses (>= 4, rounded up to even).
    system : pypulseq.Opts, optional
        System limits.
    use : str, optional
        Pulseq ``use`` tag.

    Returns
    -------
    rf : RfEvent
        The spectral-spatial pulse.
    gz : GradEvent
        The alternating selection gradient.
    gz_reph : TrapEvent
        Its rephaser.

    Raises
    ------
    ValueError
        If the requested selectivity exceeds the gradient or slew limit, or the
        spectral bandwidth is too wide for the subpulse count.

    Examples
    --------
    >>> import numpy as np
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=180, slew_unit="T/m/s")
    >>> rf, gz, gz_reph = pp.make_spsp_pulse(
    ...     np.deg2rad(30), slice_thickness=10e-3,
    ...     spectral_bandwidth=300.0, n_subpulses=12, system=system,
    ... )
    >>> rf.type, gz.channel
    ('rf', 'z')

    See Also
    --------
    make_slr_pulse : the design underneath both of its envelopes.
    sim_rf : check the spectral profile the design produces.
    """
    system = default_system(system)
    if slice_thickness <= 0 or spectral_bandwidth <= 0:
        raise ValueError("slice_thickness and spectral_bandwidth must be > 0")
    if n_subpulses < 4:
        raise ValueError("n_subpulses must be >= 4")
    if n_subpulses % 2:
        n_subpulses += 1

    grad_raster = system.grad_raster_time
    rf_raster = system.rf_raster_time
    total_duration = spectral_time_bandwidth_product / spectral_bandwidth
    lobe_duration = round(total_duration / (n_subpulses * grad_raster)) * grad_raster
    if lobe_duration < 8 * grad_raster:
        raise ValueError("spectral bandwidth is too large for the requested subpulse count")

    # Reserve a fifth of each lobe for its two ramps, on the gradient raster.
    ramp_time = max(grad_raster, round(0.1 * lobe_duration / grad_raster) * grad_raster)
    flat_time = lobe_duration - 2.0 * ramp_time
    if flat_time < 8 * rf_raster:
        raise ValueError("SPSP sublobes leave fewer than 8 RF samples")
    amplitude_hz_per_m = spatial_time_bandwidth_product / (flat_time * slice_thickness)
    if amplitude_hz_per_m > system.max_grad:
        raise ValueError("SPSP slice-selection amplitude exceeds system.max_grad")
    if amplitude_hz_per_m / ramp_time > system.max_slew * (1.0 + 1e-9):
        raise ValueError("SPSP slice-selection ramps exceed system.max_slew")

    # Design the spatial SLR pulse on the flat top, build one trapezoid, then
    # VERSE the RF onto its ramps: the time-bandwidth product survives and the
    # RF overlaps the gradient for the whole lobe.
    spatial_rf = make_slr_pulse(
        1.0,
        duration=flat_time,
        dwell=rf_raster,
        time_bw_product=spatial_time_bandwidth_product,
        pulse_type="st",
        system=system,
        use=use,
    )
    positive_lobe = _events.make_trapezoid(
        channel="z",
        amplitude=amplitude_hz_per_m,
        flat_time=flat_time,
        rise_time=ramp_time,
        fall_time=ramp_time,
        system=system,
    )
    spatial = _verse_to_trapezoid(spatial_rf.signal, positive_lobe, rf_raster)
    spectral = design_slr(
        n_subpulses, spectral_time_bandwidth_product, pulse_type="st", filter_type="ls"
    )

    shape = np.empty(n_subpulses * spatial.size, dtype=np.complex128)
    for index in range(n_subpulses):
        start = index * spatial.size
        subpulse = spatial if index % 2 == 0 else spatial[::-1]
        shape[start : start + spatial.size] = spectral[index] * subpulse

    rf = _events.make_arbitrary_rf(
        signal=shape,
        flip_angle=flip_angle,
        dwell=rf_raster,
        freq_offset=freq_offset,
        system=system,
        use=use,
    )
    gz = _alternating_extended_trapezoid(positive_lobe, n_subpulses, system)
    # The alternating full-lobe areas cancel over an even count; what is left to
    # rephase is the half-lobe up to the centre of the final subpulse.
    gz_reph = _events.make_trapezoid(
        channel="z", area=0.5 * positive_lobe.area, system=system
    )
    return rf, gz, gz_reph


def _sample_trapezoid(event, dwell: float) -> np.ndarray:
    """Sample a trapezoid at RF-raster midpoints."""
    duration = event.rise_time + event.flat_time + event.fall_time
    time = (np.arange(round(duration / dwell)) + 0.5) * dwell
    knots = np.array([0.0, event.rise_time, event.rise_time + event.flat_time, duration])
    values = np.array([0.0, event.amplitude, event.amplitude, 0.0])
    return np.interp(time, knots, values)


def _verse_to_trapezoid(signal: np.ndarray, event, dwell: float) -> np.ndarray:
    """VERSE an RF envelope designed on a flat gradient onto a trapezoid."""
    gradient = _sample_trapezoid(event, dwell)
    magnitude = np.abs(gradient)
    total = magnitude.sum()
    if total <= 0:
        raise ValueError("VERSE gradient has zero area")
    excitation_position = (np.cumsum(magnitude) - 0.5 * magnitude) / total
    source_position = (np.arange(len(signal)) + 0.5) / len(signal)
    source = np.asarray(signal, dtype=complex)
    interpolated = np.interp(excitation_position, source_position, source.real) + 1j * np.interp(
        excitation_position, source_position, source.imag
    )
    return interpolated * magnitude / np.max(magnitude)


def _alternating_extended_trapezoid(event, count: int, system):
    """Concatenate alternating trapezoids without rasterising their ramps."""
    lobe_duration = event.rise_time + event.flat_time + event.fall_time
    times: list[float] = []
    amplitudes: list[float] = []
    for index in range(count):
        start = index * lobe_duration
        sign = 1.0 if index % 2 == 0 else -1.0
        knots = (
            start,
            start + event.rise_time,
            start + event.rise_time + event.flat_time,
            start + lobe_duration,
        )
        values = (0.0, sign * event.amplitude, sign * event.amplitude, 0.0)
        for knot, value in zip(knots, values, strict=True):
            if times and np.isclose(knot, times[-1], atol=1e-15):
                amplitudes[-1] = value
            else:
                times.append(knot)
                amplitudes.append(value)
    return _events.make_extended_trapezoid(
        channel="z", times=np.asarray(times), amplitudes=np.asarray(amplitudes), system=system
    )
