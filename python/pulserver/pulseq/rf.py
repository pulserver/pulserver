"""
Pulseq rf pulse design functions.

This module contains low-level helper functions to create rf pulses,
extending PyPulseq toolbox.

"""

__all__ = [
    'make_slr_pulse',
    'make_sms_pulse',
    'make_spsp_pulse',
]

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pypulseq as pp
from numpy.typing import NDArray
from pypulseq import Opts
from sigpy import fft, resize
from sigpy.mri.rf import multiband, slr


def make_slr_pulse(
    flip_angle: float,
    duration: float = 4e-3,
    time_bw_product: float | None = None,
    bandwidth: float | None = None,
    return_gz: bool = False,
    slice_thickness: float | None = None,
    system: Opts | None = None,
    use: str = 'undefined',
    delay: float = 0.0,
    freq_offset: float = 0.0,
    phase_offset: float = 0.0,
    freq_ppm: float = 0.0,
    phase_ppm: float = 0.0,
    max_grad: float | None = None,
    max_slew: float | None = None,
    filter_type: str = 'ls',
    passband_ripple_lvl: float = 0.01,
    stopband_ripple_lvl: float = 0.01,
    cancel_alpha_phs: bool = False,
    absorb_rf_deadtime: bool = False,
) -> SimpleNamespace | tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace]:
    r"""
    Creates a radio-frequency pulse event and optionally accompanying slice select and slice select rephasing
    trapezoidal gradient events, using Shinnar-LeRoux design.

    Parameters
    ----------
    flip_angle : float
        Flip angle in radians.
    duration : float, default=4e-3
        Duration in seconds (s).
    time_bw_product : float | None, default=None
        Time-bandwidth product.
    bandwidth : float | None, default=None
        Bandwidth in Hertz (Hz).
    return_gz : bool, default=False
        Boolean flag to indicate if the slice-selective gradient has to be returned.
    slice_thickness : float | None, default=None
        Slice thickness (m) of accompanying slice select trapezoidal event.
        The slice thickness determines the area of the slice select event.
    system : Opts, default=Opts()
        System limits.
    use : str, default='undefined'
        Use of radio-frequency Gauss pulse event.
        Must be one of 'excitation', 'refocusing', 'inversion',
        'saturation', 'preparation', 'other', 'undefined'.
    delay : float, default=0.0
        Delay in seconds (s).
    freq_offset : float, default=0.0
        Frequency offset in Hertz (Hz).
    phase_offset : float, default=0.0
        Phase offset in radians.
    freq_ppm : float, default=0.0
        PPM frequency offset.
    phase_ppm : float, default=0.0
        PPM phase offset.
    max_grad : float | None, default=None
        Maximum gradient strength of accompanying slice select trapezoidal event.
    max_slew : float | None, default=None
        Maximum slew rate of accompanying slice select trapezoidal event.
    filter_type : str, default="ls"
        Type of filter to use: ``"ms"`` (sinc),
        ``"pm``, (Parks-McClellan equal-ripple),
        ``"min"`` (minphase using factored pm),
        ``"max"`` (maxphase using factored pm), ``"ls"`` (least squares).
    passband_ripple_lvl : float, default=0.01
        Passband ripple level in :math:'M_0^{-1}'.
    stopband_ripple_lvl : float, default=0.01
        Stopband ripple level in :math:'M_0^{-1}'.
    cancel_alpha_phs : bool, default=False
        For 'ex' pulses, absorb the alpha phase
        profile from beta's profile, so they cancel for a flatter
        total phase.
    absorb_rf_deadtime : bool, default=False
        If True, and 'return_gz' is True, extend gz flat time to include
        rf_dead_time.

    Returns
    -------
    rf : SimpleNamespace
        Radio-frequency pulse event with SLR designed pulse shape.
    gz : SimpleNamespace, optional
        Slice select trapezoidal gradient event accompanying the SLR radio-frequency pulse event.
    gzr : SimpleNamespace, optional
        Accompanying slice select rephasing trapezoidal gradient event.

    Raises
    ------
    ValueError
        If invalid `use` is passed.
        If `return_gz=True` and `slice_thickness` was not passed.

    Reference
    ---------
    Pauly, J., Le Roux, Patrick., Nishimura, D., and Macovski, A.(1991).
    Parameter Relations for the Shinnar-LeRoux Selective Excitation
    Pulse Design Algorithm.
    IEEE Transactions on Medical Imaging, Vol 10, No 1, 53-65.

    """
    if system is None:
        system = Opts.default
    if max_grad is None:
        max_grad = system.max_grad
    if max_slew is None:
        max_slew = system.max_slew
    if return_gz and slice_thickness is None:
        raise ValueError('User must provide slice thickness for slice-selective pulses')
    if bandwidth is None and time_bw_product is None:
        raise ValueError('User must provide bandwidth or time-bandwidth product.')
    if bandwidth is not None and time_bw_product is not None:
        raise ValueError(
            'User must provide either bandwidth or time-bandwidth product, not both.'
        )
    if bandwidth is None:
        bandwidth = time_bw_product / duration
    else:
        time_bw_product = bandwidth * duration

    # Compute number of samples
    n_samples = round(duration / system.rf_raster_time)

    # SigPy design routine
    signal = _dzrf(
        n=n_samples,
        tb=time_bw_product,
        ptype=_use2ptype(use, flip_angle),
        ftype=filter_type,
        d1=passband_ripple_lvl,
        d2=stopband_ripple_lvl,
        cancel_alpha_phs=cancel_alpha_phs,
    )

    # Wrap as Pulseq event
    items = pp.make_arbitrary_rf(
        signal,
        flip_angle,
        bandwidth,
        delay,
        system.rf_raster_time,
        freq_offset,
        True,  # signal_scaling to target flip
        max_grad,
        max_slew,
        phase_offset,
        return_gz,  # return_gz
        slice_thickness,  # slice_thickness
        system,
        time_bw_product,
        use,
        freq_ppm,
        phase_ppm,
    )

    # Create slice rewinder
    if return_gz:
        rf, gz = items

        # Recover center position
        center_pos = rf.center / rf.shape_dur

        # Calculate plateau area
        flat_area = gz.amplitude * gz.flat_time

        # Calculate area offset if gz is extended
        if absorb_rf_deadtime:
            area_offset = gz.amplitude * (system.rf_dead_time + system.rf_ringdown_time)
        else:
            area_offset = 0.0
        total_area = flat_area * (1 - center_pos) + (gz.area - flat_area) + area_offset

        # Extend gz
        if absorb_rf_deadtime:
            gz = pp.make_trapezoid(
                channel='z',
                system=system,
                rise_time=gz.rise_time,
                flat_time=gz.flat_time + system.rf_dead_time + system.rf_ringdown_time,
                fall_time=gz.fall_time,
                amplitude=gz.amplitude,
            )

        # Compensate area between rf peak and end of pulse
        gzr = pp.make_trapezoid(
            channel='z',
            system=system,
            area=-total_area,
        )

        return rf, gz, gzr

    return items  # = rf


def make_sms_pulse(
    flip_angle: float,
    n_slices: int,
    slice_thickness: float,
    duration: float = 4e-3,
    time_bw_product: float | None = None,
    bandwidth: float | None = None,
    slice_separation: float | None = None,
    system: Opts | None = None,
    use: str = 'undefined',
    delay: float = 0.0,
    freq_offset: float = 0.0,
    phase_offset: float = 0.0,
    freq_ppm: float = 0.0,
    phase_ppm: float = 0.0,
    max_grad: float | None = None,
    max_slew: float | None = None,
    filter_type: str = 'ls',
    passband_ripple_lvl: float = 0.01,
    stopband_ripple_lvl: float = 0.01,
    cancel_alpha_phs: bool = False,
    reference_phase: str = 'None',
    absorb_rf_deadtime: bool = False,
) -> SimpleNamespace | tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace]:
    r"""
    Creates a multislice radio-frequency pulse event and
    optionally accompanying slice select and slice select rephasing trapezoidal gradient events,
    using Shinnar-LeRoux design.

    Parameters
    ----------
    flip_angle : float
        Flip angle in radians.
    n_slices: int
        Number of simultanelusly excited slices.
    slice_thickness : float
        Slice thickness of accompanying slice select trapezoidal event.
        The slice thickness determines the area of the slice select event.
    duration : float, default=4e-3
        Duration in seconds (s).
    time_bw_product : float | None, default=None
        Time-bandwidth product.
    bandwidth : float | None, default=None
        Bandwidth in Hertz (Hz).
    slice_separation : float
        Distance between slices.
    system : Opts, default=Opts()
        System limits.
    use : str, default='undefined'
        Use of radio-frequency Gauss pulse event.
        Must be one of 'excitation', 'refocusing', 'inversion',
        'saturation', 'preparation', 'other', 'undefined'.
    delay : float, default=0.0
        Delay in seconds (s).
    freq_offset : float, default=0.0
        Frequency offset in Hertz (Hz).
    phase_offset : float, default=0.0
        Phase offset in radians.
    freq_ppm : float, default=0.0
        PPM frequency offset.
    phase_ppm : float, default=0.0
        PPM phase offset.
    max_grad : float | None, default=None
        Maximum gradient strength of accompanying slice select trapezoidal event.
    max_slew : float | None, default=None
        Maximum slew rate of accompanying slice select trapezoidal event.

    filter_type : str, default="ls"
        Type of filter to use: ``"ms"`` (sinc),
        ``"pm``, (Parks-McClellan equal-ripple),
        ``"min"`` (minphase using factored pm),
        ``"max"`` (maxphase using factored pm), ``"ls"`` (least squares).
    passband_ripple_lvl : float, default=0.01
        Passband ripple level in :math:'M_0^{-1}'.
    stopband_ripple_lvl : float, default=0.01
        Stopband ripple level in :math:'M_0^{-1}'.
    cancel_alpha_phs : bool, default=False
        For 'ex' pulses, absorb the alpha phase
        profile from beta's profile, so they cancel for a flatter
        total phase.
    reference_phase : str, default="None"
        Phase 0 point to use. Can be 'phs_mod' (Wong),
        'amp_mod' (Malik), 'quad_mod' (Grissom), or 'None'.
    absorb_rf_deadtime : bool, default=False
        If True, and 'return_gz' is True, extend gz flat time to include
        rf_dead_time.

    Returns
    -------
    rf : SimpleNamespace
        Radio-frequency pulse event with SLR designed pulse shape.
    gz : SimpleNamespace, optional
        Slice select trapezoidal gradient event accompanying the SLR radio-frequency pulse event.
    gzr : SimpleNamespace, optional
        Accompanying slice select rephasing trapezoidal gradient event.

    Raises
    ------
    ValueError
        If invalid `use` is passed.

    Reference
    ---------
    Pauly, J., Le Roux, Patrick., Nishimura, D., and Macovski, A.(1991).
    Parameter Relations for the Shinnar-LeRoux Selective Excitation
    Pulse Design Algorithm.
    IEEE Transactions on Medical Imaging, Vol 10, No 1, 53-65.

    """
    if slice_separation is None:
        slice_separation = n_slices * slice_thickness
    if system is None:
        system = Opts.default
    if max_grad is None:
        max_grad = system.max_grad
    if max_slew is None:
        max_slew = system.max_slew
    if bandwidth is None and time_bw_product is None:
        raise ValueError('User must provide bandwidth or time-bandwidth product.')
    if bandwidth is not None and time_bw_product is not None:
        raise ValueError(
            'User must provide either bandwidth or time-bandwidth product, not both.'
        )
    if bandwidth is None:
        bandwidth = time_bw_product / duration
    else:
        time_bw_product = bandwidth * duration

    # Compute number of samples
    n_samples = round(duration / system.rf_raster_time)

    # SigPy design routine
    signal = _dzrf(
        n=n_samples,
        tb=time_bw_product,
        ptype=_use2ptype(use, flip_angle),
        ftype=filter_type,
        d1=passband_ripple_lvl,
        d2=stopband_ripple_lvl,
        cancel_alpha_phs=cancel_alpha_phs,
    )

    # Multiband
    band_sep = time_bw_product * slice_separation / slice_thickness
    signal = _mb_rf(
        signal, n_bands=n_slices, band_sep=band_sep, phs_0_pt=reference_phase
    )

    # Wrap as Pulseq event
    rf, gz = pp.make_arbitrary_rf(
        signal,
        flip_angle,
        bandwidth,
        delay,
        system.rf_raster_time,
        freq_offset,
        True,  # signal_scaling to target flip
        max_grad,
        max_slew,
        phase_offset,
        True,  # return_gz
        slice_thickness,  # slice_thickness
        system,
        time_bw_product,
        use,
        freq_ppm,
        phase_ppm,
    )

    # Recover center position
    center_pos = rf.center / rf.shape_dur

    # Calculate plateau area
    flat_area = gz.amplitude * gz.flat_time

    # Calculate area offset if gz is extended
    if absorb_rf_deadtime:
        area_offset = gz.amplitude * (system.rf_dead_time + system.rf_ringdown_time)
    else:
        area_offset = 0.0
    total_area = flat_area * (1 - center_pos) + (gz.area - flat_area) + area_offset

    # Extend gz
    if absorb_rf_deadtime:
        gz = pp.make_trapezoid(
            channel='z',
            system=system,
            rise_time=gz.rise_time,
            flat_time=gz.flat_time + system.rf_dead_time + system.rf_ringdown_time,
            fall_time=gz.fall_time,
            amplitude=gz.amplitude,
        )

    # Compensate area between rf peak and end of pulse
    gzr = pp.make_trapezoid(
        channel='z',
        system=system,
        area=-total_area,
    )

    return rf, gz, gzr


def make_spsp_pulse(
    flip_angle: float,
    slice_thickness: float,
    freq_bandwidth: float,
    duration: float = 20e-3,
    spat_time_bw_product: float | None = None,
    spat_bandwidth: float | None = None,
    system: Opts | None = None,
    use: str = 'undefined',
    delay: float = 0.0,
    freq_offset: float = 0.0,
    phase_offset: float = 0.0,
    freq_ppm: float = 0.0,
    phase_ppm: float = 0.0,
    max_grad: float | None = None,
    max_slew: float | None = None,
    spat_filter_type: str = 'ls',
    freq_filter_type: str = 'pm',
    spat_passband_ripple_lvl: float = 0.01,
    spat_stopband_ripple_lvl: float = 0.01,
    freq_passband_ripple_lvl: float = 0.01,
    freq_stopband_ripple_lvl: float = 0.01,
    n_lobes: int = 14,
    flyback: bool = False,
) -> SimpleNamespace | tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace]:
    r"""
    Creates a spectral-spatial radio-frequency pulse event and
    accompanying slice select and slice select rephasing trapezoidal gradient events,
    using Shinnar-LeRoux design.

    Parameters
    ----------
    flip_angle : float
        Flip angle in radians.
    slice_thickness : float
        Slice thickness (m) of accompanying slice select trapezoidal event.
        The slice thickness determines the area of the slice select event.
    freq_bandwidth : float
        Spectral profile frequency bandwidth in Hertz (Hz).
    duration : float, default=4e-3
        Duration in seconds (s).
    spat_time_bw_product : float | None, default=None
        Spatial profile time-bandwidth product.
    spat_bandwidth : float
        Spatial profile frequency bandwidth in Hertz (Hz).
    system : Opts, default=Opts()
        System limits.
    use : str, default='undefined'
        Use of radio-frequency Gauss pulse event.
        Must be one of 'excitation', 'refocusing', 'inversion',
        'saturation', 'preparation', 'other', 'undefined'.
    delay : float, default=0.0
        Delay in seconds (s).
    freq_offset : float, default=0.0
        Frequency offset in Hertz (Hz).
    phase_offset : float, default=0.0
        Phase offset in radians.
    freq_ppm : float, default=0.0
        PPM frequency offset.
    phase_ppm : float, default=0.0
        PPM phase offset.
    max_grad : float | None, default=None
        Maximum gradient strength of accompanying slice select trapezoidal event.
    max_slew : float | None, default=None
        Maximum slew rate of accompanying slice select trapezoidal event.
    spat_filter_type : str, default="ls"
        Type of filter to use for spatial profile: ``"ms"`` (sinc),
        ``"pm``, (Parks-McClellan equal-ripple),
        ``"min"`` (minphase using factored pm),
        ``"max"`` (maxphase using factored pm), ``"ls"`` (least squares).
    freq_filter_type : str, default="pm"
        Type of filter to use for spectral profile: ``"ms"`` (sinc),
        ``"pm``, (Parks-McClellan equal-ripple),
        ``"min"`` (minphase using factored pm),
        ``"max"`` (maxphase using factored pm), ``"ls"`` (least squares).
    spat_passband_ripple_lvl : float, default=0.01
        Spatial profile passband ripple level in :math:'M_0^{-1}'.
    spat_stopband_ripple_lvl : float, default=0.01
        Spatial profile stopband ripple level in :math:'M_0^{-1}'.
    freq_passband_ripple_lvl : float, default=0.01
        Spectral profile passband ripple level in :math:'M_0^{-1}'.
    freq_stopband_ripple_lvl : float, default=0.01
        Spectral profile stopband ripple level in :math:'M_0^{-1}'.
    n_lobes : int, default=14
        Number of sub-lobes within the given duration.
    flyback : bool, default=False
        If ``True``, use flyback EPI slice selection grad instead of bipolar.

    Returns
    -------
    rf : SimpleNamespace
        Radio-frequency pulse event with SLR designed pulse shape.
    gz : SimpleNamespace, optional
        Slice select EPI gradient event accompanying the SLR radio-frequency pulse event.
    gzr : SimpleNamespace, optional
        Accompanying slice select rephasing trapezoidal gradient event.

    Raises
    ------
    ValueError
        If invalid `use` is passed.

    Reference
    ---------
    Pauly, J., Le Roux, Patrick., Nishimura, D., and Macovski, A.(1991).
    Parameter Relations for the Shinnar-LeRoux Selective Excitation
    Pulse Design Algorithm.
    IEEE Transactions on Medical Imaging, Vol 10, No 1, 53-65.

    """
    if system is None:
        system = Opts.default
    if max_grad is None:
        max_grad = system.max_grad
    if max_slew is None:
        max_slew = system.max_slew
    if spat_bandwidth is None and spat_time_bw_product is None:
        raise ValueError(
            'User must provide either spatial bandwidth or spatial time-bandwidth product.'
        )
    if spat_bandwidth is not None and spat_time_bw_product is not None:
        raise ValueError(
            'User must provide either spatial bandwidth or time-bandwidth product, not both.'
        )

    # Make sure lobes are even
    n_lobes = int(np.ceil(n_lobes / 2) * 2)

    # Make sure duration is even multiple both of rf and grad raster
    raster = (
        np.gcd(
            int(system.rf_raster_time * 2e6),
            int(system.grad_raster_time * 2e6),
        )
        * 1e-6
    )

    # Re-define duration as single-lobe duration
    # duration = np.round(duration / n_lobes / raster) * raster
    duration = duration / n_lobes
    duration = _round_duration(duration, system.rf_raster_time, system.grad_raster_time)

    if spat_bandwidth is None:
        spat_bandwidth = spat_time_bw_product / (n_lobes * duration)
    else:
        spat_time_bw_product = spat_bandwidth * (n_lobes * duration)

    # Compute number of samples
    n_samples = int(duration / system.rf_raster_time)

    # Generate frequency profile for base SLR 1D spatial selective pulse
    frequency_profile = _dzbeta(
        n=n_samples,
        tb=spat_time_bw_product,
        ptype=_use2ptype(use, flip_angle),
        ftype=spat_filter_type,
        d1=spat_passband_ripple_lvl,
        d2=spat_stopband_ripple_lvl,
    )
    frequency_profile = _padded_centered_fft(
        frequency_profile, 2 * n_samples, n_samples
    )

    # Generate frequency weighting (envelope) to achieve spectral selectivity
    frequency_weighting = _dzbeta(
        n=n_lobes,
        tb=(n_lobes - 1) * duration * freq_bandwidth,
        ptype=_use2ptype(use, flip_angle),
        ftype=freq_filter_type,
        d1=freq_passband_ripple_lvl,
        d2=freq_stopband_ripple_lvl,
    )

    # Generate weighted RF frequency profile
    weighted_profile = np.outer(frequency_profile, frequency_weighting) * np.sin(
        flip_angle / 2.0
    )

    # Generate hybrid time-frequency RF envelope
    hybrid_prof = np.stack([_b2rf(prof) for prof in weighted_profile], axis=-1)

    # Generate raw (non-versed) RF time envelope
    raw_signal = np.stack(
        [
            _padded_centered_fft(np.sin(0.5 * prof), 2 * n_samples, n_samples)
            for prof in hybrid_prof
        ],
        axis=-1,
    )

    # Generate single lobe grad
    amplitude = spat_bandwidth / slice_thickness
    area = amplitude * duration
    g_lobe = pp.make_trapezoid(
        channel='z',
        system=system,
        flat_time=duration,
        flat_area=area,
    )
    g_lobe_dur = pp.calc_duration(g_lobe)
    g_lobe_pad = raster * np.round(g_lobe_dur / raster)

    # Convert single lobe to extended trapezoid
    amplitudes = np.asarray([0.0, g_lobe.amplitude, g_lobe.amplitude, 0.0])
    times = np.cumsum([0.0, g_lobe.rise_time, g_lobe.flat_time, g_lobe.fall_time])
    if abs(g_lobe_pad - g_lobe_dur) > pp.eps:
        amplitudes = np.concatenate((amplitudes, (0.0,)))
        times = np.concatenate((times, (g_lobe_pad,)))
    g_lobe0 = pp.make_extended_trapezoid(
        channel='z',
        system=system,
        amplitudes=amplitudes,
        times=times,
    )
    g_lobe = deepcopy(g_lobe0)

    # Apply verse
    waveform = pp.points_to_waveform(
        g_lobe.waveform,
        system.rf_raster_time,
        g_lobe.tt,
    )
    signal = _versec(waveform, raw_signal)
    signal = np.ascontiguousarray(signal.T)  # (nlobes, nsamples)

    # Build flyback gradient if required
    if flyback:
        sign = 1
        g_rew, _, _ = pp.make_extended_trapezoid_area(
            channel='z',
            area=-g_lobe.area,
            grad_start=0.0,
            grad_end=0.0,
            system=system,
        )

        # Make sure g_rew duration aligns with both rf and grad rasters
        g_rew_dur = pp.calc_duration(g_rew)
        g_rew_pad = _round_duration(
            g_rew_dur, system.rf_raster_time, system.grad_raster_time
        )
        if abs(g_rew_pad - g_rew_dur) > pp.eps:
            g_rew.amplitudes = np.concatenate((g_rew.amplitudes, (0.0,)))
            g_rew.times = np.concatenate((g_rew.times, (g_rew_pad,)))
            g_rew.shape_dur = g_rew_pad

        # Get rf padding
        rf_pad = int(g_rew_pad / system.rf_raster_time)

        # Merge flyback with slice selection lobe
        g_rew.delay = pp.calc_duration(g_lobe)
        g_lobe = pp.add_gradients([g_lobe, g_rew], system=system)
    else:
        rf_pad = 0
        sign = -1

    # Pad RF signal
    if rf_pad:
        signal = np.pad(signal, ((0, 0), (0, rf_pad)))

    # Wrap as RF Pulseq event
    rf = pp.make_arbitrary_rf(
        signal.ravel(),
        1.0,  # flip angle, not used
        0.0,  # bandwidth, not used
        delay,
        system.rf_raster_time,
        freq_offset,
        False,  # signal_scaling to target flip
        0.0,  # max_grad, not used
        0.0,  # max_slew, not used
        phase_offset,
        False,  # return_gz
        0.0,  # slice_thickness
        system,
        0.0,  # time-bandwidth product, not used
        use,
        freq_ppm,
        phase_ppm,
    )

    # Build full grad
    gz = deepcopy(g_lobe)
    for n in range(1, n_lobes):
        _g_lobe = pp.scale_grad(g_lobe, scale=(sign) ** n)
        _g_lobe.delay = pp.calc_duration(gz)
        gz = pp.add_gradients([gz, _g_lobe], system=system)

    # Build slice rewinder
    t_center = _calc_signal_center(
        signal[-1],
        system.rf_raster_time * np.arange(signal[-1].size),
    )
    center_pos = t_center / (system.rf_raster_time * signal[-1].size)

    # Compute area
    area = (sign) ** (n_lobes - 1) * g_lobe0.area * (1 - center_pos)
    if flyback:
        area += g_rew.area

    # Compensate area between rf peak and end of pulse
    gzr = pp.make_trapezoid(
        channel='z',
        system=system,
        area=-area,
    )

    return rf, gz, gzr


# %% Internal helpers
def _use2ptype(use: str, flip_angle: float) -> str:
    """Convert PyPulseq use code to SigPy Shinnar-LeRoux pulse type."""
    small_tip = np.rad2deg(flip_angle) <= 45.0
    if use == 'refocusing':
        return 'se'
    if use == 'preparation' or use == 'inversion':
        return 'inv'
    # excitation, other and undefined
    if small_tip:
        return 'st'
    return 'ex'


_dzrf = slr.dzrf


_mb_rf = multiband.mb_rf


_b2rf = slr.b2rf


def _round_duration(T_s: float, r1_s: float, r2_s: float) -> float:
    """
    T_s  : float or array, duration in seconds
    r1_s : float, raster 1 in seconds
    r2_s : float, raster 2 in seconds
    """
    T_s = np.asarray(T_s)

    # ---- convert to integer microseconds ----
    scale = 1e6  # seconds → microseconds

    T_i = np.rint(T_s * scale).astype(np.int64)
    r1_i = round(r1_s * scale)
    r2_i = round(r2_s * scale)

    # ---- integer math ----
    L = r1_i * r2_i // np.gcd(r1_i, r2_i)

    m = L // r1_i
    step = 1 if (m % 2 == 0) else 2

    n = np.rint(T_i / L / step).astype(np.int64) * step
    D_i = n * L

    # ---- back to seconds ----
    return D_i / scale


def _dzbeta(
    n: int,
    tb: float,
    ptype: str,
    ftype: str,
    d1: float,
    d2: float,
) -> NDArray[complex]:
    """Same as dzrf, but returning beta instead."""
    bsf, d1, d2 = slr.calc_ripples(ptype, d1, d2)

    if ftype == 'ms':  # sinc
        b = slr.msinc(n, tb / 4)
    elif ftype == 'pm':  # linphase
        b = slr.dzlp(n, tb, d1, d2)
    elif ftype == 'min':  # minphase
        b = slr.dzmp(n, tb, d1, d2)
        b = b[::-1]
    elif ftype == 'max':  # maxphase
        b = slr.dzmp(n, tb, d1, d2)
    elif ftype == 'ls':  # least squares
        b = slr.dzls(n, tb, d1, d2)
    else:
        raise Exception(f'Filter type ("{ftype}") is not recognized.')

    if ptype == 'st':
        return b

    return bsf * b


def _padded_centered_fft(
    signal: NDArray[complex], isize: int, osize: int
) -> NDArray[complex]:
    """Pad with zeros to target isize and perform centered osize fft of the result."""
    return resize(fft(signal, (isize,)), (osize,))


def _versec(g: NDArray[float], rf: NDArray[complex]) -> NDArray[complex]:
    """VERSE input rf pulse to fit target g duration."""
    m, n = rf.shape

    # Match MATLAB orientation logic
    if m < n:
        rf = np.conj(rf.T)
        m, n = rf.shape

    # Compute k
    k = np.cumsum(g)
    k = (m - 1) * k / np.max(k)

    # Normalize g
    g = m * g / np.sum(g)

    # Interpolation grid
    x = np.arange(m)

    # Vectorized interpolation for all columns
    # Result shape: (len(k), n)
    interp_vals = np.vstack([np.interp(k, x, rf[:, j]) for j in range(n)]).T

    # Apply weighting
    rfv = interp_vals * g[:, None]

    return rfv


def _calc_signal_center(signal: NDArray[complex], t: NDArray[float]) -> float:
    """Similar to pulseq.calc_rf_center, but operates directly on signal."""
    rf_max = np.max(np.abs(signal))
    i_peak = np.where(np.abs(signal) >= rf_max * 0.99999)[0]
    time_center = (t[i_peak[0]] + t[i_peak[-1]]) / 2

    return time_center
