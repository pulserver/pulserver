"""
Pulseq rf pulse design functions.

This module contains low-level helper functions to create rf pulses,
extending PyPulseq toolbox.

"""

__all__ = [
    "make_slr_pulse",
    "make_sms_pulse",
    "make_spsp_pulse",
]

from types import SimpleNamespace

import numpy as np
from numpy.typing import NDArray

from pypulseq import Opts
from pypulseq.calc_duration import calc_duration
from pypulseq.calc_rf_center import calc_rf_center
from pypulseq.sigpy_pulse_opts import SigpyPulseOpts
from pypulseq.make_sigpy_pulse import sigpy_n_seq
from pypulseq.make_arbitrary_rf import make_arbitrary_rf
from pypulseq.make_trapezoid import make_trapezoid

from sigpy import resize
from sigpy import fft
from sigpy.mri.rf import slr
from sigpy.mri.rf import multiband
    
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
    
    dwell: float | None = None,
    max_grad: float | None = None,
    max_slew: float | None = None,
    
    filter_type: str = 'ls',
    passband_ripple_lvl: float = 0.01,
    stopband_ripple_lvl: float = 0.01,
    cancel_alpha_phs: bool = False,
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
    time_bw_product : float, default=0.0
        Time-bandwidth product.
    bandwidth : float, default=0.0
        Bandwidth in Hertz (Hz).
    return_gz : bool, default=False
        Boolean flag to indicate if the slice-selective gradient has to be returned.
    slice_thickness : float, default=0.0
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
        
    dwell : float | None, default=None
        RF raster time.
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
    if dwell is None:
        dwell = system.rf_raster_time
    if max_grad is None:
        max_grad = system.max_grad
    if max_slew is None:
        max_slew = system.max_slew
    if return_gz and slice_thickness is None:
        raise ValueError(
            'User must provide slice thickness for slice-selective pulses'
        )
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
    n_samples = round(duration / dwell)

    # SigPy design routine  
    signal = _dzrf(
        n=n_samples,
        tb=time_bw_product,
        ptype=_use2ptype(use),
        ftype=filter_type,
        d1=passband_ripple_lvl,
        d2=stopband_ripple_lvl,
        cancel_alpha_phs=cancel_alpha_phs,
    )
    
    # Wrap in Pulseq event
    items = make_arbitrary_rf(
        signal, 
        flip_angle,
        bandwidth,
        delay,
        dwell,
        freq_offset,
        True, # signal_scaling to target flip
        max_grad,
        max_slew,
        phase_offset,
        return_gz, # return_gz
        slice_thickness, # slice_thickness
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
        
        # Compensate area between rf peak and end of pulse
        gzr = make_trapezoid(
            channel='z',
            system=system,
            area=-flat_area * (1 - center_pos) - 0.5 * (gz.area - flat_area),
        )
        
        return rf, gz, gzr

    return items # = rf


def make_sms_pulse(
    flip_angle: float,
    n_bands: int,
    duration: float = 4e-3,
    time_bw_product: float = 4.0,
    bandwidth: float = 0.0,
    slice_thickness: float = 0.0,
    slice_separation: float | None = None,
    system: Opts | None = None,
    use: str = 'undefined',
    
    delay: float = 0.0,
    freq_offset: float = 0.0,
    phase_offset: float = 0.0,
    freq_ppm: float = 0.0,
    phase_ppm: float = 0.0,
    
    dwell: float = 0.0,
    max_grad: float = 0.0,
    max_slew: float = 0.0,
    
    filter_type: str = 'ls',
    passband_ripple_lvl: float = 0.01,
    stopband_ripple_lvl: float = 0.01,
    cancel_alpha_phs: bool = False,
    reference_phase: str = 'None',
    
) -> SimpleNamespace | tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace]:
    r"""
    Creates a multislice radio-frequency pulse event and 
    optionally accompanying slice select and slice select rephasing trapezoidal gradient events, 
    using Shinnar-LeRoux design.

    Parameters
    ----------
    flip_angle : float
        Flip angle in radians.
    n_bands: int
        Number of bands.
    duration : float, default=4e-3
        Duration in seconds (s).
    time_bw_product : float, default=4.0
        Time-bandwidth product.
    bandwidth : float, default=0.0
        Bandwidth in Hertz (Hz).
    slice_thickness : float, default=0.0
        Slice thickness of accompanying slice select trapezoidal event. 
        The slice thickness determines the area of the slice select event.
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
        
    dwell : float, default=0.0
    max_grad : float, default=0.0
        Maximum gradient strength of accompanying slice select trapezoidal event.
    max_slew : float, default=0.0
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
    if slice_separation is None:
        slice_separation = slice_thickness
    if bandwidth == 0 and time_bw_product == 0:
        raise ValueError('User must provide bandwidth or time-bandwidth product.')
    if bandwidth != 0 and time_bw_product != 0:
        raise ValueError(
            'User must provide either bandwidth or time-bandwidth product, not both.'
        )
    if bandwidth == 0:
        bandwidth = time_bw_product / duration
    else:
        time_bw_product = bandwidth * duration
        
    # Find band separation in Hz
    band_sep = slice_separation / slice_thickness * time_bw_product

    # SigPy specific options
    pulse_cfg = SigpyPulseOpts(
        pulse_type='sms',
        ptype=_use2ptype(use),
        ftype=filter_type,
        d1=passband_ripple_lvl,
        d2=stopband_ripple_lvl,
        cancel_alpha_phs=cancel_alpha_phs,
        n_bands=n_bands,
        band_sep=band_sep,
        phs_0_pt=reference_phase,
    )
    
    # First pass design w/o slice selection
    _rf, _, _ = sigpy_n_seq(
        flip_angle,
        delay,
        duration,
        freq_offset,
        0.5, # temporary center pos
        max_grad,
        max_slew,
        phase_offset,
        False, # return_gz
        0.0, # slice thickness
        system,
        time_bw_product,
        pulse_cfg,
        use,
        False, # plot
        freq_ppm,
        phase_ppm,
    )
    
    # Compute actual center post
    time_center, _ =  calc_rf_center(_rf) 
    center_pos = time_center / calc_duration(_rf)
    
    # Second pass design with slice selection
    return sigpy_n_seq(
        flip_angle,
        delay,
        duration,
        freq_offset,
        center_pos,
        max_grad,
        max_slew,
        phase_offset,
        True, # return_gz
        slice_thickness,
        system,
        time_bw_product,
        pulse_cfg,
        use,
        False, # plot
        freq_ppm,
        phase_ppm,
    )
    
def make_spsp_pulse(
    flip_angle: float,
    slice_thickness: float,
    freq_bandwidth: float,
    spat_bandwidth: float = 0.0,
    delay: float = 0.0,
    dwell: float = 0.0,
    duration: float = 4e-3,
    freq_offset: float = 0.0,
    max_grad: float = 0.0,
    max_slew: float = 0.0,
    phase_offset: float = 0.0,
    system: Opts | None = None,
    spat_time_bw_product: float = 4.0,
    use: str = 'undefined',
    freq_ppm: float = 0.0,
    phase_ppm: float = 0.0,
    spat_filter_type: str = 'ls',
    freq_filter_type: str = 'ls',
    spat_passband_ripple_lvl: float = 0.01,
    spat_stopband_ripple_lvl: float = 0.01,
    freq_passband_ripple_lvl: float = 0.01,
    freq_stopband_ripple_lvl: float = 0.01,
    spat_cancel_alpha_phs: bool = False,
    freq_cancel_alpha_phs: bool = False,
    n_lobes: int = 8,
    flyback: bool = False,
) -> SimpleNamespace | tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace]:
    ...
    # pulse_type = _use2ptype(use, flip_angle)
    
    # # 1) Get actual ripple for target pulse type
    # _, spat_passband_ripple_lvl, spat_stopband_ripple_lvl = calc_ripples(
    #     pulse_type,
    #     spat_passband_ripple_lvl,
    #     spat_stopband_ripple_lvl,
    # )
    # _, freq_passband_ripple_lvl, freq_stopband_ripple_lvl = calc_ripples(
    #     pulse_type,
    #     freq_passband_ripple_lvl,
    #     freq_stopband_ripple_lvl,
    # )
        
    # # 2) Generate SLR pulse and envelope
    # rf_lobe, gz_lobe, gz_rew = make_slr_pulse(
    #     1.0, # normalize to 1
    #     spat_bandwidth,
    #     0.0, # delay
    #     dwell,
    #     duration / n_lobes,
    #     0.0, # frequency offset
    #     max_grad,
    #     max_slew,
    #     0.0, # phase offset
    #     True, # return_gz
    #     slice_thickness,
    #     system,
    #     spat_time_bw_product,
    #     'undefined', # use
    #     0.0, # freq_ppm
    #     0.0, # phase_ppm
    #     spat_filter_type,
    #     spat_passband_ripple_lvl,
    #     spat_stopband_ripple_lvl,
    #     spat_cancel_alpha_phs,
    # )
    # rf_spect_envelope = _make_spectral_weighting(
    #     n_lobes,
    #     freq_bandwidth * (n_lobes-1) * duration / n_lobes,
    #     freq_filter_type,
    #     freq_passband_ripple_lvl,
    #     freq_stopband_ripple_lvl,
    #     freq_cancel_alpha_phs,
    # )
    
    # # Get spectrum
    # rf_lobe_spect = _padded_centered_fft(rf_lobe, 2 * rf_lobe.size, rf_lobe.size)
    
    # # Get weighted spectra
    # weigthed_spect = np.outer(rf_lobe_spect, rf_spect_envelope)
    
    
    
    

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
_b2a = slr.b2a

def _dzbeta(
    n: int, 
    tb: float, 
    ptype: str,
    ftype: str,
    d1: float,
    d2: float,
    cancel_alpha_phs: bool,
) -> NDArray[complex]:
    """Same as dzrf, but returning beta instead."""
    bsf, d1, d2 = slr.calc_ripples(ptype, d1, d2)

    if ftype == "ms": # sinc
        b = slr.msinc(n, tb / 4)
    elif ftype == "pm": # linphase
        b = slr.dzlp(n, tb, d1, d2)
    elif ftype == "min": # minphase
        b = slr.dzmp(n, tb, d1, d2)
        b = b[::-1]
    elif ftype == "max": # maxphase
        b = slr.dzmp(n, tb, d1, d2)
    elif ftype == "ls": # least squares
        b = slr.dzls(n, tb, d1, d2)
    else:
        raise Exception(f'Filter type ("{ftype}") is not recognized.')

    if ptype == "st":
        return b
    
    return bsf * b


def _dz_pins(
    n_slices: int,
    sl_thick: float,
    sl_sep: float,
    g_max: float,
    g_slew: float,
    dt: float,
    b1_max: float,
    ptype: str,
    ftype: str,
    d1: float,
    d2: float,
    gambar: float,
):
    """PINS multiband pulse design using number-of-slices API."""
    # Convert number of slices -> time bandwidth product
    tb = n_slices * sl_thick / sl_sep

    # Run SigPy routine
    rf, g = multiband.dz_pins(
        tb=tb,
        sl_sep=sl_sep,
        sl_thick=sl_thick,
        g_max=g_max,
        g_slew=g_slew,
        dt=dt,
        b1_max=b1_max,
        ptype=ptype,
        ftype=ftype,
        d1=d1,
        d2=d2,
        gambar=gambar,
    )
    
    return gambar * rf, 100.0 * gambar * g

# def _make_spectral_weighting(
#     n_lobes: int,
#     time_bw_product: float,
#     filter_type: str,
#     passband_ripple_lvl: float,
#     stopband_ripple_lvl: float,
#     cancel_alpha_phs: bool,
#     system: Opts,
# ) -> NDArray[complex]:
#     """Generate spectral weighting function to achieve spectral selectivity."""
#     rf_envelope = dzrf(
#         n_samples=n_lobes,
#         tb=time_bw_product,
#         ptype='st',
#         ftype=filter_type,
#         d1=passband_ripple_lvl,
#         d2=stopband_ripple_lvl,
#         cancel_alpha_phs=cancel_alpha_phs,
#     )
#     flip = np.sum(rf_envelope) * system.rf_raster_time * 2 * np.pi
#     return rf_envelope / flip


def _padded_centered_fft(input: NDArray[complex], isize: int, osize: int
) -> NDArray[complex]:
    """Pad with zeros to target size and perform centered fft of the result.
    
    Output has size `osize`.
    """
    return resize(fft(input, (isize,)), (osize,))
        
