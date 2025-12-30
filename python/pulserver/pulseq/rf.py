"""
Pulseq rf pulse design functions.

This module contains low-level helper functions to create rf pulses,
extending PyPulseq toolbox.

"""

__all__ = [
    "make_adiabatic_t2prep",
    "make_slr_pulse",
    "make_sms_pulse",
    "make_spsp_pulse",
]

from types import SimpleNamespace

import numpy as np

from pypulseq import Opts
from pypulseq.calc_duration import calc_duration
from pypulseq.calc_rf_center import calc_rf_center
from pypulseq.sigpy_pulse_opts import SigpyPulseOpts
from pypulseq.make_sigpy_pulse import sigpy_n_seq

from sigpy.mri.rf import calc_ripples
from sigpy.mri.rf import dzrf
    
def make_adiabatic_t2prep():
    ...
    

def make_slr_pulse(
    flip_angle: float,
    bandwidth: float = 0.0,
    delay: float = 0.0,
    dwell: float = 0.0,
    duration: float = 4e-3,
    freq_offset: float = 0.0,
    max_grad: float = 0.0,
    max_slew: float = 0.0,
    phase_offset: float = 0.0,
    return_gz: bool = False,
    slice_thickness: float = 0.0,
    system: Opts | None = None,
    time_bw_product: float = 4.0,
    use: str = 'undefined',
    freq_ppm: float = 0.0,
    phase_ppm: float = 0.0,
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
    bandwidth : float, default=0.0
        Bandwidth in Hertz (Hz).
    delay : float, default=0.0
        Delay in seconds (s).
    dwell : float, default=0.0
    duration : float, default=4e-3
        Duration in seconds (s).
    freq_offset : float, default=0.0
        Frequency offset in Hertz (Hz).
    max_grad : float, default=0.0
        Maximum gradient strength of accompanying slice select trapezoidal event.
    max_slew : float, default=0.0
        Maximum slew rate of accompanying slice select trapezoidal event.
    phase_offset : float, default=0.0
        Phase offset in Hertz (Hz).
    return_gz : bool, default=False
        Boolean flag to indicate if the slice-selective gradient has to be returned.
    slice_thickness : float, default=0.0
        Slice thickness of accompanying slice select trapezoidal event. 
        The slice thickness determines the area of the slice select event.
    system : Opts, default=Opts()
        System limits.
    time_bw_product : float, default=4.0
        Time-bandwidth product.
    use : str, default='undefined'
        Use of radio-frequency Gauss pulse event.
        Must be one of 'excitation', 'refocusing', 'inversion',
        'saturation', 'preparation', 'other', 'undefined'.
    freq_ppm : float, default=0.0
        PPM frequency offset.
    phase_ppm : float, default=0.0
        PPM phase offset.
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

    # SigPy specific options
    pulse_cfg = SigpyPulseOpts(
        pulse_type='slr',
        ptype=_use2ptype(use),
        ftype=filter_type,
        d1=passband_ripple_lvl,
        d2=stopband_ripple_lvl,
        cancel_alpha_phs=cancel_alpha_phs,
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
        return_gz,
        slice_thickness,
        system,
        time_bw_product,
        pulse_cfg,
        use,
        False, # plot
        freq_ppm,
        phase_ppm,
    )
    
def make_sms_pulse(
    flip_angle: float,
    n_bands: int,
    slice_thickness: float,
    slice_separation: float | None = None,
    bandwidth: float = 0.0,
    delay: float = 0.0,
    dwell: float = 0.0,
    duration: float = 4e-3,
    freq_offset: float = 0.0,
    max_grad: float = 0.0,
    max_slew: float = 0.0,
    phase_offset: float = 0.0,
    system: Opts | None = None,
    time_bw_product: float = 4.0,
    use: str = 'undefined',
    freq_ppm: float = 0.0,
    phase_ppm: float = 0.0,
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
    slice_thickness : float
        Slice thickness of accompanying slice select trapezoidal event. 
        The slice thickness determines the area of the slice select event.
    slice_separation : float
        Distance between slices.
    bandwidth : float, default=0.0
        Bandwidth in Hertz (Hz).
    delay : float, default=0.0
        Delay in seconds (s).
    dwell : float, default=0.0
    duration : float, default=4e-3
        Duration in seconds (s).
    freq_offset : float, default=0.0
        Frequency offset in Hertz (Hz).
    max_grad : float, default=0.0
        Maximum gradient strength of accompanying slice select trapezoidal event.
    max_slew : float, default=0.0
        Maximum slew rate of accompanying slice select trapezoidal event.
    phase_offset : float, default=0.0
        Phase offset in Hertz (Hz).
    system : Opts, default=Opts()
        System limits.
    time_bw_product : float, default=4.0
        Time-bandwidth product.
    use : str, default='undefined'
        Use of radio-frequency Gauss pulse event.
        Must be one of 'excitation', 'refocusing', 'inversion',
        'saturation', 'preparation', 'other', 'undefined'.
    freq_ppm : float, default=0.0
        PPM frequency offset.
    phase_ppm : float, default=0.0
        PPM phase offset.
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
    bandwidth: float = 0.0,
    delay: float = 0.0,
    dwell: float = 0.0,
    duration: float = 4e-3,
    freq_offset: float = 0.0,
    max_grad: float = 0.0,
    max_slew: float = 0.0,
    phase_offset: float = 0.0,
    return_gz: bool = False,
    slice_thickness: float = 0.0,
    system: Opts | None = None,
    time_bw_product: float = 4.0,
    use: str = 'undefined',
    freq_ppm: float = 0.0,
    phase_ppm: float = 0.0,
    filter_type: str = 'ls',
    passband_ripple_lvl: float = 0.01,
    stopband_ripple_lvl: float = 0.01,
    cancel_alpha_phs: bool = False,
) -> SimpleNamespace | tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace]:
    # Get pulse type
    pulse_type = _use2ptype(use, flip_angle)
    
    # 1) get ripple and amplitude scaling
    scale, passband_ripple_lvl, stopband_ripple_lvl = calc_ripples(
        pulse_type,
        passband_ripple_lvl,
        stopband_ripple_lvl,
    )
    
    # Update scale
    scale *= flip_angle * np.deg2rad(1.0)
    
    
    

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
