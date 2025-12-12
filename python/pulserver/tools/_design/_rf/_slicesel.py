"""Slice selection subroutine."""

__all__ = []

from copy import copy, deepcopy
from types import SimpleNamespace
from typing import Union

import numpy as np

from pypulseq import Opts
from pypulseq.align import align
from pypulseq.add_gradients import add_gradients
from pypulseq.make_trapezoid import make_trapezoid
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.points_to_waveform import points_to_waveform
from pypulseq.scale_grad import scale_grad

def _as_spatial_selective(
        pulse: SimpleNamespace, 
        slice_thickness: float,
        bandwidth: float = 0.0,
        max_grad: float = 0.0,
        max_slew: float = 0.0,
        system: Union[Opts, None] = None,
        time_bw_product: float = 0,
) -> tuple[SimpleNamespace, SimpleNamespace]:
    if system is None:
        system = Opts.default
    if max_grad > 0:
        system = copy(system)
        system.max_grad = max_grad
    if max_slew > 0:
        system = copy(system)
        system.max_slew = max_slew
    if bandwidth == 0 and time_bw_product == 0:
        raise ValueError('User must provide bandwidth or time-bandwidth product.')
    if bandwidth != 0 and time_bw_product != 0:
        raise ValueError('User must provide either bandwidth or time-bandwidth product, not both.')
    if bandwidth == 0:
        bandwidth = time_bw_product / pulse.shape_dur

    # Compute trapezoid params
    amplitude = bandwidth / slice_thickness
    flat_area = amplitude * pulse.shape_dur
    gz = make_trapezoid(channel='z', system=system, flat_time=pulse.shape_dur, flat_area=flat_area)
    
    # Compute rephasing params
    center_pos = pulse.center / pulse.shape_dur
    gz_rephase = make_trapezoid(
            channel='z',
            system=system,
            area=-flat_area * (1 - center_pos) - 0.5 * (gz.area - flat_area),
        )
    
    # Shift gradient within the block to align with rf pulse
    if pulse.delay > gz.rise_time:
        gz.delay = np.ceil((pulse.delay - gz.rise_time) / system.grad_raster_time) * system.grad_raster_time

    return gz, gz_rephase


def _as_adiabatic_spatial_selective(
        pulse: SimpleNamespace,
        slice_thickness: float,
        lobes: int = None,
        duration: Union[float, None] = None,
        bandwidth: float = 0.0,
        max_grad: float = 0.0,
        max_slew: float = 0.0,
        system: Union[Opts, None] = None,
        time_bw_product: float = 0,
        apodization: float = 0,
) -> tuple[SimpleNamespace, SimpleNamespace]:
    """
    Generate a slice-selective adiabatic pulse using sinc subpulses and
    PyPulseq align + add_gradients.

    Returns
    -------
    rf_block : SimpleNamespace
        Arbitrary RF block containing the full concatenated RF waveform.
    gz_combined : SimpleNamespace
        Combined slice-selection gradient for the entire pulse.
    """
    if system is None:
       system = Opts.default
    
    # Get duration       
    T = pulse.shape_dur

    # Case 1: lobes only
    if lobes is not None and duration is None:
        duration = T / lobes

    # Case 2: duration only
    elif duration is not None and lobes is None:
        lobes = int(round(T / duration))
        duration = T / lobes  # recompute exact

    # Case 3: both lobes and duration given → validate
    elif lobes is not None and duration is not None:
        raise ValueError('Please specify at most number of lobes or lobes duration, not both')

    # Case 4: neither given → choose default
    else:
        lobes = 50
        duration = T / lobes

    # TBW vs bandwidth
    if time_bw_product > 0 and bandwidth == 0:
        bandwidth = time_bw_product / duration
    elif bandwidth > 0 and time_bw_product == 0:
        time_bw_product = bandwidth * duration
    elif bandwidth > 0 and time_bw_product > 0:
        raise ValueError('Please specify either subpulse bandwidth or its TBW, not both')
    else:
        raise ValueError('Please specify subpulse bandwidth or its TBW')
        
    # Get dwell time
    dwell = T / pulse.signal.size
    
    # Extract rf_envelope
    rf_env = np.asarray(pulse.signal)
    Lenv = len(rf_env)

    # Initialize subpulses
    rf_events = []
    gz_events = []
    
    # Base sub-pulse
    rf_sub_base, gz_sub_base = make_sinc_pulse(
        flip_angle=np.pi / 2, # does not matter
        apodization=apodization,
        delay=pulse.delay,
        duration=duration,
        dwell=dwell,
        center_pos=0.5,
        freq_offset=pulse.freq_offset,
        max_grad=max_grad,
        max_slew=max_slew,
        phase_offset=pulse.phase_offset,
        return_gz=True,
        slice_thickness=slice_thickness,
        system=system,
        time_bw_product=time_bw_product,
        use=pulse.use,
        freq_ppm=pulse.freq_ppm,
        phase_ppm=pulse.phase_ppm,
    )
    rf_sub_base /= rf_sub_base.sum()

    # Generate sublobes
    for ii in range(lobes):
        idx = int(np.ceil((2*ii+1) / 2 * pulse.signal.size / lobes)) - 1
        idx = max(0, min(idx, Lenv-1))
        env_value = pulse.signal[idx]
        
        # Deep copy
        rf_sub, gz_sub = deepcopy(rf_sub_base), deepcopy(gz_sub_base)

        # Scale RF by envelope sample
        rf_sub.signal *= env_value

        # Flip gradient polarity every other lobe
        gz_sub = scale_grad(gz_sub, (-1)**ii)

        # Store blocks
        rf_events.append(rf_sub)
        gz_events.append(gz_sub)

    # Align subpulses
    aligned_gz_events = align(right=gz_events)

    # Combine gradients into single block
    gz_combined = add_gradients(gz_events, max_grad, max_slew, system)
    
    # Get full gz_waveform
    gz_waveform = points_to_waveform(gz_combined.waveform, system.grad_raster_time, gz_combined.tt)

    # Concatenate RF signals from all blocks
    rf_combined = [points_to_waveform(rf.signal, system.rf_raster_time, rf.t) for rf in rf_events]
    rf_combined = np.concatenate(rf_combined)

    # Make final arbitrary RF block
    # rf_block = pp.make_arbitrary_rf(
    #     signal=rf_combined,
    #     flip_angle=flip_angle,
    #     return_gz=False,
    #     system=system
    # )

    return rf_combined, gz_combined
        

