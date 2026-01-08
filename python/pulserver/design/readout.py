"""
"""

__all__ = [
    
]

from types import SimpleNamespace

import numpy as np

from .. import pulseq as pp

def line_readout(
    system: pp.Opts,
    fov_m: float,
    npix: int,
    receive_bandwidth_Hz = 250e3,
    oversamp: float = 1.0,
    partial_fourier_factor: float = 1.0,
    rampsamp: bool = False,
    left_ramp: bool = True,
    right_ramp: bool = True,
) -> tuple[SimpleNamespace, SimpleNamespace]:
    """
    Create basic trapezoidal readout lobe.

    Parameters
    ----------
    system : pp.Opts
        Pulseq system limits. 
    fov_m : float
        Field of view along readout direction in ``[m]``.
    npix : int
        Image size along readout direction.
    receive_bandwidth_Hz : TYPE, optional
        Receive bandwidth in ``[Hz]``. 
        The default is ``250e-3 Hz``.
    oversamp : float, optional
        Readout oversampling factor. 
        The default is ``1.0``.
    partial_fourier_factor : float, optional
        Partial Fourier Factor for asymmeric echo. 
        The default is ``1.0``.
    rampsamp : bool, optional
        If ``True``, sample points on trapezoid ramps. 
        The default is ``Fals``e.
    left_ramp : bool, optional
        Include ramp-up part of trapezoid in readout event. 
        The default is ``True``.
    right_ramp : bool, optional
        Include ramp-down part of trapezoid in readout event. 
        The default is ``True``.

    Raises
    ------
    ValueError
        If ``rampsamp`` is ``True``, and either ``left_ramp`` or ``right_ramp``
        are ``True``.

    Returns
    -------
    readout : SimpleNamespace
        Object containing readout gradient and accompanying ADC event.
    metadata : SimpleNamespace
        Object containing the following additional info
            - system : pp.Opts
                  System parameters used to generate event.
            - encoding_size : int
                  Image size along readout direction.
            - encoding_center : int
                  ADC index corresponding to k-space center along readout direction.
            - encoding_center_time : float
                  Sampling time of k-space center along readout direction in ``[s]``.
            - readout_time : float
                  Total readout time in ``[s]``.
            - pre_readout_area : float
                  Readout gradient area before echo in ``[Hz/m]``.
            - post_readout_area : float
                  Readout gradient area after echo in ``[Hz/m]``.

    """
    if rampsamp and (not(left_ramp) or not(right_ramp)):
        raise ValueError('"rampsamp" option is not compatible with cropped readout')
        
    # Save metadata
    metadata = SimpleNamespace(system=system, image_size=npix)
        
    # Get readout parameters
    readout_area, readout_time, num_samples = pp.calc_kspace_readout_params(
        fov_m,
        npix,
        receive_bandwidth_Hz,
        oversamp,
        system.adc_raster_time,
        system.grad_raster_time,
    )
    dwell_time = readout_time / num_samples
    
    # Update metadata
    metadata.encoding_size = num_samples

    # Make sure partial_fourier leads to integer number of samples
    act_num_samples = np.ceil(partial_fourier_factor * num_samples).astype(int).item()
    partial_fourier_factor = act_num_samples / num_samples
    
    # Update metadata
    metadata.readout_time = act_num_samples * dwell_time
    metadata.post_echo_area = 0.5 * readout_area
    metadata.pre_echo_area = (partial_fourier_factor - 0.5) * readout_area
    
    # Apply partial fourier undersampling
    readout_area = partial_fourier_factor * readout_area
            
    # Design readout gradient
    if rampsamp:
        gx_read = pp.make_trapezoid(
            channel='x',
            system=system,
            area=readout_area,
            duration=metadata.readout_time,
        )
        gx_read = pp.make_trapezoid(
            channel='x',
            system=system,
            area=readout_area,
            duration=pp.calc_duration(gx_read) + 2 * system.adc_dead_time
        )
        adc_delay = system.adc_dead_time
        pre_echo_area_offset = 0.0
        post_echo_area_offset = 0.0
    else:
        gx_read = pp.make_trapezoid(
            channel='x',
            system=system,
            flat_area=readout_area,
            flat_time=metadata.readout_time,
        )
        adc_delay = gx_read.rise_time
        pre_echo_area_offset = 0.5 * gx_read.rise_time * gx_read.amplitude
        post_echo_area_offset = 0.5 * gx_read.fall_time * gx_read.amplitude
        
    # Cut ramps if ADC block only includes flat part of trapezoid
    if left_ramp is False and right_ramp is False:
        gx_read = pp.make_trapezoid(
            channel='x',
            system=system,
            flat_area=readout_area,
            flat_time=gx_read.flat_time + 2 * system.adc_dead_time,
        )
        _, gx_read, _ = pp.split_waveform(gx_read, system=system)
        adc_delay = system.adc_dead_time
        pre_echo_area_offset = 0.0
        post_echo_area_offset = 0.0
    elif left_ramp is False:
        gx_read = pp.make_trapezoid(
            channel='x',
            system=system,
            flat_area=readout_area,
            flat_time=gx_read.flat_time + system.adc_dead_time,
        )
        _, gx_read = pp.split_waveform_at(
            gx_read, 
            time_point=gx_read.rise_time,
            system=system,
        )
        adc_delay = system.adc_dead_time
        pre_echo_area_offset = 0.0
    elif right_ramp is False:
        gx_read = pp.make_trapezoid(
            channel='x',
            system=system,
            flat_area=readout_area,
            flat_time=gx_read.flat_time + system.adc_dead_time,
        )
        adc_delay = gx_read.rise_time
        gx_read, _ = pp.split_waveform_at(
            gx_read, 
            time_point=gx_read.rise_time+gx_read.flat_time,
            system=system,
        )
        post_echo_area_offset = 0.0
        
    # Update metadata
    metadata.pre_echo_area += pre_echo_area_offset
    metadata.post_echo_area += post_echo_area_offset
        
    # Design Echo filter
    adc = pp.make_adc(
        system=system,
        delay=adc_delay,
        num_samples=act_num_samples,
        dwell=dwell_time
    )
    
    # Compute echo
    metadata.encoding_center = act_num_samples - (num_samples // 2)
    metadata.encoding_center_time = adc.delay + metadata.encoding_center * adc.dwell
                                                    
    # Assign parameters
    readout = SimpleNamespace(gx=gx_read, adc=adc)

    return readout, metadata
                    
# class LineReadout(BareboneReadout):
#     def __init__(
#         self,
#         system: pp.Opts,
#         fov_m: float,
#         npix: int,
#         receive_bandwidth_Hz = 250e-3,
#         oversamp: float = 1.0,
#         partial_fourier_factor: float = 1.0,
#         rampsamp: bool = False,
#         flatsamp: bool = False,
#         gx_first: float = 0.0,
#         kx_first: float = 0.0,
#         gx_last: float = 0.0,
#         kx_last: float = 0.0,
#     ):
#         super().__init__(
#             system,
#             fov_m,
#             npix,
#             receive_bandwidth_Hz,
#             oversamp,
#             partial_fourier_factor,
#             rampsamp,
#             flatsamp,
#         )
        
#         # Get target initial k-space  (excluding ramp up)
#         post_echo_area = readout_area / 2.0
#         readout_area = partial_fourier_factor * readout_area
#         pre_echo_area = readout_area - post_echo_area
        
#         # Design readout gradient
#         if rampsamp:
#             gx_read = pp.make_trapezoid(
#                 channel='x',
#                 system=system,
#                 area=readout_area,
#             )
#             gx_read = pp.make_trapezoid(
#                 channel='x',
#                 system=system,
#                 area=readout_area,
#                 duration=pp.calc_duration(gx_read) + 2 * system.adc_dead_time
#             )
#             post_echo_offset = 0.0
#             pre_echo_offset = 0.0
#             adc_delay = system.adc_dead_time
#         else:
#             gx_read = pp.make_trapezoid(
#                 channel='x',
#                 system=system,
#                 flat_area=readout_area,
#             )
#             post_echo_offset = 0.5 * (gx_read.rise_time * gx_read.amplitude)
#             pre_echo_offset = 0.5 * (gx_read.fall_time * gx_read.amplitude)
#             adc_delay = gx_read.rise_time
            
#         # Cut ramps if ADC block only includes flat part of trapezoid
#         if flatsamp:
#             gx_read = pp.make_trapezoid(
#                 channel='x',
#                 system=system,
#                 flat_area=readout_area,
#                 flat_time=gx_read.flat_time + 2 * system.adc_dead_time,
#             )
#             _, gx_read, _ = pp.split_waveform(gx_read, system=system)
#             gx_read_first = gx_read.amplitude
#             gx_read_last = gx_read.amplitude
#             post_echo_offset = 0.0
#             pre_echo_offset = 0.0
#             adc_delay = system.adc_dead_time
#         else:
#             gx_read_first = 0.0
#             gx_read_last = 0.0
        
#         # Design Echo filter
#         adc = pp.make_adc(
#             system=system,
#             delay=adc_delay,
#             num_samples=num_samples,
#             dwell_time=dwell_time
#         )
            
#         # Compute actual prewinder and rewinder area
#         post_echo_area += post_echo_offset
#         pre_echo_area += pre_echo_offset
                        
#         # Design prewinder
#         prewinder_area = -(kx_first + pre_echo_area)
#         if prewinder_area:
#             gx_prewind = _design_phasor(
#                 system=system,
#                 area=prewinder_area, 
#                 first=gx_first, 
#                 last=gx_read_first,
#             )
#         else:
#             gx_prewind = None
            
#         # Design rewinder
#         rewinder_area = kx_last - post_echo_area
#         if rewinder_area:
#             gx_rewind = _design_phasor(
#                 system=system,
#                 area=rewinder_area, 
#                 first=gx_read_last, 
#                 last=gx_last,
#             )
#         else:
#             gx_rewind = None
            
#         # Sanitize phasors
#         gx_prewind, gx_rewind = _sanitize_phasor(
#             system, 
#             gx_prewind, 
#             gx_rewind,
#         )
        
#         # Assign parameters
#         self.system = system
#         self.gx_prewind = gx_prewind
#         self.gx_rewind = gx_rewind
#         self.gx_read = gx_read
#         self.adc = adc
    
#     @property
#     def amplitude(self):
#         return self.gx_read.amplitude
        
#     @amplitude.setter
#     def amplitude(self, value: float):
#         self.gx_read = pp.scale_grad(self.gx_read, scale=value, system=self.system)
        
        
    
class EPIReadout:
    ...
    
class NonCartesianReadout:
    ...
    
    
# %% Internal Helpers
def _design_phasor(system: pp.Opts, area: float, first: float, last: float) -> SimpleNamespace:
    """Design phasor either as standard or extended trapezoid."""
    if first == 0.0 and last == 0.0: # standard trapezoid
        return pp.make_trapezoid(
            channel='x',
            system=system,
            area=area,
        )
    else:
        return pp.make_extended_trapezoid_area(
            channel='x',
            system=system,
            area=area, 
            grad_start=first, 
            grad_end=last,
        )
    
def _sanitize_phasor(
    system: pp.Opts, 
    prewinder: SimpleNamespace, 
    rewinder: SimpleNamespace,
) -> tuple[SimpleNamespace, SimpleNamespace]:
    """Enforce single rescaled phasor if possible."""
    if prewinder is None or rewinder is None:
        return prewinder, rewinder
    if prewinder.type == 'grad' or rewinder.type == 'grad':
        return prewinder, rewinder
    
    # Check if the gradient timings are the same
    same_rise = prewinder.rise_time == rewinder.rise_time
    same_flat = prewinder.flat_time == rewinder.flat_time
    same_fall = prewinder.fall_time == rewinder.fall_time
    same_delay = prewinder.delay == rewinder.delay
    
    # Enforce rewinder as scaled version of prewinder
    if same_delay and same_rise and same_flat and same_fall:
        rewinder = pp.scale_grad(
            prewinder, 
            rewinder.amplitude / prewinder.amplitude,
            system=system,
        )
        
    return prewinder, rewinder
    
    
    
    
    
    