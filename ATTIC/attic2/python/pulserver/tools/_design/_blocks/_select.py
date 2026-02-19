"""
"""

__all__ = []

import logging
import warnings

from types import SimpleNamespace

import numpy as np
import pypulseq as pp


def log_and_raise(msg, exc=ValueError): # noqa
    logging.error(msg)
    raise exc(msg)


def log_and_warn(msg, exc=UserWarning): # noqa
    logging.warning(msg)
    warnings.warn(msg, exc)
    
    
def make_spatial_selection(
    rf: SimpleNamespace,
    slice_thickness: float,
    system: pp.Opts | None = None,
    include_rf_delays_in_flat: bool = False,
    return_slice_rephase_grad: bool = True,
    lstrip: bool = False,
    rstrip: bool = False,
) -> SimpleNamespace:

    # Analyze input rf pulse
    duration = pp.calc_duration(rf)
    t_rf_center, _ = pp.calc_rf_center(rf)
    center_pos = t_rf_center / duration
    bandwidth = float(pp.calc_rf_bandwidth(rf).item())
    if bandwidth == 0.0: # block pulse
        bandwidth = 2.0 / duration # 2 / T for sinc spectrum
    
    # Compute
    amplitude = bandwidth / slice_thickness
    area = amplitude * duration
    gz_slice_select = pp.make_trapezoid(
        channel='z', 
        system=system, 
        flat_time=duration, 
        flat_area=area
    )
    if include_rf_delays_in_flat:
        gz_slice_select = pp.make_trapezoid(
            channel='z',
            system=system,
            amplitude=gz_slice_select.amplitude,
            flat_time=gz_slice_select.flat_time + system.rf_ringdown_time + system.rf_dead_time,
            rise_time=gz_slice_select.rise_time,
        )
        
    # Align RF and Slice Selection
    if rf.delay > gz_slice_select.rise_time:
        gz_slice_select.delay = np.ceil((rf.delay - gz_slice_select.rise_time) / system.grad_raster_time) * system.grad_raster_time
    if rf.delay < (gz_slice_select.rise_time + gz_slice_select.delay):
        rf.delay = gz_slice_select.rise_time + gz_slice_select.delay
        
    # Set pieces
    if rstrip or lstrip:
        if return_slice_rephase_grad:
            log_and_warn('Stripping ramps is requested - setting "return_slice_rephase_grad" to False')
        return_slice_rephase_grad = False
        try:
            grad_rise, grad_flat, grad_fall = pp.split_gradient(gz_slice_select, system)
        except Exception:
            grad_flat = None
            grad_rise, grad_fall = pp.split_gradient_at(gz_slice_select, gz_slice_select.rise_time, system)
    if lstrip:
        rf.delay -= gz_slice_select.rise_time
    if lstrip and rstrip:
        if grad_flat is None:
            log_and_raise('Gradient is too short, cannot isolate a flat portion')
        grad = grad_flat
        grad.delay = 0.0
    elif lstrip:
        grad = pp.add_gradients([grad_flat, grad_fall], system=system)
        grad.delay = 0.0
    elif rstrip:
        grad = pp.add_gradients([grad_rise, grad_flat], system=system)
    
    # Build slice rephase
    if return_slice_rephase_grad:
        gz_slice_rephase = pp.make_trapezoid(
            channel='z',
            system=system,
            area=-area * (1 - center_pos) - 0.5 * (gz_slice_select.area - area),
        )
    else:
        gz_slice_rephase = None
        
    return rf, gz_slice_select, gz_slice_rephase