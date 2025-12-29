""" """

__all__ = ['make_epi_readout']

import copy
from typing import Union

import numpy as np
import pypulseq as pp
from pypulseq import Opts

from ._utils import find_gx_flat_time_on_adc_raster


def make_epi_readout(
    fov,
    mtx,
    Ry: int = 1,
    num_shots: int = 1,
    dwell_time: float = 0.1e-6,
    osf: float = 2.0,
    ramp_sampling: bool = False,
    flyback: bool = False,
    system: Union[Opts, None] = None,
    ret_sequence: bool = False,
):
    if system is None:
        system = Opts.default

    # Parse geometry
    Nx, Ny = mtx
    fov_x, fov_y = fov

    # Compute n readouts per shot
    num_readouts_per_shot = Ny // (Ry * num_shots)

    # Get actual samples
    num_samples = int(np.ceil(osf * Nx).item())

    # Get gradient raster time
    grad_raster_time = system.grad_raster_time
    adc_raster_time = system.adc_raster_time

    # Get readout duration
    duration, _ = find_gx_flat_time_on_adc_raster(
        num_samples, dwell_time, grad_raster_time, adc_raster_time
    )

    # Get readout area based on target FOV and resolution
    delta_kx = 1 / fov_x
    kx_area = Nx * delta_kx

    # Build readout
    if ramp_sampling:
        gread = pp.make_trapezoid(channel='x', duration=duration, area=kx_area)
    else:
        gread = pp.make_trapezoid('x', flat_time=duration, flat_area=kx_area)

    # Get blip area
    delta_ky = 1 / fov_y  # Base density for given fov
    ky_blip_area = Ry * delta_ky  # Acceleration increases ky spacing linearly
    ky_area = Ny * delta_ky

    # Build phase blip
    if ramp_sampling:
        blip_delay = 0.0
    else:
        blip_delay = gread.rise_time + gread.flat_time
    gblip, _, _ = pp.make_extended_trapezoid_area(
        channel='y', area=ky_blip_area, grad_start=0.0, grad_end=0.0
    )
    gblip.delay = blip_delay

    # Build prewinder
    gx_prew, _, _ = pp.make_extended_trapezoid_area(
        channel='x', area=-0.5 * gread.area, grad_start=0.0, grad_end=0.0
    )
    pp.scale_grad(gx_prew, scale=(-1) ** (num_readouts_per_shot + 1))

    gy_prew, _, _ = pp.make_extended_trapezoid_area(
        channel='y', area=-ky_area, grad_start=0.0, grad_end=0.0
    )
    copy.deepcopy(gy_prew)

    # Find scalings
    y_scale = np.arange(-Ny // 2, Ny // 2, Ry) / Ny
    y_scale[::num_readouts_per_shot]
    np.roll(y_scale, -num_readouts_per_shot)[::num_readouts_per_shot]

    # Build ADC
    if ramp_sampling:
        adc_delay = 0.0
    else:
        adc_delay = gread.rise_time
    adc = pp.make_adc(num_samples=num_samples, duration=duration, delay=adc_delay)

    # Optionally, build sequence containing readout for a single shot
    if ret_sequence:
        seq = pp.Sequence(system=system)

        seq.add_block()
        if ramp_sampling:
            for n in range(num_readouts_per_shot):
                scale = (-1) ** n
                seq.add_block(pp.scale_grad(gread, scale), adc)
                seq.add_block(gblip)
        else:
            for n in range(num_readouts_per_shot):
                scale = (-1) ** n
                seq.add_block(pp.scale_grad(gread, scale), adc, gblip)

        return seq
