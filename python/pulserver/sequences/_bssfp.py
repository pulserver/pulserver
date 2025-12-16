"""
bssfp_pypulseq.py

Translated from the provided MATLAB True-FISP (bSSFP) example into pypulseq.

Notes / assumptions:
- This reproduces the sequence structure (alpha/2 preparation, loop of PE lines,
  RF+slice gradients, readout blocks) but avoids a few low-level gradient-splitting
  utilities from the MATLAB mr-utils that are not universally available in pypulseq
  on every version. If you have pypulseq utilities for splitting gradients and
  extended trapezoids, you can swap them in where noted.
- Timing will be close but may differ a few microseconds from the MATLAB version.
- Check the attribute names in your pypulseq installation (e.g., .flat_time, .rise_time).
"""

import copy
import logging

import numpy as np
import pypulseq as pp
from pypulseq.opts import Opts


def bSSFP2D():
    ...
    # if params is None:
    #     params = {}

    # # Prescription
    # fov_x, fov_y, fov_z = fov[0] * 1e-3, fov[1] * 1e-3, fov[2] * 1e-3
    # Nx, Ny, Nz = npix, npix, npix

    # # RF
    # flip_angle = np.deg2rad(flip_angle_deg)
    # phase_inc = np.deg2rad(phase_inc_deg)

    # # ADC dwell time
    # adc_dwell = 1e-3 / rbw # seconds

    # # RF parameters
    # rf_dur = params.get('rf_dur', 600.0e-6)  # seconds
    # rf_apo = params.get('rf_apo', 0.5)
    # rf_tbw = params.get('rf_tbw', 1.5)  # time-bandwidth product

    # # set system limits (converted to pypulseq naming)
    # if system is None:
    #     system = Opts.default

    # # Initialize sequence object
    # seq = pp.Sequence(system)


def bSSFP3D(
    fov,
    npix,
    flip_angle_deg,
    TR,
    osf=2.0,
    rbw=250.0,
    phase_inc_deg=180.0,
    system=None,
    params=None,
):
    if params is None:
        params = {}

    # Prescription
    fov_x, fov_y, fov_z = fov[0] * 1e-3, fov[1] * 1e-3, fov[2] * 1e-3
    Nx, Ny, Nz = npix, npix, npix
    Nsamples = np.ceil(osf * Nx).astype(int).item()

    # Timing
    TR *= 1e-3  # ms -> s

    # RF
    flip_angle = np.deg2rad(flip_angle_deg)
    np.deg2rad(phase_inc_deg)

    # ADC dwell time
    adc_dwell_time = 1e-3 / (2.0 * rbw)  # seconds
    readout_dur = Nsamples * adc_dwell_time

    # RF parameters
    rf_dur = params.get('rf_dur', 600.0e-6)  # seconds
    rf_tbw = params.get('rf_tbw', 1.5)  # time-bandwidth product

    # set system limits (converted to pypulseq naming)
    if system is None:
        system = Opts(
            max_grad=30,  # mT/m
            grad_unit='mT/m',
            max_slew=140,  # mT/m/ms  (1 T/m/s == 1 mT/m/ms)
            slew_unit='T/m/s',
            rf_ringdown_time=20e-6,
            rf_dead_time=100e-6,
            adc_dead_time=20e-6,
        )

    # Create hard RF pulse
    rf = pp.make_block_pulse(
        flip_angle=flip_angle, duration=rf_dur, time_bw_product=rf_tbw, system=system
    )

    # k-space step
    deltakx = 1.0 / fov_x
    deltaky = 1.0 / fov_y
    deltakz = 1.0 / fov_z

    # Calculate timing
    readout_dur = Nsamples * adc_dwell_time

    # Readout gradient: flat area = Nx * deltak, flat time = adc duration
    gx = pp.make_trapezoid(
        channel='x', flat_area=Nx * deltakx, flat_time=readout_dur, system=system
    )

    # ADC event synchronized to readout gradient
    adc = pp.make_adc(
        num_samples=Nsamples, duration=gx.flat_time, delay=gx.rise_time, system=system
    )

    # Prephaser to move the echo to center
    gx_pre = pp.make_trapezoid(channel='x', area=-gx.area / 2.0, system=system)
    copy.deepcopy(gx_pre)

    # min TR
    minTE = (
        0.5 * pp.calc_duration(rf)
        + pp.calc_duration(gx_pre)
        + 0.5 * pp.calc_duration(gx, adc)
    )
    minTR = 2 * minTE
    if minTR > TR:
        logging.warning(
            f'Minimum TR {minTR*1e3:.2f} ms is shorter than requested {TR*1e3:.2f} ms'
        )
        TR = minTR
        logging.warning(f'New TR is {TR*1e3:.2f} ms')

    # Computing TE
    TE = 0.5 * TR

    # Target delay
    delay = TE - minTE

    # Move forward readout
    gx_pre.delay += delay

    # Post readout delay
    if delay:
        delay = pp.make_delay(delay)
    else:
        delay = None

    # Phase encode areas
    (np.arange(Ny) - Ny / 2.0) * deltaky
    (np.arange(Nz) - Nz / 2.0) * deltakz

    # Initialize sequence object
    seq = pp.Sequence(system)

    # alpha/2 preparation pulse: half amplitude of RF (no readout or PE)
    rf05 = copy.deepcopy(rf)
    rf05.signal = 0.5 * rf05.signal

    # Add alpha/2 preparation block
    # Aligning delays exactly as in MATLAB requires the advanced gradient-splitting utilities.
    # We'll insert a simple block with rf05 + gz (slice select) and a short delay that approximates the intended timing.
    seq.add_block(rf05, gz)

    # A small delay to mimic preparation timing
    prep_delay_time = max(
        0.0,
        (
            TR / 2.0 - pp.calc_duration(gz)
            if hasattr(pp, 'calc_duration')
            else max(0.0, TR / 2.0 - gz_dur)
        ),
    )
    if prep_delay_time > 0:
        seq.add_block(pp.make_delay(prep_delay_time))

    # Add a second preparatory block to complete alpha/2 part (matching the MATLAB structure loosely)
    seq.add_block(pp.make_delay(1e-6))

    # Now loop over phase encodes
    # Alternate RF phase by 180 degrees for even/odd lines (bSSFP phase cycling)
    for i in range(Ny):
        # RF phase offset alternates between 0 and pi
        phase_offset = np.pi * (i % 2)
        # set phase on RF and ADC events (pypulseq stores phase as .phase_offset in many versions)
        try:
            rf.phase_offset = phase_offset
            adc.phase_offset = phase_offset
        except Exception:
            # if direct attribute not present, try event-specific setter or skip (depends on pypulseq version)
            pass

        # phase-encode gradient for first half-block (undo previous PE)
        if i == 0:
            gy_pre_prev = pp.make_trapezoid(
                channel='y', area=phase_areas[0], system=system
            )
        else:
            gy_pre_prev = pp.make_trapezoid(
                channel='y', area=phase_areas[i - 1], system=system
            )

        # current PE step
        gy = pp.make_trapezoid(
            channel='y', area=phase_areas[i], system=system, duration=gx_dur
        )

        # RF + slice selection + previous PE (first TR-part)
        seq.add_block(rf, gz, gy_pre_prev)

        # Readout block: prephaser + current PE + slice refocus remainder + ADC
        seq.add_block(gx_pre, gy, gx, adc)

    # Optionally add a trailing block to finish readout gradient
    seq.add_block(gx)

    return seq
