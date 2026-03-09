"""Shared fixtures for the pulserver pytest suite."""

import numpy as np
import pypulseq as pp
import pytest


@pytest.fixture
def simple_gre_seq():
    """Build a simple 2D GRE sequence with pypulseq."""
    sys = pp.Opts(
        max_grad=32,
        grad_unit='mT/m',
        max_slew=130,
        slew_unit='T/m/s',
        rf_ringdown_time=20e-6,
        rf_dead_time=100e-6,
        adc_dead_time=10e-6,
    )
    seq = pp.Sequence(system=sys)

    flip = 15 * np.pi / 180
    rf, gz, _ = pp.make_sinc_pulse(
        flip_angle=flip,
        duration=3e-3,
        slice_thickness=5e-3,
        apodization=0.5,
        time_bw_product=4,
        system=sys,
        return_gz=True,
    )

    gx = pp.make_trapezoid(channel='x', flat_area=128 / 0.25, flat_time=3.2e-3, system=sys)
    adc = pp.make_adc(num_samples=128, duration=gx.flat_time, delay=gx.rise_time, system=sys)
    gx_pre = pp.make_trapezoid(channel='x', area=-gx.area / 2, duration=1e-3, system=sys)
    gz_reph = pp.make_trapezoid(channel='z', area=-gz.area / 2, duration=1e-3, system=sys)

    n_pe = 16
    pe_areas = np.linspace(-0.5, 0.5, n_pe) * 128 / 0.25

    for i in range(n_pe):
        gy_pre = pp.make_trapezoid(channel='y', area=pe_areas[i], duration=1e-3, system=sys)
        seq.add_block(rf, gz)
        seq.add_block(gx_pre, gy_pre, gz_reph)
        seq.add_block(gx, adc)

    return seq
