from pathlib import Path

import numpy as np
import pulserver.io as pio
import pulserver.pulseq as ps
import pypulseq as pp
import pytest

expected_output_path = Path(__file__).parent / "expected_output"


# Dummy sequence which contains only gaussian pulses with different parameters
def seq_make_gauss_pulses():
    seq = ps.Sequence()
    seq.add_block(pp.make_gauss_pulse(flip_angle=1))
    seq.add_block(pp.make_delay(1))
    seq.add_block(pp.make_gauss_pulse(flip_angle=1, delay=1e-3))
    seq.add_block(pp.make_delay(1))
    seq.add_block(pp.make_gauss_pulse(flip_angle=np.pi / 2))
    seq.add_block(pp.make_delay(1))
    seq.add_block(pp.make_gauss_pulse(flip_angle=np.pi / 2, duration=1e-3))
    seq.add_block(pp.make_delay(1))
    seq.add_block(pp.make_gauss_pulse(flip_angle=np.pi / 2, duration=2e-3, phase_offset=np.pi / 2))
    seq.add_block(pp.make_delay(1))
    seq.add_block(pp.make_gauss_pulse(flip_angle=np.pi / 2, duration=1e-3, phase_offset=np.pi / 2, freq_offset=1e3))
    seq.add_block(pp.make_delay(1))
    seq.add_block(pp.make_gauss_pulse(flip_angle=np.pi / 2, duration=1e-3, time_bw_product=1))
    seq.add_block(pp.make_delay(1))
    seq.add_block(pp.make_gauss_pulse(flip_angle=np.pi / 2, duration=1e-3, apodization=0.1))

    return seq


# Dummy sequence which contains only sinc pulses with different parameters
def seq_make_sinc_pulses():
    seq = ps.Sequence()
    seq.add_block(pp.make_sinc_pulse(flip_angle=1))
    seq.add_block(pp.make_delay(1))
    seq.add_block(pp.make_sinc_pulse(flip_angle=1, delay=1e-3))
    seq.add_block(pp.make_delay(1))
    seq.add_block(pp.make_sinc_pulse(flip_angle=np.pi / 2))
    seq.add_block(pp.make_delay(1))
    seq.add_block(pp.make_sinc_pulse(flip_angle=np.pi / 2, duration=1e-3))
    seq.add_block(pp.make_delay(1))
    seq.add_block(pp.make_sinc_pulse(flip_angle=np.pi / 2, duration=2e-3, phase_offset=np.pi / 2))
    seq.add_block(pp.make_delay(1))
    seq.add_block(pp.make_sinc_pulse(flip_angle=np.pi / 2, duration=1e-3, phase_offset=np.pi / 2, freq_offset=1e3))
    seq.add_block(pp.make_delay(1))
    seq.add_block(pp.make_sinc_pulse(flip_angle=np.pi / 2, duration=1e-3, time_bw_product=1))
    seq.add_block(pp.make_delay(1))
    seq.add_block(pp.make_sinc_pulse(flip_angle=np.pi / 2, duration=1e-3, apodization=0.1))

    return seq


# Dummy sequence which contains only block pulses with different parameters
def seq_make_block_pulses():
    seq = ps.Sequence()
    seq.add_block(pp.make_block_pulse(flip_angle=1, duration=4e-3))
    seq.add_block(pp.make_delay(1))
    seq.add_block(pp.make_block_pulse(flip_angle=1, delay=1e-3, duration=4e-3))
    seq.add_block(pp.make_delay(1))
    seq.add_block(pp.make_block_pulse(flip_angle=np.pi / 2, duration=4e-3))
    seq.add_block(pp.make_delay(1))
    seq.add_block(pp.make_block_pulse(flip_angle=np.pi / 2, duration=1e-3))
    seq.add_block(pp.make_delay(1))
    seq.add_block(pp.make_block_pulse(flip_angle=np.pi / 2, duration=2e-3, phase_offset=np.pi / 2))
    seq.add_block(pp.make_delay(1))
    seq.add_block(pp.make_block_pulse(flip_angle=np.pi / 2, duration=1e-3, phase_offset=np.pi / 2, freq_offset=1e3))
    seq.add_block(pp.make_delay(1))
    seq.add_block(pp.make_block_pulse(flip_angle=np.pi / 2, duration=1e-3, time_bw_product=1))

    return seq


# Basic sequence with gradients in all channels, some which are identical after rounding.
def seq1():
    seq = ps.Sequence()
    seq.add_block(pp.make_block_pulse(np.pi / 4, duration=1e-3))
    seq.add_block(pp.make_trapezoid("x", area=1000))
    seq.add_block(pp.make_trapezoid("y", area=-500.00001))
    seq.add_block(pp.make_trapezoid("z", area=100))
    seq.add_block(pp.make_trapezoid("x", area=-1000), pp.make_trapezoid("y", area=500))
    seq.add_block(pp.make_trapezoid("y", area=-500), pp.make_trapezoid("z", area=1000))
    seq.add_block(pp.make_trapezoid("x", area=-1000), pp.make_trapezoid("z", area=1000.00001))

    return seq


# Basic spin-echo sequence structure
def seq2():
    seq = ps.Sequence()
    seq.add_block(pp.make_block_pulse(np.pi / 2, duration=1e-3))
    seq.add_block(pp.make_trapezoid("x", area=1000))
    seq.add_block(pp.make_trapezoid("x", area=-1000))
    seq.add_block(pp.make_block_pulse(np.pi, duration=1e-3))
    seq.add_block(pp.make_trapezoid("x", area=-500))
    seq.add_block(pp.make_trapezoid("x", area=1000, duration=10e-3), pp.make_adc(num_samples=100, duration=10e-3))

    return seq


# Basic GRE sequence with INC labels
def seq3():
    seq = ps.Sequence()

    for i in range(10):
        seq.add_block(pp.make_block_pulse(np.pi / 8, duration=1e-3))
        seq.add_block(pp.make_trapezoid("x", area=1000))
        seq.add_block(pp.make_trapezoid("y", area=-500 + i * 100))
        seq.add_block(pp.make_trapezoid("x", area=-500))
        seq.add_block(
            pp.make_trapezoid("x", area=1000, duration=10e-3),
            pp.make_adc(num_samples=100, duration=10e-3),
            pp.make_label(label="LIN", type="INC", value=1),
        )

    return seq


# Basic GRE sequence with SET labels
def seq4():
    seq = ps.Sequence()

    for i in range(10):
        seq.add_block(pp.make_block_pulse(np.pi / 8, duration=1e-3))
        seq.add_block(pp.make_trapezoid("x", area=1000))
        seq.add_block(pp.make_trapezoid("y", area=-500 + i * 100))
        seq.add_block(pp.make_trapezoid("x", area=-500))
        seq.add_block(
            pp.make_trapezoid("x", area=1000, duration=10e-3),
            pp.make_adc(num_samples=100, duration=10e-3),
            pp.make_label(label="LIN", type="SET", value=i),
        )

    return seq


# GRE sequence with preceding noise acquisition and labels
def seq5():
    sys = pp.Opts()
    seq = ps.Sequence(sys)
    rf, gz, gzr = pp.make_sinc_pulse(flip_angle=np.pi / 8, duration=1e-3, slice_thickness=3e-3, return_gz=True)
    gx = pp.make_trapezoid(channel="x", flat_area=32 * 1 / 0.3, flat_time=32 * 1e-4, system=sys)
    adc = pp.make_adc(num_samples=32, duration=gx.flat_time, delay=gx.rise_time, system=sys)
    gx_pre = pp.make_trapezoid(channel="x", area=-gx.area / 2, duration=1e-3, system=sys)
    phase_areas = -(np.arange(32) - 32 / 2) * (1 / 0.3)

    seq.add_block(pp.make_label(label="LIN", type="SET", value=0), pp.make_label(label="SLC", type="SET", value=0))
    seq.add_block(pp.make_adc(num_samples=1000, duration=1e-3), pp.make_label(label="NOISE", type="SET", value=True))
    seq.add_block(pp.make_label(label="NOISE", type="SET", value=False))
    seq.add_block(pp.make_delay(sys.rf_dead_time))

    for pe in range(32):
        gy_pre = pp.make_trapezoid(channel="y", area=phase_areas[pe], duration=1e-3, system=sys)

        seq.add_block(rf, gz)
        seq.add_block(gx_pre, gy_pre, gzr)
        seq.add_block(gx, adc, pp.make_label(label="LIN", type="SET", value=pe))

        gy_pre.amplitude = -gy_pre.amplitude
        seq.add_block(gx_pre, gy_pre, pp.make_delay(10e-3))

    return seq


# Dummy seq ending with [TRAP] section
def seq_trap_only():
    seq = ps.Sequence()
    seq.add_block(pp.make_trapezoid("x", area=1000))

    return seq


# Dummy seq ending with [ADC] section
def seq_adc_only():
    seq = ps.Sequence()
    seq.add_block(pp.make_adc(num_samples=100, duration=10e-3))

    return seq


# Dummy seq ending with [EXTENSION] section
def seq_ext_only():
    seq = ps.Sequence()
    seq.add_block(pp.make_adc(num_samples=1000, duration=1e-3), pp.make_label(label="NOISE", type="SET", value=True))

    return seq


# List of all sequence functions that will be tested with the test functions below.
sequence_zoo = [
    seq_make_gauss_pulses,
    seq_make_sinc_pulses,
    seq_make_block_pulses,
    seq1,
    seq2,
    seq3,
    seq4,
    seq5,
    seq_trap_only,
    seq_adc_only,
    seq_ext_only,
]

# Bitwise parity scope: only sequences without any extension events.
sequence_zoo_without_extensions = [
    seq_make_gauss_pulses,
    seq_make_sinc_pulses,
    seq_make_block_pulses,
    seq1,
    seq2,
    seq_trap_only,
    seq_adc_only,
]


@pytest.mark.parametrize("seq_factory", sequence_zoo_without_extensions, ids=lambda fn: fn.__name__)
def test_io_write_matches_expected_output(seq_factory):
    seq = seq_factory()
    assert len(seq.extensions_library.data) == 0

    payload = pio.write(
        seq,
        output=None,
        create_signature=True,
        remove_duplicates=True,
        check_timing=False,
    )
    rendered = payload.decode("utf-8")

    expected_file = expected_output_path / f"{seq_factory.__name__}.seq"
    expected = expected_file.read_text(encoding="utf-8")

    assert rendered == expected
