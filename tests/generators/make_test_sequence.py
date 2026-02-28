"""Generate a dummy test sequence with all kinds of the events."""

from types import SimpleNamespace

import numpy as np

import pypulseq as pp


def make_test_sequence(write: bool = False) -> pp.Sequence:
    """Create test sequence for ANSI C89 Pulseq implementation."""
    seq = pp.Sequence()

    # real valued RF event
    rf_shim = SimpleNamespace(
        type="rf_shim", shim_vector=np.ones(8, dtype=complex)
    )  # t rivial rf shim
    rf_real = pp.make_sinc_pulse(flip_angle=0.5 * np.pi)

    # complex valued RF event
    rf_cplx = pp.make_adiabatic_pulse(pulse_type="hypsec")

    # ADC event (no phase mod)
    adc = pp.make_adc(num_samples=128, dwell=10e-6)

    # ADC event (with phase mod)
    phase_mod = np.zeros(128)
    adc_phase = pp.make_adc(num_samples=128, dwell=10e-6, phase_modulation=phase_mod)

    # Trapezoid gradient (x)
    gx_trap = pp.make_trapezoid(channel="x", amplitude=10, duration=1e-3)

    # Trapezoid gradient (y)
    gy_trap = pp.make_trapezoid(channel="y", area=5, duration=1e-3)

    # Trapezoid gradient (z)
    gz_trap = pp.make_trapezoid(channel="z", area=5, duration=1e-3)

    # Arbitrary gradient (x)
    gx_arb = pp.make_arbitrary_grad(channel="x", waveform=np.array([0, 0.1, 1, 0.1, 0]))

    # Arbitrary gradient (y)
    gy_arb = pp.make_arbitrary_grad(channel="y", waveform=np.array([0, 0.1, 1, 0.1, 0]))

    # Arbitrary gradient (z)
    gz_arb = pp.make_arbitrary_grad(channel="z", waveform=np.array([0, 0.1, 1, 0.1, 0]))

    # Extended trapezoid (x)
    gx_ext = pp.make_extended_trapezoid(
        channel="x",
        amplitudes=np.array([0, 0.1, 1, 0.1, 0]),
        times=np.array([0, 10e-6, 20e-6, 2e-3 + 20e-6, 2e-3 + 30e-6]),
    )

    # Extended trapezoid (y)
    gy_ext = pp.make_extended_trapezoid(
        channel="y",
        amplitudes=np.array([0, 0.1, 1, 0.1, 0]),
        times=np.array([0, 10e-6, 20e-6, 2e-3 + 20e-6, 2e-3 + 30e-6]),
    )

    # Extended trapezoid (z)
    gz_ext = pp.make_extended_trapezoid(
        channel="z",
        amplitudes=np.array([0, 0.1, 1, 0.1, 0]),
        times=np.array([0, 10e-6, 20e-6, 2e-3 + 20e-6, 2e-3 + 30e-6]),
    )

    # Add all events to sequence
    seq.add_block(rf_real, rf_shim)
    seq.add_block(rf_cplx)
    seq.add_block(adc_phase)

    # All label types (SET, INC)
    for label in pp.get_supported_labels():
        seq.add_block(adc, pp.make_label(label, "SET", 1))
    for label in pp.get_supported_labels()[:10]:
        seq.add_block(adc, pp.make_label(label, "INC", 1))
    seq.add_block(adc, pp.make_label("TRID", "INC", 1))

    seq.add_block(gx_trap)
    seq.add_block(gy_trap)
    seq.add_block(gz_trap)
    seq.add_block(gx_arb, pp.make_rotation(np.eye(3, dtype=float)))
    seq.add_block(gy_arb, pp.make_rotation(np.eye(3, dtype=float)))
    seq.add_block(gz_arb, pp.make_rotation(np.eye(3, dtype=float)))
    seq.add_block(gx_ext)
    seq.add_block(gy_ext)
    seq.add_block(gz_ext)

    # Trigger event
    seq.add_block(pp.make_trigger(channel="physio1"))
    seq.add_block(pp.make_trigger(channel="physio2"))
    seq.add_block(pp.make_digital_output_pulse(channel="osc0"))
    seq.add_block(pp.make_digital_output_pulse(channel="osc1"))
    seq.add_block(pp.make_digital_output_pulse(channel="ext1"))

    # Soft delays (TE, TR, TI, ESP, RECTIME, T2PREP, TE2, TR2)
    for hint in ("TE", "TR", "TI", "ESP", "RECTIME", "T2PREP", "TE2", "TR2"):
        seq.add_block(pp.make_soft_delay(hint=hint))

    # Pure (hard) delay
    seq.add_block(pp.make_delay(1.0))

    # Write to file
    if write:
        seq.write("seq2.seq")

    return seq


if __name__ == "__main__":
    make_test_sequence(write=True)
