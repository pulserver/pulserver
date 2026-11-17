"""Small synthetic sequences for tests that need one readout geometry exactly."""

import math

import pypulseqpp as pp

SAMPLES = 32
DELTA_K = 5.0  # 1/m: a 0.2 m field of view


def add_readout(seq, *labels, rotation=None, moving=True):
    """Append a symmetric readout along x, rewound on both sides so k returns to zero."""
    system = pp.Opts()
    gx = pp.make_trapezoid(
        "x", flat_area=SAMPLES * DELTA_K, flat_time=3.2e-3, system=system
    )
    adc = pp.make_adc(
        num_samples=SAMPLES, duration=3.2e-3, delay=gx.rise_time, system=system
    )
    if not moving:
        seq.add_block(adc, *labels)
        return
    rewinder = pp.make_trapezoid("x", area=-gx.area / 2, duration=1e-3, system=system)
    extra = () if rotation is None else (rotation,)
    seq.add_block(rewinder, *extra)
    seq.add_block(gx, adc, *labels, *extra)
    seq.add_block(rewinder, *extra)


def write_swept_spokes(path, system, spokes=16):
    """Spokes turned about z, each with a y lobe of another shape under its x lobe.

    Each turn mixes two shapes on each board, so every repetition plays a
    waveform of its own there, and two slots per position hold less than all of
    them.
    """
    gx = pp.make_trapezoid("x", flat_area=64 / 0.22, flat_time=3.2e-3, system=system)
    gy = pp.make_trapezoid(
        "y", area=gx.area / 8, duration=pp.calc_duration(gx), system=system
    )
    adc = pp.make_adc(64, duration=3.2e-3, delay=gx.rise_time, system=system)
    pulse = pp.make_block_pulse(
        math.pi / 12, duration=1e-3, use="excitation", system=system
    )
    seq = pp.Sequence(system)
    for spoke in range(spokes):
        seq.add_block(pulse)
        seq.add_block(gx, gy, adc, pp.make_rotation(math.pi * (spoke + 0.5) / spokes))
        seq.add_block(pp.make_delay(2e-3))
    seq.write(str(path))
