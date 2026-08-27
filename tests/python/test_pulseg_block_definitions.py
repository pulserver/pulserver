"""What makes two blocks the same block definition.

A block definition is the deduplicated structure the interpreter prepares:
duration, RF, the three gradients, and the ADC. Instances of it carry only
what a playout may vary -- amplitudes, phases, rotations, and whether the ADC
acquires this time round.

The ADC is the awkward one, because a block that acquires and the dummy shot
that does not are written as different blocks in the file and are the same
block to the scanner. Three separate lines fall out of that, and these name
them: the repeating unit is recognised from the definition with its ADC left
out; a segment is split by the readouts its repetitions actually play,
because a prepared segment binds one receive filter per block position; and
a shot that acquires nothing joins whichever of those segments it likes.
"""

from __future__ import annotations

import pulserver.pypulseq as pp

SYSTEM = pp.Opts()

#: Two ADC events covering the same span of the same block: 128 samples at
#: 20 us and 64 at 40 us both run 2.56 ms from the same delay.
DWELL_A, SAMPLES_A = 20e-6, 128
DWELL_B, SAMPLES_B = 40e-6, 64


def build(pattern: str) -> pp.Sequence:
    """One TR per character: ``a``/``b`` acquire with that ADC, ``-`` does not."""
    seq = pp.Sequence(system=SYSTEM)
    rf = pp.make_block_pulse(flip_angle=0.1, duration=1e-3, system=SYSTEM)
    gx = pp.make_trapezoid(
        channel="x", flat_area=1000, flat_time=2.56e-3, system=SYSTEM
    )
    adcs = {
        "a": pp.make_adc(
            num_samples=SAMPLES_A, dwell=DWELL_A, delay=gx.rise_time, system=SYSTEM
        ),
        "b": pp.make_adc(
            num_samples=SAMPLES_B, dwell=DWELL_B, delay=gx.rise_time, system=SYSTEM
        ),
    }
    for kind in pattern:
        seq.add_block(rf)
        if kind == "-":
            seq.add_block(gx)
        else:
            seq.add_block(gx, adcs[kind])
    return seq


def test_a_non_acquiring_instance_shares_the_definition_of_the_one_that_acquires():
    # The dummy shot leading the train is the same two blocks as the shots
    # after it, so the repeating unit is two blocks and not the whole train.
    assert build("-aaa").tr_size == build("aaaa").tr_size == 2


def test_a_block_position_that_cycles_readouts_still_repeats_every_shot():
    assert build("abab").tr_size == build("aaaa").tr_size == 2
    assert build("abababab").tr_size == 2


def test_two_readouts_at_one_block_position_are_two_segments():
    assert build("abab").num_segments == 2
    assert build("aaaa").num_segments == 1


def test_a_dummy_among_two_readouts_joins_one_of_them():
    # It acquires nothing, so neither readout's filter is used for it and
    # both segments play its gradients and RF identically.
    assert build("-aba").num_segments == build("abab").num_segments == 2
    assert build("-aba").tr_size == 2
