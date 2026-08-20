"""What makes two blocks the same block definition.

A block definition is the deduplicated structure the interpreter prepares:
duration, RF, the three gradients, and the ADC. Instances of it carry only
what a playout may vary -- amplitudes, phases, rotations, and whether the ADC
acquires this time round.

The ADC is the awkward one, because a block that acquires and the dummy shot
that does not are written as different blocks in the file and are the same
block to the scanner. These name where that line falls.
"""

from __future__ import annotations

import pytest

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


def test_two_adc_events_at_one_block_position_are_two_definitions():
    # Same timing, different readouts: a structural difference, not a
    # per-instance parameter, so the repeating unit spans both.
    assert build("abab").tr_size == 4


def test_a_non_acquiring_instance_among_two_readouts_is_refused_by_name():
    # Which of the two the dummy stands in for is written nowhere, and
    # guessing would misdescribe the readout to the reconstruction.
    seq = build("-aba")
    with pytest.raises(RuntimeError, match="ADC definition"):
        _ = seq.tr_size
