"""The pTx prescan tour: every parallel-transmit pulse played at its largest channel drive."""

import numpy as np
import pypulseqpp as pp
import pytest

from pulserver import ir
from pulserver.ir import Prescan, WaveBudget

SYSTEM = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
BLOCKS = 5  # per repetition: preparation, delay, excitation, readout, delay
PREPARATION = [0.2, 0.5, 0.3, 1.0, 0.4]
EXCITATION = [0.6, 1.0, 0.8, 0.4, 0.2]


def _ptx(lead, scale):
    """A two-channel pulse led by ``lead``, the other at a third of it, at ``scale``."""
    signal = np.tile(np.hanning(100) * 60.0 * scale, (2, 1)).astype(complex)
    signal[lead] *= 3.0 * np.exp(0.3j)
    return pp.make_ptx_pulse(
        signal, dwell=SYSTEM.rf_raster_time, system=SYSTEM, use="excitation"
    )


def _converted(tmp_path, seq, name="tour.seq"):
    path = tmp_path / name
    seq.write(str(path))
    ir.convert(path, SYSTEM)
    return path


def _prepared(tmp_path, calibration=2):
    """Repetitions of a pTx preparation with no readout, then a pTx excitation and its readout."""
    adc = pp.make_adc(64, duration=2e-3, system=SYSTEM)
    seq = pp.Sequence(SYSTEM)
    for prepare, excite in zip(PREPARATION, EXCITATION, strict=True):
        seq.add_block(_ptx(0, prepare))
        seq.add_block(pp.make_delay(2e-3))
        seq.add_block(_ptx(1, excite))
        seq.add_block(adc)
        seq.add_block(pp.make_delay(5e-3))
    seq.set_definition("NumGainCalibrationReadouts", calibration)
    return _converted(tmp_path, seq)


def _holds(tour, block):
    first, blocks = tour["first_position"], tour["num_blocks"]
    return np.flatnonzero((first <= block) & (block < first + blocks))


def test_the_tour_plays_each_ptx_pulse_where_it_is_largest_readout_or_not(tmp_path):
    tour = ir.prescan_tour(_prepared(tmp_path))
    preparation = BLOCKS * int(np.argmax(PREPARATION))
    excitation = BLOCKS * int(np.argmax(EXCITATION)) + 2

    assert tour["subsequence"].size == 2
    assert _holds(tour, preparation).size == 1
    assert _holds(tour, excitation).size == 1
    assert (tour["channels"] == 2).all()
    assert (np.diff(tour["first_position"]) > 0).all()


def test_a_visit_is_a_segment_instance_the_scan_plays(tmp_path):
    path = _prepared(tmp_path)
    tour = ir.prescan_tour(path)
    blocks = ir.playout(path, WaveBudget(10**6, 4.0))["blocks"]
    starts = blocks["first_position"][blocks["position"] == 0]

    for first, count, duration in zip(
        tour["first_position"], tour["num_blocks"], tour["duration_us"], strict=True
    ):
        assert first in starts
        played = blocks["first_position"] == first
        assert played.sum() == count
        assert blocks["duration_us"][played].sum() == duration


def test_single_channel_rf_has_no_tour(tmp_path):
    seq = pp.Sequence(SYSTEM)
    for _ in range(3):
        seq.add_block(pp.make_block_pulse(0.2, duration=1e-3, system=SYSTEM))
        seq.add_block(pp.make_adc(64, duration=2e-3, system=SYSTEM))
        seq.add_block(pp.make_delay(5e-3))

    assert ir.prescan_tour(_converted(tmp_path, seq))["subsequence"].size == 0


def test_a_shimmed_pulse_is_visited_where_each_channel_is_driven_hardest(tmp_path):
    seq = pp.Sequence(SYSTEM)
    shots = [((1.0, 0.2), 1.0), ((0.2, 1.0), 0.8), ((1.0, 0.2), 0.5)]
    for weights, flip in shots:
        seq.add_block(
            pp.make_block_pulse(flip, duration=1e-3, system=SYSTEM),
            pp.make_rf_shim(np.array(weights, dtype=complex)),
        )
        seq.add_block(pp.make_adc(64, duration=2e-3, system=SYSTEM))
        seq.add_block(pp.make_delay(5e-3))
    tour = ir.prescan_tour(_converted(tmp_path, seq))

    assert tour["subsequence"].size == 2
    assert _holds(tour, 0).size == 1
    assert _holds(tour, 3).size == 1
    assert _holds(tour, 6).size == 0
    assert (tour["channels"] == 1).all()


@pytest.mark.parametrize("mode", ["played", "acquired"])
def test_the_prescan_plays_the_tour_before_its_calibration(tmp_path, mode):
    path = _prepared(tmp_path, calibration=2)
    tour = ir.prescan_tour(path)
    blocks = ir.playout(path, WaveBudget(10**6, 4.0), prescan=Prescan(tour=mode))[
        "blocks"
    ]
    toured = blocks["visit"] >= 0
    count = int(toured.sum())

    assert count == tour["num_blocks"].sum()
    assert toured[:count].all()
    np.testing.assert_array_equal(
        np.unique(blocks["first_position"][toured]), tour["first_position"]
    )
    acquired = int(blocks["adc"][toured].sum())
    assert acquired == (tour["readouts"].sum() if mode == "acquired" else 0)
    assert blocks["adc"][~toured].sum() == 2
    assert blocks["first_position"][~toured][0] == 0
