"""The time a chain plays under a sequencer's segment timing (``ir.played_time``)."""

from pathlib import Path

import pypulseqpp as pp
import pytest
from pypulseqpp import sequences

from pulserver import ir

SYSTEM = pp.Opts(
    max_grad=40.0,
    grad_unit="mT/m",
    max_slew=170.0,
    slew_unit="T/m/s",
    B0=3.0,
    rf_raster_time=1e-6,
    grad_raster_time=1e-5,
    adc_raster_time=1e-7,
    block_duration_raster=1e-5,
)
# A 7 T's: 250 us after every segment, an active one 4 us past its blocks, a
# pure delay's wait never under 500 us, on a 4 us raster.
TIMING = ir.SequencerTiming(gap_us=250, tail_us=4, min_wait_us=500, raster_us=4)


def _readouts(path, delay):
    """Three repetitions of a readout, each followed by ``delay`` s of wait or by nothing."""
    adc = pp.make_adc(num_samples=64, duration=3.2e-3, system=SYSTEM)
    scan = pp.Sequence(SYSTEM)
    for _ in range(3):
        scan.add_block(adc)
        if delay:
            scan.add_block(pp.make_delay(delay))
    return Path(sequences.write(path, [scan])[0])


def test_a_closing_wait_gives_the_segment_gaps_back(tmp_path):
    played = ir.played_time(_readouts(tmp_path / "wait.seq", 5e-3), SYSTEM, TIMING)
    assert played.blocks == pytest.approx(3 * (3.2e-3 + 5e-3))
    assert played.extra == pytest.approx(0.0, abs=1e-9)


def test_back_to_back_segments_play_the_gaps_on_top(tmp_path):
    played = ir.played_time(_readouts(tmp_path / "packed.seq", 0.0), SYSTEM, TIMING)
    gaps = played.extra / ((TIMING.gap_us + TIMING.tail_us) * 1e-6)
    assert gaps == pytest.approx(round(gaps)) and round(gaps) >= 1


def test_a_wait_too_short_to_give_the_gaps_back_plays_its_least_wait(tmp_path):
    played = ir.played_time(_readouts(tmp_path / "short.seq", 6e-4), SYSTEM, TIMING)
    assert played.extra > 0.0


def test_the_timing_refuses_negative_times():
    with pytest.raises(ValueError):
        ir.SequencerTiming(gap_us=-1)
    with pytest.raises(ValueError):
        ir.SequencerTiming(gap_us=250, raster_us=0)
