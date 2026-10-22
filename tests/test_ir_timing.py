"""How a subsequence's repetitions play in time, and its densest RF span."""

from pathlib import Path

import pypulseqpp as pp
import pytest

from pulserver.ir import convert, summary

FIXTURES = Path(__file__).parent / "fixtures" / "sequences"
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
PULSE = pp.make_block_pulse(0.1, duration=1e-3, system=SYSTEM)


def packets(path, waits, repeats):
    """One pulse and one pure delay per repetition, the delays cycling ``waits``.

    Every repetition plays the same blocks, so the repetition is one pulse and
    its delay whatever the delay's length, as a multi-slice packet whose last
    slice carries the packet's pad.
    """
    sequence = pp.Sequence(SYSTEM)
    for _ in range(repeats):
        for wait in waits:
            sequence.add_block(PULSE)
            sequence.add_block(pp.make_delay(wait))
    sequence.write(path)
    return path


def subsequence(path, **kwargs):
    return summary(path, SYSTEM, **kwargs)["subsequences"][0]


def test_every_repetition_is_timed_at_its_own_delay(tmp_path):
    x = subsequence(packets(tmp_path / "packet.seq", [4e-5, 4e-5, 4e-5, 4e-5, 0.3], 10))
    t = x["tr_timing"]
    assert (x["tr_size"], t["num_copies"]) == (2, 50)
    # The first repetition is a short one, and so are three in four after it.
    assert x["tr_duration_us"] == pytest.approx(t["shortest_copy_us"])
    assert t["shortest_copy_us"] == pytest.approx(1040.0)
    assert t["longest_copy_us"] == pytest.approx(301000.0)
    assert t["played_us"] == pytest.approx(10 * (4 * 1040.0 + 301000.0))


def test_a_long_first_repetition_is_not_the_shortest(tmp_path):
    t = subsequence(packets(tmp_path / "long_first.seq", [0.3, 4e-5], 5))["tr_timing"]
    assert t["shortest_copy_us"] == pytest.approx(1040.0)
    assert t["longest_copy_us"] == pytest.approx(301000.0)


def test_the_window_counts_the_densest_ten_seconds(tmp_path):
    # Packets of five pulses every 0.30416 s: 33 packets start within 10 s,
    # the 33rd at 9.73 s, so the densest span holds 33 * 5 plays.
    w = subsequence(packets(tmp_path / "packet.seq", [4e-5] * 4 + [0.3], 40))[
        "rf_window"
    ]
    assert w["plays"] == 33 * 5
    assert w["counts"] == [33 * 5]


def test_a_scan_shorter_than_the_window_plays_whole_inside_it(tmp_path):
    x = subsequence(packets(tmp_path / "short.seq", [4e-5] * 4 + [0.3], 3))
    assert x["rf_window"]["plays"] == 15
    assert x["rf_window"]["start_us"] == 0.0


def test_a_dense_stretch_is_found_wherever_it_plays(tmp_path):
    # 12 s of sparse pulses, then 10 s of dense ones: the window lands on the
    # dense stretch, though the scan's mean density is far lower.
    sequence = pp.Sequence(SYSTEM)
    for _ in range(12):
        sequence.add_block(PULSE)
        sequence.add_block(pp.make_delay(0.999))
    for _ in range(500):
        sequence.add_block(PULSE)
        sequence.add_block(pp.make_delay(0.019))
    path = tmp_path / "dense_tail.seq"
    sequence.write(path)
    w = subsequence(path)["rf_window"]
    assert w["plays"] == 500
    assert w["start_us"] == pytest.approx(12 * 1.0e6)


@pytest.mark.parametrize("name", sorted(p.name for p in FIXTURES.glob("*.seq")))
def test_every_fixture_plays_for_its_copies(name):
    t = subsequence(FIXTURES / name)["tr_timing"]
    assert t["num_copies"] >= 1
    assert 0.0 < t["shortest_copy_us"] <= t["longest_copy_us"] <= t["played_us"]


def test_the_cache_times_the_scan_as_the_conversion_does(tmp_path):
    path = packets(tmp_path / "cached.seq", [4e-5, 0.3], 4)
    convert(path, SYSTEM)
    cached = subsequence(path, cache_ext=".pseg")
    fresh = subsequence(path)
    assert cached["tr_timing"] == fresh["tr_timing"]
    assert cached["rf_window"] == fresh["rf_window"]
