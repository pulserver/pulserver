"""How a subsequence's repetitions play in time, and the span its RF is averaged over."""

from pathlib import Path

import numpy as np
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


def test_a_periodic_scan_is_averaged_over_one_hyper_repetition(tmp_path):
    # Packets of five pulses every 0.30416 s, for 12 s: one packet is the span.
    w = subsequence(packets(tmp_path / "packet.seq", [4e-5] * 4 + [0.3], 40))[
        "rf_window"
    ]
    assert (w["plays"], w["counts"], w["start_us"]) == (5, [5], 0.0)
    assert w["span_us"] == pytest.approx(4 * 1040.0 + 301000.0)


def test_a_long_repetition_is_averaged_over_itself_not_a_window(tmp_path):
    # An inversion shot of 3.4 s: any 10 s span starting at a play holds three
    # inversions and two trains and a half, more than the shot's mean.
    inversion = pp.make_block_pulse(np.pi, duration=1e-2, system=SYSTEM)
    sequence = pp.Sequence(SYSTEM)
    for _ in range(6):
        sequence.add_block(inversion)
        sequence.add_block(pp.make_delay(0.99))
        for _ in range(100):
            sequence.add_block(PULSE)
            sequence.add_block(pp.make_delay(0.009))
        sequence.add_block(pp.make_delay(1.4))
    path = tmp_path / "shot.seq"
    sequence.write(path)
    x = subsequence(path)
    assert x["tr_timing"]["hyper_us"] == pytest.approx(3.4e6)
    w = x["rf_window"]
    assert (w["plays"], w["counts"], w["start_us"]) == (101, [1] * 101, 0.0)
    assert w["span_us"] == pytest.approx(3.4e6)


def test_a_scan_shorter_than_the_window_plays_whole_inside_it(tmp_path):
    x = subsequence(_played(tmp_path / "short.seq", [0.01, 0.02, 0.03, 0.04]))
    w = x["rf_window"]
    assert (w["plays"], w["start_us"]) == (4, 0.0)
    assert w["span_us"] == pytest.approx(x["tr_timing"]["played_us"])


def test_a_dense_stretch_is_found_and_costed_over_the_window_alone(tmp_path):
    # 12 s of sparse pulses, then 18 s of dense ones every 30 ms, each pulse
    # led by a spoiler in its segment: the densest 10 s starts at the first
    # dense pulse, 1 ms into its instance, and holds 334 pulses. Plays are
    # costed by count, so the span is not widened to whole instances.
    spoiler = pp.make_trapezoid("z", area=1000, duration=1e-3, system=SYSTEM)
    sequence = pp.Sequence(SYSTEM)
    for wait, repeats in ((0.998, 12), (0.028, 600)):
        for _ in range(repeats):
            sequence.add_block(spoiler)
            sequence.add_block(PULSE)
            sequence.add_block(pp.make_delay(wait))
    path = tmp_path / "dense_tail.seq"
    sequence.write(path)
    w = subsequence(path)["rf_window"]
    assert w["plays"] == 334
    assert w["start_us"] == pytest.approx(12.001e6)
    assert w["span_us"] == pytest.approx(10.0e6)


def _played(path, waits):
    """One pulse and one pure delay per entry of ``waits``."""
    return packets(path, waits, 1)


@pytest.mark.parametrize(
    ("waits", "copies"),
    [
        ([4e-5] * 10, 1),
        (([4e-5] * 4 + [0.3]) * 8, 5),
        ([4e-5, 0.3] * 6, 2),
        ([0.01, 0.02, 0.03, 0.04], 4),
        (([4e-5] * 4 + [0.3]) * 3 + [4e-5] * 2, 5),
    ],
    ids=[
        "uniform",
        "pad-on-last-of-5",
        "alternating",
        "aperiodic",
        "last-period-cut-short",
    ],
)
def test_the_hyper_repetition_is_the_shortest_period_of_the_copies(
    tmp_path, waits, copies
):
    t = subsequence(_played(tmp_path / "hyper.seq", waits))["tr_timing"]
    assert t["num_copies"] == len(waits)
    assert t["hyper_copies"] == copies
    assert t["hyper_us"] == pytest.approx(sum(1000.0 + 1e6 * w for w in waits[:copies]))


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
