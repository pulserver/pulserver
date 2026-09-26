"""Where a playout holds the waves, and the samples it loads into them."""

import itertools
import math

import numpy as np
import pypulseqpp as pp
import pytest

from pulserver import ir
from pulserver.ir import WaveBudget

SYSTEM = pp.Opts(max_grad=30, grad_unit="mT/m", max_slew=120, slew_unit="T/m/s", B0=3.0)
# A playout raster other than the file's, as a scanner's may be.
RASTER_US = 4.0
SPOKES = 8
LOTS = 10**6


def _radial(path, spokes=SPOKES, rotate=True):
    """A spoke per repetition, turned about z by its own angle."""
    gx = pp.make_trapezoid("x", flat_area=64 / 0.22, flat_time=3.2e-3, system=SYSTEM)
    prephaser = pp.make_trapezoid("x", area=-gx.area / 2, duration=1e-3, system=SYSTEM)
    adc = pp.make_adc(64, duration=3.2e-3, delay=gx.rise_time, system=SYSTEM)
    pulse = pp.make_block_pulse(
        math.pi / 12, duration=1e-3, use="excitation", system=SYSTEM
    )
    seq = pp.Sequence(SYSTEM)
    for spoke in range(spokes):
        turn = [pp.make_rotation(math.pi * (spoke + 0.5) / spokes)] if rotate else []
        seq.add_block(pulse)
        seq.add_block(prephaser, *turn)
        seq.add_block(gx, adc, *turn)
        seq.add_block(pp.make_delay(2e-3))
    seq.write(str(path))
    ir.convert(path, SYSTEM)
    return path


@pytest.fixture(name="radial")
def radial_fixture(tmp_path):
    return _radial(tmp_path / "radial.seq")


def _plan(path, max_samples=LOTS, **loading):
    return ir.plan_waves(path, WaveBudget(max_samples, RASTER_US, **loading))


def _tiles(regions, axis):
    """Whether the regions holding ``axis`` fill its memory from 0 without overlap."""
    spans = sorted(
        (r["offset"][axis], r["offset"][axis] + r["samples"])
        for r in regions
        if r["offset"][axis] >= 0
    )
    starts = [start for start, _ in spans]
    return not spans or starts == [0] + [stop for _, stop in spans[:-1]]


def test_a_scan_without_rotations_holds_no_waves(tmp_path):
    path = _radial(tmp_path / "cartesian.seq", rotate=False)
    plan = _plan(path, max_samples=0)
    assert plan["mode"] == "none"
    assert plan["samples"] == (0, 0, 0)


def test_every_wave_is_held_at_once_where_the_memory_affords_it(radial):
    plan = _plan(radial)
    assert plan["mode"] == "resident"
    assert plan["samples"] == plan["resident_samples"]
    (waves,) = plan["waves"]
    for axis in range(3):
        assert _tiles(waves, axis)
        assert plan["samples"][axis] == sum(
            r["samples"] for r in waves if r["offset"][axis] >= 0
        )
    # Spokes turned about z drive x and y alone.
    assert plan["samples"][2] == 0


def test_waves_that_do_not_fit_at_once_stream_through_two_slots_per_position(radial):
    resident = _plan(radial)
    plan = _plan(radial, max(resident["streamed_samples"]))
    assert plan["mode"] == "streamed"
    assert plan["samples"] == plan["streamed_samples"]
    assert all(
        s < r
        for s, r in zip(
            plan["samples"][:2], resident["resident_samples"][:2], strict=True
        )
    )
    slots = [half for positions in plan["slots"] for pair in positions for half in pair]
    for axis in range(3):
        assert _tiles(slots, axis)
    for positions in plan["slots"]:
        for first, second in positions:
            assert first["samples"] == second["samples"]
            assert first["start_us"] == second["start_us"]


def test_a_memory_that_holds_neither_layout_is_refused(radial):
    resident = _plan(radial)
    with pytest.raises(ValueError, match="fit waveform memory"):
        _plan(radial, max(resident["streamed_samples"]) - 1)


def _instances(played, summary):
    """Each segment instance in play order: its segment and its duration, in µs."""
    sizes = [segment["num_blocks"] for segment in summary["segments"]]
    instances, block = [], 0
    while block < played["segment"].size:
        segment = int(played["segment"][block])
        stop = block + sizes[segment]
        instances.append((segment, float(played["duration_us"][block:stop].sum())))
        block = stop
    return instances


def test_each_instance_loads_its_waves_while_the_one_before_it_plays(radial):
    rate, headroom = 0.05, 0.5
    plan = _plan(
        radial,
        max(_plan(radial)["streamed_samples"]),
        load_us_per_sample=rate,
        headroom=headroom,
    )
    assert plan["loading_checked"]

    def loading(segment):
        return rate * sum(
            first["samples"] * sum(offset >= 0 for offset in first["offset"])
            for first, _ in plan["slots"][segment]
        )

    instances = _instances(
        ir.play(radial), ir.summary(radial, SYSTEM, cache_ext=".pseg")
    )
    spare = min(
        headroom * before - loading(segment)
        for (_, before), (segment, _) in itertools.pairwise(instances)
    )
    assert plan["least_spare_us"] == pytest.approx(spare, rel=1e-6)


def test_an_instance_that_cannot_load_while_the_one_before_it_plays_is_refused(
    radial,
):
    fit = _plan(radial)
    with pytest.raises(ValueError, match="longer to load"):
        _plan(radial, max(fit["streamed_samples"]), load_us_per_sample=1e3)


def test_a_resident_layout_is_not_held_to_a_load_rate(radial):
    plan = _plan(radial, load_us_per_sample=1e3)
    assert plan["mode"] == "resident"
    assert not plan["loading_checked"]


def _played_shape(played, block, axis, centres):
    """The wave a block plays on one axis over its amplitude, at ``centres``."""
    start, stop = played["gradient_span"][block, axis]
    amplitude = played["gradient_hz_per_m"][block, axis]
    shape = played["gradient_waveform_hz_per_m"][start:stop] / amplitude
    return np.interp(
        centres, played["gradient_time_us"][start:stop], shape, left=0.0, right=0.0
    )


def test_a_sampled_wave_is_the_played_wave_at_the_raster_centres(radial):
    budget = WaveBudget(LOTS, RASTER_US)
    plan = ir.plan_waves(radial, budget)
    played = ir.play(radial, waveforms=True)
    rotated = np.flatnonzero(played["wave"] >= 0)
    assert rotated.size
    for block in rotated:
        wave = (int(played["subsequence"][block]), int(played["wave"][block]))
        region = plan["waves"][wave[0]][wave[1]]
        sampled = ir.sample_wave(radial, wave, region, budget)
        centres = region["start_us"] + (np.arange(region["samples"]) + 0.5) * RASTER_US
        for axis in np.flatnonzero(played["gradient_hz_per_m"][block] != 0.0):
            np.testing.assert_allclose(
                sampled[axis], _played_shape(played, block, axis, centres), atol=1e-5
            )


@pytest.mark.parametrize(
    "fields",
    [
        (LOTS, 0.0),
        (LOTS, RASTER_US, 0.0, 0.0),
        (-1, RASTER_US),
        (LOTS, RASTER_US, -1.0),
    ],
)
def test_a_budget_without_a_raster_or_headroom_or_with_negative_room_is_refused(fields):
    with pytest.raises(ValueError):
        WaveBudget(*fields)
