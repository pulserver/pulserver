"""The rule by which a repetition's blocks become the units a machine plays."""

import math
import shutil
from pathlib import Path

import numpy as np
import pypulseqpp as pp
import pytest

from pulserver.ir import (
    Grouping,
    SequencerTiming,
    convert,
    gradient_stretch,
    play,
    played_time,
    summary,
)

SYSTEM = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
FIXTURES = Path(__file__).parent / "fixtures" / "sequences"
SEQUENCES = (
    "gre_2d_3sl.seq",
    "epi_2d_main.seq",
    "mprage_stack_of_spirals_3d.seq",
    "zte_3d.seq",
)


def _copied(name, tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / name
    shutil.copy(FIXTURES / name, path)
    return path


def _grouped(path, grouping):
    """Return the units a grouping makes, and the gradient they play over time."""
    convert(path, SYSTEM, grouping=grouping, cache_ext=".pseg")
    return (
        summary(path, SYSTEM, cache_ext=".pseg")["num_segments"],
        gradient_stretch(path, 0, "heaviest_repetition", cache_ext=".pseg"),
    )


@pytest.mark.parametrize("name", SEQUENCES)
def test_the_default_grouping_is_what_a_conversion_does_unasked(name, tmp_path):
    """Stating the rule a machine already wanted changes nothing it is given."""
    stated = _copied(name, tmp_path)
    unasked = _copied(name, tmp_path / "unasked")
    convert(stated, SYSTEM, grouping=Grouping(), cache_ext=".pseg")
    convert(unasked, SYSTEM, cache_ext=".pseg")
    assert (
        stated.with_suffix(".pseg").read_bytes()
        == unasked.with_suffix(".pseg").read_bytes()
    )


def test_keeping_a_units_edge_delays_leaves_fewer_units(tmp_path):
    """A machine that pays switching time per unit wants as few as it can have."""
    path = _copied("gre_2d_3sl.seq", tmp_path)
    split, _ = _grouped(path, Grouping())
    whole, _ = _grouped(path, Grouping(split_edge_delays=False))
    assert whole < split


def test_a_machine_that_may_begin_a_unit_under_a_gradient_gets_more_units(tmp_path):
    """Where a boundary may fall is the machine's to state, not this library's.

    A radial shell rests its gradients once a shot, so a boundary only where
    they rest leaves the shot one unit; a machine that can set a gradient on
    entry has its pick of the positions between.
    """
    path = _copied("zte_3d.seq", tmp_path)
    resting, _ = _grouped(path, Grouping())
    anywhere, _ = _grouped(path, Grouping(boundary_gradient_hz_per_m=1e9))
    assert anywhere > resting


@pytest.mark.parametrize("name", SEQUENCES)
@pytest.mark.parametrize(
    "grouping",
    [
        Grouping(split_edge_delays=False),
        Grouping(boundary_gradient_hz_per_m=1e9),
        Grouping(split_by_pulses=False, split_by_readouts=False),
    ],
    ids=["edge-delays-kept", "boundary-anywhere", "unrefined"],
)
def test_a_grouping_moves_the_boundaries_and_not_the_gradient(name, grouping, tmp_path):
    """Grouping is where a unit begins; the scan it plays is the same scan.

    Checked on the stretch from the repetition of most gradient energy, which
    starts where the repetition does whatever the grouping.
    """
    path = _copied(name, tmp_path)
    _, resting = _grouped(path, Grouping())
    _, regrouped = _grouped(path, grouping)
    assert regrouped["duration_us"] == resting["duration_us"]
    assert regrouped["energy"] == pytest.approx(resting["energy"], rel=1e-9)
    for axis in ("time_us", "gx", "gy", "gz"):
        if axis in resting and axis in regrouped:
            assert np.asarray(regrouped[axis]) == pytest.approx(
                np.asarray(resting[axis]), rel=1e-6, abs=1e-6
            )


# Spans: steady-state trains of eleven repetitions of three blocks each, the
# phase encoding and the RF phase changing from one to the next.
TRAIN = 11
BLOCKS = 3
SLICE = 1 + TRAIN * BLOCKS
# A 7 T's sequencer: 250 us after every segment and an active one 4 us past its
# blocks, with no wait in the train to give either back.
TIMING = SequencerTiming(gap_us=250, tail_us=4, min_wait_us=500, raster_us=4)


def _trains(
    tmp_path,
    *,
    slices=1,
    opened=True,
    wait=0.0,
    lead=(),
    declared=None,
    rotation=None,
    dummies=(),
):
    """``slices`` trains of ``TRAIN`` repetitions each.

    With ``opened``, a pulse of half the flip angle opens each train, so a
    train is what the scan repeats; without, a repetition is. ``lead`` events
    open each repetition, ``wait`` s follow it, and the repetitions listed in
    ``dummies`` do not acquire.
    """
    rf = pp.make_block_pulse(math.pi / 6, duration=1e-3, system=SYSTEM)
    opening = pp.make_block_pulse(math.pi / 12, duration=1e-3, system=SYSTEM)
    gx = pp.make_trapezoid("x", flat_area=64 / 0.22, flat_time=3.2e-3, system=SYSTEM)
    adc = pp.make_adc(64, duration=3.2e-3, delay=gx.rise_time, system=SYSTEM)
    encode = pp.make_trapezoid("y", area=32 / 0.22, duration=1e-3, system=SYSTEM)
    seq = pp.Sequence(SYSTEM)
    for _ in range(slices):
        if opened:
            seq.add_block(opening, pp.make_delay(2e-3))
        for index in range(TRAIN):
            rf.phase_offset = math.pi * (index % 2)
            adc.phase_offset = rf.phase_offset
            step = (index - TRAIN // 2) / TRAIN
            seq.add_block(rf, *lead)
            readout = [gx, pp.scale_grad(encode, step)]
            if index not in dummies:
                readout.append(adc)
            seq.add_block(*readout)
            events = [pp.scale_grad(encode, -step)]
            if rotation is not None:
                events.append(pp.make_rotation(rotation * (index + 1)))
            seq.add_block(*events)
            if wait:
                seq.add_block(pp.make_delay(wait))
    for name, value in (declared or {}).items():
        seq.set_definition(name, value)
    path = tmp_path / "train.seq"
    tmp_path.mkdir(parents=True, exist_ok=True)
    seq.write(str(path))
    return path


def _plays(path, grouping):
    """The blocks each segment instance of the scan holds, in play order, and what the scan plays."""
    convert(path, SYSTEM, grouping=grouping, cache_ext=".pseg")
    blocks = [
        s["num_blocks"] for s in summary(path, SYSTEM, cache_ext=".pseg")["segments"]
    ]
    played = play(path, waveforms=True)
    instances, at = [], 0
    while at < len(played["segment"]):
        instances.append(blocks[played["segment"][at]])
        at += instances[-1]
    return instances, played


def _assert_the_same_blocks(alone, spanned):
    """Everything two walks of a scan report but the segment each block is in."""
    assert set(alone) == set(spanned)
    for name in alone:
        if name != "segment":
            assert np.array_equal(
                np.asarray(alone[name]), np.asarray(spanned[name]), equal_nan=True
            ), name


def test_a_preparation_and_the_train_it_opens_play_as_one_segment_instance(tmp_path):
    """Nothing separates the opening pulse from the first repetition but a segment start."""
    path = _trains(tmp_path)
    alone, _ = _plays(path, Grouping())
    spanned, _ = _plays(path, Grouping(span_blocks=1024))
    assert alone == [1] + [BLOCKS] * TRAIN
    assert spanned == [SLICE]


def test_every_repetition_of_a_scan_plays_the_one_span(tmp_path):
    """Three slices, each an opening pulse and a train: one segment, played three times."""
    path = _trains(tmp_path, slices=3)
    instances, _ = _plays(path, Grouping(span_blocks=1024))
    assert instances == [SLICE] * 3
    assert summary(path, SYSTEM, cache_ext=".pseg")["num_segments"] == 1


def test_a_repetition_that_is_one_segment_is_not_joined_to_the_next(tmp_path):
    """A span holds the segments of one repetition, so it does not grow with their number."""
    alone, _ = _plays(_trains(tmp_path, opened=False), Grouping())
    spanned, _ = _plays(_trains(tmp_path, opened=False), Grouping(span_blocks=1024))
    assert alone == [BLOCKS] * TRAIN
    assert spanned == alone


def test_a_run_longer_than_a_span_keeps_a_segment_per_play(tmp_path):
    """A segment start inside the steady state would play one repetition time longer than the others."""
    path = _trains(tmp_path)
    alone, _ = _plays(path, Grouping())
    spanned, _ = _plays(path, Grouping(span_blocks=SLICE - 1))
    assert spanned == alone


def test_a_run_longer_than_a_span_is_cut_where_its_dummy_repetitions_start(tmp_path):
    """Repetitions that do not acquire re-establish the steady state a segment start disturbs."""
    path = _trains(tmp_path, dummies=(5, 6))
    instances, _ = _plays(path, Grouping(span_blocks=SLICE - 1))
    assert instances == [1 + 5 * BLOCKS, 6 * BLOCKS]


@pytest.mark.parametrize("slices", [1, 3])
def test_spans_move_the_boundaries_and_not_what_a_block_plays(slices, tmp_path):
    """Every block plays the same events at the same amplitudes, phases and times."""
    path = _trains(tmp_path, slices=slices, dummies=(5, 6))
    _, alone = _plays(path, Grouping())
    _, spanned = _plays(path, Grouping(span_blocks=1024))
    _, cut = _plays(path, Grouping(span_blocks=SLICE - 1))
    _assert_the_same_blocks(alone, spanned)
    _assert_the_same_blocks(alone, cut)


def test_a_train_in_one_span_plays_the_time_its_file_states(tmp_path):
    """The sequencer takes its time at every segment start: once a span."""
    path = _trains(tmp_path)
    alone = played_time(path, SYSTEM, TIMING)
    spanned = played_time(path, SYSTEM, TIMING, grouping=Grouping(span_blocks=1024))
    start = (TIMING.gap_us + TIMING.tail_us) * 1e-6
    assert alone.extra == pytest.approx((1 + TRAIN) * start)
    assert spanned.extra == pytest.approx(start)
    assert spanned.blocks == alone.blocks


def test_a_wait_ends_a_run(tmp_path):
    """A wait gives the sequencer's time back, and is played as the wait it is."""
    instances, _ = _plays(_trains(tmp_path, wait=5e-3), Grouping(span_blocks=1024))
    assert instances == [1 + BLOCKS, 1] + [BLOCKS, 1] * (TRAIN - 1)


@pytest.mark.parametrize(
    "changes",
    [
        {"lead": (pp.make_trigger("physio1", duration=5e-4),)},
        {"declared": {"EnablePmc": 1}},
        {"rotation": 0.1},
    ],
    ids=["a-trigger-waited-for", "poses-read", "a-rotation-each"],
)
def test_what_a_segment_start_is_needed_for_keeps_a_segment_per_play(changes, tmp_path):
    """A trigger starts a play, and a pose and a rotation are set at a segment start."""
    path = _trains(tmp_path, **changes)
    alone, _ = _plays(path, Grouping())
    spanned, _ = _plays(path, Grouping(span_blocks=1024))
    assert spanned == alone


@pytest.mark.parametrize("name", SEQUENCES)
def test_spans_leave_what_the_fixture_scans_play(name, tmp_path):
    path = _copied(name, tmp_path)
    _, alone = _plays(path, Grouping())
    _, spanned = _plays(path, Grouping(span_blocks=1024))
    _assert_the_same_blocks(alone, spanned)


def test_a_negative_span_is_refused():
    with pytest.raises(ValueError, match="span_blocks"):
        Grouping(span_blocks=-1)
