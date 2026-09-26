"""The two stages of a segmented playout, recorded, against what the cursor plays."""

import shutil
from pathlib import Path

import numpy as np
import pypulseqpp as pp
import pytest

from pulserver import ir
from pulserver.ir import Prescan, WaveBudget

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
# A playout raster other than the files', as a scanner's may be.
RASTER_US = 4.0
LOTS = 10**6
#: Fixtures whose waves a smaller memory streams.
WAVED = ["zte_3d.seq", "mprage_stack_of_spirals_3d.seq"]
NAMES = ["gre_2d_3sl.seq", "epi_2d_main.seq", *WAVED]


def _converted(name, directory):
    shutil.copytree(FIXTURES, directory, dirs_exist_ok=True)
    ir.convert(directory / name, SYSTEM)
    return directory / name


def _budgets(seq, ring=2):
    """The budget that holds every wave at once, and one that streams them."""
    plan = ir.plan_waves(seq, WaveBudget(LOTS, RASTER_US, slots=ring))
    budgets = {"resident": WaveBudget(LOTS, RASTER_US, slots=ring)}
    streamed = max(plan["streamed_samples"])
    if 0 < streamed < max(plan["resident_samples"]):
        budgets["streamed"] = WaveBudget(streamed, RASTER_US, slots=ring)
    return budgets


@pytest.fixture(name="played", params=NAMES, scope="module")
def played_fixture(request, tmp_path_factory):
    seq = _converted(request.param, tmp_path_factory.mktemp("playout"))
    return seq, ir.play(seq, waveforms=True)


def _layouts(seq):
    return [ir.playout(seq, budget) for budget in _budgets(seq).values()]


def test_the_scan_loop_sets_every_block_the_cursor_plays(played):
    seq, cursor = played
    for record in _layouts(seq):
        blocks = record["blocks"]
        for key in (
            "subsequence",
            "segment",
            "duration_us",
            "wave",
            "rf_amp_hz",
            "rf_phase_rad",
            "rf_freq_hz",
            "adc",
            "adc_freq_hz",
            "adc_phase_rad",
        ):
            np.testing.assert_array_equal(blocks[key], cursor[key], err_msg=key)
        waved = blocks["wave"] >= 0
        amplitude = np.where(
            waved[:, None], blocks["wave_amp_hz_per_m"], blocks["gradient_hz_per_m"]
        )
        np.testing.assert_array_equal(amplitude, cursor["gradient_hz_per_m"])
        np.testing.assert_array_equal(blocks["rotate"], 1 - cursor["norot"])


#: What each block plays, keyed as :func:`pulserver.ir.play` keys it.
WAVEFORMS = (
    "duration_us",
    "rf_use",
    "rf_delay_us",
    "rf_channels",
    "rf_center_us",
    "rf_time_us",
    "rf_waveform_hz",
    "rf_span",
    "adc_delay_us",
    "adc_dwell_ns",
    "adc_samples",
    "adc_phase_modulation_rad",
    "adc_modulation_span",
    "gradient_time_us",
    "gradient_waveform_hz_per_m",
    "gradient_span",
)


def test_every_block_plays_the_waveforms_the_cursor_plays(played):
    seq, cursor = played
    for budget in _budgets(seq).values():
        blocks = ir.playout(seq, budget, waveforms=True)["blocks"]
        for key in WAVEFORMS:
            np.testing.assert_array_equal(blocks[key], cursor[key], err_msg=key)


@pytest.mark.parametrize("name", WAVED)
def test_without_a_budget_every_wave_is_held_on_the_files_gradient_raster(
    name, tmp_path
):
    seq = _converted(name, tmp_path)
    raster = ir.summary(seq, SYSTEM, cache_ext=".pseg")["subsequences"][0][
        "grad_raster_us"
    ]
    record = ir.playout(seq)
    plan = ir.plan_waves(seq, WaveBudget(LOTS, raster))
    assert record["mode"] == "resident"
    assert record["memory_samples"] == plan["resident_samples"]
    assert record["unloaded"] == 0


def test_the_summary_states_the_gradient_raster_of_each_file(tmp_path):
    seq = _converted("zte_3d.seq", tmp_path)
    (declared,) = {1e6 * s.grad_raster_time for _, s in pp.io.read_chain(seq)}
    for summary in (
        ir.summary(seq, SYSTEM),
        ir.summary(seq, SYSTEM, cache_ext=".pseg"),
    ):
        (subsequence,) = summary["subsequences"]
        assert subsequence["grad_raster_us"] == pytest.approx(declared)


def _cursor_axis(cursor, block, axis):
    start, stop = cursor["gradient_span"][block, axis]
    return (
        cursor["gradient_time_us"][start:stop].astype(float),
        cursor["gradient_waveform_hz_per_m"][start:stop].astype(float),
    )


def test_a_wave_reads_its_own_samples_from_memory(played):
    seq, cursor = played
    for record in _layouts(seq):
        blocks = record["blocks"]
        for block in np.flatnonzero(blocks["wave"] >= 0):
            for axis in range(3):
                times, values = _cursor_axis(cursor, block, axis)
                start, stop = blocks["wave_read_span"][block, axis]
                samples = blocks["wave_read"][start:stop]
                centres = blocks["wave_start_us"][block] + RASTER_US * (
                    np.arange(samples.size) + 0.5
                )
                np.testing.assert_allclose(
                    blocks["wave_amp_hz_per_m"][block, axis] * samples,
                    np.interp(centres, times, values, left=0.0, right=0.0),
                    atol=1e-4 * max(np.abs(values).max(initial=0.0), 1.0),
                )


def test_nothing_is_read_before_it_is_loaded_or_loaded_over_what_plays(played):
    seq, _ = played
    for record in _layouts(seq):
        assert record["unloaded"] == 0
        assert record["overwrites"] == 0


# Three slots per position of the stack of spirals take more memory than
# every wave at once, which is then how a playout holds them.
@pytest.mark.parametrize(
    ("name", "ring"), [(name, 2) for name in WAVED] + [("zte_3d.seq", 3)]
)
def test_the_instances_of_a_segment_play_the_slots_of_its_ring_in_turn(
    name, ring, tmp_path
):
    seq = _converted(name, tmp_path)
    record = ir.playout(seq, _budgets(seq, ring)["streamed"])
    blocks = record["blocks"]
    assert record["mode"] == "streamed"
    assert record["slots"] == ring
    firsts = np.flatnonzero(blocks["position"] == 0)
    for segment in np.unique(blocks["segment"]):
        mine = firsts[blocks["segment"][firsts] == segment]
        np.testing.assert_array_equal(blocks["instance"][mine], np.arange(mine.size))
    np.testing.assert_array_equal(blocks["slot"], blocks["instance"] % ring)
    positions = record["positions"]
    assert positions["slot_offset"].shape[1:] == (ring, 3)
    for block in np.flatnonzero(blocks["wave"] >= 0):
        (row,) = np.flatnonzero(
            (positions["segment"] == blocks["segment"][block])
            & (positions["position"] == blocks["position"][block])
        )
        slot = blocks["slot"][block]
        np.testing.assert_array_equal(
            blocks["wave_offset"][block], positions["slot_offset"][row, slot]
        )
        assert blocks["wave_samples"][block] == positions["slot_samples"][row, slot]
    assert record["unloaded"] == 0
    assert record["overwrites"] == 0


def test_without_a_budget_the_playout_holds_the_waves_as_the_cache_lays_them_out(
    tmp_path,
):
    seq = _converted("zte_3d.seq", tmp_path)
    budget = _budgets(seq, 3)["streamed"]
    ir.convert(seq, SYSTEM, wave_budget=budget)
    stored = ir.playout(seq)
    assert (stored["mode"], stored["slots"]) == ("streamed", 3)
    given = ir.playout(seq, budget)
    np.testing.assert_array_equal(stored["blocks"]["slot"], given["blocks"]["slot"])
    np.testing.assert_array_equal(
        stored["blocks"]["wave_offset"], given["blocks"]["wave_offset"]
    )


def test_every_wave_a_position_plays_covers_the_span_it_prepares(played):
    seq, _ = played
    record = ir.playout(seq, WaveBudget(LOTS, RASTER_US))
    blocks, positions = record["blocks"], record["positions"]
    for block in np.flatnonzero(blocks["wave"] >= 0):
        (row,) = np.flatnonzero(
            (positions["segment"] == blocks["segment"][block])
            & (positions["position"] == blocks["position"][block])
        )
        assert (positions["slot_offset"][row] == -1).all()
        assert blocks["wave_samples"][block] == positions["slot_samples"][row, 0]
        assert blocks["wave_start_us"][block] == positions["slot_start_us"][row, 0]


def test_the_prescan_plays_one_subsequence_to_the_instance_completing_its_readouts(
    tmp_path,
):
    seq = _converted("gre_2d_3sl.seq", tmp_path)
    record = ir.playout(seq, WaveBudget(LOTS, RASTER_US), prescan=Prescan(0, 5))
    blocks = record["blocks"]
    assert (blocks["subsequence"] == 0).all()
    readouts = np.cumsum(blocks["adc"])
    last = np.flatnonzero(blocks["position"] == 0)[-1]
    assert readouts[-1] >= 5
    assert readouts[last - 1] < 5
    scan = ir.playout(seq, WaveBudget(LOTS, RASTER_US))["blocks"]
    assert scan["segment"].size > blocks["segment"].size


def test_the_prescan_plays_no_gradient_whose_amplitude_varies(tmp_path):
    seq = _converted("gre_2d_3sl.seq", tmp_path)
    budget = WaveBudget(LOTS, RASTER_US)
    prescan = ir.playout(seq, budget, prescan=Prescan(0, 10**6))["blocks"]
    scan = ir.playout(seq, budget)["blocks"]
    count = prescan["segment"].size
    varies = prescan["gradient_variable"] != 0
    assert (scan["gradient_hz_per_m"][:count][varies] != 0).any()
    assert (prescan["gradient_hz_per_m"][varies] == 0).all()
    np.testing.assert_array_equal(
        prescan["gradient_hz_per_m"][~varies],
        scan["gradient_hz_per_m"][:count][~varies],
    )


def _gated(path):
    """A repetition that waits for a cardiac trigger, pulses an output and reads
    out without the prescription rotation."""
    rf = pp.make_block_pulse(0.2, duration=5e-4, use="excitation", system=SYSTEM)
    gx = pp.make_trapezoid("x", flat_area=2e3, flat_time=2e-3, system=SYSTEM)
    adc = pp.make_adc(64, duration=2e-3, delay=gx.rise_time, system=SYSTEM)
    seq = pp.Sequence(SYSTEM)
    for _ in range(3):
        seq.add_block(
            pp.make_delay(1e-3),
            pp.make_trigger("physio1", duration=5e-4),
            pp.make_label(label="NOROT", type="SET", value=0),
        )
        seq.add_block(rf, pp.make_digital_output_pulse("osc0", duration=2e-4))
        seq.add_block(gx, adc, pp.make_label(label="NOROT", type="SET", value=1))
        seq.add_block(
            pp.make_delay(2e-3), pp.make_label(label="NOROT", type="SET", value=0)
        )
    seq.write(str(path))
    ir.convert(path, SYSTEM)
    return path


def test_the_scan_waits_for_a_cardiac_trigger_without_driving_an_output(tmp_path):
    path = _gated(tmp_path / "gated.seq")
    blocks = ir.playout(path, WaveBudget(LOTS, RASTER_US))["blocks"]
    trigger = (blocks["await_trigger"] != 0) & (blocks["position"] == 0)
    assert trigger.sum() == 3
    assert not blocks["digitalout"][trigger].any()


def test_the_scan_pulses_each_output_and_leaves_norot_blocks_unrotated(tmp_path):
    path = _gated(tmp_path / "gated.seq")
    blocks = ir.playout(path, WaveBudget(LOTS, RASTER_US))["blocks"]
    np.testing.assert_array_equal(blocks["rotate"], 1 - ir.play(path)["norot"])
    assert (blocks["rotate"] == 0).any()
    assert blocks["digitalout"].sum() == 3


def _turned_readouts(path):
    """A prephaser unturned in the first repetition and turned in the second:
    one segment definition, NOROT in one of its instances."""
    quarter_turn = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rf = pp.make_block_pulse(np.pi / 2, duration=2e-4, system=SYSTEM)
    prephaser = pp.make_trapezoid("x", area=-500.0, duration=1e-3, system=SYSTEM)
    readout = pp.make_trapezoid("x", flat_area=1000.0, flat_time=2e-3, system=SYSTEM)
    adc = pp.make_adc(64, duration=2e-3, delay=readout.rise_time, system=SYSTEM)
    seq = pp.Sequence(SYSTEM)
    for norot in (1, 0):
        seq.add_block(rf, pp.make_label(label="NOROT", type="SET", value=0))
        seq.add_block(prephaser, pp.make_label(label="NOROT", type="SET", value=norot))
        seq.add_block(
            readout,
            adc,
            pp.make_rotation(quarter_turn),
            pp.make_label(label="NOROT", type="SET", value=1 - norot),
        )
    seq.write(str(path))
    ir.convert(path, SYSTEM)
    return path


def test_each_instance_is_turned_as_its_own_blocks_are_labelled(tmp_path):
    path = _turned_readouts(tmp_path / "turned.seq")
    blocks = ir.playout(path, WaveBudget(LOTS, RASTER_US))["blocks"]
    norot = ir.play(path)["norot"]
    assert norot.any() and not norot.all()
    np.testing.assert_array_equal(blocks["rotate"], 1 - norot)


def test_the_prescan_waits_for_no_trigger_and_pulses_no_output(tmp_path):
    path = _gated(tmp_path / "gated.seq")
    blocks = ir.playout(path, WaveBudget(LOTS, RASTER_US), prescan=Prescan())["blocks"]
    assert blocks["adc"].sum() == 1
    assert not blocks["await_trigger"].any()
    assert not blocks["digitalout"].any()


def test_waves_that_cannot_be_loaded_in_time_are_refused_before_anything_plays(
    tmp_path,
):
    seq = _converted("zte_3d.seq", tmp_path)
    streamed = _budgets(seq)["streamed"]
    slow = WaveBudget(streamed.max_samples, RASTER_US, load_us_per_sample=1e3)
    with pytest.raises(ValueError, match="loading"):
        ir.playout(seq, slow)


def test_a_prescan_of_a_subsequence_the_scan_lacks_is_refused(tmp_path):
    seq = _converted("gre_2d_3sl.seq", tmp_path)
    with pytest.raises(ValueError):
        ir.playout(seq, WaveBudget(LOTS, RASTER_US), prescan=Prescan(5))
