"""The public Sequence over the C++ pulseq library: build, decode, window, files."""

from __future__ import annotations

import matplotlib
import numpy as np
import pulserver.pypulseq as pp
import pypulseq as upstream
import pytest
from pulserver.pypulseq._extensions import strip_extensions
from scipy.spatial.transform import Rotation

matplotlib.use("Agg")


@pytest.fixture
def system():
    return pp.Opts(
        max_grad=32,
        grad_unit="mT/m",
        max_slew=130,
        slew_unit="T/m/s",
        rf_ringdown_time=20e-6,
        rf_dead_time=100e-6,
        adc_dead_time=10e-6,
    )


@pytest.fixture
def events(system):
    """One repetition's worth of events, covering every kind that survives a file."""
    rf, gz, gzr = pp.make_sinc_pulse(
        flip_angle=np.pi / 6,
        duration=3e-3,
        slice_thickness=5e-3,
        apodization=0.5,
        time_bw_product=4,
        system=system,
        return_gz=True,
        use="excitation",
    )
    return {
        "rf": rf,
        "gz": gz,
        "gzr": gzr,
        "gx": pp.make_trapezoid(channel="x", flat_area=581.8, flat_time=3.2e-3, system=system),
        "gy": pp.make_arbitrary_grad(
            channel="y", waveform=np.sin(np.linspace(0, np.pi, 200)) * 1e5, system=system
        ),
        "adc": pp.make_adc(num_samples=128, dwell=2.4e-5, delay=1e-4, system=system),
        "trigger": pp.make_trigger(channel="physio1", duration=1e-3, system=system),
    }


@pytest.fixture
def seq(system, events):
    """Four repetitions, each rotated a little further and labelled with its line."""
    built = pp.Sequence(system=system)
    for line in range(4):
        events["rf"].phase_offset = 0.5 * line
        built.add_block(events["rf"], events["gz"])
        built.add_block(events["gzr"])
        built.add_block(
            events["gx"],
            events["gy"],
            events["adc"],
            pp.make_rotation(Rotation.from_euler("z", 30.0 * line, degrees=True)),
            *([events["trigger"]] if line == 0 else []),
            pp.make_label(type="SET", label="LIN", value=line),
        )
        built.add_block(pp.make_delay(5e-3))
    return built


def test_a_sequence_counts_its_blocks_and_their_duration(seq):
    assert seq.num_blocks == len(seq) == 16
    duration, blocks, per_column = seq.duration()
    assert blocks == 16
    assert duration == pytest.approx(float(np.sum(seq.block_durations)))
    assert per_column.tolist() == [4, 4, 4, 8, 4, 4]  # rf, gx, gy, gz, adc, ext


def test_a_decoded_block_holds_the_events_that_went_into_it(seq, events):
    block = seq.get_block(3)
    assert block.rf is None and block.adc.num_samples == 128
    assert block.gx.type == "trap" and block.gy.type == "grad"
    assert block.label_sets == [("LIN", 0)]
    assert [trigger.channel for trigger in block.triggers] == ["physio1"]
    np.testing.assert_allclose(block.gy.waveform, events["gy"].waveform, rtol=1e-12)
    np.testing.assert_allclose(block.gx.amplitude, events["gx"].amplitude, rtol=1e-12)


def test_deduplication_leaves_the_columns_at_the_precision_the_file_writes(seq, events):
    seq.remove_duplicates()
    # Six significant digits on a gradient amplitude is what `%12g` records, so
    # that -- not double precision -- is what survives a round trip.
    np.testing.assert_allclose(seq.get_block(3).gy.waveform, events["gy"].waveform, rtol=2e-6)
    seq.remove_duplicates()
    assert seq.num_blocks == 16


def test_a_rotation_is_resolved_into_the_gradients_of_a_window(seq):
    window = seq._decode_window(11, 11)  # third repetition, rotated 60 degrees
    rotated = window.get_block(1)
    # A rotated trapezoid is no longer a trapezoid, and it now has a y component.
    assert rotated.gx.type == "grad"
    assert rotated.gy is not None


def test_a_time_range_picks_the_blocks_that_overlap_it(seq):
    first, last = seq.block_range_of(time_range=(0.0, 0.01))
    assert first == 1
    edges = np.concatenate(([0.0], np.cumsum(seq.block_durations)))
    assert edges[last - 1] < 0.01  # the last block starts inside the interval


def test_a_window_can_only_be_asked_for_one_way(seq):
    with pytest.raises(ValueError, match="one of tr_index"):
        seq.block_range_of(time_range=(0.0, 0.01), block_range=(1, 2))
    with pytest.raises(ValueError, match=r"outside 1\.\.16"):
        seq.block_range_of(block_range=(1, 99))


def test_the_analysis_methods_run_on_the_window_they_are_given(seq):
    at_adc, whole, *_ = seq.calculate_kspace(block_range=(3, 3))
    assert at_adc.shape == (3, 128)
    assert whole.shape[0] == 3 and whole.shape[1] > at_adc.shape[1]
    assert len(seq.waveforms(block_range=(1, 4))) == 3
    assert seq.test_report(block_range=(1, 4)).startswith("Number of blocks: 4")
    passed, report = seq.check_timing(block_range=(1, 2))
    assert passed is (len(report) == 0)
    seq.plot(block_range=(1, 4))


def test_a_written_sequence_reads_back_to_the_same_bytes(seq, tmp_path):
    seq.remove_duplicates()
    seq.set_definition("Name", "round_trip")
    seq.set_definition("FOV", [0.22, 0.22, 0.005])
    text = seq.serialize()

    path = tmp_path / "sequence.seq"
    seq.write(path)
    assert path.read_bytes() == text

    back = pp.Sequence(system=seq.system)
    back.read(path)
    assert back.num_blocks == seq.num_blocks
    assert back.get_definition("Name") == "round_trip"
    assert back.serialize() == text


def test_the_binary_format_carries_the_same_sequence(seq, tmp_path):
    seq.remove_duplicates()
    path = tmp_path / "sequence.bin"
    seq.write(path)
    assert path.read_bytes() == seq.serialize(binary=True)

    back = pp.Sequence(system=seq.system)
    back.read(path)
    assert back.num_blocks == seq.num_blocks
    np.testing.assert_allclose(back.block_durations, seq.block_durations, rtol=1e-12)


def test_upstream_pypulseq_reads_what_was_written(seq, tmp_path):
    # Once ROTATIONS is stripped: upstream 1.5.0 has no reader for it.
    seq.remove_duplicates()
    path = tmp_path / "plain.seq"
    path.write_bytes(strip_extensions(seq.serialize()))
    native = upstream.Sequence(system=seq.system)
    native.read(str(path))
    assert len(native.block_events) == seq.num_blocks


def test_set_block_replaces_a_block_in_place(seq):
    seq.set_block(4, pp.make_delay(9e-3))
    assert seq.get_block(4).block_duration == pytest.approx(9e-3)
    assert seq.num_blocks == 16


def test_a_slotted_and_a_namespace_event_describe_the_same_block(system):
    slotted = pp.Sequence(system=system)
    plain = pp.Sequence(system=system)
    slotted.add_block(pp.make_trapezoid(channel="x", area=1000, system=system))
    plain.add_block(upstream.make_trapezoid(channel="x", area=1000, system=system))
    assert slotted.serialize(create_signature=False) == plain.serialize(create_signature=False)


def test_scaling_an_rf_amplitude_costs_no_extra_shape(system):
    """A variable flip angle train is one set of shapes at many amplitudes."""

    def shapes_for(amplitudes):
        built = pp.Sequence(system=system)
        rf = pp.make_block_pulse(flip_angle=np.pi / 2, duration=1e-3, system=system)
        for amplitude in amplitudes:
            rf.amplitude = amplitude
            built.add_block(rf)
        built.remove_duplicates()
        return built._native.num_shapes(), built._native.num_rf()

    one_shape_count, one_row = shapes_for([1e3])
    many_shape_count, many_rows = shapes_for([1e3, 5e2, 2.5e2])
    assert many_shape_count == one_shape_count
    assert (one_row, many_rows) == (1, 3)


@pytest.mark.parametrize(
    "name", ["payload", "tr_info", "num_trs", "tr_duration", "segments", "_collection"]
)
def test_the_scan_structure_properties_say_they_are_not_here_yet(seq, name):
    with pytest.raises(NotImplementedError, match="scan structure"):
        getattr(seq, name)


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("tr_block_range", (0,)),
        ("segment", (0,)),
        ("pns", ()),
        ("mech_resonances", ()),
        ("grad_spectrum", ()),
        ("plot_kspace", ()),
    ],
)
def test_the_scan_structure_methods_say_they_are_not_here_yet(seq, name, args):
    with pytest.raises(NotImplementedError, match="scan structure"):
        getattr(seq, name)(*args)
