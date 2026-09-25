from pathlib import Path

import numpy as np
import pypulseqpp as pp
import pytest
from _synthetic import SAMPLES, add_readout

from pulserver.mrd import ReadoutTable, SequenceDefinitions, _sequence, read_chain

FIXTURES = Path(__file__).parent / "fixtures" / "sequences"
SEQUENCES = sorted(path.name for path in FIXTURES.glob("*.seq"))


def fixture(name):
    seq = pp.io.read(FIXTURES / name)
    return seq, ReadoutTable.from_sequence(seq)


def synthetic(*readouts):
    seq = pp.Sequence(pp.Opts())
    for keywords in readouts:
        add_readout(seq, **keywords)
    return ReadoutTable.from_sequence(seq)


def whole_scan_k(seq, table, index):
    start = int(table.num_samples[:index].sum())
    return seq.calculate_kspace()[0][:, start : start + int(table.num_samples[index])]


def test_readouts_follow_the_adc_windows_in_play_order():
    seq, table = fixture("gre_2d_3sl.seq")
    adc = seq.waveforms_and_times(compat=False).adc
    np.testing.assert_array_equal(table.block, adc.block)
    assert table.num_samples.tolist() == [64] * 24
    assert table.readout_k(23).shape == (3, 64)
    np.testing.assert_allclose(table.dwell, 1e-5)


@pytest.mark.parametrize("range_samples", [_sequence._RANGE_SAMPLES, 1])
@pytest.mark.parametrize("name", SEQUENCES)
def test_k_integrated_from_an_excitation_is_k_integrated_from_the_first_block(
    monkeypatch, name, range_samples
):
    monkeypatch.setattr(_sequence, "_RANGE_SAMPLES", range_samples)
    seq, table = fixture(name)
    whole = seq.calculate_kspace()[0]
    # The float32 resolution of the largest k, which an MRD trajectory carries.
    tolerance = 1e-6 * np.abs(whole).max()
    start = 0
    for index in range(len(table)):
        stop = start + int(table.num_samples[index])
        np.testing.assert_allclose(
            table.readout_k(index), whole[:, start:stop], rtol=0, atol=tolerance
        )
        start = stop


def test_a_spin_echo_train_keeps_the_k_its_refocusing_pulses_reverse(monkeypatch):
    monkeypatch.setattr(_sequence, "_RANGE_SAMPLES", 1)
    system = pp.Opts()
    excitation = pp.make_block_pulse(np.pi / 2, duration=1e-3, system=system)
    refocusing = pp.make_block_pulse(
        np.pi, duration=1e-3, use="refocusing", system=system
    )
    readout = pp.make_trapezoid("x", flat_area=SAMPLES * 5.0, flat_time=3.2e-3)
    adc = pp.make_adc(num_samples=SAMPLES, duration=3.2e-3, delay=readout.rise_time)
    prephaser = pp.make_trapezoid("x", area=readout.area / 2, duration=1e-3)
    seq = pp.Sequence(system)
    for _ in range(2):
        seq.add_block(excitation)
        seq.add_block(prephaser)
        for _ in range(3):
            seq.add_block(refocusing)
            seq.add_block(readout, adc)
    table = ReadoutTable.from_sequence(seq)
    assert len(table._ranges._first) == 2
    for index in range(len(table)):
        np.testing.assert_allclose(
            table.readout_k(index), whole_scan_k(seq, table, index), atol=1e-9
        )
    assert table.center_sample.tolist() == [SAMPLES // 2] * 6


def test_an_excitation_holding_a_readout_starts_no_range(monkeypatch):
    monkeypatch.setattr(_sequence, "_RANGE_SAMPLES", 1)
    system = pp.Opts()
    readout = pp.make_trapezoid("x", flat_area=SAMPLES * 5.0, flat_time=3.2e-3)
    adc = pp.make_adc(num_samples=SAMPLES, duration=3.2e-3, delay=readout.rise_time)
    late = pp.make_block_pulse(
        np.pi / 2, duration=1e-3, delay=readout.rise_time + 3.2e-3, system=system
    )
    seq = pp.Sequence(system)
    seq.add_block(pp.make_block_pulse(np.pi / 2, duration=1e-3, system=system))
    seq.add_block(readout, adc)
    seq.add_block(readout, adc, late)
    table = ReadoutTable.from_sequence(seq)
    assert table._ranges._first.tolist() == [1]
    np.testing.assert_allclose(
        table.readout_k(1), whole_scan_k(seq, table, 1), atol=1e-9
    )
    assert np.abs(table.readout_k(1)[0]).min() > np.abs(table.readout_k(0)[0]).max()


def test_a_table_keeps_the_k_of_two_ranges_at_most(monkeypatch):
    monkeypatch.setattr(_sequence, "_RANGE_SAMPLES", 1)
    _, table = fixture("gre_2d_3sl.seq")
    for index in range(len(table)):
        table.readout_k(index)
    assert len(table._ranges._first) == len(table)
    assert len(table._ranges._kept) == _sequence._KEPT_RANGES == 2


def test_labels_are_the_values_in_force_at_each_readout():
    seq, table = fixture("epi_2d_main.seq")
    expected = seq.evaluate_labels(evolution="adc")
    assert set(table.labels) == set(expected)
    for name, values in expected.items():
        np.testing.assert_array_equal(
            table.labels[name], np.broadcast_to(values, (len(table),))
        )


def test_a_single_readout_sees_the_labels_in_force_at_it():
    """Not the values the labels end the sequence with."""
    seq = pp.Sequence(pp.Opts())
    add_readout(seq, pp.make_label("LIN", "SET", 3))
    seq.add_block(pp.make_label("LIN", "SET", 7), pp.make_delay(1e-3))

    table = ReadoutTable.from_sequence(seq)

    assert table.labels["LIN"].tolist() == [3]


def test_the_echo_index_is_the_design_centre_sample():
    seq, table = fixture("gre_2d_3sl.seq")
    design = int(np.atleast_1d(seq.get_definition("kSpaceCenterSample"))[0])
    assert set(table.center_sample.tolist()) == {design}


def test_reversed_lines_meet_the_echo_at_the_mirrored_sample():
    _, table = fixture("epi_2d_main.seq")
    reverse = table.labels["REV"] != 0
    assert reverse.any() and (~reverse).any()
    forward_centre = set(table.center_sample[~reverse].tolist())
    reverse_centre = set(table.center_sample[reverse].tolist())
    assert len(forward_centre) == len(reverse_centre) == 1
    assert forward_centre.pop() + reverse_centre.pop() == int(table.num_samples[0]) - 1


def test_a_spiral_out_echo_is_its_first_sample():
    _, table = fixture("mprage_stack_of_spirals_3d.seq")
    assert table.center_sample.tolist() == [0] * len(table)


def test_a_rotated_readout_keeps_the_echo_index_of_its_unrotated_copy():
    table = synthetic(
        {"rotation": pp.make_rotation(0.0)}, {"rotation": pp.make_rotation(np.pi / 2)}
    )
    assert table.center_sample.tolist() == [SAMPLES // 2] * 2


def test_a_readout_whose_k_does_not_move_has_no_echo_and_no_trajectory():
    table = synthetic({"moving": False})
    assert table.center_sample.tolist() == [-1]
    assert table.trajectory_dimensions.tolist() == [0]


@pytest.mark.parametrize(
    ("name", "dimensions"),
    [
        ("gre_2d_3sl.seq", 1),
        ("epi_2d_main.seq", 1),
        ("mprage_stack_of_spirals_3d.seq", 2),
        ("zte_3d.seq", 3),
    ],
)
def test_trailing_constant_axes_are_dropped(name, dimensions):
    _, table = fixture(name)
    assert set(table.trajectory_dimensions.tolist()) == {dimensions}


def test_a_leading_constant_axis_stays():
    table = synthetic(
        {"rotation": pp.make_rotation(0.0)}, {"rotation": pp.make_rotation(np.pi / 2)}
    )
    assert table.trajectory_dimensions.tolist() == [1, 2]


def test_a_chain_is_read_in_play_order():
    chain = read_chain(FIXTURES / "dedup_gre_pair.seq")
    assert [path.name for path, _ in chain] == [
        "dedup_gre_pair.seq",
        "dedup_gre_pair_b.seq",
    ]


def test_a_chain_that_returns_to_a_played_file_is_refused(tmp_path):
    seq = pp.Sequence(pp.Opts())
    add_readout(seq)
    seq.set_definition("NextSequence", "loop.seq")
    seq.write(tmp_path / "loop.seq")
    with pytest.raises(ValueError, match="returns to"):
        read_chain(tmp_path / "loop.seq")


def test_a_missing_chain_file_is_refused(tmp_path):
    seq = pp.Sequence(pp.Opts())
    add_readout(seq)
    seq.set_definition("NextSequence", "absent.seq")
    seq.write(tmp_path / "first.seq")
    with pytest.raises(FileNotFoundError, match=r"absent\.seq"):
        read_chain(tmp_path / "first.seq")


def test_definitions_are_read_in_pulseq_units():
    seq, _ = fixture("gre_2d_3sl.seq")
    definitions = SequenceDefinitions.from_sequence(seq)
    assert definitions.matrix == (64, 8, 3)
    assert definitions.fov == pytest.approx((0.22, 0.22, 0.015))
    assert definitions.te == pytest.approx((0.005,))
    assert definitions.tr == pytest.approx((0.03,))
    assert definitions.navigator_matrix is None
    assert definitions.ti == ()


def test_a_flip_angle_the_sequence_does_not_define_is_the_one_pypulseqpp_measures():
    seq, _ = fixture("gre_2d_3sl.seq")
    assert seq.get_definition("FlipAngle") == ""
    measured = tuple(seq.test_report_dict()["flip_angles_deg"])
    definitions = SequenceDefinitions.from_sequence(seq)
    assert measured
    assert definitions.flip_angle == pytest.approx(measured)
    assert definitions.te == pytest.approx((0.005,))


def test_te_and_tr_a_sequence_does_not_define_are_the_ones_pypulseqpp_measures():
    system = pp.Opts()
    seq = pp.Sequence(system)
    for _ in range(2):
        seq.add_block(
            pp.make_block_pulse(
                np.pi / 18, duration=1e-4, use="excitation", system=system
            )
        )
        add_readout(seq)
    report = seq.test_report_dict()
    definitions = SequenceDefinitions.from_sequence(seq)
    assert definitions.te == pytest.approx((report["TE"],))
    assert definitions.tr == pytest.approx((report["TR"],))
    assert definitions.flip_angle == pytest.approx((10.0,))


def test_a_sequence_without_rf_measures_no_echo_or_repetition_time():
    seq = pp.Sequence(pp.Opts())
    add_readout(seq)
    definitions = SequenceDefinitions.from_sequence(seq)
    assert (definitions.tr, definitions.te, definitions.flip_angle) == ((), (), ())
