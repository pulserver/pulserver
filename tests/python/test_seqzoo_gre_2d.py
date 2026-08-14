"""The 2D Cartesian gradient echo of the sequence zoo.

Checked through what a reconstruction actually receives -- the k-space
`calculate_kspace` reports and the labels `evaluate_labels` reports -- rather
than through the event fields the sequence was built from.
"""

from __future__ import annotations

import numpy as np
import pytest

import pulserver.pypulseq as pp
from pulserver import UIParam, dict_to_protocol, params, protocol_to_dict
from pulserver.seqzoo import gre_2d

N_X = 32
N_Y = 32


def design(**kwargs):
    kwargs.setdefault("te", 5e-3)
    kwargs.setdefault("tr", 30e-3)
    return gre_2d.main(n_x=N_X, n_y=N_Y, **kwargs)


def acquired_lines(seq, n_y=N_Y):
    """Phase-encode index of every acquisition, read back out of k-space."""
    k_traj_adc, *_ = seq.calculate_kspace()
    n_samples = k_traj_adc.shape[1] // _n_acquisitions(seq)
    ky = k_traj_adc[1].reshape(-1, n_samples)[:, 0]
    delta_ky = 1.0 / seq.get_definition("FOV")[1]
    return np.rint(ky / delta_ky).astype(int) + n_y // 2


def _n_acquisitions(seq):
    _, _, _, _, t_adc = seq.calculate_kspace()
    return len(seq.evaluate_labels(evolution="adc")["LIN"]) if len(t_adc) else 0


# ----------------------------------------------------------------------
# The sequence itself
# ----------------------------------------------------------------------


def test_the_sequence_is_valid_pulseq():
    is_ok, error_report = design().check_timing()
    assert is_ok, error_report


def test_one_repetition_per_acquired_line_and_slice():
    seq = design(n_slices=3, tr=60e-3, acceleration=2, n_acs=8)
    n_lines = len(pp.calc_sampled_lines(N_Y, 2, 8))
    assert len(seq.evaluate_labels(evolution="adc")["LIN"]) == n_lines * 3


def test_the_repetition_time_is_per_slice_not_per_slice_group():
    """Successive excitations of the *same* slice are one TR apart."""
    seq = design(n_slices=4, tr=40e-3, n_acs=0, n_dummy=0)
    assert seq.duration()[0] == pytest.approx(N_Y * 40e-3)


def test_the_acquired_lines_are_the_ones_the_sampling_plan_asked_for():
    seq = design(acceleration=2, n_acs=8)
    expected = np.asarray(
        pp.calc_sampled_lines(N_Y, 2, 8, order="calibration_first")
    )
    assert np.array_equal(acquired_lines(seq), expected)


def test_the_calibration_block_is_acquired_before_anything_else():
    """The reconstruction calibrates from it while the scan is still running."""
    seq = design(acceleration=2, n_acs=8)
    lines = acquired_lines(seq)
    assert set(lines[:8].tolist()) == set(range(N_Y // 2 - 4, N_Y // 2 + 4))


def test_the_line_counter_agrees_with_where_the_line_actually_is():
    """LIN is what the reconstruction grids by, so it must match k-space."""
    seq = design(acceleration=2, n_acs=8)
    assert np.array_equal(
        seq.evaluate_labels(evolution="adc")["LIN"], acquired_lines(seq)
    )


# ----------------------------------------------------------------------
# Parallel imaging
# ----------------------------------------------------------------------


def test_the_calibration_block_is_flagged_and_nothing_else_is():
    seq = design(acceleration=2, n_acs=8)
    labels = seq.evaluate_labels(evolution="adc")
    flagged = labels["LIN"][(labels["REF"] == 1) | (labels["IMA"] == 1)]
    assert set(flagged.tolist()) == set(range(N_Y // 2 - 4, N_Y // 2 + 4))


def test_a_calibration_line_on_the_grid_is_imaging_data_too():
    """REF alone is calibration only; IMA is calibration *and* imaging."""
    seq = design(acceleration=2, n_acs=8)
    labels = seq.evaluate_labels(evolution="adc")
    assert np.array_equal(
        labels["IMA"] == 1, (labels["REF"] == 0) & _in_calibration(labels["LIN"], 8)
    )
    assert np.all(labels["LIN"][labels["REF"] == 1] % 2 == 1)


def test_an_unaccelerated_scan_flags_nothing():
    labels = design(acceleration=1, n_acs=8).evaluate_labels(evolution="adc")
    assert not labels["REF"].any()


def test_the_calibration_block_closes_a_segment_of_its_own():
    """SEG separates calibration from the rest; LASTSEG says where it ends."""
    seq = design(n_slices=2, tr=60e-3, acceleration=2, n_acs=8)
    labels = seq.evaluate_labels(evolution="adc")
    calibration = _in_calibration(labels["LIN"], 8)
    assert np.array_equal(labels["SEG"] == 0, calibration)

    # Once per slice at the end of the calibration block, and once per slice at
    # the end of the scan -- a segment closes for every slice it spans.
    closing = labels["LIN"][labels["LASTSEG"] == 1]
    assert sorted(closing.tolist()) == sorted([N_Y // 2 + 3] * 2 + [N_Y - 2] * 2)


def test_the_last_line_of_each_slice_says_so():
    """LASTSLC is what tells a reconstruction the slice is complete."""
    seq = design(n_slices=3, tr=90e-3, acceleration=2, n_acs=8)
    labels = seq.evaluate_labels(evolution="adc")
    assert int((labels["LASTSLC"] == 1).sum()) == 3
    assert sorted(labels["SLC"][labels["LASTSLC"] == 1].tolist()) == [0, 1, 2]


def test_dummy_repetitions_are_played_without_acquiring():
    """They cost time and reach the steady state, and carry no data."""
    acquired = len(design(n_dummy=0).evaluate_labels(evolution="adc")["LIN"])
    with_dummies = design(n_dummy=5)
    assert len(with_dummies.evaluate_labels(evolution="adc")["LIN"]) == acquired
    assert with_dummies.duration()[0] == pytest.approx(
        design(n_dummy=0).duration()[0] + 5 * 30e-3
    )


def _in_calibration(lines, n_acs):
    start = max(0, N_Y // 2 - n_acs // 2)
    return (lines >= start) & (lines < start + n_acs)


# ----------------------------------------------------------------------
# Partial echo
# ----------------------------------------------------------------------


@pytest.mark.parametrize("partial_echo", [1.0, 0.75, 0.6])
def test_partial_echo_drops_the_samples_before_the_echo(partial_echo):
    seq = design(partial_echo=partial_echo)
    assert _adc_samples(seq) == max(N_X // 2 + 1, round(partial_echo * N_X))


def test_a_partial_echo_starts_further_along_the_readout():
    """The truncation is on the leading side, so k begins closer to zero."""
    full = design(partial_echo=1.0).calculate_kspace()[0][0, 0]
    partial = design(partial_echo=0.7).calculate_kspace()[0][0, 0]
    assert abs(partial) < abs(full)
    assert np.sign(partial) == np.sign(full)


def test_a_partial_echo_shortens_the_shortest_possible_te():
    full = _shortest_te(1.0)
    assert _shortest_te(0.6) < full
    with pytest.raises(ValueError, match="TE"):
        design(te=full - 1e-4)


def _shortest_te(partial_echo):
    """The TE the module reports when asked for the shortest one it can do."""
    return design(te=None, partial_echo=partial_echo).get_definition("TE")


def _adc_samples(seq):
    labels = seq.evaluate_labels(evolution="adc")
    k_traj_adc, *_ = seq.calculate_kspace()
    return k_traj_adc.shape[1] // len(labels["LIN"])


# ----------------------------------------------------------------------
# Multi-slice
# ----------------------------------------------------------------------


def test_every_slice_is_excited_at_its_own_offset():
    seq = design(n_slices=5, tr=60e-3, slice_thickness=4e-3, slice_gap=1e-3)
    labels = seq.evaluate_labels(evolution="adc")
    assert set(labels["SLC"].tolist()) == set(range(5))


def test_the_slices_are_visited_in_the_order_that_was_asked_for():
    seq = design(n_slices=4, tr=40e-3, slice_order="interleaved")
    order = seq.evaluate_labels(evolution="adc")["SLC"][:4]
    assert np.array_equal(order, pp.calc_traversal_order(4, "interleaved"))


# ----------------------------------------------------------------------
# The plugin wrapper
# ----------------------------------------------------------------------


def test_the_default_protocol_round_trips():
    protocol = gre_2d.PLUGIN.get_default_protocol(pp.Opts())
    assert params.param_int(dict_to_protocol(protocol), UIParam.NX) == 128
    assert protocol_to_dict(dict_to_protocol(protocol)) == protocol


def test_a_feasible_protocol_reports_its_scan_time():
    opts = pp.Opts()
    protocol = gre_2d.PLUGIN.get_default_protocol(opts)
    result = gre_2d.PLUGIN.validate_protocol(opts, protocol)
    assert result["valid"]
    assert result["duration"] == pytest.approx((128 + 16) * 0.25)


def test_an_impossible_echo_time_is_refused_by_name():
    opts = pp.Opts()
    protocol = gre_2d.PLUGIN.get_default_protocol(opts)
    params.set_protocol_value(protocol, UIParam.TE, 0.5)
    result = gre_2d.PLUGIN.validate_protocol(opts, protocol)
    assert not result["valid"]
    assert "TE" in result["info"]


def test_the_plugin_writes_the_same_sequence_the_repl_builds(tmp_path):
    opts = pp.Opts()
    protocol = gre_2d.PLUGIN.get_default_protocol(opts)
    params.set_protocol_value(protocol, UIParam.NX, 32)
    params.set_protocol_value(protocol, UIParam.NY, 32)
    params.set_protocol_value(protocol, UIParam.TR, 30.0)
    params.set_protocol_value(protocol, UIParam.TE, 5.0)

    written = tmp_path / "gre_2d.seq"
    gre_2d.PLUGIN.make_sequence(opts, protocol, str(written))
    assert written.exists()

    direct = tmp_path / "direct.seq"
    design(system=opts, fov=(0.22, 0.22), flip_angle_deg=12.0, n_acs=24).write(
        str(direct), remove_duplicates=False, check_timing=False
    )
    assert _blocks(written) == _blocks(direct)


def _blocks(path):
    text = path.read_text().split("[BLOCKS]")[1]
    return text.split("[")[0].strip()
