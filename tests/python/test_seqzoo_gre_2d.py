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
    flagged = labels["LIN"][labels["IMA"] == 1]
    assert set(flagged.tolist()) == set(range(N_Y // 2 - 4, N_Y // 2 + 4))


def test_every_calibration_line_is_imaging_data_too():
    """The block is a fully sampled centre of the same k-space, so all of it
    is imaging data -- the lines off the acceleration grid included."""
    seq = design(acceleration=2, n_acs=8)
    labels = seq.evaluate_labels(evolution="adc")
    assert np.array_equal(labels["IMA"] == 1, _in_calibration(labels["LIN"], 8))
    assert "REF" not in labels
    assert np.any(labels["LIN"][labels["IMA"] == 1] % 2 == 1)


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
# Slice packets
# ----------------------------------------------------------------------


def _kernel(**kwargs):
    kwargs.setdefault("te", 5e-3)
    return gre_2d.GREKernel(pp.Opts(), n_x=N_X, n_y=N_Y, **kwargs)


def test_a_tr_that_holds_every_slice_is_one_pass():
    kernel = _kernel(n_slices=4, tr=60e-3)
    assert [len(group) for group in kernel.passes] == [4]


def test_more_slices_than_the_tr_holds_are_split_rather_than_refused():
    """The failure this replaces: a slice count that did not fit raised."""
    kernel = _kernel(n_slices=10, tr=25e-3, n_acs=8)
    sizes = [len(group) for group in kernel.passes]
    assert sum(sizes) == 10
    assert max(sizes) - min(sizes) <= 1
    assert sorted(index for group in kernel.passes for index in group) == list(range(10))


def test_every_slice_sees_the_requested_repetition_time():
    """What the split is for: a short pass gets a longer repetition, not a
    shorter TR."""
    tr = 25e-3
    kernel = _kernel(n_slices=10, tr=tr, n_acs=8)
    assert len({len(group) for group in kernel.passes}) == 2, "not the interesting case"
    for group in kernel.passes:
        # A repetition is rounded up onto the block raster, so a pass can run
        # long by at most one raster tick per slice in it.
        assert len(group) * kernel.readouts[len(group)].duration == pytest.approx(
            tr, abs=len(group) * 10e-6
        )


def test_the_slices_of_a_pass_are_not_neighbours():
    kernel = _kernel(n_slices=10, tr=25e-3, n_acs=8)
    for group in kernel.passes:
        assert min(abs(a - b) for a in group for b in group if a != b) > 1


def test_the_reported_duration_is_the_one_the_scan_takes(
):
    for prescription in ({"n_slices": 10, "tr": 25e-3}, {"n_slices": 4, "tr": 60e-3}):
        kernel = _kernel(n_acs=8, n_dummy=2, **prescription)
        seq = design(n_acs=8, n_dummy=2, **prescription)
        assert seq.duration()[0] == pytest.approx(kernel.duration)


def test_every_slice_is_acquired_the_same_number_of_times_across_passes():
    seq = design(n_slices=10, tr=25e-3, n_acs=8, acceleration=2, n_dummy=0)
    counts = np.bincount(seq.evaluate_labels(evolution="adc")["SLC"])
    assert len(counts) == 10
    assert len(set(counts.tolist())) == 1


# ----------------------------------------------------------------------
# What the labels say
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "prescription",
    [
        {"acceleration": 1, "n_acs": 0},
        {"acceleration": 2, "n_acs": 8},
        {"acceleration": 2, "n_acs": 0},
        {"acceleration": 3, "n_acs": 0},
        {"acceleration": 1, "n_acs": 8, "partial_fourier": 0.75},
        {"acceleration": 2, "n_acs": 8, "partial_fourier": 0.75},
    ],
)
def test_the_line_counter_is_a_position_on_the_prescribed_grid(prescription):
    """Not a count of what was acquired.

    ``auto_label`` derives ``LIN``, and the two disagree whenever the sampling
    does not reach the grid on its own: an accelerated scan with no
    autocalibration block has no adjacent pair to read the step off, and a
    partial-Fourier one never visits the low edge.
    """
    seq = design(n_slices=2, tr=30e-3, n_dummy=0, **prescription)
    labels = seq.evaluate_labels(evolution="adc")
    assert np.array_equal(labels["LIN"], acquired_lines(seq))
    assert seq.get_definition("kSpaceCenterLine") == pytest.approx(N_Y // 2, abs=1)


def test_the_calibration_block_is_flagged_across_the_split():
    seq = design(n_slices=10, tr=25e-3, n_acs=8, acceleration=2, n_dummy=0)
    labels = seq.evaluate_labels(evolution="adc")
    assert np.array_equal(labels["SEG"] == 0, _in_calibration(labels["LIN"], 8))
    assert labels["LASTSEG"].sum() == 2 * 10
    assert labels["LASTSLC"].sum() == 10


# ----------------------------------------------------------------------
# Partial Fourier
# ----------------------------------------------------------------------


@pytest.mark.parametrize("partial_fourier", [1.0, 0.75, 0.6])
def test_partial_fourier_drops_the_lines_before_the_centre(partial_fourier):
    lines = acquired_lines(design(partial_fourier=partial_fourier, n_acs=0))
    assert lines.min() == N_Y - round(partial_fourier * N_Y)
    assert lines.max() == N_Y - 1


def test_partial_fourier_shortens_the_scan():
    assert design(partial_fourier=0.75).duration()[0] < design().duration()[0]


# ----------------------------------------------------------------------
# Where the volume sits
# ----------------------------------------------------------------------


def test_no_offset_leaves_the_sequence_exactly_as_it_was():
    """The default has to cost nothing: it is what every unshifted scan pays."""
    plain = design(n_slices=2, tr=30e-3, n_dummy=0)
    zeroed = design(n_slices=2, tr=30e-3, n_dummy=0, fov_offset=(0.0, 0.0, 0.0))
    assert np.array_equal(
        plain.get_block(1).rf.signal, zeroed.get_block(1).rf.signal
    )


def test_an_offset_moves_the_pulse_that_excites_and_the_window_that_reads():
    """Under constant gradients the shift is scalar offsets on both -- the
    waveforms themselves are shared with the unshifted scan."""
    plain = design(n_slices=1, tr=30e-3, n_dummy=0)
    slab = design(n_slices=1, tr=30e-3, n_dummy=0, fov_offset=(0.0, 0.0, 20e-3))
    assert slab.get_block(1).rf.freq_offset != plain.get_block(1).rf.freq_offset
    assert np.array_equal(slab.get_block(1).rf.signal, plain.get_block(1).rf.signal)

    in_plane = design(n_slices=1, tr=30e-3, n_dummy=0, fov_offset=(20e-3, 0.0, 0.0))
    assert in_plane.get_block(4).adc.phase_offset != plain.get_block(4).adc.phase_offset


# ----------------------------------------------------------------------
# Averages
# ----------------------------------------------------------------------


def test_the_averages_are_written_out_rather_than_left_to_the_interpreter():
    once = design(n_slices=2, n_dummy=0, n_acs=8)
    twice = design(n_slices=2, n_dummy=0, n_acs=8, n_averages=2)

    labels = twice.evaluate_labels(evolution="adc")
    assert len(labels["LIN"]) == 2 * len(once.evaluate_labels(evolution="adc")["LIN"])
    assert sorted(set(labels["AVG"].tolist())) == [0, 1]
    assert twice.get_definition("IgnoreAverages") == 1


def test_the_dummies_play_on_the_first_average_only():
    """ONCE is what says so, and it is why the scan is not 2x longer."""
    body = design(n_slices=2, n_dummy=0, n_acs=8).duration()[0]
    one = design(n_slices=2, n_dummy=8, n_acs=8).duration()[0]
    two = design(n_slices=2, n_dummy=8, n_acs=8, n_averages=2).duration()[0]

    prep = one - body
    assert prep > 0
    assert two == pytest.approx(prep + 2 * body)


@pytest.mark.parametrize("n_averages", [1, 2, 3])
def test_the_reported_duration_accounts_for_the_averages(n_averages):
    prescription = {"n_slices": 2, "n_dummy": 4, "n_acs": 8, "n_averages": n_averages}
    kernel = _kernel(tr=30e-3, **prescription)
    assert design(**prescription).duration()[0] == pytest.approx(kernel.duration)


# ----------------------------------------------------------------------
# What the file tells the scanner
# ----------------------------------------------------------------------


def test_the_receive_gain_calibration_is_declared():
    assert design(n_slices=6, tr=60e-3).get_definition(
        "NumGainCalibrationReadouts"
    ) == 6
    assert design(n_gain_calibration_readouts=2).get_definition(
        "NumGainCalibrationReadouts"
    ) == 2


def test_the_prescribed_grid_is_declared():
    """``auto_label`` reads it, and so does anything reconstructing the file."""
    seq = design(n_slices=3, tr=45e-3)
    assert seq.get_definition("Matrix") == [N_X, N_Y, 3]


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


def test_the_scanner_gets_the_binary_form_and_a_reader_gets_the_text_one(tmp_path):
    """Two destinations, one sequence: the scanner checks the file itself."""
    system = pp.Opts()
    protocol = gre_2d.PLUGIN.get_default_protocol(system)
    params.set_protocol_value(protocol, UIParam.NX, 32)
    params.set_protocol_value(protocol, UIParam.NY, 32)
    params.set_protocol_value(protocol, UIParam.TR, 30.0)
    params.set_protocol_value(protocol, UIParam.TE, 5.0)

    for name, kwargs in (("server.seq", {}), ("offline.seq", {"offline": True})):
        gre_2d.PLUGIN.make_sequence(system, protocol, str(tmp_path / name), **kwargs)

    assert (tmp_path / "server.seq").read_bytes().startswith(b"\x01pulseq")
    text = (tmp_path / "offline.seq").read_text()
    assert text.startswith("# Pulseq sequence file")
    assert "[SIGNATURE]" in text

    # Same scan either way, whatever it is written as.
    back = pp.Sequence()
    back.read(str(tmp_path / "server.seq"))
    reference = pp.Sequence()
    reference.read(str(tmp_path / "offline.seq"))
    assert back.num_blocks == reference.num_blocks
    assert back.duration()[0] == pytest.approx(reference.duration()[0])


def test_the_plugin_writes_the_same_sequence_the_repl_builds(tmp_path):
    opts = pp.Opts()
    protocol = gre_2d.PLUGIN.get_default_protocol(opts)
    params.set_protocol_value(protocol, UIParam.NX, 32)
    params.set_protocol_value(protocol, UIParam.NY, 32)
    params.set_protocol_value(protocol, UIParam.TR, 30.0)
    params.set_protocol_value(protocol, UIParam.TE, 5.0)

    written = tmp_path / "gre_2d.seq"
    gre_2d.PLUGIN.make_sequence(opts, protocol, str(written), offline=True)
    assert written.exists()

    direct = tmp_path / "direct.seq"
    design(system=opts, fov=(0.22, 0.22), flip_angle_deg=12.0, n_acs=24).write(
        str(direct)
    )
    assert _blocks(written) == _blocks(direct)


def _blocks(path):
    text = path.read_text().split("[BLOCKS]")[1]
    return text.split("[")[0].strip()
