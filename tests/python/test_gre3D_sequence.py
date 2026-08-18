"""The 3D Cartesian gradient echo of the sequence zoo.

Checked through what a reconstruction actually receives -- the k-space
`calculate_kspace` reports and the labels `evaluate_labels` reports -- rather
than through the event fields the sequence was built from.
"""

from __future__ import annotations

import numpy as np
import pytest

import pulserver.pypulseq as pp
from pulserver import UIParam, dict_to_protocol, params, protocol_to_dict
from pulserver.app import gre3D_sequence

N_X = 16
N_Y = 16
N_Z = 8
SLAB = 64e-3


def design(**kwargs):
    kwargs.setdefault("n_acs", 6)
    kwargs.setdefault("n_acs_z", 4)
    kwargs.setdefault("n_dummy", 2)
    return gre3D_sequence.main(n_x=N_X, n_y=N_Y, n_z=N_Z, slab_thickness=SLAB, **kwargs)


def kernel(**kwargs):
    kwargs.setdefault("n_acs", 6)
    kwargs.setdefault("n_acs_z", 4)
    return gre3D_sequence.GRE3DKernel(
        pp.Opts(), n_x=N_X, n_y=N_Y, n_z=N_Z, slab_thickness=SLAB, **kwargs
    )


def acquired_pairs(seq):
    """``(line, partition)`` of every acquisition, read back out of k-space."""
    k_traj_adc, *_ = seq.calculate_kspace()
    labels = seq.evaluate_labels(evolution="adc")
    n_samples = k_traj_adc.shape[1] // len(labels["LIN"])
    ky = k_traj_adc[1].reshape(-1, n_samples)[:, 0]
    kz = k_traj_adc[2].reshape(-1, n_samples)[:, 0]
    fov = seq.get_definition("FOV")
    lines = np.rint(ky * fov[1]).astype(int) + N_Y // 2
    partitions = np.rint(kz * fov[2]).astype(int) + N_Z // 2
    return list(zip(lines.tolist(), partitions.tolist(), strict=False))


def _rectangle(n_acs=6, n_acs_z=4):
    acs_y = range(N_Y // 2 - n_acs // 2, N_Y // 2 + -(-n_acs // 2))
    acs_z = range(N_Z // 2 - n_acs_z // 2, N_Z // 2 + -(-n_acs_z // 2))
    return {(line, partition) for partition in acs_z for line in acs_y}


# ----------------------------------------------------------------------
# The sequence itself
# ----------------------------------------------------------------------


def test_the_sequence_is_valid_pulseq():
    is_ok, error_report = design().check_timing()
    assert is_ok, error_report


def test_one_repetition_per_acquired_pair():
    seq = design(acceleration=2, n_acs=6)
    n_pairs = len(kernel(acceleration=2, n_acs=6).pairs)
    assert len(seq.evaluate_labels(evolution="adc")["LIN"]) == n_pairs


def test_the_acquired_pairs_are_the_ones_the_plan_asked_for():
    seq = design(acceleration=2)
    assert acquired_pairs(seq) == kernel(acceleration=2).pairs


def test_the_calibration_rectangle_is_acquired_before_anything_else():
    """The reconstruction calibrates from it while the scan is still running."""
    seq = design(acceleration=2, acceleration_z=2)
    plan = kernel(acceleration=2, acceleration_z=2)
    first = acquired_pairs(seq)[: plan.n_calibration]
    assert set(first) == set(first) & _rectangle()
    assert len(first) == plan.n_calibration


def test_the_counters_agree_with_where_the_readout_actually_is():
    """LIN and PAR are what the reconstruction grids by, so they must match
    k-space.

    The loop writes them, so this reads them back out of the gradients and
    checks the two agree. That is the whole safety argument for authoring
    them: a prephaser that does not reach the line the loop believed it
    encoded shows up here, where asserting the loop against itself would not
    see it.
    """
    seq = design(acceleration=2)
    authored = seq.evaluate_labels(evolution="adc")
    derived, _ = seq.auto_label(skip_apply=True)
    assert authored["LIN"].tolist() == derived["LIN"].tolist()
    assert authored["PAR"].tolist() == derived["PAR"].tolist()

    pairs = acquired_pairs(seq)
    assert authored["LIN"].tolist() == [line for line, _ in pairs]
    assert authored["PAR"].tolist() == [partition for _, partition in pairs]


def test_the_derived_definitions_are_what_reading_the_sequence_back_says():
    """`kSpaceCenterSample` and `SliceThickness` are stated, not detected.

    The readout designed the prephaser and the excitation knows its own
    bandwidth, so each states what it built; this checks the statement against
    what the trajectory and the pulse spectrum say.
    """
    seq = design()
    _, aux = seq.auto_label(skip_apply=True)
    assert seq.get_definition("kSpaceCenterSample") == aux["kSpaceCenterSample"]
    assert seq.get_definition("SliceThickness") == pytest.approx(
        aux["SliceThickness"], rel=1e-3
    )


def test_the_first_and_last_flags_are_left_to_the_client():
    """They are derivable from the counters and the encoding limits, and the
    ISMRMRD client sets them there. Writing them here would put an extension
    row on every acquisition to say something already known."""
    labels = design().evaluate_labels(evolution="adc")
    assert not {"FIRSTLIN", "LASTLIN", "FIRSTPAR", "LASTPAR"} & set(labels)


# ----------------------------------------------------------------------
# Parallel imaging
# ----------------------------------------------------------------------


def test_the_calibration_rectangle_is_flagged_and_nothing_else_is():
    seq = design(acceleration=2, acceleration_z=2)
    labels = seq.evaluate_labels(evolution="adc")
    pairs = acquired_pairs(seq)
    flagged = {
        pair for pair, ima in zip(pairs, labels["IMA"], strict=False) if ima == 1
    }
    assert flagged == _rectangle()


def test_the_calibration_rectangle_closes_a_segment_of_its_own():
    """SEG separates calibration from the rest, so a reconstruction can
    calibrate the moment the rectangle is complete."""
    seq = design(acceleration=2)
    labels = seq.evaluate_labels(evolution="adc")
    pairs = acquired_pairs(seq)
    in_rectangle = np.asarray([pair in _rectangle() for pair in pairs])
    assert np.array_equal(labels["SEG"] == 0, in_rectangle)

    # Where each segment closes -- the boundary the client turns into
    # LAST_IN_SEGMENT, from the counter and its limits.
    segments = np.asarray(labels["SEG"])
    closing = np.flatnonzero(np.diff(segments, append=segments[-1] + 1) != 0)
    n_calibration = int(in_rectangle.sum())
    assert closing.tolist() == [n_calibration - 1, len(pairs) - 1]


def test_undersampling_skips_pairs_on_both_axes():
    seq = design(acceleration=2, acceleration_z=2, n_acs=0, n_acs_z=0)
    lines = {line for line, _ in acquired_pairs(seq)}
    partitions = {partition for _, partition in acquired_pairs(seq)}
    assert all(line % 2 == 0 for line in lines)
    assert all(partition % 2 == 0 for partition in partitions)


def test_partial_fourier_drops_the_lines_before_the_centre():
    full = {line for line, _ in acquired_pairs(design())}
    partial = {line for line, _ in acquired_pairs(design(partial_fourier=0.75))}
    dropped = full - partial
    assert dropped and all(line < N_Y // 2 - 6 // 2 for line in dropped)


# ----------------------------------------------------------------------
# Steady state
# ----------------------------------------------------------------------


def test_dummy_repetitions_are_played_without_acquiring():
    """They cost time and reach the steady state, and carry no data."""
    with_dummies = design(n_dummy=4)
    without = design(n_dummy=0)
    n_acqs = len(without.evaluate_labels(evolution="adc")["LIN"])
    assert len(with_dummies.evaluate_labels(evolution="adc")["LIN"]) == n_acqs
    tr = with_dummies.get_definition("TR")
    assert with_dummies.duration()[0] - without.duration()[0] == pytest.approx(4 * tr)


# ----------------------------------------------------------------------
# FOV offset
# ----------------------------------------------------------------------


def test_no_offset_leaves_the_sequence_exactly_as_it_was():
    plain = design()
    still = design(fov_offset=(0.0, 0.0, 0.0))
    assert plain.num_blocks == still.num_blocks
    assert plain.get_block(1).rf.freq_offset == still.get_block(1).rf.freq_offset


def test_a_slab_offset_is_a_frequency_on_the_pulse_and_costs_no_shape():
    """The selection gradient is constant across the pulse, so the shift is
    the classic slice-offset pair and the waveforms are shared."""
    plain = design()
    shifted = design(fov_offset=(0.0, 0.0, 20e-3))
    assert shifted.get_block(1).rf.freq_offset != plain.get_block(1).rf.freq_offset
    assert shifted._native.num_shapes() == plain._native.num_shapes()

    in_plane = design(fov_offset=(20e-3, 0.0, 0.0))
    n_first_adc = next(
        i
        for i in range(1, in_plane.num_blocks + 1)
        if in_plane.get_block(i).adc is not None
    )
    assert in_plane.get_block(n_first_adc).adc.freq_offset != pytest.approx(
        plain.get_block(n_first_adc).adc.freq_offset
    )


# ----------------------------------------------------------------------
# Averages and declarations
# ----------------------------------------------------------------------


def test_the_averages_are_written_out_rather_than_left_to_the_interpreter():
    once = design()
    twice = design(n_averages=2)
    labels = twice.evaluate_labels(evolution="adc")
    assert len(labels["LIN"]) == 2 * len(once.evaluate_labels(evolution="adc")["LIN"])
    assert sorted(set(labels["AVG"].tolist())) == [0, 1]


def test_the_prescribed_grid_is_declared():
    seq = design(fov=(192e-3, 160e-3))
    assert seq.get_definition("FOV") == pytest.approx([192e-3, 160e-3, SLAB])
    assert seq.get_definition("Matrix") == pytest.approx([N_X, N_Y, N_Z])
    assert seq.get_definition("Name") == "gre_3d"


# ----------------------------------------------------------------------
# The protocol contract
# ----------------------------------------------------------------------


def test_the_default_protocol_round_trips():
    protocol = gre3D_sequence.PLUGIN.get_default_protocol(pp.Opts())
    assert protocol_to_dict(dict_to_protocol(protocol)) == protocol


def test_a_feasible_protocol_reports_its_scan_time():
    system = pp.Opts()
    report = gre3D_sequence.PLUGIN.validate_protocol(
        system, gre3D_sequence.PLUGIN.get_default_protocol(system)
    )
    assert report["valid"] is True
    assert report["duration"] > 0


def test_an_impossible_echo_time_is_refused_by_name():
    system = pp.Opts()
    protocol = gre3D_sequence.PLUGIN.get_default_protocol(system)
    params.set_protocol_value(protocol, UIParam.TE, 0.5)
    report = gre3D_sequence.PLUGIN.validate_protocol(system, protocol)
    assert report["valid"] is False
    assert report["duration"] is None
    assert "TE" in report["info"]


# ----------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------


def test_a_written_sequence_reads_back_with_its_grid(tmp_path):
    seq = design()
    path = tmp_path / "gre_3d.seq"
    seq.write(str(path))

    back = pp.Sequence()
    back.read(str(path))
    assert back.get_definition("Matrix") == pytest.approx([N_X, N_Y, N_Z])
    assert back.num_blocks == seq.num_blocks


# ----------------------------------------------------------------------
# Elliptical sampling
# ----------------------------------------------------------------------


def test_elliptical_sampling_drops_the_corners():
    """The default elliptical support omits the ky-kz corners a round object
    never fills; turning it off samples the full rectangle."""
    corners = {(0, 0), (0, N_Z - 1), (N_Y - 1, 0), (N_Y - 1, N_Z - 1)}
    disk = set(kernel(elliptical=True).pairs)
    rectangle = set(kernel(elliptical=False).pairs)
    assert not (corners & disk)
    assert corners <= rectangle
    assert disk < rectangle
