"""Encoding axes: what a scan loop's columns mean, and the counters they emit."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pypulseq")

import pulserver
import pulserver.design as design
import pulserver.pypulseq as pp
from pulserver import EncodingAxis, ScanLoop




def test_encoding_axis_declares_columns_and_rejects_nonsense():
    assert EncodingAxis("LIN").columns == 1
    assert EncodingAxis("LIN", kind="direction").columns == 3
    assert EncodingAxis("LIN", size=128).is_counter
    assert not EncodingAxis("BIN").is_counter

    with pytest.raises(ValueError, match="kind"):
        EncodingAxis("LIN", kind="wavenumber")
    with pytest.raises(ValueError, match="size"):
        EncodingAxis("LIN", size=0)
    with pytest.raises(TypeError, match="label"):
        EncodingAxis("")


def test_axes_must_account_for_every_position_column():
    positions = np.zeros((4, 2), dtype=np.intp)
    with pytest.raises(ValueError, match="position column"):
        ScanLoop(positions, (np.arange(4),), None, (EncodingAxis("LIN"),))
    with pytest.raises(TypeError, match="EncodingAxis"):
        ScanLoop(positions, (np.arange(4),), None, ("LIN", "PAR"))


def test_bare_positions_infer_the_axes_they_always_implied():
    cartesian = ScanLoop(np.zeros((4, 2), dtype=np.intp), (np.arange(4),))
    assert [axis.label for axis in cartesian.axes] == ["LIN", "PAR"]
    assert {axis.kind for axis in cartesian.axes} == {"index"}

    directions = ScanLoop(np.eye(3), (np.arange(3),))
    assert [(axis.label, axis.kind) for axis in directions.axes] == [("LIN", "direction")]
    # ...and the historical column-count dispatch still applies to them.
    assert directions.to_rotations().shape == (3, 3, 3)


def test_shipped_factories_declare_their_own_axes():
    cartesian = design.make_cartesian_sampling((64, 32, 8), acceleration=(2, 2))
    assert [(axis.label, axis.kind, axis.size) for axis in cartesian.axes] == [
        ("LIN", "index", 32),
        ("PAR", "index", 8),
    ]

    epi = design.make_epi_sampling((64, 32), acceleration=2, segments=2)
    assert [(axis.label, axis.size) for axis in epi.axes] == [("LIN", 32)]

    views = design.make_noncartesian_2d_sampling((64, 64), views=8)
    assert [(axis.label, axis.kind) for axis in views.axes] == [("LIN", "angle")]

    projections = design.make_noncartesian_projection_sampling((32, 32, 32), views=16)
    assert [(axis.label, axis.kind) for axis in projections.axes] == [("LIN", "direction")]

    slices = design.make_slice_loop(6, 3e-3)
    assert [(axis.label, axis.kind, axis.size) for axis in slices.axes] == [("SLC", "position", 6)]


def test_to_scales_defaults_to_the_encoded_matrix_extent():
    # The extent comes from the matrix, not from the sampled support, so an
    # accelerated loop still scales against the full matrix.
    loop = design.make_cartesian_sampling((64, 8), acceleration=2)
    assert loop.n_positions == 4
    assert loop.to_scales()[:, 0].tolist() == pytest.approx([-1.0, -0.5, 0.0, 0.5])
    assert np.array_equal(loop.to_scales(), loop.to_scales((8,)))

    with pytest.raises(ValueError, match="index-valued positions"):
        design.make_slice_loop(4, 3e-3).to_scales()


def test_converters_refuse_a_loop_whose_kind_cannot_mean_that():
    with pytest.raises(ValueError, match="one-dimensional positions in metres"):
        design.make_noncartesian_2d_sampling((64, 64), views=4).to_frequencies(1000.0)
    with pytest.raises(ValueError, match="to_rotations"):
        design.make_cartesian_sampling((64, 8)).to_rotations()


def test_labels_are_the_position_value_on_an_index_axis_and_the_index_elsewhere():
    cartesian = design.make_cartesian_sampling((64, 8), acceleration=2)
    assert cartesian.label_state(2) == {"LIN": 4}

    # Metres, angles and inversion times are not counters, but their order is.
    slices = design.make_slice_loop(6, 3e-3)
    assert [int(shot[0]) for shot in slices.shots] == [0, 2, 4, 1, 3, 5]
    assert slices.label_state(1) == {"SLC": 2}

    views = design.make_noncartesian_2d_sampling((64, 64), views=8)
    assert views.label_state(3) == {"LIN": 3}


def test_sms_shot_labels_its_first_band_by_default():
    sms = design.make_slice_loop(6, 3e-3, sms_factor=2)
    assert sms.shot_lengths == (2, 2, 2)
    assert sms.label_state(0) == {"SLC": 0}
    assert sms.label_state(0, position=1) == {"SLC": 3}


def test_label_limits_predict_where_the_first_last_flags_will_fire():
    # The interpreter derives FIRST_IN_*/LAST_IN_* by comparing each
    # acquisition's counter against the observed range of that counter.
    loop = design.make_cartesian_sampling((64, 32), acceleration=2)
    assert loop.label_limits() == {"LIN": (0, 30)}

    volume = design.make_cartesian_sampling((64, 32, 8))
    assert volume.label_limits() == {"LIN": (0, 31), "PAR": (0, 7)}

    # A dimension that is looped but never labelled collapses to one index,
    # which is exactly the case where both flags fire on every acquisition.
    single = design.make_counter_loop(1, label="REP")
    assert single.label_limits() == {"REP": (0, 0)}


def test_with_axes_retargets_the_counter_without_touching_the_positions():
    views = design.make_noncartesian_2d_sampling((64, 64), views=8)
    phases = views.with_axes(EncodingAxis("PHS", kind="angle"))

    assert phases.label_state(3) == {"PHS": 3}
    assert np.array_equal(phases.positions, views.positions)
    assert phases.shots == views.shots
    assert views.axes[0].label == "LIN"  # unchanged


def test_counter_loop_covers_frames_contrasts_and_averages():
    frames = design.make_counter_loop(40, label="REP")
    assert isinstance(frames, pulserver.ScanLoop)
    assert len(frames) == 40
    assert frames.shot_lengths == (1,) * 40
    assert frames.label_state(7) == {"REP": 7}
    assert frames.label_limits() == {"REP": (0, 39)}

    inversions = design.make_counter_loop([0.1, 0.3, 1.0, 2.5], label="SET")
    assert float(inversions[2][0, 0]) == 1.0
    assert inversions.label_state(2) == {"SET": 2}
    assert inversions.axes[0].kind == "value"


def test_counter_loop_reorders_the_visits_without_renumbering_the_data():
    ordered = design.make_counter_loop(6, label="AVG", order="center_out")
    visited = [int(shot[0]) for shot in ordered.shots]
    assert visited == [2, 3, 1, 4, 0, 5]
    # The counter follows the position, so shot 0 is still average 2.
    assert ordered.label_state(0) == {"AVG": 2}
    assert sorted(visited) == list(range(6))


def test_counter_loop_validates_its_inputs():
    with pytest.raises(ValueError, match="positive count or a 1D table"):
        design.make_counter_loop(0, label="REP")
    with pytest.raises(ValueError, match="positive count or a 1D table"):
        design.make_counter_loop([[0.1, 0.2]], label="SET")
    with pytest.raises(ValueError, match="order"):
        design.make_counter_loop(4, label="REP", order="spiral")


def test_counter_loop_can_carry_metres_and_partial_extents():
    offsets = design.make_counter_loop([-5e-3, 0.0, 5e-3], label="SLC", kind="position")
    assert offsets.to_frequencies(1000.0).tolist() == pytest.approx([-5.0, 0.0, 5.0])

    resumed = design.make_counter_loop(10, label="REP", size=40)
    assert resumed.sizes == (40,)


def test_a_module_consumes_the_labels_the_loops_hand_it():
    from pulserver.design import _readout as readout

    frames = design.make_counter_loop(4, label="REP")
    slices = design.make_slice_loop(6, 3e-3)
    module = readout.Line2D(
        pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s"),
        (0.22, 0.22),
        (32, 32),
        spoil_position="none",
    )
    module.set_state(lin_idx=7, **frames.label_state(2), **slices.label_state(1))

    emitted = {event.label: event.value for event in module.blocks[0] if getattr(event, "type", "") == "labelset"}
    assert emitted == {"REP": 2, "SLC": 2}
    # ...and the readout keeps emitting the counter it owns itself.
    assert any(getattr(event, "label", None) == "LIN" and event.value == 7 for block in module.blocks for event in block)
