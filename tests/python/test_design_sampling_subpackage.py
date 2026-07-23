"""Structured sampling-plan and readout-integration tests."""

from __future__ import annotations

import numpy as np
import pytest
from pulserver.design import _readout as readout
from pulserver.design import _sampling as sampling


def _labels(blocks, name):
    values = []
    for block in blocks:
        for event in block:
            if getattr(event, "type", None) in ("labelset", "labelinc") and event.label == name:
                values.append((event.type, event.value))
    return values


def test_pattern_separates_locations_and_shots_and_is_immutable():
    support = np.array([[0, 0], [1, 2], [3, 4]])
    mask = np.zeros((4, 5), bool)
    mask[tuple(support.T)] = True
    pattern = sampling.ScanLoop(support, (np.array([2, 0]), np.array([1])), mask)
    support[:] = -1
    mask[:] = False
    assert pattern.n_shots == 2
    assert pattern.n_samples == 3
    assert pattern.shot_lengths == (2, 1)
    assert np.array_equal(pattern[0], [[3, 4], [0, 0]])
    assert np.array_equal(pattern.relative(0), [[0, 0], [-3, -4]])
    assert np.array_equal(pattern.increments(0), [[-3, -4]])
    assert np.array_equal(pattern.flatten(), [[3, 4], [0, 0], [1, 2]])
    with pytest.raises(ValueError):
        pattern.positions[0, 0] = 9
    with pytest.raises(ValueError):
        pattern.shots[0][0] = 9


def test_pattern_validates_mask_and_indices():
    with pytest.raises(ValueError, match="mask sample count"):
        sampling.ScanLoop([[0]], (np.array([0]),), np.zeros(2, bool))
    with pytest.raises(IndexError):
        sampling.ScanLoop([[0]], (np.array([1]),))
    empty = sampling.ScanLoop(np.empty((0, 2)), ())
    assert empty.flatten().shape == (0, 2)


def test_generic_ordering():
    assert np.array_equal(sampling.ordering.interleaved(6), [0, 2, 4, 1, 3, 5])
    assert np.array_equal(sampling.ordering.center_out(5), [2, 1, 3, 0, 4])
    assert np.array_equal(sampling.ordering.center_out(4), [1, 2, 0, 3])
    assert np.array_equal(sampling.ordering.outside_in(4), [3, 0, 2, 1])


@pytest.mark.parametrize("ordering", ["linear", "radial", "radial_adaptive", "shuffling"])
def test_cartesian_pattern_covers_mask_once(ordering):
    mask = np.zeros((7, 5), bool)
    mask[::2, ::2] = True
    pattern = sampling.ScanLoop.from_mask(mask, train_length=4, ordering=ordering, seed=3)
    assert np.array_equal(pattern.mask, mask)
    assert pattern.n_samples == mask.sum()
    assert sorted(np.concatenate(pattern.shots)) == list(range(mask.sum()))
    assert pattern.shot_lengths[-1] <= 4


def test_sampling_pattern_and_ordering_accept_masks():
    mask = np.zeros((8, 6), dtype=bool)
    mask[::2, ::2] = True
    pattern = sampling.ScanLoop.from_mask(mask, train_length=3, ordering="radial")
    assert np.array_equal(pattern.mask, mask)
    shots = sampling.make_radial_order(mask, 3)
    assert sorted(index for shot in shots for index in shot) == list(range(mask.sum()))


def test_poisson_cpp_is_deterministic_and_keeps_acs():
    first = sampling.make_poisson_disc_mask((48, 40), 4.0, calib=(8, 6), seed=9, tol=0.15)
    second = sampling.make_poisson_disc_mask((48, 40), 4.0, calib=(8, 6), seed=9, tol=0.15)
    assert first.dtype == bool and first.shape == (48, 40)
    assert np.array_equal(first, second)
    assert first[20:28, 17:23].all()
    assert abs(first.size / first.sum() - 4.0) < 0.15


def test_relative_epi_plan_and_repeated_samples():
    pattern = sampling.ScanLoop.from_relative_shifts(
        [[2, 1], [5, 1]],
        np.array([[[0, 0], [1, 1], [0, 0]], [[0, 0], [-1, 2], [-2, 2]]]),
        shape=(8, 8),
    )
    assert pattern.n_samples == 6
    assert len(pattern.positions) == 5
    assert np.array_equal(pattern.relative(0), [[0, 0], [1, 1], [0, 0]])
    assert np.array_equal(pattern.increments(0), [[1, 1], [-1, -1]])
    with pytest.raises(IndexError):
        sampling.ScanLoop.from_relative_shifts([[0]], [[-1]], shape=(4,))


def test_segmented_skipped_caipi_has_exact_union_and_order():
    pattern = sampling.make_skipped_caipi_order((8, 8), acceleration=(2, 2), caipi_shift=1, segments=2)
    expected = sampling.make_caipirinha_mask((8, 8), 2, 2, delta=1)
    assert np.array_equal(pattern.mask, expected)
    assert pattern.n_samples == expected.sum()
    assert np.array_equal(pattern[0], [[0, 0], [4, 2]])
    assert np.array_equal(pattern.relative(0), [[0, 0], [4, 2]])


def test_epi_interleaving_accepts_arbitrary_outer_dimensions():
    pattern = sampling.epi.interleaved((8, 4), acceleration=(2, 2), num_shots=2, axis=0)
    assert pattern.n_samples == 8
    assert pattern.shot_lengths == (4, 4)
    assert all(np.all(point % 2 == 0) for point in pattern.flatten())


def test_radial_tilts_and_raga_keep_locations_and_temporal_order():
    golden = sampling.make_radial_tilt(4, scheme="golden")
    assert np.allclose(golden.flatten().ravel(), np.mod(np.arange(4) * np.pi / ((1 + np.sqrt(5)) / 2), np.pi))
    tiny = sampling.make_radial_tilt(3, scheme="tiny_golden", tiny_index=2)
    assert tiny.flatten()[1, 0] == pytest.approx(np.pi / ((1 + np.sqrt(5)) / 2 + 1))
    raga = sampling.make_radial_tilt(8, scheme="raga", tiny_index=1, approximation_order=5, segment_length=3)
    assert len(raga.positions) == 5
    assert np.array_equal(raga.flatten().ravel(), raga.positions[[0, 3, 1, 4, 2, 0, 3, 1], 0])
    assert raga.shot_lengths == (3, 3, 2)


def test_spherical_orders_and_rotation_matrices():
    means = sampling.make_golden_means_3d_tilt(13, segment_length=5)
    assert means.shot_lengths == (5, 5, 3)
    assert np.allclose(np.linalg.norm(means.positions, axis=1), 1)
    phy = sampling.make_spiral_phyllotaxis_tilt(15, 5)
    assert np.array_equal(phy.shots[2], [2, 7, 12])
    assert np.allclose(np.linalg.norm(phy.positions, axis=1), 1)
    with pytest.raises(ValueError, match="Fibonacci"):
        sampling.make_spiral_phyllotaxis_tilt(12, 4)
    directions = np.array([[1, 0, 0], [-1, 0, 0], [0, 0, 1]], float)
    rotations = sampling.ScanLoop(directions, (np.arange(3),)).to_rotations()
    assert np.allclose(rotations @ np.array([1.0, 0, 0]), directions)
    assert np.allclose(rotations @ np.swapaxes(rotations, 1, 2), np.eye(3))
    assert np.allclose(np.linalg.det(rotations), 1)


def test_slice_loop_orders_sms_shots_and_converts_positions_to_frequencies():
    loop = sampling.make_slice_loop(8, 0.005, order="interleaved", sms_factor=2)
    assert [int(shot[0]) for shot in loop.shots] == [0, 2, 1, 3]
    assert list(loop.shots[0]) == [0, 4]
    assert sorted(index for shot in loop.shots for index in shot) == list(range(8))
    offsets = loop.to_frequencies(1000)
    assert np.allclose(offsets[loop.shots[0]], [-17.5, 2.5])
    with pytest.raises(ValueError):
        sampling.make_slice_loop(7, 0.005, sms_factor=2)


def test_line_and_epi_emit_absolute_labels_from_sampling_plan():
    pp = pytest.importorskip("pypulseq")
    opts = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    line = readout.Line3D(opts, (0.22, 0.22, 0.22), (16, 16, 16), spoil_position="none")
    blocks = tuple(line.set_state(lin_idx=3, par_idx=7))
    assert _labels(blocks, "LIN") == [("labelset", 3)]
    assert _labels(blocks, "PAR") == [("labelset", 7)]

    plan = sampling.ScanLoop.from_relative_shifts([[4]], [[0], [2], [1]], shape=(16,))
    epi = readout.Epi2D(opts, (0.22, 0.22), (16, 16), 1, plan.relative(0))
    blocks = tuple(epi.set_state(lin_idx=4, labels=plan[0]))
    assert _labels(blocks, "LIN") == [("labelset", 4), ("labelset", 6), ("labelset", 5)]
    with pytest.raises(IndexError, match="LIN labels"):
        tuple(epi.set_state(lin_idx=4, labels=[4, 16, 5]))
