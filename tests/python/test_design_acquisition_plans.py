"""High-level acquisition-sampling factories."""

from __future__ import annotations

import numpy as np
import pulserver
import pulserver.pypulseq as pp
import pytest


def test_cartesian_sampling_uses_matrix_acceleration_train_and_outer_loops():
    gre = pp.make_cartesian_sampling(
        (64, 16),
        acceleration=2,
        calibration=2,
        num_slices=4,
        slice_spacing_m=5e-3,
        frames=2,
    )
    assert isinstance(gre, pp.AcquisitionPlan)
    assert isinstance(gre, pulserver.AcquisitionPlan)
    assert gre.matrix == (64, 16)
    assert gre.mask.shape == (16,)
    assert gre.sampling.inner_lengths == (1,) * gre.sampling.n_shots
    assert len(gre) == 2 * 4 * gre.sampling.n_shots
    assert gre[0].lin_idx == int(gre[0].coordinates[0, 0])
    assert gre[0].slice_indices == (0,)

    fse = pp.make_cartesian_sampling((64, 16), train_length=4, ordering="radial")
    assert all(length <= 4 for length in fse.sampling.inner_lengths)
    assert isinstance(fse[0].lin_idx, np.ndarray)

    mprage = pp.make_cartesian_sampling((64, 16, 8), train_length=8, ordering="centric")
    first_radius = np.linalg.norm(mprage[0].coordinates[0] - np.array([7.5, 3.5]))
    last_radius = np.linalg.norm(mprage[-1].coordinates[-1] - np.array([7.5, 3.5]))
    assert first_radius < last_radius


def test_cartesian_sampling_supports_3d_masks_and_custom_low_level_composition():
    plan = pp.make_cartesian_sampling(
        (64, 24, 12),
        acceleration=(2, 3),
        calibration=(4, 2),
        mask="caipi",
        caipi_shift=1,
    )
    assert plan.mask.shape == (24, 12)
    assert plan[0].par_idx is not None
    assert plan.slice_frequency_table_hz(42_000).shape == (0, 0)

    mask = np.zeros(8, dtype=bool)
    mask[::2] = True
    low_level = pp.SamplingPattern.from_mask(mask, train_length=2)
    groups = pp.make_slice_sampling(2, 5e-3)
    assert len(groups) == 2
    assert len(low_level) == low_level.n_shots
    assert not hasattr(pp.AcquisitionPlan, "from_sampling")


def test_slice_frequency_table_is_part_of_high_level_plan():
    plan = pp.make_cartesian_sampling(
        (64, 16),
        num_slices=8,
        slice_spacing_m=5e-3,
        slice_order="interleaved",
        sms_factor=2,
    )
    table = plan.slice_frequency_table_hz(1_000.0)
    assert table.shape == (4, 2)
    assert np.array_equal(table[0], plan[0].frequency_offsets_hz(1_000.0))
    assert sorted(index for group in plan.slice_groups for index in group.slice_indices) == list(range(8))
    schedule = pp.make_slice_sampling(8, 5e-3, sms_factor=2)
    assert schedule.mask.shape == (4, 8)
    assert np.all(schedule.mask.sum(axis=1) == 2)
    assert np.array_equal(schedule.frequency_table_hz(1_000.0), table)


def test_epi_sampling_covers_2d_3d_and_multiple_volumes():
    epi2d = pp.make_epi_sampling(
        (64, 16),
        acceleration=2,
        segments=2,
        frames=3,
        num_slices=2,
        slice_spacing_m=5e-3,
    )
    assert epi2d.n_frames == 3
    assert epi2d.sampling.n_samples == 8
    assert len(epi2d) == 3 * 2 * 2
    assert np.isscalar(epi2d[0].lin_idx)
    assert np.array_equal(epi2d[0].relative[0], [0])
    assert epi2d[0].labels.shape[1] == 1

    epi3d = pp.make_epi_sampling(
        (64, 16, 8),
        acceleration=(2, 2),
        segments=2,
        caipi_shift=1,
        frames=4,
    )
    assert epi3d.n_frames == 4
    assert len(epi3d) == 4 * epi3d.sampling.n_shots
    assert np.isscalar(epi3d[0].lin_idx)
    assert np.isscalar(epi3d[0].par_idx)
    assert epi3d[0].labels.shape[1] == 2


def test_noncartesian_2d_angles_can_advance_across_slices_and_frames():
    continuous = pp.make_noncartesian_2d_sampling(
        (64, 64),
        views_per_frame=4,
        frames=2,
        num_slices=2,
        slice_spacing_m=5e-3,
        scheme="golden",
    )
    assert len(continuous) == 16
    assert all(entry.rotation.shape == (3, 3) for entry in continuous)
    first_angles = [continuous[index].coordinates[0, 0] for index in (0, 4, 8)]
    assert len(set(np.round(first_angles, 12))) == 3

    repeated = pp.make_noncartesian_2d_sampling(
        (64, 64),
        views_per_frame=4,
        frames=2,
        num_slices=2,
        slice_spacing_m=5e-3,
        continuous=False,
    )
    assert repeated[0].coordinates[0, 0] == pytest.approx(repeated[4].coordinates[0, 0])
    assert repeated[0].coordinates[0, 0] == pytest.approx(repeated[8].coordinates[0, 0])


def test_stack_angles_advance_across_partitions_and_projection_covers_sphere():
    stack = pp.make_noncartesian_stack_sampling(
        (64, 64, 3),
        views_per_partition=4,
        frames=2,
    )
    assert len(stack) == 24
    assert [stack[index].par_idx for index in (0, 4, 8)] == [1, 0, 2]
    assert len({round(stack[index].coordinates[0, 0], 12) for index in (0, 4, 8, 12)}) == 4

    projection = pp.make_noncartesian_projection_sampling(
        (32, 32, 32),
        views_per_frame=8,
        frames=2,
    )
    assert len(projection) == 16
    directions = np.concatenate([entry.coordinates for entry in projection])
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)
    assert np.ptp(directions[:, 2]) > 1.0
    assert all(entry.par_idx is None for entry in projection)
