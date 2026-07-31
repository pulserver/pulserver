"""Matrix-driven sampling factories, and the explicit loops built on them."""

from __future__ import annotations

import numpy as np
import pulserver
import pulserver.design as design
import pulserver.pypulseq as pp
import pytest
from pulserver import ScanLoop
from pulserver.design import _lowlevel


def test_cartesian_sampling_covers_gre_fse_and_mprage_from_one_factory():
    gre = design.make_cartesian_sampling((64, 16), acceleration=2, calibration=2)
    assert isinstance(gre, ScanLoop)
    assert isinstance(gre, pulserver.ScanLoop)
    assert gre.mask.shape == (16,)
    assert gre.shot_lengths == (1,) * gre.n_shots
    assert gre.n_samples == int(gre.mask.sum())

    fse = design.make_cartesian_sampling((64, 16), train_length=4, ordering="radial")
    assert all(length <= 4 for length in fse.shot_lengths)
    assert fse[0].shape == (4, 1)

    mprage = design.make_cartesian_sampling((64, 16, 8), train_length=8, ordering="centric")
    centre = np.array([7.5, 3.5])
    assert np.linalg.norm(mprage[0][0] - centre) < np.linalg.norm(mprage[-1][-1] - centre)


def test_cartesian_sampling_supports_3d_masks_and_low_level_composition():
    pattern = design.make_cartesian_sampling(
        (64, 24, 12),
        acceleration=(2, 3),
        calibration=(4, 2),
        mask="caipi",
        caipi_shift=1,
    )
    assert pattern.mask.shape == (24, 12)
    assert pattern.n_dims == 2

    mask = np.zeros(8, dtype=bool)
    mask[::2] = True
    low_level = ScanLoop.from_mask(mask, train_length=2)
    assert len(low_level) == low_level.n_shots == 2

    with pytest.raises(ValueError, match="requires a 3D matrix"):
        design.make_cartesian_sampling((64, 24), mask="caipi")
    with pytest.raises(ValueError, match="calibration cannot exceed"):
        design.make_cartesian_sampling((64, 24), calibration=32)


def test_encoding_scales_drive_the_phase_encoding_template_to_the_right_area():
    system = pp.Opts()
    fov, ny = 0.24, 16
    loop = design.make_cartesian_sampling((64, ny))
    template = design.make_phase_encoding(system, "y", fov / ny)

    scales = loop.to_scales((ny,))
    assert scales.min() == pytest.approx(-1.0)
    assert np.abs(scales).max() <= 1.0
    # The whole point of the split: the factory owns the gradient, the loop
    # owns the ordering, and scaling one by the other lands on (ky - ny/2)/fov.
    for shot in loop:
        ky = int(shot[0, 0])
        assert pp.scale_grad(template, scales[ky, 0]).area == pytest.approx((ky - ny / 2) / fov)

    assert design.calc_encoding_scales(np.arange(ny), ny) == pytest.approx(scales[:, 0])
    assert design.calc_encoding_scales([[0, 4], [4, 0]], (8, 8)).tolist() == [[-1.0, 0.0], [0.0, -1.0]]
    with pytest.raises(ValueError, match="columns"):
        design.calc_encoding_scales([[0, 1]], (8,))


def test_encoding_scales_do_not_depend_on_which_views_are_played():
    # Acceleration and partial Fourier drop views from the loop; they must not
    # rescale the ones that remain, or an accelerated scan would encode a
    # different k-space than a full one at the same index.
    ny = 16
    full = design.make_cartesian_sampling((64, ny))
    accelerated = design.make_cartesian_sampling((64, ny), acceleration=4)
    partial = ScanLoop.from_mask(np.arange(ny) >= 4)

    reference = design.calc_encoding_scales(np.arange(ny), ny)
    assert full.to_scales((ny,))[:, 0] == pytest.approx(reference)
    for loop in (accelerated, partial):
        scales = loop.to_scales((ny,))
        for index, shot in enumerate(loop):
            ky = int(shot[0, 0])
            assert scales[loop.shots[index][0], 0] == pytest.approx(reference[ky])


def test_epi_sampling_covers_2d_and_3d_without_a_frame_loop():
    epi2d = design.make_epi_sampling((64, 16), acceleration=2, segments=2)
    assert epi2d.n_shots == 2
    assert epi2d.n_samples == 8
    assert np.array_equal(epi2d.relative(0)[0], [0])
    assert epi2d[0].shape[1] == 1

    epi3d = design.make_epi_sampling((64, 16, 8), acceleration=(2, 2), segments=2, caipi_shift=1)
    assert epi3d[0].shape[1] == 2
    assert epi3d.n_dims == 2

    with pytest.raises(ValueError, match="segments"):
        design.make_epi_sampling((64, 16), segments=0)


def test_noncartesian_factories_return_rotatable_patterns():
    inplane = design.make_noncartesian_2d_sampling((64, 64), views=4, scheme="golden")
    assert inplane.n_shots == 4
    assert inplane.to_rotations().shape == (4, 3, 3)

    projection = design.make_noncartesian_projection_sampling((32, 32, 32), views=8)
    directions = projection.flatten()
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)
    assert projection.to_rotations().shape == (8, 3, 3)

    assert design.make_noncartesian_2d_sampling((64, 64)).n_shots == int(np.ceil(np.pi * 64 / 2))
    with pytest.raises(ValueError, match="scheme"):
        design.make_noncartesian_projection_sampling((32, 32, 32), views=4, scheme="bogus")


def test_golden_angle_chronology_is_continuous_or_replayed_by_the_caller():
    n_frames, n_slices, n_views = 2, 2, 4
    pattern = design.make_noncartesian_2d_sampling((64, 64), views=n_frames * n_slices * n_views)

    # One iterator spanning the whole scan: every outer position sees fresh
    # angles, with no `continuous=` switch anywhere.
    chronology = iter(pattern.to_rotations())
    continuous = [next(chronology) for _ in range(n_frames) for _ in range(n_slices) for _ in range(n_views)]
    assert len({tuple(np.round(rotation[0], 12)) for rotation in continuous}) == len(continuous)

    # Re-iterating inside the loop replays the same set at every position.
    replayed = [
        rotation for _ in range(n_frames) for _ in range(n_slices) for rotation in pattern.to_rotations()[:n_views]
    ]
    assert np.allclose(replayed[0], replayed[n_views])


def test_slice_loop_is_the_same_type_as_the_k_space_loop():
    slices = design.make_slice_loop(8, 5e-3, order="interleaved", sms_factor=2)
    kspace = design.make_cartesian_sampling((64, 16))
    assert isinstance(slices, pulserver.ScanLoop)
    assert type(slices) is type(kspace)
    assert len(slices) == 4
    assert slices.shot_lengths == (2, 2, 2, 2)
    assert sorted(index for shot in slices.shots for index in shot) == list(range(8))

    # Positions are metres, so the converter is to_frequencies, not to_scales.
    offsets = slices.to_frequencies(1_000.0)
    assert offsets.shape == (8,)
    assert np.allclose(offsets[slices.shots[0]], slices[0][:, 0] * 1_000.0)
    # ...and each converter refuses positions it cannot mean anything for.
    with pytest.raises(ValueError, match="one-dimensional positions in metres"):
        design.make_cartesian_sampling((64, 16, 8)).to_frequencies(1_000.0)
    with pytest.raises(ValueError, match="spoke angles"):
        design.make_cartesian_sampling((64, 16, 8)).to_rotations()


def test_traversal_order_drives_partition_and_other_one_dimensional_loops():
    assert _lowlevel.calc_traversal_order(6, "interleaved").tolist() == [0, 2, 4, 1, 3, 5]
    assert _lowlevel.calc_traversal_order(5, "center_out").tolist() == [2, 1, 3, 0, 4]
    assert sorted(_lowlevel.calc_traversal_order(7, "random", seed=1)) == list(range(7))
    with pytest.raises(ValueError, match="unknown order"):
        _lowlevel.calc_traversal_order(4, "bogus")


def test_a_full_2d_multislice_loop_needs_only_two_scan_loops():
    system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    sampling = design.make_cartesian_sampling((32, 8), acceleration=2)
    slices = design.make_slice_loop(2, 5e-3)
    excitation = design.make_slice_selective_pulse(np.deg2rad(10), 5e-3, duration=1e-3, system=system)
    readout = design.make_line_readout(system, (0.22, 0.22), (32, 8), spoil_position="none")
    phases = iter(design.make_rf_spoiling_schedule(2 * len(slices) * len(sampling)))

    offsets = slices.to_frequencies(excitation.gradients[0].amplitude)
    seq = pp.Sequence._unstructured(system)
    for _frame in range(2):
        for slice_shot in slices.shots:
            for shot in sampling:
                phase = float(next(phases))
                excitation.set_state(
                    freq_offset_hz=float(offsets[slice_shot[0]]),
                    phase_offset_rad=phase,
                    SLC=int(slice_shot[0]),
                )
                for _block in excitation:
                    seq.add_block(*_block)
                readout.set_state(lin_idx=int(shot[0, 0]), phase_offset_rad=phase)
                for _block in readout:
                    seq.add_block(*_block)
    expected_shots = 2 * len(slices) * len(sampling)
    assert len(seq.block_events) == expected_shots * (excitation.num_blocks + readout.num_blocks)
