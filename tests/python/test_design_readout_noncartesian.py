"""Tests for class-based non-Cartesian readouts."""

from __future__ import annotations

import numpy as np
import pytest

pp = pytest.importorskip("pypulseq")

from pulserver.pypulseq import _readout as readout  # noqa: E402
from pulserver.pypulseq import traj2grad  # noqa: E402

OPTS_KW = dict(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")


def _opts():
    return pp.Opts(**OPTS_KW)


def _labels(seq, name):
    result = []
    for block_id in range(1, len(seq.block_events) + 1):
        block = seq.get_block(block_id)
        for label in (block.label or {}).values():
            if label.label == name:
                result.append((label.type, label.value))
    return result


def test_traj2grad_obeys_vector_limits():
    opts = _opts()
    u = np.linspace(0.0, 1.0, 256)
    path = 50.0 * u[:, None] * np.column_stack((np.cos(6 * np.pi * u), np.sin(6 * np.pi * u)))
    gradient = traj2grad(path, opts, end_at_zero=False)
    assert gradient.shape[1] == 3
    assert np.max(np.linalg.norm(gradient, axis=1)) <= opts.max_grad * 1.001
    slew = np.diff(gradient, axis=0) / opts.grad_raster_time
    assert np.max(np.linalg.norm(slew, axis=1)) <= opts.max_slew * 1.001


def test_numpy_trajectory_generators_have_expected_endpoints():
    radial = readout.radial_trajectory(0.2, 32)
    assert np.allclose(radial[[0, -1], 0], [-80.0, 80.0])
    assert np.all(radial[:, 1] == 0.0)

    spiral = readout.spiral_trajectory(0.2, 32, 8, density="constant", num_points=128)
    assert np.allclose(spiral[0], 0.0)
    assert np.linalg.norm(spiral[-1]) == pytest.approx(80.0)

    rosette = readout.rosette_trajectory(0.2, 32, num_points=129)
    assert np.allclose(rosette[[0, -1]], 0.0, atol=1e-12)


def test_rosette_trajectory_shape_parameters_are_internal_to_base_interleave():
    open_petals = readout.rosette_trajectory(0.2, 32, petals=5, angular_frequency_ratio=3 / 5, num_points=129)
    tight_petals = readout.rosette_trajectory(0.2, 32, petals=5, angular_frequency_ratio=7 / 5, num_points=129)
    more_petals = readout.rosette_trajectory(0.2, 32, petals=7, angular_frequency_ratio=3 / 5, num_points=129)
    assert not np.allclose(open_petals, tight_petals)
    assert not np.allclose(open_petals, more_petals)
    with pytest.raises(ValueError, match="angular_frequency_ratio"):
        readout.rosette_trajectory(0.2, 32, angular_frequency_ratio=0.0)


def test_spiral_density_modes_and_validation():
    constant = readout.spiral_trajectory(0.22, 32, 8, density="constant", num_points=128)
    variable = readout.spiral_trajectory(0.22, 32, 8, density="variable", num_points=128)
    dual = readout.spiral_trajectory(
        0.22,
        32,
        8,
        density="dual",
        inner_design_interleaves=8,
        outer_design_interleaves=16,
        transition_speed=20,
        num_points=128,
    )
    assert not np.allclose(constant, variable)
    assert not np.allclose(variable, dual)
    with pytest.raises(ValueError, match="outer_design_interleaves"):
        readout.spiral_trajectory(0.22, 32, 8, density="dual")


@pytest.mark.parametrize(
    ("variant", "has_pre", "has_rew"),
    [("outward", False, True), ("inward", True, False), ("in_out", True, True)],
)
def test_spiral_variants_have_expected_winders_and_refocus(variant, has_pre, has_rew):
    opts = _opts()
    arm = readout.Spiral(opts, 0.22, 32, 8, variant=variant, num_points=256)
    assert arm.has_prewinder is has_pre
    assert arm.has_rewinder is has_rew
    assert arm.design_interleaves == 8
    if variant == "in_out":
        assert arm.n_samples == 64

    seq = pp.Sequence(opts)
    readout.NonCartesian2D(arm)(seq, rotation_idx=3)
    _, k_full, *_ = seq.calculate_kspace()
    assert np.allclose(k_full[:, -1], 0.0, atol=1e-6)
    assert seq.check_timing()[0]
    assert _labels(seq, "LIN") == [("labelset", 3)]


def test_radial_has_both_winders_and_nyquist_shot_count():
    opts = _opts()
    spoke = readout.Radial(opts, 0.22, 64)
    assert spoke.has_prewinder and spoke.has_rewinder
    assert spoke.recommended_rotations == int(np.ceil(np.pi * 64 / 2))
    seq = pp.Sequence(opts)
    readout.NonCartesian2D(spoke)(seq)
    _, k_full, *_ = seq.calculate_kspace()
    assert np.allclose(k_full[:, -1], 0.0, atol=1e-6)
    assert seq.check_timing()[0]


def test_rosette_has_no_winders_and_derives_adc_sampling():
    arm = readout.Rosette(_opts(), 0.22, 32, petals=5)
    assert not arm.has_prewinder
    assert not arm.has_rewinder
    assert arm.design_interleaves is None
    assert arm.recommended_rotations is None
    assert arm.n_samples == arm.petals * arm.samples_per_petal
    assert arm.n_samples > 32
    assert arm.trajectory.shape == (arm.n_samples, 2)
    assert arm.design_trajectory.shape == (2049, 2)
    assert arm.bandwidth_hz_px >= arm.requested_bandwidth_hz_px
    acquired_steps = np.linalg.norm(np.diff(arm.trajectory, axis=0), axis=1)
    assert np.max(acquired_steps) <= 1.0 / 0.22 + 1e-9

    seq = pp.Sequence(arm.system)
    readout.NonCartesian2D(arm)(seq, rotation_idx=2)
    assert seq.check_timing()[0]
    assert all(abs(gradient.area) < 1e-9 for gradient in arm.gradients)
    _, k_full, *_ = seq.calculate_kspace()
    assert np.allclose(k_full[:, -1], 0.0, atol=1e-3)


def test_rosette_acquires_every_requested_petal():
    arm = readout.Rosette(_opts(), 0.22, 64, petals=5, derate=False)
    radius = np.linalg.norm(arm.trajectory, axis=1)
    kmax = 64 / (2.0 * 0.22)

    # Every equal-duration petal segment reaches the requested k-space edge;
    # the ADC also reaches the final return to the centre rather than ending
    # a dwell-raster-sized tail early.
    for petal in np.array_split(radius, arm.petals):
        assert np.max(petal) >= 0.98 * kmax
    assert radius[-1] <= 0.05 * kmax


def test_rosette_echo_spacing_stretches_gradient_and_rejects_impossible_request():
    base = readout.Rosette(_opts(), 0.22, 32, petals=5, derate=False)
    target_spacing = 1.5 * base.min_echo_spacing_s
    stretched = readout.Rosette(_opts(), 0.22, 32, petals=5, echo_spacing_s=target_spacing, derate=False)
    assert stretched.echo_spacing_s >= target_spacing
    assert stretched.echo_spacing_s < target_spacing + stretched.system.grad_raster_time / stretched.petals
    base_peak = np.max(np.hypot(base.gx.waveform, base.gy.waveform))
    stretched_peak = np.max(np.hypot(stretched.gx.waveform, stretched.gy.waveform))
    assert stretched_peak < base_peak
    with pytest.raises(ValueError, match="minimum feasible"):
        readout.Rosette(
            _opts(),
            0.22,
            32,
            petals=5,
            echo_spacing_s=0.5 * base.min_echo_spacing_s,
            derate=False,
        )


def test_slice_rephasing_is_right_aligned_with_prewinder():
    opts = _opts()
    arm = readout.Spiral(opts, 0.22, 32, 8, variant="inward", num_points=256)
    gz = pp.make_trapezoid(channel="z", area=-30.0, system=opts)
    seq = pp.Sequence(opts)
    readout.NonCartesian2D(arm, slice_rephasing=gz)(seq)
    block = seq.get_block(1)
    duration = pp.calc_duration(block)
    assert pp.calc_duration(block.gz) == pytest.approx(duration)
    assert pp.calc_duration(block.gx) == pytest.approx(duration)


def test_multiecho_labels_set_then_increment():
    opts = _opts()
    arm = readout.Radial(opts, 0.22, 32)
    seq = pp.Sequence(opts)
    readout.NonCartesian2D(arm, num_echoes=3)(seq)
    assert _labels(seq, "ECO") == [("labelset", 0), ("labelinc", 1), ("labelinc", 1)]


def test_stack_generates_full_phase_template_and_labels_indices():
    opts = _opts()
    arm = readout.Spiral(opts, 0.22, 32, 8, variant="in_out", num_points=256)
    gz_reph = pp.make_trapezoid(channel="z", area=-20.0, system=opts)
    stack = readout.StackOfTrajectories(
        arm,
        fov_z_m=0.16,
        nz=8,
        slice_rephasing=gz_reph,
        num_echoes=2,
    )
    assert stack._phase_template is not None
    seq = pp.Sequence(opts)
    stack(seq, rotation_idx=5, phase_idx=2)
    assert seq.check_timing()[0]
    assert seq.duration()[0] == pytest.approx(stack.duration)
    assert _labels(seq, "LIN") == [("labelset", 5)]
    assert _labels(seq, "PAR") == [("labelset", 2)]
    assert _labels(seq, "ECO") == [("labelset", 0), ("labelinc", 1)]
    _, k_full, *_ = seq.calculate_kspace()
    # Stack phase encoding is rewound; only the caller-provided slice
    # rephasing moment remains in this isolated readout-train test.
    assert np.allclose(k_full[:, -1], [0.0, 0.0, gz_reph.area], atol=1e-6)
    with pytest.raises(IndexError):
        stack(pp.Sequence(opts), phase_idx=8)


def test_projection_accepts_matrix_rotation_with_fast_sequence():
    arm = readout.Radial(_opts(), 0.22, 16)
    sequence = readout.Projection(arm)(rotation=np.eye(3), rotation_idx=4)
    assert len(sequence.block_events) == 3


def test_arbitrary_native_3d_path():
    opts = _opts()
    u = np.linspace(0.0, 1.0, 128)
    path = np.column_stack((30 * u * np.cos(2 * np.pi * u), 30 * u * np.sin(2 * np.pi * u), 20 * u))
    shot = readout.Arbitrary(opts, path, matrix=32)
    train = readout.NonCartesian3D(shot)
    seq = pp.Sequence(opts)
    train(seq, rotation_idx=1)
    assert shot.gz is not None
    assert seq.check_timing()[0]
    _, k_full, *_ = seq.calculate_kspace()
    assert np.allclose(k_full[:, -1], 0.0, atol=1e-6)
