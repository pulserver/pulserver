from __future__ import annotations

import math

import numpy as np
import pypulseq as pp
import pytest
from pulserver.design._lowlevel import arbgrad
from pulserver.pypulseq import Sequence, make_rotation

try:
    from scipy.spatial.transform import Rotation
except ImportError:  # pragma: no cover - scipy is an existing pulserver dependency
    Rotation = None

FOV_M = 0.256
N_PIX = 128
GAMMA_HZ_PER_T = 42.5756e6
SLEW_LIMIT_MT = 50.0  # T/m/s
GRAD_LIMIT_MT = 50e-3  # T/m
DT_S = 10e-6

SLEW_LIMIT_HZ_PIX_S = SLEW_LIMIT_MT * GAMMA_HZ_PER_T * FOV_M / N_PIX
GRAD_LIMIT_HZ_PIX = GRAD_LIMIT_MT * GAMMA_HZ_PER_T * FOV_M / N_PIX


def _assert_within_limits(waveform: arbgrad.BaseWaveform, tol: float = 1e-2) -> None:
    grad_si = arbgrad.to_gradient_tesla_per_meter(waveform, FOV_M, N_PIX, GAMMA_HZ_PER_T)
    grad_mag = np.linalg.norm(grad_si, axis=1)
    assert np.max(grad_mag) <= GRAD_LIMIT_MT * (1.0 + tol)

    slew_si = np.diff(grad_si, axis=0) / DT_S
    slew_mag = np.linalg.norm(slew_si, axis=1)
    assert np.max(slew_mag) <= SLEW_LIMIT_MT * (1.0 + tol)


def test_spiral_constant_pitch_special_case_shape_and_limits():
    """Equal k_rho_phi0/k_rho_phi1 is the constant-pitch (Archimedean) case."""
    k_rho_phi = 0.5 / (4.0 * math.pi)
    wf = arbgrad.spiral(
        FOV_M, N_PIX, SLEW_LIMIT_HZ_PIX_S, GRAD_LIMIT_HZ_PIX, DT_S, k_rho_phi0=k_rho_phi, k_rho_phi1=k_rho_phi
    )

    assert wf.k0.shape == (3,)
    assert wf.gradient.ndim == 2
    assert wf.gradient.shape[1] == 3
    assert wf.n_shots >= 1
    # Spiral from k-space center: start offset is at the origin.
    assert np.allclose(wf.k0, 0.0, atol=1e-9)
    # Gradient must ramp from and back to zero (no residual moment discontinuity).
    assert np.allclose(wf.gradient[0], 0.0, atol=1e-6)
    assert np.allclose(wf.gradient[-1], 0.0, atol=1e-6)

    _assert_within_limits(wf)

    expected_n_shots = math.ceil(N_PIX * 2.0 * math.pi * k_rho_phi)
    assert wf.n_shots == expected_n_shots


def test_spiral_variable_density_shape_and_limits():
    wf = arbgrad.spiral(FOV_M, N_PIX, SLEW_LIMIT_HZ_PIX_S, GRAD_LIMIT_HZ_PIX, DT_S)

    assert wf.gradient.shape[1] == 3
    assert wf.n_shots >= 1
    _assert_within_limits(wf)


def test_rosette_base_waveform_shape_and_limits():
    wf = arbgrad.rosette(FOV_M, N_PIX, SLEW_LIMIT_HZ_PIX_S, GRAD_LIMIT_HZ_PIX, DT_S)

    assert wf.gradient.shape[1] == 3
    assert wf.n_shots >= 1
    _assert_within_limits(wf)


@pytest.mark.parametrize(
    ("mode", "expected_increment_rad"),
    [
        ("linear", 2.0 * math.pi / 8),
        ("uniform", 2.0 * math.pi / 8),
        ("golden", math.pi * (3.0 - math.sqrt(5.0))),
        ("mri-golden", math.pi * (math.sqrt(5.0) - 1.0) / 2.0),
        ("intergaps", math.pi / 8 / 2.0),
        (0.3, 0.3),
    ],
)
def test_shot_angles_fixed_increment_modes(mode, expected_increment_rad):
    n_shots = 8
    angles = arbgrad.shot_angles(n_shots, mode=mode)

    assert angles.shape == (n_shots,)
    assert angles[0] == 0.0
    increments = np.diff(angles) % (2.0 * math.pi)
    assert np.allclose(increments, expected_increment_rad % (2.0 * math.pi))


def test_shot_angles_none_mode_is_all_zero():
    angles = arbgrad.shot_angles(5, mode="none")
    assert np.allclose(angles, 0.0)


def test_shot_angles_fibonacci_mode():
    n_shots = 21  # a Fibonacci number, per mri-nufft convention
    angles = arbgrad.shot_angles(n_shots, mode="fibonacci")

    assert angles.shape == (n_shots,)
    assert np.all(angles >= 0.0) and np.all(angles < 2.0 * math.pi)
    # Not a fixed increment: no simple repeating pattern, but every shot gets
    # a distinct angle (up to the discretization mri-nufft's lattice affords).
    assert len(np.unique(np.round(angles, 6))) == n_shots


def test_shot_angles_rejects_unknown_mode():
    with pytest.raises(ValueError):
        arbgrad.shot_angles(4, mode="not-a-mode")


@pytest.mark.skipif(Rotation is None, reason="scipy is required to build rotation quaternions")
def test_base_waveform_consumable_by_fast_sequence_with_rotation():
    """Round-trip smoke test: rotate the base waveform per shot_angles and feed
    it through pulserver.pypulseq.Sequence (the fast builder every zoo sequence
    must use), confirming the arbgrad output is directly consumable end-to-end.
    """
    opts = pp.Opts()
    wf = arbgrad.spiral(FOV_M, N_PIX, SLEW_LIMIT_HZ_PIX_S, GRAD_LIMIT_HZ_PIX, DT_S)
    grad_si = arbgrad.to_gradient_tesla_per_meter(wf, FOV_M, N_PIX, GAMMA_HZ_PER_T)
    angles = arbgrad.shot_angles(wf.n_shots, mode="golden")

    seq = Sequence(opts)
    for angle in angles[:3]:
        gx = pp.make_arbitrary_grad(channel="x", waveform=np.ascontiguousarray(grad_si[:, 0]), system=opts)
        gy = pp.make_arbitrary_grad(channel="y", waveform=np.ascontiguousarray(grad_si[:, 1]), system=opts)
        rotation = make_rotation(Rotation.from_euler("z", angle))
        seq.add_block(gx, gy, rotation)

    assert len(seq.block_events) == 3
