"""Tests for the nonlinear IRGNM-CG calibration stack."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
deepinv = pytest.importorskip("deepinv")

from pulserver.recon.calibration import (
    NLINV,
    NLINVPhysics,
    NLINVResult,
    PhasePoleCorrection,
)


class _IdentityEncoding(deepinv.physics.LinearPhysics):
    def A(self, value, **_kwargs):
        return value

    def A_adjoint(self, value, **_kwargs):
        return value

    def A_adjoint_A(self, value, **_kwargs):
        return value


def test_nlinv_physics_analytic_jacobian_matches_torch_func():
    generator = torch.Generator().manual_seed(45)
    physics = NLINVPhysics(_IdentityEncoding(), torch.ones(4, 5), 3)
    state = physics.initial_state(2, device="cpu") + 0.1 * torch.randn(
        2,
        8,
        4,
        5,
        generator=generator,
    )
    direction = torch.randn(state.shape, generator=generator)
    cotangent = torch.randn(
        physics.A(state).shape,
        generator=generator,
    )

    analytic = physics.A_jvp(state, direction)
    automatic = torch.func.jvp(physics.A, (state,), (direction,))[1]
    _, jacobian = physics.linearize(state)

    torch.testing.assert_close(analytic, automatic)
    torch.testing.assert_close(
        (analytic * cotangent).sum(),
        (direction * physics.A_vjp(state, cotangent)).sum(),
    )
    torch.testing.assert_close(
        jacobian.A_adjoint_A(direction),
        jacobian.A_adjoint(jacobian.A(direction)),
    )


def test_nlinv_returns_normalized_maps_and_named_diagnostics():
    generator = torch.Generator().manual_seed(12)
    batch, coils, height, width = 2, 2, 6, 6
    image = torch.randn(
        batch,
        1,
        height,
        width,
        generator=generator,
        dtype=torch.complex64,
    )
    maps = torch.randn(
        batch,
        coils,
        height,
        width,
        generator=generator,
        dtype=torch.complex64,
    )
    maps = maps / maps.abs().square().sum(dim=1, keepdim=True).sqrt()
    kspace = torch.fft.fftshift(
        torch.fft.fftn(
            torch.fft.ifftshift(image * maps, dim=(-2, -1)),
            dim=(-2, -1),
            norm="ortho",
        ),
        dim=(-2, -1),
    )
    # White-noise coils lie outside the Sobolev ball the solve searches, so no
    # factorization reproduces this data. The shapes and the RSS normalization
    # are still well defined, and they are what this test is about.
    model = NLINV(
        calibration_width=4,
        max_iter=3,
        cg_max_iter=5,
        cg_rtol=1e-3,
        residual_tolerance=None,
    )

    result = model(kspace, return_info=True)

    assert isinstance(result, NLINVResult)
    assert result.sensitivities.shape == (batch, coils, height, width)
    assert result.image.shape == (batch, height, width)
    assert result.calibration.shape == (batch, coils, 4, 4)
    assert torch.isfinite(result.sensitivities).all()
    torch.testing.assert_close(
        result.sensitivities.abs().square().sum(dim=1).sqrt(),
        torch.ones(batch, height, width),
        atol=2e-5,
        rtol=2e-5,
    )


def test_phase_pole_correction_preserves_encoding_and_removes_winding():
    size = 64
    y, x = torch.meshgrid(
        torch.arange(size),
        torch.arange(size),
        indexing="ij",
    )
    pole = (x - 31.3) + 1j * (y - 30.7)
    pole = pole / pole.abs().clamp_min(1e-7)
    amplitudes = torch.stack(
        [
            torch.ones_like(x),
            0.7 + 0.1 * x / size,
            0.5 + 0.1 * y / size,
        ]
    ).to(torch.float32)
    coils = (amplitudes.to(torch.complex64) * pole)[None]
    image = torch.ones(1, 1, size, size, dtype=torch.complex64) * pole.conj()
    model = PhasePoleCorrection(diameter=0.1, segments=16)

    initial_curl = model.curl(coils)
    corrected_image, corrected_coils = model(image, coils)
    corrected_curl = model.curl(corrected_coils)

    assert initial_curl.abs().max() > 0.9
    assert corrected_curl.abs().max() < 1e-4
    assert len(model.detected_poles[0]) == 1
    torch.testing.assert_close(
        corrected_image * corrected_coils,
        image * coils,
        atol=1e-6,
        rtol=1e-6,
    )


def test_phase_pole_correction_vectorizes_over_3d_planes():
    size = 32
    y, x = torch.meshgrid(
        torch.arange(size),
        torch.arange(size),
        indexing="ij",
    )
    poles = []
    for offset in (-1.0, 0.0, 1.0):
        pole = (x - (15.2 + offset)) + 1j * (y - 16.1)
        poles.append(pole / pole.abs().clamp_min(1e-7))
    pole_volume = torch.stack(poles)
    coils = pole_volume[None, None].expand(2, 3, -1, -1, -1).clone()
    image = pole_volume.conj()[None, None].expand(2, 1, -1, -1, -1).clone()
    model = PhasePoleCorrection(diameter=0.2, segments=16)

    corrected_image, corrected_coils = model(image, coils)

    assert corrected_image.shape == image.shape
    assert corrected_coils.shape == coils.shape
    torch.testing.assert_close(
        corrected_image * corrected_coils,
        image * coils,
        atol=2e-6,
        rtol=2e-6,
    )


def test_nlinv_physical_state_roundtrip_supports_gauge_projection():
    generator = torch.Generator().manual_seed(9)
    physics = NLINVPhysics(
        _IdentityEncoding(),
        torch.rand(5, 6, generator=generator).clamp_min(0.2),
        2,
    )
    image = torch.randn(2, 1, 5, 6, dtype=torch.complex64, generator=generator)
    coils = torch.randn(2, 2, 5, 6, dtype=torch.complex64, generator=generator)

    restored_image, restored_coils = physics.physical(
        physics.state_from_physical(image, coils)
    )

    torch.testing.assert_close(restored_image, image)
    torch.testing.assert_close(restored_coils, coils, atol=2e-6, rtol=2e-6)


def _smooth_calibration_case(matrix=64, coils=6):
    """Noiseless fully sampled data from smooth coils and a smooth object."""
    axis = torch.linspace(-1.0, 1.0, matrix)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    image = 0.3 + torch.where((x / 0.95) ** 2 + (y / 0.95) ** 2 < 1.0, 0.7, 0.0)
    angles = torch.linspace(0.0, 2 * torch.pi, coils + 1)[:-1]
    maps = torch.stack(
        [
            torch.exp(
                -((x - 1.5 * torch.cos(a)) ** 2 + (y - 1.5 * torch.sin(a)) ** 2) / 3.0
            )
            * torch.exp(1j * 1.5 * (torch.cos(a) * x + torch.sin(a) * y))
            for a in angles
        ]
    ).to(torch.complex64)
    maps = maps / maps.abs().square().sum(0, keepdim=True).sqrt()
    coil_images = maps * image.to(torch.complex64)
    axes = (-2, -1)
    kspace = torch.fft.fftshift(
        torch.fft.fftn(
            torch.fft.ifftshift(coil_images, dim=axes), dim=axes, norm="ortho"
        ),
        dim=axes,
    )
    return kspace[None].to(torch.complex64), maps


def _relative_residual(model, kspace):
    """The relative data residual the calibration actually left behind."""
    seen = {}
    original = type(model)._require_reproduction

    def spy(self, prediction, measured):
        norm = torch.linalg.vector_norm(measured.flatten(start_dim=1), dim=1)
        difference = torch.linalg.vector_norm(
            (prediction - measured).flatten(start_dim=1), dim=1
        )
        seen["residual"] = float((difference / norm).max())
        return original(self, prediction, measured)

    type(model)._require_reproduction = spy
    try:
        model(kspace)
    finally:
        type(model)._require_reproduction = original
    return seen["residual"]


def test_nlinv_reproduces_the_data_it_was_calibrated_on():
    """A factorization that cannot re-predict noiseless data is not converged."""
    kspace, _ = _smooth_calibration_case()
    assert _relative_residual(NLINV(calibration_width=None), kspace) < 0.15


def test_more_newton_steps_leave_a_smaller_residual():
    """Guards the damping schedule: the outer loop has to keep making progress."""
    kspace, _ = _smooth_calibration_case()
    few = _relative_residual(NLINV(calibration_width=None, max_iter=4), kspace)
    many = _relative_residual(NLINV(calibration_width=None, max_iter=12), kspace)
    assert many < 0.6 * few


def test_the_calibration_square_is_read_off_a_cartesian_mask():
    """A Cartesian acquisition states its calibration region; nothing to choose."""
    from pulserver.recon.calibration import calibration_extent

    mask = torch.zeros(64, 64, dtype=torch.bool)
    mask[24:40] = True
    assert calibration_extent(mask) == 16

    # An accelerated line just outside the block must not widen it.
    mask[0:64:2] = True
    assert calibration_extent(mask) == 16

    # The square fits inside the acquired data on every axis, so a partial echo
    # narrows it.
    partial = torch.zeros(64, 64, dtype=torch.bool)
    partial[24:40, 16:] = True
    assert calibration_extent(partial) == 16


def test_an_unsampled_centre_leaves_no_calibration_square():
    from pulserver.recon.calibration import calibration_extent

    mask = torch.zeros(16, 16, dtype=torch.bool)
    mask[0:16:2] = True
    assert calibration_extent(mask) == 0

    kspace = torch.zeros(1, 2, 16, 16, dtype=torch.complex64)
    kspace[:, :, 0:16:2] = 1.0
    with pytest.raises(ValueError, match="no fully sampled block"):
        NLINV()(kspace)


def test_a_non_cartesian_calibration_needs_an_explicit_width():
    """A spiral does not state a calibration region, so it has to be given one."""
    trajectory = torch.rand(64, 2) - 0.5
    kspace = torch.ones(1, 2, 64, dtype=torch.complex64)
    with pytest.raises(ValueError, match="has to be given as a number"):
        NLINV()(kspace, trajectory=trajectory, image_shape=(16, 16))


def test_a_starved_inner_solve_is_reported_rather_than_returned():
    kspace, _ = _smooth_calibration_case()
    starved = NLINV(
        calibration_width=None,
        max_iter=1,
        cg_max_iter=1,
        residual_tolerance=0.2,
    )
    with pytest.raises(RuntimeError, match="relative data residual"):
        starved(kspace)


def test_the_residual_check_can_be_switched_off():
    kspace, _ = _smooth_calibration_case()
    permissive = NLINV(
        calibration_width=None,
        max_iter=1,
        cg_max_iter=1,
        residual_tolerance=None,
    )
    assert permissive(kspace).shape == (1, 6, 64, 64)
