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
    model = NLINV(
        calibration_width=4,
        max_iter=3,
        cg_max_iter=5,
        cg_rtol=1e-3,
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
