"""Tests for the nonlinear IRGNM-CG calibration stack."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
deepinv = pytest.importorskip("deepinv")

from pulserver.recon.calibration import NLINV, NLINVPhysics, NLINVResult


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
