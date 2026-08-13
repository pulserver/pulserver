"""Tests for model-based simultaneous-multislice physics."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
deepinv = pytest.importorskip("deepinv")

from pulserver.recon.physics import SMS


class _CountingIdentity(deepinv.physics.LinearPhysics):
    def __init__(self, *, viewed_as_real: bool = False) -> None:
        super().__init__()
        self.viewed_as_real = viewed_as_real
        self.forward_calls = 0
        self.adjoint_calls = 0

    def A(self, value, **_kwargs):
        self.forward_calls += 1
        if not self.viewed_as_real:
            return value
        batch, channels, *spatial = value.shape
        complex_value = torch.view_as_complex(
            value.reshape(batch, channels // 2, 2, *spatial).movedim(2, -1).contiguous()
        )
        return torch.view_as_real(complex_value)

    def A_adjoint(self, value, **_kwargs):
        self.adjoint_calls += 1
        if not self.viewed_as_real:
            return value
        complex_value = torch.view_as_complex(value.contiguous())
        batch, channels, *spatial = complex_value.shape
        paired = torch.view_as_real(complex_value).movedim(-1, 2)
        return paired.reshape(batch, 2 * channels, *spatial)


def test_sms_shared_physics_folds_slices_into_one_batch_call():
    base = _CountingIdentity()
    physics = SMS(base, torch.tensor([0.0, torch.pi / 2]))
    value = torch.randn(3, 2, 1, 5, 6, dtype=torch.complex64)

    encoded = physics.A(value)
    decoded = physics.A_adjoint(encoded)

    torch.testing.assert_close(encoded, value[:, 0] + 1j * value[:, 1])
    torch.testing.assert_close(
        decoded,
        torch.stack([encoded, -1j * encoded], dim=1),
    )
    assert base.forward_calls == 1
    assert base.adjoint_calls == 1


def test_sms_is_an_exact_adjoint_with_spatial_caipi_modulation():
    generator = torch.Generator().manual_seed(13)
    phase = torch.linspace(0, torch.pi, 7)[None].expand(3, -1)
    physics = SMS(_CountingIdentity(), phase)
    image = torch.randn(
        2,
        3,
        2,
        5,
        7,
        dtype=torch.complex64,
        generator=generator,
    )
    measurement = torch.randn(
        2,
        2,
        5,
        7,
        dtype=torch.complex64,
        generator=generator,
    )

    lhs = (physics.A(image).conj() * measurement).sum().real
    rhs = (image.conj() * physics.A_adjoint(measurement)).sum().real

    torch.testing.assert_close(lhs, rhs)
    torch.testing.assert_close(
        physics.A_adjoint_A(image),
        physics.A_adjoint(physics.A(image)),
    )


def test_sms_supports_pulserver_paired_real_physics():
    base = _CountingIdentity(viewed_as_real=True)
    physics = SMS(base, torch.tensor([0.0, torch.pi]))
    complex_image = torch.randn(2, 2, 1, 4, 5, dtype=torch.complex64)
    paired = torch.view_as_real(complex_image).movedim(-1, 3).reshape(2, 2, 2, 4, 5)

    encoded = physics.A(paired)
    decoded = physics.A_adjoint(encoded)

    encoded_complex = torch.view_as_complex(encoded.contiguous())
    torch.testing.assert_close(
        encoded_complex,
        complex_image[:, 0] - complex_image[:, 1],
        atol=1e-6,
        rtol=1e-6,
    )
    assert decoded.shape == paired.shape


def test_sms_composes_distinct_slice_operators():
    first = _CountingIdentity()
    second = _CountingIdentity()
    physics = SMS([first, second], torch.zeros(2))
    value = torch.randn(1, 2, 1, 3, 4, dtype=torch.complex64)

    physics.A_adjoint(physics.A(value))

    assert first.forward_calls == second.forward_calls == 1
    assert first.adjoint_calls == second.adjoint_calls == 1


def _cartesian_base(matrix=12, coils=3):
    from pulserver.recon.physics import Cartesian2D

    maps = torch.randn(1, coils, matrix, matrix, dtype=torch.complex64)
    maps = maps / maps.abs().square().sum(1, keepdim=True).sqrt()
    return Cartesian2D(torch.ones(1, 1, matrix, matrix), maps), matrix, coils


def _noncartesian_base(matrix=12, coils=3, samples=64):
    from pulserver.recon.physics import NonCartesian2D

    maps = torch.randn(coils, matrix, matrix, dtype=torch.complex64)
    maps = maps / maps.abs().square().sum(0, keepdim=True).sqrt()
    trajectory = (torch.rand(samples, 2) - 0.5).float()
    physics = NonCartesian2D(trajectory, (matrix, matrix), coil_maps=maps, n_coils=coils)
    return physics, matrix, coils


@pytest.mark.parametrize("build", [_cartesian_base, _noncartesian_base])
def test_sms_wraps_any_in_plane_trajectory(build):
    """An SMS pulse is reconstructable whatever encodes the plane beneath it."""
    torch.manual_seed(3)
    base, matrix, _ = build()
    n_slices = 3
    physics = SMS(base, torch.rand(n_slices, 1) * 2 * torch.pi)

    image = torch.randn(1, n_slices, 2, matrix, matrix)
    measurement = physics.A(image)
    cotangent = torch.randn_like(measurement)

    torch.testing.assert_close(
        (physics.A(image) * cotangent).sum(),
        (image * physics.A_adjoint(cotangent)).sum(),
        atol=2e-4,
        rtol=2e-4,
    )
    torch.testing.assert_close(
        physics.A_adjoint_A(image),
        physics.A_adjoint(physics.A(image)),
    )


@pytest.mark.parametrize("build", [_cartesian_base, _noncartesian_base])
def test_every_physics_answers_in_the_same_measurement_layout(build):
    """Real and imaginary parts trail a measurement, whoever encoded it."""
    base, matrix, coils = build()
    measurement = base.A(torch.randn(1, 2, matrix, matrix))
    assert measurement.shape[0] == 1
    assert measurement.shape[1] == coils
    assert measurement.shape[-1] == 2
