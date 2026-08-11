"""Tests for Cartesian 3D, Wave, and Wave-Shuffling physics."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
deepinv = pytest.importorskip("deepinv")

from pulserver.recon._packed import PackedHermitianField
from pulserver.recon.physics import (
    Cartesian3D,
    WaveEncoding,
    WaveShuffling,
)


def _problem(*, multiple_maps: bool = False, real_basis: bool = False):
    generator = torch.Generator().manual_seed(81)
    batch, coils, rank, echoes = 2, 3, 3, 4
    image_shape = (3, 4, 3)
    wave_readout = 5
    maps = 2 if multiple_maps else 1
    coil_maps = torch.randn(
        coils,
        maps,
        *image_shape,
        generator=generator,
        dtype=torch.complex64,
    )
    if not multiple_maps:
        coil_maps = coil_maps[:, 0]
    wave_psf = torch.randn(
        wave_readout,
        image_shape[1],
        image_shape[2],
        generator=generator,
        dtype=torch.complex64,
    )
    basis = torch.randn(rank, echoes, generator=generator)
    if not real_basis:
        basis = torch.complex(
            basis,
            torch.randn(rank, echoes, generator=generator),
        )
    sampling = torch.tensor(
        [
            [0, 0, 0],
            [1, 2, 1],
            [3, 1, 2],
            [1, 2, 3],
            [0, 0, 2],
            [2, 2, 1],
        ]
    )
    weights = torch.tensor([1.0, 0.5, 1.2, 0.8, 0.7, 1.1])
    coefficients = torch.randn(
        batch,
        maps * rank,
        *image_shape,
        generator=generator,
        dtype=torch.complex64,
    )
    return sampling, coil_maps, wave_psf, basis, weights, coefficients


@pytest.mark.parametrize("complex_values", [False, True])
def test_packed_hermitian_field_matches_dense_and_differentiates(complex_values):
    generator = torch.Generator().manual_seed(17)
    batch, rank, locations = 5, 4, 13
    rows, columns = torch.triu_indices(rank, rank)
    values = torch.randn(rows.numel(), locations, generator=generator)
    if complex_values:
        values = torch.complex(
            values,
            torch.randn(rows.numel(), locations, generator=generator),
        )
        values[rows == columns] = values[rows == columns].real.to(values.dtype)
    spectrum = torch.randn(
        batch,
        rank,
        locations,
        generator=generator,
        dtype=torch.complex64,
        requires_grad=True,
    )
    matrix = torch.zeros(locations, rank, rank, dtype=torch.complex64)
    matrix[:, rows, columns] = values.T.to(matrix.dtype)
    off_diagonal = rows != columns
    matrix[:, columns[off_diagonal], rows[off_diagonal]] = (
        values[off_diagonal].T.conj().to(matrix.dtype)
    )

    result = PackedHermitianField(values, rank)(spectrum)
    reference = torch.einsum("lij,bjl->bil", matrix, spectrum)

    torch.testing.assert_close(result, reference)
    result.abs().square().sum().backward()
    assert spectrum.grad is not None
    assert torch.isfinite(spectrum.grad).all()


@pytest.mark.parametrize("multiple_maps", [False, True])
@pytest.mark.parametrize("viewed_as_real", [False, True])
def test_wave_shuffling_adjoint_and_packed_normal(multiple_maps, viewed_as_real):
    sampling, maps, psf, basis, weights, coefficients = _problem(
        multiple_maps=multiple_maps,
        real_basis=True,
    )
    physics = WaveShuffling(
        sampling,
        maps,
        psf,
        basis,
        line_weights=weights,
        viewed_as_real=viewed_as_real,
        coil_batch_size=2,
    )
    if viewed_as_real:
        batch, channels, *spatial = coefficients.shape
        coefficients = (
            torch.view_as_real(coefficients)
            .movedim(-1, 2)
            .reshape(
                batch,
                2 * channels,
                *spatial,
            )
        )
    data = physics.A(coefficients)
    test_data = torch.randn_like(data)

    left = (
        (data * test_data).sum()
        if viewed_as_real
        else (data.conj() * test_data).sum().real
    )
    adjoint = physics.A_adjoint(test_data)
    right = (
        (coefficients * adjoint).sum()
        if viewed_as_real
        else (coefficients.conj() * adjoint).sum().real
    )

    torch.testing.assert_close(left, right, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(
        physics.A_adjoint_A(coefficients),
        physics.A_adjoint(data),
        atol=2e-5,
        rtol=2e-5,
    )
    assert physics.normal_mode == "exact-hybrid"
    assert physics.operator.field.is_real
    assert (
        physics.operator.field.values.shape[0]
        == basis.shape[0] * (basis.shape[0] + 1) // 2
    )


def test_wave_encoding_is_rank_one_wave_physics():
    sampling, maps, psf, _basis, weights, _coefficients = _problem()
    physics = WaveEncoding(
        sampling[:, :2],
        maps,
        psf,
        line_weights=weights,
        viewed_as_real=False,
    )
    image = torch.randn(2, 1, 3, 4, 3, dtype=torch.complex64)

    data = physics.A(image)

    assert data.shape == (2, maps.shape[0], sampling.shape[0], psf.shape[0])
    torch.testing.assert_close(
        physics.A_adjoint_A(image),
        physics.A_adjoint(data),
        atol=2e-5,
        rtol=2e-5,
    )


def test_cartesian_3d_is_deepinverse_linear_physics():
    mask = torch.ones(1, 2, 3, 4, 5)
    maps = torch.ones(2, 3, 4, 5, dtype=torch.complex64)

    physics = Cartesian3D(mask, maps)

    assert isinstance(physics, deepinv.physics.LinearPhysics)
    assert physics.spatial_ndim == 3
