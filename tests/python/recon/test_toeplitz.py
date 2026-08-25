"""Tests for compact Torch Toeplitz transfer kernels."""

from __future__ import annotations

from math import prod
from typing import Any
from os import cpu_count
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from pulserver.recon._toeplitz import (
    CompactToeplitzKernel,
    support_indices,
)
from pulserver.recon.execution import CudaStreaming
import pulserver.recon.physics as physics


torch = pytest.importorskip("torch")


def _dense_apply(image, transfer, image_shape):
    spatial_shape = transfer.shape[2:]
    padded = torch.zeros(
        (image.shape[0], image.shape[1], *spatial_shape),
        dtype=image.dtype,
        device=image.device,
    )
    crop = (
        slice(None),
        slice(None),
        *(slice(0, size) for size in image_shape),
    )
    padded[crop] = image
    axes = tuple(range(2, padded.ndim))
    spectrum = torch.fft.fftn(padded, dim=axes)
    spectrum = torch.einsum("kl...,bl...->bk...", transfer, spectrum)
    return torch.fft.ifftn(spectrum, dim=axes)[crop]


def _packed_kernel(transfer, image_shape, *, chunk_size=7):
    rank = transfer.shape[0]
    rows, columns = torch.triu_indices(rank, rank, device=transfer.device)
    values = transfer[rows, columns].reshape(rows.numel(), -1)
    indices = support_indices(
        tuple(transfer.shape[2:]),
        support="full",
        device=transfer.device,
    )
    return CompactToeplitzKernel(
        values,
        indices,
        tuple(transfer.shape[2:]),
        rank,
        image_shape=image_shape,
        chunk_size=chunk_size,
    )


@pytest.mark.parametrize("complex_transfer", [False, True])
def test_precompiled_cpu_kernel_matches_torch_reference(complex_transfer):
    extension = pytest.importorskip(
        "pulserver._ext.recon_cpu",
        exc_type=ImportError,
    )
    generator = torch.Generator().manual_seed(2)
    rank, locations = 4, 37
    rows, columns = torch.triu_indices(rank, rank)
    values = torch.randn(rows.numel(), locations, generator=generator)
    if complex_transfer:
        values = torch.complex(
            values,
            torch.randn(values.shape, generator=generator),
        )
    spectrum = torch.randn(
        3,
        rank,
        locations,
        dtype=torch.complex64,
        generator=generator,
    )
    output = torch.empty_like(spectrum)
    extension.packed_hermitian_matvec(
        values.numpy(),
        spectrum.numpy(),
        output.numpy(),
    )
    matrix = torch.zeros(
        locations,
        rank,
        rank,
        dtype=values.dtype,
    )
    matrix[:, rows, columns] = values.T
    off = rows != columns
    matrix[:, columns[off], rows[off]] = values[off].T.conj()
    reference = torch.einsum(
        "lij,blj->bli",
        matrix.to(spectrum.dtype),
        spectrum.permute(0, 2, 1),
    )
    reference = reference.permute(0, 2, 1)

    torch.testing.assert_close(output, reference, atol=2e-6, rtol=2e-6)
    assert extension.simd_level() in {"scalar", "avx2", "avx512"}
    assert extension.thread_count(1) == 1
    assert extension.thread_count(32769) >= 1
    assert extension.sample_thread_count(1, 1) == 1
    assert extension.sample_thread_count(32769, 1) >= 1


def test_precompiled_cpu_kernel_parallelizes_independent_batches():
    extension = pytest.importorskip(
        "pulserver._ext.recon_cpu",
        exc_type=ImportError,
    )
    generator = torch.Generator().manual_seed(7)
    batch, rank, locations = 32, 3, 1024
    rows, columns = torch.triu_indices(rank, rank)
    values = torch.randn(rows.numel(), locations, generator=generator)
    spectrum = torch.randn(
        batch,
        rank,
        locations,
        dtype=torch.complex64,
        generator=generator,
    )
    output = torch.empty_like(spectrum)

    extension.packed_hermitian_matvec(
        values.numpy(),
        spectrum.numpy(),
        output.numpy(),
    )

    matrix = torch.zeros(locations, rank, rank)
    matrix[:, rows, columns] = values.T
    off_diagonal = rows != columns
    matrix[:, columns[off_diagonal], rows[off_diagonal]] = values[off_diagonal].T
    reference = torch.einsum(
        "lij,blj->bli",
        matrix.to(spectrum.dtype),
        spectrum.permute(0, 2, 1),
    ).permute(0, 2, 1)

    torch.testing.assert_close(output, reference, atol=2e-6, rtol=2e-6)
    expected_workers = 2 if (cpu_count() or 1) > 1 else 1
    assert extension.sample_thread_count(batch, locations) >= expected_workers


@pytest.mark.parametrize("real_transfer", [True, False])
def test_compact_kernel_matches_dense_hermitian_transfer(real_transfer):
    generator = torch.Generator().manual_seed(12)
    image_shape = (3, 4)
    spatial_shape = (6, 8)
    rank = 3
    if real_transfer:
        raw = torch.randn(
            rank,
            rank,
            *spatial_shape,
            generator=generator,
        )
        transfer = 0.5 * (raw + raw.movedim(0, 1))
    else:
        raw = torch.randn(
            rank,
            rank,
            *spatial_shape,
            generator=generator,
            dtype=torch.complex64,
        )
        transfer = 0.5 * (raw + raw.movedim(0, 1).conj())
    image = torch.randn(
        2,
        rank,
        *image_shape,
        generator=generator,
        dtype=torch.complex64,
    )

    compact = _packed_kernel(transfer, image_shape)
    result = compact.apply(image)
    reference = _dense_apply(image, transfer.to(image.dtype), image_shape)

    torch.testing.assert_close(result, reference, atol=2e-5, rtol=2e-5)
    assert compact.is_real is real_transfer
    assert compact.storage_nbytes < compact.dense_nbytes


def test_radial_support_has_expected_circle_and_sphere_reduction():
    circle = support_indices((64, 64), support="radial")
    sphere = support_indices((32, 32, 32), support="radial")
    assert circle.numel() / 64**2 == pytest.approx(np.pi / 4, abs=0.02)
    assert sphere.numel() / 32**3 == pytest.approx(np.pi / 6, abs=0.03)


def _framed_radial(frames: int, spokes: int, samples: int) -> np.ndarray:
    """One golden-angle-rotated radial frame per time point."""
    radius = np.linspace(-0.5, 0.5, samples, endpoint=False)
    offsets = np.arange(frames)[:, None] * np.pi * (3 - np.sqrt(5))
    angles = offsets + np.linspace(0, np.pi, spokes, endpoint=False)[None]
    return (
        np.stack(
            [np.cos(angles)[..., None] * radius, np.sin(angles)[..., None] * radius],
            -1,
        )
        .reshape(frames, spokes * samples, 2)
        .astype(np.float32)
    )


@pytest.mark.parametrize("complex_basis", [False, True])
@pytest.mark.parametrize("compress", [False, True])
def test_the_subspace_kernel_equals_the_framewise_normal(complex_basis, compress):
    pytest.importorskip("mrinufft")
    pytest.importorskip("finufft")
    generator = torch.Generator().manual_seed(4)
    image_shape = (16, 16)
    rank, frames = 3, 6
    trajectory = _framed_radial(frames, 21, image_shape[0])
    basis = torch.randn(rank, frames, generator=generator)
    if complex_basis:
        basis = torch.complex(basis, torch.randn(rank, frames, generator=generator))
    coefficients = torch.randn(
        1,
        rank,
        *image_shape,
        generator=generator,
        dtype=torch.complex64,
    )

    def built(toeplitz):
        return physics.Subspace(
            physics.NonCartesian2D(
                trajectory,
                image_shape,
                backend="finufft",
                toeplitz=toeplitz,
            ),
            basis,
        )

    accelerated = built({"compress": compress})
    plain = built(False)

    result = accelerated.A_adjoint_A(coefficients)
    reference = plain.A_adjoint(plain.A(coefficients)).reshape(result.shape)

    # A two-dimensional radial transfer really is dense, so compression costs
    # more here than on any trajectory it is meant for.
    tolerance = 5e-2 if compress else 2e-3
    torch.testing.assert_close(result, reference, atol=tolerance, rtol=tolerance)
    assert accelerated.toeplitz_kernel.is_real is (not complex_basis)


def test_one_gridding_pass_serves_every_frame():
    """The kernel costs one NUFFT per basis pair, not one per frame."""
    pytest.importorskip("mrinufft")
    pytest.importorskip("finufft")
    generator = torch.Generator().manual_seed(11)
    image_shape = (16, 16)
    rank = 3
    pairs = rank * (rank + 1) // 2
    plans: list[int] = []
    griddings: list[int] = []
    original = physics._psf_operator

    def counted(samples, backend, spatial_shape):
        plans.append(int(samples.shape[0]))
        operator = original(samples, backend, spatial_shape)
        adjoint = operator.adj_op

        def counting_adjoint(values):
            griddings.append(int(values.shape[-1]))
            return adjoint(values)

        operator.adj_op = counting_adjoint
        return operator

    passes = []
    for frames in (4, 12):
        plans.clear()
        griddings.clear()
        trajectory = _framed_radial(frames, 21, image_shape[0])
        basis = torch.randn(rank, frames, generator=generator)
        accelerated = physics.Subspace(
            physics.NonCartesian2D(
                trajectory,
                image_shape,
                backend="finufft",
                toeplitz=True,
            ),
            basis,
        )
        with mock.patch.object(physics, "_psf_operator", counted):
            accelerated.A_adjoint_A(
                torch.zeros(1, rank, *image_shape, dtype=torch.complex64)
            )
        assert len(plans) == 1
        assert len(griddings) == pairs
        # Every frame's samples go into the one point set that is gridded.
        assert plans[0] == frames * 21 * image_shape[0]
        assert set(griddings) == {plans[0]}
        passes.append(len(griddings))
    # Tripling the frames costs no extra pass: they share the point set.
    assert passes[0] == passes[1]


def test_cartesian_subspace_builder_matches_centered_fft_normal():
    generator = torch.Generator().manual_seed(24)
    image_shape = (4, 6)
    rank, frames = 3, 4
    basis = torch.randn(rank, frames, generator=generator)
    masks = (torch.rand(frames, *image_shape, generator=generator) > 0.35).float()
    smaps = torch.randn(
        2,
        *image_shape,
        generator=generator,
        dtype=torch.complex64,
    )
    operator = SimpleNamespace(mask=masks, coil_maps=smaps)
    frame_physics = [SimpleNamespace(operator=operator) for _ in range(frames)]
    options = physics._toeplitz_options(
        chunk_size=5,
        coil_batch_size=2,
    )
    kernel, proxy = physics._build_cartesian_subspace_toeplitz(
        frame_physics,
        basis,
        options,
    )
    image = torch.randn(
        2,
        rank,
        *image_shape,
        generator=generator,
        dtype=torch.complex64,
    )
    result = physics._apply_sense_toeplitz(
        kernel,
        image,
        proxy,
        coil_batch_size=2,
    )

    basis_complex = basis.to(image.dtype)
    expanded = torch.einsum(
        "kt,bk...->bt...",
        basis_complex.conj(),
        image,
    )
    normal_frames = torch.zeros_like(expanded)
    axes = (-2, -1)
    for frame in range(frames):
        for smap in smaps:
            coil_image = expanded[:, frame] * smap
            spectrum = torch.fft.fftshift(
                torch.fft.fftn(
                    torch.fft.ifftshift(coil_image, dim=axes),
                    dim=axes,
                    norm="ortho",
                ),
                dim=axes,
            )
            normal = torch.fft.fftshift(
                torch.fft.ifftn(
                    torch.fft.ifftshift(
                        masks[frame] * spectrum,
                        dim=axes,
                    ),
                    dim=axes,
                    norm="ortho",
                ),
                dim=axes,
            )
            normal_frames[:, frame] += smap.conj() * normal
    reference = torch.einsum(
        "kt,bt...->bk...",
        basis_complex,
        normal_frames,
    )
    torch.testing.assert_close(result, reference, atol=2e-5, rtol=2e-5)
    assert kernel.spatial_shape == image_shape


def test_cartesian_subspace_public_factory_handles_batched_sensitivity_maps():
    pytest.importorskip("deepinv")
    generator = torch.Generator().manual_seed(31)
    batch, coils, rank, frames = 3, 2, 2, 3
    image_shape = (6, 8)
    mask = torch.ones(1, 1, *image_shape)
    smaps = torch.randn(
        batch,
        coils,
        *image_shape,
        generator=generator,
        dtype=torch.complex64,
    )
    smaps /= torch.linalg.vector_norm(smaps, dim=1, keepdim=True)
    basis = torch.randn(rank, frames, generator=generator)
    coefficients = torch.randn(
        batch,
        rank,
        *image_shape,
        generator=generator,
        dtype=torch.complex64,
    )

    exact = physics.Subspace(physics.Cartesian2D(mask, smaps), basis)
    compact = physics.Toeplitz(
        physics.Subspace(physics.Cartesian2D(mask, smaps), basis),
    )

    reference = exact.A_adjoint_A(coefficients)
    result = compact.A_adjoint_A(coefficients)

    torch.testing.assert_close(result, reference, atol=3e-5, rtol=3e-5)
    assert compact.normal_mode == "exact-fft"
    assert compact.operator.toeplitz_kernel.is_real


@pytest.mark.parametrize("use_sense", [False, True])
def test_public_scalar_toeplitz_is_owned_by_pulserver(use_sense):
    pytest.importorskip("mrinufft")
    pytest.importorskip("finufft")
    generator = torch.Generator().manual_seed(91)
    rng = np.random.default_rng(91)
    image_shape = (4, 6)
    batch, coils = 2, 3
    trajectory = rng.uniform(-0.45, 0.45, (29, 2)).astype(np.float32)
    density = np.linspace(0.6, 1.4, trajectory.shape[0], dtype=np.float32)
    maps = None
    channels = coils
    if use_sense:
        maps = torch.randn(
            coils,
            *image_shape,
            generator=generator,
            dtype=torch.complex64,
        )
        maps /= torch.linalg.vector_norm(maps, dim=0, keepdim=True)
        channels = 1
    image = torch.randn(
        batch,
        channels,
        *image_shape,
        generator=generator,
        dtype=torch.complex64,
    )
    compact = physics.NonCartesian2D(
        trajectory,
        image_shape,
        coil_maps=maps,
        density=density,
        backend="finufft",
        n_coils=coils,
        n_batchs=batch,
        viewed_as_real=False,
        toeplitz=True,
    )

    reference = compact.A_adjoint(compact.A(image))
    result = compact.A_adjoint_A(image)

    torch.testing.assert_close(result, reference, atol=3e-5, rtol=3e-5)
    assert compact.normal_mode == "toeplitz"
    assert not hasattr(compact.native_operator, "gram_op")
    assert compact.operator.toeplitz_kernel.is_real


@pytest.mark.parametrize("shared", [True, False])
def test_stacked_nufft_uses_shared_or_independent_toeplitz_bank(shared):
    pytest.importorskip("mrinufft")
    pytest.importorskip("finufft")
    generator = torch.Generator().manual_seed(103)
    rng = np.random.default_rng(103)
    image_shape = (4, 6, 4)
    batch, coils = 2, 2
    if shared:
        trajectories = rng.uniform(-0.45, 0.45, (19, 2)).astype(np.float32)
        density = np.linspace(0.7, 1.2, 19, dtype=np.float32)
    else:
        trajectories = [
            rng.uniform(-0.45, 0.45, (17 + index, 2)).astype(np.float32)
            for index in range(image_shape[-1])
        ]
        density = [
            np.linspace(0.7, 1.2, len(item), dtype=np.float32) for item in trajectories
        ]
    image = torch.randn(
        batch,
        coils,
        *image_shape,
        generator=generator,
        dtype=torch.complex64,
    )
    compact = physics.NonCartesian3D(
        trajectories,
        image_shape,
        density=density,
        backend="finufft",
        n_coils=coils,
        n_batchs=batch,
        stacked=True,
        z_index=None,
        viewed_as_real=False,
        toeplitz=True,
    )

    reference = compact.A_adjoint(compact.A(image))
    result = compact.A_adjoint_A(image)

    torch.testing.assert_close(result, reference, atol=3e-5, rtol=3e-5)
    assert compact.normal_mode == "toeplitz"
    assert compact.native_operator.shared_trajectory is shared
    assert len(compact.operator.toeplitz_kernels) == (1 if shared else image_shape[-1])


def test_off_resonance_builder_matches_segmented_normal(monkeypatch):
    generator = torch.Generator().manual_seed(7)
    image_shape = (3, 3)
    spatial_shape = (6, 6)
    segments, samples = 3, 5
    features = torch.randn(samples, *spatial_shape, generator=generator)
    temporal = torch.randn(
        samples,
        segments,
        generator=generator,
        dtype=torch.complex64,
    )
    spatial = torch.randn(
        segments,
        *image_shape,
        generator=generator,
        dtype=torch.complex64,
    )
    smaps = torch.randn(
        2,
        *image_shape,
        generator=generator,
        dtype=torch.complex64,
    )
    density = torch.linspace(0.5, 1.0, samples)
    base = SimpleNamespace(shape=image_shape, smaps=smaps, density=density)
    corrected = SimpleNamespace(
        _fourier_op=base,
        B=temporal,
        C=spatial,
        n_shots=1,
    )

    def compute(_native, weights=None, *, complex_weights=False):
        weights = torch.as_tensor(weights)
        if not complex_weights:
            weights = weights.real
        return torch.einsum(
            "s,s...->...",
            weights,
            features.to(weights.dtype),
        )

    monkeypatch.setattr(physics, "_compute_toeplitz_transfer", compute)
    options = physics._toeplitz_options(
        chunk_size=4,
        coil_batch_size=2,
    )
    kernel, returned_spatial = physics._build_off_resonance_toeplitz(corrected, options)
    image = torch.randn(
        2,
        1,
        *image_shape,
        generator=generator,
        dtype=torch.complex64,
    )
    result = physics._apply_sense_toeplitz(
        kernel,
        image,
        corrected,
        right_factors=returned_spatial,
        left_factors=returned_spatial,
        coil_batch_size=2,
    )

    transfer = torch.empty(
        segments,
        segments,
        *spatial_shape,
        dtype=torch.complex64,
    )
    for left in range(segments):
        for right in range(segments):
            weights = temporal[:, left].conj() * temporal[:, right] * density
            transfer[left, right] = compute(
                base,
                weights,
                complex_weights=left != right,
            )
    reference = torch.zeros_like(image)
    for smap in smaps:
        segment_images = spatial * (smap * image[:, 0])[:, None]
        segment_normal = _dense_apply(
            segment_images,
            transfer,
            image_shape,
        )
        reference[:, 0] += (spatial.conj()[None] * segment_normal).sum(
            dim=1
        ) * smap.conj()

    torch.testing.assert_close(result, reference, atol=3e-5, rtol=3e-5)
    assert not kernel.is_real


def test_combined_subspace_off_resonance_matches_framewise_normal(monkeypatch):
    generator = torch.Generator().manual_seed(18)
    image_shape = (2, 3)
    spatial_shape = (4, 6)
    coefficients, frames, segments, samples = 2, 3, 2, 4
    basis = torch.randn(coefficients, frames, generator=generator)
    spatial = torch.randn(
        segments,
        *image_shape,
        generator=generator,
        dtype=torch.complex64,
    )
    smaps = torch.randn(
        2,
        *image_shape,
        generator=generator,
        dtype=torch.complex64,
    )
    corrected = []
    for _ in range(frames):
        base = SimpleNamespace(
            shape=image_shape,
            smaps=smaps,
            density=torch.linspace(0.7, 1.0, samples),
            features=torch.randn(
                samples,
                *spatial_shape,
                generator=generator,
            ),
        )
        corrected.append(
            SimpleNamespace(
                _fourier_op=base,
                B=torch.randn(
                    samples,
                    segments,
                    generator=generator,
                    dtype=torch.complex64,
                ),
                C=spatial,
                n_shots=1,
                shape=image_shape,
            )
        )

    def compute(native, weights=None, *, complex_weights=False):
        weights = torch.as_tensor(weights)
        if not complex_weights:
            weights = weights.real
        return torch.einsum(
            "s,s...->...",
            weights,
            native.features.to(weights.dtype),
        )

    monkeypatch.setattr(physics, "_compute_toeplitz_transfer", compute)
    frame_physics = [SimpleNamespace(native_operator=native) for native in corrected]
    options = physics._toeplitz_options(
        chunk_size=3,
        coil_batch_size=2,
    )
    kernel, factors = physics._build_subspace_off_resonance_toeplitz(
        frame_physics,
        basis,
        options,
    )
    image = torch.randn(
        2,
        coefficients,
        *image_shape,
        generator=generator,
        dtype=torch.complex64,
    )
    result = physics._apply_subspace_off_resonance_toeplitz(
        kernel,
        image,
        corrected[0],
        factors,
        coefficient_rank=coefficients,
        coil_batch_size=2,
    )

    basis_complex = basis.to(image.dtype)
    expanded = torch.einsum(
        "kt,bk...->bt...",
        basis_complex.conj(),
        image,
    )
    normal_frames = []
    for frame, native in enumerate(corrected):
        transfer = torch.empty(
            segments,
            segments,
            *spatial_shape,
            dtype=torch.complex64,
        )
        for left in range(segments):
            for right in range(segments):
                weights = (
                    native.B[:, left].conj()
                    * native.B[:, right]
                    * native._fourier_op.density
                )
                transfer[left, right] = compute(
                    native._fourier_op,
                    weights,
                    complex_weights=left != right,
                )
        frame_result = torch.zeros(
            (image.shape[0], *image_shape),
            dtype=image.dtype,
        )
        for smap in smaps:
            segment_images = spatial * (smap * expanded[:, frame])[:, None]
            segment_normal = _dense_apply(
                segment_images,
                transfer,
                image_shape,
            )
            frame_result += (spatial.conj()[None] * segment_normal).sum(
                dim=1
            ) * smap.conj()
        normal_frames.append(frame_result)
    reference = torch.einsum(
        "kt,bt...->bk...",
        basis_complex,
        torch.stack(normal_frames, dim=1),
    )

    torch.testing.assert_close(result, reference, atol=4e-5, rtol=4e-5)
    assert kernel.rank == coefficients * segments
    if torch.cuda.is_available():
        policy = CudaStreaming(
            transfer_chunk_size=3,
            physics_batch_size=1,
            transfer_precision="float32",
        )
        streamed_kernel, streamed_factors = (
            physics._build_subspace_off_resonance_toeplitz(
                frame_physics,
                basis,
                options,
                streaming=policy,
            )
        )
        streamed = physics._apply_subspace_off_resonance_toeplitz(
            streamed_kernel,
            image,
            corrected[0],
            streamed_factors,
            coefficient_rank=coefficients,
            coil_batch_size=2,
            streaming=policy,
        )
        torch.testing.assert_close(streamed, reference, atol=4e-5, rtol=4e-5)
        assert streamed_kernel.values.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_compact_kernel_cuda_matches_cpu():
    generator = torch.Generator().manual_seed(9)
    image_shape = (4, 4, 3)
    spatial_shape = (8, 8, 6)
    rank = 3
    raw = torch.randn(
        rank,
        rank,
        *spatial_shape,
        generator=generator,
    )
    transfer = 0.5 * (raw + raw.movedim(0, 1))
    image = torch.randn(
        1,
        rank,
        *image_shape,
        generator=generator,
        dtype=torch.complex64,
    )
    cpu = _packed_kernel(transfer, image_shape, chunk_size=31)
    expected = cpu.apply(image)
    cpu.cuda_transfer_precision = "float32"
    actual = cpu.apply(image.cuda()).cpu()
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
    assert cpu.last_cuda_mode == "resident"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("real_transfer", [True, False])
def test_resident_cuda_banks_match_compact_cuda(real_transfer):
    generator = torch.Generator().manual_seed(42)
    image_shape = (5, 4, 3)
    spatial_shape = (10, 8, 6)
    rank = 3
    raw = torch.randn(
        rank,
        rank,
        *spatial_shape,
        generator=generator,
        dtype=torch.float32 if real_transfer else torch.complex64,
    )
    transfer = 0.5 * (raw + raw.movedim(0, 1).conj())
    rows, columns = torch.triu_indices(rank, rank)
    values = transfer[rows, columns].reshape(rows.numel(), -1).cuda()
    indices = support_indices(spatial_shape, support="full", device="cuda")
    image = torch.randn(
        2,
        rank,
        *image_shape,
        generator=generator,
        dtype=torch.complex64,
    ).cuda()
    compact = CompactToeplitzKernel(
        values.clone(),
        indices,
        spatial_shape,
        rank,
        image_shape=image_shape,
        cuda_mode="compact",
        cuda_transfer_precision="float32",
    )
    resident = CompactToeplitzKernel(
        values.clone(),
        indices,
        spatial_shape,
        rank,
        image_shape=image_shape,
        cuda_mode="resident",
        cuda_transfer_precision="float32",
    )

    expected = compact.apply(image)
    actual = resident.apply(image)
    first_input = resident._resident_workspaces[
        resident._resident_workspace_key(image)
    ]["input"]
    repeated = resident.apply(image)

    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(repeated, expected, atol=3e-5, rtol=3e-5)
    assert compact.last_cuda_mode == "compact"
    assert resident.last_cuda_mode == "resident"
    assert (
        resident._resident_workspaces[resident._resident_workspace_key(image)][
            "input"
        ].data_ptr()
        == first_input.data_ptr()
    )


def test_resident_cuda_uses_cached_bfloat16_transfer():
    generator = torch.Generator().manual_seed(53)
    image_shape = (6, 5)
    spatial_shape = (12, 10)
    rank = 3
    raw = torch.randn(
        rank,
        rank,
        *spatial_shape,
        generator=generator,
        dtype=torch.float32,
    )
    transfer = 0.5 * (raw + raw.movedim(0, 1).conj())
    rows, columns = torch.triu_indices(rank, rank)
    values = transfer[rows, columns].reshape(rows.numel(), -1)
    indices = support_indices(spatial_shape, support="full")
    image = torch.randn(
        1,
        rank,
        *image_shape,
        generator=generator,
        dtype=torch.complex64,
    ).cuda()
    reference = CompactToeplitzKernel(
        values,
        indices,
        spatial_shape,
        rank,
        image_shape=image_shape,
        cuda_mode="resident",
        cuda_transfer_precision="float32",
    )
    automatic = CompactToeplitzKernel(
        values,
        indices,
        spatial_shape,
        rank,
        image_shape=image_shape,
        cuda_mode="resident",
        cuda_transfer_precision="bfloat16",
    )

    expected = reference.apply(image)
    result = automatic.apply(image)
    workspace = automatic._resident_workspaces[automatic._resident_workspace_key(image)]
    first_pointer = workspace["values"].data_ptr()
    repeated = automatic.apply(image)

    torch.testing.assert_close(result, expected, atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(repeated, result, atol=0, rtol=0)
    assert workspace["values"].dtype == torch.bfloat16
    assert workspace["values"].ndim == 2
    assert (
        automatic._resident_workspaces[automatic._resident_workspace_key(image)][
            "values"
        ].data_ptr()
        == first_pointer
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.skipif(
    not torch.cuda.is_bf16_supported(),
    reason="native CUDA BF16 is unavailable",
)
def test_cuda_auto_precision_uses_bfloat16_only_for_real_kernels():
    rank, locations = 3, 8192
    values = torch.randn(rank * (rank + 1) // 2, locations)
    kernel = CompactToeplitzKernel(
        values,
        torch.arange(locations, dtype=torch.int32),
        (locations,),
        rank,
        image_shape=(locations // 2,),
    )

    complex_kernel = CompactToeplitzKernel(
        torch.complex(values, torch.randn_like(values)),
        torch.arange(locations, dtype=torch.int32),
        (locations,),
        rank,
        image_shape=(locations // 2,),
    )

    assert kernel._cuda_precision("auto", "cuda") == "bfloat16"
    assert complex_kernel._cuda_precision("auto", "cuda") == "float32"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_resident_cuda_fuses_sense_factors_and_accumulation():
    generator = torch.Generator().manual_seed(62)
    image_shape = (5, 4)
    spatial_shape = (10, 8)
    rank, coils = 3, 2
    raw = torch.randn(rank, rank, *spatial_shape, generator=generator)
    transfer = 0.5 * (raw + raw.movedim(0, 1))
    rows, columns = torch.triu_indices(rank, rank)
    values = transfer[rows, columns].reshape(rows.numel(), -1).cuda()
    indices = support_indices(spatial_shape, support="full", device="cuda")
    image = torch.randn(
        1,
        rank,
        *image_shape,
        generator=generator,
        dtype=torch.complex64,
    ).cuda()
    maps = torch.randn(
        coils,
        *image_shape,
        generator=generator,
        dtype=torch.complex64,
    ).cuda()
    native = SimpleNamespace(shape=image_shape, smaps=maps)
    compact = CompactToeplitzKernel(
        values.clone(),
        indices,
        spatial_shape,
        rank,
        image_shape=image_shape,
        cuda_mode="compact",
        cuda_transfer_precision="float32",
    )
    resident = CompactToeplitzKernel(
        values.clone(),
        indices,
        spatial_shape,
        rank,
        image_shape=image_shape,
        cuda_mode="resident",
        cuda_transfer_precision="float32",
    )

    reference = physics._apply_sense_toeplitz(
        compact,
        image,
        native,
        coil_batch_size=1,
    )
    result = physics._apply_sense_toeplitz(
        resident,
        image,
        native,
        coil_batch_size=1,
    )

    torch.testing.assert_close(result, reference, atol=4e-5, rtol=4e-5)
    assert resident.last_cuda_mode == "resident"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_auto_rejects_resident_banks_outside_memory_fraction():
    image_shape = (4, 4)
    spatial_shape = (8, 8)
    indices = support_indices(spatial_shape, support="full", device="cuda")
    values = torch.ones((3, indices.numel()), device="cuda")
    image = torch.ones(
        (1, 2, *image_shape),
        dtype=torch.complex64,
        device="cuda",
    )
    automatic = CompactToeplitzKernel(
        values.clone(),
        indices,
        spatial_shape,
        2,
        image_shape=image_shape,
        cuda_max_device_fraction=1e-9,
    )
    forced = CompactToeplitzKernel(
        values.clone(),
        indices,
        spatial_shape,
        2,
        image_shape=image_shape,
        cuda_mode="resident",
        cuda_max_device_fraction=1e-9,
    )

    automatic.apply(image)

    assert automatic.last_cuda_mode == "compact"
    with pytest.raises(MemoryError, match="resident Toeplitz banks require"):
        forced.apply(image)


def _radial_to(samples: int, spokes: int, extent: float) -> np.ndarray:
    """Spokes reaching ``extent``, with a sample set closed under k -> -k."""
    radius = np.linspace(-extent, extent, samples + 1)[:-1] + extent / samples
    angles = np.arange(spokes) * np.pi / spokes
    return (
        np.stack(
            [np.stack([radius * np.cos(a), radius * np.sin(a)], -1) for a in angles]
        )
        .reshape(-1, 2)
        .astype(np.float32)
    )


def _radial(spokes: int, samples: int) -> np.ndarray:
    angles = np.linspace(0, np.pi, spokes, endpoint=False)
    radius = np.linspace(-0.5, 0.5, samples, endpoint=False)
    return (
        np.stack(
            [np.outer(np.cos(angles), radius), np.outer(np.sin(angles), radius)], -1
        )
        .reshape(-1, 2)
        .astype(np.float32)
    )


def _spiral(turns: int, samples: int) -> np.ndarray:
    angle = np.linspace(0, 2 * np.pi * turns, samples)
    radius = np.linspace(0, 0.5, samples, endpoint=False)
    return np.stack([radius * np.cos(angle), radius * np.sin(angle)], -1).astype(
        np.float32
    )


@pytest.mark.parametrize(
    "trajectory",
    [
        pytest.param(_radial(24, 32), id="radial"),
        pytest.param(_spiral(6, 512), id="spiral"),
    ],
)
def test_the_default_normal_operator_equals_the_one_it_replaces(trajectory):
    """Toeplitz is on out of the box, and it is the same operator.

    The transfer is the trajectory gridded onto the doubled grid, which is the
    Gram of the transform actually being inverted, so the accelerated normal
    answers what the plain adjoint-of-forward answers.
    """
    pytest.importorskip("mrinufft")
    pytest.importorskip("finufft")
    generator = torch.Generator().manual_seed(5)
    image_shape = (16, 16)
    maps = torch.randn(4, *image_shape, generator=generator, dtype=torch.complex64)
    maps /= torch.linalg.vector_norm(maps, dim=0, keepdim=True)
    image = torch.randn(1, 1, *image_shape, generator=generator, dtype=torch.complex64)

    default = physics.NonCartesian2D(
        trajectory, image_shape, coil_maps=maps, backend="finufft"
    )
    whole = physics.NonCartesian2D(
        trajectory,
        image_shape,
        coil_maps=maps,
        backend="finufft",
        toeplitz=True,
    )
    exact = physics.NonCartesian2D(
        trajectory, image_shape, coil_maps=maps, backend="finufft", toeplitz=False
    )

    assert default.normal_mode == "toeplitz"
    assert exact.normal_mode == "exact"
    reference = exact.A_adjoint(exact.A(image))
    scale = reference.abs().max()
    for accelerated in (default, whole):
        error = (accelerated.A_adjoint_A(image) - reference).abs().max() / scale
        assert float(error) < 2e-2


def test_a_stack_accelerates_its_planes_by_default():
    pytest.importorskip("mrinufft")
    pytest.importorskip("finufft")
    generator = torch.Generator().manual_seed(7)
    image_shape = (8, 8, 4)
    trajectory = _radial(12, 16)
    image = torch.randn(1, 1, *image_shape, generator=generator, dtype=torch.complex64)

    default = physics.NonCartesian3D(
        trajectory, image_shape, stacked=True, z_index=[0, 1], backend="finufft"
    )
    exact = physics.NonCartesian3D(
        trajectory,
        image_shape,
        stacked=True,
        z_index=[0, 1],
        backend="finufft",
        toeplitz=False,
    )

    assert default.normal_mode == "toeplitz"
    reference = exact.A_adjoint(exact.A(image))
    error = (default.A_adjoint_A(image) - reference).abs().max() / reference.abs().max()
    assert float(error) < 2e-2


def test_a_kernel_is_built_for_any_image_shape():
    """Gridding does not care whether the matrix is even.

    The transfer is an adjoint onto a grid twice the image, which is defined
    for any shape, so an odd matrix is accelerated like any other.
    """
    pytest.importorskip("mrinufft")
    pytest.importorskip("finufft")
    generator = torch.Generator().manual_seed(3)
    trajectory = _radial(12, 16)
    for size in (15, 16):
        image = torch.randn(
            1, 1, size, size, generator=generator, dtype=torch.complex64
        )
        exact = physics.NonCartesian2D(
            trajectory, (size, size), backend="finufft", toeplitz=False
        )
        accelerated = physics.NonCartesian2D(
            trajectory, (size, size), backend="finufft"
        )
        reference = exact.A_adjoint(exact.A(image))
        assert accelerated.normal_mode == "toeplitz"
        error = (
            accelerated.A_adjoint_A(image) - reference
        ).abs().max() / reference.abs().max()
        assert float(error) < 2e-2


@pytest.mark.parametrize("spokes", [51, 17])
def test_the_accelerated_solve_reproduces_the_exact_one(spokes):
    """A CG-SENSE solve does not care which normal operator it ran on.

    Density compensated and undersampled, which is what a non-Cartesian
    reconstruction actually does: the transfer kernel has to give back the
    same image the plain adjoint-of-forward would, or the acceleration is
    changing the answer rather than the cost.
    """
    pytest.importorskip("mrinufft")
    pytest.importorskip("finufft")
    from pulserver.mrd import pipe_menon_dcf
    from pulserver.recon.optim import pics

    size, coils = 32, 4
    trajectory = _radial_to(2 * size, spokes, 0.5)
    density = np.asarray(pipe_menon_dcf(trajectory, (size, size))).reshape(-1)
    generator = torch.Generator().manual_seed(61)
    maps = torch.randn(coils, size, size, generator=generator, dtype=torch.complex64)
    maps /= torch.linalg.vector_norm(maps, dim=0, keepdim=True)
    axis = torch.linspace(-1, 1, size)
    rows, columns = torch.meshgrid(axis, axis, indexing="ij")
    truth = ((rows**2 + columns**2) < 0.7).to(torch.complex64)[None]

    def solve(toeplitz):
        physics_object = physics.NonCartesian2D(
            trajectory,
            (size, size),
            coil_maps=maps,
            density=density,
            n_coils=coils,
            backend="finufft",
            toeplitz=toeplitz,
        )
        return physics_object, pics(
            data, physics_object, iterations=25, regularization=1e-3
        )

    reference_physics = physics.NonCartesian2D(
        trajectory,
        (size, size),
        coil_maps=maps,
        density=density,
        n_coils=coils,
        backend="finufft",
        toeplitz=False,
    )
    data = reference_physics.A(truth)

    exact_physics, exact = solve(False)
    accelerated_physics, accelerated = solve(True)

    assert exact_physics.normal_mode == "exact"
    assert accelerated_physics.normal_mode == "toeplitz"
    difference = (accelerated - exact).norm() / exact.norm()
    assert float(difference) < 5e-2


def test_the_kernel_is_stored_over_the_support_the_scan_reached():
    """A projection scan leaves a ball, so the cube's corners are not stored.

    The kernel is gridded onto the doubled grid, so it holds weight where the
    trajectory landed and in the neighbourhood the interpolation spread into.
    That is not a setting: an encoding with a trajectory keeps that support,
    and one without keeps every location.
    """
    pytest.importorskip("mrinufft")
    pytest.importorskip("finufft")
    size, views = 20, 900
    index = np.arange(views)
    height = 1 - 2 * (index + 0.5) / views
    azimuth = index * np.pi * (3 - np.sqrt(5))
    radial = np.sqrt(np.maximum(1 - height**2, 0))
    directions = np.stack(
        [radial * np.cos(azimuth), radial * np.sin(azimuth), height], -1
    )
    reach = np.linspace(-0.5, 0.5, 2 * size, endpoint=False)
    trajectory = (directions[:, None, :] * reach[None, :, None]).reshape(-1, 3)

    generator = torch.Generator().manual_seed(71)
    image = torch.randn(
        1, 1, size, size, size, generator=generator, dtype=torch.complex64
    )
    exact = physics.NonCartesian3D(
        trajectory.astype(np.float32),
        (size,) * 3,
        backend="finufft",
        toeplitz=False,
        viewed_as_real=False,
    )
    accelerated = physics.NonCartesian3D(
        trajectory.astype(np.float32),
        (size,) * 3,
        backend="finufft",
        viewed_as_real=False,
    )
    reference = exact.A_adjoint(exact.A(image))
    result = accelerated.A_adjoint_A(image)
    kernel = accelerated.operator.toeplitz_kernel

    assert kernel.n_locations < 8 * size**3
    error = (result - reference).abs().max() / reference.abs().max()
    assert float(error) < 2e-2


@pytest.mark.parametrize("complex_basis", [False, True])
@pytest.mark.parametrize("coils", [1, 3])
def test_a_dynamic_scan_encodes_through_one_plan_over_every_sample(
    complex_basis, coils
):
    """One plan over the whole trajectory answers what a frame at a time does.

    A subspace acquisition is linear in its data, so weighting each frame's
    samples by its basis coefficient and gridding all of them at once is the
    same encoding as gridding frame by frame -- with one plan rather than one
    per frame, and without a volume accumulated per frame.
    """
    pytest.importorskip("mrinufft")
    pytest.importorskip("finufft")
    generator = torch.Generator().manual_seed(19)
    image_shape = (12, 12)
    rank, frames = 3, 7
    trajectory = _framed_radial(frames, 9, image_shape[0])
    basis = torch.randn(rank, frames, generator=generator)
    if complex_basis:
        basis = torch.complex(basis, torch.randn(rank, frames, generator=generator))
    maps = None
    if coils > 1:
        maps = torch.randn(
            coils, *image_shape, generator=generator, dtype=torch.complex64
        )
        maps = maps / maps.abs().pow(2).sum(0, keepdim=True).sqrt()

    built = physics.Subspace(
        physics.NonCartesian2D(
            trajectory,
            image_shape,
            coil_maps=maps,
            n_coils=coils,
            backend="finufft",
            toeplitz=False,
        ),
        basis,
    )
    assert built.operator.flat_encoding is not None

    coefficients = torch.randn(
        1, rank, *image_shape, generator=generator, dtype=torch.complex64
    )
    flat_measurement = built.A(coefficients)
    flat_image = built.A_adjoint(flat_measurement)

    built.operator.__dict__["flat_encoding"] = None
    framewise_measurement = built.A(coefficients)
    framewise_image = built.A_adjoint(framewise_measurement)

    torch.testing.assert_close(
        flat_measurement, framewise_measurement, atol=2e-5, rtol=2e-5
    )
    torch.testing.assert_close(flat_image, framewise_image, atol=2e-5, rtol=2e-5)
    # The adjoint answers the shape the normal operator does, one axis per
    # coefficient and none for coils.
    assert flat_image.shape == (1, rank, *image_shape)


def test_the_encoding_operator_survives_giving_its_plan_up():
    """A build takes the encoding plan and gives it back.

    The gridding plan answers on a grid eight times the encoding one, so the
    two cannot be the same object; they take turns instead, and encoding has
    to work exactly as before once the build is done.
    """
    pytest.importorskip("mrinufft")
    pytest.importorskip("finufft")
    generator = torch.Generator().manual_seed(41)
    image_shape = (12, 12)
    rank, frames = 3, 5
    trajectory = _framed_radial(frames, 9, image_shape[0])
    basis = torch.randn(rank, frames, generator=generator)
    built = physics.Subspace(
        physics.NonCartesian2D(
            trajectory, image_shape, backend="finufft", toeplitz=True
        ),
        basis,
    )
    coefficients = torch.randn(
        1, rank, *image_shape, generator=generator, dtype=torch.complex64
    )
    before = built.A(coefficients)

    built.A_adjoint_A(coefficients)

    torch.testing.assert_close(built.A(coefficients), before, atol=2e-6, rtol=2e-6)


def test_a_released_plan_is_made_by_the_transform_that_needs_it():
    """Nothing plans until a transform runs, and then exactly one does."""
    made: list[int] = []
    pointed: list[tuple[Any, int]] = []
    executed: list[str] = []

    class _Plan:
        def __init__(self, typ: int) -> None:
            self.typ = typ

        def execute(self, first, second):
            executed.append(f"{self.typ}:{first}{second}")
            return first

    class _Raw:
        def __init__(self) -> None:
            self.plans = [None, None, None]

        def _make_plan(self, typ, **_settings):
            made.append(typ)
            self.plans[typ] = _Plan(typ)

        def _set_pts(self, typ, samples):
            pointed.append((samples, typ))

    raw = _Raw()
    physics._plans_made_when_asked(raw, {"eps": 1e-6}, "build")

    assert made == []

    assert raw.type1("a", "b") == "a"
    assert made == [1]
    assert pointed == [("build", 1)]
    assert executed == ["1:ab"]

    raw.type2("c", "d")

    assert made == [1, 2]
    assert executed == ["1:ab", "2:cd"]


def test_an_unplanned_operator_can_still_be_aimed_at_new_samples():
    """Points wait for the plan instead of asking an absent one to take them."""
    pointed: list[tuple[Any, int]] = []

    class _Plan:
        def execute(self, first, _second):
            return first

    class _Raw:
        def __init__(self) -> None:
            self.plans = [None, None, None]

        def _make_plan(self, typ, **_settings):
            self.plans[typ] = _Plan()

        def _set_pts(self, typ, samples):
            pointed.append((samples, typ))

    raw = _Raw()
    physics._plans_made_when_asked(raw, {"eps": 1e-6}, "build")

    raw._set_pts(1, "frame")
    raw._set_pts(2, "frame")

    assert pointed == []

    raw.type1("a", "b")

    assert pointed == [("frame", 1)]

    raw._set_pts(1, "next")

    assert pointed == [("frame", 1), ("next", 1)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_a_build_leaves_the_card_to_the_kernel_it_built():
    """The encoding plans go for the build and do not come back on their own.

    A kernel exists to stand in for the transform, so the transform's plan is
    only worth the memory once someone encodes again.
    """
    generator = torch.Generator().manual_seed(43)
    image_shape = (12, 12)
    rank, frames = 3, 5
    trajectory = torch.from_numpy(_framed_radial(frames, 9, image_shape[0])).cuda()
    basis = torch.randn(rank, frames, generator=generator).cuda()
    built = physics.Subspace(
        physics.NonCartesian2D(
            trajectory, image_shape, backend="cufinufft", toeplitz=True
        ),
        basis,
    )
    coefficients = torch.randn(
        1, rank, *image_shape, generator=generator, dtype=torch.complex64
    ).cuda()
    before = built.A(coefficients)
    encoding = built.frame_physics[0].provider.physics.native_operator
    raw = physics._base_fourier_operator(encoding).raw_op

    built.A_adjoint_A(coefficients)

    assert raw.plans[1] is None
    torch.testing.assert_close(built.A(coefficients), before, atol=2e-6, rtol=2e-6)


def test_a_kernel_hands_the_allocator_its_blocks_back_exactly_once():
    """A whole application settles the allocator, and only the first one does."""
    generator = torch.Generator().manual_seed(19)
    image_shape = (3, 4)
    spatial_shape = (6, 8)
    rank = 2
    raw = torch.randn(rank, rank, *spatial_shape, generator=generator)
    transfer = 0.5 * (raw + raw.movedim(0, 1))
    kernel = _packed_kernel(transfer, image_shape)

    with mock.patch("torch.cuda.empty_cache") as released:
        kernel.settle_allocator()
        kernel.settle_allocator()
        kernel.settle_allocator()

    assert released.call_count == 1


def test_the_resident_layout_asks_for_two_banks_and_a_workspace():
    """The estimate covers the banks it allocates, not a third one it does not."""
    generator = torch.Generator().manual_seed(23)
    image_shape = (4, 5)
    spatial_shape = (8, 10)
    rank = 3
    raw = torch.randn(rank, rank, *spatial_shape, generator=generator)
    kernel = _packed_kernel(0.5 * (raw + raw.movedim(0, 1)), image_shape)
    image = torch.zeros(2, rank, *image_shape, dtype=torch.complex64)

    required = kernel._resident_additional_bytes(image)

    volume = 2 * prod(spatial_shape) * image.element_size()
    banks = rank * volume
    result = image.numel() * image.element_size()
    assert required == 2 * banks + volume + result


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_a_device_that_refuses_the_banks_falls_back_instead_of_raising():
    """A tight estimate is only safe if being wrong costs speed, not the run."""
    generator = torch.Generator().manual_seed(24)
    image_shape = (4, 5)
    spatial_shape = (8, 10)
    rank = 3
    raw = torch.randn(rank, rank, *spatial_shape, generator=generator)
    transfer = 0.5 * (raw + raw.movedim(0, 1))
    kernel = _packed_kernel(transfer, image_shape).to("cuda")
    kernel.cuda_transfer_precision = "float32"
    image = torch.randn(
        2, rank, *image_shape, generator=generator, dtype=torch.complex64
    ).cuda()
    expected = _dense_apply(image, transfer.to(image.dtype).cuda(), image_shape)

    with mock.patch.object(
        kernel,
        "_apply_cuda_resident",
        side_effect=RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"),
    ):
        result = kernel.apply(image)

    torch.testing.assert_close(result, expected, atol=2e-5, rtol=2e-5)
    assert kernel._resident_denied
    assert kernel.last_cuda_mode == "compact"
    assert kernel._select_cuda_mode(image) == "compact"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_a_narrow_encoding_replaces_the_canonical_values_rather_than_joining_them():
    """Half-width storage is only a saving if the full-width copy leaves."""
    generator = torch.Generator().manual_seed(31)
    image_shape = (4, 5)
    spatial_shape = (8, 10)
    rank = 3
    raw = torch.randn(rank, rank, *spatial_shape, generator=generator)
    kernel = _packed_kernel(0.5 * (raw + raw.movedim(0, 1)), image_shape).to("cuda")
    assert kernel.values.device.type == "cuda"

    encoded = kernel._encoded_values_for("cuda", "bfloat16")

    assert encoded.dtype is torch.bfloat16
    assert encoded.device.type == "cuda"
    assert kernel.values.device.type == "cpu"
    assert kernel._encoded_values_for("cuda", "bfloat16") is encoded


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_maps_are_read_where_they_rest_and_staged_a_coil_at_a_time():
    """Maps the caller left on the host are not moved to the device whole."""
    generator = torch.Generator().manual_seed(37)
    image_shape = (8, 8)
    coils, rank = 4, 2
    maps = torch.randn(coils, *image_shape, generator=generator, dtype=torch.complex64)
    maps = maps / maps.abs().pow(2).sum(0, keepdim=True).sqrt()
    operator = SimpleNamespace(shape=image_shape, smaps=maps, uses_sense=True)
    image = torch.zeros(1, rank, *image_shape, dtype=torch.complex64).cuda()

    held = physics._sense_maps(operator, image)

    assert held.device.type == "cpu"

    raw = torch.randn(
        rank, rank, *[2 * size for size in image_shape], generator=generator
    )
    kernel = _packed_kernel(0.5 * (raw + raw.movedim(0, 1)), image_shape).to("cuda")
    kernel.cuda_transfer_precision = "float32"

    staged = physics._apply_sense_toeplitz(kernel, image, operator)
    whole = physics._apply_sense_toeplitz(
        kernel,
        image,
        SimpleNamespace(shape=image_shape, smaps=maps.cuda(), uses_sense=True),
    )

    assert maps.device.type == "cpu"
    torch.testing.assert_close(staged, whole, atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("batched_maps", [False, True])
def test_coils_split_over_devices_sum_to_the_undivided_answer(batched_maps):
    """Coils are independent until the sum, so dividing them must change nothing.

    One device standing in for several exercises the division, the per-device
    kernel views and the sum; it cannot speak for what two cards do between
    themselves.
    """
    generator = torch.Generator().manual_seed(51)
    image_shape = (4, 5)
    spatial_shape = (8, 10)
    rank, coils, batch = 3, 5, 2
    raw = torch.randn(rank, rank, *spatial_shape, generator=generator)
    kernel = _packed_kernel(0.5 * (raw + raw.movedim(0, 1)), image_shape).to("cuda")
    kernel.cuda_transfer_precision = "float32"
    shape = (batch, coils, *image_shape) if batched_maps else (coils, *image_shape)
    maps = torch.randn(*shape, generator=generator, dtype=torch.complex64).cuda()
    image = torch.randn(
        batch, rank, *image_shape, generator=generator, dtype=torch.complex64
    ).cuda()
    operator = SimpleNamespace(shape=image_shape, smaps=maps, uses_sense=True)

    whole = physics._apply_sense_toeplitz(kernel, image, operator)

    policy = CudaStreaming(streams=2, pin_memory=False)
    with mock.patch.object(
        type(policy),
        "torch_devices",
        property(lambda self: (torch.device("cuda:0"),) * 2),
    ):
        assert policy.device_count == 2
        divided = physics._apply_sense_toeplitz(
            kernel, image, operator, streaming=policy
        )

    torch.testing.assert_close(divided, whole, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("real_transfer", [True, False])
def test_the_dense_host_multiply_equals_the_gathered_one(real_transfer):
    """Widening the multiply to the whole grid must not change the answer."""
    generator = torch.Generator().manual_seed(61)
    image_shape = (3, 4)
    spatial_shape = (6, 8)
    rank = 3
    raw = torch.randn(rank, rank, *spatial_shape, generator=generator)
    if real_transfer:
        transfer = 0.5 * (raw + raw.movedim(0, 1))
    else:
        imaginary = torch.randn(rank, rank, *spatial_shape, generator=generator)
        full = torch.complex(raw, imaginary)
        transfer = 0.5 * (full + full.movedim(0, 1).conj())
    image = torch.randn(
        2, rank, *image_shape, generator=generator, dtype=torch.complex64
    )

    kernel = _packed_kernel(transfer, image_shape)
    kernel.host_dense = "always"
    dense = kernel.apply(image)
    kernel.host_dense = "never"
    gathered = kernel.apply(image)
    reference = _dense_apply(image, transfer.to(image.dtype), image_shape)

    torch.testing.assert_close(dense, gathered, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(dense, reference, atol=2e-5, rtol=2e-5)


def test_the_host_multiply_stays_lean_when_the_grid_would_not_fit():
    """The dense form is taken only when the host has room for it."""
    generator = torch.Generator().manual_seed(62)
    image_shape = (3, 4)
    spatial_shape = (6, 8)
    rank = 2
    raw = torch.randn(rank, rank, *spatial_shape, generator=generator)
    kernel = _packed_kernel(0.5 * (raw + raw.movedim(0, 1)), image_shape)
    image = torch.zeros(1, rank, *image_shape, dtype=torch.complex64)

    assert kernel._host_multiply_is_dense(image)

    kernel.host_max_memory_fraction = 1e-12

    assert not kernel._host_multiply_is_dense(image)


def test_asking_a_dynamic_physics_for_toeplitz_turns_it_on():
    """The explicit opt-in reaches a subspace physics built without it.

    A subspace physics decides whether it has a kernel from what its frames
    report, so a frame asked to use one has to say so afterwards.
    """
    pytest.importorskip("mrinufft")
    pytest.importorskip("finufft")
    generator = torch.Generator().manual_seed(31)
    image_shape = (12, 12)
    rank, frames = 3, 6
    built = physics.Subspace(
        physics.NonCartesian2D(
            _framed_radial(frames, 9, image_shape[0]),
            image_shape,
            backend="finufft",
            toeplitz=False,
        ),
        torch.randn(rank, frames, generator=generator),
    )
    assert built.normal_mode == "exact"

    turned = physics.Toeplitz(built)

    assert turned.normal_mode == "toeplitz"
    assert turned.operator.use_toeplitz
    coefficients = torch.randn(
        1, rank, *image_shape, generator=generator, dtype=torch.complex64
    )
    torch.testing.assert_close(
        turned.A_adjoint_A(coefficients),
        built.A_adjoint(built.A(coefficients)),
        atol=2e-4,
        rtol=2e-4,
    )


@pytest.mark.parametrize("planned", [1, 2])
@pytest.mark.parametrize("kind", ["scalar", "subspace"])
def test_a_non_cartesian_operator_takes_any_number_of_images(kind, planned):
    """A batch of three answers what three separate calls answer.

    An mri-nufft plan is sized by ``n_trans``; ``n_batchs`` only says how to
    fold the input. A call carrying a different number of images than the
    operator was told to expect is served either way, so the leading axis is a
    batch axis here as it is for a Cartesian operator.
    """
    pytest.importorskip("mrinufft")
    pytest.importorskip("finufft")
    generator = torch.Generator().manual_seed(29)
    image_shape = (12, 12)
    coils, batch = 3, 3
    maps = torch.randn(coils, *image_shape, generator=generator, dtype=torch.complex64)
    maps = maps / maps.abs().pow(2).sum(0, keepdim=True).sqrt()

    if kind == "scalar":
        trajectory = _framed_radial(1, 9, image_shape[0])[0]
        built = physics.NonCartesian2D(
            trajectory,
            image_shape,
            coil_maps=maps,
            backend="finufft",
            n_batchs=planned,
        )
        image = torch.randn(
            batch, *image_shape, generator=generator, dtype=torch.complex64
        )
    else:
        rank, frames = 3, 7
        trajectory = _framed_radial(frames, 9, image_shape[0])
        built = physics.Subspace(
            physics.NonCartesian2D(
                trajectory,
                image_shape,
                coil_maps=maps,
                backend="finufft",
                n_batchs=planned,
            ),
            torch.randn(rank, frames, generator=generator),
        )
        image = torch.randn(
            batch, rank, *image_shape, generator=generator, dtype=torch.complex64
        )

    measurement = built.A(image)
    adjoint = built.A_adjoint(measurement)
    normal = built.A_adjoint_A(image)

    one_at_a_time = torch.cat([built.A(image[i : i + 1]) for i in range(batch)])
    torch.testing.assert_close(measurement, one_at_a_time, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(
        adjoint,
        torch.cat([built.A_adjoint(measurement[i : i + 1]) for i in range(batch)]),
        atol=2e-5,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        normal,
        torch.cat([built.A_adjoint_A(image[i : i + 1]) for i in range(batch)]),
        atol=2e-5,
        rtol=2e-5,
    )


def test_one_plan_serves_a_dynamic_scan_however_many_frames_it_has():
    """The encoding plans once, not once per frame."""
    pytest.importorskip("mrinufft")
    pytest.importorskip("finufft")
    generator = torch.Generator().manual_seed(23)
    image_shape = (12, 12)
    rank = 3
    for frames in (4, 16):
        trajectory = _framed_radial(frames, 9, image_shape[0])
        basis = torch.randn(rank, frames, generator=generator)
        built = physics.Subspace(
            physics.NonCartesian2D(
                trajectory, image_shape, backend="finufft", toeplitz=False
            ),
            basis,
        )
        provider = built.operator.frame_physics[0].provider
        built.A_adjoint(
            torch.zeros(1, frames, 1, 9 * image_shape[0], dtype=torch.complex64)
        )
        # The frame-at-a-time provider is never asked for an operator, so the
        # plan it would build per frame is never built at all.
        assert provider.shared is None
        assert not provider.cache


def test_the_noncartesian_solve_keeps_the_whole_kernel():
    """CG needs a positive-definite normal, which truncation cannot promise.

    The exact normal operator of a SENSE solve has eigenvalues at zero, so a
    kernel cut to the locations the samples reached carries the smallest ones
    negative and CG stops on a non-positive recurrence. One plane's kernel is
    small enough that keeping all of it costs nothing worth having, so the
    recipe asks for the whole thing.
    """
    import numpy as np

    from pulserver.recon import NonCartesian2D, noncartesian_recon

    size, spokes = 64, int(np.ceil(np.pi / 2 * 64))
    angles = np.linspace(0, np.pi, spokes, endpoint=False)
    radius = np.linspace(-0.5, 0.5, 2 * size, endpoint=False)
    trajectory = (
        np.stack(
            [np.outer(np.cos(angles), radius), np.outer(np.sin(angles), radius)], -1
        )
        .reshape(-1, 2)
        .astype(np.float32)
    )

    truth = np.zeros((size, size), dtype=np.complex64)
    rows, columns = np.mgrid[0:size, 0:size]
    truth[((rows - size / 2) ** 2 + (columns - size / 2) ** 2) < (size / 3) ** 2] = 1.0

    axes = np.meshgrid(*(np.linspace(-1, 1, size),) * 2, indexing="ij")
    maps = np.stack(
        [
            np.exp(-((axes[0] - x) ** 2 + (axes[1] - y) ** 2))
            for x, y in ((-0.6, -0.6), (-0.6, 0.6), (0.6, -0.6), (0.6, 0.6))
        ]
    ).astype(np.complex64)
    maps /= np.sqrt((abs(maps) ** 2).sum(0, keepdims=True))

    encoding = NonCartesian2D(
        trajectory, (size, size), coil_maps=torch.as_tensor(maps), n_coils=4
    )
    measured = encoding.A(torch.as_tensor(truth)[None]).detach()[0]

    solved = noncartesian_recon(
        np.asarray(measured.cpu()), trajectory, (size, size), mode="pics"
    )

    assert np.isfinite(solved).all()
    inside = np.abs(truth) > 0
    assert np.abs(solved)[inside].mean() > 3 * np.abs(solved)[~inside].mean()


def test_the_off_resonance_kernel_equals_the_normal_it_stands_for():
    """The segmented kernel is the Gram of the corrected operator.

    One transfer per pair of interpolation segments, weighted by the temporal
    factors of the pair, between the spatial factor of the segment it enters
    and the conjugate of the one it leaves by. Held against the operator it
    replaces, because a fast normal that disagrees is a different operator.
    """
    import numpy as np

    from pulserver.recon import NonCartesian2D, OffResonance

    size, spokes, samples = 64, 48, 64
    angles = np.linspace(0, np.pi, spokes, endpoint=False)
    radius = np.linspace(-0.5, 0.5, samples, endpoint=False)
    trajectory = (
        np.stack(
            [np.outer(np.cos(angles), radius), np.outer(np.sin(angles), radius)], -1
        )
        .reshape(-1, 2)
        .astype(np.float32)
    )
    readout_time = np.tile(np.linspace(0, 8e-3, samples, dtype=np.float32), spokes)
    field_map = np.zeros((size, size), dtype=np.float32)
    field_map[:, size // 2 :] = 150.0
    maps = torch.ones(4, size, size, dtype=torch.complex64) / 2.0

    def corrected(toeplitz):
        base = NonCartesian2D(
            trajectory, (size, size), coil_maps=maps, n_coils=4, toeplitz=toeplitz
        )
        return OffResonance(base, field_map, readout_time)

    image = torch.randn(1, 1, size, size, dtype=torch.complex64)
    exact = corrected(False)
    reference = exact.A_adjoint(exact.A(image))
    scale = reference.abs().max()

    whole = corrected({"compress": False}).A_adjoint_A(image)
    assert float((whole - reference).abs().max() / scale) < 1e-4

    # Keeping only the locations the samples reached is an approximation, and
    # a far coarser one than the transform's own error.
    compressed = corrected({"compress": True}).A_adjoint_A(image)
    assert float((compressed - reference).abs().max() / scale) < 1e-2


def test_the_correction_is_worth_making():
    """A corrected solve beats an uncorrected one on off-resonant data.

    The kernel agreeing with its operator says nothing about whether the
    operator is the right one, so this holds the physics as well: a readout
    long enough to accumulate phase reconstructs blurred when the field is
    ignored and sharp when it is not.
    """
    import numpy as np

    from pulserver import recon as recon_module

    size, spokes, samples = 64, 48, 64
    angles = np.linspace(0, np.pi, spokes, endpoint=False)
    radius = np.linspace(-0.5, 0.5, samples, endpoint=False)
    trajectory = (
        np.stack(
            [np.outer(np.cos(angles), radius), np.outer(np.sin(angles), radius)], -1
        )
        .reshape(-1, 2)
        .astype(np.float32)
    )
    readout_time = np.tile(np.linspace(0, 16e-3, samples, dtype=np.float32), spokes)
    field_map = np.zeros((size, size), dtype=np.float32)
    field_map[:, size // 2 :] = 500.0

    truth = torch.zeros(1, size, size, dtype=torch.complex64)
    rows, columns = torch.meshgrid(
        torch.arange(size), torch.arange(size), indexing="ij"
    )
    truth[0][((rows - size / 2) ** 2 + (columns - size / 2) ** 2) < (size / 3) ** 2] = (
        1.0
    )
    truth[0][
        ((rows - size / 2 - 8) ** 2 + (columns - size / 2 + 8) ** 2) < (size / 12) ** 2
    ] = 2.0
    maps = torch.ones(4, size, size, dtype=torch.complex64) / 2.0

    base = recon_module.NonCartesian2D(
        trajectory, (size, size), coil_maps=maps, n_coils=4
    )
    corrected = recon_module.OffResonance(base, field_map, readout_time)
    measured = corrected.A(truth)

    def error(operator):
        solved = recon_module.pics(
            measured, operator, regularization=1e-3, iterations=10
        )[0]
        solved = solved.detach().abs().squeeze()
        scale = (solved * truth[0].abs()).sum() / solved.pow(2).sum().clamp_min(1e-12)
        return float((scale * solved - truth[0].abs()).norm() / truth[0].abs().norm())

    assert error(corrected) < 0.5 * error(base)
