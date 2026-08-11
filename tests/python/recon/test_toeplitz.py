"""Tests for compact Torch Toeplitz transfer kernels."""

from __future__ import annotations

from os import cpu_count
from types import SimpleNamespace

import numpy as np
import pytest

from pulserver.recon._toeplitz import CompactToeplitzKernel, support_indices
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
        "pulserver._ext._recon_cpu_wrapper",
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
        "pulserver._ext._recon_cpu_wrapper",
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


@pytest.mark.parametrize("complex_basis", [False, True])
def test_subspace_builder_matches_framewise_normal(complex_basis):
    generator = torch.Generator().manual_seed(4)
    image_shape = (3, 3)
    spatial_shape = (6, 6)
    rank, frames = 3, 4
    scalar_kernels = [
        torch.randn(*spatial_shape, generator=generator) for _ in range(frames)
    ]
    natives = [
        SimpleNamespace(
            shape=image_shape,
            smaps=None,
            compute_toeplitz_kernel=lambda kernel=kernel: kernel,
        )
        for kernel in scalar_kernels
    ]
    frame_physics = [SimpleNamespace(native_operator=native) for native in natives]
    basis = torch.randn(rank, frames, generator=generator)
    if complex_basis:
        basis = torch.complex(
            basis,
            torch.randn(rank, frames, generator=generator),
        )
    options = physics._toeplitz_options(support="full", chunk_size=5)
    kernel = physics._build_subspace_toeplitz(frame_physics, basis, options)
    coefficients = torch.randn(
        2,
        rank,
        *image_shape,
        generator=generator,
        dtype=torch.complex64,
    )

    result = kernel.apply(coefficients)
    basis_complex = basis.to(coefficients.dtype)
    expanded = torch.einsum(
        "kt,bk...->bt...",
        basis_complex.conj(),
        coefficients,
    )
    normal_frames = []
    for frame in range(frames):
        scalar = scalar_kernels[frame][None, None]
        normal_frames.append(
            _dense_apply(
                expanded[:, frame : frame + 1],
                scalar.to(coefficients.dtype),
                image_shape,
            )
        )
    reference = torch.einsum(
        "kt,bt...->bk...",
        basis_complex,
        torch.cat(normal_frames, dim=1),
    )

    torch.testing.assert_close(result, reference, atol=2e-5, rtol=2e-5)
    assert kernel.is_real is (not complex_basis)


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
        support="full",
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
        support="full",
    )
    real_coefficients = exact.operator._image_as_real(coefficients)

    reference = exact.A_adjoint_A(real_coefficients)
    result = compact.A_adjoint_A(real_coefficients)

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
        support="full",
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
        support="full",
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_resident_cuda_fp16_transfer_uses_fp32_accumulation():
    generator = torch.Generator().manual_seed(52)
    image_shape = (6, 5)
    spatial_shape = (12, 10)
    rank = 3
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
    fp32 = CompactToeplitzKernel(
        values.clone(),
        indices,
        spatial_shape,
        rank,
        image_shape=image_shape,
        cuda_mode="resident",
        cuda_transfer_precision="float32",
    )
    fp16 = CompactToeplitzKernel(
        values.clone(),
        indices,
        spatial_shape,
        rank,
        image_shape=image_shape,
        cuda_mode="resident",
        cuda_transfer_precision="float16",
    )

    reference = fp32.apply(image)
    result = fp16.apply(image)

    torch.testing.assert_close(result, reference, atol=2e-3, rtol=2e-3)
    workspace = fp16._resident_workspaces[fp16._resident_workspace_key(image)]
    assert workspace["values"].dtype == torch.float16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.skipif(
    not torch.cuda.is_bf16_supported(),
    reason="native CUDA BF16 is unavailable",
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
