"""Tests for host-backed CUDA reconstruction execution."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from pulserver.recon.optim import pics
from pulserver.recon.denoisers import LLR, TGV, TV, Wavelet
from pulserver.recon.execution import CudaStreaming
from pulserver.recon.physics import (
    Cartesian2D,
    MRIPhysics,
    Subspace,
)
from pulserver.recon._cuda_streaming import (
    CudaBatchDenoiser,
    _CudaWorkerMemoryError,
    tensor_nbytes,
)
from pulserver.recon._toeplitz import CompactToeplitzKernel, support_indices
from pulserver.recon.physics import _apply_sense_toeplitz


torch = pytest.importorskip("torch")


def test_streaming_policy_validation_and_tensor_size():
    with pytest.raises(ValueError, match="one or two"):
        CudaStreaming(streams=3)
    with pytest.raises(ValueError, match="CUDA"):
        CudaStreaming(device="cpu")
    value = torch.empty(2, 3, 4, dtype=torch.complex64)
    assert tensor_nbytes(value) == 2 * 3 * 4 * 8


def test_streaming_policy_discovers_devices_unless_one_is_explicit(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    automatic = CudaStreaming()
    explicit = CudaStreaming(device="cuda:1")

    assert tuple(map(str, automatic.torch_devices)) == ("cuda:0", "cuda:1")
    assert tuple(map(str, automatic.execution_devices)) == (
        "cuda:0",
        "cuda:1",
        "cuda:0",
        "cuda:1",
    )
    assert tuple(map(str, explicit.torch_devices)) == ("cuda:1",)
    assert automatic.for_device("cuda:1").device_count == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_exact_batch_denoiser_preserves_order_and_tensor_conditions():
    class Conditioned(torch.nn.Module):
        def forward(self, value, *, sigma=None, gain=None):
            sigma = sigma.reshape(-1, 1, 1, 1)
            return value + sigma + gain

    model = Conditioned().eval()
    value = torch.arange(7 * 2 * 3 * 4, dtype=torch.float32).reshape(7, 2, 3, 4)
    sigma = torch.arange(7, dtype=torch.float32)
    gain = torch.full_like(value, 2.0)
    executor = CudaStreaming(streams=2).wrap_batch_denoiser(
        model,
        max_batch_size=2,
    )

    result = executor(value, sigma=sigma, gain=gain)

    expected = value + sigma[:, None, None, None] + gain
    torch.testing.assert_close(result, expected)
    assert result.device.type == "cpu"
    assert len(executor._models) == 2
    assert executor._batch_limits == {}


def test_batch_denoiser_retains_single_stream_fallback(monkeypatch):
    policy = SimpleNamespace(
        execution_devices=("cuda:0", "cuda:1", "cuda:0", "cuda:1"),
        torch_devices=("cuda:0", "cuda:1"),
        ensure_available=lambda: None,
    )
    executor = CudaBatchDenoiser(torch.nn.Identity().eval(), policy)
    calls = []

    def execute(value, *, sigma, kwargs, devices):
        del sigma, kwargs
        calls.append(devices)
        if len(calls) == 1:
            raise _CudaWorkerMemoryError("concurrent workers exceed VRAM")
        return value

    monkeypatch.setattr(executor, "_forward_on_devices", execute)
    monkeypatch.setattr(executor, "_release_worker_models", lambda: None)
    value = torch.zeros(4, 1, 2, 2)

    assert executor(value) is value
    assert executor(value) is value
    assert calls == [
        policy.execution_devices,
        policy.torch_devices,
        policy.torch_devices,
    ]


def test_physics_replicas_split_leading_batch_across_devices():
    class Replica:
        def __init__(self, offset):
            self.offset = offset

        def A(self, value):
            return value + self.offset

    selected = MRIPhysics(
        SimpleNamespace(A=lambda value: value),
        native_operator=None,
        kind="test",
        spatial_ndim=2,
    )
    selected.streaming_policy = SimpleNamespace(
        torch_devices=(torch.device("cuda:0"), torch.device("cuda:1")),
    )
    selected._streaming_replicas = {
        "cuda:0": Replica(10),
        "cuda:1": Replica(20),
    }
    value = torch.arange(4)[:, None]

    result = selected.A(value)

    torch.testing.assert_close(result[:2], value[:2] + 10)
    torch.testing.assert_close(result[2:], value[2:] + 20)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_streamed_compact_toeplitz_matches_in_core_cuda():
    generator = torch.Generator().manual_seed(42)
    image_shape = (4, 4, 4)
    spatial_shape = (8, 8, 8)
    rank = 3
    indices = support_indices(spatial_shape, support="radial")
    rows, columns = torch.triu_indices(rank, rank)
    raw = torch.randn(rank, rank, *spatial_shape, generator=generator)
    transfer = 0.5 * (raw + raw.transpose(0, 1))
    kernel = CompactToeplitzKernel(
        transfer[rows, columns].reshape(rows.numel(), -1)[:, indices],
        indices,
        spatial_shape,
        rank,
        image_shape=image_shape,
        chunk_size=31,
    )
    image = torch.randn(
        1,
        rank,
        *image_shape,
        generator=generator,
        dtype=torch.complex64,
    )

    reference = kernel.apply(image)
    result = kernel.apply_streamed(
        image,
        CudaStreaming(
            streams=2,
            transfer_chunk_size=29,
            transfer_precision="float32",
        ),
    )
    reduced = kernel.apply_streamed(
        image,
        CudaStreaming(
            streams=2,
            transfer_chunk_size=29,
            transfer_precision=(
                "bfloat16" if torch.cuda.is_bf16_supported() else "float32"
            ),
        ),
    )

    torch.testing.assert_close(result, reference, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(reduced, reference, atol=2e-3, rtol=2e-3)
    assert result.device.type == "cpu"
    assert kernel.values.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_streamed_complex_hermitian_toeplitz_matches_in_core_cuda():
    generator = torch.Generator().manual_seed(43)
    image_shape = (4, 4, 4)
    spatial_shape = (8, 8, 8)
    rank = 3
    indices = support_indices(spatial_shape, support="radial")
    rows, columns = torch.triu_indices(rank, rank)
    raw = torch.randn(
        rank,
        rank,
        *spatial_shape,
        generator=generator,
        dtype=torch.complex64,
    )
    transfer = 0.5 * (raw + raw.transpose(0, 1).conj())
    kernel = CompactToeplitzKernel(
        transfer[rows, columns].reshape(rows.numel(), -1)[:, indices],
        indices,
        spatial_shape,
        rank,
        image_shape=image_shape,
        chunk_size=31,
    )
    image = torch.randn(
        1,
        rank,
        *image_shape,
        generator=generator,
        dtype=torch.complex64,
    )

    reference = kernel.apply(image)
    full_cuda = kernel.apply(image.cuda()).cpu()
    kernel.to("cpu")
    result = kernel.apply_streamed(
        image,
        CudaStreaming(
            streams=2,
            transfer_chunk_size=29,
            transfer_precision="float32",
        ),
    )

    torch.testing.assert_close(full_cuda, reference, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(result, reference, atol=3e-5, rtol=3e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_streamed_sense_fuses_spatial_factors_on_cuda():
    generator = torch.Generator().manual_seed(44)
    image_shape = (4, 4, 4)
    spatial_shape = (8, 8, 8)
    rank = 3
    indices = support_indices(spatial_shape, support="radial")
    rows, columns = torch.triu_indices(rank, rank)
    raw = torch.randn(rank, rank, *spatial_shape, generator=generator)
    transfer = 0.5 * (raw + raw.transpose(0, 1))
    kernel = CompactToeplitzKernel(
        transfer[rows, columns].reshape(rows.numel(), -1)[:, indices],
        indices,
        spatial_shape,
        rank,
        image_shape=image_shape,
        chunk_size=31,
    )
    image = torch.randn(
        1,
        rank,
        *image_shape,
        generator=generator,
        dtype=torch.complex64,
    )
    maps = torch.randn(
        2,
        *image_shape,
        generator=generator,
        dtype=torch.complex64,
    )
    reference = sum(
        kernel.apply(image * coil_map[None, None]) * coil_map.conj()[None, None]
        for coil_map in maps
    )
    result = _apply_sense_toeplitz(
        kernel,
        image.pin_memory(),
        SimpleNamespace(shape=image_shape, smaps=maps),
        streaming=CudaStreaming(
            streams=2,
            transfer_chunk_size=29,
            transfer_precision="float32",
        ),
    )

    torch.testing.assert_close(result, reference, atol=3e-5, rtol=3e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_streamed_llr_is_exact_for_aligned_nonoverlapping_slabs():
    generator = torch.Generator().manual_seed(51)
    image = torch.randn(
        1,
        3,
        8,
        8,
        8,
        generator=generator,
        dtype=torch.complex64,
    )
    reference_model = LLR(
        dimension=3,
        block_size=2,
        cycle_spins=False,
        block_batch_size=8,
    )
    streamed_model = LLR(
        dimension=3,
        block_size=2,
        cycle_spins=False,
        block_batch_size=8,
    )
    policy = CudaStreaming(
        denoiser_slab_size=4,
        denoiser_halo=0,
    )

    reference = reference_model(image.to("cuda"), 0.1).cpu()
    result = policy.wrap_denoiser(streamed_model)(image, 0.1)

    torch.testing.assert_close(result, reference, atol=0, rtol=0)


@pytest.mark.parametrize(
    "model",
    [
        pytest.param(TV(n_it_max=2, crit=0), id="tv"),
        pytest.param(TGV(n_it_max=2, crit=0), id="tgv"),
        pytest.param(
            Wavelet(dimension=3, wavelet="db2", level=1),
            id="wavelet",
        ),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_global_denoisers_run_on_overlapping_streamed_slabs(model):
    pytest.importorskip("ptwt")
    image = torch.randn(1, 2, 12, 12, 12)
    policy = CudaStreaming(
        denoiser_slab_size=6,
        denoiser_halo=2,
    )

    result = policy.wrap_denoiser(model)(image, 0.05)

    assert result.shape == image.shape
    assert result.device.type == "cpu"
    assert torch.isfinite(result).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_pics_keeps_streamed_fista_iterates_on_cpu():
    class IdentityPhysics:
        def __init__(self):
            self.policy = None

        def enable_streaming(self, policy):
            self.policy = policy

        @staticmethod
        def A(value):
            return value

        @staticmethod
        def A_adjoint(value):
            return value

        @staticmethod
        def A_adjoint_A(value):
            return value

    class IdentityDenoiser(torch.nn.Module):
        @staticmethod
        def forward(value, _sigma):
            return value

    data = torch.randn(1, 2, 8, 8)
    physics = IdentityPhysics()
    policy = CudaStreaming(
        denoiser_slab_size=4,
        denoiser_halo=0,
    )

    result = pics(
        data,
        physics,
        IdentityDenoiser(),
        regularization=0.1,
        iterations=2,
        streaming=policy,
    )

    assert physics.policy is policy
    assert result.device.type == "cpu"
    assert result.shape == data.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_streamed_subspace_adjoint_projects_frames_incrementally():
    class FrameIdentity:
        use_toeplitz = False

        @staticmethod
        def A_adjoint(value):
            return value

        @staticmethod
        def A(value):
            return value

        @staticmethod
        def A_adjoint_A(value):
            return value

    native = type("Native", (), {"shape": (6, 6), "smaps": None})()
    base = MRIPhysics(
        FrameIdentity(),
        native_operator=native,
        kind="noncartesian2d",
        spatial_ndim=2,
        viewed_as_real=False,
    )
    basis = torch.randn(3, 5)
    measurements = torch.randn(1, 5, 1, 6, 6, dtype=torch.complex64)
    coefficients = torch.randn(1, 3, 6, 6, dtype=torch.complex64)
    exact = Subspace(base, basis)
    reference = exact.A_adjoint(measurements)
    forward_reference = exact.A(coefficients)
    normal_reference = exact.A_adjoint_A(coefficients)
    streamed = Subspace(base, basis)
    CudaStreaming().configure_physics(streamed)

    result = streamed.A_adjoint(measurements)
    forward_result = streamed.A(coefficients)
    normal_result = streamed.A_adjoint_A(coefficients)

    torch.testing.assert_close(result, reference)
    torch.testing.assert_close(forward_result, forward_reference)
    torch.testing.assert_close(normal_result, normal_reference)
    assert result.device.type == "cpu"
    assert forward_result.device.type == "cpu"
    assert normal_result.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_streamed_dynamic_subspace_uses_bounded_frame_operator_lru():
    class FrameIdentity:
        use_toeplitz = False

        @staticmethod
        def A(value):
            return value

        @staticmethod
        def A_adjoint(value):
            return value

        @staticmethod
        def A_adjoint_A(value):
            return value

    rebuilds = []
    trajectory = torch.zeros(5, 2, 4, 2)
    native = type("Native", (), {"shape": (6, 6), "smaps": None})()

    def rebuild(_trajectory, frame_index):
        rebuilds.append(frame_index)
        return MRIPhysics(
            FrameIdentity(),
            native_operator=native,
            kind="noncartesian2d",
            spatial_ndim=2,
            viewed_as_real=False,
        )

    base = MRIPhysics(
        FrameIdentity(),
        native_operator=native,
        kind="noncartesian2d",
        spatial_ndim=2,
        viewed_as_real=False,
        trajectory=trajectory,
        rebuild=rebuild,
    )
    basis = torch.randn(3, 5)
    coefficients = torch.randn(1, 3, 6, 6, dtype=torch.complex64)
    policy = CudaStreaming(frame_cache_size=2)

    streamed = Subspace(base, basis, streaming=policy)
    assert rebuilds == []

    result = streamed.A(coefficients)
    provider = streamed.operator.frame_physics[0].provider

    # The frame operator here is an identity stub, so a frame's image comes
    # back as it went in: five frames of one 6x6 image.
    assert result.shape == (1, 5, 6, 6)
    assert rebuilds == [0, 1, 2, 3, 4]
    assert list(provider.cache) == [3, 4]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_streamed_cartesian_physics_chunks_multislice_batch():
    generator = torch.Generator().manual_seed(72)
    batch, coils = 3, 2
    shape = (8, 10)
    mask = torch.ones(1, 1, *shape)
    maps = torch.randn(
        batch,
        coils,
        *shape,
        generator=generator,
        dtype=torch.complex64,
    )
    maps /= torch.linalg.vector_norm(maps, dim=1, keepdim=True)
    image = torch.randn(batch, *shape, generator=generator, dtype=torch.complex64)
    physics = Cartesian2D(mask, maps)

    forward_reference = physics.A(image)
    adjoint_reference = physics.A_adjoint(forward_reference)
    normal_reference = physics.A_adjoint_A(image)
    CudaStreaming(physics_batch_size=1).configure_physics(physics)

    forward = physics.A(image)
    adjoint = physics.A_adjoint(forward_reference)
    normal = physics.A_adjoint_A(image)

    torch.testing.assert_close(forward, forward_reference, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(adjoint, adjoint_reference, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(normal, normal_reference, atol=2e-5, rtol=2e-5)
    assert forward.device.type == "cpu"
    assert adjoint.device.type == "cpu"
    assert normal.device.type == "cpu"
