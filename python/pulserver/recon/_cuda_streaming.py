"""Host-backed, bounded-memory CUDA execution helpers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from importlib import import_module
from math import prod
from typing import Any


def _torch() -> Any:
    try:
        return import_module("torch")
    except ImportError as error:
        raise ImportError(
            "CUDA streaming requires Torch; install pulserver[recon-cuda]."
        ) from error


def _spatial_tuple(
    value: int | tuple[int, ...],
    dimension: int,
    *,
    name: str,
) -> tuple[int, ...]:
    result = (value,) * dimension if isinstance(value, int) else tuple(value)
    if len(result) != dimension or any(item < 0 for item in result):
        raise ValueError(f"{name} must contain {dimension} non-negative integers")
    return result


@dataclass(frozen=True)
class CudaStreaming:
    """Configuration for host-backed double-buffered CUDA execution.

    Reconstruction iterates and input k-space remain on CPU. Two CUDA streams
    alternate pinned-host transfers and bounded GPU work. Toeplitz transforms
    one full padded coefficient volume at a time and uses fused packed
    Hermitian CUDA multiplication. ``transfer_chunk_size`` bounds packed
    transfer staging. ``transfer_precision`` preserves the source dtype in
    ``"auto"`` mode; FP16/BF16 storage is explicit and still accumulates in
    FP32. Denoisers operate on overlapping slabs along their first spatial
    axis.
    """

    device: str = "cuda"
    streams: int = 2
    pin_memory: bool = True
    transfer_chunk_size: int = 1048576
    physics_batch_size: int = 1
    spectrum_residency: str = "auto"
    kernel_residency: str = "auto"
    transfer_precision: str = "auto"
    max_device_fraction: float = 0.85
    denoiser_slab_size: int = 32
    denoiser_halo: int | tuple[int, ...] = 8
    frame_cache_size: int = 2
    result_device: str = "cpu"

    def __post_init__(self) -> None:
        torch = _torch()
        device = torch.device(self.device)
        if device.type != "cuda":
            raise ValueError("CudaStreaming.device must be a CUDA device")
        if self.streams not in {1, 2}:
            raise ValueError("CudaStreaming.streams must be one or two")
        if self.transfer_chunk_size < 1:
            raise ValueError("transfer_chunk_size must be positive")
        if self.physics_batch_size < 1:
            raise ValueError("physics_batch_size must be positive")
        if self.kernel_residency not in {"auto", "host", "device"}:
            raise ValueError("kernel_residency must be 'auto', 'host', or 'device'")
        if self.transfer_precision not in {
            "auto",
            "float32",
            "float16",
            "bfloat16",
        }:
            raise ValueError(
                "transfer_precision must be 'auto', 'float32', 'float16', "
                "or 'bfloat16'"
            )
        if self.spectrum_residency not in {"auto", "host", "device"}:
            raise ValueError("spectrum_residency must be 'auto', 'host', or 'device'")
        if not 0.0 < self.max_device_fraction <= 1.0:
            raise ValueError("max_device_fraction must be in (0, 1]")
        if self.denoiser_slab_size < 1:
            raise ValueError("denoiser_slab_size must be positive")
        if self.frame_cache_size < 1:
            raise ValueError("frame_cache_size must be positive")
        if self.result_device not in {"cpu", "cuda"}:
            raise ValueError("result_device must be 'cpu' or 'cuda'")

    @property
    def torch_device(self) -> Any:
        """Configured :class:`torch.device`."""
        torch = _torch()
        device = torch.device(self.device)
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        return device

    def ensure_available(self) -> None:
        """Raise a clear error when the configured CUDA device is unavailable."""
        torch = _torch()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA streaming requested but CUDA is unavailable")
        torch.empty(0, device=self.torch_device)

    def configure_physics(self, physics: Any) -> Any:
        """Attach this policy to a Pulserver physics facade."""
        self.ensure_available()
        enable = getattr(physics, "enable_streaming", None)
        if enable is None:
            raise TypeError("physics does not support host-backed CUDA streaming")
        enable(self)
        return physics

    def wrap_denoiser(self, model: Any) -> CudaStreamedDenoiser:
        """Wrap a denoiser with overlapping double-buffered CUDA slabs."""
        self.ensure_available()
        if isinstance(model, CudaStreamedDenoiser):
            return model
        return CudaStreamedDenoiser(model, self)


class CudaStreamedDenoiser:
    """Run a denoiser on overlapping host-backed slabs using two CUDA streams.

    Cropping each denoised slab to its non-overlapping core prevents blending
    seams. Local block denoisers are exact when the halo covers their block
    support and slab boundaries align with the block grid. TV, TGV, and
    wavelet transforms are global proximals; slab execution is therefore an
    overlap approximation whose accuracy is controlled by ``denoiser_halo``.
    """

    def __init__(self, model: Any, policy: CudaStreaming) -> None:
        self.model = model
        self.policy = policy
        self._models = (deepcopy(model), deepcopy(model))

    def __call__(self, x: Any, sigma: Any, **kwargs: Any) -> Any:
        return self.forward(x, sigma, **kwargs)

    @staticmethod
    def _clear_state(model: Any) -> None:
        for name in ("x2", "u2", "r2"):
            if hasattr(model, name):
                setattr(model, name, None)
        if hasattr(model, "restart"):
            model.restart = True

    def _slabs(self, shape: tuple[int, ...]) -> list[tuple[int, int, int, int]]:
        depth = shape[0]
        halo = _spatial_tuple(
            self.policy.denoiser_halo,
            len(shape),
            name="denoiser_halo",
        )[0]
        slabs = []
        for core_start in range(0, depth, self.policy.denoiser_slab_size):
            core_stop = min(core_start + self.policy.denoiser_slab_size, depth)
            slabs.append(
                (
                    core_start,
                    core_stop,
                    max(0, core_start - halo),
                    min(depth, core_stop + halo),
                )
            )
        return slabs

    def forward(self, x: Any, sigma: Any, **kwargs: Any) -> Any:
        torch = _torch()
        if not isinstance(x, torch.Tensor):
            raise TypeError("streamed denoisers require a torch.Tensor")
        if x.device.type != "cpu":
            return self.model(x, sigma, **kwargs)
        if x.ndim < 4:
            raise ValueError("streamed denoisers require 2D or 3D image tensors")

        self.policy.ensure_available()
        spatial_shape = tuple(int(item) for item in x.shape[2:])
        slabs = self._slabs(spatial_shape)
        output = torch.empty_like(
            x,
            device="cpu",
            pin_memory=self.policy.pin_memory,
        )
        streams = [
            torch.cuda.Stream(device=self.policy.torch_device)
            for _ in range(self.policy.streams)
        ]

        def run_slot(slot: int) -> None:
            torch.cuda.set_device(self.policy.torch_device)
            model = self._models[slot].to(self.policy.torch_device)
            stream = streams[slot]
            for slab_index in range(slot, len(slabs), self.policy.streams):
                core_start, core_stop, input_start, input_stop = slabs[slab_index]
                source = x[:, :, input_start:input_stop]
                host_input = torch.empty_like(
                    source,
                    device="cpu",
                    pin_memory=self.policy.pin_memory,
                )
                host_input.copy_(source)
                host_output = torch.empty_like(
                    host_input,
                    pin_memory=self.policy.pin_memory,
                )
                with torch.cuda.stream(stream):
                    device_input = host_input.to(
                        self.policy.torch_device,
                        non_blocking=self.policy.pin_memory,
                    )
                    device_sigma = (
                        sigma.to(self.policy.torch_device, non_blocking=True)
                        if isinstance(sigma, torch.Tensor)
                        else sigma
                    )
                    device_output = model(device_input, device_sigma, **kwargs)
                    host_output.copy_(
                        device_output,
                        non_blocking=self.policy.pin_memory,
                    )
                stream.synchronize()
                local_start = core_start - input_start
                local_stop = local_start + core_stop - core_start
                output[:, :, core_start:core_stop].copy_(
                    host_output[:, :, local_start:local_stop]
                )
                self._clear_state(model)

        with ThreadPoolExecutor(max_workers=self.policy.streams) as executor:
            futures = [
                executor.submit(run_slot, slot) for slot in range(self.policy.streams)
            ]
            for future in futures:
                future.result()

        return output

    @property
    def workspace_voxels(self) -> int:
        """Maximum input voxels staged by both stream slots, excluding batch/channels."""
        halo = (
            self.policy.denoiser_halo
            if isinstance(self.policy.denoiser_halo, int)
            else self.policy.denoiser_halo[0]
        )
        return self.policy.streams * (self.policy.denoiser_slab_size + 2 * halo)

    def extra_repr(self) -> str:
        return (
            f"device={self.policy.device}, slab={self.policy.denoiser_slab_size}, "
            f"halo={self.policy.denoiser_halo}, streams={self.policy.streams}"
        )


def tensor_nbytes(value: Any) -> int:
    """Return tensor storage bytes without allocating a temporary."""
    return int(prod(value.shape) * value.element_size())
