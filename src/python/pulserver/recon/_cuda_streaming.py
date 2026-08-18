"""Host-backed, bounded-memory CUDA execution helpers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass, replace
from importlib import import_module
from math import prod
from typing import Any


def _torch() -> Any:
    try:
        return import_module("torch")
    except ImportError as error:
        raise ImportError(
            "CUDA streaming requires Torch with CUDA support; install "
            "pulserver[cuda] on a CUDA host."
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

    Reconstruction iterates and input k-space remain on CPU. By default every
    visible GPU participates; ``cuda:N`` or ``devices=(...)`` constrains the
    set explicitly. Each device owns ``streams`` CUDA streams which alternate
    pinned-host transfers and bounded GPU work. Toeplitz transforms
    one full padded coefficient volume at a time and uses fused packed
    Hermitian CUDA multiplication. ``transfer_chunk_size`` bounds packed
    transfer staging. ``transfer_precision="auto"`` uses BF16 coefficient
    storage for real Toeplitz kernels on natively capable GPUs and retains
    complex64 for complex-Hermitian kernels. Spectra and accumulation remain
    complex64/FP32. Denoisers operate on overlapping slabs along their first
    spatial axis.
    """

    device: str = "cuda"
    devices: tuple[str, ...] | None = None
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
        if self.devices is not None:
            if not self.devices:
                raise ValueError("CudaStreaming.devices cannot be empty")
            normalized = tuple(str(torch.device(item)) for item in self.devices)
            if any(torch.device(item).type != "cuda" for item in normalized):
                raise ValueError("CudaStreaming.devices must contain CUDA devices")
            if len(set(normalized)) != len(normalized):
                raise ValueError("CudaStreaming.devices must be unique")
            object.__setattr__(self, "devices", normalized)
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
                "transfer_precision must be 'auto', 'float32', 'float16', or 'bfloat16'"
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
        """Primary configured :class:`torch.device`."""
        return self.torch_devices[0]

    @property
    def torch_devices(self) -> tuple[Any, ...]:
        """CUDA devices participating in streamed execution.

        An explicit ``devices`` tuple always wins.  Otherwise ``cuda:N`` is
        deliberately single-device, while the default unindexed ``cuda``
        discovers every visible GPU.  This makes a two-GPU recon host useful
        without changing application code and still gives callers a precise
        opt-out.
        """
        torch = _torch()
        if self.devices is not None:
            return tuple(torch.device(item) for item in self.devices)
        configured = torch.device(self.device)
        if configured.index is not None:
            return (configured,)
        count = torch.cuda.device_count()
        if count:
            return tuple(torch.device("cuda", index) for index in range(count))
        return (torch.device("cuda", torch.cuda.current_device()),)

    @property
    def device_count(self) -> int:
        """Number of GPUs selected by this policy."""
        return len(self.torch_devices)

    @property
    def execution_devices(self) -> tuple[Any, ...]:
        """One device entry per stream worker, ordered round-robin by GPU."""
        return tuple(
            device for _ in range(self.streams) for device in self.torch_devices
        )

    def for_device(self, device: Any, *, streams: int | None = None) -> CudaStreaming:
        """Return an equivalent policy constrained to one CUDA device."""
        selected = str(device)
        return replace(
            self,
            device=selected,
            devices=(selected,),
            streams=self.streams if streams is None else streams,
        )

    def ensure_available(self) -> None:
        """Raise a clear error when the configured CUDA device is unavailable."""
        torch = _torch()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA streaming requested but CUDA is unavailable")
        for device in self.torch_devices:
            torch.empty(0, device=device)

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

    def wrap_batch_denoiser(
        self,
        model: Any,
        *,
        max_batch_size: int | None = None,
    ) -> CudaBatchDenoiser:
        """Wrap an inference denoiser with exact multi-GPU context batching."""
        self.ensure_available()
        if isinstance(model, CudaBatchDenoiser):
            return model
        return CudaBatchDenoiser(
            model,
            self,
            max_batch_size=max_batch_size,
        )


class CudaBatchDenoiser:
    """Distribute independent denoiser batches over CUDA streams and devices.

    The input and result remain on pinned host memory. Each worker owns a
    model replica and CUDA stream. An out-of-memory batch is bisected until it
    fits, and the successful size is retained as the worker's limit for later
    calls. This executor is inference-only because gradients from replicas
    cannot be reduced into the source model.
    """

    def __init__(
        self,
        model: Any,
        policy: CudaStreaming,
        *,
        max_batch_size: int | None = None,
    ) -> None:
        if max_batch_size is not None and (
            not isinstance(max_batch_size, int)
            or isinstance(max_batch_size, bool)
            or max_batch_size < 1
        ):
            raise ValueError("max_batch_size must be a positive integer or None")
        self.model = model
        self.policy = policy
        self.max_batch_size = max_batch_size
        self._models: tuple[Any, ...] | None = None
        self._batch_limits: dict[int, int] = {}
        self._single_stream = False

    def __call__(
        self,
        x: Any,
        sigma: Any = None,
        **kwargs: Any,
    ) -> Any:
        return self.forward(x, sigma=sigma, **kwargs)

    def forward(
        self,
        x: Any,
        sigma: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Denoise a flattened CPU batch and restore its original ordering."""
        torch = _torch()
        if not isinstance(x, torch.Tensor):
            raise TypeError("CUDA batch denoisers require a torch.Tensor")
        if x.device.type != "cpu":
            return self.model(x, sigma=sigma, **kwargs)
        if x.ndim < 3 or x.shape[0] < 1:
            raise ValueError("CUDA batch denoisers require a non-empty image batch")
        if getattr(self.model, "training", False):
            raise RuntimeError("CUDA batch denoisers require model.eval()")
        self.policy.ensure_available()
        devices = (
            self.policy.torch_devices
            if self._single_stream
            else self.policy.execution_devices
        )
        devices = devices[: min(len(devices), x.shape[0])]
        try:
            return self._forward_on_devices(
                x,
                sigma=sigma,
                kwargs=kwargs,
                devices=devices,
            )
        except _CudaWorkerMemoryError:
            fallback_devices = self.policy.torch_devices[
                : min(len(self.policy.torch_devices), x.shape[0])
            ]
            if tuple(map(str, devices)) == tuple(map(str, fallback_devices)):
                raise
            self._release_worker_models()
            self._batch_limits.clear()
            self._single_stream = True
            return self._forward_on_devices(
                x,
                sigma=sigma,
                kwargs=kwargs,
                devices=fallback_devices,
            )

    def _forward_on_devices(
        self,
        x: Any,
        *,
        sigma: Any,
        kwargs: dict[str, Any],
        devices: tuple[Any, ...],
    ) -> Any:
        torch = _torch()
        available_workers = len(devices)
        balanced_batch = max(
            1,
            (x.shape[0] + available_workers - 1) // available_workers,
        )
        initial_batch = (
            balanced_batch
            if self.max_batch_size is None
            else min(self.max_batch_size, balanced_batch)
        )
        tasks = [
            (start, min(start + initial_batch, x.shape[0]))
            for start in range(0, x.shape[0], initial_batch)
        ]
        worker_count = min(available_workers, len(tasks))
        devices = devices[:worker_count]
        models = self._worker_models(worker_count)
        assignments = [tasks[index::worker_count] for index in range(worker_count)]

        def run_worker(slot: int) -> list[tuple[int, Any]]:
            device = devices[slot]
            torch.cuda.set_device(device)
            try:
                model = models[slot].to(device).eval()
            except torch.cuda.OutOfMemoryError as error:
                torch.cuda.empty_cache()
                raise _CudaWorkerMemoryError(
                    "a CUDA model replica does not fit; select fewer streams or "
                    "a smaller model"
                ) from error
            stream = torch.cuda.Stream(device=device)
            outputs = []
            for start, stop in assignments[slot]:
                outputs.extend(
                    self._run_batch(
                        slot,
                        model,
                        stream,
                        device,
                        x,
                        sigma,
                        kwargs,
                        start,
                        stop,
                    )
                )
            return outputs

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(run_worker, slot) for slot in range(worker_count)
            ]
            completed = [item for future in futures for item in future.result()]
        completed.sort(key=lambda item: item[0])
        return torch.cat([value for _, value in completed], dim=0)

    def _worker_models(self, worker_count: int) -> tuple[Any, ...]:
        if self._models is None or len(self._models) < worker_count:
            self._models = tuple(deepcopy(self.model) for _ in range(worker_count))
        return self._models

    def _release_worker_models(self) -> None:
        torch = _torch()
        models, self._models = self._models, None
        if models is not None:
            for model in models:
                with suppress(Exception):
                    model.to("cpu")
        for device in self.policy.torch_devices:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()

    def _run_batch(
        self,
        slot: int,
        model: Any,
        stream: Any,
        device: Any,
        x: Any,
        sigma: Any,
        kwargs: dict[str, Any],
        start: int,
        stop: int,
    ) -> list[tuple[int, Any]]:
        torch = _torch()
        limit = self._batch_limits.get(slot)
        if limit is not None and stop - start > limit:
            results = []
            for selected_start in range(start, stop, limit):
                results.extend(
                    self._run_batch(
                        slot,
                        model,
                        stream,
                        device,
                        x,
                        sigma,
                        kwargs,
                        selected_start,
                        min(selected_start + limit, stop),
                    )
                )
            return results
        device_input = None
        device_sigma = None
        device_kwargs = None
        device_output = None
        try:
            host_input = _pinned(x[start:stop], enabled=self.policy.pin_memory)
            with torch.cuda.device(device), torch.cuda.stream(stream):
                device_input = host_input.to(
                    device,
                    non_blocking=self.policy.pin_memory,
                )
                device_sigma = _condition_to_device(
                    sigma,
                    start=start,
                    stop=stop,
                    total=x.shape[0],
                    device=device,
                    pin_memory=self.policy.pin_memory,
                )
                device_kwargs = {
                    name: _condition_to_device(
                        value,
                        start=start,
                        stop=stop,
                        total=x.shape[0],
                        device=device,
                        pin_memory=self.policy.pin_memory,
                    )
                    for name, value in kwargs.items()
                }
                with torch.inference_mode():
                    device_output = model(
                        device_input,
                        sigma=device_sigma,
                        **device_kwargs,
                    )
                if not isinstance(device_output, torch.Tensor):
                    raise TypeError("CUDA batch denoisers must return a torch.Tensor")
                host_output = torch.empty_like(
                    device_output,
                    device="cpu",
                    pin_memory=self.policy.pin_memory,
                )
                host_output.copy_(
                    device_output,
                    non_blocking=self.policy.pin_memory,
                )
            stream.synchronize()
            if slot in self._batch_limits:
                self._batch_limits[slot] = max(
                    self._batch_limits[slot],
                    stop - start,
                )
            return [(start, host_output)]
        except torch.cuda.OutOfMemoryError as error:
            device_input = None
            device_sigma = None
            device_kwargs = None
            device_output = None
            with suppress(torch.cuda.OutOfMemoryError):
                stream.synchronize()
            torch.cuda.empty_cache()
            size = stop - start
            if size == 1:
                raise _CudaWorkerMemoryError(
                    "one model sample does not fit on the selected CUDA device"
                ) from error
            reduced = max(1, size // 2)
            current = self._batch_limits.get(slot)
            self._batch_limits[slot] = (
                reduced if current is None else min(current, reduced)
            )
            middle = start + reduced
            return [
                *self._run_batch(
                    slot,
                    model,
                    stream,
                    device,
                    x,
                    sigma,
                    kwargs,
                    start,
                    middle,
                ),
                *self._run_batch(
                    slot,
                    model,
                    stream,
                    device,
                    x,
                    sigma,
                    kwargs,
                    middle,
                    stop,
                ),
            ]


class _CudaWorkerMemoryError(MemoryError):
    pass


def _pinned(value: Any, *, enabled: bool) -> Any:
    torch = _torch()
    if not enabled or value.device.type != "cpu" or value.is_pinned():
        return value
    result = torch.empty_like(value, pin_memory=True)
    result.copy_(value)
    return result


def _condition_to_device(
    value: Any,
    *,
    start: int,
    stop: int,
    total: int,
    device: Any,
    pin_memory: bool,
) -> Any:
    torch = _torch()
    if not isinstance(value, torch.Tensor):
        return value
    selected = (
        value[start:stop] if value.ndim > 0 and value.shape[0] == total else value
    )
    selected = _pinned(selected, enabled=pin_memory)
    return selected.to(
        device,
        non_blocking=pin_memory and selected.device.type == "cpu",
    )


class CudaStreamedDenoiser:
    """Run a denoiser on overlapping host-backed slabs across CUDA devices.

    Cropping each denoised slab to its non-overlapping core prevents blending
    seams. Local block denoisers are exact when the halo covers their block
    support and slab boundaries align with the block grid. TV, TGV, and
    wavelet transforms are global proximals; slab execution is therefore an
    overlap approximation whose accuracy is controlled by ``denoiser_halo``.
    """

    def __init__(self, model: Any, policy: CudaStreaming) -> None:
        self.model = model
        self.policy = policy
        self._models = tuple(deepcopy(model) for _ in policy.execution_devices)

    def __call__(self, x: Any, sigma: Any, **kwargs: Any) -> Any:
        return self.forward(x, sigma, **kwargs)

    @staticmethod
    def _clear_state(model: Any) -> None:
        for name in ("x2", "u2", "r2"):
            if hasattr(model, name):
                setattr(model, name, None)
        if hasattr(model, "restart"):
            model.restart = True
        children = getattr(model, "children", None)
        if children is not None:
            for child in children():
                CudaStreamedDenoiser._clear_state(child)

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
        devices = self.policy.execution_devices
        streams = [torch.cuda.Stream(device=device) for device in devices]

        def run_slot(slot: int) -> None:
            device = devices[slot]
            torch.cuda.set_device(device)
            model = self._models[slot].to(device)
            stream = streams[slot]
            for slab_index in range(slot, len(slabs), len(streams)):
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
                        device,
                        non_blocking=self.policy.pin_memory,
                    )
                    device_sigma = (
                        sigma.to(device, non_blocking=True)
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

        with ThreadPoolExecutor(max_workers=len(streams)) as executor:
            futures = [executor.submit(run_slot, slot) for slot in range(len(streams))]
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
        return len(self.policy.execution_devices) * (
            self.policy.denoiser_slab_size + 2 * halo
        )

    def extra_repr(self) -> str:
        return (
            f"devices={tuple(map(str, self.policy.torch_devices))}, "
            f"slab={self.policy.denoiser_slab_size}, "
            f"halo={self.policy.denoiser_halo}, streams={self.policy.streams}"
        )


def tensor_nbytes(value: Any) -> int:
    """Return tensor storage bytes without allocating a temporary."""
    return int(prod(value.shape) * value.element_size())
