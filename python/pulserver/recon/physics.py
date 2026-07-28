"""MRI physics factories with a uniform DeepInverse-facing interface.

The public factories own the mri-nufft/DeepInverse integration boundary.
Callers never need to construct an mri-nufft autodiff wrapper themselves.
Subspace, off-resonance, and Toeplitz behavior are composed as decorators so
that the API does not grow one class for every possible combination.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
from importlib import import_module
from math import prod
from types import MethodType, SimpleNamespace
from typing import Any

from ._toeplitz import CompactToeplitzKernel, as_torch, support_indices

__all__ = [
    "Cartesian2D",
    "MRIPhysics",
    "NonCartesian2D",
    "NonCartesian3D",
    "OffResonance",
    "Subspace",
    "Toeplitz",
    "cartesian_2d",
    "noncartesian_2d",
    "noncartesian_3d",
    "off_resonance",
    "subspace",
    "toeplitz",
]


def _require_deepinv() -> Any:
    try:
        return import_module("deepinv.physics")
    except ImportError as error:
        raise ImportError(
            "MRI physics operators require DeepInverse; install "
            "pulserver[recon-cpu] or pulserver[recon-cuda]."
        ) from error


def _require_mrinufft() -> Any:
    try:
        return import_module("mrinufft")
    except ImportError as error:
        raise ImportError(
            "Non-Cartesian MRI physics requires mri-nufft; install "
            "pulserver[recon-cpu] or pulserver[recon-cuda]."
        ) from error


def _image_as_real(value: Any) -> Any:
    """Pack complex image channels as DeepInverse real channels."""
    torch = import_module("torch")
    batch, channels, *spatial = value.shape
    value = torch.view_as_real(value).movedim(-1, 2)
    return value.reshape(batch, channels * 2, *spatial)


def _image_as_cpx(value: Any) -> Any:
    """Unpack DeepInverse real image channels to complex channels."""
    torch = import_module("torch")
    batch, channels, *spatial = value.shape
    if channels % 2:
        raise ValueError("real-view images require pairs of complex channels")
    value = value.reshape(batch, channels // 2, 2, *spatial).movedim(2, -1)
    return torch.view_as_complex(value.contiguous())


def _kspace_as_real(value: Any) -> Any:
    """Expose complex k-space through a trailing real/imaginary dimension."""
    torch = import_module("torch")
    return torch.view_as_real(value)


def _kspace_as_cpx(value: Any) -> Any:
    """Restore complex k-space from its trailing real/imaginary dimension."""
    torch = import_module("torch")
    return torch.view_as_complex(value.contiguous())


def _toeplitz_options(
    *,
    support: str = "radial",
    radius: float = 1.0,
    chunk_size: int = 65536,
    coil_batch_size: int = 1,
    cuda_mode: str = "auto",
    cuda_max_device_fraction: float = 0.85,
    cuda_transfer_precision: str = "auto",
) -> dict[str, Any]:
    if support not in {"full", "radial"}:
        raise ValueError("Toeplitz support must be 'full' or 'radial'")
    if radius <= 0.0 or radius > 1.0:
        raise ValueError("Toeplitz radius must be in the interval (0, 1]")
    if chunk_size <= 0:
        raise ValueError("Toeplitz chunk_size must be positive")
    if coil_batch_size <= 0:
        raise ValueError("Toeplitz coil_batch_size must be positive")
    if cuda_mode not in {"auto", "resident", "compact"}:
        raise ValueError("Toeplitz cuda_mode must be 'auto', 'resident', or 'compact'")
    if not 0.0 < cuda_max_device_fraction <= 1.0:
        raise ValueError("Toeplitz cuda_max_device_fraction must be in (0, 1]")
    if cuda_transfer_precision not in {
        "auto",
        "float32",
        "float16",
        "bfloat16",
    }:
        raise ValueError(
            "Toeplitz cuda_transfer_precision must be 'auto', 'float32', "
            "'float16', or 'bfloat16'"
        )
    return {
        "support": support,
        "radius": float(radius),
        "chunk_size": int(chunk_size),
        "coil_batch_size": int(coil_batch_size),
        "cuda_mode": cuda_mode,
        "cuda_max_device_fraction": float(cuda_max_device_fraction),
        "cuda_transfer_precision": cuda_transfer_precision,
    }


def _toeplitz_request(value: bool | dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    if isinstance(value, dict):
        return True, _toeplitz_options(**value)
    return bool(value), _toeplitz_options()


class MRIPhysics:
    """A small, transparent facade over a DeepInverse linear physics object.

    The facade records reconstruction-specific metadata and forwards the
    numerical contract (``A``, ``A_adjoint``, ``A_adjoint_A``, ``A_dagger``)
    to DeepInverse. ``native_operator`` is the underlying mri-nufft operator
    for non-Cartesian acquisitions and is intentionally read-only.
    """

    def __init__(
        self,
        operator: Any,
        *,
        native_operator: Any | None,
        kind: str,
        spatial_ndim: int,
        viewed_as_real: bool = True,
        modifiers: tuple[str, ...] = (),
        trajectory: Any | None = None,
        rebuild: Callable[[Any, int | None], MRIPhysics] | None = None,
        toeplitz_options: dict[str, Any] | None = None,
    ) -> None:
        self.operator = operator
        self.native_operator = native_operator
        self.kind = kind
        self.spatial_ndim = spatial_ndim
        self.viewed_as_real = viewed_as_real
        self.modifiers = modifiers
        self.trajectory = trajectory
        self._rebuild = rebuild
        self.toeplitz_options = (
            dict(toeplitz_options) if toeplitz_options is not None else None
        )
        self.streaming_policy = None
        self._streaming_methods: set[str] = set()
        self._streaming_parameters: dict[str, Any] | None = None

    @property
    def normal_mode(self) -> str:
        """Return ``"toeplitz"``, ``"exact-fft"``, or ``"exact"``."""
        if "toeplitz" in self.modifiers:
            if self.kind == "cartesian2d":
                return "exact-fft"
            if getattr(self.operator, "use_toeplitz", False):
                return "toeplitz"
        return "exact"

    def A(self, x: Any, **kwargs: Any) -> Any:
        """Apply the forward encoding operator."""
        return self._stream_call("A", self.operator.A, x, **kwargs)

    def A_adjoint(self, y: Any, **kwargs: Any) -> Any:
        """Apply the adjoint encoding operator."""
        return self._stream_call(
            "A_adjoint",
            self.operator.A_adjoint,
            y,
            **kwargs,
        )

    def A_adjoint_A(self, x: Any, **kwargs: Any) -> Any:
        """Apply the normal operator, using Toeplitz acceleration when valid."""
        return self._stream_call(
            "A_adjoint_A",
            self.operator.A_adjoint_A,
            x,
            **kwargs,
        )

    def _stream_call(
        self,
        name: str,
        method: Any,
        value: Any,
        **kwargs: Any,
    ) -> Any:
        """Stage one operator unit when the wrapped operator is not stream-aware."""
        policy = self.streaming_policy
        if policy is None or name in self._streaming_methods:
            return method(value, **kwargs)
        torch = import_module("torch")
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
            return method(value, **kwargs)
        if self._streaming_parameters is not None:
            starts = list(range(0, value.shape[0], policy.physics_batch_size))
            outputs: list[Any | None] = [None] * len(starts)
            streams = [
                torch.cuda.Stream(device=policy.torch_device)
                for _ in range(policy.streams)
            ]
            for chunk_index, start in enumerate(starts):
                stop = min(start + policy.physics_batch_size, value.shape[0])
                slot = chunk_index % policy.streams
                streams[slot].synchronize()
                host_parameters = {}
                for parameter_name, parameter in self._streaming_parameters.items():
                    selected = parameter
                    if (
                        isinstance(parameter, torch.Tensor)
                        and parameter.ndim
                        and parameter.shape[0] == value.shape[0]
                    ):
                        selected = parameter[start:stop]
                    if (
                        policy.pin_memory
                        and isinstance(selected, torch.Tensor)
                        and selected.device.type == "cpu"
                        and not selected.is_pinned()
                    ):
                        pinned_parameter = torch.empty_like(
                            selected,
                            pin_memory=True,
                        )
                        pinned_parameter.copy_(selected)
                        selected = pinned_parameter
                    host_parameters[parameter_name] = selected
                host = value[start:stop]
                if policy.pin_memory and not host.is_pinned():
                    pinned = torch.empty_like(host, pin_memory=True)
                    pinned.copy_(host)
                    host = pinned
                with torch.cuda.stream(streams[slot]):
                    device_parameters = {
                        parameter_name: (
                            parameter.to(
                                policy.torch_device,
                                non_blocking=(
                                    policy.pin_memory
                                    and isinstance(parameter, torch.Tensor)
                                    and parameter.is_pinned()
                                ),
                            )
                            if isinstance(parameter, torch.Tensor)
                            else parameter
                        )
                        for parameter_name, parameter in host_parameters.items()
                    }
                    device_value = host.to(
                        policy.torch_device,
                        non_blocking=policy.pin_memory,
                    )
                    device_result = self._cartesian_device_call(
                        name,
                        device_value,
                        device_parameters["mask"],
                        device_parameters["coil_maps"],
                        kwargs,
                    )
                    outputs[chunk_index] = torch.empty_like(
                        device_result,
                        device="cpu",
                        pin_memory=policy.pin_memory,
                    )
                    outputs[chunk_index].copy_(
                        device_result,
                        non_blocking=policy.pin_memory,
                    )
            for stream in streams:
                stream.synchronize()
            return torch.cat(
                [output for output in outputs if output is not None],
                dim=0,
            )
        return self._stage_call(method, value, policy, kwargs)

    def _cartesian_device_call(
        self,
        name: str,
        value: Any,
        mask: Any,
        coil_maps: Any,
        kwargs: dict[str, Any],
    ) -> Any:
        """Apply one bounded Cartesian SENSE batch without mutating DeepInv."""
        torch = import_module("torch")

        def image_as_complex(image: Any) -> Any:
            return torch.view_as_complex(image.movedim(1, -1).contiguous())

        def image_as_real(image: Any) -> Any:
            return torch.view_as_real(image).movedim(-1, 1)

        def forward(image: Any) -> Any:
            coil_images = coil_maps * image_as_complex(image)[:, None]
            spectrum = self.operator.fft(coil_images, dim=(-2, -1))
            return mask[:, :, None] * image_as_real(spectrum)

        def adjoint(kspace: Any) -> Any:
            masked = image_as_complex(mask[:, :, None] * kspace)
            coil_images = self.operator.ifft(masked, dim=(-2, -1))
            if kwargs.get("rss", False):
                return self.operator.rss(image_as_real(coil_images), multicoil=True)
            image = (coil_maps.conj() * coil_images).sum(dim=1)
            return image_as_real(image)

        if name == "A":
            return forward(value)
        if name == "A_adjoint":
            result = adjoint(value)
            return self.operator.crop(result, crop=kwargs.get("crop", False))
        if name == "A_adjoint_A":
            return adjoint(forward(value))
        raise ValueError(f"unsupported streamed Cartesian method {name!r}")

    @staticmethod
    def _stage_call(
        method: Any,
        value: Any,
        policy: Any,
        kwargs: dict[str, Any],
    ) -> Any:
        """Move one bounded call to CUDA and return a CPU tensor."""
        torch = import_module("torch")
        if policy.pin_memory and not value.is_pinned():
            host = torch.empty_like(value, pin_memory=True)
            host.copy_(value)
        else:
            host = value
        device_value = host.to(
            policy.torch_device,
            non_blocking=policy.pin_memory,
        )
        result = method(device_value, **kwargs)
        if isinstance(result, torch.Tensor):
            result = result.to("cpu")
        return result

    def A_vjp(self, x: Any, v: Any, **kwargs: Any) -> Any:
        """Return the vector-Jacobian product required by DeepInverse."""
        method = getattr(self.operator, "A_vjp", None)
        if method is not None:
            return method(x, v, **kwargs)
        return self.A_adjoint(v, **kwargs)

    def A_dagger(self, y: Any, **kwargs: Any) -> Any:
        """Apply the least-squares pseudo-inverse supplied by DeepInverse."""
        return self.operator.A_dagger(y, **kwargs)

    def __call__(self, x: Any, **kwargs: Any) -> Any:
        return self.A(x, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.operator, name)

    def to(self, *args: Any, **kwargs: Any) -> MRIPhysics:
        """Move the underlying Torch physics object and return ``self``."""
        self.operator.to(*args, **kwargs)
        return self

    def enable_streaming(self, policy: Any) -> MRIPhysics:
        """Keep reconstruction state on CPU and use bounded CUDA workspaces."""
        self.streaming_policy = policy
        enable = getattr(self.operator, "enable_streaming", None)
        if enable is not None:
            enable(policy)
            self._streaming_methods = set(
                getattr(self.operator, "streaming_methods", ())
            )
        else:
            self.operator.streaming_policy = policy
            self._streaming_methods = set()
        return self

    def rebuild(
        self,
        trajectory: Any,
        frame_index: int | None = None,
    ) -> MRIPhysics:
        """Rebuild a non-Cartesian factory for a frame-specific trajectory."""
        if self._rebuild is None:
            return self
        return self._rebuild(trajectory, frame_index)


class _FramePhysicsProvider:
    """Small LRU of frame-specific native operators for streamed dynamics."""

    def __init__(
        self,
        physics: MRIPhysics,
        trajectory: Any,
        policy: Any,
    ) -> None:
        self.physics = physics
        self.trajectory = trajectory
        self.policy = policy
        self.cache: OrderedDict[int, MRIPhysics] = OrderedDict()
        self.toeplitz_options = physics.toeplitz_options

    def get(self, index: int) -> MRIPhysics:
        if index in self.cache:
            result = self.cache.pop(index)
            self.cache[index] = result
            return result
        result = self.physics.rebuild(self.trajectory[index], index)
        result.enable_streaming(self.policy)
        if self.toeplitz_options is not None:
            toeplitz(result, **self.toeplitz_options)
        self.cache[index] = result
        while len(self.cache) > self.policy.frame_cache_size:
            self.cache.popitem(last=False)
        return result


class _LazyFramePhysics:
    """Frame facade resolving through a shared bounded LRU."""

    def __init__(self, provider: _FramePhysicsProvider, index: int) -> None:
        self.provider = provider
        self.index = index
        self.kind = provider.physics.kind
        self.viewed_as_real = provider.physics.viewed_as_real
        self.modifiers = provider.physics.modifiers

    @property
    def normal_mode(self) -> str:
        if "toeplitz" in self.modifiers:
            return "exact-fft" if self.kind == "cartesian2d" else "toeplitz"
        return self.provider.physics.normal_mode

    @property
    def native_operator(self) -> Any:
        return self.provider.get(self.index).native_operator

    def A(self, value: Any) -> Any:
        return self.provider.get(self.index).A(value)

    def A_adjoint(self, value: Any) -> Any:
        return self.provider.get(self.index).A_adjoint(value)

    def A_adjoint_A(self, value: Any) -> Any:
        return self.provider.get(self.index).A_adjoint_A(value)

    def enable_streaming(self, policy: Any) -> None:
        self.provider.policy = policy

    def enable_toeplitz(self, options: dict[str, Any]) -> None:
        self.provider.toeplitz_options = options
        self.modifiers = tuple(
            dict.fromkeys((*self.modifiers, "toeplitz"))
        )


def _native_linear_physics(
    native_operator: Any,
    *,
    viewed_as_real: bool,
    use_toeplitz: bool = False,
    native_toeplitz_valid: bool = True,
) -> Any:
    """Adapt an mri-nufft operator to DeepInverse without exposing glue."""
    physics_module = _require_deepinv()
    try:
        import_module("mrinufft.operators.autodiff")
    except ImportError as error:
        raise ImportError(
            "The mri-nufft DeepInverse adapter requires Torch and DeepInverse."
        ) from error

    # Prefer mri-nufft's maintained DeepInverse/autograd interface. The custom
    # adapter below is only needed for compositions (notably stacked NUFFT)
    # that correctly implement forward/adjoint operations but do not expose
    # an autograd wrapper. PICS itself uses explicit adjoints, so both paths
    # have identical reconstruction semantics.
    if getattr(native_operator, "autograd_available", False):
        inner = native_operator.make_deepinv_phy()

        class _MRIViewPhysics(physics_module.LinearPhysics):
            def __init__(self) -> None:
                super().__init__()
                self.__dict__["inner"] = inner
                self.viewed_as_real = viewed_as_real
                self.use_toeplitz = use_toeplitz and native_toeplitz_valid

            def A(self, x: Any, **kwargs: Any) -> Any:
                if self.viewed_as_real:
                    x = _image_as_cpx(x)
                result = self.inner.A(x, **kwargs)
                return _kspace_as_real(result) if self.viewed_as_real else result

            def A_adjoint(self, y: Any, **kwargs: Any) -> Any:
                if self.viewed_as_real:
                    y = _kspace_as_cpx(y)
                result = self.inner.A_adjoint(y, **kwargs)
                return _image_as_real(result) if self.viewed_as_real else result

            def A_dagger(self, y: Any, **kwargs: Any) -> Any:
                if self.viewed_as_real:
                    y = _kspace_as_cpx(y)
                result = self.inner.A_dagger(y, **kwargs)
                return _image_as_real(result) if self.viewed_as_real else result

            def A_adjoint_A(self, x: Any, **kwargs: Any) -> Any:
                del kwargs
                if not self.use_toeplitz:
                    return self.A_adjoint(self.A(x))
                if self.viewed_as_real:
                    x = _image_as_cpx(x)
                normal = native_operator.gram_op(x, toeplitz=True)
                return _image_as_real(normal) if self.viewed_as_real else normal

        return _MRIViewPhysics()

    class _MRIThinPhysics(physics_module.LinearPhysics):
        def __init__(self) -> None:
            super().__init__()
            self.__dict__["native_operator"] = native_operator
            self.viewed_as_real = viewed_as_real
            self.use_toeplitz = use_toeplitz and native_toeplitz_valid

        def A(self, x: Any, **kwargs: Any) -> Any:
            del kwargs
            if self.viewed_as_real:
                x = _image_as_cpx(x)
            result = self.native_operator.op(x)
            return _kspace_as_real(result) if self.viewed_as_real else result

        def A_adjoint(self, y: Any, **kwargs: Any) -> Any:
            del kwargs
            if self.viewed_as_real:
                y = _kspace_as_cpx(y)
            result = self.native_operator.adj_op(y)
            return _image_as_real(result) if self.viewed_as_real else result

        def A_adjoint_A(self, x: Any, **kwargs: Any) -> Any:
            del kwargs
            if not self.use_toeplitz:
                return self.A_adjoint(self.A(x))
            if self.viewed_as_real:
                x = _image_as_cpx(x)
            result = self.native_operator.gram_op(x, toeplitz=True)
            return _image_as_real(result) if self.viewed_as_real else result

    # Keep the explicit reference so static analyzers and mocked test modules
    # both validate that this is a DeepInverse physics implementation.
    assert issubclass(_MRIThinPhysics, physics_module.LinearPhysics)
    return _MRIThinPhysics()


def _base_fourier_operator(native_operator: Any) -> Any:
    """Return the undecorated Fourier operator beneath mri-nufft wrappers."""
    return getattr(native_operator, "_fourier_op", native_operator)


def _compute_toeplitz_transfer(
    native_operator: Any,
    weights: Any | None = None,
    *,
    complex_weights: bool = False,
) -> Any:
    """Compute a scalar transfer, including complex cross-weight support."""
    if weights is None:
        method = getattr(native_operator, "compute_toeplitz_kernel", None)
        if method is None:
            raise RuntimeError(
                "This mri-nufft version does not expose Toeplitz kernels. "
                "Install a version providing compute_toeplitz_kernel()."
            )
        return method()

    try:
        compute = import_module("mrinufft.operators.toeplitz").compute_toeplitz_kernel
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "This mri-nufft version does not expose weighted Toeplitz kernels."
        ) from error

    real_kernel = compute(native_operator, weights=weights.real)
    if not complex_weights:
        return real_kernel
    imaginary_kernel = compute(native_operator, weights=weights.imag)
    return real_kernel + 1j * imaginary_kernel


def _sense_maps(native_operator: Any, reference: Any) -> Any:
    """Return sensitivity maps as a Torch tensor on ``reference.device``."""
    torch = import_module("torch")
    base = _base_fourier_operator(native_operator)
    maps = getattr(base, "smaps", None)
    if maps is None:
        return torch.ones(
            (1, *base.shape),
            dtype=reference.dtype,
            device=reference.device,
        )
    maps = as_torch(maps, device=reference.device).to(reference.dtype)
    spatial_ndim = len(base.shape)
    if maps.ndim == spatial_ndim:
        return maps[None]
    if maps.ndim in {spatial_ndim + 1, spatial_ndim + 2}:
        return maps
    raise ValueError(
        "sensitivity maps must have shape (coils, *image_shape) or "
        "(batch, coils, *image_shape)"
    )


def _apply_sense_toeplitz(
    kernel: CompactToeplitzKernel,
    image: Any,
    native_operator: Any,
    *,
    right_factors: Any | None = None,
    left_factors: Any | None = None,
    coil_batch_size: int = 1,
    streaming: Any | None = None,
) -> Any:
    """Apply a compact transfer between optional spatial factor banks."""
    torch = import_module("torch")
    maps = _sense_maps(native_operator, image)
    batched_maps = maps.ndim == image.ndim
    if batched_maps:
        if maps.shape[0] == 1:
            maps = maps.expand(image.shape[0], *maps.shape[1:])
        elif maps.shape[0] != image.shape[0]:
            raise ValueError(
                "batched sensitivity maps must have one entry per image batch"
            )
        n_coils = maps.shape[1]
    else:
        n_coils = maps.shape[0]
    result_rank = 1 if left_factors is not None else kernel.rank
    result = torch.zeros(
        (image.shape[0], result_rank, *kernel.image_shape),
        dtype=image.dtype,
        device=image.device,
    )
    if right_factors is not None:
        right_factors = as_torch(right_factors, device=image.device).to(image.dtype)
        right_factors = right_factors.reshape(kernel.rank, *kernel.image_shape)
    if left_factors is not None:
        left_factors = as_torch(left_factors, device=image.device).to(image.dtype)
        left_factors = left_factors.reshape(kernel.rank, *kernel.image_shape)

    staged_coils = None
    if (
        streaming is not None
        and image.device.type == "cpu"
        and streaming.pin_memory
        and coil_batch_size > 1
    ):
        staged_coils = torch.empty(
            (
                image.shape[0],
                min(coil_batch_size, n_coils),
                image.shape[1],
                *kernel.image_shape,
            ),
            dtype=image.dtype,
            device="cpu",
            pin_memory=True,
        )

    for start in range(0, n_coils, coil_batch_size):
        if batched_maps:
            coil_maps = maps[:, start : start + coil_batch_size]
            left = image[:, None]
            right = coil_maps[:, :, None]
        else:
            coil_maps = maps[start : start + coil_batch_size]
            left = image[:, None]
            right = coil_maps[None, :, None]
        coil_count = coil_maps.shape[1] if batched_maps else coil_maps.shape[0]
        resident_sense = (
            streaming is None
            and image.device.type == "cuda"
            and coil_count == 1
            and right_factors is None
            and left_factors is None
            and kernel._select_cuda_mode(image) == "resident"
        )
        if resident_sense:
            maps_batch = (
                coil_maps[:, 0]
                if batched_maps
                else coil_maps[0][None].expand(
                    image.shape[0],
                    *kernel.image_shape,
                )
            )
            factor = maps_batch[:, None].expand_as(image)
            kernel._last_cuda_mode = "resident"
            kernel._apply_cuda_resident(
                image,
                right_factor=factor,
                left_factor=factor.conj(),
                output=result,
            )
            continue
        fused_streaming = (
            streaming is not None
            and image.device.type == "cpu"
            and coil_count == 1
        )
        if fused_streaming:
            maps_batch = (
                coil_maps[:, 0]
                if batched_maps
                else coil_maps[0][None].expand(
                    image.shape[0],
                    *kernel.image_shape,
                )
            )
            fused_right = maps_batch[:, None].expand(
                image.shape[0],
                kernel.rank,
                *kernel.image_shape,
            )
            if right_factors is not None:
                fused_right = fused_right * right_factors[None]
            fused_left = maps_batch.conj()[:, None].expand(
                image.shape[0],
                kernel.rank,
                *kernel.image_shape,
            )
            if left_factors is not None:
                fused_left = fused_left * left_factors.conj()[None]
            transformed = kernel.apply_streamed(
                image,
                streaming,
                right_factor=fused_right,
                left_factor=fused_left,
            )
        elif staged_coils is None:
            coil_images = left * right
        else:
            coil_images = staged_coils[:, :coil_count]
            torch.mul(left, right, out=coil_images)
        if not fused_streaming:
            coil_images = coil_images.flatten(0, 1)
            if right_factors is not None:
                coil_images = coil_images * right_factors[None]
            transformed = (
                kernel.apply_streamed(coil_images, streaming)
                if streaming is not None and coil_images.device.type == "cpu"
                else kernel.apply(coil_images)
            )
        if left_factors is not None:
            transformed = (
                transformed.sum(dim=1, keepdim=True)
                if fused_streaming
                else (left_factors.conj()[None] * transformed).sum(
                    dim=1,
                    keepdim=True,
                )
            )
        transformed = transformed.unflatten(0, (image.shape[0], coil_count))
        if fused_streaming:
            result += transformed.sum(dim=1)
        else:
            result += (
                (transformed * coil_maps.conj()[None, :, None]).sum(dim=1)
                if not batched_maps
                else (transformed * coil_maps.conj()[:, :, None]).sum(dim=1)
            )
    return result


def _selected_transfer(
    transfer: Any,
    indices: Any,
    *,
    streaming: Any | None,
) -> Any:
    """Select retained locations and optionally move them to host storage."""
    torch = import_module("torch")
    transfer = as_torch(transfer).flatten()
    selected = torch.index_select(
        transfer,
        0,
        indices.to(transfer.device, dtype=torch.int64),
    )
    return selected.to("cpu") if streaming is not None else selected


def _build_subspace_toeplitz(
    frame_physics: Sequence[MRIPhysics | _LazyFramePhysics],
    basis: Any,
    options: dict[str, Any],
    streaming: Any | None = None,
) -> CompactToeplitzKernel:
    """Mix scalar frame transfers directly into packed coefficient matrices."""
    torch = import_module("torch")
    basis = torch.as_tensor(basis)
    rank, _ = basis.shape
    rows, columns = torch.triu_indices(rank, rank, device=basis.device)

    lazy = any(isinstance(item, _LazyFramePhysics) for item in frame_physics)
    grouped: dict[int, tuple[Any, Any]] = {}
    if not lazy:
        for frame, item in enumerate(frame_physics):
            native = item.native_operator
            if native is None or hasattr(native, "B"):
                raise RuntimeError(
                    "A combined subspace kernel requires undecorated frame NUFFTs."
                )
            coefficients = basis[rows, frame] * basis[columns, frame].conj()
            key = id(native)
            if key in grouped:
                grouped[key] = (native, grouped[key][1] + coefficients)
            else:
                grouped[key] = (native, coefficients)
        entries = iter(grouped.values())
    else:
        # Resolve one native NUFFT at a time through the bounded frame LRU.
        # A generator is important here: a list would retain every CUDA plan.
        entries = (
            (
                item.native_operator,
                basis[rows, frame] * basis[columns, frame].conj(),
            )
            for frame, item in enumerate(frame_physics)
        )

    image_shape = None
    spatial_shape = None
    packed = None
    indices = None
    for native, coefficients in entries:
        if native is None or hasattr(native, "B"):
            raise RuntimeError(
                "A combined subspace kernel requires undecorated frame NUFFTs."
            )
        current_shape = tuple(int(size) for size in native.shape)
        if image_shape is None:
            image_shape = current_shape
            spatial_shape = tuple(2 * size for size in image_shape)
        elif current_shape != image_shape:
            raise ValueError("all subspace frames must share one image shape")
        scalar = as_torch(_compute_toeplitz_transfer(native)).flatten()
        if indices is None:
            assert spatial_shape is not None
            indices = support_indices(
                spatial_shape,
                support=options["support"],
                radius=options["radius"],
                device="cpu" if streaming is not None else scalar.device,
            )
        selected = _selected_transfer(
            scalar,
            indices,
            streaming=streaming,
        )
        coefficients = coefficients.to(selected.device)
        dtype = torch.promote_types(coefficients.dtype, selected.dtype)
        if streaming is not None:
            selected = selected.to(dtype)
            if packed is None:
                packed = torch.zeros(
                    (coefficients.numel(), selected.numel()),
                    dtype=dtype,
                    device="cpu",
                )
            for packed_index, coefficient in enumerate(coefficients):
                packed[packed_index].add_(
                    selected,
                    alpha=coefficient.item(),
                )
        else:
            contribution = coefficients[:, None].to(dtype) * selected[None].to(dtype)
            packed = contribution if packed is None else packed + contribution
    assert (
        packed is not None
        and indices is not None
        and image_shape is not None
        and spatial_shape is not None
    )

    values = (
        packed.to(basis.dtype) if basis.is_complex() else packed.real.to(basis.dtype)
    )
    return CompactToeplitzKernel(
        values,
        indices,
        spatial_shape,
        rank,
        image_shape=image_shape,
        chunk_size=options["chunk_size"],
        cuda_mode=options["cuda_mode"],
        cuda_max_device_fraction=options["cuda_max_device_fraction"],
        cuda_transfer_precision=options["cuda_transfer_precision"],
    )


def _build_cartesian_subspace_toeplitz(
    frame_physics: Sequence[MRIPhysics],
    basis: Any,
    options: dict[str, Any],
    streaming: Any | None = None,
) -> tuple[CompactToeplitzKernel, Any]:
    """Build an exact packed Cartesian subspace transfer without 2x padding."""
    torch = import_module("torch")
    basis = torch.as_tensor(basis)
    rank, n_frames = basis.shape
    operator = frame_physics[0].operator
    mask = as_torch(operator.mask)
    maps = as_torch(operator.coil_maps)
    if streaming is not None:
        mask = mask.to("cpu")
        maps = maps.to("cpu")
    image_shape = tuple(int(size) for size in maps.shape[-2:])
    # DeepInverse represents Cartesian masks as (batch, real/imag, H, W).
    # The two channels are identical; retain only one before interpreting the
    # leading dimension as shared/per-frame masks.
    if mask.ndim >= 4 and mask.shape[-3] == 2:
        mask = mask.select(-3, 0)
    masks = mask.reshape(-1, *image_shape)
    if masks.shape[0] == 1:
        masks = masks.expand(n_frames, *image_shape)
    elif masks.shape[0] != n_frames:
        raise ValueError(
            "Cartesian subspace mask must be shared or have one mask per frame"
        )
    masks = torch.fft.ifftshift(masks, dim=(-2, -1)).abs().square()
    indices = support_indices(
        image_shape,
        support=options["support"],
        radius=options["radius"],
        device=masks.device,
    )
    rows, columns = torch.triu_indices(rank, rank, device=basis.device)
    packed = torch.zeros(
        (rows.numel(), indices.numel()),
        dtype=torch.promote_types(basis.dtype, masks.dtype),
        device=masks.device,
    )
    basis = basis.to(masks.device)
    for frame in range(n_frames):
        mixing = (
            basis[rows.to(masks.device), frame]
            * basis[columns.to(masks.device), frame].conj()
        )
        sampled_mask = torch.index_select(masks[frame].flatten(), 0, indices)
        packed += mixing[:, None] * sampled_mask[None]
    packed = (
        packed.to(basis.dtype) if basis.is_complex() else packed.real.to(basis.dtype)
    )
    kernel = CompactToeplitzKernel(
        packed,
        indices,
        image_shape,
        rank,
        image_shape=image_shape,
        chunk_size=options["chunk_size"],
        cuda_mode=options["cuda_mode"],
        cuda_max_device_fraction=options["cuda_max_device_fraction"],
        cuda_transfer_precision=options["cuda_transfer_precision"],
    )
    proxy = SimpleNamespace(
        shape=image_shape,
        smaps=maps,
    )
    return kernel, proxy


def _off_resonance_scalar_transfers(
    corrected_operator: Any,
    options: dict[str, Any],
    indices: Any | None = None,
    streaming: Any | None = None,
) -> tuple[Any, Any]:
    """Return upper-triangular segment transfers at retained locations."""
    torch = import_module("torch")
    base = _base_fourier_operator(corrected_operator)
    temporal = corrected_operator.B
    rank = int(temporal.shape[1])
    image_shape = tuple(int(size) for size in base.shape)
    spatial_shape = tuple(2 * size for size in image_shape)
    rows, columns = torch.triu_indices(rank, rank)
    temporal_is_complex = as_torch(temporal).is_complex()
    density = getattr(base, "density", None)
    packed = []
    kernel_device = indices.device if indices is not None else None
    for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
        weights = temporal[:, row].conj() * temporal[:, column]
        if corrected_operator.n_shots > 1:
            try:
                weights = weights.reshape(1, -1).repeat(
                    corrected_operator.n_shots, axis=0
                )
            except TypeError:
                weights = weights.reshape(1, -1).repeat(corrected_operator.n_shots, 1)
        weights = weights.reshape(-1)
        if density is not None:
            weights = weights * density
        complex_weights = temporal_is_complex and row != column
        scalar = as_torch(
            _compute_toeplitz_transfer(
                base,
                weights,
                complex_weights=complex_weights,
            )
        ).flatten()
        if indices is None:
            kernel_device = "cpu" if streaming is not None else scalar.device
            indices = support_indices(
                spatial_shape,
                support=options["support"],
                radius=options["radius"],
                device=kernel_device,
            )
        packed.append(
            _selected_transfer(
                scalar,
                indices,
                streaming=streaming,
            )
        )
    assert indices is not None
    values = torch.stack(packed)
    temporal_dtype = as_torch(temporal).dtype
    values = (
        values.to(temporal_dtype)
        if temporal_is_complex
        else values.real.to(temporal_dtype)
    )
    return values, indices


def _build_off_resonance_toeplitz(
    corrected_operator: Any,
    options: dict[str, Any],
    streaming: Any | None = None,
) -> tuple[CompactToeplitzKernel, Any]:
    """Build packed interpolation-segment cross-transfer kernels."""
    base = _base_fourier_operator(corrected_operator)
    spatial = corrected_operator.C
    rank = int(corrected_operator.B.shape[1])
    image_shape = tuple(int(size) for size in base.shape)
    spatial_shape = tuple(2 * size for size in image_shape)
    values, indices = _off_resonance_scalar_transfers(
        corrected_operator,
        options,
        streaming=streaming,
    )
    kernel = CompactToeplitzKernel(
        values,
        indices,
        spatial_shape,
        rank,
        image_shape=image_shape,
        chunk_size=options["chunk_size"],
        cuda_mode=options["cuda_mode"],
        cuda_max_device_fraction=options["cuda_max_device_fraction"],
        cuda_transfer_precision=options["cuda_transfer_precision"],
    )
    return kernel, spatial


def _build_subspace_off_resonance_toeplitz(
    frame_physics: Sequence[MRIPhysics],
    basis: Any,
    options: dict[str, Any],
    streaming: Any | None = None,
) -> tuple[CompactToeplitzKernel, Any]:
    """Combine shared spatial off-resonance factors with a temporal subspace."""
    torch = import_module("torch")
    basis = torch.as_tensor(basis)
    coefficient_rank, n_frames = basis.shape
    if len(frame_physics) != n_frames:
        raise ValueError("basis frame count does not match frame physics")
    first = frame_physics[0].native_operator
    if first is None or not hasattr(first, "B"):
        raise RuntimeError("expected off-resonance-corrected frame operators")
    spatial_factors = first.C
    if any(
        item.native_operator is None
        or not hasattr(item.native_operator, "B")
        or item.native_operator.C is not spatial_factors
        for item in frame_physics
    ):
        raise RuntimeError(
            "combined subspace/off-resonance Toeplitz requires shared "
            "spatial interpolation factors"
        )

    segment_rank = int(first.B.shape[1])
    combined_rank = coefficient_rank * segment_rank
    rows, columns = torch.triu_indices(combined_rank, combined_rank)
    out_coefficients = rows // segment_rank
    in_coefficients = columns // segment_rank
    out_segments = rows % segment_rank
    in_segments = columns % segment_rank
    segment_rows, segment_columns = torch.triu_indices(
        segment_rank,
        segment_rank,
    )
    segment_lookup = torch.empty(
        (segment_rank, segment_rank),
        dtype=torch.int64,
    )
    packed_segment = torch.arange(segment_rows.numel())
    segment_lookup[segment_rows, segment_columns] = packed_segment
    segment_lookup[segment_columns, segment_rows] = packed_segment
    lookup = segment_lookup[out_segments, in_segments]
    conjugate = out_segments > in_segments

    image_shape = tuple(int(size) for size in first.shape)
    spatial_shape = tuple(2 * size for size in image_shape)
    packed = None
    indices = None
    for frame, item in enumerate(frame_physics):
        native = item.native_operator
        assert native is not None
        if tuple(native.shape) != image_shape:
            raise ValueError("all frames must share one image shape")
        segment_values, indices = _off_resonance_scalar_transfers(
            native,
            options,
            indices,
            streaming,
        )
        device = segment_values.device
        basis_device = basis.to(device)
        mixing = (
            basis_device[out_coefficients.to(device), frame]
            * basis_device[in_coefficients.to(device), frame].conj()
        )
        dtype = torch.promote_types(mixing.dtype, segment_values.dtype)
        if streaming is not None:
            if packed is None:
                packed = torch.zeros(
                    (mixing.numel(), segment_values.shape[1]),
                    dtype=dtype,
                    device="cpu",
                )
            for packed_index, coefficient in enumerate(mixing):
                selected = segment_values[lookup[packed_index]]
                if conjugate[packed_index]:
                    selected = selected.conj()
                packed[packed_index].add_(
                    selected.to(dtype),
                    alpha=coefficient.item(),
                )
        else:
            selected = segment_values[lookup.to(device)]
            mask = conjugate.to(device)
            selected[mask] = selected[mask].conj()
            contribution = mixing[:, None].to(dtype) * selected.to(dtype)
            packed = contribution if packed is None else packed + contribution
    assert packed is not None and indices is not None
    packed = packed.to(torch.promote_types(basis.dtype, as_torch(first.B).dtype))
    kernel = CompactToeplitzKernel(
        packed,
        indices,
        spatial_shape,
        combined_rank,
        image_shape=image_shape,
        chunk_size=options["chunk_size"],
        cuda_mode=options["cuda_mode"],
        cuda_max_device_fraction=options["cuda_max_device_fraction"],
        cuda_transfer_precision=options["cuda_transfer_precision"],
    )
    return kernel, spatial_factors


def _apply_subspace_off_resonance_toeplitz(
    kernel: CompactToeplitzKernel,
    image: Any,
    native_operator: Any,
    spatial_factors: Any,
    *,
    coefficient_rank: int,
    coil_batch_size: int,
    streaming: Any | None = None,
) -> Any:
    """Apply a combined coefficient/segment transfer with SENSE maps."""
    torch = import_module("torch")
    maps = _sense_maps(native_operator, image)
    segment_rank = kernel.rank // coefficient_rank
    spatial_factors = as_torch(
        spatial_factors,
        device=image.device,
    ).to(image.dtype)
    spatial_factors = spatial_factors.reshape(
        segment_rank,
        *kernel.image_shape,
    )
    result = torch.zeros_like(image)
    for start in range(0, maps.shape[0], coil_batch_size):
        coil_maps = maps[start : start + coil_batch_size]
        coil_images = image[:, None] * coil_maps[None, :, None]
        expanded = coil_images[:, :, :, None] * spatial_factors[None, None, None]
        expanded = expanded.flatten(0, 1).flatten(1, 2)
        transformed = (
            kernel.apply_streamed(expanded, streaming)
            if streaming is not None and expanded.device.type == "cpu"
            else kernel.apply(expanded)
        )
        transformed = transformed.unflatten(
            1,
            (coefficient_rank, segment_rank),
        )
        transformed = (transformed * spatial_factors.conj()[None, None]).sum(dim=2)
        transformed = transformed.unflatten(
            0,
            (image.shape[0], coil_maps.shape[0]),
        )
        result += (transformed * coil_maps.conj()[None, :, None]).sum(dim=1)
    return result


def cartesian_2d(
    mask: Any,
    coil_maps: Any,
    *,
    toeplitz: bool | dict[str, Any] = False,
    **kwargs: Any,
) -> MRIPhysics:
    """Create 2D Cartesian SENSE physics.

    Leading dimensions are handled by DeepInverse as batch dimensions, so
    slices, contrasts, and dynamic frames are reconstructed independently.
    Cartesian normal operations already use exact FFTs; ``toeplitz=True`` is
    accepted for API symmetry and reports ``normal_mode == "exact-fft"``.
    """
    toeplitz_enabled, options = _toeplitz_request(toeplitz)
    physics_module = _require_deepinv()
    device = getattr(coil_maps, "device", getattr(mask, "device", "cpu"))
    operator = physics_module.MultiCoilMRI(
        mask=mask,
        coil_maps=coil_maps,
        three_d=False,
        device=device,
        **kwargs,
    )
    result = MRIPhysics(
        operator,
        native_operator=None,
        kind="cartesian2d",
        spatial_ndim=2,
        modifiers=("toeplitz",) if toeplitz_enabled else (),
        toeplitz_options=options if toeplitz_enabled else None,
    )
    result._streaming_parameters = {
        "mask": getattr(operator, "mask", mask),
        "coil_maps": getattr(operator, "coil_maps", coil_maps),
    }
    return result


Cartesian2D = cartesian_2d


def _noncartesian(
    trajectory: Any,
    image_shape: tuple[int, ...],
    *,
    spatial_ndim: int,
    coil_maps: Any | None,
    density: Any | None,
    backend: str,
    n_coils: int,
    n_batchs: int,
    stacked: bool,
    z_index: Any,
    toeplitz: bool | dict[str, Any],
    viewed_as_real: bool,
    streaming: Any | None,
    operator_kwargs: dict[str, Any],
) -> MRIPhysics:
    toeplitz_enabled, toeplitz_config = _toeplitz_request(toeplitz)
    if len(image_shape) != spatial_ndim:
        raise ValueError(
            f"image_shape must have {spatial_ndim} entries, got {image_shape!r}"
        )
    trajectory_dim = getattr(trajectory, "shape", (None,))[-1]
    if trajectory_dim is not None and trajectory_dim != spatial_ndim:
        raise ValueError(
            f"trajectory must end in {spatial_ndim} coordinates, got {trajectory_dim}"
        )
    if stacked and spatial_ndim != 3:
        raise ValueError("stacked trajectories are only supported by NonCartesian3D")

    mrinufft = _require_mrinufft()
    trajectory_shape = getattr(trajectory, "shape", ())
    frame_stacked = streaming is not None and len(trajectory_shape) >= 4
    native_trajectory = trajectory[0] if frame_stacked else trajectory
    native_density = density
    density_shape = getattr(density, "shape", ())
    if (
        frame_stacked
        and density is not None
        and density_shape
        and density_shape[0] == trajectory_shape[0]
    ):
        native_density = density[0]
    common = {
        "samples": native_trajectory,
        "shape": image_shape,
        "smaps": coil_maps,
        "n_coils": n_coils,
        "n_batchs": n_batchs,
        "squeeze_dims": False,
        **operator_kwargs,
    }
    if stacked:
        native = mrinufft.get_operator("stacked")(
            backend=backend,
            z_index=z_index,
            **common,
        )
    else:
        native = mrinufft.get_operator(backend)(density=native_density, **common)

    # mri-nufft's generic Toeplitz Gram implementation is valid for the base
    # NUFFT. Stacked FFT/NUFFT composition currently has no equivalent native
    # kernel, so it retains the exact normal operation.
    native_toeplitz_valid = not stacked
    operator = _native_linear_physics(
        native,
        viewed_as_real=viewed_as_real,
        use_toeplitz=toeplitz_enabled,
        native_toeplitz_valid=native_toeplitz_valid,
    )

    def rebuild(
        new_trajectory: Any,
        frame_index: int | None = None,
    ) -> MRIPhysics:
        frame_density = density
        density_shape = getattr(density, "shape", ())
        trajectory_shape = getattr(trajectory, "shape", ())
        frame_samples = prod(getattr(new_trajectory, "shape", (0,))[:-1])
        if (
            frame_index is not None
            and density is not None
            and len(density_shape) > 1
            and trajectory_shape
            and density_shape[0] == trajectory_shape[0]
        ):
            frame_density = density[frame_index]
        elif (
            frame_index is not None
            and density is not None
            and trajectory_shape
            and prod(density_shape) == trajectory_shape[0] * frame_samples
        ):
            start = frame_index * frame_samples
            frame_density = density.reshape(-1)[start : start + frame_samples]
        return _noncartesian(
            new_trajectory,
            image_shape,
            spatial_ndim=spatial_ndim,
            coil_maps=coil_maps,
            density=frame_density,
            backend=backend,
            n_coils=n_coils,
            n_batchs=n_batchs,
            stacked=stacked,
            z_index=z_index,
            toeplitz=toeplitz,
            viewed_as_real=viewed_as_real,
            streaming=streaming,
            operator_kwargs=operator_kwargs,
        )

    result = MRIPhysics(
        operator,
        native_operator=native,
        kind=f"noncartesian{spatial_ndim}d",
        spatial_ndim=spatial_ndim,
        viewed_as_real=viewed_as_real,
        modifiers=(("stacked",) if stacked else ())
        + (("toeplitz",) if toeplitz_enabled else ()),
        trajectory=trajectory,
        rebuild=rebuild,
        toeplitz_options=toeplitz_config if toeplitz_enabled else None,
    )
    if streaming is not None:
        result.enable_streaming(streaming)
    return result


def noncartesian_2d(
    trajectory: Any,
    image_shape: tuple[int, int],
    *,
    coil_maps: Any | None = None,
    density: Any | None = None,
    backend: str = "finufft",
    n_coils: int = 1,
    n_batchs: int = 1,
    toeplitz: bool | dict[str, Any] = False,
    viewed_as_real: bool = True,
    streaming: Any | None = None,
    **kwargs: Any,
) -> MRIPhysics:
    """Create 2D non-Cartesian mri-nufft physics."""
    result = _noncartesian(
        trajectory,
        image_shape,
        spatial_ndim=2,
        coil_maps=coil_maps,
        density=density,
        backend=backend,
        n_coils=n_coils,
        n_batchs=n_batchs,
        stacked=False,
        z_index=None,
        toeplitz=toeplitz,
        viewed_as_real=viewed_as_real,
        streaming=streaming,
        operator_kwargs=kwargs,
    )
    enabled, options = _toeplitz_request(toeplitz)
    return Toeplitz(result, **options) if enabled else result


NonCartesian2D = noncartesian_2d


def noncartesian_3d(
    trajectory: Any,
    image_shape: tuple[int, int, int],
    *,
    coil_maps: Any | None = None,
    density: Any | None = None,
    backend: str = "finufft",
    n_coils: int = 1,
    n_batchs: int = 1,
    stacked: bool = False,
    z_index: Any = "auto",
    toeplitz: bool | dict[str, Any] = False,
    viewed_as_real: bool = True,
    streaming: Any | None = None,
    **kwargs: Any,
) -> MRIPhysics:
    """Create full-3D or stack-of-trajectories mri-nufft physics."""
    result = _noncartesian(
        trajectory,
        image_shape,
        spatial_ndim=3,
        coil_maps=coil_maps,
        density=density,
        backend=backend,
        n_coils=n_coils,
        n_batchs=n_batchs,
        stacked=stacked,
        z_index=z_index,
        toeplitz=toeplitz,
        viewed_as_real=viewed_as_real,
        streaming=streaming,
        operator_kwargs=kwargs,
    )
    enabled, options = _toeplitz_request(toeplitz)
    return Toeplitz(result, **options) if enabled else result


NonCartesian3D = noncartesian_3d


def _subspace_linear_physics(
    frame_physics: Sequence[MRIPhysics | _LazyFramePhysics],
    basis: Any,
    *,
    viewed_as_real: bool,
    toeplitz_config: dict[str, Any] | None,
) -> Any:
    physics_module = _require_deepinv()
    try:
        torch = import_module("torch")
    except ImportError as error:
        raise ImportError("Subspace physics requires Torch.") from error

    class _SubspaceLinearPhysics(physics_module.LinearPhysics):
        def __init__(self) -> None:
            super().__init__()
            self.__dict__["frame_physics"] = tuple(frame_physics)
            self.__dict__["basis"] = torch.as_tensor(basis)
            self.viewed_as_real = viewed_as_real
            self.use_toeplitz = bool(frame_physics) and all(
                item.normal_mode in {"toeplitz", "exact-fft"} for item in frame_physics
            )
            self._toeplitz_options = (
                dict(toeplitz_config)
                if toeplitz_config is not None
                else _toeplitz_options()
            )
            self.toeplitz_kernel = None
            if all(item.kind == "cartesian2d" for item in frame_physics):
                self._compact_toeplitz = "cartesian-subspace"
            elif any(
                isinstance(item, _LazyFramePhysics) for item in frame_physics
            ):
                self._compact_toeplitz = (
                    "subspace-off-resonance"
                    if "off_resonance" in frame_physics[0].modifiers
                    else "subspace"
                )
            elif all(
                (native := item.native_operator) is not None
                and not hasattr(native, "B")
                for item in frame_physics
            ):
                self._compact_toeplitz = "subspace"
            elif (
                frame_physics
                and all(
                    (native := item.native_operator) is not None
                    and hasattr(native, "B")
                    for item in frame_physics
                )
                and all(
                    item.native_operator.C is frame_physics[0].native_operator.C
                    for item in frame_physics
                )
            ):
                self._compact_toeplitz = "subspace-off-resonance"
            else:
                self._compact_toeplitz = None
            self._toeplitz_spatial_factors = None
            self._toeplitz_native_proxy = None
            self.streaming_policy = None
            self.streaming_methods = {"A", "A_adjoint", "A_adjoint_A"}

        def enable_toeplitz(self, options: dict[str, Any]) -> None:
            self._toeplitz_options = dict(options)
            self.toeplitz_kernel = None
            self._toeplitz_spatial_factors = None
            self._toeplitz_native_proxy = None
            for item in self.frame_physics:
                if isinstance(item, _LazyFramePhysics):
                    item.enable_toeplitz(options)
                else:
                    toeplitz(item, **options)
            self.use_toeplitz = bool(self.frame_physics) and all(
                item.normal_mode in {"toeplitz", "exact-fft"}
                for item in self.frame_physics
            )

        def enable_streaming(self, policy: Any) -> None:
            self.streaming_policy = policy
            unique_frames = {id(frame): frame for frame in self.frame_physics}.values()
            for frame in unique_frames:
                frame.enable_streaming(policy)

        @staticmethod
        def _image_as_cpx(x: Any) -> Any:
            batch, channels, *spatial = x.shape
            if channels % 2:
                raise ValueError(
                    "real-view subspace images need 2 channels per coefficient"
                )
            x = x.reshape(batch, channels // 2, 2, *spatial).movedim(2, -1)
            return torch.view_as_complex(x.contiguous())

        @staticmethod
        def _image_as_real(x: Any) -> Any:
            batch, channels, *spatial = x.shape
            x = torch.view_as_real(x).movedim(-1, 2)
            return x.reshape(batch, channels * 2, *spatial)

        def _expand(self, coefficients: Any) -> Any:
            basis_t = self.basis.to(
                device=coefficients.device, dtype=coefficients.dtype
            )
            return torch.einsum("kt,bk...->bt...", basis_t.conj(), coefficients)

        def _project(self, frames: Any) -> Any:
            basis_t = self.basis.to(device=frames.device, dtype=frames.dtype)
            return torch.einsum("kt,bt...->bk...", basis_t, frames)

        def A(self, x: Any, **kwargs: Any) -> Any:
            del kwargs
            coefficients = self._image_as_cpx(x) if self.viewed_as_real else x
            if (
                self.streaming_policy is not None
                and isinstance(coefficients, torch.Tensor)
                and coefficients.device.type == "cpu"
            ):
                basis_cpu = self.basis.to(
                    device="cpu",
                    dtype=coefficients.dtype,
                )
                measurements = None
                for index, frame_physics_item in enumerate(self.frame_physics):
                    frame = (
                        basis_cpu[:, index]
                        .conj()
                        .reshape(1, -1, *([1] * (coefficients.ndim - 2)))
                        * coefficients
                    ).sum(dim=1, keepdim=True)
                    if frame_physics_item.viewed_as_real:
                        frame = self._image_as_real(frame)
                    measurement = frame_physics_item.A(frame)
                    if measurements is None:
                        measurements = torch.empty(
                            (
                                measurement.shape[0],
                                len(self.frame_physics),
                                *measurement.shape[1:],
                            ),
                            dtype=measurement.dtype,
                            device="cpu",
                            pin_memory=self.streaming_policy.pin_memory,
                        )
                    measurements[:, index].copy_(measurement)
                assert measurements is not None
                return measurements
            frames = self._expand(coefficients)
            measurements = []
            for index, physics in enumerate(self.frame_physics):
                frame = frames[:, index : index + 1]
                if physics.viewed_as_real:
                    frame = self._image_as_real(frame)
                measurements.append(physics.A(frame))
            return torch.stack(measurements, dim=1)

        def A_adjoint(self, y: Any, **kwargs: Any) -> Any:
            del kwargs
            if (
                self.streaming_policy is not None
                and isinstance(y, torch.Tensor)
                and y.device.type == "cpu"
            ):
                policy = self.streaming_policy
                streams = [
                    torch.cuda.Stream(device=policy.torch_device)
                    for _ in range(policy.streams)
                ]
                staged: list[Any | None] = [None] * policy.streams
                events: list[Any | None] = [None] * policy.streams

                def prefetch(frame_index: int) -> None:
                    slot = frame_index % policy.streams
                    streams[slot].synchronize()
                    source = y[:, frame_index]
                    if policy.pin_memory and not source.is_pinned():
                        host = torch.empty_like(source, pin_memory=True)
                        host.copy_(source)
                    else:
                        host = source
                    with torch.cuda.stream(streams[slot]):
                        staged[slot] = host.to(
                            policy.torch_device,
                            non_blocking=policy.pin_memory,
                        )
                        events[slot] = torch.cuda.Event()
                        events[slot].record(streams[slot])

                prefetch(0)
                coefficients = None
                basis_cpu = self.basis.to("cpu")
                for index, frame_physics_item in enumerate(self.frame_physics):
                    if index + 1 < len(self.frame_physics):
                        prefetch(index + 1)
                    slot = index % policy.streams
                    events[slot].synchronize()
                    frame = frame_physics_item.A_adjoint(staged[slot])
                    if frame_physics_item.viewed_as_real:
                        frame = self._image_as_cpx(frame)
                    frame = frame.to("cpu")
                    if coefficients is None:
                        coefficients = torch.zeros(
                            (
                                frame.shape[0],
                                self.basis.shape[0],
                                *frame.shape[2:],
                            ),
                            dtype=frame.dtype,
                            device="cpu",
                            pin_memory=policy.pin_memory,
                        )
                    coefficients += (
                        basis_cpu[:, index]
                        .to(frame.dtype)
                        .reshape(1, -1, *([1] * (frame.ndim - 2)))
                        * frame
                    )
                assert coefficients is not None
                return (
                    self._image_as_real(coefficients)
                    if self.viewed_as_real
                    else coefficients
                )
            frames = []
            for index, physics in enumerate(self.frame_physics):
                frame = physics.A_adjoint(y[:, index])
                if physics.viewed_as_real:
                    frame = self._image_as_cpx(frame)
                frames.append(frame)
            coefficients = self._project(torch.cat(frames, dim=1))
            return (
                self._image_as_real(coefficients)
                if self.viewed_as_real
                else coefficients
            )

        def A_adjoint_A(self, x: Any, **kwargs: Any) -> Any:
            del kwargs
            coefficients = self._image_as_cpx(x) if self.viewed_as_real else x
            if self.use_toeplitz and self._compact_toeplitz is not None:
                if self.toeplitz_kernel is None:
                    if self._compact_toeplitz == "cartesian-subspace":
                        (
                            self.toeplitz_kernel,
                            self._toeplitz_native_proxy,
                        ) = _build_cartesian_subspace_toeplitz(
                            self.frame_physics,
                            self.basis,
                            self._toeplitz_options,
                            self.streaming_policy,
                        )
                    elif self._compact_toeplitz == "subspace":
                        self.toeplitz_kernel = _build_subspace_toeplitz(
                            self.frame_physics,
                            self.basis,
                            self._toeplitz_options,
                            self.streaming_policy,
                        )
                    else:
                        (
                            self.toeplitz_kernel,
                            self._toeplitz_spatial_factors,
                        ) = _build_subspace_off_resonance_toeplitz(
                            self.frame_physics,
                            self.basis,
                            self._toeplitz_options,
                            self.streaming_policy,
                        )
                if self._compact_toeplitz in {
                    "cartesian-subspace",
                    "subspace",
                }:
                    selected_native = (
                        self._toeplitz_native_proxy
                        if self._compact_toeplitz == "cartesian-subspace"
                        else self.frame_physics[0].native_operator
                    )
                    result = _apply_sense_toeplitz(
                        self.toeplitz_kernel,
                        coefficients,
                        selected_native,
                        coil_batch_size=self._toeplitz_options["coil_batch_size"],
                        streaming=self.streaming_policy,
                    )
                else:
                    result = _apply_subspace_off_resonance_toeplitz(
                        self.toeplitz_kernel,
                        coefficients,
                        self.frame_physics[0].native_operator,
                        self._toeplitz_spatial_factors,
                        coefficient_rank=self.basis.shape[0],
                        coil_batch_size=self._toeplitz_options["coil_batch_size"],
                        streaming=self.streaming_policy,
                    )
                return self._image_as_real(result) if self.viewed_as_real else result
            if (
                self.streaming_policy is not None
                and isinstance(coefficients, torch.Tensor)
                and coefficients.device.type == "cpu"
            ):
                basis_cpu = self.basis.to(
                    device="cpu",
                    dtype=coefficients.dtype,
                )
                result = torch.zeros_like(
                    coefficients,
                    device="cpu",
                    pin_memory=self.streaming_policy.pin_memory,
                )
                for index, frame_physics_item in enumerate(self.frame_physics):
                    frame = (
                        basis_cpu[:, index]
                        .conj()
                        .reshape(1, -1, *([1] * (coefficients.ndim - 2)))
                        * coefficients
                    ).sum(dim=1, keepdim=True)
                    if frame_physics_item.viewed_as_real:
                        frame = self._image_as_real(frame)
                    normal = frame_physics_item.A_adjoint_A(frame)
                    if frame_physics_item.viewed_as_real:
                        normal = self._image_as_cpx(normal)
                    result += (
                        basis_cpu[:, index]
                        .reshape(1, -1, *([1] * (normal.ndim - 2)))
                        * normal.to("cpu")
                    )
                return self._image_as_real(result) if self.viewed_as_real else result
            frames = self._expand(coefficients)
            normal_frames = []
            for index, physics in enumerate(self.frame_physics):
                frame = frames[:, index : index + 1]
                if physics.viewed_as_real:
                    frame = self._image_as_real(frame)
                normal = physics.A_adjoint_A(frame)
                if physics.viewed_as_real:
                    normal = self._image_as_cpx(normal)
                normal_frames.append(normal)
            result = self._project(torch.cat(normal_frames, dim=1))
            return self._image_as_real(result) if self.viewed_as_real else result

    return _SubspaceLinearPhysics()


def subspace(
    physics: MRIPhysics,
    basis: Any,
    *,
    streaming: Any | None = None,
) -> MRIPhysics:
    """Decorate Cartesian or non-Cartesian physics with a low-rank subspace.

    ``basis`` has shape ``(rank, frames)``. Coefficient images use shape
    ``(batch, 2 * rank, *spatial)`` in the default real view, and measurements
    gain a frame dimension at axis 1. If a non-Cartesian trajectory has a
    leading frame dimension, one mri-nufft operator is built per frame.
    """
    if "subspace" in physics.modifiers:
        raise ValueError("physics already has a subspace decorator")
    shape = getattr(basis, "shape", ())
    if len(shape) != 2:
        raise ValueError("basis must have shape (rank, frames)")
    n_frames = int(shape[1])

    if streaming is None:
        streaming = physics.streaming_policy

    frame_physics: list[MRIPhysics | _LazyFramePhysics]
    trajectory = physics.trajectory
    trajectory_shape = getattr(trajectory, "shape", ())
    if (
        trajectory is not None
        and len(trajectory_shape) >= 3
        and trajectory_shape[0] == n_frames
    ):
        if streaming is None:
            frame_physics = [
                physics.rebuild(trajectory[index], index) for index in range(n_frames)
            ]
        else:
            provider = _FramePhysicsProvider(physics, trajectory, streaming)
            frame_physics = [
                _LazyFramePhysics(provider, index) for index in range(n_frames)
            ]
    else:
        frame_physics = [physics] * n_frames

    operator = _subspace_linear_physics(
        frame_physics,
        basis,
        viewed_as_real=physics.viewed_as_real,
        toeplitz_config=physics.toeplitz_options,
    )
    result = MRIPhysics(
        operator,
        native_operator=None,
        kind=physics.kind,
        spatial_ndim=physics.spatial_ndim,
        viewed_as_real=physics.viewed_as_real,
        modifiers=(*physics.modifiers, "subspace"),
        trajectory=trajectory,
        toeplitz_options=physics.toeplitz_options,
    )
    if streaming is not None:
        result.enable_streaming(streaming)
    return result


Subspace = subspace


def _configure_off_resonance_toeplitz(
    operator: Any,
    corrected_operator: Any,
    *,
    enabled: bool,
    options: dict[str, Any],
) -> Any:
    """Install a lazy segmented Toeplitz normal on a DeepInverse adapter."""
    operator.use_toeplitz = enabled
    operator.toeplitz_kernel = None
    operator._toeplitz_options = dict(options)
    operator.streaming_policy = None
    operator.streaming_methods = {"A_adjoint_A"}

    def enable_toeplitz(self: Any, new_options: dict[str, Any]) -> None:
        self.use_toeplitz = True
        self._toeplitz_options = dict(new_options)
        self.toeplitz_kernel = None

    def enable_streaming(self: Any, policy: Any) -> None:
        self.streaming_policy = policy

    def segmented_normal(self: Any, x: Any, **kwargs: Any) -> Any:
        del kwargs
        if not self.use_toeplitz:
            return self.A_adjoint(self.A(x))
        if self.viewed_as_real:
            x = _image_as_cpx(x)
        if self.toeplitz_kernel is None:
            self.toeplitz_kernel, spatial_factors = _build_off_resonance_toeplitz(
                corrected_operator,
                self._toeplitz_options,
                self.streaming_policy,
            )
            self._toeplitz_spatial_factors = spatial_factors
        result = _apply_sense_toeplitz(
            self.toeplitz_kernel,
            x,
            corrected_operator,
            right_factors=self._toeplitz_spatial_factors,
            left_factors=self._toeplitz_spatial_factors,
            coil_batch_size=self._toeplitz_options["coil_batch_size"],
            streaming=self.streaming_policy,
        )
        return _image_as_real(result) if self.viewed_as_real else result

    operator.enable_toeplitz = MethodType(enable_toeplitz, operator)
    operator.enable_streaming = MethodType(enable_streaming, operator)
    operator.A_adjoint_A = MethodType(segmented_normal, operator)
    return operator


def off_resonance(
    physics: MRIPhysics,
    field_map: Any,
    readout_time: Any,
    *,
    r2star_map: Any | None = None,
    mask: Any | None = None,
    interpolator: str | dict[str, Any] | tuple[Any, Any] = "svd",
) -> MRIPhysics:
    """Decorate non-Cartesian physics with mri-nufft off-resonance correction."""
    if physics.native_operator is None:
        raise TypeError("OffResonance requires base non-Cartesian physics")
    if "subspace" in physics.modifiers:
        raise ValueError(
            "Apply OffResonance before Subspace so field correction occurs "
            "in every acquired frame."
        )
    if "off_resonance" in physics.modifiers:
        raise ValueError("physics already has an off-resonance decorator")
    try:
        corrected_class = import_module("mrinufft.operators").MRIFourierCorrected
    except ImportError as error:
        raise ImportError("Off-resonance physics requires mri-nufft.") from error

    corrected_interpolator = interpolator
    corrected_readout_time = readout_time
    trajectory_shape = getattr(physics.trajectory, "shape", ())
    time_shape = getattr(readout_time, "shape", ())
    frame_samples = (
        prod(trajectory_shape[1:-1]) if len(trajectory_shape) >= 4 else 0
    )
    dynamic_readout = bool(
        frame_samples
        and time_shape
        and prod(time_shape) != frame_samples
        and (
            time_shape[0] == trajectory_shape[0]
            or prod(time_shape) == trajectory_shape[0] * frame_samples
        )
    )
    if (
        physics.streaming_policy is not None
        and dynamic_readout
    ):
        corrected_readout_time = (
            readout_time[0]
            if time_shape[0] == trajectory_shape[0]
            else readout_time.reshape(-1)[:frame_samples]
        )
    if isinstance(interpolator, tuple):
        # mri-nufft's array-interface decorator currently turns nested tuples
        # into lists before MRIFourierCorrected can recognize its documented
        # ``(B, C)`` input. A callable preserves the supplied arrays and also
        # lets frame rebuilds reuse one spatial factor bank by identity.
        temporal_factors, spatial_factors = interpolator
        temporal_shape = getattr(temporal_factors, "shape", ())
        if (
            physics.streaming_policy is not None
            and len(trajectory_shape) >= 4
            and temporal_shape
            and temporal_shape[0] == trajectory_shape[0]
        ):
            temporal_factors = temporal_factors[0]
        elif (
            physics.streaming_policy is not None
            and frame_samples
            and temporal_shape
            and temporal_shape[0] == trajectory_shape[0] * frame_samples
        ):
            temporal_factors = temporal_factors[:frame_samples]

        def supplied_interpolator(**_kwargs: Any) -> tuple[Any, Any, None]:
            return temporal_factors, spatial_factors, None

        corrected_interpolator = supplied_interpolator

    native = corrected_class(
        physics.native_operator,
        b0_map=field_map,
        readout_time=corrected_readout_time,
        r2star_map=r2star_map,
        mask=mask,
        interpolator=corrected_interpolator,
    )
    toeplitz_enabled = "toeplitz" in physics.modifiers
    options = physics.toeplitz_options or _toeplitz_options()
    operator = _native_linear_physics(
        native,
        viewed_as_real=physics.viewed_as_real,
        use_toeplitz=False,
        native_toeplitz_valid=False,
    )
    operator = _configure_off_resonance_toeplitz(
        operator,
        native,
        enabled=toeplitz_enabled,
        options=options,
    )

    def rebuild(
        new_trajectory: Any,
        frame_index: int | None = None,
    ) -> MRIPhysics:
        frame_readout_time = readout_time
        frame_interpolator = interpolator
        time_shape = getattr(readout_time, "shape", ())
        trajectory_shape = getattr(physics.trajectory, "shape", ())
        frame_samples = prod(getattr(new_trajectory, "shape", (0,))[:-1])
        if (
            frame_index is not None
            and len(time_shape) > 1
            and trajectory_shape
            and time_shape[0] == trajectory_shape[0]
        ):
            frame_readout_time = readout_time[frame_index]
        elif (
            frame_index is not None
            and time_shape
            and trajectory_shape
            and prod(time_shape) == trajectory_shape[0] * frame_samples
        ):
            start = frame_index * frame_samples
            frame_readout_time = readout_time.reshape(-1)[start : start + frame_samples]
        if frame_index is not None and not dynamic_readout:
            temporal_factors = native.B
            if temporal_factors.shape[0] == frame_samples:
                frame_interpolator = (temporal_factors, native.C)
            elif (
                trajectory_shape
                and temporal_factors.shape[0] == trajectory_shape[0] * frame_samples
            ):
                start = frame_index * frame_samples
                frame_interpolator = (
                    temporal_factors[start : start + frame_samples],
                    native.C,
                )
        return off_resonance(
            physics.rebuild(new_trajectory, frame_index),
            field_map,
            frame_readout_time,
            r2star_map=r2star_map,
            mask=mask,
            interpolator=frame_interpolator,
        )

    result = MRIPhysics(
        operator,
        native_operator=native,
        kind=physics.kind,
        spatial_ndim=physics.spatial_ndim,
        viewed_as_real=physics.viewed_as_real,
        modifiers=(*physics.modifiers, "off_resonance"),
        trajectory=physics.trajectory,
        rebuild=rebuild,
        toeplitz_options=options if toeplitz_enabled else None,
    )
    if physics.streaming_policy is not None:
        result.enable_streaming(physics.streaming_policy)
    return result


OffResonance = off_resonance


def toeplitz(
    physics: MRIPhysics,
    *,
    support: str = "radial",
    radius: float = 1.0,
    chunk_size: int = 65536,
    coil_batch_size: int = 1,
    cuda_mode: str = "auto",
    cuda_max_device_fraction: float = 0.85,
    cuda_transfer_precision: str = "auto",
) -> MRIPhysics:
    """Enable a Toeplitz normal operator wherever the backend supports it.

    Subspace and off-resonance decorators use a Torch-native matrix-valued
    transfer kernel. Its Hermitian upper triangle is packed, real bases retain
    real storage, and only ``support`` locations are stored. ``support="full"``
    preserves the complete embedding. ``support="radial"`` keeps the centered
    circle/sphere with normalized ``radius`` and therefore assumes the same
    radial/spherical filtering in the reconstruction model.
    """
    options = _toeplitz_options(
        support=support,
        radius=radius,
        chunk_size=chunk_size,
        coil_batch_size=coil_batch_size,
        cuda_mode=cuda_mode,
        cuda_max_device_fraction=cuda_max_device_fraction,
        cuda_transfer_precision=cuda_transfer_precision,
    )
    if "toeplitz" not in physics.modifiers:
        physics.modifiers = (*physics.modifiers, "toeplitz")
    physics.toeplitz_options = options
    enable = getattr(physics.operator, "enable_toeplitz", None)
    if enable is not None:
        enable(options)
    elif physics.native_operator is not None and "stacked" not in physics.modifiers:
        physics.operator.use_toeplitz = True
    return physics


Toeplitz = toeplitz
