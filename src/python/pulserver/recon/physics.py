"""MRI physics classes with a uniform DeepInverse-facing interface.

The public classes own the mri-nufft/DeepInverse integration boundary.
Callers never need to construct an mri-nufft autodiff wrapper themselves.
Subspace, off-resonance, and Toeplitz behavior are composed as decorators so
that the API does not grow one class for every possible combination.
"""

from __future__ import annotations

__all__ = [
    "SMS",
    "Cartesian2D",
    "Cartesian3D",
    "MRIPhysics",
    "NonCartesian2D",
    "NonCartesian3D",
    "OffResonance",
    "Subspace",
    "Toeplitz",
    "WaveEncoding",
    "WaveShuffling",
    "available_nufft_backends",
]

from collections import OrderedDict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import cache, wraps
from contextlib import ExitStack, contextmanager, suppress
from importlib import import_module
from itertools import chain
from math import prod, sqrt
from types import MethodType, SimpleNamespace
from typing import Any

import deepinv

from ._toeplitz import (
    CompactToeplitzKernel,
    _device_is_full,
    as_torch,
    occupancy_indices,
    support_indices,
)
from .execution import _resolve_device
from ._views import image_as_cpx as _image_as_cpx
from ._views import image_as_real as _image_as_real
from ._views import kspace_as_cpx as _kspace_as_cpx
from ._views import kspace_as_real as _kspace_as_real
from ._stacked import _StackedNUFFTLinearPhysics
from ._wave import _WaveLinearPhysics
from ._sms import _SMSLinearPhysics


def _require_deepinv() -> Any:
    try:
        return import_module("deepinv.physics")
    except ImportError as error:
        raise ImportError(
            "MRI physics operators require DeepInverse, which ships with "
            "pulserver; reinstall the package to restore it."
        ) from error


def _require_mrinufft() -> Any:
    try:
        module = import_module("mrinufft")
    except ImportError as error:
        raise ImportError(
            "Non-Cartesian MRI physics requires mri-nufft, which ships "
            "with pulserver; reinstall the package to restore it."
        ) from error
    from ._torch_cufinufft import register_torch_cufinufft

    register_torch_cufinufft()
    return module


def available_nufft_backends() -> list[str]:
    """Return available MRI-NUFFT backends, including Pulserver's Torch CUDA adapter.

    Examples
    --------
    Which non-Cartesian backends this installation can actually use -- the CUDA
    ones appear only where a card and its libraries are present.

    >>> import pulserver.recon as recon
    >>> "finufft" in recon.available_nufft_backends()
    True
    """
    try:
        _require_mrinufft()
    except ImportError:
        return []
    list_backends = import_module("mrinufft.operators").list_backends
    public_names = {
        "cufinufft" if name == "cufinufft-torch" else name
        for name in list_backends(available_only=True)
    }
    return sorted(public_names)


def _resolve_nufft_backend(
    backend: str,
) -> str:
    """Select FINUFFT on CPU and the private Torch CUFINUFFT adapter on CUDA."""
    selected = backend.lower()
    if selected == "cufinufft":
        return "cufinufft-torch"
    if selected != "auto":
        return selected
    try:
        torch = import_module("torch")
    except ImportError:
        return "finufft"
    return "cufinufft-torch" if torch.cuda.is_available() else "finufft"


def _cartesian_image_as_real(value: Any) -> Any:
    """Pack a complex Cartesian image into DeepInverse's channel-one real layout.

    The Cartesian operators are DeepInverse ``MultiCoilMRI`` objects, which
    carry the real and imaginary parts of an image in a dedicated axis at
    position one -- ``(batch, 2, ...)`` -- rather than interleaved into the
    channel axis the way :func:`_image_as_real` does for the mri-nufft path.
    This is the single conversion the complex-native Cartesian boundary uses.
    """
    torch = import_module("torch")
    return torch.view_as_real(value).movedim(-1, 1)


def _cartesian_image_as_cpx(value: Any) -> Any:
    """Restore a complex Cartesian image from the channel-one real layout."""
    torch = import_module("torch")
    return torch.view_as_complex(value.movedim(1, -1).contiguous())


def _operator_device(physics: Any) -> Any:
    """Where a physics object computes, defaulting to the CPU.

    An operator carries tensors of several kinds -- a mask, sensitivities, a
    trajectory, a stack of per-slice operators -- and placing it moves the ones
    the arithmetic runs on while small host-side ones stay behind. So any
    tensor on an accelerator is the answer, and only an operator with none of
    them computes on the host.
    """
    torch = import_module("torch")
    buffers = getattr(physics, "buffers", None)
    parameters = getattr(physics, "parameters", None)
    if callable(buffers) and callable(parameters):
        for tensor in chain(buffers(), parameters()):
            if tensor.device.type != "cpu":
                return tensor.device
    return torch.device("cpu")


def _mirror_array_namespace(method: Callable) -> Callable:
    """Let a physics method accept NumPy and answer in the caller's namespace.

    Torch tensors pass straight through, so the operator's internal recursion
    is untouched and there is no per-call overhead once inside the stack. A
    NumPy array is moved onto the operator's device in its complex or real
    working dtype -- a no-op, and hence zero-copy, when it already is that
    dtype on the host -- and the Torch result is handed back as NumPy. This is
    what lets a reconstruction pass raw arrays straight to ``A``/``A_adjoint``
    without shuttling them across the boundary by hand.
    """

    @wraps(method)
    def wrapper(self: Any, value: Any, *args: Any, **kwargs: Any) -> Any:
        torch = import_module("torch")
        if isinstance(value, torch.Tensor):
            return method(self, value, *args, **kwargs)
        numpy = import_module("numpy")
        if not isinstance(value, numpy.ndarray):
            return method(self, value, *args, **kwargs)
        tensor = torch.as_tensor(value)
        if tensor.is_complex():
            tensor = tensor.to(torch.complex64)
        elif tensor.dtype.is_floating_point:
            tensor = tensor.to(torch.float32)
        tensor = tensor.to(_operator_device(self))
        result = method(self, tensor, *args, **kwargs)
        if isinstance(result, torch.Tensor):
            return result.detach().to("cpu").numpy()
        return result

    return wrapper


def _measurement_to_trailing(value: Any) -> Any:
    """Move a measurement's real/imaginary axis from channel one to the end.

    Pulserver carries the real and imaginary parts of a *measurement* in a
    trailing axis, ``(batch, coils, ..., 2)``; DeepInverse carries them in
    channel position one, ``(batch, 2, coils, ...)``. Images use the packed
    channel form on both sides, so only measurements cross this boundary.

    Parameters
    ----------
    value
        Measurement in the DeepInverse channel-first layout.

    Returns
    -------
    torch.Tensor
        The same measurement in Pulserver's trailing layout.
    """
    return value.movedim(1, -1)


def _measurement_to_channels(value: Any) -> Any:
    """Move a measurement's real/imaginary axis from the end to channel one.

    The inverse of :func:`_measurement_to_trailing`.
    """
    return value.movedim(-1, 1)


class _TrailingRealView(deepinv.physics.LinearPhysics):
    """Present a DeepInverse MRI operator in Pulserver's measurement layout.

    Wraps the operator rather than converting at each call site so that every
    physics object in the package answers with the same measurement shape,
    whatever library implements it underneath. It stays a Torch module so that
    the wrapped operator keeps taking part in ``.to()`` and in the attribute
    forwarding :class:`MRIPhysics` performs.
    """

    def __init__(self, operator: Any) -> None:
        super().__init__()
        self.operator = operator

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("operator"), name)

    def A(self, x: Any, **kwargs: Any) -> Any:
        """Encode an image, answering in the trailing layout."""
        return _measurement_to_trailing(self.operator.A(x, **kwargs))

    def A_adjoint(self, y: Any, **kwargs: Any) -> Any:
        """Decode a measurement given in the trailing layout."""
        return self.operator.A_adjoint(_measurement_to_channels(y), **kwargs)

    def A_adjoint_A(self, x: Any, **kwargs: Any) -> Any:
        """Apply the normal operator, which never leaves image space."""
        return self.operator.A_adjoint_A(x, **kwargs)


class _CartesianComplexView(deepinv.physics.LinearPhysics):
    """Present a trailing-real Cartesian operator as complex-native.

    Wraps a :class:`_TrailingRealView` so images and measurements cross the
    boundary as native complex tensors -- image ``(batch, [coils,] *spatial)``,
    measurement ``(batch, coils, *kspace)`` -- packing to the two-channel real
    layout DeepInverse's ``MultiCoilMRI`` works in only for the duration of one
    call. It is the complex-facing dual of :class:`_TrailingRealView`, and the
    only place the Cartesian real/complex conversion happens once the whole
    stack is complex by default.
    """

    def __init__(self, operator: Any) -> None:
        super().__init__()
        self.operator = operator

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("operator"), name)

    def A(self, x: Any, **kwargs: Any) -> Any:
        """Encode a complex image, answering with complex k-space."""
        real = self.operator.A(_cartesian_image_as_real(x), **kwargs)
        return _kspace_as_cpx(real)

    def A_adjoint(self, y: Any, **kwargs: Any) -> Any:
        """Decode complex k-space, answering with a complex image."""
        image = self.operator.A_adjoint(_kspace_as_real(y), **kwargs)
        return _cartesian_image_as_cpx(image)

    def A_adjoint_A(self, x: Any, **kwargs: Any) -> Any:
        """Apply the exact FFT normal operator, staying in complex image space."""
        result = self.operator.A_adjoint_A(_cartesian_image_as_real(x), **kwargs)
        return _cartesian_image_as_cpx(result)


def _toeplitz_options(
    *,
    compress: bool = True,
    chunk_size: int = 65536,
    coil_batch_size: int = 1,
    cuda_mode: str = "auto",
    cuda_max_device_fraction: float = 0.85,
    cuda_transfer_precision: str = "auto",
) -> dict[str, Any]:
    """Validate how a Toeplitz kernel is applied.

    Nothing here says how the kernel is built. It is gridded onto the doubled
    grid the way BART and MRISubspaceRecon.jl build theirs, and that is not a
    choice. What is left is what it is stored and executed on: whether the
    locations the trajectory never reached are dropped, how much is unpacked
    at a time, how many coils share a pass, and what a CUDA device holds.

    ``compress`` is BART's ``--compress-psf``. Dropping the untouched
    locations is what makes a large three-dimensional kernel fit, and it
    perturbs the normal operator by what it discards -- immaterial for a solve
    whose spectrum is clear of zero, and enough to cost a rank-deficient one
    its positive definiteness. A calibration solved over a small window turns
    it off for that reason.
    """
    if chunk_size <= 0:
        raise ValueError("Toeplitz chunk_size must be positive")
    if coil_batch_size <= 0:
        raise ValueError("Toeplitz coil_batch_size must be positive")
    if cuda_mode not in {"auto", "resident", "compact"}:
        raise ValueError("Toeplitz cuda_mode must be 'auto', 'resident', or 'compact'")
    if not 0.0 < cuda_max_device_fraction <= 1.0:
        raise ValueError("Toeplitz cuda_max_device_fraction must be in (0, 1]")
    if cuda_transfer_precision not in {"auto", "float32", "bfloat16"}:
        raise ValueError(
            "Toeplitz cuda_transfer_precision must be 'auto', 'float32', or 'bfloat16'"
        )
    return {
        "compress": bool(compress),
        "chunk_size": int(chunk_size),
        "coil_batch_size": int(coil_batch_size),
        "cuda_mode": cuda_mode,
        "cuda_max_device_fraction": float(cuda_max_device_fraction),
        "cuda_transfer_precision": cuda_transfer_precision,
    }


#: Cells either side of a sample that the backend's interpolation spreads
#: into. The gridded transfer is non-zero within this reach of the trajectory
#: and nowhere else, which is the support the kernel is stored over.
_SPREAD_HALF_WIDTH = 4


def _support_locations(
    samples: Any,
    spatial_shape: tuple[int, ...],
    device: Any,
    compress: bool = True,
) -> Any:
    """The locations a gridded transfer holds weight at.

    Where the trajectory landed on the doubled grid, plus the neighbourhood the
    interpolation spread into. An encoding with no trajectory to read, which is
    what a Cartesian one leaves, keeps every location, and so does one that
    asks not to be compressed.
    """
    if samples is None or not compress:
        return support_indices(spatial_shape, support="full", radius=1.0, device=device)
    return occupancy_indices(samples, spatial_shape, width=_SPREAD_HALF_WIDTH).to(
        device
    )


def _toeplitz_request(
    value: bool | str | dict[str, Any],
) -> tuple[bool, bool, dict[str, Any]]:
    """Decode a ``toeplitz=`` argument into enabled, best-effort and options.

    ``"auto"`` asks for the acceleration wherever it is available, so a shape
    or a backend that cannot carry a transfer kernel falls back to the exact
    normal operator rather than raising. ``True`` and an options mapping ask
    for it outright, and say so if it cannot be built.
    """
    if isinstance(value, dict):
        return True, False, _toeplitz_options(**value)
    if value == "auto":
        return True, True, _toeplitz_options()
    return bool(value), False, _toeplitz_options()


class MRIPhysics(deepinv.physics.LinearPhysics):
    """DeepInverse-native facade over an MRI linear physics object.

    The facade records reconstruction-specific metadata and forwards the
    numerical contract (``A``, ``A_adjoint``, ``A_adjoint_A``, ``A_dagger``)
    to DeepInverse. ``native_operator`` is the underlying mri-nufft operator
    for non-Cartesian acquisitions and is intentionally read-only.

    Examples
    --------
    Every operator on this page is one of these: it can encode an image, take the
    adjoint of a measurement, and apply its own normal operator -- which is what a
    solver spends its time in.

    .. plot::

       import pulserver.recon as recon
       from _figures import images, phantom, radial_spokes

       truth, coil_maps = phantom(64, coils=4)
       physics = recon.NonCartesian2D(
           radial_spokes(64, 24), (64, 64), coil_maps=coil_maps[0]
       )

       measured = physics.A(truth)
       adjoint = physics.A_adjoint(measured)
       normal = physics.A_adjoint_A(truth)

       images(
           [
               ("truth", truth[0]),
               ("A then A-adjoint", adjoint[0, 0]),
               ("the normal operator", normal[0, 0]),
           ],
           title="what a physics operator does",
       )
    """

    def __init__(
        self,
        operator: Any,
        *,
        native_operator: Any | None,
        kind: str,
        spatial_ndim: int,
        viewed_as_real: bool = False,
        modifiers: tuple[str, ...] = (),
        trajectory: Any | None = None,
        rebuild: Callable[[Any, int | None], MRIPhysics] | None = None,
        toeplitz_options: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
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
        self._replicate: Callable[[Any, Any], MRIPhysics] | None = None
        self._streaming_replicas: dict[str, MRIPhysics] = {}

    @property
    def normal_mode(self) -> str:
        """Return the implementation used by the normal operator."""
        if self.kind in {"wave", "wave-shuffling"}:
            return "exact-hybrid"
        if "toeplitz" in self.modifiers:
            if self.kind.startswith("cartesian"):
                return "exact-fft"
            if getattr(self.operator, "use_toeplitz", False):
                return "toeplitz"
        return "exact"

    @_mirror_array_namespace
    def A(self, x: Any, **kwargs: Any) -> Any:
        """Apply the forward encoding operator."""
        return self._stream_call("A", self.operator.A, x, **kwargs)

    @_mirror_array_namespace
    def A_adjoint(self, y: Any, **kwargs: Any) -> Any:
        """Apply the adjoint encoding operator."""
        return self._stream_call(
            "A_adjoint",
            self.operator.A_adjoint,
            y,
            **kwargs,
        )

    @_mirror_array_namespace
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
        if self._streaming_replicas:
            devices = policy.torch_devices
            worker_count = min(len(devices), value.shape[0])
            boundaries = [
                (index * value.shape[0]) // worker_count
                for index in range(worker_count + 1)
            ]

            def apply_part(index: int) -> Any:
                start, stop = boundaries[index], boundaries[index + 1]
                replica = self._streaming_replicas[str(devices[index])]
                return getattr(replica, name)(value[start:stop], **kwargs)

            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                outputs = list(executor.map(apply_part, range(worker_count)))
            return torch.cat(outputs, dim=0)
        if self._streaming_parameters is not None:
            starts = list(range(0, value.shape[0], policy.physics_batch_size))
            outputs: list[Any | None] = [None] * len(starts)
            devices = policy.execution_devices
            streams = [torch.cuda.Stream(device=device) for device in devices]
            for chunk_index, start in enumerate(starts):
                stop = min(start + policy.physics_batch_size, value.shape[0])
                slot = chunk_index % len(streams)
                device = devices[slot]
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
                with torch.cuda.device(device), torch.cuda.stream(streams[slot]):
                    device_parameters = {
                        parameter_name: (
                            parameter.to(
                                device,
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
                        device,
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
        """Apply one bounded Cartesian SENSE batch without mutating DeepInv.

        The kernel below works in the two-channel real layout throughout. When
        the physics is complex-native (the default), the complex image or
        measurement is packed to that layout on the way in and unpacked on the
        way out, so the one arithmetic path serves both representations.
        """
        image_as_complex = _cartesian_image_as_cpx
        image_as_real = _cartesian_image_as_real
        real_view = self.viewed_as_real

        if not real_view:
            value = (
                image_as_real(value) if name != "A_adjoint" else _kspace_as_real(value)
            )

        def forward(image: Any) -> Any:
            coil_images = coil_maps * image_as_complex(image)[:, None]
            axes = tuple(range(-self.spatial_ndim, 0))
            spectrum = self.operator.fft(coil_images, dim=axes)
            return mask[:, :, None] * image_as_real(spectrum)

        def adjoint(kspace: Any) -> Any:
            masked = image_as_complex(mask[:, :, None] * kspace)
            axes = tuple(range(-self.spatial_ndim, 0))
            coil_images = self.operator.ifft(masked, dim=axes)
            if kwargs.get("rss", False):
                return self.operator.rss(image_as_real(coil_images), multicoil=True)
            image = (coil_maps.conj() * coil_images).sum(dim=1)
            return image_as_real(image)

        if name == "A":
            result = _measurement_to_trailing(forward(value))
            return result if real_view else _kspace_as_cpx(result)
        if name == "A_adjoint":
            result = adjoint(_measurement_to_channels(value))
            result = self.operator.crop(result, crop=kwargs.get("crop", False))
            return result if real_view else image_as_complex(result)
        if name == "A_adjoint_A":
            result = adjoint(forward(value))
            return result if real_view else image_as_complex(result)
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
        if value.shape[0] > policy.physics_batch_size:
            return torch.cat(
                [
                    MRIPhysics._stage_call(
                        method,
                        value[start : start + policy.physics_batch_size],
                        policy,
                        kwargs,
                    )
                    for start in range(
                        0,
                        value.shape[0],
                        policy.physics_batch_size,
                    )
                ],
                dim=0,
            )
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
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("operator"), name)

    def to(self, *args: Any, **kwargs: Any) -> MRIPhysics:
        """Move the underlying Torch physics object and return ``self``."""
        super().to(*args, **kwargs)
        return self

    def enable_streaming(self, policy: Any) -> MRIPhysics:
        """Keep reconstruction state on CPU and use bounded CUDA workspaces."""
        self.streaming_policy = policy
        if self._replicate is not None and getattr(policy, "device_count", 1) > 1:
            self._streaming_replicas = {
                str(device): self._replicate(device, policy.for_device(device))
                for device in policy.torch_devices
            }
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
        """Rebuild non-Cartesian physics for a frame-specific trajectory."""
        if self._rebuild is None:
            return self
        return self._rebuild(trajectory, frame_index)


def _init_from(physics: MRIPhysics, source: MRIPhysics) -> None:
    """Initialize a facade in place from a built one, sharing operator state.

    The wrapper and the source deliberately alias the same operator, rebuild
    hook and streaming state, so enabling a kernel or a policy on either is
    visible through both.
    """
    MRIPhysics.__init__(
        physics,
        source.operator,
        native_operator=source.native_operator,
        kind=source.kind,
        spatial_ndim=source.spatial_ndim,
        viewed_as_real=source.viewed_as_real,
        modifiers=source.modifiers,
        trajectory=source.trajectory,
        rebuild=source._rebuild,
        toeplitz_options=source.toeplitz_options,
    )
    physics.streaming_policy = source.streaming_policy
    physics._streaming_methods = source._streaming_methods
    physics._streaming_parameters = source._streaming_parameters
    physics._replicate = source._replicate
    physics._streaming_replicas = source._streaming_replicas


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
        self.shared: MRIPhysics | None = None
        self.target: int | None = None
        native = _base_fourier_operator(physics.native_operator)
        self.has_density = getattr(native, "density", None) is not None
        # A ragged acquisition has to keep a plan per frame: a NUFFT is planned
        # for a fixed number of points.
        self.shareable = (
            hasattr(native, "update_samples")
            and len(
                {int(prod(getattr(frame, "shape", (0,))[:-1])) for frame in trajectory}
            )
            == 1
        )

    def samples(self, index: int) -> Any:
        """One frame's sample set, without building the plan that reads it."""
        _require_mrinufft()
        return import_module("mrinufft._utils").proper_trajectory(
            self.trajectory[index],
            normalize="pi",
        )

    def density(self, index: int) -> Any:
        """One frame's sample weights, without building its plan."""
        native = _base_fourier_operator(self.physics.native_operator)
        return _frame_density(
            getattr(native, "density", None),
            self.trajectory,
            index,
            prod(getattr(self.trajectory, "shape", (0, 0))[1:-1]),
        )

    def _build(self, index: int) -> MRIPhysics:
        result = self.physics.rebuild(self.trajectory[index], index)
        if self.policy is not None:
            result.enable_streaming(self.policy)
        if self.toeplitz_options is not None:
            _enable_toeplitz(result, **self.toeplitz_options)
        return result

    def get(self, index: int) -> MRIPhysics:
        """The physics for one frame, planned once and retargeted after that.

        Frames of a dynamic acquisition differ only in where their samples
        fall, so they share the plan -- by far the most expensive part of a
        frame, and on CUDA the part that holds device memory for as long as
        the physics lives. Callers use one frame at a time.
        """
        if self.shared is not None:
            if self.target != index:
                native = _base_fourier_operator(self.shared.native_operator)
                native.update_samples(self.samples(index))
                if self.has_density:
                    native.density = self.density(index)
                self.target = index
            return self.shared
        if index in self.cache:
            result = self.cache.pop(index)
            self.cache[index] = result
            return result
        result = self._build(index)
        if self.shareable:
            self.shared, self.target = result, index
            return result
        self.cache[index] = result
        limit = 1 if self.policy is None else self.policy.frame_cache_size
        while len(self.cache) > limit:
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
        self.spatial_ndim = provider.physics.spatial_ndim

    @property
    def normal_mode(self) -> str:
        if "toeplitz" in self.modifiers:
            return "exact-fft" if self.kind.startswith("cartesian") else "toeplitz"
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
        self.modifiers = tuple(dict.fromkeys((*self.modifiers, "toeplitz")))

    @property
    def samples(self) -> Any:
        """This frame's trajectory, in the units a NUFFT plans on."""
        return self.provider.samples(self.index)

    @property
    def density(self) -> Any:
        """This frame's sample weights, if the acquisition carries any."""
        return self.provider.density(self.index)

    @property
    def backend(self) -> str:
        """The NUFFT backend the frames are planned on."""
        reference = _base_fourier_operator(self.provider.physics.native_operator)
        return getattr(reference, "backend", "finufft")

    @property
    def image_shape(self) -> tuple[int, ...]:
        """The image grid every frame shares."""
        reference = _base_fourier_operator(self.provider.physics.native_operator)
        return tuple(int(size) for size in reference.shape)

    @property
    def coil_view(self) -> Any:
        """The coils and the grid, without a plan built to read them off.

        Applying a kernel needs the sensitivities and the image shape, and the
        frames share both with the acquisition they came from.
        """
        reference = _base_fourier_operator(self.provider.physics.native_operator)
        return SimpleNamespace(
            shape=tuple(int(size) for size in reference.shape),
            smaps=getattr(reference, "smaps", None),
            uses_sense=getattr(reference, "uses_sense", False),
            n_coils=int(getattr(reference, "n_coils", 1) or 1),
        )


def _plan_batch_width(native_operator: Any, batch: int) -> int:
    """How many images an mri-nufft operator can be pointed at in one call.

    ``n_batchs`` only tells the operator how to fold its input; the plan is
    sized by ``n_trans``, which has to divide the transforms a call asks for.
    """
    trans = int(getattr(native_operator, "n_trans", 1) or 1)
    coils = int(getattr(native_operator, "n_coils", 1) or 1)
    if trans == 1 or (batch * coils) % trans == 0:
        return batch
    return int(getattr(native_operator, "n_batchs", 1) or 1)


@contextmanager
def _batches_of(native_operator: Any, width: int) -> Any:
    """The operator seen as folding its input into ``width`` images."""
    held = getattr(native_operator, "n_batchs", None)
    if held is None or int(held) == width:
        yield
        return
    native_operator.n_batchs = width
    try:
        yield
    finally:
        native_operator.n_batchs = held


def _over_batches(apply: Any, native_operator: Any, value: Any) -> Any:
    """Apply an operator to any number of images, whatever it plans for.

    The leading axis is the batch on both sides of a NUFFT, so a call the plan
    cannot take in one go is served in groups of the width it can, the last of
    them padded up and cut back.
    """
    torch = import_module("torch")
    batch = int(value.shape[0])
    width = _plan_batch_width(native_operator, batch)
    if width == batch:
        with _batches_of(native_operator, batch):
            return apply(value)
    results = []
    for start in range(0, batch, width):
        group = value[start : start + width]
        short = width - int(group.shape[0])
        if short:
            group = torch.cat((group, group[-1:].expand(short, *group.shape[1:])), 0)
        with _batches_of(native_operator, width):
            outcome = apply(group)
        results.append(outcome[: width - short] if short else outcome)
    return torch.cat(results, 0)


def _native_linear_physics(
    native_operator: Any,
    *,
    viewed_as_real: bool,
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
    # adapter below is only needed for third-party operators that correctly
    # implement forward/adjoint operations but do not expose an autograd
    # wrapper. PICS itself uses explicit adjoints, so both paths have identical
    # reconstruction semantics.
    if getattr(native_operator, "autograd_available", False):
        inner = native_operator.make_deepinv_phy()

        class _MRIViewPhysics(physics_module.LinearPhysics):
            def __init__(self) -> None:
                super().__init__()
                self.__dict__["inner"] = inner
                self.viewed_as_real = viewed_as_real
                self.use_toeplitz = False

            def A(self, x: Any, **kwargs: Any) -> Any:
                if self.viewed_as_real:
                    x = _image_as_cpx(x)
                result = _over_batches(
                    lambda value: self.inner.A(value, **kwargs),
                    native_operator,
                    x,
                )
                return _kspace_as_real(result) if self.viewed_as_real else result

            def A_adjoint(self, y: Any, **kwargs: Any) -> Any:
                if self.viewed_as_real:
                    y = _kspace_as_cpx(y)
                result = _over_batches(
                    lambda value: self.inner.A_adjoint(value, **kwargs),
                    native_operator,
                    y,
                )
                return _image_as_real(result) if self.viewed_as_real else result

            def A_dagger(self, y: Any, **kwargs: Any) -> Any:
                if self.viewed_as_real:
                    y = _kspace_as_cpx(y)
                result = _over_batches(
                    lambda value: self.inner.A_dagger(value, **kwargs),
                    native_operator,
                    y,
                )
                return _image_as_real(result) if self.viewed_as_real else result

            def A_adjoint_A(self, x: Any, **kwargs: Any) -> Any:
                del kwargs
                return self.A_adjoint(self.A(x))

        return _MRIViewPhysics()

    class _MRIThinPhysics(physics_module.LinearPhysics):
        def __init__(self) -> None:
            super().__init__()
            self.__dict__["native_operator"] = native_operator
            self.viewed_as_real = viewed_as_real
            self.use_toeplitz = False

        def A(self, x: Any, **kwargs: Any) -> Any:
            del kwargs
            if self.viewed_as_real:
                x = _image_as_cpx(x)
            result = _over_batches(self.native_operator.op, self.native_operator, x)
            return _kspace_as_real(result) if self.viewed_as_real else result

        def A_adjoint(self, y: Any, **kwargs: Any) -> Any:
            del kwargs
            if self.viewed_as_real:
                y = _kspace_as_cpx(y)
            result = _over_batches(self.native_operator.adj_op, self.native_operator, y)
            return _image_as_real(result) if self.viewed_as_real else result

        def A_adjoint_A(self, x: Any, **kwargs: Any) -> Any:
            del kwargs
            return self.A_adjoint(self.A(x))

    # Keep the explicit reference so static analyzers and mocked test modules
    # both validate that this is a DeepInverse physics implementation.
    assert issubclass(_MRIThinPhysics, physics_module.LinearPhysics)
    return _MRIThinPhysics()


def _mrinufft_norm_factor(shape: tuple[int, ...]) -> float:
    """The normalization an mri-nufft operator on ``shape`` divides by."""
    return sqrt(prod(shape) * 2 ** len(shape))


def _frame_density(
    density: Any,
    trajectory: Any,
    frame_index: int | None,
    frame_samples: int,
) -> Any:
    """One frame's share of a density given for a whole dynamic trajectory.

    A density may be given per frame, flat over every sample, or once for a
    trajectory every frame shares; only the first two are split.
    """
    if frame_index is None or density is None:
        return density
    density_shape = getattr(density, "shape", ())
    trajectory_shape = getattr(trajectory, "shape", ())
    if not trajectory_shape:
        return density
    if len(density_shape) > 1 and density_shape[0] == trajectory_shape[0]:
        return density[frame_index]
    if prod(density_shape) == trajectory_shape[0] * frame_samples:
        start = frame_index * frame_samples
        return density.reshape(-1)[start : start + frame_samples]
    return density


def _base_fourier_operator(native_operator: Any) -> Any:
    """Return the undecorated Fourier operator beneath mri-nufft wrappers."""
    return getattr(native_operator, "_fourier_op", native_operator)


_PSF_OPERATOR_SLOT: dict[tuple[Any, ...], Any] = {}

# The transfer is held as complex64 and then cut to the support the scan
# reached, so gridding it tighter than this buys nothing that survives either
# step and costs several times the build.
_PSF_TOLERANCE = 1e-4

# The tolerance to ask the narrow spreading grid for. Interpolation width is
# set by the pair, not by either alone: the narrow grid reaches the same
# five-wide kernel here that the wide grid reaches at _PSF_TOLERANCE, so the
# fallback costs a factor on the transfer's accuracy rather than on the
# spreading, which is what it would cost at the tighter tolerance.
_NARROW_PSF_TOLERANCE = 1e-3
_NARROW_PSF_UPSAMPLING = 1.25


def _psf_operator(
    samples: Any,
    backend: str,
    spatial_shape: tuple[int, ...],
) -> Any:
    """A NUFFT on the doubled grid, for gridding the transfer onto it.

    One plan is kept per (backend, grid, sample count) and retargeted at each
    trajectory it is asked for. Planning a NUFFT is the expensive part of
    building a kernel, and holding a second plan on the doubled grid is what
    makes a build run out of device memory.
    """
    mrinufft = _require_mrinufft()
    shape = tuple(int(size) for size in spatial_shape)
    key = (backend, shape, int(samples.shape[0]))
    operator = _PSF_OPERATOR_SLOT.get(key)
    if operator is not None:
        operator.update_samples(samples)
        return operator
    build = mrinufft.get_operator(backend)
    _yield_cached_device_memory(getattr(samples, "device", None))
    settings: dict[str, Any] = _psf_settings(shape, samples)
    try:
        operator = build(
            samples=samples,
            shape=shape,
            density=None,
            n_coils=1,
            squeeze_dims=False,
            **settings,
        )
    except TypeError:
        operator = build(
            samples=samples,
            shape=shape,
            density=None,
            n_coils=1,
            squeeze_dims=False,
        )
    # One slot: a plan on the doubled grid is the largest device allocation a
    # build makes, and holding a second one is what makes a build run out.
    _PSF_OPERATOR_SLOT.clear()
    _PSF_OPERATOR_SLOT[key] = operator
    return operator


def _yield_cached_device_memory(device: Any) -> None:
    """Hand the allocator's spare blocks back to the driver.

    A NUFFT plan is allocated outside Torch, so blocks Torch is holding for
    reuse are neither available to it nor counted as free -- and Torch does
    not release them when another library runs out. What a build measures and
    what it can take are both only true once these are returned.
    """
    torch = import_module("torch")
    if "cuda" not in str(device):
        return
    with suppress(RuntimeError):
        torch.cuda.empty_cache()


def _psf_settings(shape: tuple[int, ...], samples: Any) -> dict[str, Any]:
    """What to plan the gridding NUFFT with, given what the device has room for.

    A NUFFT spreads onto a grid of its own on the way to the one it answers
    on; that grid is internal and does not touch the transfer, so it is chosen
    for what it costs. The wide one is the default. On the doubled grid a
    kernel is built on it is eight times the transfer, so at these sizes it
    stops fitting, and the narrow one is asked for a looser tolerance -- which
    keeps its interpolation kernel the width the wide one has, and spends the
    difference on the transfer rather than on every point spread onto it.
    """
    torch = import_module("torch")
    narrow = {"eps": _NARROW_PSF_TOLERANCE, "upsampfac": _NARROW_PSF_UPSAMPLING}
    # NumPy answers `device` with a plain string, Torch with an object.
    device = getattr(samples, "device", None)
    if "cuda" not in str(device):
        return {"eps": _PSF_TOLERANCE}
    free, _ = torch.cuda.mem_get_info(device)
    spreading = 8 * (2 ** len(shape)) * prod(shape)
    wide = spreading + 8 * prod(shape) + 8 * int(samples.shape[0])
    return {"eps": _PSF_TOLERANCE} if wide < 0.6 * free else narrow


def _within_psf_plans(build: Any) -> Any:
    """Release the gridding plan a builder makes when its build ends."""

    @wraps(build)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with _psf_plans():
            return build(*args, **kwargs)

    return wrapper


@contextmanager
def _psf_plans() -> Any:
    """Hold one gridding plan for the length of a build, then release it.

    A plan on the doubled grid is the largest device allocation a build makes
    -- larger than the kernel it produces -- and the solve that follows needs
    that memory for its own transforms.
    """
    try:
        yield
    finally:
        _PSF_OPERATOR_SLOT.clear()
        with suppress(ImportError, AttributeError):
            torch = import_module("torch")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _compute_toeplitz_transfer(
    native_operator: Any,
    weights: Any | None = None,
    *,
    complex_weights: bool = False,
) -> Any:
    """The transfer a Toeplitz normal operator multiplies by.

    The point-spread function is the adjoint of the sample weights taken on a
    grid twice the image in every dimension -- ones for a plain normal, the
    density for a compensated one, a basis product for a subspace frame or an
    off-resonance segment -- and the transfer is its transform.

    Gridding is what puts the weight where the trajectory is. The adjoint
    interpolates each sample onto the doubled grid with the backend's own
    kernel, so the transfer holds weight where the scan reached and in the rim
    that interpolation spreads into, and nowhere else. That is the same
    operator the forward NUFFT applies, so the normal is the Gram of the
    transform actually being inverted.
    """
    del complex_weights
    torch = import_module("torch")
    base = _base_fourier_operator(native_operator)
    image_shape = tuple(int(size) for size in base.shape)
    spatial_shape = tuple(2 * size for size in image_shape)
    operator = _psf_operator(
        base.samples,
        getattr(base, "backend", "finufft"),
        spatial_shape,
    )

    if weights is None:
        # The plain normal is weighted by whatever the operator itself carries:
        # its adjoint applies the density once, so the Gram does too.
        weights = getattr(base, "density", None)
    if weights is None:
        values = torch.ones(
            operator.n_samples,
            dtype=torch.complex64,
            device=as_torch(base.samples).device,
        )
    else:
        values = as_torch(weights).reshape(-1).to(torch.complex64)

    # Backends differ on whether they take a bare sample vector, so the
    # batch and coil axes are stated and dropped again.
    psf = as_torch(operator.adj_op(values.reshape(1, 1, -1))).reshape(spatial_shape)
    axes = tuple(range(len(spatial_shape)))
    # ``adj_op`` answers a centred image and divides by the doubled grid's own
    # normalization, while the normal operator this stands in for carries the
    # image grid's twice -- once in the forward and once in the adjoint.
    scale = float(operator.norm_factor) / float(base.norm_factor) ** 2
    return torch.fft.fftn(torch.fft.ifftshift(psf, dim=axes), dim=axes) * scale


def _sense_maps(native_operator: Any, reference: Any) -> Any:
    """Return sensitivity maps as a Torch tensor, on whatever device holds them.

    A normal application reads one coil at a time, so maps the caller left on
    the host are staged coil by coil rather than moved whole -- the difference
    is the whole bank against one map of it.
    """
    torch = import_module("torch")
    base = _base_fourier_operator(native_operator)
    maps = getattr(base, "smaps", None)
    if maps is None:
        return torch.ones(
            (1, *base.shape),
            dtype=reference.dtype,
            device=reference.device,
        )
    maps = as_torch(maps).to(reference.dtype)
    spatial_ndim = len(base.shape)
    if maps.ndim == spatial_ndim:
        return maps[None]
    if maps.ndim in {spatial_ndim + 1, spatial_ndim + 2}:
        return maps
    raise ValueError(
        "sensitivity maps must have shape (coils, *image_shape) or "
        "(batch, coils, *image_shape)"
    )


def _frame_coil_view(frame: MRIPhysics | _LazyFramePhysics) -> Any:
    """What a kernel needs of a frame: its coils and its grid, never a plan."""
    view = getattr(frame, "coil_view", None)
    return frame.native_operator if view is None else view


def _coils_split_across_devices(
    kernel: CompactToeplitzKernel,
    image: Any,
    maps: Any,
    streaming: Any,
    *,
    batched_maps: bool,
    n_coils: int,
) -> Any:
    """Sum a normal application over coils divided between CUDA devices.

    Coils are independent until the sum that ends them, so each device is given
    a share of them, its own copy of the transfer and its own copy of the
    image, and returns the part of the sum it computed.

    This has not been run on a machine with more than one GPU. What it assumes
    of a second device is what ``for_device`` and ``_apply_sense_toeplitz``
    already assume of the first.
    """
    devices = streaming.torch_devices[: min(streaming.device_count, n_coils)]
    edges = [(index * n_coils) // len(devices) for index in range(len(devices) + 1)]

    def share(position: int) -> Any:
        device = devices[position]
        start, stop = edges[position], edges[position + 1]
        coils = (maps[:, start:stop] if batched_maps else maps[start:stop]).to(device)
        held = SimpleNamespace(
            shape=kernel.image_shape,
            smaps=coils,
            uses_sense=True,
        )
        return _apply_sense_toeplitz(
            kernel.for_device(device),
            image.to(device),
            held,
            coil_batch_size=1,
        )

    with ThreadPoolExecutor(max_workers=len(devices)) as workers:
        parts = list(workers.map(share, range(len(devices))))
    total = parts[0].to(image.device)
    for part in parts[1:]:
        total = total + part.to(image.device)
    return total


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
    if streaming is not None and streaming.device_count > 1:
        # Coils are independent until their final SENSE reduction.  Group at
        # least one coil per device so even a single-image reconstruction can
        # fan its Toeplitz work across a multi-GPU recon host.
        coil_batch_size = max(coil_batch_size, streaming.device_count)
    maps = _sense_maps(native_operator, image)
    # An image is (batch, *spatial) and unbatched maps are (coils, *spatial),
    # so the two carry the same rank; only the maps' own rank separates them.
    batched_maps = maps.ndim == len(kernel.image_shape) + 2
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
    if (
        streaming is not None
        and streaming.device_count > 1
        and n_coils > 1
        and left_factors is None
        and right_factors is None
    ):
        return _coils_split_across_devices(
            kernel,
            image,
            maps,
            streaming,
            batched_maps=batched_maps,
            n_coils=n_coils,
        )
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
            coil_maps = maps[:, start : start + coil_batch_size].to(image.device)
            left = image[:, None]
            right = coil_maps[:, :, None]
        else:
            coil_maps = maps[start : start + coil_batch_size].to(image.device)
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
            try:
                kernel._apply_cuda_resident(
                    image,
                    right_factor=factor,
                    left_factor=factor.conj(),
                    output=result,
                )
            except RuntimeError as error:
                if kernel.cuda_mode == "resident" or not _device_is_full(error):
                    raise
                kernel._resident_refused()
            else:
                kernel._last_cuda_mode = "resident"
                continue
        fused_streaming = (
            streaming is not None and image.device.type == "cpu" and coil_count == 1
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
    kernel.settle_allocator()
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


@_within_psf_plans
def _build_scalar_toeplitz(
    native_operator: Any,
    options: dict[str, Any],
    streaming: Any | None = None,
) -> CompactToeplitzKernel:
    """Build Pulserver's compact rank-one NUFFT transfer."""
    base = _base_fourier_operator(native_operator)
    image_shape = tuple(int(size) for size in base.shape)
    spatial_shape = tuple(2 * size for size in image_shape)
    transfer = as_torch(_compute_toeplitz_transfer(base)).flatten()
    indices = _support_locations(
        getattr(base, "samples", None),
        spatial_shape,
        "cpu" if streaming is not None else transfer.device,
        options["compress"],
    )
    values = _selected_transfer(transfer, indices, streaming=streaming).real[None]
    kernel = CompactToeplitzKernel(
        values,
        indices,
        spatial_shape,
        1,
        image_shape=image_shape,
        chunk_size=options["chunk_size"],
        cuda_mode=options["cuda_mode"],
        cuda_max_device_fraction=options["cuda_max_device_fraction"],
        cuda_transfer_precision=options["cuda_transfer_precision"],
    )
    return kernel


def _configure_base_toeplitz(
    operator: Any,
    native_operator: Any,
    *,
    enabled: bool,
    best_effort: bool = False,
    options: dict[str, Any],
) -> Any:
    """Install Pulserver's lazy scalar normal on a native NUFFT adapter.

    The kernel is built on the first normal-operator call, so an operator that
    only ever encodes or decodes pays nothing for carrying one. ``best_effort``
    is what ``toeplitz="auto"`` asks for: a shape the backend cannot embed
    circulantly reverts to the exact normal instead of raising.
    """
    operator.use_toeplitz = enabled
    operator.toeplitz_best_effort = best_effort
    operator.toeplitz_kernel = None
    operator._toeplitz_options = dict(options)
    operator.streaming_policy = None
    operator.streaming_methods = {"A_adjoint_A"}

    def enable_toeplitz(self: Any, new_options: dict[str, Any]) -> None:
        self.use_toeplitz = True
        self.toeplitz_best_effort = False
        self._toeplitz_options = dict(new_options)
        self.toeplitz_kernel = None

    def enable_streaming(self: Any, policy: Any) -> None:
        self.streaming_policy = policy
        self.toeplitz_kernel = None

    def scalar_normal(self: Any, x: Any, **kwargs: Any) -> Any:
        del kwargs
        if not self.use_toeplitz:
            return self.A_adjoint(self.A(x))
        image = _image_as_cpx(x) if self.viewed_as_real else x
        if self.toeplitz_kernel is None:
            try:
                self.toeplitz_kernel = _build_scalar_toeplitz(
                    native_operator,
                    self._toeplitz_options,
                    self.streaming_policy,
                )
            except (ValueError, NotImplementedError):
                if not self.toeplitz_best_effort:
                    raise
                self.use_toeplitz = False
                return self.A_adjoint(self.A(x))
        base = _base_fourier_operator(native_operator)
        if getattr(base, "uses_sense", False):
            result = _apply_sense_toeplitz(
                self.toeplitz_kernel,
                image,
                native_operator,
                coil_batch_size=self._toeplitz_options["coil_batch_size"],
                streaming=self.streaming_policy,
            )
        else:
            batch, channels, *spatial = image.shape
            flattened = image.reshape(batch * channels, 1, *spatial)
            result = (
                self.toeplitz_kernel.apply_streamed(
                    flattened,
                    self.streaming_policy,
                )
                if self.streaming_policy is not None and flattened.device.type == "cpu"
                else self.toeplitz_kernel.apply(flattened)
            ).reshape(batch, channels, *spatial)
        self.toeplitz_kernel.settle_allocator()
        return _image_as_real(result) if self.viewed_as_real else result

    operator.enable_toeplitz = MethodType(enable_toeplitz, operator)
    operator.enable_streaming = MethodType(enable_streaming, operator)
    operator.A_adjoint_A = MethodType(scalar_normal, operator)
    return operator


def _subspace_frame_blocks(
    frame_physics: Sequence[MRIPhysics | _LazyFramePhysics],
    basis: Any,
    rows: Any,
    columns: Any,
) -> tuple[list[tuple[Any, Any, Any]], str, tuple[int, ...]]:
    """Group the frames onto the distinct trajectories they were acquired on.

    Returns one entry per distinct trajectory -- its samples, its sample
    weights and the coefficient every upper-triangular basis pair enters it
    with, summed over the frames that share it -- alongside the backend and
    the image grid they all agree on.
    """
    torch = import_module("torch")
    order: list[Any] = []
    blocks: dict[Any, tuple[Any, Any, Any]] = {}
    backend = None
    image_shape = None
    for frame, item in enumerate(frame_physics):
        coefficients = basis[rows, frame] * basis[columns, frame].conj()
        if isinstance(item, _LazyFramePhysics):
            key: Any = ("lazy", id(item.provider), item.index)
            samples, weights = item.samples, item.density
            frame_backend, frame_shape = item.backend, item.image_shape
        else:
            native = item.native_operator
            if native is None or hasattr(native, "B"):
                raise RuntimeError(
                    "A combined subspace kernel requires undecorated frame NUFFTs."
                )
            base = _base_fourier_operator(native)
            key = id(base)
            samples, weights = base.samples, getattr(base, "density", None)
            frame_backend = getattr(base, "backend", "finufft")
            frame_shape = tuple(int(size) for size in base.shape)
        if image_shape is None:
            backend, image_shape = frame_backend, frame_shape
        elif frame_shape != image_shape:
            raise ValueError("all subspace frames must share one image shape")
        if key in blocks:
            held = blocks[key]
            blocks[key] = (held[0], held[1], held[2] + coefficients.to(held[2].device))
        else:
            order.append(key)
            blocks[key] = (samples, weights, coefficients)
    if image_shape is None:
        raise ValueError("a subspace kernel needs at least one frame")
    assert backend is not None
    del torch
    return [blocks[key] for key in order], backend, image_shape


def _centring_signs(indices: Any, spatial_shape: tuple[int, ...]) -> Any:
    """The sign that centres a transfer, at the locations it is kept over.

    Shifting a point-spread function by half the grid before transforming it
    multiplies every output by ``(-1)`` raised to the sum of its coordinates,
    so the shift never has to be performed and no copy of the doubled grid is
    made to hold it.
    """
    torch = import_module("torch")
    flat = as_torch(indices).to(torch.int64)
    parity = torch.zeros_like(flat)
    stride = 1
    for size in reversed(spatial_shape):
        parity = parity + (flat // stride) % size
        stride *= size
    return torch.where(parity % 2 == 0, 1.0, -1.0).to(torch.complex64)


def _subspace_pair_transfers(
    blocks: Sequence[tuple[Any, Any, Any]],
    backend: str,
    image_shape: tuple[int, ...],
    samples: Any,
    counts: Sequence[int],
    indices: Any,
    *,
    streaming: Any | None = None,
    keep_complex: bool = True,
) -> Any:
    """Grid one transfer per upper-triangular basis pair, over every sample.

    A pair's transfer is the adjoint of one weight per sample -- the frame's
    basis product, times whatever density the acquisition carries -- so the
    whole dynamic acquisition grids in a single pass and the count of NUFFTs
    is the size of the basis, not the length of the scan.

    Each is cut to ``indices`` as it is gridded and put down on the host in the
    form it is kept in, so a build holds one row of the device rather than the
    whole packed set twice over -- once complex and once made real.
    """
    torch = import_module("torch")
    spatial_shape = tuple(2 * size for size in image_shape)
    # One weight per sample, assembled in a single pass: a dynamic acquisition
    # has as many blocks as it has frames, and touching each of them per basis
    # pair is thousands of launches for one vector.
    weights = None
    if any(block[1] is not None for block in blocks):
        pieces = []
        for (_, density, _), count in zip(blocks, counts, strict=True):
            if density is None:
                pieces.append(torch.ones(count, device=samples.device))
                continue
            piece = as_torch(density).reshape(-1).to(samples.device)
            if piece.numel() != count:
                raise ValueError("density and samples must have the same length")
            pieces.append(piece)
        weights = torch.cat(pieces).to(torch.complex64)
    repeats = torch.tensor(counts, device=samples.device)
    coefficients = torch.stack([block[2] for block in blocks], dim=1)

    operator = _psf_operator(samples, backend, spatial_shape)
    axes = tuple(range(len(spatial_shape)))
    signs = _centring_signs(indices, spatial_shape)
    scale = float(operator.norm_factor) / _mrinufft_norm_factor(image_shape) ** 2
    n_pairs = int(blocks[0][2].numel())

    packed = None
    coefficients = coefficients.to(device=samples.device, dtype=torch.complex64)
    for pair in range(n_pairs):
        values = torch.repeat_interleave(coefficients[pair], repeats)
        if weights is not None:
            values = values * weights
        values_view = values.reshape(1, 1, -1)
        del values
        psf = as_torch(operator.adj_op(values_view)).reshape(spatial_shape)
        # Transformed in place, and the centring folded into a sign on the
        # locations kept: shifting the point-spread function by half the grid
        # is the same as alternating the sign of what comes out, and a copy of
        # the doubled grid is the largest thing a build holds after the plan.
        torch.fft.fftn(psf, dim=axes, out=psf)
        selected = _selected_transfer(psf.reshape(-1), indices, streaming=streaming)
        del psf
        row = selected * signs * scale
        del selected
        if not keep_complex:
            row = row.real
        if packed is None:
            packed = torch.empty(
                (n_pairs, row.numel()),
                dtype=row.dtype,
                device="cpu",
            )
        packed[pair].copy_(row)
        del row
    assert packed is not None
    return packed


_PLAN_SETTINGS = "_pulserver_plan_settings"
_LAZY_PLANS = "_pulserver_lazy_plans"


def _remember_plan_settings(native_operator: Any, settings: dict[str, Any]) -> None:
    """Keep what an operator was planned with, so it can be planned again."""
    base = _base_fourier_operator(native_operator)
    with suppress(AttributeError):
        setattr(base, _PLAN_SETTINGS, dict(settings))


def _plans_made_when_asked(raw: Any, settings: dict[str, Any], samples: Any) -> None:
    """Let each transform of ``raw`` plan itself the first time it runs.

    Points are held rather than set while a plan is absent, so an operator can
    be aimed at new samples without a plan to aim, and arrives at the transform
    pointed where its last caller asked.
    """
    state = getattr(raw, _LAZY_PLANS, None)
    if state is not None:
        state["settings"] = dict(settings)
        state["samples"] = {1: samples, 2: samples}
        return

    state = {"settings": dict(settings), "samples": {1: samples, 2: samples}}
    set_pts = raw._set_pts

    def points(typ: Any, new_samples: Any) -> None:
        if typ in state["samples"] and raw.plans[typ] is None:
            state["samples"][typ] = new_samples
            return
        set_pts(typ, new_samples)

    def planned(typ: int) -> Any:
        if raw.plans[typ] is None:
            raw._make_plan(typ, **state["settings"])
            set_pts(typ, state["samples"][typ])
        return raw.plans[typ]

    def type1(coefficients: Any, grid: Any) -> Any:
        return planned(1).execute(coefficients, grid)

    def type2(grid: Any, coefficients: Any) -> Any:
        return planned(2).execute(grid, coefficients)

    raw._set_pts = points
    raw.type1 = type1
    raw.type2 = type2
    setattr(raw, _LAZY_PLANS, state)


@contextmanager
def _plans_given_up(native_operator: Any) -> Any:
    """Release an operator's NUFFT plans, and plan again when one is asked for.

    A plan is bound to the grid it answers on, so an encoding plan and the
    gridding plan of a kernel built from the same points cannot be one object.
    They can, however, take turns: the samples outlive the plan, so the
    encoding side gives its plan up for the length of a build. It takes the
    plan back on its next transform rather than at the end of the build,
    because what a kernel is for is standing in for that transform -- a solve
    that has one applies it many times over and encodes no further.
    """
    base = _base_fourier_operator(native_operator)
    raw = getattr(base, "raw_op", None)
    settings = getattr(base, _PLAN_SETTINGS, None)
    samples = getattr(base, "_samples", None)
    held = getattr(raw, "plans", None)
    reusable = (
        settings is not None
        and samples is not None
        and held is not None
        and getattr(raw, "grad_plan", None) is None
        and "cuda" in str(getattr(samples, "device", ""))
    )
    if not reusable:
        yield
        return
    # The plans go only when nothing refers to them, this frame included.
    width = len(held)
    del held
    raw.plans = [None] * width
    _yield_cached_device_memory(getattr(samples, "device", None))
    try:
        yield
    finally:
        _plans_made_when_asked(raw, settings, samples)


def _maps_parked_on_host(native_operator: Any) -> None:
    """Send an operator's sensitivities to the host for a kernel's lifetime.

    Sensitivities have the same lifetime on a device as the plans beside them:
    a normal operator that is a kernel reads its own copy a coil at a time, and
    the operator they belong to encodes once, if at all. It stages them back a
    coil at a time when it is next asked to.
    """
    base = _base_fourier_operator(native_operator)
    maps = getattr(base, "_smaps", None)
    device = getattr(maps, "device", None)
    # A host array carries a device too, spelled as a bare string.
    if maps is None or getattr(device, "type", "cpu") == "cpu":
        return
    with suppress(AttributeError, RuntimeError):
        base._smaps = maps.to("cpu")


@contextmanager
def _frames_release_their_plans(
    frame_physics: Sequence[MRIPhysics | _LazyFramePhysics],
) -> Any:
    """Give a build the device to itself.

    Building a kernel needs the samples and the basis, not the operator that
    encodes with them -- and that operator holds a plan the size of the one the
    gridding is about to ask for. Two of them on a card sized for one is what
    turns a transform into several. Frames plan again the next time they are
    asked to encode, which costs one plan against the several a build spends
    starved of memory.
    """
    providers = {
        id(item.provider): item.provider
        for item in frame_physics
        if isinstance(item, _LazyFramePhysics)
    }
    for provider in providers.values():
        provider.shared = None
        provider.target = None
        provider.cache.clear()
        _maps_parked_on_host(provider.physics.native_operator)
    if providers:
        with suppress(ImportError, AttributeError):
            torch = import_module("torch")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    with ExitStack() as encoding:
        for provider in providers.values():
            encoding.enter_context(_plans_given_up(provider.physics.native_operator))
        yield


@_within_psf_plans
def _build_subspace_toeplitz(
    frame_physics: Sequence[MRIPhysics | _LazyFramePhysics],
    basis: Any,
    options: dict[str, Any],
    streaming: Any | None = None,
) -> CompactToeplitzKernel:
    """Grid one transfer per basis pair and pack them as coefficient matrices."""
    torch = import_module("torch")
    basis = torch.as_tensor(basis)
    rank, _ = basis.shape
    rows, columns = torch.triu_indices(rank, rank, device=basis.device)

    blocks, backend, image_shape = _subspace_frame_blocks(
        frame_physics,
        basis,
        rows,
        columns,
    )
    stack = ExitStack()
    stack.enter_context(_frames_release_their_plans(frame_physics))
    spatial_shape = tuple(2 * size for size in image_shape)
    # The support is the union of what the frames reached, read off their
    # trajectories and needing none of their transfers -- so it is known before
    # the first one is gridded, and each can be cut as it comes.
    ndim = len(image_shape)
    counts = [as_torch(block[0]).reshape(-1, ndim).shape[0] for block in blocks]
    samples = torch.cat([as_torch(block[0]).reshape(-1, ndim) for block in blocks])
    blocks = [(None, block[1], block[2]) for block in blocks]
    indices = _support_locations(
        samples,
        spatial_shape,
        "cpu" if streaming is not None else samples.device,
        options["compress"],
    )
    packed = _subspace_pair_transfers(
        blocks,
        backend,
        image_shape,
        samples,
        counts,
        indices,
        streaming=streaming,
        keep_complex=bool(basis.is_complex()),
    )

    stack.close()
    values = (
        packed.to(basis.dtype) if basis.is_complex() else packed.real.to(basis.dtype)
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
    return kernel


@_within_psf_plans
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
    spatial_ndim = getattr(frame_physics[0], "spatial_ndim", maps.ndim - 1)
    image_shape = tuple(int(size) for size in maps.shape[-spatial_ndim:])
    # DeepInverse represents Cartesian masks as (batch, real/imag, H, W).
    # The two channels are identical; retain only one before interpreting the
    # leading dimension as shared/per-frame masks.
    channel_axis = -(spatial_ndim + 1)
    if mask.ndim >= spatial_ndim + 2 and mask.shape[channel_axis] == 2:
        mask = mask.select(channel_axis, 0)
    masks = mask.reshape(-1, *image_shape)
    if masks.shape[0] == 1:
        masks = masks.expand(n_frames, *image_shape)
    elif masks.shape[0] != n_frames:
        raise ValueError(
            "Cartesian subspace mask must be shared or have one mask per frame"
        )
    masks = torch.fft.ifftshift(masks, dim=(-2, -1)).abs().square()
    # A Cartesian mask fills its own grid, so there is nothing to leave out.
    indices = _support_locations(None, image_shape, masks.device, options["compress"])
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
    """Return upper-triangular segment transfers at their retained locations."""
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
            indices = _support_locations(
                getattr(base, "samples", None),
                spatial_shape,
                kernel_device,
                options["compress"],
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


@_within_psf_plans
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
    if streaming is not None and streaming.device_count > 1:
        coil_batch_size = max(coil_batch_size, streaming.device_count)
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
        coil_maps = maps[start : start + coil_batch_size].to(image.device)
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
    kernel.settle_allocator()
    return result


class _CoilwiseCartesianMRI(deepinv.physics.MultiCoilMRI):
    """A Cartesian MRI operator with no sensitivity maps: the coil axis passes
    through untouched.

    DeepInverse's :class:`~deepinv.physics.MultiCoilMRI` collapses the coils in
    its adjoint -- a sensitivity combination when maps are given, and a plain
    sum when they are not. A coil sum is not a meaningful image, so with no maps
    this variant keeps the coils instead: the adjoint returns one image per coil
    and the forward encodes each coil independently. That matches the
    convention the non-Cartesian (mri-nufft) operators already follow, and
    leaves the coil combination to the caller as an explicit step.
    """

    def _spatial_dims(self) -> tuple[int, ...]:
        return (-3, -2, -1) if self.three_d else (-2, -1)

    def A(self, x: Any, mask: Any = None, **kwargs: Any) -> Any:
        """Encode each coil independently: a masked FFT, coils untouched."""
        self.update_parameters(mask=mask, check_coil_maps=False, **kwargs)
        spectrum = self.fft(self.to_torch_complex(x), dim=self._spatial_dims())
        return self.mask[:, :, None] * self.from_torch_complex(spectrum)

    def A_adjoint(
        self, y: Any, mask: Any = None, crop: bool = False, **kwargs: Any
    ) -> Any:
        """Return one image per coil, without combining them."""
        self.update_parameters(mask=mask, check_coil_maps=False, **kwargs)
        masked = self.to_torch_complex(self.mask[:, :, None] * y)
        coil_images = self.ifft(masked, dim=self._spatial_dims())
        return self.crop(self.from_torch_complex(coil_images), crop=crop)

    def A_adjoint_A(self, x: Any, **kwargs: Any) -> Any:
        """Per-coil normal operator: an exact FFT round trip on each coil."""
        return self.A_adjoint(self.A(x, **kwargs))


def _single_precision(value: Any) -> Any:
    """``value`` in the precision every operator here works in.

    A trajectory is what a NUFFT plans on and sensitivities are what it
    applies, so a double-precision one plans a double-precision transform
    and then meets single-precision data -- which the backend reports as a
    dtype mismatch, from inside a plan, far from the call that caused it.
    Whatever arrives, NumPy or Torch, a sequence of either, leaves single.
    """
    import numpy
    import torch

    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return type(value)(_single_precision(item) for item in value)
    if isinstance(value, torch.Tensor):
        if value.dtype == torch.complex128:
            return value.to(torch.complex64)
        if value.dtype == torch.float64:
            return value.to(torch.float32)
        return value
    if isinstance(value, numpy.ndarray):
        if value.dtype == numpy.complex128:
            return value.astype(numpy.complex64, copy=False)
        if value.dtype == numpy.float64:
            return value.astype(numpy.float32, copy=False)
        return value
    return value


def _init_cartesian(
    physics: MRIPhysics,
    mask: Any,
    coil_maps: Any,
    *,
    spatial_ndim: int,
    toeplitz: bool | dict[str, Any] = False,
    viewed_as_real: bool = False,
    **kwargs: Any,
) -> None:
    """Initialize Cartesian physics in place.

    With ``coil_maps`` this is SENSE: the adjoint combines the coils through the
    sensitivities. With ``coil_maps=None`` it is a coil-wise operator whose
    adjoint returns one image per coil, for an explicit coil combination
    afterwards (see :class:`_CoilwiseCartesianMRI`).

    Leading dimensions are handled by DeepInverse as batch dimensions, so
    slices, contrasts, and dynamic frames are reconstructed independently.
    Cartesian normal operations already use exact FFTs; ``toeplitz=True`` is
    accepted for API symmetry and reports ``normal_mode == "exact-fft"``.
    """
    toeplitz_enabled, _, options = _toeplitz_request(toeplitz)
    physics_module = _require_deepinv()
    # Direct imports rather than the module's ``import_module`` so the array
    # boundary keeps working when a test stubs the latter for operator
    # selection.
    import numpy
    import torch

    requested_device = _resolve_device(kwargs.pop("device", None))
    if isinstance(mask, numpy.ndarray):
        mask = torch.as_tensor(mask).to(torch.float32)
    if isinstance(coil_maps, numpy.ndarray):
        coil_maps = torch.as_tensor(coil_maps).to(torch.complex64)
    mask, coil_maps = _single_precision(mask), _single_precision(coil_maps)
    if requested_device is not None:
        if hasattr(mask, "to"):
            mask = mask.to(requested_device)
        if coil_maps is not None and hasattr(coil_maps, "to"):
            coil_maps = coil_maps.to(requested_device)
    device = getattr(coil_maps, "device", getattr(mask, "device", "cpu"))
    operator_class = (
        _CoilwiseCartesianMRI if coil_maps is None else physics_module.MultiCoilMRI
    )
    operator = operator_class(
        mask=mask,
        coil_maps=coil_maps,
        three_d=spatial_ndim == 3,
        device=device,
        **kwargs,
    )
    boundary = _TrailingRealView(operator)
    if not viewed_as_real:
        boundary = _CartesianComplexView(boundary)
    MRIPhysics.__init__(
        physics,
        boundary,
        native_operator=None,
        kind=f"cartesian{spatial_ndim}d",
        spatial_ndim=spatial_ndim,
        viewed_as_real=viewed_as_real,
        modifiers=("toeplitz",) if toeplitz_enabled else (),
        toeplitz_options=options if toeplitz_enabled else None,
    )
    physics._streaming_parameters = {
        "mask": getattr(operator, "mask", mask),
        "coil_maps": getattr(operator, "coil_maps", coil_maps),
    }


class Cartesian2D(MRIPhysics):
    """Two-dimensional Cartesian physics.

    Parameters
    ----------
    mask
        Sampling mask over the encoded grid, shaped ``(h, w)``, ``(c, h, w)``
        or ``(batch, c, h, w)``. Non-zero marks an acquired position.
    coil_maps
        Complex sensitivities shaped ``(coils, h, w)`` or
        ``(batch, coils, h, w)``. ``None`` (the default) is a coil-wise
        operator with no sensitivities: the adjoint returns one image per coil
        rather than a combined image, ``img_size=(h, w)`` is then required, and
        the coil combination is the caller's own explicit step.
    toeplitz
        Accepted for symmetry with the non-Cartesian operators. A Cartesian
        normal operator is already an exact FFT, so this only changes what
        ``normal_mode`` reports.
    **kwargs
        Forwarded to :class:`deepinv.physics.MultiCoilMRI`.

    Notes
    -----
    Everything is native complex: a SENSE image is ``(batch, h, w)``, a
    coil-wise (no-maps) image keeps its coils, ``(batch, coils, h, w)``, and a
    measurement is ``(batch, coils, h, w)`` -- the complex layout every physics
    in this package answers in. Leading dimensions beyond the batch are
    independent problems, so slices, contrasts and frames reconstruct together.

    Examples
    --------
    >>> import torch
    >>> from pulserver.recon.physics import Cartesian2D
    >>> physics = Cartesian2D(
    ...     torch.ones(1, 1, 8, 8),
    ...     torch.ones(1, 3, 8, 8, dtype=torch.complex64) / 3 ** 0.5,
    ... )
    >>> physics.A(torch.randn(1, 8, 8, dtype=torch.complex64)).shape
    torch.Size([1, 3, 8, 8])

    With no maps the adjoint keeps one image per coil, for an explicit
    combination afterwards:

    >>> coil_wise = Cartesian2D(torch.ones(1, 1, 8, 8), img_size=(8, 8))
    >>> coil_wise.A_adjoint(torch.randn(1, 4, 8, 8, dtype=torch.complex64)).shape
    torch.Size([1, 4, 8, 8])

    What the operator does, on DeepInverse's phantom: measure the object
    through each element of the array, and bring it back. Without maps the
    adjoint keeps the coils apart, which is what a calibration wants to see:

    .. plot::

       import torch
       import pulserver.recon as recon
       from _figures import images, phantom

       truth, coil_maps = phantom(64, coils=4)
       mask = torch.ones(1, 1, 64, 64)
       coil_wise = recon.Cartesian2D(mask, img_size=(64, 64))
       measured = recon.Cartesian2D(mask, coil_maps).A(truth)
       coils = coil_wise.A_adjoint(measured)
       images(
           [("object", truth), ("coil 0", coils[0, 0]), ("coil 2", coils[0, 2])],
           title="Cartesian2D, fully sampled, four elements",
       )
    """

    def __init__(
        self,
        mask: Any,
        coil_maps: Any = None,
        *,
        toeplitz: bool | dict[str, Any] = False,
        **kwargs: Any,
    ) -> None:
        _init_cartesian(
            self,
            mask,
            coil_maps,
            spatial_ndim=2,
            toeplitz=toeplitz,
            **kwargs,
        )


class Cartesian3D(MRIPhysics):
    """Three-dimensional Cartesian physics.

    Parameters
    ----------
    mask
        Sampling mask over the encoded volume, trailing ``(d, h, w)``.
    coil_maps
        Complex sensitivities shaped ``(coils, d, h, w)`` or with a leading
        batch. ``None`` (the default) is a coil-wise operator; see
        :class:`Cartesian2D`.
    toeplitz
        Accepted for symmetry; see :class:`Cartesian2D`.
    **kwargs
        Forwarded to :class:`deepinv.physics.MultiCoilMRI`.

    Notes
    -----
    Native complex throughout: a SENSE image is ``(batch, d, h, w)``, a
    coil-wise image ``(batch, coils, d, h, w)``, and a measurement
    ``(batch, coils, d, h, w)``. See :class:`Cartesian2D` for the layout
    convention.

    Examples
    --------
    >>> import torch
    >>> import pulserver.recon as recon
    >>> physics = recon.Cartesian3D(
    ...     torch.ones(1, 1, 8, 8, 8),
    ...     torch.ones(1, 2, 8, 8, 8, dtype=torch.complex64) / 2 ** 0.5,
    ... )
    >>> physics.A(torch.zeros(1, 8, 8, 8, dtype=torch.complex64)).shape
    torch.Size([1, 2, 8, 8, 8])
    """

    def __init__(
        self,
        mask: Any,
        coil_maps: Any = None,
        *,
        toeplitz: bool | dict[str, Any] = False,
        **kwargs: Any,
    ) -> None:
        _init_cartesian(
            self,
            mask,
            coil_maps,
            spatial_ndim=3,
            toeplitz=toeplitz,
            **kwargs,
        )


class SMS(MRIPhysics):
    """Model-based simultaneous-multislice MRI physics.

    A shared base physics is vectorized over the product of batch and slice
    axes, allowing its existing dual-GPU streaming policy to distribute all
    slices together. A sequence of physics objects represents slices with
    distinct trajectories or sampling operators and is composed exactly.

    Parameters
    ----------
    physics
        One shared MRI physics object or one object per simultaneously excited
        slice.
    caipi_encoding
        Complex CAIPI modulation or phase in radians. Its first axis is slice;
        remaining axes broadcast over the trailing measurement dimensions.
    n_slices
        Slice count when a shared physics and no encoding tensor are supplied.
    streaming
        Optional Pulserver CUDA streaming policy forwarded to the base physics.

    Examples
    --------
    Simultaneous multi-slice: the slices are excited together and arrive summed,
    with a CAIPI phase telling them apart. The operator carries that phase, so a
    solve unfolds the slices rather than a separate unaliasing step doing it.

    >>> import torch
    >>> import pulserver.recon as recon
    >>> base = recon.Cartesian2D(torch.ones(16, 16))
    >>> physics = recon.SMS(base, n_slices=2)
    >>> isinstance(physics, recon.MRIPhysics)
    True
    """

    def __init__(
        self,
        physics: MRIPhysics | Sequence[MRIPhysics],
        caipi_encoding: Any | None = None,
        *,
        n_slices: int | None = None,
        streaming: Any | None = None,
    ) -> None:
        selected = list(physics) if isinstance(physics, Sequence) else physics
        operator = _SMSLinearPhysics(selected, caipi_encoding, n_slices)
        base = selected[0] if isinstance(selected, list) else selected
        super().__init__(
            operator,
            native_operator=None,
            kind="sms",
            spatial_ndim=int(getattr(base, "spatial_ndim", 2)),
            viewed_as_real=operator.viewed_as_real,
            modifiers=tuple(dict.fromkeys((*getattr(base, "modifiers", ()), "sms"))),
        )
        if streaming is not None:
            self.enable_streaming(streaming)


def _stacked_trajectory_bank(
    trajectory: Any,
    z_index: Any,
    stack_size: int,
) -> tuple[list[Any], Any, list[Any] | None]:
    """Resolve shared or plane-specific 2D trajectories and stack indices."""
    numpy = import_module("numpy")

    def host(value: Any) -> Any:
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        return numpy.asarray(value)

    shape = getattr(trajectory, "shape", ())
    if not shape and isinstance(trajectory, (list, tuple)) and trajectory:
        shape = getattr(trajectory[0], "shape", ())
    coordinate_dim = shape[-1] if shape else None
    if coordinate_dim == 3:
        samples = host(trajectory).reshape(-1, 3)
        z_coordinates = numpy.asarray(
            import_module("mrinufft._utils").proper_trajectory(
                samples[:, 2],
                normalize="unit",
            )
        ).reshape(-1)
        _, first = numpy.unique(z_coordinates, return_index=True)
        ordered_z = z_coordinates[numpy.sort(first)]
        groups = [
            numpy.flatnonzero(numpy.isclose(z_coordinates, value))
            for value in ordered_z
        ]
        planes = [samples[group, :2] for group in groups]
        indices = numpy.rint(ordered_z * stack_size + stack_size // 2).astype(
            numpy.int64
        )
        if numpy.any(indices < 0) or numpy.any(indices >= stack_size):
            raise ValueError("stacked trajectory contains an out-of-grid z coordinate")
        if numpy.unique(indices).size != indices.size:
            raise ValueError("stacked trajectory maps multiple planes to one z index")
        return planes, indices, groups
    if coordinate_dim != 2:
        raise ValueError("stacked trajectories must end in two or three coordinates")

    if z_index is None or (isinstance(z_index, str) and z_index == "auto"):
        indices = numpy.arange(stack_size, dtype=numpy.int64)
    else:
        try:
            indices = numpy.arange(stack_size, dtype=numpy.int64)[host(z_index)]
        except IndexError as error:
            raise ValueError("z_index must select valid stack entries") from error
        indices = numpy.asarray(indices, dtype=numpy.int64).reshape(-1)
    if indices.size == 0:
        raise ValueError("z_index must select at least one stack entry")
    if numpy.unique(indices).size != indices.size:
        raise ValueError("z_index must not contain duplicate stack entries")

    explicit_sequence = isinstance(trajectory, (list, tuple))
    array = None if explicit_sequence else host(trajectory)
    banked_array = (
        array is not None and array.ndim >= 4 and array.shape[0] == indices.size
    )
    if explicit_sequence or banked_array:
        entries = list(trajectory) if explicit_sequence else list(array)
        if len(entries) != indices.size:
            raise ValueError("one 2D trajectory is required per selected stack plane")
        planes = [host(entry).reshape(-1, 2) for entry in entries]
    else:
        shared = host(trajectory).reshape(-1, 2)
        planes = [shared] * indices.size
    return planes, indices, None


def _stacked_density_bank(
    density: Any | None,
    trajectories: Sequence[Any],
    sample_groups: Sequence[Any] | None,
) -> list[Any | None]:
    """Resolve shared, banked, or flattened stack density weights."""
    if density is None:
        return [None] * len(trajectories)
    numpy = import_module("numpy")

    def host(value: Any) -> Any:
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        return numpy.asarray(value)

    counts = [int(numpy.asarray(item).reshape(-1, 2).shape[0]) for item in trajectories]
    if isinstance(density, (list, tuple)):
        if len(density) != len(trajectories):
            raise ValueError("one density array is required per stack trajectory")
        result = [host(item).reshape(-1) for item in density]
    else:
        weights = host(density)
        if weights.size == counts[0] and all(count == counts[0] for count in counts):
            result = [weights.reshape(-1)] * len(trajectories)
        elif (
            weights.ndim >= 2
            and weights.shape[0] == len(trajectories)
            and all(weights[index].size == count for index, count in enumerate(counts))
        ):
            result = [weights[index].reshape(-1) for index in range(len(counts))]
        elif weights.size == sum(counts):
            flattened = weights.reshape(-1)
            if sample_groups is not None:
                result = [flattened[group] for group in sample_groups]
            else:
                result = []
                start = 0
                for count in counts:
                    result.append(flattened[start : start + count])
                    start += count
        else:
            raise ValueError("stacked density does not match the trajectory bank")
    if any(item.size != count for item, count in zip(result, counts, strict=True)):
        raise ValueError("stacked density must have one weight per sample")
    return result


def _stacked_linear_physics(
    mrinufft: Any,
    trajectory: Any,
    image_shape: tuple[int, int, int],
    *,
    coil_maps: Any | None,
    density: Any | None,
    backend: str,
    n_coils: int,
    n_batchs: int,
    z_index: Any,
    viewed_as_real: bool,
    toeplitz_enabled: bool,
    toeplitz_options: dict[str, Any],
    operator_kwargs: dict[str, Any],
) -> tuple[Any, Any]:
    """Build shared-batch or independent plane NUFFTs for a stack."""
    numpy = import_module("numpy")
    trajectories, indices, groups = _stacked_trajectory_bank(
        trajectory,
        z_index,
        image_shape[-1],
    )
    densities = _stacked_density_bank(density, trajectories, groups)
    shared = all(
        numpy.array_equal(trajectories[0], item) for item in trajectories[1:]
    ) and all(
        (densities[0] is None and item is None)
        or (
            densities[0] is not None
            and item is not None
            and numpy.array_equal(densities[0], item)
        )
        for item in densities[1:]
    )

    common = {
        "shape": image_shape[:2],
        "smaps": None,
        "n_batchs": n_batchs,
        "squeeze_dims": False,
        **operator_kwargs,
    }
    if shared:
        native_operators = [
            mrinufft.get_operator(backend)(
                samples=trajectories[0],
                density=densities[0],
                n_coils=n_coils * len(indices),
                **common,
            )
        ]
    else:
        native_operators = [
            mrinufft.get_operator(backend)(
                samples=samples,
                density=weights,
                n_coils=n_coils,
                **common,
            )
            for samples, weights in zip(trajectories, densities, strict=True)
        ]
    plane_physics = [
        _native_linear_physics(native, viewed_as_real=False)
        for native in native_operators
    ]
    operator = _StackedNUFFTLinearPhysics(
        plane_physics,
        native_operators,
        indices,
        image_shape,
        coil_maps=coil_maps,
        n_coils=n_coils,
        viewed_as_real=viewed_as_real,
        toeplitz=toeplitz_enabled,
        toeplitz_options=toeplitz_options,
        shared_operator=shared,
    )
    native_proxy = SimpleNamespace(
        shape=image_shape,
        smaps=coil_maps,
        plane_operators=tuple(native_operators),
        z_index=indices,
        shared_trajectory=shared,
        stacked=True,
    )
    return operator, native_proxy


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
    toeplitz: bool | str | dict[str, Any],
    viewed_as_real: bool,
    streaming: Any | None,
    operator_kwargs: dict[str, Any],
) -> MRIPhysics:
    toeplitz_enabled, best_effort, toeplitz_config = _toeplitz_request(toeplitz)
    if len(image_shape) != spatial_ndim:
        raise ValueError(
            f"image_shape must have {spatial_ndim} entries, got {image_shape!r}"
        )
    trajectory = _single_precision(trajectory)
    density = _single_precision(density)
    coil_maps = _single_precision(coil_maps)
    trajectory_shape = getattr(trajectory, "shape", ())
    if not trajectory_shape and isinstance(trajectory, (list, tuple)) and trajectory:
        trajectory_shape = getattr(trajectory[0], "shape", ())
    trajectory_dim = trajectory_shape[-1] if trajectory_shape else None
    valid_dimensions = {2, 3} if stacked else {spatial_ndim}
    if trajectory_dim is not None and trajectory_dim not in valid_dimensions:
        raise ValueError(
            f"trajectory must end in {sorted(valid_dimensions)} coordinates, "
            f"got {trajectory_dim}"
        )
    if stacked and spatial_ndim != 3:
        raise ValueError("stacked trajectories are only supported by NonCartesian3D")
    map_shape = getattr(coil_maps, "shape", ())
    if coil_maps is not None:
        if len(map_shape) == spatial_ndim + 1:
            inferred_coils = int(map_shape[0])
        elif len(map_shape) == spatial_ndim + 2:
            inferred_coils = int(map_shape[1])
        else:
            raise ValueError(
                "coil_maps must have shape (coils, *image_shape) or "
                "(batch, coils, *image_shape)"
            )
        if n_coils not in {1, inferred_coils}:
            raise ValueError("n_coils conflicts with the sensitivity-map bank")
        n_coils = inferred_coils

    mrinufft = _require_mrinufft()
    backend = _resolve_nufft_backend(backend)
    operator_kwargs = dict(operator_kwargs)
    if backend == "cufinufft-torch" and "gpu_device_id" not in operator_kwargs:
        selected_device = (
            streaming.torch_device
            if streaming is not None and hasattr(streaming, "torch_device")
            else getattr(trajectory, "device", None)
        )
        if getattr(selected_device, "type", None) == "cuda":
            operator_kwargs["gpu_device_id"] = selected_device.index or 0
    native_coil_maps = coil_maps
    if (
        backend == "finufft"
        and hasattr(coil_maps, "detach")
        and getattr(coil_maps.device, "type", None) == "cpu"
    ):
        native_coil_maps = coil_maps.detach().numpy()
    frame_stacked = not stacked and streaming is not None and len(trajectory_shape) >= 4
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
    elif (
        density is not None
        and len(trajectory_shape) >= 3
        and len(density_shape) > 1
        and prod(density_shape) == prod(trajectory_shape[:-1])
    ):
        # The operator plans on every frame's samples at once, so a density
        # given one row per frame is flat to it.
        native_density = density.reshape(-1)
    if stacked:
        operator, native = _stacked_linear_physics(
            mrinufft,
            native_trajectory,
            image_shape,
            coil_maps=coil_maps,
            density=native_density,
            backend=backend,
            n_coils=n_coils,
            n_batchs=n_batchs,
            z_index=z_index,
            viewed_as_real=viewed_as_real,
            toeplitz_enabled=toeplitz_enabled,
            toeplitz_options=toeplitz_config,
            operator_kwargs=operator_kwargs,
        )
    else:
        native = mrinufft.get_operator(backend)(
            samples=native_trajectory,
            shape=image_shape,
            smaps=native_coil_maps,
            density=native_density,
            n_coils=n_coils,
            n_batchs=n_batchs,
            squeeze_dims=False,
            **operator_kwargs,
        )
        _remember_plan_settings(native, operator_kwargs)
        operator = _native_linear_physics(native, viewed_as_real=viewed_as_real)
        operator = _configure_base_toeplitz(
            operator,
            native,
            enabled=toeplitz_enabled,
            best_effort=best_effort,
            options=toeplitz_config,
        )

    def rebuild(
        new_trajectory: Any,
        frame_index: int | None = None,
    ) -> MRIPhysics:
        frame_density = _frame_density(
            density,
            trajectory,
            frame_index,
            prod(getattr(new_trajectory, "shape", (0,))[:-1]),
        )
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

    def replicate(device: Any, device_policy: Any) -> MRIPhysics:
        replica_kwargs = dict(operator_kwargs)
        replica_kwargs["gpu_device_id"] = device.index
        return _noncartesian(
            trajectory,
            image_shape,
            spatial_ndim=spatial_ndim,
            coil_maps=coil_maps,
            density=density,
            backend=backend,
            n_coils=n_coils,
            n_batchs=n_batchs,
            stacked=stacked,
            z_index=z_index,
            toeplitz=toeplitz,
            viewed_as_real=viewed_as_real,
            streaming=device_policy,
            operator_kwargs=replica_kwargs,
        )

    if backend == "cufinufft-torch":
        result._replicate = replicate
    if streaming is not None:
        result.enable_streaming(streaming)
    return result


class NonCartesian2D(MRIPhysics):
    """Two-dimensional non-Cartesian MRI physics, over MRI-NUFFT.

    Parameters
    ----------
    trajectory
        K-space coordinates shaped ``(samples, 2)`` or ``(shots, samples, 2)``,
        in MRI-NUFFT's ``[-0.5, 0.5)`` units.
    image_shape
        Reconstructed matrix, ``(h, w)``.
    coil_maps
        Complex sensitivities shaped ``(coils, h, w)``. ``None`` encodes a
        single channel.
    density
        Density-compensation weights, or a name MRI-NUFFT recognises. See
        :func:`pulserver.mrd.pipe_menon_dcf`.
    backend
        MRI-NUFFT backend. ``"auto"`` picks FINUFFT on CPU and Pulserver's
        Torch-native CUFINUFFT adapter on a CUDA host.
    n_coils, n_batchs
        Coil and batch counts the backend plans for.
    toeplitz
        How the normal operator is computed. ``"auto"``, the default, builds a
        transfer kernel on the first normal-operator call -- exact, and what
        makes an iterative solve worth running -- and falls back to the plain
        adjoint-of-forward for a shape no kernel can embed. ``False`` is the
        plain one outright, ``True`` insists on the kernel, and a dict is the
        kernel with these options.
    viewed_as_real
        Exchange images and measurements through real views.
    streaming
        Optional :class:`pulserver.recon.execution.CudaStreaming` policy.
    **kwargs
        Forwarded to the MRI-NUFFT operator.

    Notes
    -----
    Images are ``(batch, 2, h, w)``, measurements ``(batch, coils, k, 2)``.

    Examples
    --------
    >>> import numpy as np
    >>> import torch
    >>> import pulserver.recon as recon
    >>> angles = np.linspace(0, np.pi, 8, endpoint=False)
    >>> radius = np.linspace(-0.5, 0.5, 32)
    >>> trajectory = np.stack(
    ...     [np.outer(np.cos(angles), radius), np.outer(np.sin(angles), radius)], -1
    ... ).reshape(-1, 2)

    With no maps the adjoint grids each coil onto the image matrix, which is
    what a density-compensated first estimate is made of:

    >>> physics = recon.NonCartesian2D(trajectory, (16, 16))
    >>> physics.A_adjoint(torch.ones(1, 1, 256, dtype=torch.complex64)).shape
    torch.Size([1, 1, 16, 16])

    Golden-angle spokes, gridded and then solved. The adjoint needs the
    density compensation because the samples crowd the centre; the solve does
    not, because the operator's normal equations already account for it:

    .. plot::

       import numpy as np
       import torch
       import pulserver.recon as recon
       import pulserver.mrd as mrd
       from _figures import images, phantom

       truth, coil_maps = phantom(64, coils=4)
       angles = np.pi * (np.arange(48) * 0.618034 % 1.0)
       radius = np.linspace(-0.5, 0.5, 128)
       trajectory = np.stack(
           [np.outer(np.cos(angles), radius), np.outer(np.sin(angles), radius)], -1
       ).reshape(-1, 2)

       physics = recon.NonCartesian2D(trajectory, (64, 64), coil_maps=coil_maps)
       measured = physics.A(truth)
       weights = torch.as_tensor(
           np.asarray(mrd.pipe_menon_dcf(trajectory, (64, 64))), dtype=torch.complex64
       )
       images(
           [
               ("object", truth),
               ("density-compensated adjoint", physics.A_adjoint(measured * weights)),
               ("CG-SENSE", recon.pics(measured, physics, iterations=15)),
           ],
           title="NonCartesian2D, 48 golden-angle spokes",
       )
    """

    def __init__(
        self,
        trajectory: Any,
        image_shape: tuple[int, int],
        *,
        coil_maps: Any | None = None,
        density: Any | None = None,
        backend: str = "auto",
        n_coils: int = 1,
        n_batchs: int = 1,
        toeplitz: bool | str | dict[str, Any] = "auto",
        viewed_as_real: bool = False,
        streaming: Any | None = None,
        **kwargs: Any,
    ) -> None:
        base = _noncartesian(
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
        enabled, best_effort, options = _toeplitz_request(toeplitz)
        if enabled:
            _enable_toeplitz(base, best_effort=best_effort, **options)
        _init_from(self, base)


class NonCartesian3D(MRIPhysics):
    """Three-dimensional or stack-of-NUFFTs MRI physics.

    With ``stacked=True``, one 2D trajectory array is batched across selected
    stack-frequency planes. A Python sequence supplies independent plane
    trajectories; a 3D-coordinate trajectory is grouped by its Cartesian
    stack coordinate. Shared and plane-specific density layouts follow the
    same convention.

    Parameters
    ----------
    trajectory
        K-space coordinates ending in three components, or -- under
        ``stacked`` -- one 2D trajectory or a sequence of per-plane ones.
    image_shape
        Reconstructed matrix, ``(d, h, w)``.
    coil_maps
        Complex sensitivities shaped ``(coils, d, h, w)``.
    density
        Density-compensation weights, shared or per plane.
    backend
        MRI-NUFFT backend.
    n_coils, n_batchs
        Coil and batch counts the backend plans for.
    stacked
        Encode a stack of 2D trajectories rather than a full 3D one.
    z_index
        Stack-frequency planes to encode. ``"auto"`` takes them from the
        trajectory.
    toeplitz
        How the normal operator is computed. ``"auto"``, the default, builds a
        transfer kernel on the first normal-operator call -- exact, and what
        makes an iterative solve worth running -- and falls back to the plain
        adjoint-of-forward for a shape no kernel can embed. ``False`` is the
        plain one outright, ``True`` insists on the kernel, and a dict is the
        kernel with these options.
    viewed_as_real
        Exchange images and measurements through real views.
    streaming
        Optional :class:`pulserver.recon.execution.CudaStreaming` policy.
    **kwargs
        Forwarded to the MRI-NUFFT operator.

    Notes
    -----
    Images are ``(batch, 2, d, h, w)``, measurements ``(batch, coils, k, 2)``.

    Examples
    --------
    A koosh ball is a projection scan: every spoke starts at the centre, so the
    adjoint alone is heavily weighted there and needs the density compensation a
    solve applies for it.

    .. plot::

       import matplotlib.pyplot as plt
       import pulserver.recon as recon
       from _figures import images, koosh_spokes, volume

       truth, coil_maps = volume(24, coils=4, depth=24)
       trajectory = koosh_spokes(24, 60)

       physics = recon.NonCartesian3D(trajectory, (24, 24, 24), coil_maps=coil_maps[0])
       measured = physics.A(truth)
       adjoint = physics.A_adjoint(measured)
       solved = recon.pics(measured, physics, iterations=10)

       middle = 12
       images(
           [
               ("truth", truth[0, middle]),
               ("adjoint", adjoint[0, 0, middle]),
               ("CG-SENSE, 10 iterations", solved[0, 0, middle]),
           ],
           title="NonCartesian3D over a 60-spoke koosh ball",
       )
    """

    def __init__(
        self,
        trajectory: Any,
        image_shape: tuple[int, int, int],
        *,
        coil_maps: Any | None = None,
        density: Any | None = None,
        backend: str = "auto",
        n_coils: int = 1,
        n_batchs: int = 1,
        stacked: bool = False,
        z_index: Any = "auto",
        toeplitz: bool | str | dict[str, Any] = "auto",
        viewed_as_real: bool = False,
        streaming: Any | None = None,
        **kwargs: Any,
    ) -> None:
        base = _noncartesian(
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
        enabled, best_effort, options = _toeplitz_request(toeplitz)
        if enabled:
            _enable_toeplitz(base, best_effort=best_effort, **options)
        _init_from(self, base)


class _FlatSubspaceEncoding:
    """Encode a dynamic acquisition through one plan over every sample.

    A subspace acquisition is a NUFFT per frame only if you insist on frames.
    The transform is linear in the data, so weighting a frame's samples by its
    basis coefficient and gridding the whole trajectory at once gives the same
    answer with one plan instead of one per frame -- and without accumulating a
    volume per frame, which is what a frame-at-a-time adjoint really spends.

    Sample sets that large are held one group of coils at a time: the data for
    every coil at once is the largest array in a reconstruction, and there is
    no reason for a second one beside it.
    """

    def __init__(self, physics: MRIPhysics, trajectory: Any) -> None:
        self.physics = physics
        shape = tuple(int(size) for size in trajectory.shape)
        self.n_frames = shape[0]
        self.per_frame = prod(shape[1:-1])
        native = _base_fourier_operator(physics.native_operator)
        self.n_coils = int(getattr(native, "n_coils", 1))
        self.uses_sense = getattr(native, "smaps", None) is not None

    def _chunks(self, reference: Any) -> list[range]:
        """Coil groups sized to what the device has room for."""
        if not self.uses_sense:
            return [range(self.n_coils)]
        torch = import_module("torch")
        per_coil = 8 * self.n_frames * self.per_frame * max(reference.shape[0], 1)
        if reference.device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(reference.device)
            budget = int(0.2 * free)
        else:
            budget = 4 * 1024**3
        width = max(1, min(self.n_coils, budget // max(2 * per_coil, 1)))
        return [
            range(start, min(start + width, self.n_coils))
            for start in range(0, self.n_coils, width)
        ]

    @contextmanager
    def _restricted(self, coils: range) -> Any:
        """The encoding operator seen as carrying only these coils."""
        native = _base_fourier_operator(self.physics.native_operator)
        if not self.uses_sense or len(coils) == self.n_coils:
            yield self.physics
            return
        held = native.smaps
        native.smaps = held[coils.start : coils.stop]
        try:
            yield self.physics
        finally:
            native.smaps = held

    def encode(self, coefficients: Any, basis: Any) -> Any:
        """Measurements ``(batch, frames, coils, samples)`` from coefficients."""
        torch = import_module("torch")
        batch, rank = coefficients.shape[0], coefficients.shape[1]
        weights = basis.to(device=coefficients.device, dtype=coefficients.dtype).conj()
        measurements = None
        for coils in self._chunks(coefficients):
            part = None
            for index in range(rank):
                with self._restricted(coils) as operator:
                    full = operator.A(coefficients[:, index])
                full = full.reshape(batch, len(coils), self.n_frames, self.per_frame)
                scaled = full * weights[index].reshape(1, 1, -1, 1)
                part = scaled if part is None else part + scaled
                del full, scaled
            part = part.transpose(1, 2)
            if measurements is None:
                measurements = torch.empty(
                    (batch, self.n_frames, self.n_coils, self.per_frame),
                    dtype=part.dtype,
                    device=part.device,
                )
            measurements[:, :, coils.start : coils.stop] = part
            del part
        assert measurements is not None
        return measurements

    def decode(self, measurements: Any, basis: Any, rank: int) -> Any:
        """Coefficients ``(batch, rank, *image)`` from measurements."""
        torch = import_module("torch")
        batch = measurements.shape[0]
        weights = basis.to(device=measurements.device, dtype=measurements.dtype)
        coefficients = None
        for coils in self._chunks(measurements):
            block = measurements[:, :, coils.start : coils.stop]
            for index in range(rank):
                weighted = block * weights[index].reshape(1, -1, 1, 1)
                weighted = weighted.transpose(1, 2).reshape(batch, len(coils), -1)
                with self._restricted(coils) as operator:
                    image = operator.A_adjoint(weighted)
                del weighted
                image = image.reshape(batch, *image.shape[-self.physics.spatial_ndim :])
                if coefficients is None:
                    coefficients = torch.zeros(
                        (batch, rank, *image.shape[1:]),
                        dtype=image.dtype,
                        device=image.device,
                    )
                coefficients[:, index] += image
                del image
            del block
        assert coefficients is not None
        return coefficients


def _subspace_linear_physics(
    frame_physics: Sequence[MRIPhysics | _LazyFramePhysics],
    basis: Any,
    *,
    viewed_as_real: bool,
    toeplitz_config: dict[str, Any] | None,
    flat_encoding: _FlatSubspaceEncoding | None = None,
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
            self.__dict__["flat_encoding"] = flat_encoding
            self.__dict__["spatial_rank"] = int(
                getattr(frame_physics[0], "spatial_ndim", 2)
            )
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
            if all(item.kind.startswith("cartesian") for item in frame_physics):
                self._compact_toeplitz = "cartesian-subspace"
            elif any("stacked" in item.modifiers for item in frame_physics):
                # Each frame already owns a compact plane-kernel bank. The
                # general subspace loop composes those exact normals without
                # materializing a dense rank-by-rank 3D transfer.
                self._compact_toeplitz = None
            elif any(isinstance(item, _LazyFramePhysics) for item in frame_physics):
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
                    _enable_toeplitz(item, **options)
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
                    else:
                        frame = frame[:, 0]
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
            if self.flat_encoding is not None:
                return self.flat_encoding.encode(coefficients, self.basis)
            frames = self._expand(coefficients)
            measurements = []
            for index, physics in enumerate(self.frame_physics):
                frame = frames[:, index : index + 1]
                if physics.viewed_as_real:
                    frame = self._image_as_real(frame)
                else:
                    # A complex frame physics takes the image without the
                    # subspace coefficient axis, ``(batch, *spatial)``.
                    frame = frame[:, 0]
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
                    else:
                        frame = frame.reshape(
                            frame.shape[0], 1, *frame.shape[-self.spatial_rank :]
                        )
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
            if self.flat_encoding is not None:
                coefficients = self.flat_encoding.decode(
                    y,
                    self.basis,
                    int(self.basis.shape[0]),
                )
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
                else:
                    # One coefficient axis, whether or not the frame physics
                    # answered with a coil axis of its own.
                    frame = frame.reshape(
                        frame.shape[0], 1, *frame.shape[-self.spatial_rank :]
                    )
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
                        else _frame_coil_view(self.frame_physics[0])
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
                        _frame_coil_view(self.frame_physics[0]),
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
                        normal = self._image_as_cpx(normal)
                    else:
                        normal = frame_physics_item.A_adjoint_A(frame[:, 0])[:, None]
                    result += basis_cpu[:, index].reshape(
                        1, -1, *([1] * (normal.ndim - 2))
                    ) * normal.to("cpu")
                return self._image_as_real(result) if self.viewed_as_real else result
            frames = self._expand(coefficients)
            normal_frames = []
            for index, physics in enumerate(self.frame_physics):
                frame = frames[:, index : index + 1]
                if physics.viewed_as_real:
                    frame = self._image_as_real(frame)
                    normal = physics.A_adjoint_A(frame)
                    normal = self._image_as_cpx(normal)
                else:
                    normal = physics.A_adjoint_A(frame[:, 0])[:, None]
                normal_frames.append(normal)
            result = self._project(torch.cat(normal_frames, dim=1))
            return self._image_as_real(result) if self.viewed_as_real else result

    return _SubspaceLinearPhysics()


def _subspace(
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
    flat_encoding: _FlatSubspaceEncoding | None = None
    trajectory = physics.trajectory
    trajectory_shape = getattr(trajectory, "shape", ())
    if (
        trajectory is not None
        and len(trajectory_shape) >= 3
        and trajectory_shape[0] == n_frames
    ):
        provider = _FramePhysicsProvider(physics, trajectory, streaming)
        frame_physics = [
            _LazyFramePhysics(provider, index) for index in range(n_frames)
        ]
        native = _base_fourier_operator(physics.native_operator)
        if (
            physics.kind.startswith("noncartesian")
            and "stacked" not in physics.modifiers
            and not physics.viewed_as_real
            and native is not None
            and not hasattr(native, "B")
        ):
            flat_encoding = _FlatSubspaceEncoding(physics, trajectory)
    else:
        frame_physics = [physics] * n_frames

    operator = _subspace_linear_physics(
        frame_physics,
        basis,
        viewed_as_real=physics.viewed_as_real,
        toeplitz_config=physics.toeplitz_options,
        flat_encoding=flat_encoding,
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


class Subspace(MRIPhysics):
    """Subspace encoding composed with frame-wise MRI physics.

    Solves for a small number of temporal coefficients instead of one image
    per frame, with the base physics applied to each expanded frame.

    Parameters
    ----------
    physics
        Base MRI physics, applied per frame.
    basis
        Temporal basis shaped ``(rank, frames)`` -- rank first. Rows are the
        retained singular vectors of a signal dictionary.
    **kwargs
        Forwarded to the base physics wrapper.

    Notes
    -----
    Coefficient images are native complex, ``(batch, rank, *image_shape)`` --
    one complex channel per retained coefficient.

    Examples
    --------
    >>> import torch
    >>> from pulserver.recon.physics import Cartesian2D, Subspace
    >>> base = Cartesian2D(
    ...     torch.ones(1, 1, 8, 8),
    ...     torch.ones(1, 2, 8, 8, dtype=torch.complex64) / 2 ** 0.5,
    ... )
    >>> physics = Subspace(base, torch.randn(3, 5, dtype=torch.complex64))
    >>> physics.A(torch.randn(1, 3, 8, 8, dtype=torch.complex64)).shape
    torch.Size([1, 5, 2, 8, 8])
    """

    def __init__(self, physics: MRIPhysics, basis: Any, **kwargs: Any) -> None:
        _init_from(self, _subspace(physics, basis, **kwargs))


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


def _host_array(value: Any) -> Any:
    """A field model as NumPy, contiguous, for the interpolation to plan on.

    The decomposition the interpolator runs hands back a reversed view, which
    is not something a Torch array can wrap. Planning on the host settles it;
    the factors it produces are moved to the operator afterwards.
    """
    if value is None:
        return None
    numpy = import_module("numpy")
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach().cpu().numpy()
    return numpy.ascontiguousarray(value)


@cache
def _torch_corrected_class() -> Any:
    """mri-nufft's off-resonance operator, summed in Torch.

    The correction itself is upstream's -- the field model, the interpolators
    fitted from it, the segment count. What is replaced is the two loops that
    add the segments up, because upstream writes them for NumPy and CuPy and
    reaches for CuPy to put them on a GPU. Ours is one Torch operator over
    Torch factors, so the sum runs wherever the samples already are and the
    package needs no second array library to use a card.
    """
    corrected_class = import_module("mrinufft.operators").MRIFourierCorrected
    numpy = import_module("numpy")
    torch = import_module("torch")

    class TorchFourierCorrected(corrected_class):  # type: ignore[misc,valid-type]
        """Off-resonance correction whose segment sum is Torch throughout."""

        def to_torch(self, device: Any) -> None:
            """Hold the interpolators as Torch tensors on ``device``."""

            def held(factor: Any) -> Any:
                if not isinstance(factor, torch.Tensor):
                    factor = torch.as_tensor(numpy.asarray(factor))
                factor = factor.to(torch.complex64, copy=False)
                return factor if device is None else factor.to(device)

            self.B, self.C = held(self.B), held(self.C)

        def _staged(self, factor: Any, like: Any) -> Any:
            return factor.to(device=like.device, dtype=like.dtype)

        def op(self, data: Any, *args: Any) -> Any:
            """Distorted k-space: the segments, weighted and summed."""
            data = torch.as_tensor(data)
            batches, coils = self.n_batchs, self.n_coils
            shots, per_shot = self.n_shots, self.n_samples_per_shot
            kspace = None
            for segment in range(self.n_interpolators):
                spatial = self._staged(self.C[segment], data)
                partial = self._fourier_op.op(spatial * data, *args)
                partial = partial.reshape(batches, coils, shots, per_shot)
                temporal = self._staged(self.B[:, segment], partial)
                weighted = (temporal * partial).reshape(batches, coils, self.n_samples)
                kspace = weighted if kspace is None else kspace + weighted
            return self._safe_squeeze(kspace)

        def adj_op(self, coeffs: Any, *args: Any) -> Any:
            """The adjoint of that sum, segment by conjugated segment."""
            coeffs = torch.as_tensor(coeffs)
            batches, coils = self.n_batchs, self.n_coils
            shots, per_shot = self.n_shots, self.n_samples_per_shot
            folded = coeffs.reshape(batches, coils, shots, per_shot)
            image = None
            for segment in range(self.n_interpolators):
                temporal = self._staged(self.B[:, segment], folded).conj()
                weighted = (temporal * folded).reshape(batches, coils, self.n_samples)
                partial = self._fourier_op.adj_op(weighted, *args)
                spatial = self._staged(self.C[segment], partial).conj()
                contribution = spatial * partial
                image = contribution if image is None else image + contribution
            return self._safe_squeeze(image)

    return TorchFourierCorrected


def _off_resonance(
    physics: MRIPhysics,
    field_map: Any,
    readout_time: Any,
    *,
    r2star_map: Any | None = None,
    mask: Any | None = None,
    interpolator: str | dict[str, Any] | tuple[Any, Any] = "svd",
) -> MRIPhysics:
    """Decorate non-Cartesian physics with mri-nufft off-resonance correction.

    The field model is fitted with a dense decomposition. The partial one is
    an ARPACK routine that answers a reversed view, which is not something a
    Torch array can wrap, and the matrix it decomposes is small enough -- one
    row per readout sample, one column per field bin -- that the dense
    factorization is the simpler thing to depend on.
    """
    if physics.native_operator is None:
        raise TypeError("OffResonance requires base non-Cartesian physics")
    if "stacked" in physics.modifiers:
        raise ValueError(
            "OffResonance for stack-of-NUFFTs needs a stack-frequency field "
            "model and is not yet a valid composition."
        )
    if "subspace" in physics.modifiers:
        raise ValueError(
            "Apply OffResonance before Subspace so field correction occurs "
            "in every acquired frame."
        )
    if "off_resonance" in physics.modifiers:
        raise ValueError("physics already has an off-resonance decorator")
    try:
        _torch_corrected_class()
    except ImportError as error:
        raise ImportError("Off-resonance physics requires mri-nufft.") from error

    corrected_interpolator = interpolator
    if isinstance(interpolator, str):
        corrected_interpolator = {"name": interpolator, "partial_svd": False}
    elif isinstance(interpolator, dict):
        corrected_interpolator = {"partial_svd": False, **interpolator}
    corrected_readout_time = readout_time
    trajectory_shape = getattr(physics.trajectory, "shape", ())
    time_shape = getattr(readout_time, "shape", ())
    frame_samples = prod(trajectory_shape[1:-1]) if len(trajectory_shape) >= 4 else 0
    dynamic_readout = bool(
        frame_samples
        and time_shape
        and prod(time_shape) != frame_samples
        and (
            time_shape[0] == trajectory_shape[0]
            or prod(time_shape) == trajectory_shape[0] * frame_samples
        )
    )
    if physics.streaming_policy is not None and dynamic_readout:
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

    native = _torch_corrected_class()(
        physics.native_operator,
        b0_map=_host_array(field_map),
        readout_time=_host_array(corrected_readout_time),
        r2star_map=_host_array(r2star_map),
        mask=_host_array(mask),
        interpolator=corrected_interpolator,
    )
    native.to_torch(getattr(physics.native_operator, "device", None))
    toeplitz_enabled = "toeplitz" in physics.modifiers
    options = physics.toeplitz_options or _toeplitz_options()
    operator = _native_linear_physics(
        native,
        viewed_as_real=physics.viewed_as_real,
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
        return OffResonance(
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


class OffResonance(MRIPhysics):
    """Multi-frequency-interpolation off-resonance correction.

    Parameters
    ----------
    physics
        Base **non-Cartesian** physics. Off-resonance evolves along the
        readout, which a Cartesian encode samples too briefly for this model
        to be the right correction; reverse-polarity distortion correction
        (:func:`pulserver.recon.postprocessing.run_pyhysco`) is the Cartesian
        EPI route.
    field_map
        Off-resonance in Hz over the image grid.
    readout_time
        Time of every sample relative to the echo, in seconds.
    **kwargs
        Segmentation options forwarded to the interpolation.

    Raises
    ------
    TypeError
        If ``physics`` is Cartesian.

    Examples
    --------
    A long readout accumulates phase wherever the field is off resonance, so a
    spiral or a radial train blurs where the field is wrong. The correction is
    a time-segmented model of that phase: the field map, and the time each
    sample was taken.

    .. plot::

       import numpy as np
       import pulserver.recon as recon
       from _figures import images, phantom, radial_spokes

       truth, coil_maps = phantom(64, coils=4)
       spokes = radial_spokes(64, 48)
       trajectory = np.asarray(spokes).reshape(-1, 2).astype(np.float32)

       # Half the field offset by 500 Hz, over a 16 ms readout.
       field_map = np.zeros((64, 64), dtype=np.float32)
       field_map[:, 32:] = 500.0
       readout_time = np.tile(
           np.linspace(0, 16e-3, np.asarray(spokes).shape[1], dtype=np.float32),
           np.asarray(spokes).shape[0],
       )

       encoding = recon.NonCartesian2D(
           trajectory, (64, 64), coil_maps=coil_maps[0], n_coils=4
       )
       corrected = recon.OffResonance(encoding, field_map, readout_time)
       measured = corrected.A(truth)

       ignored = recon.pics(measured, encoding, iterations=10)[0]
       modelled = recon.pics(measured, corrected, iterations=10)[0]

       images(
           [
               ("truth", truth[0]),
               ("field ignored", ignored),
               ("field modelled", modelled),
           ],
           title="A 16 ms readout through a 500 Hz offset",
       )

    The field model is fitted once, when the operator is built, so a solve pays
    for it only at the start.
    """

    def __init__(
        self,
        physics: MRIPhysics,
        field_map: Any,
        readout_time: Any,
        **kwargs: Any,
    ) -> None:
        _init_from(
            self,
            _off_resonance(physics, field_map, readout_time, **kwargs),
        )


def _enable_toeplitz(
    physics: MRIPhysics,
    *,
    best_effort: bool = False,
    compress: bool = True,
    chunk_size: int = 65536,
    coil_batch_size: int = 1,
    cuda_mode: str = "auto",
    cuda_max_device_fraction: float = 0.85,
    cuda_transfer_precision: str = "auto",
) -> None:
    """Give a physics object a Toeplitz normal operator.

    The kernel is the trajectory gridded onto a grid twice the image in every
    dimension, stored over the locations it reached. Subspace and off-resonance
    decorators carry a matrix-valued transfer built the same way, whose
    Hermitian upper triangle is packed and whose real bases keep real storage.
    An even transfer -- what a trajectory closed under ``k -> -k`` leaves -- is
    stored over half its locations and mirrored as it is applied.

    None of that is a choice. What the arguments settle is execution: how much
    is unpacked at a time, how many coils share a pass, and what a CUDA device
    holds.
    """
    options = _toeplitz_options(
        compress=compress,
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
    elif physics.native_operator is not None:
        physics.operator.use_toeplitz = True
    if best_effort and hasattr(physics.operator, "toeplitz_best_effort"):
        physics.operator.toeplitz_best_effort = True


class Toeplitz(MRIPhysics):
    """A physics object whose normal operator uses a precomputed kernel.

    The non-Cartesian operators already build one, so this is the spelling for
    building it with different options, and for accelerating a physics that
    was made without one.

    Parameters
    ----------
    physics
        Base physics to accelerate.
    **options
        Toeplitz options: ``compress``, ``chunk_size``, ``coil_batch_size``
        and the CUDA transfer settings.

    Examples
    --------
    The kernel is the normal operator, not an approximation of it: a scan's
    transfer gridded onto a doubled grid, so ``A^H A`` becomes pad, transform,
    multiply, transform back, crop -- and gives the same answer the two transforms
    would have.

    >>> import torch
    >>> import pulserver.recon as recon
    >>> import sys; sys.path.insert(0, "docs")
    >>> from _figures import phantom, radial_spokes
    >>> truth, coil_maps = phantom(64, coils=4)
    >>> plain = recon.NonCartesian2D(
    ...     radial_spokes(64, 24), (64, 64), coil_maps=coil_maps[0], toeplitz=False
    ... )
    >>> kernelled = recon.Toeplitz(plain)
    >>> exact = plain.A_adjoint_A(truth)
    >>> through_kernel = kernelled.A_adjoint_A(truth)
    >>> error = torch.linalg.vector_norm(exact - through_kernel)
    >>> bool(error / torch.linalg.vector_norm(exact) < 1e-5)
    True
    """

    def __init__(self, physics: MRIPhysics, **kwargs: Any) -> None:
        _enable_toeplitz(physics, **kwargs)
        _init_from(self, physics)


class WaveShuffling(MRIPhysics):
    """Three-dimensional Wave-Shuffling subspace physics.

    ``sampling`` contains ``(phase, partition, echo)`` indices and ``basis``
    follows Pulserver's ``(rank, echoes)`` convention. The forward and adjoint
    gather/scatter only acquired lines. The normal operator uses the exact
    packed temporal kernel in hybrid k-space and never materializes an echo
    train or a dense ``rank x rank`` field.

    Parameters
    ----------
    sampling
        Acquired ``(phase, partition, echo)`` indices.
    coil_maps
        Complex sensitivities over the reconstructed volume.
    wave_psf
        Wave point-spread function: a tensor, or a
        :class:`pulserver.recon.calibration.WavePSFResult`.
    basis
        Temporal basis shaped ``(rank, echoes)``.
    **kwargs
        Forwarded to the wave-shuffling operator.

    Examples
    --------
    Wave encoding and a temporal subspace at once: the corkscrew separates the
    aliasing, the basis carries the contrast change through the echo train, and
    one solve recovers the coefficient images both describe::

        import pulserver.recon as recon

        physics = recon.WaveShuffling(sampling, coil_maps, wave_psf, basis)
        coefficients = recon.pics(measured, physics)
    """

    def __init__(
        self,
        sampling: Any,
        coil_maps: Any,
        wave_psf: Any,
        basis: Any,
        *,
        line_weights: Any | None = None,
        viewed_as_real: bool = False,
        coil_batch_size: int = 1,
        cuda_transfer_precision: str = "auto",
        streaming: Any | None = None,
    ) -> None:
        operator = _WaveLinearPhysics(
            sampling,
            coil_maps,
            wave_psf,
            basis,
            line_weights=line_weights,
            viewed_as_real=viewed_as_real,
            coil_batch_size=coil_batch_size,
            cuda_transfer_precision=cuda_transfer_precision,
        )
        super().__init__(
            operator,
            native_operator=None,
            kind="wave-shuffling",
            spatial_ndim=3,
            viewed_as_real=viewed_as_real,
            modifiers=("wave", "subspace"),
        )
        if streaming is not None:
            self.enable_streaming(streaming)


class WaveEncoding(MRIPhysics):
    """Wave-CAIPI encoding: a corkscrew gradient spread along the readout.

    Parameters
    ----------
    sampling
        Acquired ``(phase, partition, echo)`` indices, or ``(phase,
        partition)`` for a single-echo scan.
    coil_maps
        Complex sensitivities over the reconstructed volume.
    wave_psf
        Wave point-spread function: a tensor, or a
        :class:`pulserver.recon.calibration.WavePSFResult` from
        :class:`pulserver.recon.calibration.WavePSFCalibration`.
    line_weights
        Optional per-line weights over the acquired samples.
    viewed_as_real
        Exchange images and measurements through real views.
    coil_batch_size
        Coils processed together by the hybrid-space normal operator.
    cuda_transfer_precision
        Precision of host-to-device transfers when streaming.
    streaming
        Optional :class:`pulserver.recon.execution.CudaStreaming` policy.

    Examples
    --------
    Wave encoding plays sinusoidal gradients during the readout, so each line is
    smeared along the encoded axes by a corkscrew point-spread function. Spreading
    the aliasing that way is what lets a higher acceleration still separate::

        import pulserver.recon as recon

        physics = recon.WaveEncoding(sampling, coil_maps, wave_psf)
        image = recon.pics(measured, physics)
    """

    def __init__(
        self,
        sampling: Any,
        coil_maps: Any,
        wave_psf: Any,
        *,
        line_weights: Any | None = None,
        viewed_as_real: bool = False,
        coil_batch_size: int = 1,
        cuda_transfer_precision: str = "auto",
        streaming: Any | None = None,
    ) -> None:
        sampling_tensor = as_torch(sampling)
        echoes = (
            int(sampling_tensor[:, 2].max()) + 1
            if sampling_tensor.ndim == 2
            and sampling_tensor.shape[1] == 3
            and sampling_tensor.shape[0]
            else 1
        )
        basis = as_torch(coil_maps).real.new_ones((1, echoes))
        operator = _WaveLinearPhysics(
            sampling,
            coil_maps,
            wave_psf,
            basis,
            line_weights=line_weights,
            viewed_as_real=viewed_as_real,
            coil_batch_size=coil_batch_size,
            cuda_transfer_precision=cuda_transfer_precision,
        )
        super().__init__(
            operator,
            native_operator=None,
            kind="wave",
            spatial_ndim=3,
            viewed_as_real=viewed_as_real,
            modifiers=("wave",),
        )
        if streaming is not None:
            self.enable_streaming(streaming)
