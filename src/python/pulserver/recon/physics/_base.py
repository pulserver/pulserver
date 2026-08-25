"""The interface every physics model presents to a solver.

:class:`MRIPhysics` is the encoding operator ``A``: it takes an image to the
samples a scan would measure, takes them back, and answers ``A^H A`` by
whichever route the model has been given."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
from typing import Any

import deepinv

from .._views import kspace_as_cpx as _kspace_as_cpx
from .._views import kspace_as_real as _kspace_as_real

from ._common import (
    _cartesian_image_as_cpx,
    _cartesian_image_as_real,
    _measurement_to_channels,
    _measurement_to_trailing,
    _mirror_array_namespace,
)


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
