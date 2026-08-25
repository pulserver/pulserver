"""Private compositional stack-of-NUFFTs physics."""

from __future__ import annotations

__all__: list[str] = []

from math import sqrt
from typing import Any

import deepinv
import torch

from ._views import image_as_cpx as _image_as_cpx
from ._views import image_as_real as _image_as_real
from ._views import kspace_as_cpx as _kspace_as_cpx
from ._views import kspace_as_real as _kspace_as_real


class _StackedNUFFTLinearPhysics(deepinv.physics.LinearPhysics):
    """Compose a Cartesian stack transform with one or more 2D NUFFTs."""

    def __init__(
        self,
        plane_physics: list[Any],
        native_operators: list[Any],
        z_indices: Any,
        image_shape: tuple[int, int, int],
        *,
        coil_maps: Any | None,
        n_coils: int,
        viewed_as_real: bool,
        toeplitz: bool,
        toeplitz_options: dict[str, Any],
        shared_operator: bool,
    ) -> None:
        super().__init__()
        if not plane_physics or len(plane_physics) != len(native_operators):
            raise ValueError("stacked physics requires matching non-empty plane banks")
        if not shared_operator and len(z_indices) != len(plane_physics):
            raise ValueError("z indices and plane NUFFTs do not match")
        self.plane_physics = torch.nn.ModuleList(plane_physics)
        self.__dict__["native_operators"] = tuple(native_operators)
        self.register_buffer(
            "z_indices",
            torch.as_tensor(z_indices, dtype=torch.int64).reshape(-1),
            persistent=False,
        )
        if coil_maps is None:
            self.register_buffer("coil_maps", None, persistent=False)
        else:
            self.register_buffer(
                "coil_maps",
                torch.as_tensor(coil_maps),
                persistent=False,
            )
        self.image_shape = tuple(int(size) for size in image_shape)
        self.n_coils = int(n_coils)
        self.viewed_as_real = viewed_as_real
        self.shared_operator = bool(shared_operator)
        self.use_toeplitz = bool(toeplitz)
        self._toeplitz_options = dict(toeplitz_options)
        self.toeplitz_kernels: dict[int, Any] = {}
        self.streaming_policy = None
        self.streaming_methods = {"A_adjoint_A"}

    def enable_toeplitz(self, options: dict[str, Any]) -> None:
        """Enable the compact normal and invalidate its lazy kernel bank."""
        self.use_toeplitz = True
        self._toeplitz_options = dict(options)
        self.toeplitz_kernels.clear()

    def enable_streaming(self, policy: Any) -> None:
        """Use bounded device workspaces for independent stack planes."""
        self.streaming_policy = policy
        self.toeplitz_kernels.clear()

    def A(self, x: Any, **kwargs: Any) -> Any:
        """Apply the stack FFT followed by the plane NUFFT bank."""
        del kwargs
        image = _image_as_cpx(x) if self.viewed_as_real else x
        planes = self._gather(self._fftz(self._coil_images(image)))
        if self.shared_operator:
            batch, coils, stacks, *spatial = planes.shape
            flattened = planes.reshape(batch, coils * stacks, *spatial)
            measurement = self.plane_physics[0].A(flattened)
            result = measurement.reshape(batch, coils, stacks, -1).flatten(-2)
        else:
            result = torch.cat(
                [
                    physics.A(planes[:, :, index])
                    for index, physics in enumerate(self.plane_physics)
                ],
                dim=-1,
            )
        return _kspace_as_real(result) if self.viewed_as_real else result

    def A_adjoint(self, y: Any, **kwargs: Any) -> Any:
        """Apply the plane adjoints followed by the inverse stack FFT."""
        del kwargs
        measurement = _kspace_as_cpx(y) if self.viewed_as_real else y
        batch, coils = measurement.shape[:2]
        if self.shared_operator:
            samples = self.native_operators[0].n_samples
            planes = (
                self.plane_physics[0]
                .A_adjoint(
                    measurement.reshape(batch, coils * self.z_indices.numel(), samples)
                )
                .reshape(
                    batch,
                    coils,
                    self.z_indices.numel(),
                    *self.image_shape[:2],
                )
            )
        else:
            planes = []
            start = 0
            for native, physics in zip(
                self.native_operators,
                self.plane_physics,
                strict=True,
            ):
                stop = start + native.n_samples
                planes.append(physics.A_adjoint(measurement[..., start:stop]))
                start = stop
            if start != measurement.shape[-1]:
                raise ValueError("stacked k-space sample count does not match physics")
            planes = torch.stack(planes, dim=2)
        coil_images = self._ifftz(self._scatter(planes))
        image = self._combine_coils(coil_images)
        return _image_as_real(image) if self.viewed_as_real else image

    def A_adjoint_A(self, x: Any, **kwargs: Any) -> Any:
        """Apply Pulserver's scalar Toeplitz bank in stack-frequency space."""
        del kwargs
        if not self.use_toeplitz:
            return self.A_adjoint(self.A(x))
        image = _image_as_cpx(x) if self.viewed_as_real else x
        if self.coil_maps is None:
            result = self._normal_coils(image)
        else:
            maps = self._maps_for(image)
            result = torch.zeros_like(image)
            coil_batch = self._toeplitz_options["coil_batch_size"]
            for start in range(0, maps.shape[1], coil_batch):
                selected = maps[:, start : start + coil_batch]
                coils = image * selected
                result += (self._normal_coils(coils) * selected.conj()).sum(
                    dim=1,
                    keepdim=True,
                )
        return _image_as_real(result) if self.viewed_as_real else result

    def _normal_coils(self, coil_images: Any) -> Any:
        planes = self._gather(self._fftz(coil_images))
        if self.shared_operator:
            batch, coils, stacks, *spatial = planes.shape
            flattened = planes.reshape(batch * coils * stacks, 1, *spatial)
            transformed = self._apply_kernel(self.native_operators[0], flattened)
            planes = transformed.reshape(batch, coils, stacks, *spatial)
        else:
            planes = torch.stack(
                [
                    self._apply_kernel(
                        native, planes[:, :, index].flatten(0, 1)[:, None]
                    )
                    .squeeze(1)
                    .unflatten(0, planes.shape[:2])
                    for index, native in enumerate(self.native_operators)
                ],
                dim=2,
            )
        return self._ifftz(self._scatter(planes))

    def _apply_kernel(self, native: Any, image: Any) -> Any:
        key = id(native)
        kernel = self.toeplitz_kernels.get(key)
        if kernel is None:
            from .physics._kernel import _build_scalar_toeplitz

            kernel = _build_scalar_toeplitz(
                native,
                self._toeplitz_options,
                self.streaming_policy,
            )
            self.toeplitz_kernels[key] = kernel
        if self.streaming_policy is not None and image.device.type == "cpu":
            return kernel.apply_streamed(image, self.streaming_policy)
        return kernel.apply(image)

    def _coil_images(self, image: Any) -> Any:
        if self.coil_maps is None:
            if image.shape[1] != self.n_coils:
                raise ValueError(
                    f"calibrationless stacked input needs {self.n_coils} channels"
                )
            return image
        return image * self._maps_for(image)

    def _combine_coils(self, coil_images: Any) -> Any:
        if self.coil_maps is None:
            return coil_images
        return (coil_images * self._maps_for(coil_images).conj()).sum(
            dim=1,
            keepdim=True,
        )

    def _maps_for(self, reference: Any) -> Any:
        maps = self.coil_maps.to(device=reference.device, dtype=reference.dtype)
        if maps.ndim == reference.ndim - 1:
            maps = maps[None]
        if maps.shape[0] == 1:
            return maps.expand(reference.shape[0], *maps.shape[1:])
        if maps.shape[0] != reference.shape[0]:
            raise ValueError("batched sensitivity maps must match the image batch")
        return maps

    def _gather(self, spectrum: Any) -> Any:
        indices = self.z_indices.to(spectrum.device)
        return torch.index_select(spectrum, -1, indices).movedim(-1, 2)

    def _scatter(self, planes: Any) -> Any:
        spectrum = torch.zeros(
            (*planes.shape[:2], *self.image_shape),
            dtype=planes.dtype,
            device=planes.device,
        )
        indices = self.z_indices.to(planes.device)
        spectrum.index_copy_(-1, indices, planes.movedim(2, -1))
        return spectrum

    @staticmethod
    def _fftz(value: Any) -> Any:
        return torch.fft.fftshift(
            torch.fft.fft(
                torch.fft.ifftshift(value, dim=-1),
                dim=-1,
                norm="ortho",
            ),
            dim=-1,
        ) / sqrt(2.0)

    @staticmethod
    def _ifftz(value: Any) -> Any:
        return torch.fft.fftshift(
            torch.fft.ifft(
                torch.fft.ifftshift(value, dim=-1),
                dim=-1,
                norm="ortho",
            ),
            dim=-1,
        ) / sqrt(2.0)
