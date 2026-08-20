"""Private Torch-native MRI-NUFFT CUFINUFFT backend.

The adapter deliberately remains private: users select ``backend="auto"`` or
``backend="cufinufft"`` through :mod:`pulserver.recon.physics`.  It subclasses
MRI-NUFFT's CUFINUFFT operator and raw-plan wrapper so backend registration,
shape conventions, autodiff integration, and the guru-plan lifecycle remain
owned upstream.  Only the CuPy-specific allocation and SENSE loops are
replaced with Torch operations.
"""

from __future__ import annotations

__all__ = ["register_torch_cufinufft"]

from functools import partial
from typing import Any

import numpy as np


def register_torch_cufinufft() -> bool:
    """Register the private backend and return whether CUDA can execute it."""
    try:
        import cufinufft
        import torch
        from mrinufft._utils import proper_trajectory
        from mrinufft.density import get_density
        from mrinufft.operators.base import FourierOperatorBase
        from mrinufft.operators.interfaces.cufinufft import (
            MRICufiNUFFT,
            RawCufinufftPlan,
        )
    except ImportError:
        return False

    registered = FourierOperatorBase.interfaces.get("cufinufft-torch")
    if registered is not None:
        return bool(registered[0])

    class RawTorchCufinufftPlan(RawCufinufftPlan):
        """MRI-NUFFT raw plan using CUFINUFFT's generic CUDA-array API."""

        def _make_plan(self, typ: int, **kwargs: Any) -> None:
            dtype = "complex128" if self._dtype == torch.float64 else "complex64"
            self.plans[typ] = cufinufft.Plan(
                typ,
                self.shape,
                self.n_trans,
                self.eps,
                dtype=dtype,
                **kwargs,
            )

        def _set_pts(self, typ: int | str, samples: Any) -> None:
            plan = self.grad_plan if typ == "grad" else self.plans[typ]
            if plan is None:
                raise ValueError(f"Plan of type {typ} has not been initialized")
            if samples.ndim != 2 or samples.shape[1] != self.ndim:
                raise ValueError(
                    f"samples must have shape (N, {self.ndim}), got "
                    f"{tuple(samples.shape)}"
                )
            coordinates = tuple(
                samples[:, axis].contiguous() for axis in range(self.ndim)
            )
            plan.setpts(*coordinates)

    class MRITorchCufiNUFFT(MRICufiNUFFT):
        """CUFINUFFT MRI operator backed exclusively by Torch CUDA tensors."""

        backend = "cufinufft-torch"
        available = torch.cuda.is_available()
        autograd_available = True

        def __init__(
            self,
            samples: Any,
            shape: tuple[int, ...],
            density: Any = False,
            n_coils: int = 1,
            n_batchs: int = 1,
            smaps: Any = None,
            smaps_cached: bool = True,
            verbose: bool = False,
            squeeze_dims: bool = False,
            n_trans: int = 1,
            **kwargs: Any,
        ) -> None:
            del smaps_cached, verbose
            FourierOperatorBase.__init__(self)
            # Some MRI-NUFFT releases initialize this cache only in concrete
            # backend constructors; this adapter intentionally bypasses the
            # CuPy constructor.
            self._toeplitz_kernel = None
            gpu_device_id = int(kwargs.pop("gpu_device_id", 0))
            self.device = torch.device("cuda", gpu_device_id)
            torch.empty(0, device=self.device)
            source = torch.as_tensor(samples)
            real_dtype = (
                torch.float64 if source.dtype == torch.float64 else torch.float32
            )
            self._samples = proper_trajectory(
                source.to(self.device, dtype=real_dtype),
                normalize="pi",
            ).contiguous()
            self.shape = shape
            self.dtype = np.float64 if real_dtype == torch.float64 else np.float32
            self.n_batchs = n_batchs
            self.n_coils = n_coils
            self.n_trans = int(n_trans)
            self.squeeze_dims = squeeze_dims
            self._plan_kwargs = {"gpu_device_id": gpu_device_id, **kwargs}
            if self.n_trans < 1 or (n_batchs * n_coils) % self.n_trans:
                raise ValueError("n_trans must divide n_batchs*n_coils")
            self.smaps = smaps
            if self.uses_sense and self.n_coils % self.n_trans:
                raise ValueError("n_trans must divide n_coils when using SENSE")
            self.compute_density(density)
            self.raw_op = RawTorchCufinufftPlan(
                self._samples,
                self.shape,
                n_trans=self.n_trans,
                **self._plan_kwargs,
            )

        @property
        def torch_cpx_dtype(self) -> Any:
            return torch.complex128 if self.dtype == np.float64 else torch.complex64

        @FourierOperatorBase.smaps.setter
        def smaps(self, value: Any) -> None:
            if value is None:
                self._smaps = None
                return
            maps = torch.as_tensor(
                value, dtype=self.torch_cpx_dtype, device=self.device
            )
            if tuple(maps.shape[1:]) != self.shape:
                raise ValueError("sensitivity-map shape does not match image shape")
            self.n_coils = maps.shape[0]
            self._smaps = maps.contiguous()

        @FourierOperatorBase.density.setter
        def density(self, value: Any) -> None:
            if value is None:
                self._density = None
                return
            weights = torch.as_tensor(value, device=self.device)
            if weights.numel() != self.n_samples:
                raise ValueError("density and samples must have the same length")
            self._density = weights.reshape(-1).contiguous()

        def compute_density(self, method: Any = None) -> None:
            if method is None or method is False:
                self._density = None
                self._density_method = None
                return
            if isinstance(method, (np.ndarray, torch.Tensor)):
                self.density = method
                self._density_method = None
                return
            kwargs: dict[str, Any] = {}
            if method is True or method == "pipe":
                density_method = get_density("pipe")
                kwargs["backend"] = "finufft"
            elif isinstance(method, dict):
                options = dict(method)
                name = options.pop("name")
                density_method = get_density(name)
                kwargs.update(options)
                if name == "pipe":
                    kwargs.setdefault("backend", "finufft")
            elif isinstance(method, str):
                density_method = get_density(method)
            elif callable(method):
                density_method = method
            else:
                raise ValueError(f"unknown density method {method!r}")
            host_samples = self.samples.detach().cpu().numpy()
            self.density = density_method(host_samples, self.shape, **kwargs)
            self._density_method = partial(density_method, **kwargs)

        def update_samples(self, new_samples: Any, *, unsafe: bool = False) -> None:
            samples = torch.as_tensor(new_samples, device=self.device)
            self._samples = (
                (samples if unsafe else proper_trajectory(samples, normalize="pi"))
                .to(dtype=self.samples.dtype)
                .contiguous()
            )
            if hasattr(self, "raw_op"):
                self.raw_op._set_pts(1, self._samples)
                self.raw_op._set_pts(2, self._samples)
            self._toeplitz_kernel = None

        def _op(self, image: Any, coeffs: Any) -> Any:
            return self.raw_op.type2(image, coeffs)

        def _adj_op(self, coeffs: Any, image: Any) -> Any:
            return self.raw_op.type1(coeffs, image)

        def _input(self, value: Any) -> tuple[Any, bool]:
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value)
            host = value.device.type == "cpu"
            return value.to(self.device, dtype=self.torch_cpx_dtype), host

        def op(self, data: Any, ksp: Any = None) -> Any:
            self.check_shape(image=data)
            data, return_host = self._input(data)
            batch, coils, samples = data.shape[0], self.n_coils, self.n_samples
            if ksp is None:
                ksp = torch.empty(
                    (batch, coils, samples),
                    dtype=self.torch_cpx_dtype,
                    device=self.device,
                )
            else:
                ksp = torch.as_tensor(ksp, device=self.device)
            if self.uses_sense:
                images = data.reshape(batch, *self.shape)
                for batch_index in range(batch):
                    for start in range(0, coils, self.n_trans):
                        stop = start + self.n_trans
                        coil_images = images[batch_index][None] * self.smaps[start:stop]
                        self._op(coil_images, ksp[batch_index, start:stop])
            else:
                images = data.reshape(batch * coils, *self.shape)
                flat = ksp.reshape(batch * coils, samples)
                for start in range(0, batch * coils, self.n_trans):
                    stop = start + self.n_trans
                    self._op(images[start:stop], flat[start:stop])
            ksp.mul_(1.0 / self.norm_factor)
            result = self._safe_squeeze(ksp)
            return result.cpu() if return_host else result

        def adj_op(self, coeffs: Any, image: Any = None) -> Any:
            self.check_shape(ksp=coeffs)
            coeffs, return_host = self._input(coeffs)
            batch, coils, samples = coeffs.shape[0], self.n_coils, self.n_samples
            flat = coeffs.reshape(batch * coils, samples)
            if self.density is not None:
                flat = flat * self.density[None]
            if self.uses_sense:
                result = torch.zeros(
                    (batch, 1, *self.shape),
                    dtype=self.torch_cpx_dtype,
                    device=self.device,
                )
                coil_images = torch.empty(
                    (self.n_trans, *self.shape),
                    dtype=self.torch_cpx_dtype,
                    device=self.device,
                )
                for batch_index in range(batch):
                    for start in range(0, coils, self.n_trans):
                        stop = start + self.n_trans
                        self._adj_op(
                            flat[
                                batch_index * coils + start : batch_index * coils + stop
                            ],
                            coil_images,
                        )
                        result[batch_index, 0].add_(
                            (coil_images * self.smaps[start:stop].conj()).sum(dim=0)
                        )
            else:
                result = torch.empty(
                    (batch * coils, *self.shape),
                    dtype=self.torch_cpx_dtype,
                    device=self.device,
                )
                for start in range(0, batch * coils, self.n_trans):
                    stop = start + self.n_trans
                    self._adj_op(flat[start:stop], result[start:stop])
                result = result.reshape(batch, coils, *self.shape)
            result.mul_(1.0 / self.norm_factor)
            if image is not None:
                torch.as_tensor(image, device=self.device).copy_(result)
            result = self._safe_squeeze(result)
            return result.cpu() if return_host else result

        def compute_toeplitz_kernel(self) -> Any:
            """Compute MRI-NUFFT's circulant embedding with Torch operations."""
            if any(size % 2 for size in self.shape):
                raise ValueError("Toeplitz kernels require even image dimensions")
            weights = (
                self.density.to(self.torch_cpx_dtype)
                if self.density is not None
                else torch.ones(
                    self.n_samples,
                    dtype=self.torch_cpx_dtype,
                    device=self.device,
                )
            )
            plan = RawTorchCufinufftPlan(
                self.samples,
                self.shape,
                n_trans=1,
                **self._plan_kwargs,
            )
            temporary = torch.empty(
                self.shape,
                dtype=self.torch_cpx_dtype,
                device=self.device,
            )

            def adjoint(shifts: tuple[int, ...]) -> Any:
                shift = torch.tensor(
                    shifts,
                    dtype=self.samples.dtype,
                    device=self.device,
                )
                modulated = weights * torch.exp(1j * (self.samples @ shift))
                plan.type1(modulated, temporary)
                return temporary.clone()

            if self.ndim == 2:
                n0, n1 = self.shape
                h0, h1 = n0 // 2, n1 // 2
                kernel = torch.zeros(
                    (2 * n0, 2 * n1),
                    dtype=self.torch_cpx_dtype,
                    device=self.device,
                )
                kernel[:n0, :n1] = adjoint((h0, h1))
                kernel[:n0, n1 + 1 :] = adjoint((h0, -h1))[:, 1:]
                symmetric = torch.roll(
                    torch.roll(kernel.flip((0, 1)), 1, 0), 1, 1
                ).conj()
                kernel[n0 + 1 :] = symmetric[n0 + 1 :]
            elif self.ndim == 3:
                n0, n1, n2 = self.shape
                h0, h1, h2 = n0 // 2, n1 // 2, n2 // 2
                kernel = torch.zeros(
                    (2 * n0, 2 * n1, 2 * n2),
                    dtype=self.torch_cpx_dtype,
                    device=self.device,
                )
                kernel[:n0, :n1, :n2] = adjoint((h0, h1, h2))
                kernel[:n0, :n1, n2 + 1 :] = adjoint((h0, h1, -h2))[:, :, 1:]
                kernel[:n0, n1 + 1 :, :n2] = adjoint((h0, -h1, h2))[:, 1:]
                kernel[:n0, n1 + 1 :, n2 + 1 :] = adjoint((h0, -h1, -h2))[:, 1:, 1:]
                symmetric = torch.roll(
                    torch.roll(torch.roll(kernel.flip((0, 1, 2)), 1, 0), 1, 1),
                    1,
                    2,
                ).conj()
                kernel[n0 + 1 :] = symmetric[n0 + 1 :]
            else:
                raise ValueError("Toeplitz kernels support only 2D and 3D")
            self._toeplitz_kernel = (
                torch.fft.fftn(kernel, norm="ortho").real / self.norm_factor
            )
            return self._toeplitz_kernel

        def _gram_raw(self, image: Any) -> Any:
            if self._toeplitz_kernel is None:
                self.compute_toeplitz_kernel()
            padded = torch.zeros(
                self._toeplitz_kernel.shape,
                dtype=image.dtype,
                device=self.device,
            )
            crop = tuple(slice(0, size) for size in self.shape)
            padded[crop] = image
            padded = torch.fft.fftn(padded, norm="ortho")
            padded.mul_(self._toeplitz_kernel)
            return torch.fft.ifftn(padded, norm="ortho")[crop]

        def gram_op(self, data: Any, image: Any = None, toeplitz: bool = True) -> Any:
            del image
            if not toeplitz:
                return self.adj_op(self.op(data))
            data, return_host = self._input(data)
            batch, coils = data.shape[0], self.n_coils
            if self.uses_sense:
                images = data.reshape(batch, *self.shape)
                result = torch.zeros(
                    (batch, 1, *self.shape),
                    dtype=self.torch_cpx_dtype,
                    device=self.device,
                )
                for batch_index in range(batch):
                    for coil in range(coils):
                        encoded = images[batch_index] * self.smaps[coil]
                        result[batch_index, 0].add_(
                            self._gram_raw(encoded) * self.smaps[coil].conj()
                        )
            else:
                images = data.reshape(batch * coils, *self.shape)
                result = torch.stack([self._gram_raw(item) for item in images])
                result = result.reshape(batch, coils, *self.shape)
            result = self._safe_squeeze(result)
            return result.cpu() if return_host else result

    # Class creation performs MRI-NUFFT's normal backend registration.
    return bool(MRITorchCufiNUFFT.available)
