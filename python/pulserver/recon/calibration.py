"""Nonlinear MRI calibration built from Pulserver physics and optimizers."""

from __future__ import annotations

__all__ = ["NLINV", "NLINVPhysics", "NLINVResult"]

from dataclasses import dataclass
from math import prod
from typing import Any

import torch

import deepinv

from .optim import IRGNM, ConjugateGradient
from .physics import NonCartesian2D, NonCartesian3D


@dataclass(frozen=True)
class NLINVResult:
    """Outputs of an NLINV calibration.

    Parameters
    ----------
    sensitivities
        RSS-normalized coil sensitivity maps.
    image
        Reconstructed image with the sensitivity scaling absorbed.
    calibration
        Synthesized Cartesian calibration k-space on the solve matrix.
    """

    sensitivities: torch.Tensor
    image: torch.Tensor
    calibration: torch.Tensor


class NLINVPhysics(deepinv.physics.Physics):
    r"""Joint nonlinear image and coil-sensitivity physics.

    The state contains one image channel followed by one Sobolev-coordinate
    channel per coil. It is exposed as paired real channels by default, which
    gives DeepInverse and :mod:`torch.func` an unambiguous real Jacobian while
    all MRI operations remain complex64 internally.

    Parameters
    ----------
    encoding
        Coil-preserving linear Fourier physics. Its input has shape
        ``(batch, coils, *image_shape)``.
    sobolev_weights
        Real Fourier-domain smoothing weights over ``image_shape``.
    n_coils
        Number of receive coils.
    viewed_as_real
        Expose joint states and measurements through real views.
    """

    def __init__(
        self,
        encoding: deepinv.physics.LinearPhysics,
        sobolev_weights: torch.Tensor,
        n_coils: int,
        *,
        viewed_as_real: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(encoding, deepinv.physics.LinearPhysics):
            raise TypeError("encoding must be a DeepInverse LinearPhysics")
        weights = torch.as_tensor(sobolev_weights, dtype=torch.float32)
        if weights.ndim not in {2, 3} or any(size <= 0 for size in weights.shape):
            raise ValueError("sobolev_weights must describe a 2D or 3D matrix")
        if n_coils <= 0:
            raise ValueError("n_coils must be positive")
        self.encoding = encoding
        self.register_buffer("sobolev_weights", weights.contiguous())
        self.n_coils = int(n_coils)
        self.image_shape = tuple(int(size) for size in weights.shape)
        self.viewed_as_real = bool(viewed_as_real)

    def A(self, state: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Apply the nonlinear map ``F(rho, c_i) = E(rho * c_i)``."""
        del kwargs
        image, coils = self.physical(state)
        result = self.encoding.A(image * coils)
        return torch.view_as_real(result) if self.viewed_as_real else result

    def A_jvp(
        self,
        state: torch.Tensor,
        vector: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the analytic Jacobian-vector product."""
        image, coils = self.physical(state)
        delta_image, delta_coefficients = self._split_state(vector)
        delta_coils = self._sobolev_forward(delta_coefficients)
        result = self.encoding.A(image * delta_coils + delta_image * coils)
        return torch.view_as_real(result) if self.viewed_as_real else result

    def A_vjp(
        self,
        state: torch.Tensor,
        vector: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the analytic real vector-Jacobian product."""
        if self.viewed_as_real:
            vector = torch.view_as_complex(vector.contiguous())
        image, coils = self.physical(state)
        coil_images = self.encoding.A_adjoint(vector)
        image_gradient = (coil_images * coils.conj()).sum(dim=1, keepdim=True)
        coil_gradient = self._sobolev_adjoint(coil_images * image.conj())
        return self._join_state(image_gradient, coil_gradient)

    def linearize(
        self,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, deepinv.physics.LinearPhysics]:
        """Return the prediction and an analytic matrix-free Jacobian."""
        return self.A(state), _NLINVJacobian(self, state)

    def physical(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the image and physical smooth coil maps from a joint state."""
        image, coefficients = self._split_state(state)
        return image, self._sobolev_forward(coefficients)

    def initial_state(self, batch_size: int, *, device: Any) -> torch.Tensor:
        """Return the standard unit-image, zero-coil initialization."""
        state = torch.zeros(
            (batch_size, self.n_coils + 1, *self.image_shape),
            dtype=torch.complex64,
            device=device,
        )
        state[:, 0] = 1.0
        return self._join_state(state[:, :1], state[:, 1:])

    # %% private module subroutines

    def _split_state(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.viewed_as_real:
            batch, channels, *spatial = state.shape
            if channels != 2 * (self.n_coils + 1):
                raise ValueError("joint real-view state has the wrong channel count")
            state = state.reshape(
                batch,
                self.n_coils + 1,
                2,
                *spatial,
            ).movedim(2, -1)
            state = torch.view_as_complex(state.contiguous())
        if (
            state.shape[1] != self.n_coils + 1
            or tuple(state.shape[2:]) != self.image_shape
        ):
            raise ValueError("joint state shape does not match NLINV physics")
        return state[:, :1], state[:, 1:]

    def _join_state(
        self,
        image: torch.Tensor,
        coefficients: torch.Tensor,
    ) -> torch.Tensor:
        state = torch.cat([image, coefficients], dim=1)
        if not self.viewed_as_real:
            return state
        batch, channels, *spatial = state.shape
        state = torch.view_as_real(state).movedim(-1, 2)
        return state.reshape(batch, 2 * channels, *spatial)

    def _sobolev_forward(self, coefficients: torch.Tensor) -> torch.Tensor:
        axes = tuple(range(-len(self.image_shape), 0))
        weighted = self.sobolev_weights.to(coefficients.dtype) * coefficients
        return _ifftc(weighted, axes)

    def _sobolev_adjoint(self, coils: torch.Tensor) -> torch.Tensor:
        axes = tuple(range(-len(self.image_shape), 0))
        return self.sobolev_weights.to(coils.dtype).conj() * _fftc(coils, axes)


class NLINV(torch.nn.Module):
    """Estimate coil sensitivities with IRGNM-CG.

    The class owns only NLINV setup and post-processing. Nonlinear iteration
    is delegated to :class:`pulserver.recon.optim.IRGNM`, its linearized systems
    to :class:`pulserver.recon.optim.ConjugateGradient`, and Fourier normals to
    the selected Pulserver physics (including non-Cartesian Toeplitz normals).

    Parameters
    ----------
    spatial_ndim
        Cartesian or non-Cartesian spatial dimensionality.
    calibration_width
        Centered calibration width. ``None`` uses the full input matrix.
    max_iter
        Number of IRGNM outer iterations.
    cg_max_iter, cg_rtol
        Inner implicit-CG accuracy.
    damping, damping_decay
        Initial Tikhonov damping and geometric decay.
    sobolev_width, sobolev_degree
        Smooth-coil Sobolev filter parameters.
    toeplitz
        Enable the mri-nufft Toeplitz normal for non-Cartesian calibration.
    backend
        MRI-NUFFT backend. ``"auto"`` selects FINUFFT or Torch CUFINUFFT.
    """

    def __init__(
        self,
        *,
        spatial_ndim: int = 2,
        calibration_width: int | None = 24,
        max_iter: int = 8,
        cg_max_iter: int = 12,
        cg_rtol: float = 1e-2,
        damping: float = 1.0,
        damping_decay: float = 2.0 / 3.0,
        sobolev_width: float = 200.0,
        sobolev_degree: int = 32,
        toeplitz: bool = True,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        if spatial_ndim not in {2, 3}:
            raise ValueError("spatial_ndim must be 2 or 3")
        if calibration_width is not None and calibration_width <= 0:
            raise ValueError("calibration_width must be positive or None")
        if max_iter <= 0 or cg_max_iter <= 0:
            raise ValueError("iteration counts must be positive")
        if cg_rtol < 0.0:
            raise ValueError("cg_rtol must be non-negative")
        if damping < 0.0 or not 0.0 < damping_decay <= 1.0:
            raise ValueError("damping must be non-negative and decay must be in (0, 1]")
        if sobolev_width <= 0.0 or sobolev_degree <= 0:
            raise ValueError("Sobolev width and degree must be positive")
        self.spatial_ndim = spatial_ndim
        self.calibration_width = calibration_width
        self.max_iter = max_iter
        self.cg_max_iter = cg_max_iter
        self.cg_rtol = float(cg_rtol)
        self.damping = float(damping)
        self.damping_decay = float(damping_decay)
        self.sobolev_width = float(sobolev_width)
        self.sobolev_degree = int(sobolev_degree)
        self.toeplitz = bool(toeplitz)
        self.backend = backend

    def forward(
        self,
        kspace: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        trajectory: torch.Tensor | None = None,
        image_shape: tuple[int, ...] | None = None,
        density: torch.Tensor | None = None,
        encoding: deepinv.physics.LinearPhysics | None = None,
        return_info: bool = False,
    ) -> torch.Tensor | NLINVResult:
        """Estimate sensitivity maps from Cartesian or non-Cartesian data.

        Parameters
        ----------
        kspace
            Complex data with shape ``(..., coils, *samples)``.
        mask
            Cartesian sampling mask. It is inferred from nonzero data by
            default.
        trajectory
            Non-Cartesian coordinates ending in ``spatial_ndim``.
        image_shape
            Reconstructed matrix, required for non-Cartesian data.
        density
            Optional density/preconditioning weights passed to MRI-NUFFT.
        encoding
            Optional coil-preserving custom DeepInverse linear physics.
        return_info
            Return image and synthesized calibration data with the maps.

        Returns
        -------
        torch.Tensor or NLINVResult
            RSS-normalized sensitivities, or all calibration outputs.
        """
        if not isinstance(kspace, torch.Tensor) or not kspace.is_complex():
            raise TypeError("kspace must be a complex torch.Tensor")
        if kspace.dtype != torch.complex64:
            kspace = kspace.to(torch.complex64)

        prepared = self._prepare(
            kspace,
            mask=mask,
            trajectory=trajectory,
            image_shape=image_shape,
            density=density,
            encoding=encoding,
        )
        data, selected_encoding, solve_shape, output_shape, leading_shape = prepared
        batch, n_coils = data.shape[:2]
        weights = _sobolev_weights(
            solve_shape,
            output_shape,
            self.sobolev_width,
            self.sobolev_degree,
            device=data.device,
        )
        physics = NLINVPhysics(selected_encoding, weights, n_coils)
        scale = 100.0 / torch.linalg.vector_norm(
            data.flatten(start_dim=1),
            dim=1,
        ).clamp_min(1e-12)
        scale_view = scale.reshape(batch, *([1] * (data.ndim - 1)))
        scaled_data = torch.view_as_real(data * scale_view)
        initialization = physics.initial_state(batch, device=data.device)
        damping = tuple(
            self.damping * self.damping_decay**index for index in range(self.max_iter)
        )
        solver = IRGNM(
            ConjugateGradient(
                max_iter=self.cg_max_iter,
                rtol=self.cg_rtol,
                batch_dim=0,
            ),
            damping=damping,
            max_iter=self.max_iter,
        )
        state = solver(scaled_data, physics, init=initialization)
        image, coils = physics.physical(state)
        calibration = _fftc(
            image * coils,
            tuple(range(-self.spatial_ndim, 0)),
        )
        calibration = calibration / scale.reshape(
            batch,
            *([1] * (calibration.ndim - 1)),
        )

        image = _resize_fourier_image(image, output_shape)
        coils = _resize_fourier_image(coils, output_shape)
        rss = coils.abs().square().sum(dim=1, keepdim=True).sqrt().clamp_min(1e-10)
        sensitivities = coils / rss
        reconstructed = (
            image
            * rss
            / scale.reshape(
                batch,
                *([1] * (image.ndim - 1)),
            )
        )

        sensitivities = _restore_leading(
            sensitivities,
            leading_shape,
            (n_coils, *output_shape),
        )
        reconstructed = _restore_leading(
            reconstructed[:, 0],
            leading_shape,
            output_shape,
        )
        calibration = _restore_leading(
            calibration,
            leading_shape,
            (n_coils, *solve_shape),
        )
        if not return_info:
            return sensitivities
        return NLINVResult(
            sensitivities=sensitivities,
            image=reconstructed,
            calibration=calibration,
        )

    # %% private module subroutines

    def _prepare(
        self,
        kspace: torch.Tensor,
        *,
        mask: torch.Tensor | None,
        trajectory: torch.Tensor | None,
        image_shape: tuple[int, ...] | None,
        density: torch.Tensor | None,
        encoding: deepinv.physics.LinearPhysics | None,
    ) -> tuple[
        torch.Tensor,
        deepinv.physics.LinearPhysics,
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
    ]:
        if encoding is not None:
            if image_shape is None:
                raise ValueError("image_shape is required with a custom encoding")
            output_shape = tuple(map(int, image_shape))
            leading_shape, data = _flatten_batches(
                kspace,
                self.spatial_ndim + 1,
            )
            return data, encoding, output_shape, output_shape, leading_shape
        if trajectory is None:
            output_shape = tuple(
                int(size) for size in kspace.shape[-self.spatial_ndim :]
            )
            leading_shape, data = _flatten_batches(kspace, self.spatial_ndim + 1)
            solve_shape = tuple(
                min(self.calibration_width, size)
                if self.calibration_width is not None
                else size
                for size in output_shape
            )
            data = _resize_centered(data, solve_shape)
            if mask is None:
                selected_mask = data.abs().square().sum(dim=(0, 1)) > 0
            else:
                selected_mask = _resize_centered(
                    torch.as_tensor(mask, device=data.device),
                    solve_shape,
                )
                if selected_mask.ndim > self.spatial_ndim:
                    selected_mask = selected_mask.reshape(
                        -1,
                        *solve_shape,
                    )[0]
            selected_encoding = _CartesianCalibrationEncoding(selected_mask)
            return data, selected_encoding, solve_shape, output_shape, leading_shape

        if image_shape is None or len(image_shape) != self.spatial_ndim:
            raise ValueError("image_shape is required for non-Cartesian NLINV")
        output_shape = tuple(map(int, image_shape))
        trajectory = torch.as_tensor(
            trajectory, device=kspace.device, dtype=torch.float32
        )
        if trajectory.shape[-1] != self.spatial_ndim:
            raise ValueError(
                "trajectory coordinate dimension does not match spatial_ndim"
            )
        sample_shape = tuple(int(size) for size in trajectory.shape[:-1])
        leading_shape, data = _flatten_batches(kspace, len(sample_shape) + 1)
        data = data.reshape(data.shape[0], data.shape[1], -1)
        trajectory = trajectory.reshape(-1, self.spatial_ndim)
        selected_density = None if density is None else density.reshape(-1)
        solve_shape = output_shape
        if self.calibration_width is not None:
            solve_shape = tuple(
                min(self.calibration_width, size) for size in output_shape
            )
            scaled = trajectory * torch.tensor(
                output_shape,
                device=trajectory.device,
                dtype=trajectory.dtype,
            )
            retained = torch.linalg.vector_norm(scaled, dim=-1) <= min(solve_shape) / 2
            trajectory = trajectory[retained]
            data = data[..., retained]
            if selected_density is not None:
                selected_density = selected_density[retained]
        physics_class = NonCartesian2D if self.spatial_ndim == 2 else NonCartesian3D
        selected_encoding = physics_class(
            trajectory,
            solve_shape,
            density=selected_density,
            backend=self.backend,
            n_coils=data.shape[1],
            n_batchs=data.shape[0],
            toeplitz=self.toeplitz,
            viewed_as_real=False,
        )
        return data, selected_encoding, solve_shape, output_shape, leading_shape


# %% private module subroutines


class _NLINVJacobian(deepinv.physics.LinearPhysics):
    def __init__(self, physics: NLINVPhysics, state: torch.Tensor) -> None:
        super().__init__()
        self.__dict__["physics"] = physics
        self.__dict__["state"] = state

    def A(self, vector: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        del kwargs
        return self.physics.A_jvp(self.state, vector)

    def A_adjoint(self, vector: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        del kwargs
        return self.physics.A_vjp(self.state, vector)

    def A_adjoint_A(self, vector: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        del kwargs
        image, coils = self.physics.physical(self.state)
        delta_image, delta_coefficients = self.physics._split_state(vector)
        delta_coils = self.physics._sobolev_forward(delta_coefficients)
        derivative_images = image * delta_coils + delta_image * coils
        method = getattr(self.physics.encoding, "A_adjoint_A", None)
        normal_images = (
            method(derivative_images)
            if method is not None
            else self.physics.encoding.A_adjoint(
                self.physics.encoding.A(derivative_images)
            )
        )
        image_gradient = (normal_images * coils.conj()).sum(dim=1, keepdim=True)
        coil_gradient = self.physics._sobolev_adjoint(normal_images * image.conj())
        return self.physics._join_state(image_gradient, coil_gradient)


class _CartesianCalibrationEncoding(deepinv.physics.LinearPhysics):
    def __init__(self, mask: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mask", torch.as_tensor(mask).to(torch.complex64))
        self.axes = tuple(range(-self.mask.ndim, 0))

    def A(self, image: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        del kwargs
        return self.mask * _fftc(image, self.axes)

    def A_adjoint(self, kspace: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        del kwargs
        return _ifftc(self.mask.conj() * kspace, self.axes)

    def A_adjoint_A(self, image: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        del kwargs
        return self.A_adjoint(self.A(image))


def _sobolev_weights(
    solve_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    width: float,
    degree: int,
    *,
    device: Any,
) -> torch.Tensor:
    scale = width / max(output_shape) ** 2
    grids = torch.meshgrid(
        *[
            torch.arange(
                -size // 2,
                size - size // 2,
                dtype=torch.float32,
                device=device,
            )
            for size in solve_shape
        ],
        indexing="ij",
    )
    radius = sum(grid.square() for grid in grids)
    return (1.0 + scale * radius).pow(-degree / 2)


def _fftc(value: torch.Tensor, axes: tuple[int, ...]) -> torch.Tensor:
    value = torch.fft.ifftshift(value, dim=axes)
    value = torch.fft.fftn(value, dim=axes, norm="ortho")
    return torch.fft.fftshift(value, dim=axes)


def _ifftc(value: torch.Tensor, axes: tuple[int, ...]) -> torch.Tensor:
    value = torch.fft.ifftshift(value, dim=axes)
    value = torch.fft.ifftn(value, dim=axes, norm="ortho")
    return torch.fft.fftshift(value, dim=axes)


def _resize_centered(value: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    spatial_ndim = len(shape)
    result_shape = (*value.shape[:-spatial_ndim], *shape)
    result = torch.zeros(result_shape, dtype=value.dtype, device=value.device)
    source_slices = [slice(None)] * value.ndim
    target_slices = [slice(None)] * value.ndim
    for offset, target_size in enumerate(shape, start=value.ndim - spatial_ndim):
        source_size = value.shape[offset]
        count = min(source_size, target_size)
        source_start = (source_size - count) // 2
        target_start = (target_size - count) // 2
        source_slices[offset] = slice(source_start, source_start + count)
        target_slices[offset] = slice(target_start, target_start + count)
    result[tuple(target_slices)] = value[tuple(source_slices)]
    return result


def _resize_fourier_image(
    value: torch.Tensor,
    shape: tuple[int, ...],
) -> torch.Tensor:
    if tuple(value.shape[-len(shape) :]) == shape:
        return value
    axes = tuple(range(-len(shape), 0))
    return _ifftc(_resize_centered(_fftc(value, axes), shape), axes)


def _flatten_batches(
    value: torch.Tensor,
    single_ndim: int,
) -> tuple[tuple[int, ...], torch.Tensor]:
    if value.ndim < single_ndim:
        raise ValueError("kspace has too few dimensions")
    leading_shape = tuple(int(size) for size in value.shape[: value.ndim - single_ndim])
    single_shape = tuple(int(size) for size in value.shape[-single_ndim:])
    batch = prod(leading_shape) if leading_shape else 1
    return leading_shape, value.reshape(batch, *single_shape)


def _restore_leading(
    value: torch.Tensor,
    leading_shape: tuple[int, ...],
    single_shape: tuple[int, ...],
) -> torch.Tensor:
    if not leading_shape:
        return value.reshape(*single_shape)
    return value.reshape(*leading_shape, *single_shape)
