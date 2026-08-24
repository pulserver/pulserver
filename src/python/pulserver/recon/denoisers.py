"""DeepInverse-style denoiser classes used by :func:`pulserver.recon.pics`."""

from __future__ import annotations

__all__ = [
    "LLR",
    "TGV",
    "TV",
    "AverageDenoiser",
    "Positive",
    "Wavelet",
]

from collections.abc import Sequence
from importlib import import_module
from typing import Any

try:
    _ModuleBase = import_module("torch").nn.Module
except ImportError:  # Keep the optional module importable without recon extras.
    _ModuleBase = object


def _models() -> Any:
    try:
        return import_module("deepinv.models")
    except ImportError as error:
        raise ImportError(
            "Reconstruction denoisers require DeepInverse, which ships "
            "with pulserver; reinstall the package to restore it."
        ) from error


class Wavelet(_ModuleBase):
    """DeepInverse 2D or 3D orthogonal-wavelet denoiser.

    Batch entries are independent, which makes the same object suitable for
    slices, contrasts, and dynamic frames. ``dimension`` selects only the
    spatial transform dimensionality.

    Examples
    --------
    .. plot::

       import pulserver.recon as recon
       from _figures import images, phantom, radial_spokes

       truth, coil_maps = phantom(64, coils=4)
       physics = recon.NonCartesian2D(
           radial_spokes(64, 16), (64, 64), coil_maps=coil_maps[0]
       )
       measured = physics.A(truth)

       images(
           [
               ("truth", truth[0]),
               ("no prior", recon.pics(measured, physics, iterations=8)[0, 0]),
               (
                   "wavelet",
                   recon.pics(
                       measured, physics, recon.Wavelet(), iterations=8,
                       regularization=0.01,
                   )[0, 0],
               ),
           ],
           title="a wavelet prior on a 16-spoke radial scan",
       )
    """

    def __init__(
        self,
        *,
        dimension: int = 2,
        wavelet: str = "db8",
        level: int = 3,
        complex_data: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if dimension not in (2, 3):
            raise ValueError("dimension must be 2 or 3")
        self.model = _models().WaveletDenoiser(
            wvdim=dimension,
            wv=wavelet,
            level=level,
            is_complex=complex_data,
            **kwargs,
        )

    def forward(self, x: Any, sigma: Any, **kwargs: Any) -> Any:
        return self.model(x, sigma, **kwargs)


def _model_arguments(models: tuple[Any, ...]) -> tuple[Any, ...]:
    if len(models) == 1 and isinstance(models[0], Sequence):
        return tuple(models[0])
    return models


class AverageDenoiser(_ModuleBase):
    """Equal-weight ensemble of denoiser modules.

    Applies every denoiser to the same ``(x, sigma)`` pair and returns their
    arithmetic mean, so it is directly compatible with DeepInverse's
    plug-and-play prior and with :func:`pulserver.recon.pics`. Denoisers may
    be passed positionally or as one sequence.

    Examples
    --------
    Several priors applied as one, by averaging what each of them returns.

    >>> import pulserver.recon as recon
    >>> combined = recon.AverageDenoiser(recon.TV(), recon.Wavelet())
    >>> isinstance(combined, recon.AverageDenoiser)
    True
    """

    def __init__(self, *denoisers: Any) -> None:
        super().__init__()
        selected = _model_arguments(denoisers)
        if not selected:
            raise ValueError("at least one denoiser is required")
        torch = import_module("torch")
        self.models = torch.nn.ModuleList(selected)

    def forward(self, x: Any, sigma: Any, **kwargs: Any) -> Any:
        result = self.models[0](x, sigma, **kwargs)
        for model in self.models[1:]:
            result = result + model(x, sigma, **kwargs)
        return result / len(self.models)


class Positive(_ModuleBase):
    r"""Indicator prior for the non-negative real or fixed-phase cone.

    The proximity operator is an orthogonal projection, so this module can be
    used directly with Pulserver's FISTA, ADMM, and PDHG implementations. For
    complex data, positivity means a non-negative magnitude along ``phase``.
    When no phase is supplied, it means a non-negative real value with zero
    imaginary component.

    Parameters
    ----------
    phase
        Fixed complex phase map or real phase in radians. It must broadcast to
        the complex-valued input.
    viewed_as_real
        Interpret channel pairs ``(real, imaginary)`` along dimension one as
        complex channels, matching Pulserver's default MRI physics interface.
    tolerance
        Numerical tolerance used only when evaluating the indicator cost.

    Examples
    --------
    The proximity operator is a projection, so it is its own illustration: what
    is below the cone is moved onto it and everything else is left alone.

    >>> import torch
    >>> import pulserver.recon as recon
    >>> recon.Positive().prox(torch.tensor([[-1.0, 0.5], [2.0, -0.25]]), 1.0)
    tensor([[0.0000, 0.5000],
            [2.0000, 0.0000]])
    """

    def __init__(
        self,
        phase: Any | None = None,
        *,
        viewed_as_real: bool = False,
        tolerance: float = 1e-7,
    ) -> None:
        super().__init__()
        if tolerance < 0.0:
            raise ValueError("tolerance must be non-negative")
        torch = import_module("torch")
        selected_phase = None
        if phase is not None:
            selected_phase = torch.as_tensor(phase)
            if not selected_phase.is_complex():
                if not selected_phase.is_floating_point():
                    selected_phase = selected_phase.to(torch.float32)
                selected_phase = torch.polar(
                    torch.ones_like(selected_phase),
                    selected_phase,
                )
            else:
                magnitude = selected_phase.abs()
                epsilon = torch.finfo(selected_phase.real.dtype).eps
                selected_phase = torch.where(
                    magnitude > epsilon,
                    selected_phase / magnitude.clamp_min(epsilon),
                    torch.ones_like(selected_phase),
                )
        self.register_buffer("phase", selected_phase)
        self.viewed_as_real = bool(viewed_as_real)
        self.tolerance = float(tolerance)

    def forward(self, value: Any, *_args: Any, **_kwargs: Any) -> Any:
        """Evaluate the positivity indicator independently for each batch."""
        torch = import_module("torch")
        complex_value, restore = self._complex_view(value)
        rotated = self._demodulate(complex_value)
        feasible = rotated.real >= -self.tolerance
        if rotated.is_complex():
            feasible = feasible & (rotated.imag.abs() <= self.tolerance)
        axes = tuple(range(1, feasible.ndim))
        feasible = feasible.all(dim=axes) if axes else feasible
        zero = torch.zeros_like(feasible, dtype=value.real.dtype)
        infinity = torch.full_like(zero, torch.inf)
        del restore
        return torch.where(feasible, zero, infinity)

    def prox(
        self,
        value: Any,
        *_args: Any,
        gamma: Any | None = None,
        **_kwargs: Any,
    ) -> Any:
        """Project ``value`` onto the configured positivity cone."""
        del gamma
        complex_value, restore = self._complex_view(value)
        phase = self._phase_for(complex_value)
        rotated = complex_value if phase is None else complex_value * phase.conj()
        projected = rotated.real.clamp_min(0)
        if complex_value.is_complex():
            projected = projected.to(complex_value.dtype)
        if phase is not None:
            projected = projected * phase
        return restore(projected)

    def prox_conjugate(
        self,
        value: Any,
        *_args: Any,
        gamma: Any,
        lamb: Any = 1.0,
        **_kwargs: Any,
    ) -> Any:
        """Project onto the convex conjugate through Moreau's identity."""
        torch = import_module("torch")
        selected_lambda = torch.as_tensor(
            lamb,
            device=value.device,
            dtype=value.real.dtype,
        )
        if not bool(torch.all(selected_lambda > 0)):
            if bool(torch.all(selected_lambda == 0)):
                return torch.zeros_like(value)
            raise ValueError("positivity weights must be non-negative")
        return value - gamma * self.prox(value / gamma)

    # %% private module subroutines

    def _complex_view(self, value: Any) -> tuple[Any, Any]:
        torch = import_module("torch")
        if not self.viewed_as_real:
            return value, lambda result: result
        if value.ndim < 2 or value.shape[1] % 2:
            raise ValueError(
                "paired-real positivity requires an even channel dimension"
            )
        batch, channels, *spatial = value.shape
        paired = value.reshape(batch, channels // 2, 2, *spatial).movedim(2, -1)
        complex_value = torch.view_as_complex(paired.contiguous())

        def restore(result: Any) -> Any:
            result = torch.view_as_real(result).movedim(-1, 2)
            return result.reshape(batch, channels, *spatial)

        return complex_value, restore

    def _phase_for(self, value: Any) -> Any | None:
        if self.phase is None:
            return None
        if not value.is_complex():
            raise TypeError(
                "fixed-phase positivity requires complex or paired-real input"
            )
        return self.phase.to(device=value.device, dtype=value.dtype)

    def _demodulate(self, value: Any) -> Any:
        phase = self._phase_for(value)
        return value if phase is None else value * phase.conj()


# The LLR block/cycle-spinning design is adapted from SetsompopLab/MRF.
# See NOTICE_LLR.md for the BSD-3-Clause copyright and license text.
def _spatial_parameter(
    value: int | Sequence[int],
    dimension: int,
    *,
    name: str,
) -> tuple[int, ...]:
    if isinstance(value, int):
        result = (value,) * dimension
    else:
        result = tuple(int(item) for item in value)
    if len(result) != dimension or any(item < 1 for item in result):
        raise ValueError(f"{name} must contain {dimension} positive integers")
    return result


class LLR(_ModuleBase):
    """Locally low-rank blockwise singular-value denoiser.

    Treats the channel axis as the low-rank contrast/subspace axis and
    soft-thresholds singular values independently in overlapping spatial
    blocks. Inputs have shape ``(batch, channels, *spatial)`` and may be real
    or complex Torch tensors. ``dimension`` must be 2 or 3.

    ``cycle_spins=True`` cyclically shifts the block origin between calls,
    matching the blocking-artifact suppression used by the MRF reference
    implementation. ``block_batch_size`` bounds the simultaneous patch/SVD
    workspace for large 3D volumes; use ``None`` to process every block in one
    vectorized batch.

    Examples
    --------
    .. plot::

       import pulserver.recon as recon
       from _figures import images, phantom, radial_spokes

       truth, coil_maps = phantom(64, coils=4)
       physics = recon.NonCartesian2D(
           radial_spokes(64, 16), (64, 64), coil_maps=coil_maps[0]
       )
       measured = physics.A(truth)

       images(
           [
               ("truth", truth[0]),
               ("no prior", recon.pics(measured, physics, iterations=8)[0, 0]),
               (
                   "LLR",
                   recon.pics(
                       measured, physics, recon.LLR(), iterations=8,
                       regularization=0.01,
                   )[0, 0],
               ),
           ],
           title="locally low rank on a 16-spoke radial scan",
       )
    """

    def __init__(
        self,
        *,
        dimension: int = 2,
        block_size: int | Sequence[int] = 8,
        stride: int | Sequence[int] | None = None,
        cycle_spins: bool = True,
        block_batch_size: int | None = 1024,
    ) -> None:
        super().__init__()
        if dimension not in (2, 3):
            raise ValueError("dimension must be 2 or 3")
        blocks = _spatial_parameter(block_size, dimension, name="block_size")
        strides = (
            blocks
            if stride is None
            else _spatial_parameter(stride, dimension, name="stride")
        )
        if block_batch_size is not None and (
            not isinstance(block_batch_size, int)
            or isinstance(block_batch_size, bool)
            or block_batch_size < 1
        ):
            raise ValueError("block_batch_size must be a positive integer or None")
        self.dimension = dimension
        self.block_size = blocks
        self.stride = strides
        self.cycle_spins = cycle_spins
        self.block_batch_size = block_batch_size
        self._call_count = 0
        self._patch_index_key: tuple[Any, ...] | None = None
        self._patch_coordinates: tuple[Any, Any, Any] | None = None

    def _coordinates(
        self,
        spatial_shape: tuple[int, ...],
        device: Any,
    ) -> tuple[Any, Any, Any]:
        torch = import_module("torch")
        key = (*spatial_shape, str(device))
        if key == self._patch_index_key:
            return self._patch_coordinates

        starts = []
        for size, block, step in zip(
            spatial_shape,
            self.block_size,
            self.stride,
            strict=True,
        ):
            if block > size:
                raise ValueError(
                    f"block_size {self.block_size!r} exceeds input "
                    f"spatial shape {spatial_shape!r}"
                )
            axis_starts = list(range(0, size - block + 1, step))
            if axis_starts[-1] != size - block:
                axis_starts.append(size - block)
            starts.append(torch.tensor(axis_starts, dtype=torch.long, device=device))

        start_grid = torch.meshgrid(*starts, indexing="ij")
        offset_grid = torch.meshgrid(
            *[
                torch.arange(block, dtype=torch.long, device=device)
                for block in self.block_size
            ],
            indexing="ij",
        )
        flat_strides = torch.tensor(
            [
                int(torch.tensor(spatial_shape[index + 1 :]).prod().item())
                if index + 1 < self.dimension
                else 1
                for index in range(self.dimension)
            ],
            dtype=torch.long,
            device=device,
        )
        start_coordinates = torch.stack(
            [axis.reshape(-1) for axis in start_grid],
            dim=-1,
        )
        offset_coordinates = torch.stack(
            [axis.reshape(-1) for axis in offset_grid],
            dim=-1,
        )
        self._patch_index_key = key
        self._patch_coordinates = (
            start_coordinates,
            offset_coordinates,
            flat_strides,
        )
        return self._patch_coordinates

    def forward(self, x: Any, sigma: Any, **kwargs: Any) -> Any:
        del kwargs
        torch = import_module("torch")
        if x.ndim != self.dimension + 2:
            raise ValueError(
                f"expected a {self.dimension}D image tensor with "
                f"{self.dimension + 2} axes, got shape {tuple(x.shape)!r}"
            )

        shifts = tuple(self._call_count % block for block in self.block_size)
        self._call_count += 1

        batch, channels = x.shape[:2]
        spatial_shape = tuple(int(item) for item in x.shape[2:])
        start_coordinates, offset_coordinates, flat_strides = self._coordinates(
            spatial_shape,
            x.device,
        )

        real_dtype = x.real.dtype
        threshold = torch.as_tensor(
            sigma,
            dtype=real_dtype,
            device=x.device,
        ).squeeze()
        if threshold.ndim == 0:
            threshold = threshold.expand(batch)
        elif threshold.ndim != 1 or threshold.shape[0] != batch:
            raise ValueError("sigma must be a scalar or contain one value per batch")
        if torch.any(threshold < 0):
            raise ValueError("sigma must be non-negative")

        flat_input = x.reshape(batch, channels, -1)
        output = torch.zeros_like(x).reshape(batch, channels, -1)
        non_overlapping = all(
            step == block and size % block == 0
            for size, block, step in zip(
                spatial_shape,
                self.block_size,
                self.stride,
                strict=True,
            )
        )
        weights = (
            None
            if non_overlapping
            else torch.zeros(
                output.shape[-1],
                dtype=real_dtype,
                device=x.device,
            )
        )
        block_count = start_coordinates.shape[0]
        blocks_per_batch = self.block_batch_size or block_count
        shifted_blocks = self.cycle_spins and any(shifts)
        if shifted_blocks:
            shift = torch.tensor(shifts, dtype=torch.long, device=x.device)
            spatial = torch.tensor(
                spatial_shape,
                dtype=torch.long,
                device=x.device,
            )
        else:
            offset_indices = (offset_coordinates * flat_strides).sum(dim=-1)
        for first in range(0, block_count, blocks_per_batch):
            starts = start_coordinates[first : first + blocks_per_batch]
            if shifted_blocks:
                coordinates = (
                    starts[:, None, :] + offset_coordinates[None, :, :] - shift
                ) % spatial
                indices = (coordinates * flat_strides).sum(dim=-1)
            else:
                start_indices = (starts * flat_strides).sum(dim=-1)
                indices = start_indices[:, None] + offset_indices[None, :]
            patches = flat_input[:, :, indices]
            matrices = patches.permute(0, 2, 1, 3)
            if channels <= matrices.shape[-1]:
                gram = matrices @ matrices.mH
                eigenvalues, u = torch.linalg.eigh(gram)
                singular_values = torch.sqrt(torch.clamp(eigenvalues, min=0))
                cutoff = threshold[:, None, None]
                denominator = singular_values.clamp_min(torch.finfo(real_dtype).eps)
                factors = torch.where(
                    singular_values > torch.finfo(real_dtype).eps,
                    torch.clamp(
                        1 - cutoff / denominator,
                        min=0,
                    ),
                    0,
                )
                matrices = u @ (factors.unsqueeze(-1) * (u.mH @ matrices))
            else:
                u, singular_values, vh = torch.linalg.svd(
                    matrices,
                    full_matrices=False,
                )
                shrunk = torch.clamp(
                    singular_values - threshold[:, None, None],
                    min=0,
                )
                matrices = (u * shrunk.unsqueeze(-2)) @ vh

            flat_indices = indices.reshape(-1)
            values = matrices.permute(0, 2, 1, 3).reshape(
                batch,
                channels,
                -1,
            )
            expanded_indices = flat_indices[None, None, :].expand_as(values)
            if non_overlapping:
                output.scatter_(2, expanded_indices, values)
            else:
                output.scatter_add_(2, expanded_indices, values)
                weights.scatter_add_(
                    0,
                    flat_indices,
                    torch.ones_like(flat_indices, dtype=real_dtype),
                )
        if weights is not None:
            output = output / weights.clamp_min(torch.finfo(real_dtype).eps)[None, None]
        return output.reshape_as(x)


class TV(_ModuleBase):
    """DeepInverse's spatially 2D/3D-agnostic total-variation denoiser.

    Examples
    --------
    >>> import torch
    >>> import pulserver.recon as recon
    >>> prior = recon.TV()
    >>> prior(torch.zeros(1, 1, 8, 8), 0.1).shape
    torch.Size([1, 1, 8, 8])
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.model = _models().TVDenoiser(**kwargs)

    def forward(self, x: Any, sigma: Any, **kwargs: Any) -> Any:
        return self.model(x, sigma, **kwargs)


class TGV(_ModuleBase):
    """DeepInverse's spatially 2D/3D-agnostic total-generalized-variation denoiser.

    Examples
    --------
    .. plot::

       import pulserver.recon as recon
       from _figures import images, phantom, radial_spokes

       truth, coil_maps = phantom(64, coils=4)
       physics = recon.NonCartesian2D(
           radial_spokes(64, 16), (64, 64), coil_maps=coil_maps[0]
       )
       measured = physics.A(truth)

       images(
           [
               ("truth", truth[0]),
               ("no prior", recon.pics(measured, physics, iterations=8)[0, 0]),
               (
                   "TGV",
                   recon.pics(
                       measured, physics, recon.TGV(), iterations=8,
                       regularization=0.01,
                   )[0, 0],
               ),
           ],
           title="total generalized variation on a 16-spoke radial scan",
       )
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.model = _models().TGVDenoiser(**kwargs)

    def forward(self, x: Any, sigma: Any, **kwargs: Any) -> Any:
        return self.model(x, sigma, **kwargs)
