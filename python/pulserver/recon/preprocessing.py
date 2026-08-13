"""Array-level MRI preprocessing utilities.

Functions preserve Torch tensors (including their device) and NumPy arrays.
MRPro containers are delegated to their maintained methods when applicable.
"""

from __future__ import annotations

__all__ = [
    "POCS",
    "EPIPhaseCorrection",
    "Homodyne",
    "SmsEpiInputs",
    "cartesian_3d_to_2d",
    "coil_compress",
    "correct_epi_eddy_currents",
    "epi_ramp_interpolate",
    "estimate_epi_eddy_phase",
    "grid_cartesian",
    "noise_prewhiten",
    "pipe_menon_dcf",
    "remove_readout_oversampling",
]

from importlib import import_module
from typing import Any

from ._sms import SmsEpiInputs


class EPIPhaseCorrection:
    """Shot-resolved odd/even EPI navigator phase correction.

    The estimator fits the readout phase independently for every shot after
    combining all remaining navigator dimensions. Optional causal smoothing
    acts on the polynomial coefficients, retaining slow drift without merging
    genuinely distinct shot offsets.

    Parameters
    ----------
    polynomial_order
        Readout phase-polynomial order.
    shot_axis
        Navigator shot axis. ``None`` estimates one global correction.
    readout_axis
        Readout sample axis.
    temporal_smoothing
        Causal coefficient smoothing in ``[0, 1)``. Zero disables smoothing.
    """

    def __init__(
        self,
        *,
        polynomial_order: int = 1,
        shot_axis: int | None = 0,
        readout_axis: int = -1,
        temporal_smoothing: float = 0.0,
    ) -> None:
        if polynomial_order < 0:
            raise ValueError("polynomial_order must be non-negative")
        if not 0.0 <= temporal_smoothing < 1.0:
            raise ValueError("temporal_smoothing must lie in [0, 1)")
        self.polynomial_order = int(polynomial_order)
        self.shot_axis = shot_axis
        self.readout_axis = int(readout_axis)
        self.temporal_smoothing = float(temporal_smoothing)

    def fit(self, positive_navigator: Any, negative_navigator: Any) -> Any:
        """Estimate one smooth odd/even phase curve per navigator shot."""
        if positive_navigator.shape != negative_navigator.shape:
            raise ValueError("positive and negative navigators must have equal shape")
        if self.shot_axis is None:
            return estimate_epi_eddy_phase(
                positive_navigator,
                negative_navigator,
                readout_axis=self.readout_axis,
                polynomial_order=self.polynomial_order,
            )
        xp, is_torch = _matching_libraries(
            positive_navigator,
            negative_navigator,
        )
        shot_axis = self.shot_axis % positive_navigator.ndim
        readout_axis = self.readout_axis % positive_navigator.ndim
        if shot_axis == readout_axis:
            raise ValueError("shot_axis and readout_axis must be distinct")
        cross = positive_navigator * negative_navigator.conj()
        if is_torch:
            cross = cross.movedim((shot_axis, readout_axis), (0, -1))
            if cross.ndim > 2:
                cross = cross.sum(dim=tuple(range(1, cross.ndim - 1)))
        else:
            cross = xp.moveaxis(cross, (shot_axis, readout_axis), (0, -1))
            if cross.ndim > 2:
                cross = cross.sum(axis=tuple(range(1, cross.ndim - 1)))
        phase = _unwrap_last(xp.angle(cross), xp, is_torch)
        return _fit_phase_curves(
            phase,
            self.polynomial_order,
            self.temporal_smoothing,
            xp,
            is_torch,
        )

    def correct(
        self,
        positive_readouts: Any,
        negative_readouts: Any,
        phase: Any | None = None,
    ) -> tuple[Any, Any, Any]:
        """Apply symmetric phase correction to shot-resolved polarities."""
        if positive_readouts.shape != negative_readouts.shape:
            raise ValueError("positive and negative readouts must have equal shape")
        if phase is None:
            phase = self.fit(positive_readouts, negative_readouts)
        xp, is_torch = _matching_libraries(positive_readouts, negative_readouts)
        phase = (
            xp.as_tensor(
                phase,
                device=positive_readouts.device,
                dtype=positive_readouts.real.dtype,
            )
            if is_torch
            else xp.asarray(phase, dtype=positive_readouts.real.dtype)
        )
        shape = [1] * positive_readouts.ndim
        readout_axis = self.readout_axis % positive_readouts.ndim
        shape[readout_axis] = positive_readouts.shape[readout_axis]
        if self.shot_axis is not None:
            shot_axis = self.shot_axis % positive_readouts.ndim
            shape[shot_axis] = positive_readouts.shape[shot_axis]
            expected = (shape[shot_axis], shape[readout_axis])
            if phase.shape != expected:
                raise ValueError(f"shot-resolved phase must have shape {expected}")
            phase = phase.reshape(
                *phase.shape,
                *([1] * (positive_readouts.ndim - 2)),
            )
            phase = (
                phase.movedim((0, 1), (shot_axis, readout_axis))
                if is_torch
                else xp.moveaxis(phase, (0, 1), (shot_axis, readout_axis))
            )
        elif phase.ndim != 1:
            raise ValueError("global EPI phase must be one-dimensional")
        else:
            phase = phase.reshape(shape)
        return (
            positive_readouts * xp.exp(-0.5j * phase),
            negative_readouts * xp.exp(0.5j * phase),
            phase,
        )

    def __call__(
        self,
        positive_readouts: Any,
        negative_readouts: Any,
        phase: Any | None = None,
    ) -> tuple[Any, Any, Any]:
        return self.correct(positive_readouts, negative_readouts, phase)


class Homodyne:
    """Homodyne reconstruction for one-sided Cartesian partial Fourier.

    Parameters
    ----------
    dimension
        Number of spatial Fourier dimensions.
    partial_axis
        Axis containing the partial-Fourier acquisition.
    """

    def __init__(self, *, dimension: int = 2, partial_axis: int = -2) -> None:
        if dimension not in {1, 2, 3}:
            raise ValueError("dimension must be 1, 2, or 3")
        self.dimension = int(dimension)
        self.partial_axis = int(partial_axis)

    def __call__(self, kspace: Any, mask: Any | None = None) -> Any:
        """Reconstruct an image from partial-Fourier Cartesian k-space."""
        axes, partial_axis = _spatial_axes(
            kspace.ndim,
            self.dimension,
            self.partial_axis,
        )
        acquired = _partial_fourier_mask(kspace, mask, partial_axis)
        lowpass, weight = _homodyne_masks(acquired)
        lowpass = _broadcast_line(lowpass, kspace.ndim, partial_axis)
        weight = _broadcast_line(weight, kspace.ndim, partial_axis)
        reference = _centered_fftn(kspace * lowpass, axes=axes, inverse=True)
        phase = _unit_phase(reference)
        weighted = _centered_fftn(kspace * weight, axes=axes, inverse=True)
        projected = (weighted * phase.conj()).real
        return projected * phase


class POCS:
    """Projection-onto-convex-sets partial-Fourier reconstruction.

    Parameters
    ----------
    dimension
        Number of spatial Fourier dimensions.
    partial_axis
        Axis containing the partial-Fourier acquisition.
    iterations
        Maximum number of data-consistency/phase-projection iterations.
    tolerance
        Relative iterate-change tolerance. Set to zero for a fixed count.
    positive
        Also project the demodulated image onto the non-negative real cone.
    """

    def __init__(
        self,
        *,
        dimension: int = 2,
        partial_axis: int = -2,
        iterations: int = 12,
        tolerance: float = 1e-5,
        positive: bool = True,
    ) -> None:
        if dimension not in {1, 2, 3}:
            raise ValueError("dimension must be 1, 2, or 3")
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        if tolerance < 0.0:
            raise ValueError("tolerance must be non-negative")
        self.dimension = int(dimension)
        self.partial_axis = int(partial_axis)
        self.iterations = int(iterations)
        self.tolerance = float(tolerance)
        self.positive = bool(positive)

    def __call__(self, kspace: Any, mask: Any | None = None) -> Any:
        """Reconstruct an image while preserving every acquired sample."""
        xp, is_torch = _torch_or_numpy(kspace)
        axes, partial_axis = _spatial_axes(
            kspace.ndim,
            self.dimension,
            self.partial_axis,
        )
        acquired = _partial_fourier_mask(kspace, mask, partial_axis)
        lowpass, _ = _homodyne_masks(acquired)
        acquired = _broadcast_line(acquired, kspace.ndim, partial_axis)
        lowpass = _broadcast_line(lowpass, kspace.ndim, partial_axis)
        phase = _unit_phase(_centered_fftn(kspace * lowpass, axes=axes, inverse=True))
        estimate = kspace.copy() if not is_torch else kspace.clone()
        previous = _centered_fftn(estimate, axes=axes, inverse=True)
        for _ in range(self.iterations):
            demodulated = (previous * phase.conj()).real
            if self.positive:
                demodulated = (
                    demodulated.clamp_min(0) if is_torch else xp.maximum(demodulated, 0)
                )
            projected = demodulated * phase
            proposed = _centered_fftn(projected, axes=axes, inverse=False)
            estimate = xp.where(acquired, kspace, proposed)
            updated = _centered_fftn(estimate, axes=axes, inverse=True)
            if self.tolerance and _relative_change(updated, previous) <= self.tolerance:
                previous = updated
                break
            previous = updated
        return previous


def _torch_or_numpy(array: Any) -> tuple[Any, bool]:
    try:
        torch = import_module("torch")
    except ImportError:
        torch = None
    if torch is not None and isinstance(array, torch.Tensor):
        return torch, True
    return import_module("numpy"), False


def _centered_fft(
    data: Any,
    *,
    axis: int,
    inverse: bool,
) -> Any:
    xp, is_torch = _torch_or_numpy(data)
    if is_torch:
        shifted = xp.fft.ifftshift(data, dim=axis)
        transform = xp.fft.ifft if inverse else xp.fft.fft
        return xp.fft.fftshift(
            transform(shifted, dim=axis, norm="ortho"),
            dim=axis,
        )
    shifted = xp.fft.ifftshift(data, axes=axis)
    transform = xp.fft.ifft if inverse else xp.fft.fft
    return xp.fft.fftshift(
        transform(shifted, axis=axis, norm="ortho"),
        axes=axis,
    )


def cartesian_3d_to_2d(
    kspace: Any,
    *,
    readout_axis: int = -1,
) -> Any:
    """Convert Cartesian 3D k-space into independent 2D hybrid-space planes.

    A centered inverse FFT is applied along readout. The returned readout
    positions can be moved into the batch dimension before constructing
    :func:`pulserver.recon.physics.Cartesian2D` physics.
    """
    return _centered_fft(kspace, axis=readout_axis, inverse=True)


def remove_readout_oversampling(
    data: Any,
    target_size: int | None = None,
    *,
    readout_axis: int = -1,
) -> Any:
    """Remove readout oversampling by centered image-domain cropping.

    MRPro ``KData`` objects delegate to ``KData.remove_readout_os()`` and infer
    the target from their header. Raw arrays require ``target_size``.
    """
    method = getattr(data, "remove_readout_os", None)
    if callable(method):
        if target_size is not None:
            raise ValueError("target_size is inferred from an MRPro KData header")
        return method()
    if target_size is None:
        raise ValueError("target_size is required for raw arrays")
    current = data.shape[readout_axis]
    if not 0 < target_size <= current:
        raise ValueError(f"target_size must be in [1, {current}], got {target_size}")
    if target_size == current:
        return data
    image = _centered_fft(data, axis=readout_axis, inverse=True)
    start = (current - target_size) // 2
    selection = [slice(None)] * image.ndim
    selection[readout_axis] = slice(start, start + target_size)
    cropped = image[tuple(selection)]
    return _centered_fft(cropped, axis=readout_axis, inverse=False)


def coil_compress(
    kspace: Any,
    n_coils: int | float,
    *,
    trajectory: Any | None = None,
    calibration_radius: float | None = None,
) -> tuple[Any, Any]:
    """Apply mri-nufft SVD coil compression and return data plus matrix."""
    try:
        function = import_module("mrinufft.extras").coil_compression
    except ImportError as error:
        raise ImportError(
            "Coil compression requires mri-nufft; install "
            "pulserver[recon-cpu] or pulserver[recon-cuda]."
        ) from error
    return function(
        kspace,
        n_coils,
        traj=trajectory,
        krad_thresh=calibration_radius,
    )


def noise_prewhiten(
    kspace: Any,
    noise: Any,
    *,
    coil_axis: int = -2,
    scale_factor: float = 1.0,
) -> Any:
    """Decorrelate receiver coils using the measured noise covariance.

    MRPro ``KData`` delegates to ``KData.prewhiten``. Raw Torch/NumPy arrays
    are whitened with a Cholesky solve and preserve their input container.
    """
    method = getattr(kspace, "prewhiten", None)
    if callable(method):
        return method(noise, scale_factor)

    xp, is_torch = _torch_or_numpy(kspace)
    _, noise_is_torch = _torch_or_numpy(noise)
    if is_torch != noise_is_torch:
        raise TypeError("kspace and noise must use the same array library")

    if is_torch:
        moved_noise = noise.movedim(coil_axis, 0)
        noise_flat = moved_noise.reshape(moved_noise.shape[0], -1)
        covariance = noise_flat @ noise_flat.conj().transpose(0, 1)
        covariance = covariance / noise_flat.shape[-1]
        cholesky = xp.linalg.cholesky(covariance)
        moved_data = kspace.movedim(coil_axis, 0)
        flat = moved_data.reshape(moved_data.shape[0], -1)
        whitened = xp.linalg.solve_triangular(
            cholesky,
            flat,
            upper=False,
        )
        whitened = whitened * scale_factor**0.5
        return whitened.reshape(moved_data.shape).movedim(0, coil_axis)

    moved_noise = xp.moveaxis(noise, coil_axis, 0)
    noise_flat = moved_noise.reshape(moved_noise.shape[0], -1)
    covariance = noise_flat @ noise_flat.conj().T / noise_flat.shape[-1]
    cholesky = xp.linalg.cholesky(covariance)
    moved_data = xp.moveaxis(kspace, coil_axis, 0)
    flat = moved_data.reshape(moved_data.shape[0], -1)
    whitened = xp.linalg.solve(cholesky, flat) * scale_factor**0.5
    return xp.moveaxis(whitened.reshape(moved_data.shape), 0, coil_axis)


def epi_ramp_interpolate(
    data: Any,
    sample_positions: Any,
    target_positions: Any,
    *,
    readout_axis: int = -1,
) -> Any:
    """Linearly interpolate EPI ramp samples onto a uniform readout grid."""
    xp, is_torch = _torch_or_numpy(data)
    if is_torch:
        source = xp.as_tensor(
            sample_positions,
            device=data.device,
            dtype=data.real.dtype,
        )
        target = xp.as_tensor(
            target_positions,
            device=data.device,
            dtype=data.real.dtype,
        )
        if source.ndim != 1 or target.ndim != 1:
            raise ValueError("sample_positions and target_positions must be 1D")
        if source.numel() != data.shape[readout_axis]:
            raise ValueError("sample_positions length must match the readout")
        if not bool(xp.all(source[1:] > source[:-1])):
            raise ValueError("sample_positions must be strictly increasing")
        right = xp.searchsorted(source, target).clamp(1, source.numel() - 1)
        left = right - 1
        weight = (target - source[left]) / (source[right] - source[left])
        moved = data.movedim(readout_axis, -1)
        result = moved[..., left] * (1 - weight) + moved[..., right] * weight
        return result.movedim(-1, readout_axis)

    source = xp.asarray(sample_positions)
    target = xp.asarray(target_positions)
    if source.ndim != 1 or target.ndim != 1:
        raise ValueError("sample_positions and target_positions must be 1D")
    if source.size != data.shape[readout_axis]:
        raise ValueError("sample_positions length must match the readout")
    if not xp.all(source[1:] > source[:-1]):
        raise ValueError("sample_positions must be strictly increasing")
    right = xp.searchsorted(source, target).clip(1, source.size - 1)
    left = right - 1
    weight = (target - source[left]) / (source[right] - source[left])
    moved = xp.moveaxis(data, readout_axis, -1)
    result = moved[..., left] * (1 - weight) + moved[..., right] * weight
    return xp.moveaxis(result, -1, readout_axis)


def _unwrap(phase: Any, xp: Any, is_torch: bool) -> Any:
    if not is_torch:
        return xp.unwrap(phase)
    delta = phase[1:] - phase[:-1]
    wrapped = (delta + xp.pi) % (2 * xp.pi) - xp.pi
    wrapped = xp.where((wrapped == -xp.pi) & (delta > 0), xp.pi, wrapped)
    correction = xp.cumsum(wrapped - delta, dim=0)
    result = phase.clone()
    result[1:] += correction
    return result


def estimate_epi_eddy_phase(
    positive_navigator: Any,
    negative_navigator: Any,
    *,
    readout_axis: int = -1,
    polynomial_order: int = 1,
) -> Any:
    """Estimate a smooth odd/even EPI phase difference from navigator pairs."""
    if positive_navigator.shape != negative_navigator.shape:
        raise ValueError("positive and negative navigators must have equal shape")
    if polynomial_order < 0:
        raise ValueError("polynomial_order must be non-negative")
    xp, is_torch = _torch_or_numpy(positive_navigator)
    negative_xp, negative_is_torch = _torch_or_numpy(negative_navigator)
    if xp is not negative_xp and is_torch != negative_is_torch:
        raise TypeError("navigator arrays must use the same array library")

    cross = positive_navigator * negative_navigator.conj()
    axes = tuple(
        index for index in range(cross.ndim) if index != readout_axis % cross.ndim
    )
    cross = cross.sum(dim=axes) if is_torch else cross.sum(axis=axes)
    phase = _unwrap(xp.angle(cross), xp, is_torch)
    n_readout = phase.shape[0]
    if is_torch:
        coordinate = xp.linspace(
            -1,
            1,
            n_readout,
            device=phase.device,
            dtype=phase.dtype,
        )
        design = xp.stack(
            [coordinate**degree for degree in range(polynomial_order + 1)],
            dim=1,
        )
        coefficients = xp.linalg.lstsq(design, phase[:, None]).solution
        return (design @ coefficients)[:, 0]
    coordinate = xp.linspace(-1, 1, n_readout)
    coefficients = xp.polynomial.polynomial.polyfit(
        coordinate,
        phase,
        polynomial_order,
    )
    return xp.polynomial.polynomial.polyval(coordinate, coefficients)


def correct_epi_eddy_currents(
    positive_readouts: Any,
    negative_readouts: Any,
    phase: Any | None = None,
    *,
    readout_axis: int = -1,
    polynomial_order: int = 1,
) -> tuple[Any, Any, Any]:
    """Apply symmetric odd/even phase correction to EPI readout polarities."""
    if phase is None:
        phase = estimate_epi_eddy_phase(
            positive_readouts,
            negative_readouts,
            readout_axis=readout_axis,
            polynomial_order=polynomial_order,
        )
    xp, is_torch = _torch_or_numpy(positive_readouts)
    phase = (
        xp.as_tensor(
            phase,
            device=positive_readouts.device,
            dtype=positive_readouts.real.dtype,
        )
        if is_torch
        else xp.asarray(phase)
    )
    shape = [1] * positive_readouts.ndim
    shape[readout_axis] = phase.shape[0]
    phase = phase.reshape(shape)
    positive_factor = xp.exp(-0.5j * phase)
    negative_factor = xp.exp(0.5j * phase)
    return (
        positive_readouts * positive_factor,
        negative_readouts * negative_factor,
        phase.reshape(-1),
    )


# %% private module subroutines


def _spatial_axes(
    ndim: int,
    dimension: int,
    partial_axis: int,
) -> tuple[tuple[int, ...], int]:
    if ndim < dimension:
        raise ValueError("input has fewer dimensions than the spatial transform")
    axes = tuple(range(ndim - dimension, ndim))
    selected = partial_axis % ndim
    if selected not in axes:
        raise ValueError("partial_axis must be one of the spatial dimensions")
    return axes, selected


def _centered_fftn(
    data: Any,
    *,
    axes: tuple[int, ...],
    inverse: bool,
) -> Any:
    xp, is_torch = _torch_or_numpy(data)
    if is_torch:
        shifted = xp.fft.ifftshift(data, dim=axes)
        transform = xp.fft.ifftn if inverse else xp.fft.fftn
        return xp.fft.fftshift(
            transform(shifted, dim=axes, norm="ortho"),
            dim=axes,
        )
    shifted = xp.fft.ifftshift(data, axes=axes)
    transform = xp.fft.ifftn if inverse else xp.fft.fftn
    return xp.fft.fftshift(
        transform(shifted, axes=axes, norm="ortho"),
        axes=axes,
    )


def _partial_fourier_mask(data: Any, mask: Any | None, axis: int) -> Any:
    xp, is_torch = _torch_or_numpy(data)
    if mask is None:
        occupied = data.abs() > 0 if is_torch else xp.abs(data) > 0
        reduce_axes = tuple(index for index in range(data.ndim) if index != axis)
        acquired = (
            occupied.any(dim=reduce_axes)
            if is_torch
            else occupied.any(axis=reduce_axes)
        )
    else:
        acquired = (
            xp.as_tensor(mask, device=data.device, dtype=xp.bool)
            if is_torch
            else xp.asarray(mask, dtype=bool)
        )
    if acquired.ndim != 1 or acquired.shape[0] != data.shape[axis]:
        raise ValueError("partial-Fourier mask must be one-dimensional")
    indices = (
        acquired.nonzero(as_tuple=False).reshape(-1)
        if is_torch
        else xp.flatnonzero(acquired)
    )
    count = indices.numel() if is_torch else indices.size
    if count == 0:
        raise ValueError("partial-Fourier mask contains no acquired samples")
    first = int(indices[0])
    last = int(indices[-1])
    if count != last - first + 1:
        raise ValueError("partial-Fourier samples must form one contiguous interval")
    if first != 0 and last != acquired.shape[0] - 1:
        raise ValueError("partial-Fourier samples must omit only one k-space edge")
    center = acquired.shape[0] // 2
    if not bool(acquired[center]):
        raise ValueError("partial-Fourier samples must include the k-space center")
    return acquired


def _homodyne_masks(acquired: Any) -> tuple[Any, Any]:
    xp, is_torch = _torch_or_numpy(acquired)
    count = acquired.shape[0]
    indices = xp.arange(count, device=acquired.device) if is_torch else xp.arange(count)
    center = count // 2
    partner = (2 * center - indices) % count
    symmetric = acquired & acquired[partner]
    weight = acquired.to(dtype=xp.float32) if is_torch else acquired.astype(float)
    partner_values = (
        acquired[partner].to(dtype=weight.dtype)
        if is_torch
        else acquired[partner].astype(weight.dtype)
    )
    weight = weight * (2 - partner_values)
    return symmetric, weight


def _broadcast_line(line: Any, ndim: int, axis: int) -> Any:
    shape = [1] * ndim
    shape[axis] = line.shape[0]
    return line.reshape(shape)


def _unit_phase(image: Any) -> Any:
    xp, is_torch = _torch_or_numpy(image)
    magnitude = image.abs() if is_torch else xp.abs(image)
    epsilon = xp.finfo(image.real.dtype).eps
    return (
        image / magnitude.clip(min=epsilon)
        if is_torch
        else image
        / xp.clip(
            magnitude,
            epsilon,
            None,
        )
    )


def _relative_change(current: Any, previous: Any) -> float:
    xp, is_torch = _torch_or_numpy(current)
    if is_torch:
        numerator = xp.linalg.vector_norm((current - previous).reshape(-1))
        denominator = xp.linalg.vector_norm(previous.reshape(-1)).clamp_min(1e-12)
        return float((numerator / denominator).detach().cpu())
    numerator = xp.linalg.norm((current - previous).reshape(-1))
    denominator = max(float(xp.linalg.norm(previous.reshape(-1))), 1e-12)
    return float(numerator / denominator)


def _matching_libraries(first: Any, second: Any) -> tuple[Any, bool]:
    xp, is_torch = _torch_or_numpy(first)
    second_xp, second_is_torch = _torch_or_numpy(second)
    if xp is not second_xp and is_torch != second_is_torch:
        raise TypeError("arrays must use the same array library")
    return xp, is_torch


def _unwrap_last(phase: Any, xp: Any, is_torch: bool) -> Any:
    if not is_torch:
        return xp.unwrap(phase, axis=-1)
    delta = phase[..., 1:] - phase[..., :-1]
    wrapped = (delta + xp.pi) % (2 * xp.pi) - xp.pi
    wrapped = xp.where((wrapped == -xp.pi) & (delta > 0), xp.pi, wrapped)
    correction = xp.cumsum(wrapped - delta, dim=-1)
    result = phase.clone()
    result[..., 1:] += correction
    return result


def _fit_phase_curves(
    phase: Any,
    order: int,
    smoothing: float,
    xp: Any,
    is_torch: bool,
) -> Any:
    coordinate = (
        xp.linspace(
            -1,
            1,
            phase.shape[-1],
            device=phase.device,
            dtype=phase.dtype,
        )
        if is_torch
        else xp.linspace(-1, 1, phase.shape[-1], dtype=phase.dtype)
    )
    design = (
        xp.stack([coordinate**degree for degree in range(order + 1)], dim=1)
        if is_torch
        else xp.stack([coordinate**degree for degree in range(order + 1)], axis=1)
    )
    coefficients = (
        xp.linalg.lstsq(design, phase.T).solution.T
        if is_torch
        else xp.linalg.lstsq(design, phase.T, rcond=None)[0].T
    )
    if smoothing:
        filtered = coefficients.clone() if is_torch else coefficients.copy()
        for index in range(1, filtered.shape[0]):
            filtered[index] = (
                smoothing * filtered[index - 1]
                + (1.0 - smoothing) * coefficients[index]
            )
        coefficients = filtered
    return coefficients @ design.T


def pipe_menon_dcf(
    trajectory: Any,
    image_shape: tuple[int, ...],
    *,
    backend: str = "finufft",
    **kwargs: Any,
) -> Any:
    """Estimate Pipe--Menon density-compensation weights with MRI-NUFFT.

    ``kwargs`` are passed unchanged to the selected backend's ``pipe``
    implementation, including options such as ``max_iter`` and
    normalization. The returned array remains in the array/device ecosystem
    selected by MRI-NUFFT.
    """
    if len(image_shape) not in (2, 3) or any(int(item) < 1 for item in image_shape):
        raise ValueError("image_shape must contain two or three positive entries")
    try:
        density = import_module("mrinufft.density")
    except ImportError as error:
        raise ImportError(
            "Pipe-Menon DCF estimation requires mri-nufft; install "
            "pulserver[recon-cpu] or pulserver[recon-cuda]."
        ) from error
    return density.pipe(
        trajectory,
        tuple(int(item) for item in image_shape),
        backend=backend,
        **kwargs,
    )



def grid_cartesian(
    acquisitions: Any,
    encodes: Any,
    shape: Any,
    *,
    partitions: Any = None,
    echo_position: int | None = None,
) -> tuple[Any, Any]:
    """Scatter acquisitions onto a zero-filled Cartesian grid, with their mask.

    Step one of every Cartesian reconstruction: acquisitions arrive ordered by
    acquisition, each carrying the phase encode it belongs to, and the
    reconstruction needs them on a grid together with a record of which
    positions were actually sampled.

    Partial echo is handled by ``echo_position``. Truncating the samples before
    the echo leaves the acquired window right-aligned against a readout axis
    twice as wide as the part that follows the echo, which is where this places
    them.

    Parameters
    ----------
    acquisitions
        K-space shaped ``(acquisition, coil, sample)``.
    encodes
        Phase encode of each acquisition, as indices into the first grid axis.
    shape
        Phase-encode extent, or ``(n_y, n_z)`` for a 3D acquisition. The
        readout extent comes from the samples and ``echo_position``.
    partitions
        Partition of each acquisition, for a 3D acquisition. Required when
        ``shape`` has two entries and refused when it has one.
    echo_position
        Sample index the echo sits at, as MRD's ``center_sample`` reports it.
        ``None`` treats the readout as a full echo centred on its midpoint.

    Returns
    -------
    grid : numpy.ndarray
        Zero-filled ``(coil, *shape, n_x)`` k-space.
    mask : numpy.ndarray
        Boolean array shaped like ``grid`` without its coil axis, true where a
        sample was acquired.

    Raises
    ------
    ValueError
        If the acquisitions are ragged, the counters do not match the
        acquisition count, or ``partitions`` disagrees with ``shape``.

    Examples
    --------
    >>> import numpy as np
    >>> from pulserver.recon.preprocessing import grid_cartesian
    >>> data = np.ones((4, 2, 16), dtype=complex)
    >>> grid, mask = grid_cartesian(data, [0, 2, 4, 6], 8)
    >>> grid.shape, mask.shape, int(mask.sum())
    ((2, 8, 16), (8, 16), 64)

    A partial echo lands against the full readout width:

    >>> data = np.ones((8, 2, 12), dtype=complex)
    >>> grid, mask = grid_cartesian(data, range(8), 8, echo_position=4)
    >>> grid.shape, bool(mask[0, 0]), bool(mask[0, -1])
    ((2, 8, 16), False, True)
    """
    numpy = import_module("numpy")

    acquisitions = numpy.asarray(acquisitions)
    if acquisitions.ndim != 3:
        raise ValueError(
            "acquisitions must be (acquisition, coil, sample); ragged "
            "acquisitions have to be padded or split first"
        )
    extents = (int(shape),) if numpy.isscalar(shape) else tuple(int(s) for s in shape)
    if len(extents) not in (1, 2):
        raise ValueError("shape must hold one or two phase-encode extents")
    if (partitions is None) != (len(extents) == 1):
        raise ValueError(
            "partitions is required for a 3D shape and refused for a 2D one"
        )

    counters = [numpy.asarray(encodes, dtype=int)]
    if partitions is not None:
        counters.append(numpy.asarray(partitions, dtype=int))
    for counter in counters:
        if counter.shape != (acquisitions.shape[0],):
            raise ValueError("every counter needs one entry per acquisition")

    n_samples = acquisitions.shape[-1]
    position = n_samples // 2 if echo_position is None else int(echo_position)
    n_x = 2 * (n_samples - position)
    acquired = slice(n_x - n_samples, None)

    grid = numpy.zeros((acquisitions.shape[1], *extents, n_x), dtype=numpy.complex64)
    grid[(slice(None), *counters, acquired)] = numpy.moveaxis(acquisitions, 0, 1)

    mask = numpy.zeros((*extents, n_x), dtype=bool)
    mask[(*counters, acquired)] = True
    return grid, mask
