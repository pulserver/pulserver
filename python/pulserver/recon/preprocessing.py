"""Array-level MRI preprocessing utilities.

Functions preserve Torch tensors (including their device) and NumPy arrays.
MRPro containers are delegated to their maintained methods when applicable.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "cartesian_3d_to_2d",
    "coil_compress",
    "correct_epi_eddy_currents",
    "epi_ramp_interpolate",
    "estimate_epi_eddy_phase",
    "noise_prewhiten",
    "remove_readout_oversampling",
]


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
    :func:`pulserver.recon.Cartesian2D` physics.
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
