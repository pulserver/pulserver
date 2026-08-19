"""Array-level MRI preprocessing utilities.

Functions preserve Torch tensors (including their device) and NumPy arrays.
MRPro containers are delegated to their maintained methods when applicable.
"""

from __future__ import annotations

__all__ = [
    "POCS",
    "CartesianGridder",
    "EPIPhaseCorrection",
    "EpiAcquisitionGroups",
    "Homodyne",
    "SmsEpiInputs",
    "cartesian_3d_to_2d",
    "coil_compress",
    "correct_epi_eddy_currents",
    "correct_lines",
    "echo_count",
    "encoded_shape",
    "encoded_volume",
    "epi_ramp_interpolate",
    "estimate_epi_eddy_phase",
    "fftc",
    "fill_partial_echo",
    "grid_cartesian",
    "ifftc",
    "noise_prewhiten",
    "odd_even_fit",
    "partition_epi_acquisitions",
    "pipe_menon_dcf",
    "receiver_channels",
    "recon_shape",
    "recon_volume",
    "remove_readout_oversampling",
]

from collections.abc import Iterable
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from ._fourier import centered_fftn as _centered_fftn
from ._fourier import torch_or_numpy as _torch_or_numpy
from ._mrd.metadata import acquisition_label, has_acquisition_flag
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


def fill_partial_echo(
    kspace: Any,
    readout: Any,
    iterations: int = 12,
    *,
    dimension: int,
) -> Any:
    """Recover the readout edge a partial echo never acquired.

    Parameters
    ----------
    kspace
        K-space over the full readout width, coil-wise or combined.
    readout
        Which readout samples were acquired, over the full width.
    iterations
        POCS iterations.
    dimension
        How many trailing axes of ``kspace`` are spatial: 2 for a slice, 3 for
        a slab. Required rather than inferred, because whether a leading axis
        is coils or partitions is the caller's to know, and guessing it wrong
        fills the wrong axis and says nothing.

    Returns
    -------
    array
        The partial-Fourier image, in the namespace of ``kspace``: the
        reconstruction whose re-encoding reproduces every acquired sample.
    """
    return POCS(dimension=dimension, partial_axis=-1, iterations=iterations)(
        kspace, readout
    )


def fftc(data: Any, *, axes: int | tuple[int, ...] = (-2, -1)) -> Any:
    """Centered orthonormal FFT over one or more axes.

    The ``ifftshift -> fft(norm="ortho") -> fftshift`` an MRI reconstruction
    means by "the Fourier transform", so a plugin states the transform once
    rather than re-deriving the shifts. Torch tensors (device preserved) and
    NumPy arrays both pass through.

    Parameters
    ----------
    data
        The array to transform.
    axes
        Axis or axes to transform over. Default is the last two.

    Returns
    -------
    array
        The transform, in the namespace of ``data``.

    See Also
    --------
    ifftc : the inverse.
    cartesian_3d_to_2d : the readout-decoupling this builds on.
    """
    axes = (axes,) if isinstance(axes, int) else tuple(axes)
    return _centered_fftn(data, axes=axes, inverse=False)


def ifftc(data: Any, *, axes: int | tuple[int, ...] = (-2, -1)) -> Any:
    """Centered orthonormal inverse FFT over one or more axes.

    The inverse of :func:`fftc`; see it for the convention. A single-axis call
    along the readout is the decoupling :func:`cartesian_3d_to_2d` performs.

    Parameters
    ----------
    data
        The array to transform.
    axes
        Axis or axes to transform over. Default is the last two.

    Returns
    -------
    array
        The inverse transform, in the namespace of ``data``.
    """
    axes = (axes,) if isinstance(axes, int) else tuple(axes)
    return _centered_fftn(data, axes=axes, inverse=True)


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
    return _centered_fftn(kspace, axes=(readout_axis,), inverse=True)


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
    image = _centered_fftn(data, axes=(readout_axis,), inverse=True)
    start = (current - target_size) // 2
    selection = [slice(None)] * image.ndim
    selection[readout_axis] = slice(start, start + target_size)
    cropped = image[tuple(selection)]
    return _centered_fftn(cropped, axes=(readout_axis,), inverse=False)


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
            "Coil compression requires mri-nufft, which ships with "
            "pulserver; reinstall the package to restore it."
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
            "Pipe-Menon DCF estimation requires mri-nufft, which ships "
            "with pulserver; reinstall the package to restore it."
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

    n_x, acquired = _cartesian_extent(acquisitions.shape[-1], echo_position)

    grid = numpy.zeros((acquisitions.shape[1], *extents, n_x), dtype=numpy.complex64)
    grid[(slice(None), *counters, acquired)] = numpy.moveaxis(acquisitions, 0, 1)

    mask = numpy.zeros((*extents, n_x), dtype=bool)
    mask[(*counters, acquired)] = True
    return grid, mask


@dataclass(frozen=True)
class EpiAcquisitionGroups:
    """EPI acquisitions partitioned by the roles their MRD flags declare.

    ``phase_correction`` holds the blip-nulled navigator (``NAV``/``REF``),
    ``reverse_polarity`` the optional ``SET=1`` reference volume, and
    ``single_band_reference`` the multiband ``REF`` data. Everything else is
    ``imaging``.
    """

    phase_correction: list[Any]
    single_band_reference: list[Any]
    reverse_polarity: list[Any]
    imaging: list[Any]


def partition_epi_acquisitions(
    acquisitions: Iterable[Any], *, reverse_polarity_set: int = 1
) -> EpiAcquisitionGroups:
    """Sort an EPI stream into the roles its flags and ``idx.set`` declare.

    What every EPI reconstruction does before anything else: the navigator, the
    reverse-polarity reference and the multiband reference each want different
    treatment from the imaging lines, and the sequence has already said which
    is which.

    Parameters
    ----------
    acquisitions
        The acquisitions of one EPI measurement, in arrival order.
    reverse_polarity_set
        ``idx.set`` value reserved for the reverse phase-encode reference.

    Returns
    -------
    EpiAcquisitionGroups
        The four roles, each in arrival order.
    """
    phase_correction: list[Any] = []
    single_band_reference: list[Any] = []
    reverse_polarity: list[Any] = []
    imaging: list[Any] = []
    for acquisition in acquisitions:
        if _any_flag(acquisition, ("ACQ_IS_PHASECORR_DATA", "ACQ_IS_NAVIGATION_DATA")):
            phase_correction.append(acquisition)
        elif acquisition_label(acquisition, "set", 0) == reverse_polarity_set:
            reverse_polarity.append(acquisition)
        elif _any_flag(
            acquisition,
            (
                "ACQ_IS_PARALLEL_CALIBRATION",
                "ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING",
            ),
        ):
            single_band_reference.append(acquisition)
        else:
            imaging.append(acquisition)
    return EpiAcquisitionGroups(
        phase_correction, single_band_reference, reverse_polarity, imaging
    )


def odd_even_fit(navigator_lines: list[Any]) -> tuple[float, float]:
    """Fit the odd/even linear phase from blip-nulled navigator lines.

    The middle line was read backwards; against the mean of its two
    like-polarity neighbours, its hybrid-space phase difference is the
    gradient-delay ramp plus a constant -- the two numbers every reversed
    line is corrected by.

    Parameters
    ----------
    navigator_lines : list of numpy.ndarray
        Three ``(coils, samples)`` lines, polarity ``+ - +``, reversed lines
        already flipped back into readout order.

    Returns
    -------
    tuple of float
        Phase slope (radians per sample) and intercept (radians).
    """
    import numpy as np

    forward = 0.5 * (_hybrid(navigator_lines[0]) + _hybrid(navigator_lines[2]))
    backward = _hybrid(navigator_lines[1])
    cross = np.sum(forward * np.conj(backward), axis=0)

    weights = np.abs(cross)
    phase = np.unwrap(np.angle(cross))
    samples = np.arange(phase.size)
    keep = weights > 0.1 * weights.max()
    slope, intercept = np.polyfit(samples[keep], phase[keep], 1, w=weights[keep])
    return float(slope), float(intercept)


def correct_lines(
    lines: list[tuple[Any, bool]], slope: float, intercept: float
) -> list[Any]:
    """Flip and phase-correct a train's lines into a consistent readout.

    Parameters
    ----------
    lines : list of tuple
        ``(data, reversed)`` per line, data ``(coils, samples)``.
    slope, intercept
        The odd/even fit of :func:`odd_even_fit`.

    Returns
    -------
    list of numpy.ndarray
        The corrected lines, all in forward readout order.
    """
    import numpy as np

    corrected = []
    for data, backwards in lines:
        row = np.asarray(data)
        if backwards:
            row = row[..., ::-1]
            hybrid = _hybrid(row)
            ramp = slope * np.arange(hybrid.shape[-1]) + intercept
            hybrid = hybrid * np.exp(1j * ramp)
            row = fftc(hybrid, axes=-1)
        corrected.append(row.astype(np.complex64))
    return corrected


def encoded_shape(header: Any, *, encoding: int = 0) -> tuple[int, int, int]:
    """Return the ``(n_slices, n_y, n_x)`` grid an MRD header describes.

    The encoded space, so the readout extent is the oversampled one the scanner
    actually digitises and the phase-encode extent covers the whole prescribed
    matrix rather than the lines one acceleration happens to sample. This is
    the grid a reconstruction allocates; :func:`recon_shape` is what it crops to
    at the end.

    Parameters
    ----------
    header
        Parsed MRD XML header.
    encoding
        Encoding space to read. Navigator data lives in its own.

    Returns
    -------
    tuple of int
        Slices, phase encodes, readout samples.

    Raises
    ------
    ValueError
        If the header carries no such encoded space.
    """
    space = _encoding_space(header, encoding, "encodedSpace")
    limits = getattr(_encoding(header, encoding), "encodingLimits", None)
    slices = getattr(limits, "slice", None)
    n_slices = 1 if slices is None else int(slices.maximum) + 1
    return n_slices, int(space.matrixSize.y), int(space.matrixSize.x)


def recon_shape(header: Any, *, encoding: int = 0) -> tuple[int, int]:
    """Return the ``(n_y, n_x)`` image matrix an MRD header asks for.

    The reconstructed space, which is the encoded one with readout oversampling
    and any phase field-of-view oversampling taken back off.

    Parameters
    ----------
    header
        Parsed MRD XML header.
    encoding
        Encoding space to read.

    Returns
    -------
    tuple of int
        Phase encodes, readout samples.

    Raises
    ------
    ValueError
        If the header carries no such reconstructed space.
    """
    space = _encoding_space(header, encoding, "reconSpace")
    return int(space.matrixSize.y), int(space.matrixSize.x)


def encoded_volume(header: Any, *, encoding: int = 0) -> tuple[int, int, int]:
    """Return the ``(n_z, n_y, n_x)`` grid an MRD header describes for a slab.

    The volume counterpart of :func:`encoded_shape`: the partition extent comes
    from the encoded matrix rather than the slice counter, because a 3D scan
    encodes z instead of stepping through it.

    Parameters
    ----------
    header
        Parsed MRD XML header.
    encoding
        Encoding space to read.

    Returns
    -------
    tuple of int
        Partitions, phase encodes, readout samples.

    Raises
    ------
    ValueError
        If the header carries no such encoded space.
    """
    return _volume(header, encoding, "encodedSpace")


def recon_volume(header: Any, *, encoding: int = 0) -> tuple[int, int, int]:
    """Return the ``(n_z, n_y, n_x)`` image matrix an MRD header asks for.

    Parameters
    ----------
    header
        Parsed MRD XML header.
    encoding
        Encoding space to read.

    Returns
    -------
    tuple of int
        Partitions, phase encodes, readout samples.

    Raises
    ------
    ValueError
        If the header carries no such reconstructed space.
    """
    return _volume(header, encoding, "reconSpace")


def echo_count(header: Any, *, encoding: int = 0) -> int:
    """Return how many echoes the MRD header declares.

    A sequence's ``ECO`` label arrives as the ``contrast`` counter, so its
    encoding limit is the echo count. A header without one describes a
    single-echo scan.

    Parameters
    ----------
    header
        Parsed MRD XML header.
    encoding
        Encoding space to read.

    Returns
    -------
    int
        Echoes per repetition.
    """
    try:
        limits = header.encoding[encoding].encodingLimits.contrast
    except (AttributeError, IndexError, TypeError):
        return 1
    return 1 if limits is None else int(limits.maximum) + 1


def receiver_channels(header: Any) -> int:
    """Return the number of receive channels an MRD header declares.

    Parameters
    ----------
    header
        Parsed MRD XML header.

    Returns
    -------
    int
        Coils in every acquisition of the scan.

    Raises
    ------
    ValueError
        If the header does not declare a channel count.
    """
    system = getattr(header, "acquisitionSystemInformation", None)
    channels = getattr(system, "receiverChannels", None)
    if channels is None:
        raise ValueError("MRD header declares no receiverChannels")
    return int(channels)


def _any_flag(acquisition: Any, flags: tuple[str, ...]) -> bool:
    return any(has_acquisition_flag(acquisition, flag) for flag in flags)


def _hybrid(rows: Any) -> Any:
    """Rows into hybrid space: inverse FFT along the readout."""
    import numpy as np

    return ifftc(np.asarray(rows), axes=-1)


def _encoding(header: Any, index: int) -> Any:
    try:
        return header.encoding[index]
    except (AttributeError, IndexError, TypeError) as error:
        raise ValueError(
            f"MRD header carries no encoding {index}; an inline reconstruction "
            "needs the header the scanner sends ahead of the data"
        ) from error


def _volume(header: Any, index: int, name: str) -> tuple[int, int, int]:
    matrix = _encoding_space(header, index, name).matrixSize
    return int(matrix.z), int(matrix.y), int(matrix.x)


def _encoding_space(header: Any, index: int, name: str) -> Any:
    space = getattr(_encoding(header, index), name, None)
    if space is None or getattr(space, "matrixSize", None) is None:
        raise ValueError(f"MRD encoding {index} carries no {name} matrix size")
    return space


def _cartesian_extent(n_samples: int, echo_position: int | None) -> tuple[int, slice]:
    """Full readout width, and where the acquired samples land in it.

    Truncating the samples before the echo leaves the acquired window
    right-aligned against a readout axis twice as wide as the part that follows
    the echo.
    """
    n_samples = int(n_samples)
    position = n_samples // 2 if echo_position is None else int(echo_position)
    n_x = 2 * (n_samples - position)
    return n_x, slice(n_x - n_samples, None)


class CartesianGridder:
    """A scan's Cartesian k-space, filled one acquisition at a time.

    Allocated up front from the grid the header describes, so a reconstruction
    can place each acquisition the moment it arrives and read a finished unit
    out by index without a second pass. Indexing returns the ``(kspace, mask)``
    of one position along the leading axis -- one slice of a multi-slice scan --
    and :meth:`result` the whole buffer.

    A readout shorter than the grid is right-aligned in it: truncating the
    samples before the echo is what a partial echo does, so the acquired window
    ends where a full one would.

    Parameters
    ----------
    shape
        Full grid without the coil axis, readout last: ``(n_y, n_x)`` for a
        single plane, ``(n_slices, n_y, n_x)`` for a multi-slice scan,
        ``(n_z, n_y, n_x)`` for a volume.
    coils
        Number of receive channels.

    Attributes
    ----------
    kspace : numpy.ndarray
        Zero-filled ``(coils, *shape)`` k-space.
    mask : numpy.ndarray
        Boolean array shaped ``shape``, true where a sample was acquired.

    Raises
    ------
    ValueError
        If ``shape`` has fewer than two axes, or an acquisition does not fit
        the grid it is placed in.

    Examples
    --------
    >>> import numpy as np
    >>> from pulserver.recon.preprocessing import CartesianGridder
    >>> buffer = CartesianGridder((2, 8, 16), coils=4)
    >>> for line in range(0, 8, 2):
    ...     buffer.add(np.ones((4, 16)), 1, line)
    >>> kspace, mask = buffer[1]
    >>> kspace.shape, mask.shape, int(mask.sum())
    ((4, 8, 16), (8, 16), 64)

    An unfilled position is empty, which is what makes the mask the record of
    what the scan actually sampled:

    >>> bool(buffer[0][1].any())
    False

    A partial echo lands against the full readout width:

    >>> partial = CartesianGridder((4, 16), coils=1)
    >>> partial.add(np.ones((1, 12)), 0)
    >>> bool(partial.mask[0, 0]), bool(partial.mask[0, -1])
    (False, True)
    """

    def __init__(self, shape: Any, *, coils: int) -> None:
        numpy = import_module("numpy")

        extents = tuple(int(s) for s in shape)
        if len(extents) < 2:
            raise ValueError("shape needs at least a phase-encode and a readout axis")
        coils = int(coils)
        if coils < 1:
            raise ValueError("coils must be positive")
        self.shape = extents
        self.coils = coils
        self.kspace = numpy.zeros((coils, *extents), dtype=numpy.complex64)
        self.mask = numpy.zeros(extents, dtype=bool)

    def add(self, acquisition: Any, *index: int) -> None:
        """Place one ``(coil, sample)`` acquisition at its position in the grid.

        Parameters
        ----------
        acquisition
            K-space of one readout, shaped ``(coils, samples)``.
        *index
            Position along every axis of ``shape`` except the readout, in the
            same order.

        Raises
        ------
        ValueError
            If the acquisition is not two-dimensional, the index does not name
            a position, or either exceeds the grid.
        """
        numpy = import_module("numpy")

        acquisition = numpy.asarray(acquisition)
        if acquisition.ndim != 2:
            raise ValueError("each acquisition must be (coil, sample)")
        if acquisition.shape[0] != self.coils:
            raise ValueError(
                f"acquisition carries {acquisition.shape[0]} coils, grid holds "
                f"{self.coils}"
            )
        if len(index) != len(self.shape) - 1:
            raise ValueError(
                f"grid needs {len(self.shape) - 1} index values, got {len(index)}"
            )
        n_x = self.shape[-1]
        n_samples = acquisition.shape[-1]
        if n_samples > n_x:
            raise ValueError(
                f"acquisition has {n_samples} samples, readout axis holds {n_x}"
            )
        position = tuple(int(value) for value in index)
        acquired = slice(n_x - n_samples, None)
        self.kspace[(slice(None), *position, acquired)] = acquisition
        self.mask[(*position, acquired)] = True

    def __getitem__(self, index: Any) -> tuple[Any, Any]:
        """Return the ``(kspace, mask)`` at one position along the leading axis."""
        return self.kspace[:, index], self.mask[index]

    def result(self) -> tuple[Any, Any]:
        """Return the whole ``(kspace, mask)`` buffer."""
        return self.kspace, self.mask
