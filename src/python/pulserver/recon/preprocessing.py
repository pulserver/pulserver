"""Array-level MRI preprocessing utilities.

Functions preserve Torch tensors (including their device) and NumPy arrays.
"""

from __future__ import annotations

__all__ = [
    "POCS",
    "Homodyne",
    "coil_compress",
    "correct_lines",
    "epi_ramp_interpolate",
    "epi_ramp_positions",
    "estimate_epi_phase",
    "fftc",
    "fill_partial_echo",
    "ifftc",
    "noise_prewhiten",
    "pipe_menon_dcf",
    "remove_readout_oversampling",
]

from importlib import import_module
from typing import Any

from ._fourier import centered_fftn as _centered_fftn
from ._fourier import torch_or_numpy as _torch_or_numpy


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
    method: str = "pocs",
) -> Any:
    """Recover the readout edge a partial echo never acquired.

    Both estimators rest on the same fact -- an image whose phase varies slowly
    is nearly conjugate symmetric in k-space, so the missing edge is implied by
    the acquired one. :class:`POCS` iterates towards an image that reproduces
    every acquired sample; :class:`Homodyne` reaches an answer in one pass by
    weighting the acquired half and demodulating the low-resolution phase.
    POCS is the more faithful of the two and Homodyne the cheaper, which is the
    choice a scanner-side reconstruction is actually making.

    Parameters
    ----------
    kspace
        K-space over the full readout width, coil-wise or combined.
    readout
        Which readout samples were acquired, over the full width.
    iterations
        POCS iterations. Homodyne takes one pass and ignores this.
    dimension
        How many trailing axes of ``kspace`` are spatial: 2 for a slice, 3 for
        a slab. Required rather than inferred, because whether a leading axis
        is coils or partitions is the caller's to know, and guessing it wrong
        fills the wrong axis and says nothing.
    method
        ``"pocs"`` or ``"homodyne"``.

    Returns
    -------
    array
        The partial-Fourier image, in the namespace of ``kspace``: for POCS,
        the reconstruction whose re-encoding reproduces every acquired sample.

    Raises
    ------
    ValueError
        If ``method`` names neither estimator.
    """
    if method == "pocs":
        return POCS(dimension=dimension, partial_axis=-1, iterations=iterations)(
            kspace, readout
        )
    if method == "homodyne":
        return Homodyne(dimension=dimension, partial_axis=-1)(kspace, readout)
    raise ValueError(f"method must be pocs or homodyne, got {method!r}")


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
    """
    axes = (axes,) if isinstance(axes, int) else tuple(axes)
    return _centered_fftn(data, axes=axes, inverse=False)


def ifftc(data: Any, *, axes: int | tuple[int, ...] = (-2, -1)) -> Any:
    """Centered orthonormal inverse FFT over one or more axes.

    The inverse of :func:`fftc`; see it for the convention. A single-axis call
    along the readout decouples a Cartesian volume into independent planes.

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


def remove_readout_oversampling(
    data: Any,
    target_size: int,
    *,
    readout_axis: int = -1,
) -> Any:
    """Remove readout oversampling by centered image-domain cropping.

    A scanner digitises more samples than the prescribed matrix so the readout
    filter has room to roll off, and those extra samples are field of view, not
    resolution: the crop is in the image domain, and what comes back is the
    same k-space over the prescribed width.

    Parameters
    ----------
    data
        K-space with the readout along ``readout_axis``.
    target_size
        Samples the prescribed matrix asks for --
        :attr:`~pulserver.recon.ReconBuffer.image_shape` last entry.
    readout_axis
        Which axis the readout runs along.

    Returns
    -------
    array
        K-space over ``target_size`` samples, in the namespace of ``data``.

    Raises
    ------
    ValueError
        If ``target_size`` is not within the samples there are.
    """
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
    """Compress the receive array onto its principal channels.

    A receive array measures the same object through every element, so the
    channels are strongly correlated and most of what they carry lives in far
    fewer of them. The principal components of the sample covariance are those
    channels: keeping the leading ones is the linear combination that retains
    the most energy for a given count, and everything downstream -- the
    sensitivities, the solve, the buffers -- then runs on the smaller array.

    Parameters
    ----------
    kspace
        The measurement, ``(coils, samples)``. Torch tensors keep their device.
    n_coils
        Virtual channels to keep, as a count, or the fraction of the energy to
        retain when given as a float in ``(0, 1]``. A count beyond the physical
        channels keeps them all.
    trajectory
        Where each sample was taken, ``(samples, dimensions)``. Only needed
        with ``calibration_radius``.
    calibration_radius
        Estimate the components from the samples inside this fraction of the
        maximum k-space radius, rather than from every sample -- the centre is
        where the array's correlations are, and where the object is brightest.

    Returns
    -------
    compressed : array
        ``(n_coils, samples)`` in the virtual basis.
    matrix : array
        ``(n_coils, coils)``, the basis itself, for compressing anything else
        the same way -- the acquisitions that arrive after a calibration
        established it, or the sensitivities they are solved against.

    Raises
    ------
    ValueError
        If ``kspace`` is not two-dimensional, ``n_coils`` asks for nothing, or
        ``calibration_radius`` is given without a trajectory.
    """
    xp, _ = _torch_or_numpy(kspace)
    if kspace.ndim != 2:
        raise ValueError(f"kspace must be (coils, samples), got {kspace.shape}")
    if (calibration_radius is None) != (trajectory is None):
        raise ValueError("calibration_radius and trajectory are given together")

    region = kspace
    if calibration_radius is not None:
        radius = xp.sqrt(xp.sum(trajectory**2, axis=-1))
        region = kspace[:, radius < calibration_radius * radius.max()]

    # Eigenvectors of the sample covariance, largest eigenvalue first. eigh
    # returns them as columns, and ascending, so the order is applied to the
    # second axis and the rows of the result are the virtual channels.
    values, vectors = xp.linalg.eigh(region @ region.conj().T)
    order = xp.argsort(-values)
    energies = [float(values[int(index)]) for index in order]

    keep = _retained_channels(n_coils, energies)
    matrix = vectors[:, order[:keep]].conj().T
    return matrix @ kspace, matrix


def _retained_channels(n_coils: int | float, energies: list[float]) -> int:
    """How many principal channels a count or an energy fraction asks for."""
    if isinstance(n_coils, float) and not float(n_coils).is_integer():
        if not 0.0 < n_coils <= 1.0:
            raise ValueError(f"an energy fraction must lie in (0, 1], got {n_coils}")
        total = sum(energies)
        if total <= 0.0:
            return 1
        running = 0.0
        for count, energy in enumerate(energies, start=1):
            running += energy
            if running / total >= n_coils:
                return count
        return len(energies)
    keep = int(n_coils)
    if keep < 1:
        raise ValueError(f"n_coils must keep at least one channel, got {n_coils}")
    return min(keep, len(energies))


def noise_prewhiten(
    kspace: Any,
    noise: Any,
    *,
    coil_axis: int = -2,
    scale_factor: float = 1.0,
) -> Any:
    """Decorrelate receiver coils using the measured noise covariance.

    A receive array's channels see correlated noise, and a solve that assumes
    they do not weights them wrongly. The noise scan measures that covariance;
    whitening is the Cholesky solve that turns it into the identity, so every
    channel afterwards carries unit, independent noise.

    Parameters
    ----------
    kspace
        The measurement, with the channels along ``coil_axis``.
    noise
        The noise scan, channels along the same axis.
    coil_axis
        Which axis the channels run along.
    scale_factor
        Applied to the whitened data, for a caller keeping a known noise level.

    Returns
    -------
    array
        The whitened measurement, in the namespace of ``kspace``.

    Raises
    ------
    TypeError
        If the measurement and the noise are not in the same array library.
    """
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


def epi_ramp_positions(
    n_samples: int,
    dwell: float,
    *,
    ramp_up: float,
    flat_top: float,
    ramp_down: float,
    delay: float = 0.0,
) -> tuple[Any, Any]:
    """Where a ramp-sampled EPI readout took its samples, and where they belong.

    Sampling only the plateau throws away the time the ramps take, which is why
    a readout worth playing samples across them too. The cost is that k no
    longer advances at a constant rate: during a ramp the gradient is still
    rising, so the early samples are packed together in k and the plateau's are
    evenly spread. Placing them on the grid as if they were uniform is a
    geometric distortion along the readout, and the fix is to say where each
    one actually landed and resample.

    k is the integral of the gradient, so the position of a sample is the area
    the trapezoid has swept by the time it is taken.

    Parameters
    ----------
    n_samples
        Samples the ADC took.
    dwell
        Seconds between them.
    ramp_up, flat_top, ramp_down
        The read lobe's timing, in seconds.
    delay
        Seconds from the start of the lobe to the first sample.

    Returns
    -------
    sampled : numpy.ndarray
        Where each sample was taken, in k, normalised so the whole lobe sweeps
        ``[-0.5, 0.5]``.
    uniform : numpy.ndarray
        The grid they belong on: ``n_samples`` evenly spaced positions spanning
        what was actually swept.

    Raises
    ------
    ValueError
        If the lobe has no duration, the dwell is not positive, or the samples
        run past the end of the lobe.

    See Also
    --------
    epi_ramp_interpolate : resamples a readout from one onto the other.
    """
    import numpy as np

    if n_samples < 2:
        raise ValueError(f"a readout is at least two samples, got {n_samples}")
    if dwell <= 0.0:
        raise ValueError(f"dwell must be positive, got {dwell}")
    if min(ramp_up, flat_top, ramp_down) < 0.0 or ramp_up + flat_top + ramp_down <= 0.0:
        raise ValueError("the read lobe must have a positive duration")

    times = delay + (np.arange(n_samples) + 0.5) * dwell
    duration = ramp_up + flat_top + ramp_down
    if times[0] < 0.0 or times[-1] > duration:
        raise ValueError(
            f"the samples span {times[0]:.3e}..{times[-1]:.3e} s, outside the "
            f"{duration:.3e} s lobe"
        )

    # Area swept by each phase of the trapezoid, at unit amplitude. Every
    # branch is evaluated, so a phase of zero duration divides by one and is
    # then discarded rather than overflowing on the way.
    rising = np.square(times) / (2.0 * (ramp_up or 1.0))
    plateau = 0.5 * ramp_up + (times - ramp_up)
    fallen = np.clip((times - ramp_up - flat_top) / (ramp_down or 1.0), 0.0, 1.0)
    falling = (
        0.5 * ramp_up + flat_top + 0.5 * ramp_down * (1.0 - np.square(1.0 - fallen))
    )
    swept = np.where(
        times <= ramp_up,
        rising,
        np.where(times <= ramp_up + flat_top, plateau, falling),
    )
    total = 0.5 * ramp_up + flat_top + 0.5 * ramp_down
    sampled = swept / total - 0.5
    return sampled, np.linspace(sampled[0], sampled[-1], n_samples)


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


def estimate_epi_phase(
    navigator_lines: list[Any],
    *,
    polynomial_order: int = 1,
) -> Any:
    """Fit the odd/even phase an EPI readout carries, from a blip-nulled navigator.

    Reversing a readout does not reverse the delays it was played through, so a
    line read backwards carries a phase its forward neighbours do not, and
    leaving it there is what puts a ghost at half the field of view. The
    navigator measures it directly: three blip-nulled lines of alternating
    polarity see the same object, so the phase difference between the middle
    one and its neighbours is that phase and nothing else.

    A first-order fit is the correction every product reconstruction applies,
    because the term that dominates is the gradient and ADC delay and a delay
    is linear in the readout. Raising the order picks up what eddy currents
    leave beyond it, which an oblique or a strongly driven readout has more of.

    The fit is weighted by the cross-correlation magnitude and ignores the
    samples below a tenth of its peak: a phase difference where there is no
    signal is noise, and letting the readout's empty edges into an unweighted
    fit is what drags a high-order one off.

    Parameters
    ----------
    navigator_lines : list of numpy.ndarray
        Three ``(coils, samples)`` lines, polarity ``+ - +``, reversed lines
        already flipped back into readout order.
    polynomial_order
        Order of the phase polynomial. ``1`` is the gradient-delay ramp and a
        constant.

    Returns
    -------
    numpy.ndarray
        Coefficients of the phase, lowest order first, in a coordinate running
        from ``-1`` to ``1`` across the readout -- so a fit measured on the
        navigator applies to a readout of any length.

    Raises
    ------
    ValueError
        If fewer than three navigator lines are given, or the order is
        negative.
    """
    import numpy as np

    if len(navigator_lines) < 3:
        raise ValueError(
            f"the navigator is three lines of alternating polarity, got "
            f"{len(navigator_lines)}"
        )
    if polynomial_order < 0:
        raise ValueError("polynomial_order must be non-negative")

    forward = 0.5 * (_hybrid(navigator_lines[0]) + _hybrid(navigator_lines[2]))
    backward = _hybrid(navigator_lines[1])
    cross = np.sum(forward * np.conj(backward), axis=0)

    weights = np.abs(cross)
    phase = np.unwrap(np.angle(cross))
    coordinate = np.linspace(-1.0, 1.0, phase.size)
    keep = weights > 0.1 * weights.max()
    return np.polynomial.polynomial.polyfit(
        coordinate[keep], phase[keep], polynomial_order, w=weights[keep]
    )


def correct_lines(lines: list[tuple[Any, bool]], phase: Any = None) -> list[Any]:
    """Flip and phase-correct a train's lines into a consistent readout.

    The forward lines define the grid, so a reversed one is rotated onto them
    rather than both being met in the middle: the correction is one-sided, and
    the image does not move.

    Parameters
    ----------
    lines : list of tuple
        ``(data, reversed)`` per line, data ``(coils, samples)``.
    phase
        The polynomial coefficients :func:`estimate_epi_phase` returned.
        ``None`` -- before a navigator has arrived -- flips a reversed line
        without demodulating it.

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
            if phase is not None:
                hybrid = _hybrid(row)
                coordinate = np.linspace(-1.0, 1.0, hybrid.shape[-1])
                ramp = np.polynomial.polynomial.polyval(coordinate, phase)
                row = fftc(hybrid * np.exp(1j * ramp), axes=-1)
        corrected.append(row.astype(np.complex64))
    return corrected


def _hybrid(rows: Any) -> Any:
    """Rows into hybrid space: inverse FFT along the readout."""
    import numpy as np

    return ifftc(np.asarray(rows), axes=-1)
