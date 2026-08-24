"""Array-level MRI preprocessing utilities.

Functions preserve Torch tensors (including their device) and NumPy arrays.
"""

from __future__ import annotations

__all__ = [
    "POCS",
    "Homodyne",
    "coil_compress",
    "correct_lines",
    "epi_ramp_operator",
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

    Examples
    --------
    >>> import numpy as np
    >>> import pulserver.mrd as mrd
    >>> truncated = np.zeros((1, 16, 16), dtype=complex)
    >>> truncated[..., :10] = 1.0
    >>> readout = np.zeros(16)
    >>> readout[:10] = 1.0
    >>> mrd.Homodyne(dimension=2, partial_axis=-1)(truncated, readout).shape
    (1, 16, 16)

    One pass rather than POCS's iteration; ``fill_partial_echo`` reaches it by
    name, and its figure is the two side by side.
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

    Examples
    --------
    >>> import numpy as np
    >>> import pulserver.mrd as mrd
    >>> truncated = np.zeros((1, 16, 16), dtype=complex)
    >>> truncated[..., :10] = 1.0
    >>> readout = np.zeros(16)
    >>> readout[:10] = 1.0
    >>> mrd.POCS(dimension=2, partial_axis=-1, iterations=4)(truncated, readout).shape
    (1, 16, 16)

    ``fill_partial_echo`` reaches this by name, and its figure is the two
    estimators side by side.
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

    Examples
    --------
    >>> import numpy as np
    >>> import pulserver.mrd as mrd
    >>> image = np.zeros((16, 16), dtype=complex)
    >>> image[6:10, 6:10] = 1.0
    >>> truncated = mrd.fftc(image)
    >>> truncated[:, 12:] = 0
    >>> readout = np.ones(16)
    >>> readout[12:] = 0

    Both estimators answer for the same truncation, and the name is what a
    plugin exposes as its ``partial_fourier`` setting:

    >>> pocs = mrd.fill_partial_echo(truncated[None], readout, dimension=2)
    >>> homodyne = mrd.fill_partial_echo(
    ...     truncated[None], readout, dimension=2, method="homodyne"
    ... )
    >>> pocs.shape == homodyne.shape
    True

    What the truncation costs, and what the conjugate symmetry buys back:

    .. plot::

       import numpy as np
       import pulserver.mrd as mrd
       from _figures import images, phantom

       truth = phantom(64)[0][0].numpy()
       truncated = mrd.fftc(truth)
       truncated[:, 40:] = 0.0
       readout = np.ones(64)
       readout[40:] = 0.0
       images(
           [
               ("object", truth),
               ("zero-filled", mrd.ifftc(truncated)),
               ("POCS", mrd.fill_partial_echo(truncated[None], readout, dimension=2)),
               (
                   "Homodyne",
                   mrd.fill_partial_echo(
                       truncated[None], readout, dimension=2, method="homodyne"
                   ),
               ),
           ],
           title="fill_partial_echo: two estimators for the same truncation",
       )
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

    Examples
    --------
    >>> import numpy as np
    >>> import pulserver.mrd as mrd
    >>> image = np.zeros((8, 8), dtype=complex)
    >>> image[4, 4] = 1.0
    >>> kspace = mrd.fftc(image)

    A point at the centre of the image is flat in k-space, and the round trip
    is exact:

    >>> bool(np.allclose(np.abs(kspace), np.abs(kspace).mean()))
    True
    >>> bool(np.allclose(mrd.ifftc(kspace), image, atol=1e-12))
    True
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

    Examples
    --------
    >>> import numpy as np
    >>> import pulserver.mrd as mrd
    >>> kspace = np.ones((8, 8), dtype=complex)
    >>> image = mrd.ifftc(kspace)
    >>> bool(np.allclose(mrd.fftc(image), kspace, atol=1e-12))
    True
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

    Examples
    --------
    >>> import numpy as np
    >>> import pulserver.mrd as mrd
    >>> readout = mrd.ifftc(np.ones((4, 128)), axes=-1)
    >>> mrd.remove_readout_oversampling(readout, 64).shape
    (4, 64)
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

    Examples
    --------
    >>> import numpy as np
    >>> import pulserver.mrd as mrd
    >>> rng = np.random.default_rng(0)
    >>> lines = rng.normal(size=(8, 256)) + 1j * rng.normal(size=(8, 256))

    Eight channels carrying four independent signals compress onto four
    without loss, and the basis comes back to apply to everything that
    follows:

    >>> lines[4:] = lines[:4]
    >>> compressed, basis = mrd.coil_compress(lines, 4)
    >>> compressed.shape, basis.shape
    ((4, 256), (4, 8))
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

    Examples
    --------
    >>> import numpy as np
    >>> import pulserver.mrd as mrd
    >>> rng = np.random.default_rng(0)
    >>> noise = rng.normal(size=(4, 512)) + 1j * rng.normal(size=(4, 512))

    Whitening the noise scan against itself leaves an identity covariance,
    which is what every readout that follows is measured against:

    >>> whitened = mrd.noise_prewhiten(noise, noise, coil_axis=0)
    >>> covariance = whitened @ whitened.conj().T / whitened.shape[-1]
    >>> bool(np.allclose(covariance, np.eye(4), atol=5e-2))
    True
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


def epi_ramp_operator(
    sample_positions: Any,
    target_positions: Any,
    support: int,
    *,
    regularization: float = 1e-6,
) -> Any:
    """Resample a readout from where it was taken onto where it belongs.

    A readout is band-limited: it is the transform of an object that occupies
    ``support`` pixels and nothing outside them. So samples taken anywhere
    determine it everywhere, and moving them onto the grid is not an
    approximation but a change of basis -- the least-squares inverse of the
    non-uniform transform, followed by the uniform one. The operator's entries
    are the sinc-like kernels that implies.

    Linear interpolation is the cheap stand-in for this and is visibly worse:
    over a readout whose ramps take half its duration, this resampling is exact
    to numerical precision where a linear one leaves seven percent.

    What makes it exact is that the samples outnumber the pixels they have to
    determine, which is what readout oversampling buys. Where they do not --
    where the fast part of the sweep steps further than ``1 / support`` -- the
    readout has aliased and no resampling recovers it; ``regularization`` keeps
    the solve from amplifying that, it does not undo it.

    One lobe is played for every readout of a train, so the operator is built
    once and applied to each.

    Parameters
    ----------
    sample_positions
        Where each sample was taken, in k, normalised so the readout spans at
        most ``[-0.5, 0.5]``. The trajectory an acquisition carries: a client
        attaches one exactly when the gradient was still moving under the ADC,
        which is when a readout needs this.
    target_positions
        Where they belong -- the uniform grid, in the same units.
    support
        Pixels the object occupies along the readout: the reconstructed matrix,
        not the oversampled one the scanner digitised.
    regularization
        Tikhonov weight on the normal equations, relative to the sample count.

    Returns
    -------
    numpy.ndarray
        ``(target, source)``. Applying it to a ``(coils, samples)`` readout is
        ``readout @ operator.T``.

    Raises
    ------
    ValueError
        If either position set does not describe a readout, or ``support`` is
        not positive.

    Examples
    --------
    >>> import numpy as np
    >>> import pulserver.mrd as mrd
    >>> sampled = np.sin(np.linspace(-np.pi / 2, np.pi / 2, 16))
    >>> uniform = np.linspace(-1, 1, 16)
    >>> operator = mrd.epi_ramp_operator(sampled, uniform, 3)
    >>> operator.shape
    (16, 16)
    """
    import numpy as np

    sample_positions = np.asarray(sample_positions, dtype=float).reshape(-1)
    target_positions = np.asarray(target_positions, dtype=float).reshape(-1)
    support = int(support)
    if sample_positions.size < 2 or target_positions.size < 2:
        raise ValueError("both position sets must describe a readout")
    if support < 1:
        raise ValueError(f"support must be positive, got {support}")

    grid = np.arange(support) - support // 2
    taken = np.exp(-2j * np.pi * np.outer(sample_positions, grid))
    wanted = np.exp(-2j * np.pi * np.outer(target_positions, grid))
    normal = taken.conj().T @ taken + regularization * sample_positions.size * np.eye(
        support
    )
    return (wanted @ np.linalg.solve(normal, taken.conj().T)).astype(np.complex64)


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

    Examples
    --------
    >>> import numpy as np
    >>> import pulserver.mrd as mrd
    >>> angles = np.linspace(0, np.pi, 8, endpoint=False)
    >>> radius = np.linspace(-0.5, 0.5, 32)
    >>> trajectory = np.stack(
    ...     [np.outer(np.cos(angles), radius), np.outer(np.sin(angles), radius)], -1
    ... ).reshape(-1, 2)
    >>> weights = np.asarray(mrd.pipe_menon_dcf(trajectory, (16, 16)))
    >>> weights.shape, weights.dtype.kind
    ((256,), 'f')
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
    weights = density.pipe(
        trajectory,
        tuple(int(item) for item in image_shape),
        backend=backend,
        **kwargs,
    )
    # The estimator works in the complex domain and answers there; a density
    # is real, and a complex one propagates into every operator that carries
    # it and into the transfer kernels built from it. The real part of a
    # complex array is a strided view, which the backends will not take, so it
    # is materialized here.
    real = getattr(weights, "real", None)
    if real is None:
        return weights
    contiguous = getattr(real, "contiguous", None)
    if callable(contiguous):
        return contiguous()
    return import_module("numpy").ascontiguousarray(real)


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

    Examples
    --------
    >>> import numpy as np
    >>> import pulserver.mrd as mrd
    >>> line = np.exp(-np.linspace(-2, 2, 32) ** 2).astype(np.complex64)
    >>> line = line[None].repeat(2, 0)
    >>> ramp = np.exp(1j * 0.15 * np.arange(32)).astype(np.complex64)
    >>> fit = mrd.estimate_epi_phase([line * ramp, line, line * ramp])
    >>> fit.shape
    (2,)
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

    Examples
    --------
    A reversed line carries the delay it was played through, and leaving it
    there is what puts a copy of the object at half the field of view.
    ``phase=None`` is the same train flipped but not demodulated, which is
    what the ghost is:

    .. plot::

       import numpy as np
       import pulserver.mrd as mrd
       from _figures import images, phantom

       truth = phantom(64)[0][0].numpy()
       kspace = mrd.fftc(truth)
       ramp = 0.9 * np.linspace(-1.0, 1.0, 64) + 0.4

       def backwards(line):
           hybrid = mrd.ifftc(line, axes=-1)
           return mrd.fftc(hybrid * np.exp(-1j * ramp), axes=-1)[..., ::-1]

       train = [
           (line[None], False) if index % 2 == 0 else (backwards(line[None]), True)
           for index, line in enumerate(kspace)
       ]
       middle = kspace[32][None]
       phase = mrd.estimate_epi_phase([middle, backwards(middle)[..., ::-1], middle])

       def placed(fit):
           return np.stack([mrd.correct_lines([line], fit)[0][0] for line in train])

       images(
           [
               ("object", truth),
               ("flipped only", mrd.ifftc(placed(None))),
               ("phase corrected", mrd.ifftc(placed(phase))),
           ],
           title="an odd/even phase, and the ghost it leaves",
       )

    Examples
    --------
    >>> import numpy as np
    >>> import pulserver.mrd as mrd
    >>> line = np.ones((2, 32), dtype=np.complex64)
    >>> corrected = mrd.correct_lines([(line, False), (line, True)])
    >>> len(corrected), corrected[0].shape
    (2, (2, 32))
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
