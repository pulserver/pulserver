"""Centered orthonormal Fourier transforms shared across the reconstruction.

The one convention an MRI reconstruction means by "the Fourier transform":
``ifftshift -> fft(norm="ortho") -> fftshift``, over any set of axes, for
Torch tensors (device preserved) and NumPy arrays alike.
"""

from __future__ import annotations

__all__ = [
    "centered_fftn",
    "fftc",
    "ifftc",
    "resize_centered",
    "resize_centered_axis",
    "torch_or_numpy",
]

from importlib import import_module
from typing import Any


def torch_or_numpy(array: Any) -> tuple[Any, bool]:
    """Return the array's namespace module and whether it is Torch."""
    try:
        torch = import_module("torch")
    except ImportError:
        torch = None
    if torch is not None and isinstance(array, torch.Tensor):
        return torch, True
    return import_module("numpy"), False


def centered_fftn(data: Any, *, axes: tuple[int, ...], inverse: bool) -> Any:
    """Centered orthonormal FFT or inverse FFT over the given axes."""
    xp, is_torch = torch_or_numpy(data)
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


def fftc(value: Any, axes: int | tuple[int, ...]) -> Any:
    """Centered orthonormal FFT over one axis or several."""
    axes = (axes,) if isinstance(axes, int) else tuple(axes)
    return centered_fftn(value, axes=axes, inverse=False)


def ifftc(value: Any, axes: int | tuple[int, ...]) -> Any:
    """Centered orthonormal inverse FFT over one axis or several."""
    axes = (axes,) if isinstance(axes, int) else tuple(axes)
    return centered_fftn(value, axes=axes, inverse=True)


def resize_centered(value: Any, shape: tuple[int, ...]) -> Any:
    """Zero-pad or crop the trailing axes symmetrically about their centres."""
    xp, is_torch = torch_or_numpy(value)
    spatial_ndim = len(shape)
    result_shape = (*value.shape[:-spatial_ndim], *shape)
    result = (
        xp.zeros(result_shape, dtype=value.dtype, device=value.device)
        if is_torch
        else xp.zeros(result_shape, dtype=value.dtype)
    )
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


def resize_centered_axis(value: Any, size: int, *, axis: int) -> Any:
    """Zero-pad or crop one axis symmetrically about its centre."""
    axis %= value.ndim
    if value.shape[axis] == size:
        return value
    return resize_centered(value, (size, *value.shape[axis + 1 :]))
