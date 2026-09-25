"""Image-space operations on reconstruction outputs, for NumPy and Torch arrays."""

from __future__ import annotations

__all__ = ["as_numpy", "center_crop", "coil_combine"]

from typing import Any


def center_crop(image: Any, shape: Any) -> Any:
    """Crop the trailing axes of an image to a centred window.

    Parameters
    ----------
    image
        NumPy or Torch array.
    shape
        Extent of each trailing axis to keep; leading axes pass through.

    Returns
    -------
    array
        A view of ``image``. Along each axis the sample at index ``n // 2``,
        the centre of a centred FFT of ``n`` samples, is at index
        ``size // 2`` of the window.

    Raises
    ------
    ValueError
        If ``shape`` has more axes than ``image``, or an extent is not in
        ``1..image.shape[axis]``.

    Examples
    --------
    >>> import numpy as np
    >>> from pulserver.mrd import center_crop
    >>> image = np.arange(2 * 8).reshape(2, 8)
    >>> center_crop(image, (4,)).shape
    (2, 4)
    >>> center_crop(np.arange(8), (4,))
    array([2, 3, 4, 5])
    """
    extents = tuple(int(size) for size in shape)
    if len(extents) > image.ndim:
        raise ValueError(f"shape has {len(extents)} axes, image has {image.ndim}")
    selection = [slice(None)] * image.ndim
    for axis, size in enumerate(extents, start=image.ndim - len(extents)):
        current = image.shape[axis]
        if not 0 < size <= current:
            raise ValueError(f"cannot crop axis {axis} of {current} samples to {size}")
        start = current // 2 - size // 2
        selection[axis] = slice(start, start + size)
    return image[tuple(selection)]


def coil_combine(
    coil_images: Any,
    coil_maps: Any | None = None,
    *,
    coil_axis: int = -3,
) -> Any:
    """Combine the coil axis of a stack of coil images.

    With sensitivity maps, returns the matched filter ``sum(conj(s) * x)``,
    complex and not normalised by the map energy. Without maps, returns the
    root sum of squares, real.

    Parameters
    ----------
    coil_images
        NumPy or Torch array with coils along ``coil_axis``.
    coil_maps
        Sensitivities broadcastable against ``coil_images``.
    coil_axis
        Coil axis; the default fits ``(..., coil, y, x)``.

    Returns
    -------
    array
        ``coil_images`` without ``coil_axis``.

    Examples
    --------
    >>> import numpy as np
    >>> from pulserver.mrd import coil_combine
    >>> coil_images = np.ones((4, 8, 8), dtype=complex)
    >>> float(coil_combine(coil_images)[0, 0])
    2.0
    >>> maps = np.full((4, 8, 8), 0.5, dtype=complex)
    >>> complex(coil_combine(coil_images, maps)[0, 0])
    (2+0j)
    """
    if coil_maps is None:
        squared = (coil_images * coil_images.conj()).real
        return squared.sum(coil_axis) ** 0.5
    return (coil_maps.conj() * coil_images).sum(coil_axis)


def as_numpy(array: Any) -> Any:
    """Return an array as NumPy on the host; Torch tensors are detached first.

    Examples
    --------
    >>> import pulserver.mrd as mrd
    >>> mrd.as_numpy([1.0, 2.0])
    array([1., 2.])
    """
    import numpy as np

    detach = getattr(array, "detach", None)
    if callable(detach):
        array = detach()
    return array.cpu().numpy() if hasattr(array, "cpu") else np.asarray(array)
