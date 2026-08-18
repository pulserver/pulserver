"""Image-space operations that follow a reconstruction.

Everything here takes reconstructed images or coil images and returns images:
combining the coil dimension away, unwarping gradient nonlinearity, and
correcting susceptibility distortion.
"""

from __future__ import annotations

__all__ = [
    "CoefficientAccessor",
    "GradientCoefficients",
    "Gradunwarp",
    "ImageGeometry",
    "as_numpy",
    "center_crop",
    "coil_combine",
    "run_pyhysco",
]

from typing import Any

from ._distortion import run_pyhysco
from ._gradunwarp import (
    CoefficientAccessor,
    GradientCoefficients,
    Gradunwarp,
    ImageGeometry,
)


def center_crop(image: Any, shape: Any) -> Any:
    """Crop an image's trailing axes to a centred window.

    The last step of a Cartesian reconstruction: the image comes off the
    encoded grid, which carries readout oversampling and whatever field-of-view
    oversampling the phase-encode direction was prescribed with, and the
    prescribed matrix is the centred part of it.

    Parameters
    ----------
    image
        Image, NumPy or Torch, with any number of leading axes.
    shape
        Extent of the trailing axes to keep. Shorter than ``image.ndim``, so
        coils, slices and contrasts pass through untouched.

    Returns
    -------
    array
        A view of ``image`` cropped to ``shape``.

    Raises
    ------
    ValueError
        If ``shape`` has more axes than the image, or asks for more samples
        than an axis holds.

    Examples
    --------
    >>> import numpy as np
    >>> from pulserver.recon.postprocessing import center_crop
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
        start = (current - size) // 2
        selection[axis] = slice(start, start + size)
    return image[tuple(selection)]


def coil_combine(
    coil_images: Any,
    coil_maps: Any | None = None,
    *,
    coil_axis: int = -3,
) -> Any:
    """Collapse the coil dimension of a stack of coil images.

    With sensitivity maps this is the matched filter, ``sum(s* x)``, which
    keeps the image complex and is what a SENSE-style reconstruction produces.
    Without them it is the root sum of squares, which needs no calibration and
    returns a real magnitude — the right answer whenever no maps are available
    and the only one available inline before calibration has run.

    Parameters
    ----------
    coil_images
        Coil images, NumPy or Torch, with the coil dimension at ``coil_axis``.
    coil_maps
        Sensitivity maps broadcastable against ``coil_images``. ``None``
        selects the root sum of squares.
    coil_axis
        Dimension holding the coils. The default is the third from last, which
        is where a ``(..., coil, y, x)`` stack carries it.

    Returns
    -------
    array
        The combined image, with ``coil_axis`` removed. Complex when maps were
        supplied, real magnitude otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> from pulserver.recon.postprocessing import coil_combine
    >>> coil_images = np.ones((4, 8, 8), dtype=complex)
    >>> float(coil_combine(coil_images)[0, 0])
    2.0

    Matched-filter combination keeps the phase:

    >>> maps = np.full((4, 8, 8), 0.5, dtype=complex)
    >>> complex(coil_combine(coil_images, maps)[0, 0])
    (2+0j)
    """
    if coil_maps is None:
        squared = (coil_images * coil_images.conj()).real
        return squared.sum(coil_axis) ** 0.5
    return (coil_maps.conj() * coil_images).sum(coil_axis)


def as_numpy(array: Any) -> Any:
    """Bring a reconstruction back to NumPy, whether it is Torch or already so.

    A reconstruction that ran on a GPU answers in the namespace it ran in, and
    an image on its way out to DICOM has to be NumPy.

    Parameters
    ----------
    array
        A Torch tensor, on any device, or anything NumPy accepts.

    Returns
    -------
    numpy.ndarray
        The same values, on the host.
    """
    import numpy as np

    return array.cpu().numpy() if hasattr(array, "cpu") else np.asarray(array)
