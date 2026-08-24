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
    "image_result",
    "run_pyhysco",
]

from typing import Any

from ..mrd._images import as_numpy, center_crop
from ._distortion import run_pyhysco
from ._gradunwarp import (
    CoefficientAccessor,
    GradientCoefficients,
    Gradunwarp,
    ImageGeometry,
)


def image_result(
    image: Any,
    buffer: Any = None,
    *,
    series_index: int = 0,
    image_index: int | None = None,
    image_type: str = "magnitude",
    dicom: bool = True,
) -> Any:
    """Package one reconstructed image as the result a plugin returns.

    The last step of every reconstruction, and the same three things each time:
    take the magnitude, crop the encoded grid back to the matrix the header
    asked for, and put the two in-plane axes in the order MRD reads them --
    which is the transpose of the order the physics reconstructed in.

    Parameters
    ----------
    image
        The reconstructed image or volume, complex or real, NumPy or Torch.
    buffer
        The :class:`~pulserver.recon.ReconBuffer` it came out of, whose
        ``image_shape`` is the matrix to crop to. ``None`` crops nothing.
    series_index
        Output image-series index.
    image_index
        Explicit image index. ``None`` lets the runtime assign one.
    image_type
        ``"magnitude"``, which takes the modulus, or any other MRD image type,
        which passes the values through.
    dicom
        Convert to DICOM before sending.

    Returns
    -------
    ReconResult
        Ready for :meth:`~pulserver.recon.ReconPlugin.recon` to return.

    Examples
    --------
    >>> import numpy as np
    >>> import pulserver.recon as recon
    >>> result = recon.image_result(np.ones((4, 6), dtype=complex))
    >>> result.data.shape
    (6, 4)
    """
    import numpy as np

    from .plugin import ReconResult

    data = as_numpy(image)
    if image_type == "magnitude":
        data = np.abs(data)
    if buffer is not None:
        data = center_crop(data, buffer.image_shape)
    return ReconResult(
        np.swapaxes(data, -1, -2),
        reference=-1,
        series_index=series_index,
        image_index=image_index,
        image_type=image_type,
        dicom=dicom,
    )
