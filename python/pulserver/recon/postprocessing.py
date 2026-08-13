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
