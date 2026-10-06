"""Gradient nonlinearity correction of a reconstructed image."""

from __future__ import annotations

__all__ = ["gradient_unwarped", "states_gradient_coefficients"]

import logging
from typing import Any

#: User parameters a gradient coil's spherical-harmonic table is stated in. A
#: GE interpreter writes ``gradient_coefficients``; the others are the names
#: bartorch also reads, so a stream from another interpreter is corrected too.
COEFFICIENT_PARAMETERS = (
    "gradient_coefficients",
    "gradientcoefficients",
    "gradunwarp_coefficients",
    "coeff_dat",
)

_log = logging.getLogger("pulserver.recon")


def states_gradient_coefficients(header: Any) -> bool:
    """Whether an MRD header carries a gradient coil's coefficient table."""
    parameters = getattr(
        getattr(header, "userParameters", None), "userParameterString", None
    )
    return any(
        str(getattr(parameter, "name", "")).strip().lower() in COEFFICIENT_PARAMETERS
        and getattr(parameter, "value", None)
        for parameter in parameters or ()
    )


def gradient_unwarped(image: Any, context: Any, data: Any) -> Any:
    """Return an image resampled from where its voxels were acquired to where they belong.

    The coil's spherical-harmonic field is evaluated at the scanner position of
    every voxel of the reconstruction matrix, and the image is resampled there.
    A complex image has its real and imaginary parts resampled separately.

    Geometry comes from the header's reconstruction space and from the unit's
    reference acquisition, so the correction is in the frame the series was
    acquired in, oblique prescriptions included.

    Parameters
    ----------
    image
        ``(..., [partitions,] lines, columns)``, NumPy array or Torch tensor.
    context
        The plugin's :class:`ReconContext`, whose header states the table.
    data
        The plugin's :class:`ReconData`, whose buffer states the geometry.

    Returns
    -------
    The corrected image, of the type and on the device it was given in.
    ``image`` itself where the header states no table, where the unit has no
    reference acquisition, or where the correction fails: an uncorrected image
    reaches the scanner, and the reason is logged.
    """
    header = getattr(context, "header", None)
    if header is None or not states_gradient_coefficients(header):
        return image

    buffer = getattr(data, "data", None)
    reference = getattr(buffer, "reference", None)
    if reference is None:
        _log.warning(
            "the header states a gradient coil table and the unit has no "
            "reference acquisition; the image is not corrected"
        )
        return image

    try:
        import torch
        from bartorch.tools import Gradunwarp

        correction = Gradunwarp.from_mrd(header, reference)
        tensor = torch.as_tensor(image)
        # The correction works on the encoding's own matrix, (partitions,
        # lines, columns). A plugin that reconstructed one partition drops that
        # axis, so it is restored for the resampling and dropped again after.
        flattened = (
            tensor.ndim == len(correction.shape) - 1 and correction.shape[0] == 1
        )
        if flattened:
            tensor = tensor[None]
        corrected = correction(tensor)
        if flattened:
            corrected = corrected[0]
    except Exception as error:
        _log.warning("the image was not corrected for gradient nonlinearity: %s", error)
        return image

    # NumPy arrays carry a ``device`` of their own under the array API, so the
    # type is asked for rather than duck-typed.
    return corrected if isinstance(image, torch.Tensor) else corrected.cpu().numpy()
