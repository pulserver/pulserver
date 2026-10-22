"""Gradient nonlinearity correction of a reconstructed image."""

from __future__ import annotations

__all__ = ["gradient_unwarped", "states_gradient_coefficients"]

import logging
from typing import Any

import numpy as np

#: User parameters a gradient coil's spherical-harmonic table is stated in:
#: ``gradient_coefficients`` is the name a scanner-side interpreter writes; the
#: others are names bartorch also reads.
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
    """Correct an image for gradient nonlinearity.

    The coil's spherical-harmonic field is evaluated at the scanner position of
    every voxel of the reconstruction matrix, and the image is resampled at the
    displaced positions; a complex image has its real and imaginary parts
    resampled separately. Geometry comes from the header's reconstruction space
    and the unit's reference acquisition, so the correction is in the frame the
    series was acquired in, oblique prescriptions included. An encoding of one
    partition, a 2D slice, is corrected in its plane: the displacement across
    the slice moves the slice, which resampling cannot undo.

    Parameters
    ----------
    image : array_like
        ``(..., [partitions,] lines, columns)``, NumPy array or Torch tensor.
    context : ReconContext
        The plugin's :class:`ReconContext`, whose header states the table.
    data : ReconData
        The plugin's :class:`ReconData`, whose buffer states the geometry.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        The corrected image, of the type and on the device of ``image``.
        ``image`` itself where the header states no table, where the unit has
        no reference acquisition, or where the correction fails; the reason is
        logged and an uncorrected image is sent on.
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
        if correction.shape[0] == 1:
            correction = _in_plane(correction, header, reference)
        corrected = correction(torch.as_tensor(image))
    except Exception as error:
        _log.warning("the image was not corrected for gradient nonlinearity: %s", error)
        return image

    # NumPy arrays carry a ``device`` of their own under the array API, so the
    # type is asked for rather than duck-typed.
    return corrected if isinstance(image, torch.Tensor) else corrected.cpu().numpy()


def _in_plane(correction: Any, header: Any, reference: Any) -> Any:
    """Return ``correction``, of one partition, on the slice's ``(lines, columns)`` grid.

    Same table, matrix, field of view and centre, with ``position`` the centre
    of voxel ``n // 2`` as :meth:`bartorch.tools.Gradunwarp.from_mrd` reads it.
    """
    from bartorch.tools import Gradunwarp

    space = header.encoding[int(getattr(reference, "encoding_space_ref", 0) or 0)]
    matrix, extent = space.reconSpace.matrixSize, space.reconSpace.fieldOfView_mm
    shape = np.array([int(matrix.y), int(matrix.x)])
    fov_mm = np.array([float(extent.y), float(extent.x)])
    orientation = np.stack(
        [np.asarray(reference.phase_dir, float), np.asarray(reference.read_dir, float)],
        axis=1,
    )
    offset = (shape - 1) / 2 - shape // 2
    centre = np.asarray(reference.position, float) + orientation @ (
        offset * fov_mm / shape
    )
    return Gradunwarp(
        correction.coefficients,
        tuple(shape),
        tuple(fov_mm),
        orientation,
        tuple(centre),
        jacobian=correction.jacobian,
    )
