"""The prescription entries: where the interpreter asks for the field of view to be."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ._keys import FloatKey

#: The translation of the field-of-view centre along the logical readout,
#: phase and slice axes, in mm.
FOV_OFFSET = (
    str(FloatKey.FOV_OFFSET_X),
    str(FloatKey.FOV_OFFSET_Y),
    str(FloatKey.FOV_OFFSET_Z),
)

#: The rotation from the logical to the physical axes, row-major:
#: ``fov_rotation_ij`` is element ``(i, j)`` of ``R`` in physical = R logical.
FOV_ROTATION = tuple(
    str(FloatKey[f"FOV_ROTATION_{i}{j}"]) for i in (1, 2, 3) for j in (1, 2, 3)
)

#: The entries a protocol carries for the scanner's prescription. The
#: interpreter fills them from its prescription; no sequence argument receives
#: them. Pulserver applies the translation when it builds the IR, and checks
#: the gradients in the physical frame the rotation gives.
PRESCRIPTION = FOV_OFFSET + FOV_ROTATION

# The protocol carries floats to six significant digits.
_ORTHONORMAL = 1e-4


def prescribed_offset(values: Mapping[str, Any]) -> tuple[float, float, float]:
    """Return the field-of-view offset protocol values carry, in metres.

    Along the logical readout, phase and slice axes; zero along an axis whose
    entry is absent.
    """
    x, y, z = (float(values.get(name, 0.0)) * 1e-3 for name in FOV_OFFSET)
    return x, y, z


def prescribed_rotation(values: Mapping[str, Any]) -> np.ndarray:
    """Return the ``(3, 3)`` rotation from logical to physical axes protocol values carry.

    An absent entry is the identity's. The matrix is returned as the nearest
    orthonormal one, with its determinant's sign, so a reflection stays one.

    Raises
    ------
    ValueError
        If the entries are not orthonormal to the six significant digits the
        protocol carries.
    """
    identity = np.eye(3).ravel()
    matrix = np.array(
        [float(values.get(name, identity[k])) for k, name in enumerate(FOV_ROTATION)]
    ).reshape(3, 3)
    if not np.allclose(matrix @ matrix.T, np.eye(3), atol=_ORTHONORMAL):
        raise ValueError(
            f"the prescription rotation {matrix.tolist()} is not orthonormal"
        )
    left, _, right = np.linalg.svd(matrix)
    return left @ right
