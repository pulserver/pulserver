"""Shared normalization for Pulserver rotation events."""

from __future__ import annotations

import numpy as np

from ._make_rotation import make_rotation


def normalize_rotation(rotation):
    """Normalize a matrix or SciPy rotation to a Pulserver rotation event."""
    if rotation is None or getattr(rotation, "type", None) == "rot3D":
        return rotation

    if tuple(np.shape(rotation)) == (3, 3):
        from scipy.spatial.transform import Rotation

        rotation = Rotation.from_matrix(np.asarray(rotation, dtype=float))
    if not hasattr(rotation, "as_quat"):
        raise TypeError("rotation must be a 3x3 matrix, scipy Rotation, or pulserver rotation event")
    return make_rotation(rotation)
