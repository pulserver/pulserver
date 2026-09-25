"""The prescriptions and the phantom the virtual scanner is tested with."""

import numpy as np
from scipy.spatial.transform import Rotation

from pulserver import virtual

#: A field-of-view offset along the logical axes, in metres.
OFFSET = np.array([0.02, -0.012, 0.0])
#: A prescription turned 20 degrees in plane about z, then tilted 30 degrees about x.
OBLIQUE = Rotation.from_euler("zx", [20.0, 30.0], degrees=True).as_matrix()
#: The oblique prescription with its phase-encoding axis reversed: a reflection.
REFLECTED = OBLIQUE @ np.diag([1.0, -1.0, 1.0])
#: The prescriptions a scan is repeated under, with their test ids.
ORIENTATIONS = {"axial": np.eye(3), "oblique": OBLIQUE, "reflected": REFLECTED}


def phantom(position=(0.0, 0.0, 0.0), rotation=None, coils=2):
    """An object without a mirror symmetry, with its origin at ``position`` and its axes turned by ``rotation``."""
    return virtual.Phantom(
        [
            virtual.Ellipse((0.0, 0.0, 0.0), (0.08, 0.06), 0.3),
            virtual.Ellipse((0.03, 0.02, 0.0), (0.02, 0.015), 0.0, 0.5),
        ],
        coils=coils,
        rotation=rotation,
        position=position,
    )


def posed(rotation, coils=2):
    """The phantom posed where a prescription of ``OFFSET`` and ``rotation`` places the field of view."""
    return phantom(rotation @ OFFSET, rotation, coils)
