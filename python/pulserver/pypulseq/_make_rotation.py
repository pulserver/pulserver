"""Rotation extension event constructor.

The event holds the caller's rotation object itself, so ``add_block`` keys its
quaternion cache on that object's identity and asks SciPy for the four numbers
once per orientation rather than once per block.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

__all__ = ["make_rotation"]


def make_rotation(rot_quaternion: Any) -> SimpleNamespace:
    """Create a rotation extension event.

    Attaching one of these to a block rotates every gradient in it, without
    redesigning any waveform. That is what makes the non-Cartesian readouts
    cheap: one base spoke or interleaf is designed, and each shot is the same
    waveform under a different rotation.

    Parameters
    ----------
    rot_quaternion : Any
        Rotation object exposing ``as_quat(canonical=True, scalar_first=True)``
        — e.g. :class:`scipy.spatial.transform.Rotation`.

    Returns
    -------
    types.SimpleNamespace
        Rotation extension event (``type == 'rot3D'``).

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.spatial.transform import Rotation
    >>> import pulserver.pypulseq as pp
    >>> event = pp.make_rotation(Rotation.from_euler("z", np.pi / 4))
    >>> event.type
    'rot3D'

    Rotate a radial base spoke over a golden-angle plan::

        for angle in pp.calc_golden_angles(377):
            readout(seq, rotation=Rotation.from_euler("z", angle))

    Reusing one rotation object across blocks costs one ``as_quat`` call for all
    of them; a fresh object per shot costs one each.

    See Also
    --------
    pulserver.ScanLoop.to_rotations : plan directions to rotation matrices.
    """
    event = SimpleNamespace()
    event.type = "rot3D"
    event.rot_quaternion = rot_quaternion
    return event
