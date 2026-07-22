"""Rotation extension event constructor.

This helper mirrors the pypulseq-style event factory pattern and returns a
SimpleNamespace consumed by pulserver's fast Sequence path.
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

        for angle in pp.make_radial_tilt(377, scheme="golden").support[:, 0]:
            readout(seq, rotation=Rotation.from_euler("z", angle))

    See Also
    --------
    pulserver.SamplingPattern.to_rotations : plan directions to rotation matrices.
    """
    event = SimpleNamespace()
    event.type = "rot3D"
    event.rot_quaternion = rot_quaternion
    return event
