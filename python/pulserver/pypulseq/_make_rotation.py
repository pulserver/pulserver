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

    Parameters
    ----------
    rot_quaternion:
        Quaternion/rotation object exposing ``as_quat(canonical=True,
        scalar_first=True)`` (e.g. scipy Rotation).
    """
    event = SimpleNamespace()
    event.type = "rot3D"
    event.rot_quaternion = rot_quaternion
    return event
