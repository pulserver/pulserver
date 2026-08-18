"""Rotating a block's gradients by a general 3x3 matrix.

:func:`pypulseq.rotate.rotate` turns them about a single axis; a ``ROTATIONS``
block extension carries a full quaternion, which needs this.

The body works in :class:`~types.SimpleNamespace` events, as
:func:`pypulseq.scale_grad` and :func:`pypulseq.add_gradients` do.
:mod:`pulserver.pypulseq` exports it through the interoperation decorator, so a
caller passes and receives slotted events.
"""

from __future__ import annotations

__all__ = ["rotate3D"]

from types import SimpleNamespace

import numpy as np
import pypulseq as pp
from pypulseq.add_gradients import add_gradients
from pypulseq.scale_grad import scale_grad

from ._opts import default_system

_AXES = ("x", "y", "z")


def _grad_abs_mag(grad: SimpleNamespace) -> float:
    """Peak absolute amplitude of a trapezoid or arbitrary gradient."""
    if grad.type == "trap":
        return abs(grad.amplitude)
    return float(np.max(np.abs(grad.waveform)))


def rotate3D(
    *args: SimpleNamespace,
    rotation_matrix: np.ndarray,
    system: pp.Opts | None = None,
) -> list[SimpleNamespace]:
    """Rotate the gradients in ``args`` by a 3x3 rotation matrix.

    Non-gradient events are passed through untouched, mirroring
    :func:`pypulseq.rotate.rotate`.

    Parameters
    ----------
    args : SimpleNamespace
        Block events. At most one gradient per axis.
    rotation_matrix : numpy.ndarray
        3x3 rotation matrix.
    system : pypulseq.Opts, optional
        System limits used when summing projections onto a common axis.

    Returns
    -------
    list of SimpleNamespace
        Bypassed events first, then the rotated gradients.
    """
    system = default_system(system)

    rotation_matrix = np.asarray(rotation_matrix, dtype=float)
    if rotation_matrix.shape != (3, 3):
        raise ValueError("The rotation matrix must have shape (3, 3).")

    to_rotate: dict[str, SimpleNamespace] = {}
    bypass: list[SimpleNamespace] = []

    for event in args:
        if event.type not in ("grad", "trap"):
            bypass.append(event)
            continue
        if event.channel not in _AXES:
            raise ValueError(f"Invalid event channel. Expected one of {list(_AXES)}")
        if event.channel in to_rotate:
            raise ValueError(
                f"More than one gradient provided for channel {event.channel}"
            )
        to_rotate[event.channel] = event

    max_mag = max((_grad_abs_mag(g) for g in to_rotate.values()), default=0.0)
    fthresh = 1e-6
    thresh = fthresh * max_mag

    rotated: list[SimpleNamespace] = []
    for j in range(3):
        out = None
        for i in range(3):
            if _AXES[i] not in to_rotate or abs(rotation_matrix[j, i]) < fthresh:
                continue
            scaled = scale_grad(grad=to_rotate[_AXES[i]], scale=rotation_matrix[j, i])
            scaled.channel = _AXES[j]
            out = scaled if out is None else add_gradients((out, scaled), system=system)
        if out is not None and _grad_abs_mag(out) >= thresh:
            rotated.append(out)

    return [*bypass, *rotated]
