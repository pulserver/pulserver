"""The explicit saturation bands a console prescribes, and a sequence function's arguments for those it plays."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ..design import FloatParam, IntParam
from ..design._entries import Entry
from ..protocol import ProtocolKey, UIParam, prescribed_offset, prescribed_rotation

#: The normal each band starts with, along the physical axes, before the console turns it.
_NORMALS = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)) * 2


def band_entries(count: int) -> dict[ProtocolKey, Entry]:
    """Return the entries of ``count`` explicit saturation bands, 1 to 6.

    ``exsat_mask`` turns band ``n`` on with its bit ``n - 1``; each band has its
    normal along the physical axes, its centre's distance from the isocentre
    along it and its thickness, in mm. They bind no argument of a sequence
    function: :func:`band_arguments` turns them into its own.
    """
    entries: dict[ProtocolKey, Entry] = {
        UIParam.EXSAT_MASK: IntParam(
            "exsat_mask", range_min=0, range_max=2**count - 1, default=0
        )
    }
    for n in range(1, count + 1):
        for key, component in zip(
            UIParam.exsat_normal(n), _NORMALS[n - 1], strict=True
        ):
            entries[key] = FloatParam(
                key.value,
                range_min=-1.0,
                range_max=1.0,
                range_incr=1e-3,
                default=component,
            )
        entries[UIParam.exsat_loc(n)] = FloatParam(
            f"exsat{n}_loc",
            unit="mm",
            scale=1e-3,
            range_min=-500.0,
            range_max=500.0,
            default=0.0,
        )
        entries[UIParam.exsat_thickness(n)] = FloatParam(
            f"exsat{n}_thickness",
            unit="mm",
            scale=1e-3,
            range_min=5.0,
            range_max=200.0,
            default=40.0,
        )
    return entries


def band_arguments(protocol: Mapping[ProtocolKey, Any], plays: int) -> dict[str, Any]:
    """Return a protocol's arguments with the bands it turns on as the sequence function's ``sat<k>_*``, in play order.

    A band's normal along the logical axes is ``R.T @ n``, ``R`` the
    prescription rotation and ``n`` its normal along the physical axes, and its
    position from the field-of-view centre, where the design places it before
    the offset moves the sequence, is its distance from the isocentre less the
    prescribed offset along that normal; positions and thicknesses in metres.

    Raises
    ------
    ValueError
        If more bands are turned on than the function plays, or a band's
        normal is zero.
    """
    values = protocol.arguments
    rotation = prescribed_rotation(protocol)
    offset = np.asarray(prescribed_offset(protocol))
    mask = int(values.get("exsat_mask", 0))
    arguments = {k: v for k, v in values.items() if not k.startswith("exsat")}
    played = [n for n in range(1, 7) if mask & (1 << (n - 1))]
    if len(played) > plays:
        raise ValueError(
            f"{len(played)} saturation bands are turned on; at most {plays} play"
        )
    for k, n in enumerate(played, 1):
        normal = np.array(
            [values[key.value] for key in UIParam.exsat_normal(n)], dtype=float
        )
        length = float(np.linalg.norm(normal))
        if length == 0.0:
            raise ValueError(f"saturation band {n} has no normal")
        logical = rotation.T @ (normal / length)
        arguments |= {
            f"sat{k}_normal_x": float(logical[0]),
            f"sat{k}_normal_y": float(logical[1]),
            f"sat{k}_normal_z": float(logical[2]),
            f"sat{k}_position": values[f"exsat{n}_loc"] - float(logical @ offset),
            f"sat{k}_thickness": values[f"exsat{n}_thickness"],
        }
    return arguments
