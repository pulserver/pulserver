"""The explicit saturation bands a console prescribes, as the arguments of a sequence function that plays them."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any

from ..design import ConfigParam, FloatParam, IntParam
from ..design._entries import Entry
from ..protocol import ProtocolKey, UIParam

#: The normal each band starts with, along the physical axes, before the console turns it.
_NORMALS = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)) * 2

#: A band's starting position and thickness, in mm: outside a head's field of
#: view, so a band the operator has not yet placed saturates nothing imaged.
_POSITION_MM, _THICKNESS_MM = 150.0, 40.0


def band_entries(count: int) -> dict[ProtocolKey, Entry]:
    """Return the entries of ``count`` explicit saturation bands, 1 to 6, none of them on.

    ``exsat_mask`` has bit ``n - 1`` set for each band ``n`` the operator
    places on the console's graphic Rx, up to ``count``. Each band has its
    normal along the physical axes, its centre's distance from the isocentre
    along it and its thickness, in mm.
    """
    entries: dict[ProtocolKey, Entry] = {
        UIParam.ENABLE_SATURATION_UI: ConfigParam(1),
        UIParam.EXSAT_MASK: IntParam(
            "exsat_mask", range_min=0, range_max=2**count - 1, default=0
        ),
    }
    for n in range(1, count + 1):
        for key in UIParam.exsat_normal(n):
            entries[key] = FloatParam(
                key.value, range_min=-1.0, range_max=1.0, range_incr=1e-3
            )
        entries[UIParam.exsat_loc(n)] = FloatParam(
            f"exsat{n}_loc",
            unit="mm",
            scale=1e-3,
            range_min=-500.0,
            range_max=500.0,
        )
        entries[UIParam.exsat_thickness(n)] = FloatParam(
            f"exsat{n}_thickness",
            unit="mm",
            scale=1e-3,
            range_min=5.0,
            range_max=200.0,
        )
    return entries


def explicit_bands(app: Callable[..., Any], count: int) -> Callable[..., Any]:
    """Return ``app`` taking the console's ``count`` bands, ``exsat<n>_*`` and ``exsat_mask``, in place of its ``sat<n>_*``.

    A band is in the physical frame, as the console prescribes it and as
    ``app`` plays it: its normal along the physical axes, its position from
    the isocentre and its thickness, in metres. The bands ``exsat_mask``
    sets are played and the others are passed at zero thickness, which
    ``app`` leaves off. A mask setting a band past ``count`` raises
    ``ValueError``.
    """
    signature = inspect.signature(app)
    own = [p for p in signature.parameters.values() if not p.name.startswith("sat")]
    given = []
    for n in range(1, count + 1):
        given += [
            inspect.Parameter(
                f"exsat{n}_normal_{axis}",
                inspect.Parameter.KEYWORD_ONLY,
                default=component,
            )
            for axis, component in zip("xyz", _NORMALS[n - 1], strict=True)
        ]
        given += [
            inspect.Parameter(
                f"exsat{n}_loc",
                inspect.Parameter.KEYWORD_ONLY,
                default=1e-3 * _POSITION_MM,
            ),
            inspect.Parameter(
                f"exsat{n}_thickness",
                inspect.Parameter.KEYWORD_ONLY,
                default=1e-3 * _THICKNESS_MM,
            ),
        ]
    given.append(
        inspect.Parameter("exsat_mask", inspect.Parameter.KEYWORD_ONLY, default=0)
    )
    stated = signature.replace(parameters=[*own, *given])

    @functools.wraps(app)
    def banded(*args: Any, **kwargs: Any) -> Any:
        bound = stated.bind(*args, **kwargs)
        bound.apply_defaults()
        values = dict(bound.arguments)
        mask = int(values.pop("exsat_mask"))
        if mask >> count:
            raise ValueError(
                f"the console places explicit saturation bands past band {count}, "
                f"the last this sequence plays (mask {mask})"
            )
        for n in range(1, count + 1):
            on = bool(mask >> (n - 1) & 1)
            for axis in "xyz":
                values[f"sat{n}_normal_{axis}"] = values.pop(f"exsat{n}_normal_{axis}")
            values[f"sat{n}_position"] = values.pop(f"exsat{n}_loc")
            thickness = values.pop(f"exsat{n}_thickness")
            values[f"sat{n}_thickness"] = thickness if on else 0.0
        return app(**values)

    banded.__signature__ = stated  # type: ignore[attr-defined]
    return banded
