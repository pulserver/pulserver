"""A 3D sequence function prescribed as a console prescribes a slab: by its locations and the thickness of each."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any


def slab(app: Callable[..., Any]) -> Callable[..., Any]:
    """Return ``app`` taking the thickness of one location, ``slice_thickness``, in place of the slab's, ``fov_z``.

    The slab is ``n_z * slice_thickness``, in metres, so that a slab keeps its
    locations' thickness as their number changes. The default thickness is
    the app's default slab over its default locations.
    """
    signature = inspect.signature(app)
    fov_z, n_z = signature.parameters["fov_z"], signature.parameters["n_z"]
    parameters = [
        p.replace(name="slice_thickness", default=fov_z.default / n_z.default)
        if p.name == "fov_z"
        else p
        for p in signature.parameters.values()
    ]
    stated = signature.replace(parameters=parameters)

    @functools.wraps(app)
    def by_locations(*args: Any, **kwargs: Any) -> Any:
        bound = stated.bind(*args, **kwargs)
        bound.apply_defaults()
        values = dict(bound.arguments)
        values["fov_z"] = values["n_z"] * values.pop("slice_thickness")
        return app(**values)

    by_locations.__signature__ = stated  # type: ignore[attr-defined]
    return by_locations
