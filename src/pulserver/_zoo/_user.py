"""The controls the shipped sequences share that a console reaches through its user CVs, each at the same user CV in every plugin."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Mapping
from typing import Any

from ..design import Description, FloatParam, IntParam
from ..design._entries import Entry
from ..protocol import ProtocolKey, UIParam

#: Each shared control: the argument it binds, its name and its entry, at the
#: user CV of its place here.
SHARED: tuple[tuple[str, str, Callable[[], Entry]], ...] = (
    ("n_dummy", "Dummy scans", lambda: IntParam("n_dummy", range_min=0, range_max=256)),
    (
        "partial_fourier_x",
        "Partial echo",
        lambda: FloatParam(
            "partial_fourier_x", range_min=0.75, range_max=1.0, range_incr=0.01
        ),
    ),
    (
        "partial_fourier_y",
        "Partial Fourier y",
        lambda: FloatParam(
            "partial_fourier_y", range_min=0.75, range_max=1.0, range_incr=0.01
        ),
    ),
    (
        "partial_fourier_z",
        "Partial Fourier z",
        lambda: FloatParam(
            "partial_fourier_z", range_min=0.75, range_max=1.0, range_incr=0.01
        ),
    ),
    ("n_acs_y", "ACS lines y", lambda: IntParam("n_acs_y", range_min=0, range_max=128)),
    ("n_acs_z", "ACS lines z", lambda: IntParam("n_acs_z", range_min=0, range_max=128)),
    (
        "readout_oversampling",
        "Readout oversampling",
        lambda: FloatParam(
            "readout_oversampling", range_min=1.0, range_max=4.0, range_incr=0.1
        ),
    ),
    (
        "esp",
        "Echo spacing, 0 the shortest",
        lambda: FloatParam(
            "esp",
            unit="ms",
            scale=1e-3,
            range_min=0.0,
            range_max=100.0,
            range_incr=0.01,
        ),
    ),
    (
        "refocusing_angle_deg",
        "Refocusing flip",
        lambda: FloatParam(
            "refocusing_angle_deg", unit="deg", range_min=60.0, range_max=180.0
        ),
    ),
)

#: The first user CV a plugin's own controls take, after the shared ones.
OWN = 12


def user_entries(
    app: Callable[..., Any], protocol: Mapping[ProtocolKey, Entry]
) -> dict[ProtocolKey, Entry]:
    """Return the shared controls of the arguments ``app`` takes and ``protocol`` does not bind, each named, at its user CV."""
    taken = inspect.signature(app).parameters
    bound = {getattr(entry, "argument", None) for entry in protocol.values()}
    entries: dict[ProtocolKey, Entry] = {}
    for index, (argument, name, entry) in enumerate(SHARED):
        if argument in taken and argument not in bound:
            entries[UIParam.user_name(index)] = Description(name)
            entries[UIParam.user_value(index)] = entry()
    return entries


def shortest_at_zero(app: Callable[..., Any], argument: str) -> Callable[..., Any]:
    """Return ``app`` taking 0 for ``argument`` where it takes ``None``, its shortest choice, which a user CV cannot hold."""
    signature = inspect.signature(app)
    parameters = [
        p.replace(default=0.0) if p.name == argument and p.default is None else p
        for p in signature.parameters.values()
    ]
    stated = signature.replace(parameters=parameters)

    @functools.wraps(app)
    def shortest(*args: Any, **kwargs: Any) -> Any:
        bound = stated.bind(*args, **kwargs)
        bound.apply_defaults()
        values = dict(bound.arguments)
        if values.get(argument) == 0.0:
            values[argument] = None
        return app(**values)

    shortest.__signature__ = stated  # type: ignore[attr-defined]
    return shortest
