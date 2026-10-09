"""The controls the shipped sequences share that a console reaches through its user CVs, each at the same user CV in every plugin."""

from __future__ import annotations

import dataclasses
import functools
import inspect
from collections.abc import Callable, Mapping
from typing import Any

from ..design import Description, FloatParam, IntParam
from ..design._entries import Entry
from ..protocol import ProtocolKey, UIParam

#: Each shared control: the arguments it binds, whichever a sequence function
#: takes, its name and its entry for that argument, at the user CV of its place
#: here.
SHARED: tuple[tuple[tuple[str, ...], str, Callable[[str], Entry]], ...] = (
    (
        ("n_dummy",),
        "Dummy scans, -1 until steady state",
        lambda argument: IntParam(argument, range_min=-1, range_max=4096, automatic=-1),
    ),
    (
        ("partial_fourier_x",),
        "Partial echo",
        lambda argument: FloatParam(
            argument, range_min=0.75, range_max=1.0, range_incr=0.01
        ),
    ),
    (
        ("partial_fourier_y",),
        "Partial Fourier y",
        lambda argument: FloatParam(
            argument, range_min=0.75, range_max=1.0, range_incr=0.01
        ),
    ),
    (
        ("partial_fourier_z",),
        "Partial Fourier z",
        lambda argument: FloatParam(
            argument, range_min=0.75, range_max=1.0, range_incr=0.01
        ),
    ),
    (
        ("n_acs_y",),
        "ACS lines y",
        lambda argument: IntParam(argument, range_min=0, range_max=128),
    ),
    (
        ("n_acs_z",),
        "ACS lines z",
        lambda argument: IntParam(argument, range_min=0, range_max=128),
    ),
    (
        ("readout_oversampling",),
        "Readout oversampling",
        lambda argument: FloatParam(
            argument, range_min=1.0, range_max=4.0, range_incr=0.1
        ),
    ),
    (
        ("esp", "echo_spacing"),
        "Echo spacing, 0 the shortest",
        lambda argument: FloatParam(
            argument,
            unit="ms",
            scale=1e-3,
            range_min=0.0,
            range_max=100.0,
            range_incr=0.01,
        ),
    ),
    (
        ("refocusing_angle_deg",),
        "Refocusing flip",
        lambda argument: FloatParam(
            argument, unit="deg", range_min=60.0, range_max=180.0
        ),
    ),
    (
        ("caipi_shift",),
        "CAIPI shift",
        lambda argument: IntParam(argument, range_min=0, range_max=3),
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
    for index, (arguments, name, entry) in enumerate(SHARED):
        for argument in arguments:
            if argument in taken and argument not in bound:
                made = entry(argument)
                # A function that takes no None has no choice of its own.
                if getattr(made, "automatic", None) is not None and (
                    taken[argument].default is not None
                ):
                    made = dataclasses.replace(
                        made, automatic=None, range_min=made.automatic + 1
                    )
                    name = name.split(",")[0]
                entries[UIParam.user_name(index)] = Description(name)
                entries[UIParam.user_value(index)] = made
                break
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
