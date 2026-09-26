"""Scanner sequences: a pypulseqpp application bound to the scanner protocol."""

from __future__ import annotations

import importlib.util
import inspect
import math
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pypulseqpp as pp
from pypulseqpp import sequences

from ..protocol import (
    FOV_OFFSET,
    FOV_ROTATION,
    PRESCRIPTION,
    InputMode,
    Kind,
    Parameter,
    Validation,
    prescribed_rotation,
)
from ..protocol._keys import WIRE_NAMES

Preset = float | Callable[[pp.Opts], float] | None
#: Range of each prescription entry, in mm either side of the isocentre.
OFFSET_LIMIT_MM = 1000.0

# FLT_DIG: the significant decimal digits a float32 parameter holds through a round trip.
_SIGNIFICANT_DIGITS = 6
_MICROSECOND = Decimal("1e-6")


@dataclass(frozen=True)
class FloatParam:
    """A float UI entry bound to an ``init_sequence`` argument.

    The UI value is the argument divided by ``scale``, in ``unit``; a dropdown
    when it has options.
    """

    argument: str
    unit: str = ""
    scale: float = 1.0
    range_min: float = 0.0
    range_max: float = math.inf
    range_incr: float = 1.0
    options: tuple[float, ...] = ()


@dataclass(frozen=True)
class TimeParam:
    """A time UI entry bound to an ``init_sequence`` argument in seconds.

    Values, ranges and options are integer microseconds, the unit of the
    scanner's time parameters, so the value a parameter holds is the value the design
    reported. Each key of ``presets`` is a dropdown preset; its value is the
    time it requests: seconds, ``None`` for the application's own shortest
    choice, or a function of the scanner limits. The entry is a dropdown when
    it has options or presets.
    """

    argument: str
    range_min: int = 0
    range_max: int = 2**31 - 1
    range_incr: int = 1
    options: tuple[int, ...] = ()
    presets: Mapping[int, Preset] = field(default_factory=dict)


@dataclass(frozen=True)
class IntParam:
    """An integer UI entry bound to an ``init_sequence`` argument; a dropdown when it has options."""

    argument: str
    unit: str = ""
    range_min: int = 0
    range_max: int = 2**31 - 1
    range_incr: int = 1
    options: tuple[int, ...] = ()


@dataclass(frozen=True)
class BoolParam:
    """A checkbox UI entry bound to an ``init_sequence`` argument."""

    argument: str


@dataclass(frozen=True)
class StringListParam:
    """A choice among option strings, bound to an ``init_sequence`` argument.

    The argument receives the chosen string, not its index.
    """

    argument: str
    options: tuple[str, ...]


@dataclass(frozen=True)
class ConfigParam:
    """A value the sequence declares to the interpreter; never shown or edited."""

    value: int


@dataclass(frozen=True)
class Description:
    """A read-only text row in the UI."""

    text: str


Entry = (
    FloatParam
    | TimeParam
    | IntParam
    | BoolParam
    | StringListParam
    | ConfigParam
    | Description
)


def _ui_float(value: float) -> float:
    return float(f"{value:.{_SIGNIFICANT_DIGITS}g}")


def _to_ui(value: float, scale: float) -> float:
    return _ui_float(float(Decimal(repr(float(value))) / Decimal(repr(scale))))


def _to_si(value: float, scale: float) -> float:
    return float(Decimal(repr(_ui_float(value))) * Decimal(repr(scale)))


def _to_microseconds(seconds: float) -> int:
    """Round to the nearest microsecond, ties to even."""
    return int((Decimal(repr(float(seconds))) / _MICROSECOND).to_integral_value())


def _to_seconds(microseconds: int) -> float:
    return float(Decimal(int(microseconds)) * _MICROSECOND)


def _parameter(name: str, entry: Entry, defaults: Mapping[str, Any]) -> Parameter:
    if isinstance(entry, ConfigParam):
        return Parameter(Kind.CONFIG, entry.value, InputMode.OFF)
    if isinstance(entry, Description):
        return Parameter(Kind.DESCRIPTION, entry.text)
    if entry.argument not in defaults:
        raise ValueError(
            f"{name} binds {entry.argument!r}, which init_sequence does not take"
        )
    default = defaults[entry.argument]
    if isinstance(entry, TimeParam):
        options = (*entry.presets, *entry.options)
        if default is None:
            shortest = [key for key, preset in entry.presets.items() if preset is None]
            if not shortest:
                raise ValueError(
                    f"{name} defaults to None but offers no preset requesting it"
                )
            value = shortest[0]
        else:
            value = _to_microseconds(default)
        mode = InputMode.DROPDOWN if options else InputMode.TYPEIN
        return Parameter(
            Kind.INT,
            value,
            mode,
            entry.range_min,
            entry.range_max,
            entry.range_incr,
            "us",
            options,
        )
    if isinstance(entry, FloatParam):
        mode = InputMode.DROPDOWN if entry.options else InputMode.TYPEIN
        return Parameter(
            Kind.FLOAT,
            _to_ui(default, entry.scale),
            mode,
            entry.range_min,
            entry.range_max,
            entry.range_incr,
            entry.unit,
            entry.options,
        )
    if isinstance(entry, IntParam):
        mode = InputMode.DROPDOWN if entry.options else InputMode.TYPEIN
        return Parameter(
            Kind.INT,
            default,
            mode,
            entry.range_min,
            entry.range_max,
            entry.range_incr,
            entry.unit,
            entry.options,
        )
    if isinstance(entry, BoolParam):
        return Parameter(Kind.BOOL, default)
    return Parameter(
        Kind.STRINGLIST, default, InputMode.DROPDOWN, options=entry.options
    )


class ScannerSequence:
    """A pypulseqpp application exposed to the scanner UI.

    A subclass sets :attr:`app` and :attr:`ui`, whose keys are the
    interpreter's parameter names: members of
    :class:`~pulserver.protocol.UIParam` or :class:`~pulserver.protocol.ConfigKey`,
    or user-entry keys. They are stored as plain strings. Entries a request
    omits keep the application's defaults. An entry resolves to the value its
    argument took in the design, as the application records it with
    ``SequenceApp.resolve``, and otherwise keeps the requested value. Times
    travel as integer microseconds; other float values are read and reported
    to six significant digits, the precision of a float32 parameter. Either way
    a reply stored in scanner parameters and sent back resolves to itself.

    Attributes
    ----------
    recon : str
        Reconstruction plugin the data of this sequence is reconstructed with,
        recorded in every design generated from it. Empty leaves the choice
        to the reconstruction client.

    Raises
    ------
    ValueError
        When a subclass is defined with a ``ui`` key the interpreter does not
        know, which its parser would drop, or with one of the prescription
        entries, which pulserver applies itself.
    """

    app: ClassVar[type[sequences.SequenceApp]]
    ui: ClassVar[Mapping[str, Entry]]
    recon: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "ui" not in cls.__dict__:
            return
        unknown = sorted(str(name) for name in cls.ui if name not in WIRE_NAMES)
        if unknown:
            raise ValueError(
                f"{cls.__name__} declares entries the interpreter does not know: "
                f"{unknown}"
            )
        reserved = sorted(str(name) for name in cls.ui if str(name) in PRESCRIPTION)
        if reserved:
            raise ValueError(
                f"{cls.__name__} binds the prescription entries {reserved}, which "
                "pulserver applies when it builds the IR"
            )
        cls.ui = {str(name): entry for name, entry in cls.ui.items()}

    def listing(self) -> dict[str, Parameter]:
        """Return the protocol with its schema, valued at the application's defaults.

        An argument defaulting to ``None`` shows the preset that requests ``None``.
        The ``PRESCRIPTION`` entries of :mod:`pulserver.protocol` follow the
        declared ones, not editable in the UI: the offset at zero, the rotation
        at the identity.
        """
        defaults = self.app.protocol()
        listing = {
            name: _parameter(name, entry, defaults) for name, entry in self.ui.items()
        }
        for name in FOV_OFFSET:
            listing[name] = Parameter(
                Kind.FLOAT,
                0.0,
                InputMode.OFF,
                -OFFSET_LIMIT_MM,
                OFFSET_LIMIT_MM,
                0.1,
                "mm",
            )
        for name, value in zip(FOV_ROTATION, np.eye(3).ravel(), strict=True):
            listing[name] = Parameter(
                Kind.FLOAT, float(value), InputMode.OFF, -1.0, 1.0, 1e-6, ""
            )
        return listing

    def validate(self, system: pp.Opts, request: Mapping[str, Any]) -> Validation:
        """Resolve a request into the protocol the application will play.

        A valid reply carries the resolved values, as the application's
        ``resolved`` reports them, and its ``scan_time()``, which plays the
        chain only when the application states no ``duration``. An invalid
        one carries the request and the error the design raised.

        Raises
        ------
        ValueError
            If the request names an entry the UI does not declare.
        """
        return self.resolve(system, request)[1]

    def generate(
        self, system: pp.Opts, request: Mapping[str, Any], directory: Path
    ) -> tuple[Validation, list[str]]:
        """Write the resolved design into ``directory``, as :meth:`write` does.

        Returns
        -------
        Validation
            As :meth:`validate` returns it.
        list of str
            Written paths in play order, prescans first; empty for an invalid
            request, for which nothing is written.
        """
        app, validation = self.resolve(system, request)
        if app is None:
            return validation, []
        return validation, self.write(app, directory)

    @staticmethod
    def write(app: sequences.SequenceApp, directory: Path | str) -> list[str]:
        """Write a resolved application into ``directory`` as signed binary Pulseq.

        The first file is ``sequence.seq``. The files keep the ``.seq`` name a
        chain names them by; what is in them is the binary form, which the
        scanner IR conversion reads. Returns the written paths in play order,
        prescans first.
        """
        return app.write(Path(directory) / "sequence.seq", offline=False)

    def resolve(
        self, system: pp.Opts, request: Mapping[str, Any]
    ) -> tuple[sequences.SequenceApp | None, Validation]:
        """Return the application a request constructs, and the validation of :meth:`validate`.

        The application is ``None`` for an invalid request. It is constructed
        once and not designed, so :meth:`write` designs it.

        Raises
        ------
        ValueError
            If the request names an entry the UI does not declare.
        """
        listing = self.listing()
        unknown = set(request) - set(listing)
        if unknown:
            raise ValueError(f"not entries of this protocol: {sorted(unknown)}")
        values = {name: p.value for name, p in listing.items() if p.editable}
        values.update(request)
        try:
            prescribed_rotation(values)
            app = self.app(system, **self._arguments(system, values))
            scan_time = app.scan_time()
        except Exception as error:  # a design refuses a protocol by raising
            return None, Validation(
                False, None, str(error) or type(error).__name__, values
            )

        resolved = dict(values)
        readback = app.resolved
        for name, entry in self.ui.items():
            value = readback.get(getattr(entry, "argument", None))
            if value is None:
                continue
            if isinstance(entry, TimeParam):
                resolved[name] = _to_microseconds(value)
            elif isinstance(entry, FloatParam):
                resolved[name] = _to_ui(value, entry.scale)
            elif isinstance(entry, IntParam):
                resolved[name] = int(value)
            else:
                resolved[name] = value
        return app, Validation(True, scan_time, "", resolved)

    def _arguments(self, system: pp.Opts, values: Mapping[str, Any]) -> dict[str, Any]:
        arguments = {}
        for name, entry in self.ui.items():
            if isinstance(entry, ConfigParam | Description):
                continue
            value = values[name]
            if isinstance(entry, TimeParam):
                if entry.presets and value < 0:
                    if value not in entry.presets:
                        raise ValueError(f"{name} does not offer preset {value}")
                    preset = entry.presets[value]
                    value = preset(system) if callable(preset) else preset
                else:
                    value = _to_seconds(value)
            elif isinstance(entry, FloatParam):
                value = _to_si(value, entry.scale)
            arguments[entry.argument] = value
        return arguments


def load_plugin(path: Path) -> ScannerSequence:
    """Import a plugin file and instantiate the one ScannerSequence it defines.

    The module is registered as ``pulserver_plugin_<file stem>``; loading
    another file with the same stem replaces it.

    Raises
    ------
    ValueError
        If the file defines no ScannerSequence subclass, or more than one.
    """
    path = Path(path)
    name = f"pulserver_plugin_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    found = [
        obj
        for obj in vars(module).values()
        if inspect.isclass(obj)
        and issubclass(obj, ScannerSequence)
        and obj.__module__ == name
    ]
    if len(found) != 1:
        raise ValueError(
            f"{path} defines {len(found)} ScannerSequence subclasses; a plugin defines one"
        )
    return found[0]()
