"""Scanner sequences: a pypulseqpp application bound to the scanner protocol."""

from __future__ import annotations

import difflib
import importlib.util
import inspect
import math
import sys
import warnings
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pypulseqpp as pp
from pypulseqpp import sequences

from .. import ir
from ..protocol import (
    FOV_OFFSET,
    FOV_ROTATION,
    PRESCRIPTION,
    InputMode,
    Kind,
    Parameter,
    ProtocolKey,
    RfPulse,
    TEPreset,
    TRPreset,
    Validation,
    prescribed_rotation,
)
from ..protocol._keys import WIRE_NAMES, StrEnum

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
    when it has options. ``default`` is the UI value the protocol starts at;
    ``None`` starts it at the application's default.
    """

    argument: str
    unit: str = ""
    scale: float = 1.0
    range_min: float = 0.0
    range_max: float = math.inf
    range_incr: float = 1.0
    options: tuple[float, ...] = ()
    default: float | None = None


@dataclass(frozen=True)
class TimeParam:
    """A time UI entry bound to an ``init_sequence`` argument in seconds.

    Values, ranges and options are integer microseconds, the unit of the
    scanner's time parameters, so the value a parameter holds is the value the design
    reported. Each key of ``presets`` is a dropdown preset; its value is the
    time it requests: seconds, ``None`` for the application's own shortest
    choice, or a function of the scanner limits. The entry is a dropdown when
    it has options or presets. ``default``, microseconds or a key of
    ``presets``, is the value the protocol starts at; ``None`` starts it at the
    application's default.
    """

    argument: str
    range_min: int = 0
    range_max: int = 2**31 - 1
    range_incr: int = 1
    options: tuple[int, ...] = ()
    presets: Mapping[int, Preset] = field(default_factory=dict)
    default: int | None = None


@dataclass(frozen=True)
class IntParam:
    """An integer UI entry bound to an ``init_sequence`` argument; a dropdown when it has options.

    ``default`` is the value the protocol starts at; ``None`` starts it at the
    application's default.
    """

    argument: str
    unit: str = ""
    range_min: int = 0
    range_max: int = 2**31 - 1
    range_incr: int = 1
    options: tuple[int, ...] = ()
    default: int | None = None


@dataclass(frozen=True)
class BoolParam:
    """A checkbox UI entry bound to an ``init_sequence`` argument.

    ``default`` is the value the protocol starts at; ``None`` starts it at the
    application's default.
    """

    argument: str
    default: bool | None = None


@dataclass(frozen=True)
class ChoiceParam:
    """A choice among the members of a ``StrEnum``, bound to an ``init_sequence`` argument.

    The argument receives the chosen member, a ``str`` equal to its option. The
    options are the members in definition order, and the wire carries the index
    of the chosen one. ``default`` is the member the protocol starts at;
    ``None`` starts it at the application's default, which has to be a member
    or the value of one.
    """

    argument: str
    choices: type[StrEnum]
    default: StrEnum | None = None


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
    | ChoiceParam
    | ConfigParam
    | Description
)


def StringListParam(
    argument: str, options: tuple[str, ...], default: str | None = None
) -> ChoiceParam:
    """Return a :class:`ChoiceParam` over a ``StrEnum`` built from option strings.

    Deprecated: declare the enum and use :class:`ChoiceParam`. The argument
    receives the member of the chosen option, a ``str`` equal to it.
    ``default`` is the option the protocol starts at; ``None`` starts it at the
    application's default.

    Warns
    -----
    DeprecationWarning
        On every call.
    """
    warnings.warn(
        "StringListParam is deprecated; use ChoiceParam with a StrEnum",
        DeprecationWarning,
        stacklevel=2,
    )
    choices = StrEnum(
        "Options", {f"OPTION_{n}": option for n, option in enumerate(options)}
    )
    return ChoiceParam(argument, choices, None if default is None else choices(default))


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


def _member(key: ProtocolKey, entry: ChoiceParam, value: Any) -> StrEnum:
    """Return the member of the entry's choices that ``value`` is or names."""
    try:
        return entry.choices(value)
    except ValueError:
        options = ", ".join(entry.choices)
        raise ValueError(f"{key}: {value!r} is not one of {options}") from None


def _parameter(
    name: ProtocolKey, entry: Entry, defaults: Mapping[str, Any]
) -> Parameter:
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
        if entry.default is not None:
            value = int(entry.default)
            if value < 0 and value not in entry.presets:
                raise ValueError(
                    f"{name} defaults to preset {value}, which it does not offer"
                )
        elif default is None:
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
            _to_ui(default, entry.scale)
            if entry.default is None
            else _ui_float(entry.default),
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
            default if entry.default is None else int(entry.default),
            mode,
            entry.range_min,
            entry.range_max,
            entry.range_incr,
            entry.unit,
            entry.options,
        )
    if isinstance(entry, BoolParam):
        return Parameter(Kind.BOOL, default if entry.default is None else entry.default)
    default = default if entry.default is None else entry.default
    return Parameter(
        Kind.STRINGLIST,
        _member(name, entry, default),
        InputMode.DROPDOWN,
        options=tuple(entry.choices),
    )


class Protocol(Mapping[ProtocolKey, Any]):
    """The values of a scanner sequence's protocol, in the units of its application's arguments.

    An immutable mapping from :data:`~pulserver.protocol.ProtocolKey` to value.
    A time is in seconds and a float in the unit of the argument it binds,
    metres for a length; an integer is an ``int``, a checkbox a ``bool`` and a
    choice a member of its enum. A time entry showing a preset holds what the
    preset requests, a number of seconds or ``None`` for the application's own
    shortest choice, and :meth:`preset` names the preset. The prescription
    entries bind no argument and keep the units of the wire: mm for the
    offset, unitless for the rotation.

    The wire carries integer microseconds for a time and the entry's own unit
    for a float, to six significant digits. :meth:`from_wire` and
    :meth:`to_wire` are the only conversions between the two.

    Parameters
    ----------
    entries
        The declared entries the values belong to, as
        :attr:`ScannerSequence.ui` holds them.
    values
        The value of each key, in argument units.
    presets
        The preset each time key shows, by the key its entry declares it with.
    """

    def __init__(
        self,
        entries: Mapping[ProtocolKey, Entry],
        values: Mapping[ProtocolKey, Any],
        presets: Mapping[ProtocolKey, TEPreset | TRPreset] | None = None,
    ) -> None:
        self._entries = entries
        self._values = dict(values)
        self._presets = dict(presets or {})

    def __getitem__(self, key: ProtocolKey) -> Any:
        return self._values[key]

    def __iter__(self) -> Iterator[ProtocolKey]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        items = ", ".join(f"{key}: {value!r}" for key, value in self._values.items())
        return f"Protocol({{{items}}})"

    @classmethod
    def from_wire(
        cls,
        entries: Mapping[ProtocolKey, Entry],
        values: Mapping[ProtocolKey, Any],
        system: pp.Opts,
    ) -> Protocol:
        """Return the protocol that wire values stand for.

        Parameters
        ----------
        entries
            The declared entries: :attr:`ScannerSequence.ui`.
        values
            Wire values by key: integer microseconds or a preset code for a
            time, the entry's unit for a float, an option for a choice. A key
            of a config or description entry, which holds no value, is left
            out.
        system
            The scanner limits a callable preset is a function of.

        Raises
        ------
        ValueError
            If a key is neither a declared entry nor a prescription entry, a
            time is a preset its entry does not offer, or a choice is not one
            of its options.
        """
        converted: dict[ProtocolKey, Any] = {}
        presets: dict[ProtocolKey, TEPreset | TRPreset] = {}
        for key, value in values.items():
            entry = entries.get(key)
            if isinstance(entry, ConfigParam | Description):
                continue
            if entry is None:
                if key not in PRESCRIPTION:
                    raise ValueError(f"{key} is not an entry of this protocol")
                converted[key] = float(value)
            elif isinstance(entry, TimeParam):
                if entry.presets and value < 0:
                    code = next((c for c in entry.presets if c == value), None)
                    if code is None:
                        raise ValueError(f"{key} does not offer preset {value}")
                    preset = entry.presets[code]
                    presets[key] = code
                    converted[key] = preset(system) if callable(preset) else preset
                else:
                    converted[key] = _to_seconds(value)
            elif isinstance(entry, FloatParam):
                converted[key] = _to_si(value, entry.scale)
            elif isinstance(entry, IntParam):
                converted[key] = int(value)
            elif isinstance(entry, BoolParam):
                converted[key] = bool(value)
            else:
                converted[key] = _member(key, entry, value)
        return cls(entries, converted, presets)

    def to_wire(self) -> dict[ProtocolKey, float | int | bool | str]:
        """Return the wire values of the protocol, the inverse of :meth:`from_wire`.

        A time showing a preset is its preset code. A float is rounded to six
        significant digits, the precision of a float32 parameter, and a time
        to the nearest microsecond, ties to even.
        """
        wire: dict[ProtocolKey, Any] = {}
        for key, value in self._values.items():
            entry = self._entries.get(key)
            if key in self._presets:
                wire[key] = self._presets[key]
            elif isinstance(entry, TimeParam):
                wire[key] = _to_microseconds(value)
            elif isinstance(entry, FloatParam):
                wire[key] = _to_ui(value, entry.scale)
            elif isinstance(entry, IntParam):
                wire[key] = int(value)
            elif isinstance(entry, BoolParam):
                wire[key] = bool(value)
            else:
                wire[key] = value
        return wire

    @property
    def arguments(self) -> dict[str, Any]:
        """The values by the name of the ``init_sequence`` argument each binds.

        The prescription entries bind no argument and are left out.
        """
        arguments = {}
        for key, value in self._values.items():
            argument = getattr(self._entries.get(key), "argument", None)
            if argument is not None:
                arguments[argument] = value
        return arguments

    def preset(self, key: ProtocolKey) -> TEPreset | TRPreset | None:
        """Return the preset a time key shows, or ``None`` where it shows none.

        The preset is the key of the entry's ``presets`` the wire value
        selected: a :class:`~pulserver.protocol.TEPreset` or
        :class:`~pulserver.protocol.TRPreset` member.
        """
        return self._presets.get(key)

    def replace(self, changes: Mapping[ProtocolKey, Any]) -> Protocol:
        """Return a protocol holding ``changes`` in place of the values they name.

        The values are in argument units, and a time given in seconds no
        longer shows a preset.

        Raises
        ------
        ValueError
            If a key is not in the protocol, or a choice is not one of its
            options.
        """
        unknown = sorted(key for key in changes if key not in self._values)
        if unknown:
            raise ValueError(f"not entries of this protocol: {', '.join(unknown)}")
        values, presets = dict(self._values), dict(self._presets)
        for key, value in changes.items():
            entry = self._entries.get(key)
            if isinstance(entry, ChoiceParam):
                value = _member(key, entry, value)
            values[key] = value
            presets.pop(key, None)
        return Protocol(self._entries, values, presets)


class ScannerSequence:
    """A pypulseqpp application exposed to the scanner UI.

    A subclass sets :attr:`app` and, for the arguments the operator edits,
    :attr:`ui`; without ``ui`` the application plays its defaults. The keys of
    ``ui`` are the interpreter's parameter names: the members of
    :data:`~pulserver.protocol.ProtocolKey`, which
    :class:`~pulserver.protocol.UIParam` collects. A plain string naming an
    entry is stored as its member. Entries a request omits keep their initial
    values, the entry's ``default`` or else the application's. An entry
    resolves to the value its argument took in the design, as the application
    records it with ``SequenceApp.resolve``, and otherwise keeps the requested
    value. Times travel as integer microseconds; other float values are read
    and reported to six significant digits, the precision of a float32
    parameter. Either way a reply stored in scanner parameters and sent back
    resolves to itself.

    Attributes
    ----------
    follows : tuple of str
        Which protocol parameter drives the angle of each pulse the sequence
        plays: one entry per distinct pulse, in the order the sequence first
        plays them, ``""`` for one the operator does not move. A sequence that
        excites and then refocuses under the one control declares
        ``(UIParam.FLIP, UIParam.FLIP)``.

        It is declared rather than worked out: which control drives a pulse is
        not a property of the pulse. Empty leaves a scanner to cost the RF when
        it has the design, which is what it does with any sequence.

        The angles are not declared. They are read from a design at the
        application's defaults, and a pulse's share of its parameter is the
        angle it was designed at over the value that parameter was designed
        with -- so a refocusing train whose angles vary gets one share each
        without any of them being written down.

    Raises
    ------
    ValueError
        When a subclass is defined with a ``ui`` key the interpreter does not
        know, which its parser would drop, or with one of the prescription
        entries, which pulserver applies itself.

    Warns
    -----
    DeprecationWarning
        When a subclass sets ``recon``, which has no effect: the reconstruction
        is named by the reconstruction client or, on a console, by the scan.
    """

    app: ClassVar[type[sequences.SequenceApp]]
    ui: ClassVar[Mapping[ProtocolKey, Entry]] = {}
    follows: ClassVar[tuple[ProtocolKey | str, ...]] = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "recon" in cls.__dict__:
            warnings.warn(
                f"{cls.__name__} sets recon, which has no effect: the reconstruction "
                "is named by the reconstruction client or by the scan",
                DeprecationWarning,
                stacklevel=2,
            )
        if "ui" not in cls.__dict__:
            return
        unknown = sorted(name for name in cls.ui if name not in WIRE_NAMES)
        if unknown:
            names = {name.lower(): name for name in WIRE_NAMES}
            close = sorted(
                {
                    names[m]
                    for n in unknown
                    for m in difflib.get_close_matches(n.lower(), names)
                }
            )
            hint = f"; did you mean {close}?" if close else ""
            raise ValueError(
                f"{cls.__name__} declares entries the interpreter does not know: "
                f"{unknown}{hint}"
            )
        ui = {WIRE_NAMES[name]: entry for name, entry in cls.ui.items()}
        reserved = sorted(key.value for key in ui if key in PRESCRIPTION)
        if reserved:
            raise ValueError(
                f"{cls.__name__} binds the prescription entries {reserved}, which "
                "pulserver applies when it builds the IR"
            )
        cls.ui = ui

    def listing(self) -> dict[ProtocolKey, Parameter]:
        """Return the protocol with its schema, valued at the application's defaults.

        An argument defaulting to ``None`` shows the preset that requests ``None``.
        The ``PRESCRIPTION`` entries of :mod:`pulserver.protocol` follow the
        declared ones, not editable in the UI: the offset at zero, the rotation
        at the identity.
        """
        defaults = self.app.protocol()
        listing = {
            key: _parameter(key, entry, defaults) for key, entry in self.ui.items()
        }
        for key in FOV_OFFSET:
            listing[key] = Parameter(
                Kind.FLOAT,
                0.0,
                InputMode.OFF,
                -OFFSET_LIMIT_MM,
                OFFSET_LIMIT_MM,
                0.1,
                "mm",
            )
        for key, value in zip(FOV_ROTATION, np.eye(3).ravel(), strict=True):
            listing[key] = Parameter(
                Kind.FLOAT, float(value), InputMode.OFF, -1.0, 1.0, 1e-6, ""
            )
        return listing

    def rf_pulses(
        self, seq_path: Path | str, designed: Mapping[str, Any]
    ) -> list[RfPulse]:
        """Return the RF a design plays, as a scanner costs it while prescribing.

        The pulses and their angles are read off the sequence a design wrote,
        so nothing is designed to find them; which control drives each is
        :attr:`follows`, declared. A pulse takes a share of its parameter: the
        angle it was designed at over the value the parameter was designed
        with, so a refocusing train whose angles vary gets one share each.

        Parameters
        ----------
        seq_path
            The design's first sequence file.
        designed
            The prescription as designed, by the application's own argument
            names: ``SequenceApp.resolved``.

        Returns
        -------
        list of RfPulse
            One entry per distinct pulse and angle. Empty where
            :attr:`follows` declares nothing, which leaves a scanner to cost
            the RF only once it has the design itself.

        Raises
        ------
        ValueError
            When the sequence does not play as many distinct pulses as
            :attr:`follows` declares, which would silently cost pulses against
            the wrong controls.
        """
        if not self.follows:
            return []
        played = ir.played_rf(pp.io.read(Path(seq_path)))
        order: list[int] = []
        for definition, _ in played:
            if definition not in order:
                order.append(definition)
        if len(order) != len(self.follows):
            raise ValueError(
                f"{type(self).__name__} declares {len(self.follows)} pulses but its "
                f"sequence plays {len(order)} distinct ones"
            )
        driving = {}
        for parameter in self.follows:
            entry = self.ui.get(parameter or "")
            driving[parameter or ""] = (
                float(designed.get(entry.argument, 0.0) or 0.0) if entry else 0.0
            )

        found: dict[tuple[int, int], RfPulse] = {}
        for definition, angle in played:
            # One entry per angle: a definition played at several angles is a
            # train, and each of its pulses is costed at its own.
            key = (definition, round(angle * 1e3))
            if key in found:
                continue
            parameter = self.follows[order.index(definition)] or ""
            drives = driving[parameter]
            found[key] = RfPulse(
                definition=definition,
                flip_deg=angle,
                follows=parameter if drives else "",
                factor=angle / drives if drives else 1.0,
            )
        return list(found.values())

    def validate(
        self, system: pp.Opts, request: Mapping[ProtocolKey, Any]
    ) -> Validation:
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
        self, system: pp.Opts, request: Mapping[ProtocolKey, Any], directory: Path
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
        self, system: pp.Opts, request: Mapping[ProtocolKey, Any]
    ) -> tuple[sequences.SequenceApp | None, Validation]:
        """Return the application a request constructs, and the validation of :meth:`validate`.

        The request is in wire values and so is the validation, which a
        :class:`Protocol` converts to and from the argument units the
        application takes. The application is ``None`` for an invalid request.
        It is constructed once and not designed, so :meth:`write` designs it.

        Raises
        ------
        ValueError
            If the request names an entry the UI does not declare.
        """
        listing = self.listing()
        unknown = set(request) - set(listing)
        if unknown:
            raise ValueError(f"not entries of this protocol: {sorted(unknown)}")
        values = {key: p.value for key, p in listing.items() if p.editable}
        values.update(request)
        try:
            prescribed_rotation(values)
            protocol = Protocol.from_wire(self.ui, values, system)
            app = self.app(system, **protocol.arguments)
            scan_time = app.scan_time()
        except Exception as error:  # a design refuses a protocol by raising
            return None, Validation(
                False, None, str(error) or type(error).__name__, values
            )

        readback = app.resolved
        recorded = {}
        for key, entry in self.ui.items():
            value = readback.get(getattr(entry, "argument", None))
            if value is not None:
                recorded[key] = value
        return app, Validation(
            True, scan_time, "", protocol.replace(recorded).to_wire()
        )


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
