"""The entries a scanner-sequence plugin declares its protocol with, and the values they hold."""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import pypulseqpp as pp

from ..protocol import (
    PRESCRIPTION,
    InputMode,
    Kind,
    Parameter,
    ProtocolKey,
    TEPreset,
    TRPreset,
)
from ..protocol._keys import StrEnum

Preset = float | Callable[[pp.Opts], float] | None

# FLT_DIG: the significant decimal digits a float32 parameter holds through a round trip.
_SIGNIFICANT_DIGITS = 6
_MICROSECOND = Decimal("1e-6")


@dataclass(frozen=True)
class FloatParam:
    """A float UI entry bound to an argument of the sequence function.

    The UI value is the argument divided by ``scale``, in ``unit``; a dropdown
    when it has options. ``default`` is the UI value the protocol starts at;
    ``None`` starts it at the sequence function's default.
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
    """A time UI entry bound to an argument of the sequence function, in seconds.

    Values, ranges and options are integer microseconds, the unit of the
    scanner's time parameters, so the value a parameter holds is the value the design
    reported. Each key of ``presets`` is a dropdown preset; its value is the
    time it requests: seconds, ``None`` for the sequence function's own shortest
    choice, or a function of the scanner limits. The entry is a dropdown when
    it has options or presets. ``default``, microseconds or a key of
    ``presets``, is the value the protocol starts at; ``None`` starts it at the
    sequence function's default.
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
    """An integer UI entry bound to an argument of the sequence function; a dropdown when it has options.

    ``default`` is the value the protocol starts at; ``None`` starts it at the
    sequence function's default.
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
    """A checkbox UI entry bound to an argument of the sequence function.

    ``default`` is the value the protocol starts at; ``None`` starts it at the
    sequence function's default.
    """

    argument: str
    default: bool | None = None


@dataclass(frozen=True)
class ChoiceParam:
    """A choice among the members of a ``StrEnum``, bound to an argument of the sequence function.

    The argument receives the chosen member, a ``str`` equal to its option. The
    options are the members in definition order, and the wire carries the index
    of the chosen one. ``default`` is the member the protocol starts at;
    ``None`` starts it at the sequence function's default, which has to be a member
    or the value of one.
    """

    argument: str
    choices: type[StrEnum]
    default: StrEnum | None = None


@dataclass(frozen=True)
class StatedParam:
    """A choice the sequence states, which binds no argument: a 3D sequence's ``imaging_mode`` is ``3d``.

    It is listed as a choice among all the members of its enum, so that the
    wire carries the value's index among them, as the interpreter reads it
    once at its start; any other value is refused.
    """

    value: StrEnum


@dataclass(frozen=True)
class AveragesParam:
    """The number of signal averages, NEX: how many times the main sequence plays, binding no argument.

    Every plugin has it unless it declares its own. The main sequence of a
    design is played that many times, written into its block table once it
    is designed (:meth:`pypulseqpp.Sequence.expand_repeats`), each repetition
    past the first numbered by ``AVG``. It travels as a float, as the
    interpreter's NEX does, and holds a whole number from 1.
    """

    range_max: int = 16


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
    | StatedParam
    | AveragesParam
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
    sequence function's default.

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


def _averages(key: ProtocolKey, entry: AveragesParam, value: Any) -> int:
    """Return the number of averages a value states.

    Raises
    ------
    ValueError
        If it is not a whole number from 1 to the entry's maximum.
    """
    count = float(value)
    if count != round(count) or not 1 <= count <= entry.range_max:
        raise ValueError(
            f"{key} is a whole number of averages from 1 to {entry.range_max}, not {value}"
        )
    return int(count)


def _stated(key: ProtocolKey, entry: StatedParam, value: Any) -> StrEnum:
    """Return the stated member a value names, by option or index.

    Raises
    ------
    ValueError
        If the value names another member, or none.
    """
    members = list(type(entry.value))
    if (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value < len(members)
    ):
        value = members[value]
    if value != entry.value:
        raise ValueError(f"{key} is {entry.value}, not {value}")
    return entry.value


def _parameter(
    name: ProtocolKey, entry: Entry, defaults: Mapping[str, Any]
) -> Parameter:
    if isinstance(entry, ConfigParam):
        return Parameter(Kind.CONFIG, entry.value, InputMode.OFF)
    if isinstance(entry, Description):
        return Parameter(Kind.DESCRIPTION, entry.text)
    if isinstance(entry, AveragesParam):
        return Parameter(
            Kind.FLOAT, 1.0, InputMode.TYPEIN, 1.0, float(entry.range_max), 1.0, ""
        )
    if isinstance(entry, StatedParam):
        return Parameter(
            Kind.STRINGLIST,
            entry.value,
            InputMode.DROPDOWN,
            options=tuple(type(entry.value)),
        )
    # An entry that states its default may bind a name the app does not take,
    # for the plugin's own hooks to read.
    if entry.argument not in defaults and getattr(entry, "default", None) is None:
        raise ValueError(
            f"{name} binds {entry.argument!r}, which the app does not take"
        )
    default = defaults.get(entry.argument)
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
    """The values of a plugin's protocol, in the units of the sequence function's arguments.

    An immutable mapping from :data:`~pulserver.protocol.ProtocolKey` to value.
    A time is in seconds and a float in the unit of the argument it binds,
    metres for a length; an integer is an ``int``, a checkbox a ``bool`` and a
    choice a member of its enum. A time entry showing a preset holds what the
    preset requests, a number of seconds or ``None`` for the sequence function's own
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
        :attr:`SequencePlugin.protocol` holds them.
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
            The declared entries: :attr:`SequencePlugin.protocol`.
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
            elif isinstance(entry, StatedParam):
                converted[key] = _stated(key, entry, value)
            elif isinstance(entry, AveragesParam):
                converted[key] = _averages(key, entry, value)
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
            elif isinstance(entry, AveragesParam):
                wire[key] = float(value)
            else:
                wire[key] = value
        return wire

    @property
    def arguments(self) -> dict[str, Any]:
        """The values by the name of the sequence function argument each binds.

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
            elif isinstance(entry, StatedParam):
                value = _stated(key, entry, value)
            elif isinstance(entry, AveragesParam):
                value = _averages(key, entry, value)
            values[key] = value
            presets.pop(key, None)
        return Protocol(self._entries, values, presets)
