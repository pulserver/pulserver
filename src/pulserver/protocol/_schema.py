"""Protocol parameters as the scanner UI declares and edits them."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, IntEnum

MAX_DROPDOWN_OPTIONS = 5


class Kind(str, Enum):
    """Type tag of a protocol entry on the wire."""

    FLOAT = "float"
    INT = "int"
    BOOL = "bool"
    STRINGLIST = "stringlist"
    CONFIG = "config"
    DESCRIPTION = "description"


class InputMode(str, Enum):
    """How a numeric entry is presented in the scanner UI."""

    OFF = "off"
    TYPEIN = "typein"
    DROPDOWN = "dropdown"


class TEPreset(IntEnum):
    """Echo-time dropdown entries the UI shows as words.

    Values are the scanner UI's preset codes; they travel as negative dropdown
    options of a time entry.
    """

    MIN_FULL = -1
    MINIMUM = -2
    IN_PHASE = -3
    OUT_PHASE = -4
    MAXIMUM = -5


class TRPreset(IntEnum):
    """Repetition-time dropdown entries the UI shows as words.

    The same code means different presets for TE and TR: ``-1`` is
    ``TRPreset.MINIMUM`` here and ``TEPreset.MIN_FULL`` for an echo time.
    """

    MINIMUM = -1


EDITABLE = frozenset({Kind.FLOAT, Kind.INT, Kind.BOOL, Kind.STRINGLIST})


@dataclass(frozen=True)
class Parameter:
    """One protocol entry: its kind, current value and UI schema.

    Mirrors ``pulseg_protocol_value`` in the interpreter. Numeric entries carry
    a mode, range, increment, unit and, as a dropdown, up to five options; a
    negative option of a time entry is a preset. A stringlist's options are
    its choices and its value is one of them, as the option object: a member
    where the options are the members of an enum, a string otherwise.
    ``config`` and ``description`` entries are declared by the sequence and
    never edited.

    Raises
    ------
    ValueError
        If a dropdown has no options or more than five, or a stringlist value
        is not one of its options.
    """

    kind: Kind
    value: float | int | bool | str
    mode: InputMode = InputMode.TYPEIN
    range_min: float = 0.0
    range_max: float = math.inf
    range_incr: float = 1.0
    unit: str = ""
    options: tuple = ()

    def __post_init__(self) -> None:
        kind = Kind(self.kind)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "mode", InputMode(self.mode))
        if kind in (Kind.FLOAT, Kind.INT):
            cast = float if kind is Kind.FLOAT else int
            object.__setattr__(self, "value", cast(self.value))
            object.__setattr__(self, "options", tuple(cast(o) for o in self.options))
            if self.mode is InputMode.DROPDOWN and not (
                1 <= len(self.options) <= MAX_DROPDOWN_OPTIONS
            ):
                raise ValueError(
                    f"a dropdown takes 1 to {MAX_DROPDOWN_OPTIONS} options, "
                    f"got {len(self.options)}"
                )
        elif kind is Kind.STRINGLIST:
            options = tuple(self.options)
            if self.value not in options:
                raise ValueError(f"{self.value!r} is not one of {', '.join(options)}")
            object.__setattr__(self, "options", options)
            object.__setattr__(self, "value", options[options.index(self.value)])
        elif kind is Kind.BOOL:
            object.__setattr__(self, "value", bool(self.value))
        elif kind is Kind.CONFIG:
            object.__setattr__(self, "value", int(self.value))

    @property
    def editable(self) -> bool:
        """Whether the entry travels in request and reply value blocks."""
        return self.kind in EDITABLE

    def coerce(self, text: str) -> float | int | bool | str:
        """Read a value of this entry from its wire text.

        A stringlist accepts an option, or failing an exact match, an option
        index, and returns the option object.

        Raises
        ------
        ValueError
            If the text is not a value of this entry.
        """
        text = text.strip()
        if self.kind is Kind.FLOAT:
            return float(text)
        if self.kind in (Kind.INT, Kind.CONFIG):
            return int(text)
        if self.kind is Kind.BOOL:
            return text.lower() in ("true", "1")
        if self.kind is Kind.STRINGLIST:
            if text in self.options:
                return self.options[self.options.index(text)]
            if not text.isdecimal() or int(text) >= len(self.options):
                raise ValueError(f"{text!r} is not one of {', '.join(self.options)}")
            return self.options[int(text)]
        return text.replace("\\n", "\n")
