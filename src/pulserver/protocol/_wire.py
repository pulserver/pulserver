"""Text form of a protocol on the interpreter wire.

The block grammar is the one ``pulseg_protocol_parse`` reads: listings carry
``name: kind|…`` schema lines, value blocks carry ``name: value`` lines. Keys
and values are typed everywhere else: this module is where they become text,
and where text becomes them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ._keys import WIRE_NAMES, ProtocolKey
from ._prescription import PRESCRIPTION
from ._schema import InputMode, Kind, Parameter

#: First and last lines of every protocol block, listing or values.
PROTOCOL_BEGIN = "[Protocol]"
PROTOCOL_END = "[Protocol End]"

#: The block was once delimited by the name of a tool that drove it. A server
#: and an interpreter are deployed separately and may differ in age, so both
#: spellings are read; only the one above is written.
_FORMER_BEGIN = "[NimPulseqGUI Protocol]"
_FORMER_END = "[NimPulseqGUI Protocol End]"


@dataclass(frozen=True)
class Validation:
    """Reply of the ``validate`` design call.

    Attributes
    ----------
    duration
        Scan time in seconds; ``None`` when the plugin reports none.
    values
        The resolved protocol for a valid request, the request itself for an
        invalid one, as wire values by key.
    """

    valid: bool
    duration: float | None
    info: str
    values: dict[ProtocolKey, float | int | bool | str]


def _key(name: str) -> ProtocolKey | str:
    """Return the key a wire name stands for; a name the interpreter does not know stays text."""
    return WIRE_NAMES.get(name, name)


def _number(value: float) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    value = float(value)
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return repr(value)


def _listing_line(name: ProtocolKey, p: Parameter) -> str:
    if p.kind in (Kind.FLOAT, Kind.INT):
        fields = [
            p.kind.value,
            p.mode.value,
            _number(p.value),
            _number(p.range_min),
            _number(p.range_max),
            _number(p.range_incr),
            p.unit,
        ]
        if p.mode is InputMode.DROPDOWN:
            fields += [_number(o) for o in p.options]
    elif p.kind is Kind.BOOL:
        fields = ["bool", "true" if p.value else "false"]
    elif p.kind is Kind.STRINGLIST:
        fields = ["stringlist", str(p.options.index(p.value)), *p.options]
    elif p.kind is Kind.CONFIG:
        fields = ["config", str(p.value)]
    else:
        fields = ["description", str(p.value).replace("\n", "\\n")]
    return f"{name}: {'|'.join(fields)}"


def _block_lines(text: str) -> list[tuple[str, str]]:
    """Return the name and value of each line in a protocol block.

    Lines may be commented out with ``#``, as in a ``.seq`` file header.
    """
    entries, inside = [], False
    for raw in text.splitlines():
        line = raw.strip().lstrip("#").strip()
        if PROTOCOL_END in line or _FORMER_END in line:
            break
        if PROTOCOL_BEGIN in line or _FORMER_BEGIN in line:
            inside = True
            continue
        if inside and ": " in line:
            name, value = line.split(": ", 1)
            entries.append((name.strip(), value.strip()))
    return entries


def format_listing(parameters: Mapping[ProtocolKey, Parameter]) -> str:
    """Format a protocol with its schema, as the ``list`` design call replies it."""
    lines = [_listing_line(name, p) for name, p in parameters.items()]
    return "\n".join([PROTOCOL_BEGIN, *lines, PROTOCOL_END]) + "\n"


def parse_listing(text: str) -> dict[ProtocolKey, Parameter]:
    """Read a protocol with its schema from a listing block."""
    parameters = {}
    for wire_name, value in _block_lines(text):
        name = _key(wire_name)
        kind, *fields = value.split("|")
        kind = Kind(kind)
        if kind in (Kind.FLOAT, Kind.INT):
            cast = float if kind is Kind.FLOAT else int
            mode, number, low, high, incr, unit, *options = fields
            parameters[name] = Parameter(
                kind,
                cast(number),
                InputMode(mode),
                cast(low),
                cast(high),
                cast(incr),
                unit,
                tuple(cast(o) for o in options if o),
            )
        elif kind is Kind.BOOL:
            parameters[name] = Parameter(kind, fields[0] == "true")
        elif kind is Kind.STRINGLIST:
            index, *options = fields
            parameters[name] = Parameter(
                kind, options[int(index)], InputMode.DROPDOWN, options=tuple(options)
            )
        elif kind is Kind.CONFIG:
            parameters[name] = Parameter(kind, int(fields[0]), InputMode.OFF)
        else:
            text_value = "|".join(fields).replace("\\n", "\n")
            parameters[name] = Parameter(kind, text_value)
    return parameters


def format_values(
    values: Mapping[ProtocolKey, float | int | bool | str],
    listing: Mapping[ProtocolKey, Parameter],
) -> str:
    """Format a value block, as requests and replies carry it.

    A stringlist travels as its option index, as the interpreter sends it.
    Read-only entries are left out.
    """
    lines = []
    for name, value in values.items():
        p = listing[name]
        if not p.editable:
            continue
        if p.kind is Kind.BOOL:
            text = "true" if value else "false"
        elif p.kind is Kind.STRINGLIST:
            text = str(p.options.index(value))
        else:
            text = _number(int(value) if p.kind is Kind.INT else float(value))
        lines.append(f"{name}: {text}")
    return "\n".join([PROTOCOL_BEGIN, *lines, PROTOCOL_END]) + "\n"


def parse_values(
    text: str, listing: Mapping[ProtocolKey, Parameter]
) -> dict[ProtocolKey, float | int | bool | str]:
    """Read a value block against the listing it was edited from.

    Lines for read-only entries are ignored. A stringlist's value is the
    option object the listing holds, which is a member of the choices' enum
    where the listing comes from a scanner sequence.

    Raises
    ------
    ValueError
        If a line names a parameter the listing does not declare, or carries a
        value that parameter cannot take.
    """
    values = {}
    for name, value in _block_lines(text):
        if name not in listing:
            raise ValueError(f"{name!r} is not a parameter of this protocol")
        p = listing[name]
        if p.editable:
            values[_key(name)] = p.coerce(value)
    return values


def format_prescription(values: Mapping[ProtocolKey, float]) -> list[str]:
    """Format prescription entries as the ``name: value`` lines of a block.

    The offset is in mm and the rotation unitless, as the entries of a
    protocol carry them.
    """
    return [f"{name}: {_number(value)}" for name, value in values.items()]


def parse_prescription(text: str) -> dict[ProtocolKey, float]:
    """Read the prescription entries out of the ``name: value`` lines of a block.

    Other lines are left out.

    Raises
    ------
    ValueError
        If a prescription line is not a number.
    """
    found = {}
    for line in text.splitlines():
        name, _, value = line.partition(": ")
        if name in PRESCRIPTION:
            found[_key(name)] = float(value)
    return found


def format_validation(
    validation: Validation, listing: Mapping[ProtocolKey, Parameter]
) -> str:
    """Format a ``validate`` reply: status line, info line, value block.

    Whitespace in the info text, newlines included, is folded to single spaces.
    """
    if validation.valid:
        duration = "?" if validation.duration is None else repr(validation.duration)
        status = f"VALID {duration}"
    else:
        status = "INVALID"
    info = " ".join(validation.info.split())
    return f"{status}\nINFO {info}\n" + format_values(validation.values, listing)


def parse_validation(text: str, listing: Mapping[ProtocolKey, Parameter]) -> Validation:
    """Read a ``validate`` reply."""
    status, info, block = text.split("\n", 2)
    word, _, duration = status.partition(" ")
    valid = word == "VALID"
    return Validation(
        valid=valid,
        duration=float(duration) if valid and duration not in ("", "?") else None,
        info=info.removeprefix("INFO").strip(),
        values=parse_values(block, listing),
    )


PULSES_BEGIN = "[RfPulses]"
PULSES_END = "[RfPulses End]"


@dataclass(frozen=True)
class RfPulse:
    """One pulse a sequence plays, and what its flip angle follows.

    The scanner costs the RF of a scan before it runs it. What the operator
    moves between one costing and the next is a flip angle, and sometimes the
    structure: another echo in the train, another line of the matrix, so a
    repetition holding a different number of pulses in a different order.

    **A pulse is a definition of the design it was read from.** Which knob
    drives a pulse is not a property of the pulse, so it is stated by the
    sequence's author and not worked out here: an angle is the thing being
    moved, a position moves when another echo joins a train, and what a pulse
    is for is shared by pulses that are not the same. :attr:`definition` is the
    design's own identity for it, and a scanner reads the same number from that
    design's cache.

    The shape does not travel. It is in the design's own cache, which the
    scanner reads, and every statistic a pulse is costed from is computed from
    that shape at unit peak and does not move with the angle.

    Attributes
    ----------
    definition
        The design's RF definition this pulse is an instance of.
    flip_deg
        The angle it was designed at. The angle played where ``follows`` is
        empty.
    follows
        Protocol parameter whose value the angle takes; empty is a pulse whose
        angle the operator does not move.
    factor
        What that value is scaled by: an inversion at twice the excitation, a
        refocusing at four fifths of it.
    """

    definition: int
    flip_deg: float
    follows: str = ""
    factor: float = 1.0


def format_pulses(pulses: Sequence[RfPulse], design: str = "") -> str:
    """Format the RF a design plays, as the ``list`` design call replies it.

    ``design`` names the design whose cache holds the shapes, which is what a
    scanner reads them from, and by the definition number each pulse states.
    """
    lines = [PULSES_BEGIN, f"design {design or '-'}"]
    for pulse in pulses:
        lines.append(
            f"{pulse.follows or '-'} {pulse.factor:.7g} "
            f"{pulse.flip_deg:.7g} {pulse.definition:d}"
        )
    lines.append(PULSES_END)
    return "\n".join(lines) + "\n"


def parse_pulses(text: str) -> tuple[str, list[RfPulse]]:
    """Read the design and the RF it plays from a listing.

    Returns an empty design and no pulses where the text holds no block, which
    is a sequence the scanner is not given the RF of in advance.

    Raises
    ------
    ValueError
        If a line is not a pulse, or states a value that is not a number.
    """
    found: list[RfPulse] = []
    design = ""
    inside = False
    for line in text.splitlines():
        stripped = line.strip()
        if not inside:
            inside = stripped == PULSES_BEGIN
            continue
        if stripped == PULSES_END:
            break
        if not stripped:
            continue
        parts = stripped.split()
        if parts[0] == "design":
            design = "" if len(parts) < 2 or parts[1] == "-" else parts[1]
            continue
        if len(parts) < 4:
            raise ValueError(f"an RF pulse is four values: {stripped!r}")
        found.append(
            RfPulse(
                definition=int(parts[3]),
                flip_deg=float(parts[2]),
                follows="" if parts[0] == "-" else parts[0],
                factor=float(parts[1]),
            )
        )
    return design, found
