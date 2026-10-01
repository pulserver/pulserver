"""Text form of a protocol on the interpreter wire.

The block grammar is the one ``pulseg_protocol_parse`` reads: listings carry
``name: kind|…`` schema lines, value blocks carry ``name: value`` lines.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

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
        invalid one.
    """

    valid: bool
    duration: float | None
    info: str
    values: dict[str, float | int | bool | str]


def _number(value: float) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    value = float(value)
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return repr(value)


def _listing_line(name: str, p: Parameter) -> str:
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


def format_listing(parameters: Mapping[str, Parameter]) -> str:
    """Format a protocol with its schema, as the ``list`` design call replies it."""
    lines = [_listing_line(name, p) for name, p in parameters.items()]
    return "\n".join([PROTOCOL_BEGIN, *lines, PROTOCOL_END]) + "\n"


def parse_listing(text: str) -> dict[str, Parameter]:
    """Read a protocol with its schema from a listing block."""
    parameters = {}
    for name, value in _block_lines(text):
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
    values: Mapping[str, float | int | bool | str], listing: Mapping[str, Parameter]
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
    text: str, listing: Mapping[str, Parameter]
) -> dict[str, float | int | bool | str]:
    """Read a value block against the listing it was edited from.

    Lines for read-only entries are ignored.

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
            values[name] = p.coerce(value)
    return values


def format_validation(validation: Validation, listing: Mapping[str, Parameter]) -> str:
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


def parse_validation(text: str, listing: Mapping[str, Parameter]) -> Validation:
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

    The scanner costs its RF before a scan from the pulses the sequence will
    play. Which pulses those are, and the shape of each, is settled by the
    design; what the operator moves is an angle. So a pulse travels once, and
    the scanner reads the angle from wherever ``follows`` names for as long as
    the sequence is prescribed.

    The envelope is the magnitude normalised to a peak of one. Every statistic
    a scanner costs a pulse from is computed from that shape alone and does not
    move with the angle, which is why one envelope serves every prescription.

    Attributes
    ----------
    envelope
        Magnitude, normalised to a peak of one, on the RF raster.
    duration_us
        How long the pulse plays.
    flip_deg
        The angle it is designed at. The angle played where ``follows`` is
        empty.
    follows
        Protocol parameter whose value the angle takes; empty is a pulse whose
        angle the operator does not move.
    factor
        What that value is scaled by: an inversion at twice the excitation, a
        refocusing at four fifths of it.
    bandwidth_hz
        Bandwidth at half the spectral peak; 0 where it is not stated.
    count
        How many times the sequence plays it.
    """

    envelope: tuple[float, ...]
    duration_us: float
    flip_deg: float
    follows: str = ""
    factor: float = 1.0
    bandwidth_hz: float = 0.0
    count: int = 1


def format_pulses(pulses: Sequence[RfPulse]) -> str:
    """Format the pulses a sequence plays, as the ``list`` design call replies them."""
    lines = [PULSES_BEGIN]
    for pulse in pulses:
        head = (
            f"{pulse.follows or '-'} {pulse.factor:.7g} {pulse.flip_deg:.7g} "
            f"{pulse.duration_us:.7g} {pulse.bandwidth_hz:.7g} {pulse.count:d} "
            f"{len(pulse.envelope)}"
        )
        samples = " ".join(f"{v:.6g}" for v in pulse.envelope)
        lines.append(f"{head} {samples}".rstrip())
    lines.append(PULSES_END)
    return "\n".join(lines) + "\n"


def parse_pulses(text: str) -> list[RfPulse]:
    """Read the pulses a sequence plays from a listing.

    Returns an empty list where the text holds no block, which is a sequence
    whose RF the scanner is not given in advance.

    Raises
    ------
    ValueError
        If a line states fewer samples than it carries, or a value that is not
        a number.
    """
    found: list[RfPulse] = []
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
        if len(parts) < 7:
            raise ValueError(f"an RF pulse is at least seven values: {stripped!r}")
        follows = "" if parts[0] == "-" else parts[0]
        factor, flip, duration, bandwidth = (float(v) for v in parts[1:5])
        played = int(parts[5])
        count = int(parts[6])
        samples = tuple(float(v) for v in parts[7 : 7 + count])
        if len(samples) != count:
            raise ValueError(
                f"an RF pulse states {count} samples and carries {len(samples)}"
            )
        found.append(
            RfPulse(
                envelope=samples,
                duration_us=duration,
                flip_deg=flip,
                follows=follows,
                factor=factor,
                bandwidth_hz=bandwidth,
                count=played,
            )
        )
    return found
