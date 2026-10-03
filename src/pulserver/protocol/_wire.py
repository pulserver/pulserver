"""Text form of a protocol on the interpreter wire.

The block grammar is the one ``pulseg_protocol_parse`` reads: listings carry
``name: kind|…`` schema lines, value blocks carry ``name: value`` lines. Keys
and values are typed everywhere else: this module is where they become text,
and where text becomes them.

The RF blocks, ``[RfDefinitions]`` and ``[RfLayout]``, hold lists of numbers
rather than entries. A number is ASCII decimal with nine significant digits,
numbers are separated by single spaces, and each list is one line, so a reader
takes tokens and not lines of a fixed size.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np

from ._keys import WIRE_NAMES, ProtocolKey
from ._prescription import PRESCRIPTION
from ._schema import InputMode, Kind, Parameter

if TYPE_CHECKING:
    import pypulseqpp as pp

    from ..design import RfLayout

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
    rf_layout
        The RF layout of a valid evaluation; ``None`` where the plugin states
        none or plays no RF, for an invalid request, and for a reply read from
        text. Excluded from equality and from the representation.
    """

    valid: bool
    duration: float | None
    info: str
    values: dict[ProtocolKey, float | int | bool | str]
    rf_layout: RfLayout | None = field(default=None, compare=False, repr=False)


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
    validation: Validation,
    listing: Mapping[ProtocolKey, Parameter],
    *,
    rf_layout: bool = False,
    listed_peak_hz: Sequence[float] = (),
) -> str:
    """Format a ``validate`` reply: status line, info line, value block.

    Whitespace in the info text, newlines included, is folded to single spaces.
    With ``rf_layout``, a valid reply that holds an RF layout ends with its
    ``[RfLayout]`` block, as :func:`format_rf_layout` formats it over
    ``listed_peak_hz``.
    """
    if validation.valid:
        duration = "?" if validation.duration is None else repr(validation.duration)
        status = f"VALID {duration}"
    else:
        status = "INVALID"
    info = " ".join(validation.info.split())
    text = f"{status}\nINFO {info}\n" + format_values(validation.values, listing)
    if rf_layout and validation.valid and validation.rf_layout is not None:
        text += format_rf_layout(validation.rf_layout, listed_peak_hz)
    return text


def parse_validation(text: str, listing: Mapping[ProtocolKey, Parameter]) -> Validation:
    """Read a ``validate`` reply.

    Blocks after the value block are left to :func:`parse_rf_layout`.
    """
    status, info, block = text.split("\n", 2)
    word, _, duration = status.partition(" ")
    valid = word == "VALID"
    return Validation(
        valid=valid,
        duration=float(duration) if valid and duration not in ("", "?") else None,
        info=info.removeprefix("INFO").strip(),
        values=parse_values(block, listing),
    )


@dataclass(frozen=True, eq=False)
class RfDefinitionRecord:
    """One RF definition of an ``[RfDefinitions]`` block.

    Attributes
    ----------
    index
        The number of the definition, from 0, that the runs of an RF layout
        name it by.
    use
        The RF use word of pypulseqpp.
    flip_deg
        The flip angle of the base instance, in degrees.
    peak_hz
        The peak RF amplitude of the base instance, in Hz.
    bandwidth_hz
        The bandwidth :func:`pypulseqpp.calc_rf_bandwidth` measures on the sum
        of the channels, in Hz; ``0.0`` where the sum is zero.
    delay
        The delay of the RF event in its block, in seconds.
    center
        The pulse centre, in seconds from the start of the event.
    duration
        The shape duration, in seconds from the start of the event.
    time
        ``(samples,)``: the sample times, in seconds from the start of the event.
    waveform
        ``(channels, samples)``: the complex waveform of each channel at unit
        peak magnitude.
    """

    index: int
    use: str
    flip_deg: float
    peak_hz: float
    bandwidth_hz: float
    delay: float
    center: float
    duration: float
    time: np.ndarray
    waveform: np.ndarray


@dataclass(frozen=True)
class RfLayoutRecord:
    """The contents of an ``[RfLayout]`` block, one entry per instance of one TR in play order.

    Attributes
    ----------
    period
        The TR in seconds.
    definition
        The number of the definition each instance plays.
    amplitude
        The peak RF amplitude of each instance over the ``peak_hz`` the listing
        states for its definition.
    control
        The control each amplitude is proportional to, or ``None``.
    """

    period: float
    definition: tuple[int, ...]
    amplitude: tuple[float, ...]
    control: tuple[ProtocolKey | None, ...]


_RF_DEFINITIONS = ("[RfDefinitions]", "[RfDefinitions End]")
_RF_LAYOUT = ("[RfLayout]", "[RfLayout End]")


def _g(value: float) -> str:
    """Return a number as the RF blocks write it: nine significant digits, no negative zero."""
    return f"{float(value) + 0.0:.9g}"


def _numbers(values: np.ndarray) -> str:
    return " ".join(_g(value) for value in values)


def _duration(time: np.ndarray) -> float:
    """Return the shape duration of an RF event from its sample times.

    A time base starting at zero ends with the shape. Samples centred on the
    raster extend half a step beyond the last.
    """
    if time.size == 1:
        return 2.0 * float(time[0])
    if time[0] == 0.0:
        return float(time[-1])
    return float(time[-1] + (time[-1] - time[-2]) / 2)


def _bandwidth(definition: pp.RfDefinition) -> float:
    """Return the bandwidth of the sum of the channels of a definition about its stated centre, ``0.0`` for a zero sum."""
    from pypulseqpp import calc_rf_bandwidth

    signal = definition.waveform.sum(axis=0)
    if not signal.any():
        return 0.0
    event = SimpleNamespace(
        t=definition.time,
        signal=signal,
        center=definition.center,
        freq_offset=0.0,
        freq_ppm=0.0,
        phase_offset=0.0,
    )
    return float(calc_rf_bandwidth(event))


def format_rf_definitions(instances: pp.RfInstances) -> str:
    """Format the RF definitions of an evaluation, as the ``list`` design call replies them.

    One ``definition`` line per definition gives its number, use, flip angle,
    peak amplitude, bandwidth, delay, centre, duration, channel count and
    sample count. The lines after it list the sample times, then the real and
    the imaginary parts of each channel, one line each. Returns an empty string
    for no definitions.
    """
    if not instances.definitions:
        return ""
    begin, end = _RF_DEFINITIONS
    lines = [begin]
    for index, definition in enumerate(instances.definitions):
        channels, samples = definition.waveform.shape
        fields = [
            str(index),
            definition.use,
            _g(definition.flip_deg),
            _g(definition.peak_hz),
            _g(_bandwidth(definition)),
            _g(definition.delay),
            _g(definition.center),
            _g(_duration(definition.time)),
            str(channels),
            str(samples),
        ]
        lines.append("definition " + " ".join(fields))
        lines.append(_numbers(definition.time))
        for channel in definition.waveform:
            lines.append(_numbers(channel.real))
            lines.append(_numbers(channel.imag))
    lines.append(end)
    return "\n".join(lines) + "\n"


def _over_listed_peaks(
    layout: RfLayout, listed_peak_hz: Sequence[float]
) -> list[float]:
    """Return per definition the ratio of the layout's ``peak_hz`` to the listed one.

    The ratio is 1.0 where no peak is listed for the definition, or the listed
    peak is zero.
    """
    factors = []
    for index, definition in enumerate(layout.instances.definitions):
        listed = listed_peak_hz[index] if index < len(listed_peak_hz) else 0.0
        factors.append(definition.peak_hz / listed if listed > 0 else 1.0)
    return factors


def format_rf_layout(layout: RfLayout, listed_peak_hz: Sequence[float] = ()) -> str:
    """Format the RF layout of a validation, as the ``validate`` design call replies it.

    The instances are run-length encoded in play order: consecutive instances
    of one definition and control, with amplitudes that print alike, are one
    ``run`` line holding their count, so a train of equal pulses is one line.
    No samples are carried; a run names its definition by the number the
    ``[RfDefinitions]`` block of the listing gives it.

    The amplitude of a run is the peak RF amplitude of its instances over
    ``listed_peak_hz[definition]``, the ``peak_hz`` in Hz the listing states for
    the definition, so ``amplitude * peak_hz * waveform`` is what the instance
    plays. A definition that ``listed_peak_hz`` does not reach, or lists as zero,
    keeps the amplitude of the layout, which is over the ``peak_hz`` of the
    layout's own definition. Returns an empty string for a layout without
    instances.
    """
    instances = layout.instances
    if not len(instances.definition):
        return ""
    numbers = instances.definition.tolist()
    factors = _over_listed_peaks(layout, listed_peak_hz)
    rows = zip(
        (str(number) for number in numbers),
        (
            _g(amplitude * factors[number])
            for amplitude, number in zip(instances.amplitude, numbers, strict=True)
        ),
        ("-" if control is None else control.value for control in layout.control),
        strict=True,
    )
    begin, end = _RF_LAYOUT
    lines = [begin, f"period {_g(layout.period)}"]
    for row, group in itertools.groupby(rows):
        lines.append("run " + " ".join(row) + f" {len(list(group))}")
    lines.append(end)
    return "\n".join(lines) + "\n"


def _block(text: str, begin: str, end: str) -> list[str] | None:
    """Return the stripped lines between the first ``begin`` line and ``end``, ``None`` where there is no such block."""
    lines: list[str] = []
    inside = False
    for raw in text.splitlines():
        line = raw.strip()
        if not inside:
            inside = line == begin
        elif line == end:
            return lines
        else:
            lines.append(line)
    if inside:
        raise ValueError(f"{begin} is not closed by {end}")
    return None


def parse_rf_definitions(text: str) -> list[RfDefinitionRecord]:
    """Read the ``[RfDefinitions]`` block of a ``list`` reply.

    Blocks other than the one it reads are ignored. Returns an empty list where
    the text holds none.

    Raises
    ------
    ValueError
        If the block is not closed, a definition line is not ten values, or
        the lines of a definition are not the ones its channel and sample
        counts state.
    """
    lines = _block(text, *_RF_DEFINITIONS)
    found = []
    at = 0
    while lines is not None and at < len(lines):
        head = lines[at].split()
        if head[:1] != ["definition"] or len(head) != 11:
            raise ValueError(f"not a definition line of an RF block: {lines[at]!r}")
        channels, samples = int(head[9]), int(head[10])
        rows = lines[at + 1 : at + 2 + 2 * channels]
        if len(rows) != 1 + 2 * channels:
            raise ValueError(f"definition {head[1]} is cut short")
        values = [np.array(row.split(), dtype=float) for row in rows]
        if any(row.size != samples for row in values):
            raise ValueError(f"definition {head[1]} does not hold {samples} samples")
        flip, peak, bandwidth, delay, center, duration = (float(v) for v in head[3:9])
        found.append(
            RfDefinitionRecord(
                index=int(head[1]),
                use=head[2],
                flip_deg=flip,
                peak_hz=peak,
                bandwidth_hz=bandwidth,
                delay=delay,
                center=center,
                duration=duration,
                time=values[0],
                waveform=np.array(values[1::2]) + 1j * np.array(values[2::2]),
            )
        )
        at += 2 + 2 * channels
    return found


def parse_rf_layout(text: str) -> RfLayoutRecord | None:
    """Read the ``[RfLayout]`` block of a ``validate`` reply.

    The runs are decoded to one entry per instance. Blocks other than the one
    it reads are ignored. Returns ``None`` where the text holds none.

    Raises
    ------
    ValueError
        If the block is not closed, states no period, or holds a line that is
        neither the period nor a run of four values with a positive count.
    """
    lines = _block(text, *_RF_LAYOUT)
    if lines is None:
        return None
    period = None
    definition: list[int] = []
    amplitude: list[float] = []
    control: list[ProtocolKey | None] = []
    for line in filter(None, lines):
        word, *fields = line.split()
        if word == "period" and len(fields) == 1:
            period = float(fields[0])
        elif word == "run" and len(fields) == 4 and int(fields[3]) > 0:
            count = int(fields[3])
            definition += [int(fields[0])] * count
            amplitude += [float(fields[1])] * count
            control += [None if fields[2] == "-" else _key(fields[2])] * count
        else:
            raise ValueError(f"not a line of an RF layout: {line!r}")
    if period is None:
        raise ValueError("an RF layout states its period")
    return RfLayoutRecord(period, tuple(definition), tuple(amplitude), tuple(control))
