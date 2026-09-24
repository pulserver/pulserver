"""The event and shape libraries of a sequence, as the IR conversion reads them."""

from __future__ import annotations

__all__ = [
    "BlockExtensions",
    "SequenceLibraries",
    "Shape",
    "SpecificationLibraries",
    "block_extensions",
    "conversion_payload",
    "sequence_libraries",
    "specification_libraries",
]

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pypulseqpp as pp
from numpy.typing import NDArray
from pypulseqpp import _ext as _core

#: Use tag of an RF event, as a Pulseq file writes it. A pulse the file does
#: not label is 0, which is what a reader that switches on the tag treats as
#: "unknown"; "undefined" is what pypulseqpp calls the same thing.
_RF_USE = {
    "": 0,
    "undefined": 0,
    "excitation": 1,
    "refocusing": 2,
    "inversion": 3,
    "saturation": 4,
    "preparation": 5,
    "other": 6,
}

_TRAPEZOID = 0
_ARBITRARY = 1

#: Bands an RF spectrum row has room for, as the cache stores them.
MAX_BANDS = 8

#: Labels that count a position in the scan, in the order Pulseq numbers them.
COUNTER_LABELS = ("SLC", "SEG", "REP", "AVG", "SET", "ECO", "PHS", "LIN", "PAR", "ACQ")

#: Labels that state something about a block rather than counting one.
FLAG_LABELS = (
    "TRID",
    "NAV",
    "REV",
    "SMS",
    "REF",
    "IMA",
    "NOISE",
    "PMC",
    "NOROT",
    "NOPOS",
    "NOSCL",
    "ONCE",
)

#: Extension specification naming each kind of row a block can point at.
_SPECIFICATIONS = {
    "rotation": "ROTATIONS",
    "rf_shim": "RF_SHIMS",
    "trigger": "TRIGGERS",
    "soft_delay": "DELAYS",
}

#: Label ids the raw library rows carry. Past ACQ these are the scanner
#: converter's own numbering, which is not pypulseqpp's: a file names its
#: labels, so only the number they are held under differs.
LABEL_IDS = {
    "SLC": 1,
    "SEG": 2,
    "REP": 3,
    "AVG": 4,
    "SET": 5,
    "ECO": 6,
    "PHS": 7,
    "LIN": 8,
    "PAR": 9,
    "ACQ": 10,
    "NAV": 11,
    "REV": 12,
    "SMS": 13,
    "REF": 14,
    "IMA": 15,
    "NOISE": 16,
    "PMC": 17,
    "NOROT": 18,
    "NOPOS": 19,
    "NOSCL": 20,
    "ONCE": 21,
    "TRID": 22,
    "OFF": 23,
}

#: Kind each extension specification is, as the converter numbers the kinds.
_EXTENSION_KINDS = {
    "TRIGGERS": 1,
    "ROTATIONS": 2,
    "LABELSET": 3,
    "LABELINC": 4,
    "RF_SHIMS": 5,
    "DELAYS": 6,
}


#: Hint a soft delay names, as a Pulseq file numbers it; anything else is -1.
_HINTS = {
    "TE": 1,
    "TR": 2,
    "TI": 3,
    "ESP": 4,
    "RECTIME": 5,
    "T2PREP": 6,
    "TE2": 7,
}


@dataclass(frozen=True)
class Shape:
    """One entry of the shape library.

    Attributes
    ----------
    num_uncompressed_samples
        Samples :attr:`samples` decompresses to.
    samples
        Run-length encoded on the derivative, as Pulseq stores a shape.
    """

    num_uncompressed_samples: int
    samples: NDArray[np.float64]


@dataclass(frozen=True)
class SequenceLibraries:
    """The libraries of one sequence, in the layout the IR conversion keys on.

    Event ids are the file's own, so a row's position is the id a block names
    and row 0 is id 1. A library entry no block plays is not recoverable and
    is absent: the rows reach as far as the largest id in use.

    Shape ids are minted here in first-use order. pypulseqpp hands a decoded
    event its samples but not the shape it is stored under, so what survives
    is which events share a shape, not the number the file gave it; shapes a
    file left duplicated are one here.

    Attributes
    ----------
    blocks : NDArray[np.float64]
        ``(N, 7)``: duration in block-duration rasters, then the rf, gx, gy,
        gz, adc and extension ids, 0 for an event the block does not play.
    rf : NDArray[np.float64]
        ``(R, 10)``: amplitude in Hz; the magnitude, phase and time shape ids;
        centre and delay in µs; the frequency and phase ppm offsets; the
        frequency offset in Hz and the phase offset in rad.
    rf_use : NDArray[np.int32]
        ``(R,)``: 0 unknown, 1 excitation, 2 refocusing, 3 inversion,
        4 saturation, 5 preparation, 6 other.
    rf_spectra : NDArray[np.float64]
        ``(R, 3 + MAX_BANDS)``: bandwidth in Hz, number of bands, widest band's
        bandwidth in Hz, then each band's offset from the carrier in Hz, as
        ``pypulseqpp.calc_rf_bandwidth`` measures them; unused offsets are 0.
    grad : NDArray[np.float64]
        ``(G, 7)``, by type in column 0. A trapezoid (0): amplitude in Hz/m,
        rise, flat and fall times in µs, delay in µs. An arbitrary gradient
        (1): amplitude in Hz/m, the waveform's first and last values in Hz/m,
        the waveform and time shape ids, delay in µs.
    adc : NDArray[np.float64]
        ``(A, 8)``: sample count; dwell in ns; delay in µs; the frequency and
        phase ppm offsets; the frequency offset in Hz; the phase offset in
        rad; the phase modulation shape id.
    shapes : tuple[Shape, ...]
        Indexed by shape id minus one.
    """

    blocks: NDArray[np.float64]
    rf: NDArray[np.float64]
    rf_use: NDArray[np.int32]
    rf_spectra: NDArray[np.float64]
    grad: NDArray[np.float64]
    adc: NDArray[np.float64]
    shapes: tuple[Shape, ...]


def sequence_libraries(sequence: Any) -> SequenceLibraries:
    """Read one ``pypulseqpp.Sequence`` into the libraries the IR conversion keys on.

    Each event id is decoded once, from the first block that plays it.

    Raises
    ------
    ValueError
        If an event id is played on no block the sequence can decode.
    """
    core = sequence._native
    events = np.asarray(core.block_events(), dtype=np.int64)
    durations = np.asarray(core.block_durations(), dtype=np.float64)

    blocks = np.zeros((durations.size, 7), dtype=np.float64)
    blocks[:, 0] = np.rint(durations / core.block_duration_raster())
    blocks[:, 1:] = events

    shapes = _ShapeTable()
    rf, rf_use, rf_spectra = _rf_library(core, events, shapes)
    grad = _grad_library(core, events, shapes)
    adc = _adc_library(core, events, shapes)
    return SequenceLibraries(
        blocks, rf, rf_use, rf_spectra, grad, adc, shapes.entries()
    )


# %% private module subroutines


class _ShapeTable:
    """Interns decompressed waveforms, handing out 1-based ids in first-use order."""

    def __init__(self) -> None:
        self._ids: dict[bytes, int] = {}
        self._entries: list[Shape] = []

    def intern(self, samples: NDArray[np.float64]) -> int:
        samples = np.ascontiguousarray(samples, dtype=np.float64)
        key = samples.tobytes()
        if key not in self._ids:
            self._entries.append(
                Shape(int(samples.size), np.asarray(_core.compress_shape(samples)))
            )
            self._ids[key] = len(self._entries)
        return self._ids[key]

    def entries(self) -> tuple[Shape, ...]:
        return tuple(self._entries)


def _decoded(
    core: Any, events: NDArray[np.int64], column: int | tuple[int, ...]
) -> Any:
    """Yield ``(id, event)`` for every id played in ``column``, in id order.

    ``column`` is an index into the block table's event columns, or several
    when one library serves more than one, as the three gradient axes do.
    """
    columns = (column,) if isinstance(column, int) else column
    names = {0: "rf", 1: "gx", 2: "gy", 3: "gz", 4: "adc"}
    first: dict[int, tuple[int, str]] = {}
    for index, row in enumerate(events):
        for which in columns:
            identifier = int(row[which])
            if identifier > 0 and identifier not in first:
                first[identifier] = (index + 1, names[which])
    for identifier in sorted(first):
        block, name = first[identifier]
        event = core.decode_block(block)[name]
        if event is None:
            raise ValueError(f"block {block} does not decode the {name} it names")
        yield identifier, event


def _rows(decoded: list[tuple[int, Any]], width: int) -> NDArray[np.float64]:
    """Rows reaching the largest id played; an id no block plays keeps a zero row."""
    return np.zeros((max((i for i, _ in decoded), default=0), width), dtype=np.float64)


def _micro(seconds: float) -> float:
    return float(np.rint(float(seconds) * 1e6))


def _time_shape(times: NDArray[np.float64], raster: float, shapes: _ShapeTable) -> int:
    """Return the shape id of a vector of sample times, 0 when it lies on the raster.

    Times are stored in raster units. The grid ``0.5, 1.5, ...`` is what an
    event with no time shape plays, so a file that stores that grid explicitly
    reads back as an event with none.
    """
    ticks = np.asarray(times, dtype=np.float64) / raster
    if ticks.size and np.allclose(ticks, np.arange(ticks.size) + 0.5):
        return 0
    return shapes.intern(ticks)


def _rf_library(
    core: Any, events: NDArray[np.int64], shapes: _ShapeTable
) -> tuple[NDArray[np.float64], NDArray[np.int32], NDArray[np.float64]]:
    decoded = list(_decoded(core, events, 0))
    rows = _rows(decoded, 10)
    uses = np.zeros(rows.shape[0], dtype=np.int32)
    spectra = np.zeros((rows.shape[0], 3 + MAX_BANDS), dtype=np.float64)
    raster = core.rf_raster_time()
    measured: dict[tuple[float, float, float], NDArray[np.float64]] = {}
    for identifier, event in decoded:
        row = rows[identifier - 1]
        row[0] = event.amplitude
        row[1] = shapes.intern(np.asarray(event.magnitude))
        row[2] = shapes.intern(np.asarray(event.phase))
        row[3] = _time_shape(np.asarray(event.t), raster, shapes)
        row[4] = _micro(event.center)
        row[5] = _micro(event.delay)
        row[6] = event.freq_ppm
        row[7] = event.phase_ppm
        row[8] = event.freq_offset
        row[9] = event.phase_offset
        uses[identifier - 1] = _RF_USE[event.use]
        # The spectrum's shape depends on the waveform alone: amplitude scales
        # it and a frequency offset moves it, neither of which the bands see.
        key = (row[1], row[2], row[3])
        if key not in measured:
            measured[key] = _spectrum_row(event, raster)
        spectra[identifier - 1] = measured[key]
    return rows, uses, spectra


def _spectrum_row(event: Any, raster: float) -> NDArray[np.float64]:
    """Bandwidth, band count, widest band and band offsets of one RF event, in Hz.

    Measured at the carrier: the event's frequency offsets are zeroed, which
    is harmless because a decoded event is a snapshot the sequence does not
    hold, and its offsets are already in the library row.
    """
    event.freq_offset = 0.0
    event.freq_ppm = 0.0
    result = pp.calc_rf_bandwidth(event, dt=raster, compat=False)
    row = np.zeros(3 + MAX_BANDS, dtype=np.float64)
    count = min(result.num_bands, MAX_BANDS)
    row[0] = result.bandwidth
    row[1] = max(result.num_bands, 1)
    row[2] = result.band_bandwidths.max() if result.num_bands else result.bandwidth
    # Measured on 10 Hz bins; below a millihertz an offset is the float32 of a
    # binary file's shape samples, not a property of the pulse.
    row[3 : 3 + count] = np.round(result.band_offsets[:count], 3)
    return row


def _grad_library(
    core: Any, events: NDArray[np.int64], shapes: _ShapeTable
) -> NDArray[np.float64]:
    decoded = list(_decoded(core, events, (1, 2, 3)))
    rows = _rows(decoded, 7)
    raster = core.grad_raster_time()
    for identifier, event in decoded:
        row = rows[identifier - 1]
        if event.type == "trap":
            row[0] = _TRAPEZOID
            row[1] = event.amplitude
            row[2] = _micro(event.rise_time)
            row[3] = _micro(event.flat_time)
            row[4] = _micro(event.fall_time)
            row[5] = _micro(event.delay)
            continue
        waveform = np.asarray(event.waveform, dtype=np.float64)
        # A gradient of no amplitude plays nothing whatever its stored shape.
        normalised = (
            waveform / event.amplitude if event.amplitude else np.zeros_like(waveform)
        )
        row[0] = _ARBITRARY
        row[1] = event.amplitude
        row[2] = event.first
        row[3] = event.last
        row[4] = shapes.intern(normalised)
        row[5] = _time_shape(np.asarray(event.tt), raster, shapes)
        row[6] = _micro(event.delay)
    return rows


def _adc_library(
    core: Any, events: NDArray[np.int64], shapes: _ShapeTable
) -> NDArray[np.float64]:
    decoded = list(_decoded(core, events, 4))
    rows = _rows(decoded, 8)
    for identifier, event in decoded:
        row = rows[identifier - 1]
        row[0] = event.num_samples
        row[1] = float(np.rint(event.dwell * 1e9))
        row[2] = _micro(event.delay)
        row[3] = event.freq_ppm
        row[4] = event.phase_ppm
        row[5] = event.freq_offset
        row[6] = event.phase_offset
        modulation = np.asarray(event.phase_modulation, dtype=np.float64)
        if modulation.size and modulation.size != event.num_samples:
            raise ValueError(
                f"ADC {identifier} acquires {event.num_samples:g} samples but its "
                f"phase modulation has {modulation.size}"
            )
        row[7] = shapes.intern(modulation) if modulation.size else 0
    return rows


@dataclass(frozen=True)
class BlockExtensions:
    """Per block, what its extension chain resolves to.

    Attributes
    ----------
    labelset, labelinc : dict[str, NDArray[np.int32]]
        One array per label of :data:`COUNTER_LABELS`, holding what the block
        sets or increments it by. 0 where the block says nothing about it,
        which is also what setting it to zero looks like.
    flags : dict[str, NDArray[np.int32]]
        One array per label of :data:`FLAG_LABELS`; -1 where the block states
        none. ``TRID`` is an identifier rather than a flag and carries the
        block's own value, not a running one.
    rotation, rf_shim, trigger, soft_delay : NDArray[np.int32]
        Row of that specification the block points at, counted from 0; -1 for
        none.
    """

    labelset: dict[str, NDArray[np.int32]]
    labelinc: dict[str, NDArray[np.int32]]
    flags: dict[str, NDArray[np.int32]]
    rotation: NDArray[np.int32]
    rf_shim: NDArray[np.int32]
    trigger: NDArray[np.int32]
    soft_delay: NDArray[np.int32]


def block_extensions(sequence: Any) -> BlockExtensions:
    """Resolve every block's extension chain.

    Each distinct chain is resolved once, from the first block that plays it:
    blocks sharing a chain head resolve to the same thing.
    """
    core = sequence._native
    kinds = _declared_types(core)
    heads = np.asarray(core.block_events(), dtype=np.int64)[:, 5]
    resolved = {0: _Chain()}
    for index, head in enumerate(heads):
        if int(head) not in resolved:
            resolved[int(head)] = _resolve(core, int(head), index + 1, kinds)

    count = heads.size
    labelset = {name: np.zeros(count, dtype=np.int32) for name in COUNTER_LABELS}
    labelinc = {name: np.zeros(count, dtype=np.int32) for name in COUNTER_LABELS}
    flags = {name: np.full(count, -1, dtype=np.int32) for name in FLAG_LABELS}
    points = {name: np.full(count, -1, dtype=np.int32) for name in _SPECIFICATIONS}
    for index, head in enumerate(heads):
        chain = resolved[int(head)]
        for name, value in chain.labelset.items():
            labelset[name][index] = value
        for name, value in chain.labelinc.items():
            labelinc[name][index] = value
        for name, value in chain.flags.items():
            flags[name][index] = value
        for name, value in chain.points.items():
            points[name][index] = value
    return BlockExtensions(labelset, labelinc, flags, **points)


@dataclass
class _Chain:
    """One extension chain, resolved."""

    labelset: dict[str, int] = field(default_factory=dict)
    labelinc: dict[str, int] = field(default_factory=dict)
    flags: dict[str, int] = field(default_factory=dict)
    points: dict[str, int] = field(default_factory=dict)


def _declared_types(core: Any) -> dict[str, int]:
    """Return the type number the file declared for each specification it carries.

    Read from the numbers the file declared rather than asked for by name:
    asking by name mints one for a specification the file does not carry, and
    reading a sequence does not change it.
    """
    declared = {
        core.extension_type_name(number): number
        for number in range(1, 8)
        if core.extension_type_name(number)
    }
    return {name: declared.get(name, -1) for name in _EXTENSION_KINDS}


def _resolve(core: Any, head: int, block: int, types: dict[str, int]) -> _Chain:
    """Resolve one chain: its labels from the block that plays it, its rows from the chain."""
    chain = _Chain()
    links = np.asarray(core.extension_chain(head), dtype=np.int64).reshape(2, -1)
    kinds = {
        name: types[specification] for name, specification in _SPECIFICATIONS.items()
    }
    for name, kind in kinds.items():
        referenced = links[1][links[0] == kind]
        if referenced.size:
            # The chain names a 1-based row; the tables count from 0.
            chain.points[name] = int(referenced[-1]) - 1
    for label in core.decode_block(block).get("label") or ():
        target = chain.labelset if label.setting else chain.labelinc
        if label.label in FLAG_LABELS:
            chain.flags[label.label] = int(label.value)
        else:
            target[label.label] = int(label.value)
    return chain


@dataclass(frozen=True)
class SpecificationLibraries:
    """The rows a block's extension chain points at, indexed by the file's ids.

    Row 0 is id 1, as in :class:`SequenceLibraries`. A row no block points at
    is not recoverable and is absent.

    Attributes
    ----------
    rotations : NDArray[np.float64]
        ``(R, 4)``: the quaternion turning the block's gradients.
    triggers : NDArray[np.float64]
        ``(T, 4)``: control code, channel code, delay in µs, duration in µs.
    rf_shims : tuple[NDArray[np.float64], ...]
        Per row, magnitude and phase alternating, one pair per transmit
        channel; phase in rad.
    soft_delays : NDArray[np.float64]
        ``(S, 4)``: the delay's number, offset in µs, factor, and the id of
        the hint it names; -1 for a hint the format does not number.
    labelset, labelinc : NDArray[np.float64]
        ``(L, 2)``: the value and the label it applies to, numbered as
        :data:`LABEL_IDS` numbers it.
    referenced : dict[str, tuple[int, ...]]
        Per table, the ids some chain points at. A row between them that no
        chain names is not recoverable and reads as zeros.
    """

    rotations: NDArray[np.float64]
    triggers: NDArray[np.float64]
    rf_shims: tuple[NDArray[np.float64], ...]
    soft_delays: NDArray[np.float64]
    labelset: NDArray[np.float64]
    labelinc: NDArray[np.float64]
    referenced: dict[str, tuple[int, ...]]


def specification_libraries(sequence: Any) -> SpecificationLibraries:
    """Read the rows every extension chain of a sequence points at.

    Each chain is decoded once and its events are attributed to the ids the
    chain names, in chain order, which is the order a decoded block lists them
    in.
    """
    core = sequence._native
    types = _declared_types(core)
    heads = np.asarray(core.block_events(), dtype=np.int64)[:, 5]
    kinds = {
        name: types[specification] for name, specification in _SPECIFICATIONS.items()
    }
    kinds["labelset"] = types["LABELSET"]
    kinds["labelinc"] = types["LABELINC"]

    rows: dict[str, dict[int, Any]] = {name: {} for name in kinds}
    seen: set[int] = set()
    for index, head in enumerate(heads):
        head = int(head)
        if head == 0 or head in seen:
            continue
        seen.add(head)
        _read_chain(core, head, index + 1, kinds, rows)

    return SpecificationLibraries(
        rotations=_table(rows["rotation"], 4),
        triggers=_table(rows["trigger"], 4),
        rf_shims=_ragged(rows["rf_shim"]),
        soft_delays=_table(rows["soft_delay"], 4),
        labelset=_table(rows["labelset"], 2),
        labelinc=_table(rows["labelinc"], 2),
        referenced={
            table: tuple(sorted(rows[name]))
            for name, table in (
                ("rotation", "rotations"),
                ("trigger", "triggers"),
                ("rf_shim", "rf_shims"),
                ("soft_delay", "soft_delays"),
                ("labelset", "labelset"),
                ("labelinc", "labelinc"),
            )
        },
    )


def _read_chain(
    core: Any,
    head: int,
    block: int,
    kinds: dict[str, int],
    rows: dict[str, dict[int, Any]],
) -> None:
    links = np.asarray(core.extension_chain(head), dtype=np.int64).reshape(2, -1)
    decoded = core.decode_block(block)
    labels = list(decoded.get("label") or ())
    triggers = list(decoded.get("trig") or ())
    for name, kind in kinds.items():
        referenced = links[1][links[0] == kind]
        for position, identifier in enumerate(referenced):
            identifier = int(identifier)
            if identifier in rows[name]:
                continue
            row = _specification_row(name, position, decoded, labels, triggers)
            if row is not None:
                rows[name][identifier] = row


def _specification_row(
    name: str,
    position: int,
    decoded: dict[str, Any],
    labels: list[Any],
    triggers: list[Any],
) -> Any:
    """One row of a specification, taken from the event at ``position`` of its kind."""
    if name == "rotation":
        return np.asarray(decoded["rotation"].quaternion, dtype=np.float64)
    if name == "trigger":
        event = triggers[position]
        return np.array(
            [
                event.control,
                event.channel_code,
                _micro(event.delay),
                _micro(event.duration),
            ]
        )
    if name == "rf_shim":
        shim = np.asarray(decoded["rf_shim"].shim_vector)
        return np.stack([np.abs(shim), np.angle(shim)], axis=1).reshape(-1)
    if name == "soft_delay":
        event = decoded["soft_delay"]
        return np.array(
            [
                event.numID,
                _micro(event.offset),
                event.factor,
                _HINTS.get(event.hint, -1),
            ]
        )
    matching = [label for label in labels if label.setting == (name == "labelset")]
    event = matching[position]
    return np.array([event.value, LABEL_IDS.get(event.label, -1)])


def _table(rows: dict[int, Any], width: int) -> NDArray[np.float64]:
    table = np.zeros((max(rows, default=0), width), dtype=np.float64)
    for identifier, row in rows.items():
        table[identifier - 1] = row
    return table


def _ragged(rows: dict[int, Any]) -> tuple[NDArray[np.float64], ...]:
    empty = np.zeros(0, dtype=np.float64)
    return tuple(
        rows.get(identifier + 1, empty) for identifier in range(max(rows, default=0))
    )


def conversion_payload(sequence: Any) -> dict[str, Any]:
    """Everything one sequence file contributes to a conversion.

    The libraries, the specification tables and the chain rows that link a
    block to them, in the layout a parsed file holds: times in µs, fields of
    view in cm, rasters in µs.

    Chain rows are minted here. pypulseqpp names a block's chain by its head
    and hands back the links it resolves to, not the rows they are stored in,
    so the chain is written out again -- one run of rows per distinct head,
    each block pointing at the head of its own.
    """
    core = sequence._native
    libraries = sequence_libraries(sequence)
    specifications = specification_libraries(sequence)
    chains, heads = _chain_rows(core)
    blocks = libraries.blocks.copy()
    blocks[:, 6] = heads
    rf, grad, adc, rf_use, rf_spectra = _compact(blocks, libraries)
    declared = core.definitions()

    return {
        "version": [
            core.version_major(),
            core.version_minor(),
            core.version_revision(),
        ],
        "rasters": [
            1e6 * core.rf_raster_time(),
            1e6 * core.grad_raster_time(),
            1e6 * core.adc_raster_time(),
            1e6 * core.block_duration_raster(),
        ],
        "reserved": {
            "fov": [100.0 * value for value in _numbers(declared, "FOV", 3)],
            "matrix": _numbers(declared, "Matrix", 3),
            "nav_fov": [100.0 * value for value in _numbers(declared, "NavFOV", 3)],
            "nav_matrix": _numbers(declared, "NavMatrix", 3),
            "total_duration": _numbers(declared, "TotalDuration", 1)[0],
            "enable_pmc": int(_numbers(declared, "EnablePmc", 1)[0]),
            "num_gain_cal_readouts": int(
                _numbers(declared, "NumGainCalibrationReadouts", 1)[0]
            ),
            "enable_sar_burst_mode": int(
                _numbers(declared, "EnableSarBurstMode", 1)[0]
            ),
            "vop_sar_ratio": 0.0,
            "vop_global_sar_ratio": 0.0,
            "name": str(declared.get("Name", "")),
            "next_sequence": str(declared.get("NextSequence", "")),
        },
        "definitions": {name: _texts(value) for name, value in declared.items()},
        "blocks": blocks,
        "rf": rf,
        "rf_use": rf_use,
        "rf_spectra": rf_spectra,
        "grad": grad,
        "adc": adc,
        "shapes": [
            (shape.num_uncompressed_samples, shape.samples)
            for shape in libraries.shapes
        ],
        "extensions": chains,
        "extension_map": _extension_map(core),
        "triggers": specifications.triggers,
        "rotations": specifications.rotations,
        "labelset": specifications.labelset,
        "labelinc": specifications.labelinc,
        "soft_delays": specifications.soft_delays,
        "rf_shims": list(specifications.rf_shims),
    }


def _compact(
    blocks: NDArray[np.float64], libraries: SequenceLibraries
) -> tuple[Any, Any, Any, Any, Any]:
    """Drop the library rows no block plays, renumbering the block table in place.

    A row a decoded sequence never named is not recoverable, and a placeholder
    kept in its place would deduplicate into a definition of its own that
    nothing plays. Numbering the played rows again keeps the libraries to what
    the scan actually asks for.
    """
    rf, rf_map = _played(libraries.rf, blocks, (1,))
    grad, grad_map = _played(libraries.grad, blocks, (2, 3, 4))
    adc, adc_map = _played(libraries.adc, blocks, (5,))
    played_rf = [old - 1 for old in sorted(rf_map, key=rf_map.get)]
    uses = np.array([libraries.rf_use[row] for row in played_rf], dtype=np.int32)
    spectra = libraries.rf_spectra[played_rf]
    for columns, mapping in (((1,), rf_map), ((2, 3, 4), grad_map), ((5,), adc_map)):
        for column in columns:
            blocks[:, column] = [
                mapping.get(int(value), 0) for value in blocks[:, column]
            ]
    return rf, grad, adc, uses, spectra


def _played(
    library: NDArray[np.float64], blocks: NDArray[np.float64], columns: tuple[int, ...]
) -> tuple[NDArray[np.float64], dict[int, int]]:
    """Return the rows some block names, and what each of their ids becomes."""
    named = sorted(
        {int(value) for column in columns for value in blocks[:, column]} - {0}
    )
    mapping = {old: new for new, old in enumerate(named, start=1)}
    if not named:
        return library[:0], mapping
    return library[[old - 1 for old in named]], mapping


def _extension_map(core: Any) -> list[int]:
    """Return the type number the file gave each kind of specification, -1 for absent."""
    mapping = [-1] * 8
    for name, number in _declared_types(core).items():
        mapping[_EXTENSION_KINDS[name]] = number
    return mapping


def _chain_rows(core: Any) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the chain rows blocks point at, and the row each block starts at."""
    heads = np.asarray(core.block_events(), dtype=np.int64)[:, 5]
    rows: list[list[float]] = []
    start: dict[int, int] = {}
    for head in sorted({int(value) for value in heads} - {0}):
        links = np.asarray(core.extension_chain(head), dtype=np.int64).reshape(2, -1)
        start[head] = len(rows) + 1
        for position in range(links.shape[1]):
            last = position + 1 == links.shape[1]
            rows.append(
                [
                    float(links[0, position]),
                    float(links[1, position]),
                    0.0 if last else float(len(rows) + 2),
                ]
            )
    table = np.asarray(rows, dtype=np.float64).reshape(-1, 3)
    return table, np.array(
        [start.get(int(head), 0) for head in heads], dtype=np.float64
    )


def _numbers(declared: dict[str, Any], name: str, count: int) -> list[float]:
    value = declared.get(name)
    if value is None or isinstance(value, str):
        return [0.0] * count
    values = list(value) if isinstance(value, (list, tuple, np.ndarray)) else [value]
    return [float(values[i]) if i < len(values) else 0.0 for i in range(count)]


def _texts(value: Any) -> list[str]:
    """Return a definition's values as the text a file carries them in."""
    if isinstance(value, str):
        return [value]
    values = list(value) if isinstance(value, (list, tuple, np.ndarray)) else [value]
    return [item if isinstance(item, str) else repr(float(item)) for item in values]
