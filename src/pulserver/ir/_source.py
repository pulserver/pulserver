"""The event and shape libraries of a sequence, as the IR conversion reads them from pypulseqpp."""

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


#: The layout of ``pypulseqpp.io.SequenceLibraries`` this module reads.
_LAYOUT = 1


@dataclass(frozen=True)
class Shape:
    """One entry of the shape library.

    Attributes
    ----------
    num_uncompressed_samples
        Samples :attr:`samples` stands for.
    samples
        As pypulseqpp stores the shape: run-length encoded on its derivative
        when there are fewer values than samples, the samples themselves when
        there are as many.
    """

    num_uncompressed_samples: int
    samples: NDArray[np.float64]


@dataclass(frozen=True)
class SequenceLibraries:
    """The libraries of one sequence, in the layout the IR conversion keys on.

    Event and shape ids are the sequence's own, so a row's position is the id
    a block names and row 0 is id 1. Every row the sequence holds is here,
    played or not. A time grid pypulseqpp states by rule rather than as a
    shape, an arbitrary gradient sampled every half raster, is appended to
    the shapes as a shape of its own.

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
        ``pypulseqpp.calc_rf_bandwidth`` measures them; unused offsets are 0,
        and so is every row no block plays.
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

    Raises
    ------
    ValueError
        If pypulseqpp exports its tables in another layout, or an ADC's phase
        modulation has not one phase per sample.
    """
    return _sequence_libraries(sequence, _tables(sequence))


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

    Each distinct chain is resolved once: blocks sharing a chain head resolve
    to the same thing. A label outside :data:`COUNTER_LABELS` and
    :data:`FLAG_LABELS` is not carried.
    """
    tables = _tables(sequence)
    kinds = _declared_types(tables)
    heads = tables.blocks[:, 5]
    resolved = {0: _Chain()}
    for head in {int(value) for value in heads} - {0}:
        resolved[head] = _resolve(tables, head, kinds)

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


@dataclass(frozen=True)
class SpecificationLibraries:
    """The specification tables of a sequence, indexed by its ids.

    Row 0 is id 1, as in :class:`SequenceLibraries`, and every row the
    sequence holds is here.

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
        :data:`LABEL_IDS` numbers it; -1 for a label it does not number.
    referenced : dict[str, tuple[int, ...]]
        Per table, the ids some block's chain points at.
    """

    rotations: NDArray[np.float64]
    triggers: NDArray[np.float64]
    rf_shims: tuple[NDArray[np.float64], ...]
    soft_delays: NDArray[np.float64]
    labelset: NDArray[np.float64]
    labelinc: NDArray[np.float64]
    referenced: dict[str, tuple[int, ...]]


def specification_libraries(sequence: Any) -> SpecificationLibraries:
    """Read the specification tables of a sequence and the rows its chains name."""
    return _specification_libraries(_tables(sequence))


def conversion_payload(sequence: Any) -> dict[str, Any]:
    """Everything one sequence file contributes to a conversion.

    The libraries, the specification tables and the chain rows that link a
    block to them, in the layout a parsed file holds: times in µs, fields of
    view in cm, rasters in µs. The chain rows are the sequence's own, and
    each block names the head of its chain.
    """
    tables = _tables(sequence)
    libraries = _sequence_libraries(sequence, tables)
    specifications = _specification_libraries(tables)
    blocks = libraries.blocks.copy()
    rf, grad, adc, rf_use, rf_spectra = _compact(blocks, libraries)
    declared = sequence.definitions

    return {
        "version": [
            sequence.version_major,
            sequence.version_minor,
            sequence.version_revision,
        ],
        "rasters": [
            1e6 * sequence.rf_raster_time,
            1e6 * sequence.grad_raster_time,
            1e6 * sequence.adc_raster_time,
            1e6 * sequence.block_duration_raster,
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
        "extensions": np.asarray(tables.extensions, dtype=np.float64).reshape(-1, 3),
        "extension_map": _extension_map(tables),
        "triggers": specifications.triggers,
        "rotations": specifications.rotations,
        "labelset": specifications.labelset,
        "labelinc": specifications.labelinc,
        "soft_delays": specifications.soft_delays,
        "rf_shims": list(specifications.rf_shims),
    }


# %% private module subroutines


def _tables(sequence: Any) -> Any:
    tables = sequence.libraries()
    if tables.layout != _LAYOUT:
        raise ValueError(
            f"pypulseqpp exports its libraries in layout {tables.layout}; the IR "
            f"conversion reads layout {_LAYOUT}"
        )
    return tables


def _micro(seconds: Any) -> Any:
    return np.rint(np.asarray(seconds, dtype=np.float64) * 1e6)


def _sequence_libraries(sequence: Any, tables: Any) -> SequenceLibraries:
    blocks = np.zeros((tables.blocks.shape[0], 7), dtype=np.float64)
    blocks[:, 0] = np.rint(tables.block_durations / sequence.block_duration_raster)
    blocks[:, 1:] = tables.blocks

    shapes = _ShapeTable(tables.shapes)
    rf, rf_use, rf_spectra = _rf_library(sequence, tables, blocks)
    grad = _grad_library(tables, shapes)
    adc = _adc_library(tables)
    return SequenceLibraries(
        blocks, rf, rf_use, rf_spectra, grad, adc, shapes.entries()
    )


class _ShapeTable:
    """pypulseqpp's shapes under its own ids, then the time grids appended after them."""

    def __init__(self, exported: Any) -> None:
        self._entries = [
            Shape(int(shape.num_samples), np.asarray(shape.data, dtype=np.float64))
            for shape in exported
        ]
        self._grids: dict[int, int] = {}

    def half_raster(self, count: int) -> int:
        """Return the id of a time shape of ``count`` samples every half raster."""
        if count not in self._grids:
            self._entries.append(Shape(count, 0.5 * np.arange(1.0, count + 1.0)))
            self._grids[count] = len(self._entries)
        return self._grids[count]

    def entries(self) -> tuple[Shape, ...]:
        return tuple(self._entries)


def _rf_library(
    sequence: Any, tables: Any, blocks: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.int32], NDArray[np.float64]]:
    rows = np.array(tables.rf, dtype=np.float64).reshape(-1, 10)
    rows[:, 4:6] = _micro(rows[:, 4:6])
    uses = np.array([_RF_USE[use] for use in tables.rf_use], dtype=np.int32)
    spectra = np.zeros((rows.shape[0], 3 + MAX_BANDS), dtype=np.float64)
    raster = sequence.rf_raster_time
    measured: dict[tuple[float, float, float], NDArray[np.float64]] = {}
    played = blocks[:, 1].astype(np.int64)
    for identifier in np.unique(played[played > 0]):
        row = rows[identifier - 1]
        # The spectrum's shape depends on the waveform alone: amplitude scales
        # it and a frequency offset moves it, neither of which the bands see.
        key = (row[1], row[2], row[3])
        if key not in measured:
            block = int(np.argmax(played == identifier)) + 1
            measured[key] = _spectrum_row(sequence.get_block(block).rf, raster)
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


def _grad_library(tables: Any, shapes: _ShapeTable) -> NDArray[np.float64]:
    count = tables.trapezoid_ids.size + tables.arbitrary_gradient_ids.size
    rows = np.zeros((count, 7), dtype=np.float64)
    trapezoids = np.zeros((tables.trapezoid_ids.size, 7), dtype=np.float64)
    trapezoids[:, 0] = _TRAPEZOID
    trapezoids[:, 1] = tables.trapezoids[:, 0]
    trapezoids[:, 2:6] = _micro(tables.trapezoids[:, 1:5])
    rows[tables.trapezoid_ids - 1] = trapezoids
    for identifier, row in zip(
        tables.arbitrary_gradient_ids, tables.arbitrary_gradients, strict=True
    ):
        time_shape = int(row[4])
        if time_shape == -1:
            time_shape = shapes.half_raster(tables.shapes[int(row[3]) - 1].num_samples)
        rows[identifier - 1] = [
            _ARBITRARY,
            row[0],
            row[1],
            row[2],
            row[3],
            time_shape,
            _micro(row[5]),
        ]
    return rows


def _adc_library(tables: Any) -> NDArray[np.float64]:
    rows = np.array(tables.adc, dtype=np.float64).reshape(-1, 8)
    rows[:, 1] = np.rint(rows[:, 1] * 1e9)
    rows[:, 2] = _micro(rows[:, 2])
    for identifier, row in enumerate(rows, start=1):
        modulation = int(row[7])
        size = tables.shapes[modulation - 1].num_samples if modulation else 0
        if size and size != row[0]:
            raise ValueError(
                f"ADC {identifier} acquires {row[0]:g} samples but its "
                f"phase modulation has {size}"
            )
    return rows


@dataclass
class _Chain:
    """One extension chain, resolved."""

    labelset: dict[str, int] = field(default_factory=dict)
    labelinc: dict[str, int] = field(default_factory=dict)
    flags: dict[str, int] = field(default_factory=dict)
    points: dict[str, int] = field(default_factory=dict)


def _declared_types(tables: Any) -> dict[str, int]:
    """Return the type number the sequence gives each specification; -1 for none."""
    return {name: tables.extension_types.get(name, -1) for name in _EXTENSION_KINDS}


def _links(extensions: NDArray[np.int32], head: int) -> list[tuple[int, int]]:
    """Return the ``(type, row)`` of every link of the chain from ``head``, in order.

    A chain ends at a next of 0, and at a link it has already visited.
    """
    links: list[tuple[int, int]] = []
    seen: set[int] = set()
    node = head
    while 0 < node <= len(extensions) and node not in seen:
        seen.add(node)
        kind, row, node = (int(value) for value in extensions[node - 1])
        links.append((kind, row))
    return links


def _resolve(tables: Any, head: int, kinds: dict[str, int]) -> _Chain:
    """Resolve one chain: the last row it names of each kind, and its labels in order."""
    chain = _Chain()
    links = _links(tables.extensions, head)
    for name, specification in _SPECIFICATIONS.items():
        referenced = [row for kind, row in links if kind == kinds[specification]]
        if referenced:
            # The chain names a 1-based row; the tables count from 0.
            chain.points[name] = referenced[-1] - 1
    labels = {
        kinds["LABELSET"]: (
            tables.label_set_values,
            tables.label_set_labels,
            chain.labelset,
        ),
        kinds["LABELINC"]: (
            tables.label_inc_values,
            tables.label_inc_labels,
            chain.labelinc,
        ),
    }
    for kind, row in links:
        if kind < 0 or kind not in labels:
            continue
        values, names, target = labels[kind]
        value, label = int(values[row - 1]), names[row - 1]
        if label in FLAG_LABELS:
            chain.flags[label] = value
        elif label in COUNTER_LABELS:
            target[label] = value
    return chain


def _specification_libraries(tables: Any) -> SpecificationLibraries:
    kinds = _declared_types(tables)
    referenced: dict[str, set[int]] = {kind: set() for kind in _EXTENSION_KINDS}
    numbers = {number: kind for kind, number in kinds.items() if number >= 0}
    for head in {int(value) for value in tables.blocks[:, 5]} - {0}:
        for kind, row in _links(tables.extensions, head):
            if kind in numbers:
                referenced[numbers[kind]].add(row)

    triggers = np.array(tables.triggers, dtype=np.float64).reshape(-1, 4)
    triggers[:, 2:4] = _micro(triggers[:, 2:4])
    soft_delays = np.zeros((len(tables.soft_delay_hints), 4), dtype=np.float64)
    soft_delays[:, 0] = tables.soft_delays[:, 0]
    soft_delays[:, 1] = _micro(tables.soft_delays[:, 1])
    soft_delays[:, 2] = tables.soft_delays[:, 2]
    soft_delays[:, 3] = [_HINTS.get(hint, -1) for hint in tables.soft_delay_hints]

    return SpecificationLibraries(
        rotations=np.array(tables.rotations, dtype=np.float64).reshape(-1, 4),
        triggers=triggers,
        rf_shims=tuple(np.array(row, dtype=np.float64) for row in tables.rf_shims),
        soft_delays=soft_delays,
        labelset=_label_rows(tables.label_set_values, tables.label_set_labels),
        labelinc=_label_rows(tables.label_inc_values, tables.label_inc_labels),
        referenced={
            table: tuple(sorted(referenced[kind]))
            for kind, table in (
                ("ROTATIONS", "rotations"),
                ("TRIGGERS", "triggers"),
                ("RF_SHIMS", "rf_shims"),
                ("DELAYS", "soft_delays"),
                ("LABELSET", "labelset"),
                ("LABELINC", "labelinc"),
            )
        },
    )


def _label_rows(values: Any, labels: Any) -> NDArray[np.float64]:
    rows = np.zeros((len(labels), 2), dtype=np.float64)
    rows[:, 0] = values
    rows[:, 1] = [LABEL_IDS.get(label, -1) for label in labels]
    return rows


def _compact(
    blocks: NDArray[np.float64], libraries: SequenceLibraries
) -> tuple[Any, Any, Any, Any, Any]:
    """Drop the library rows no block plays, renumbering the block table in place.

    A row no block plays would deduplicate into a definition of its own that
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


def _extension_map(tables: Any) -> list[int]:
    """Return the type number the sequence gives each kind of specification, -1 for absent."""
    mapping = [-1] * 8
    for name, number in _declared_types(tables).items():
        mapping[_EXTENSION_KINDS[name]] = number
    return mapping


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
