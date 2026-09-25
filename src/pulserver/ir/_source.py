"""The event and shape libraries of a sequence, as the IR conversion reads them from pypulseqpp."""

from __future__ import annotations

__all__ = [
    "SequenceLibraries",
    "Shape",
    "SpecificationLibraries",
    "conversion_payload",
    "sequence_libraries",
    "specification_libraries",
]

from dataclasses import dataclass
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

#: Flags the IR keeps per block, in the order of the payload's block_flags
#: columns.
_BLOCK_FLAGS = ("NOROT", "NOPOS", "PMC", "NAV", "TRID")

#: Labels the IR records per readout, in the order of the converter's label
#: state, which label_column_map indexes.
_READOUT_LABELS = (
    "SLC",
    "PHS",
    "REP",
    "AVG",
    "SEG",
    "SET",
    "ECO",
    "PAR",
    "LIN",
    "ACQ",
    "OFF",
)

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
        and so is every row whose shapes no block plays at a nonzero amplitude.
    rf_flip_deg : NDArray[np.float64]
        ``(R,)``: each row's flip angle in degrees, as
        ``pypulseqpp.Sequence.rf_flip_angles`` gives it.
    rf_channels : NDArray[np.int32]
        ``(R,)``: the transmit channels each row holds, as
        ``pypulseqpp.Sequence.rf_channels`` counts them.
    rf_b1sq_integral : NDArray[np.float64]
        ``(R,)``: the integral of each row's squared envelope scaled to unit
        peak, in s: ``pypulseqpp.calc_rf_power``'s energy over its peak power,
        both summed over the channels; 0 for every row whose shapes no block
        plays at a nonzero amplitude.
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
    rf_flip_deg: NDArray[np.float64]
    rf_channels: NDArray[np.int32]
    rf_b1sq_integral: NDArray[np.float64]
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


def conversion_payload(sequence: Any, system: pp.Opts) -> dict[str, Any]:
    """Everything one sequence file contributes to a conversion.

    The libraries, the specification tables and the chain rows that link a
    block to them, in the layout a parsed file holds: times in µs, fields of
    view in cm, rasters in µs. The chain rows are the sequence's own, and
    each block names the head of its chain.

    RF and ADC frequency and phase offsets are absolute: the ppm offsets are
    resolved at the gamma and B0 of ``system`` by
    ``pypulseqpp.io.SequenceLibraries.absolute_offsets``, and the ppm columns
    are zero. The repetition the conversion segments is the one
    ``pypulseqpp.Sequence.repetition`` finds. Each block's rotation and shim,
    the flags in force at it, the gradient its RF pulse plays under and the
    labels at each readout are pypulseqpp's ``block_rotations``,
    ``block_shims``, ``evaluate_labels``, ``rf_gradients`` and
    ``label_blocks``; the conversion reads the chain rows for triggers alone.

    Raises
    ------
    ValueError
        If pypulseqpp exports its tables in another layout, an ADC's phase
        modulation has not one phase per sample, or the repetition does not
        start at the first block.
    """
    tables = _tables(sequence)
    libraries = _sequence_libraries(sequence, tables)
    _resolve_ppm(libraries, tables, system)
    specifications = _specification_libraries(tables)
    blocks = libraries.blocks.copy()
    rf, grad, adc, rf_use, rf_spectra, rf_flip_deg, rf_channels, rf_b1sq = _compact(
        blocks, libraries
    )
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
            "repetition_size": _repetition_size(sequence),
            "name": str(declared.get("Name", "")),
            "next_sequence": str(declared.get("NextSequence", "")),
        },
        "definitions": {name: _texts(value) for name, value in declared.items()},
        "blocks": blocks,
        "rf": rf,
        "rf_use": rf_use,
        "rf_spectra": rf_spectra,
        "rf_flip_deg": rf_flip_deg,
        "rf_channels": rf_channels,
        "rf_b1sq_integral": rf_b1sq,
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
        **_block_states(sequence, blocks),
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
    rf, rf_use, rf_spectra, rf_b1sq = _rf_library(sequence, tables, blocks)
    grad = _grad_library(tables, shapes)
    adc = _adc_library(tables)
    return SequenceLibraries(
        blocks,
        rf,
        rf_use,
        rf_spectra,
        np.asarray(sequence.rf_flip_angles(), dtype=np.float64),
        np.asarray(sequence.rf_channels(), dtype=np.int32),
        rf_b1sq,
        grad,
        adc,
        shapes.entries(),
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
) -> tuple[
    NDArray[np.float64], NDArray[np.int32], NDArray[np.float64], NDArray[np.float64]
]:
    rows = np.array(tables.rf, dtype=np.float64).reshape(-1, 10)
    rows[:, 4:6] = _micro(rows[:, 4:6])
    uses = np.array([_RF_USE[use] for use in tables.rf_use], dtype=np.int32)
    spectra = np.zeros((rows.shape[0], 3 + MAX_BANDS), dtype=np.float64)
    integrals = np.zeros(rows.shape[0], dtype=np.float64)
    raster = sequence.rf_raster_time
    # The bands and the unit-peak integral depend on the waveform alone: the
    # amplitude scales the spectrum and cancels from the integral, and a
    # frequency offset only moves the spectrum. Both are measured once per set
    # of shapes, on a row that plays something: a pulse of zero amplitude has
    # no spectrum, and its row keeps none.
    measured: dict[tuple[float, float, float], tuple[NDArray[np.float64], float]] = {}
    played = blocks[:, 1].astype(np.int64)
    identifiers = np.unique(played[played > 0])
    for identifier in identifiers:
        row = rows[identifier - 1]
        key = (row[1], row[2], row[3])
        if row[0] != 0.0 and key not in measured:
            block = int(np.argmax(played == identifier)) + 1
            event = sequence.get_block(block).rf
            energy, peak, _ = pp.calc_rf_power(event, dt=raster)
            measured[key] = (
                _spectrum_row(event, raster),
                energy / peak if peak else 0.0,
            )
    for identifier in identifiers:
        row = rows[identifier - 1]
        found = measured.get((row[1], row[2], row[3]))
        if found is not None:
            spectra[identifier - 1] = found[0]
            integrals[identifier - 1] = found[1]
    return rows, uses, spectra, integrals


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


def _block_states(
    sequence: Any, blocks: NDArray[np.float64]
) -> dict[str, NDArray[Any]]:
    """Per block, the rows it plays, the flags in force and the gradient under its RF; per readout, the labels.

    ``block_rotations`` and ``block_shims`` count the ROTATIONS and RF_SHIMS
    rows from 0, -1 for none. ``block_flags`` holds :data:`_BLOCK_FLAGS` and
    ``adc_labels``, one row per acquiring block in block order,
    :data:`_READOUT_LABELS`: the values in force once the block's own labels
    apply, PMC starting at 1 and every other label at 0, with OFF as 0 or 1.
    ``trid_set`` is 1 at a block that sets TRID, which is where a repetition of
    that group starts even when it sets the value already in force.
    ``rf_steady`` is 1 at a block whose RF pulse plays under a gradient that
    holds one value along every channel axis, and ``rf_gradient`` is the
    gradient along x, y and z at the pulse's centre, in Hz/m.
    """
    count = blocks.shape[0]
    start = dict.fromkeys((*_BLOCK_FLAGS, *_READOUT_LABELS), 0)
    start["PMC"] = 1
    found = sequence.evaluate_labels(init=start, evolution="blocks")
    state = {
        name: np.broadcast_to(np.asarray(found[name], dtype=np.int32), (count,))
        for name in start
    }
    labels = np.stack([state[name] for name in _READOUT_LABELS], axis=1)
    labels = labels[blocks[:, 5] > 0]
    labels[:, -1] = labels[:, -1] != 0
    trid_set = np.zeros(count, dtype=np.int32)
    trid_set[np.asarray(sequence.label_blocks("TRID"), dtype=np.int64) - 1] = 1
    under = sequence.rf_gradients()
    pulsed = under.block.astype(np.int64) - 1
    rf_steady = np.zeros(count, dtype=np.int32)
    rf_steady[pulsed] = under.steady.all(axis=1)
    rf_gradient = np.zeros((count, 3), dtype=np.float64)
    rf_gradient[pulsed] = under.gradient
    return {
        "block_rotations": np.asarray(sequence.block_rotations(), dtype=np.int32) - 1,
        "block_shims": np.asarray(sequence.block_shims(), dtype=np.int32) - 1,
        "block_flags": np.stack([state[name] for name in _BLOCK_FLAGS], axis=1),
        "trid_set": trid_set,
        "rf_steady": rf_steady,
        "rf_gradient": rf_gradient,
        "adc_labels": labels,
    }


def _repetition_size(sequence: Any) -> int:
    size, start = sequence.repetition()
    if start != 1:
        raise ValueError(
            f"the sequence repeats from block {start}; the IR segments a "
            "repetition that starts at the first block"
        )
    return int(size)


def _resolve_ppm(libraries: SequenceLibraries, tables: Any, system: pp.Opts) -> None:
    """Fold the ppm offsets of the RF and ADC rows into their absolute offsets, in place."""
    rf_offsets, adc_offsets = tables.absolute_offsets(system)
    libraries.rf[:, 8:10] = rf_offsets
    libraries.rf[:, 6:8] = 0.0
    libraries.adc[:, 5:7] = adc_offsets
    libraries.adc[:, 3:5] = 0.0


def _compact(
    blocks: NDArray[np.float64], libraries: SequenceLibraries
) -> tuple[Any, ...]:
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
    flips = libraries.rf_flip_deg[played_rf]
    channels = libraries.rf_channels[played_rf]
    integrals = libraries.rf_b1sq_integral[played_rf]
    for columns, mapping in (((1,), rf_map), ((2, 3, 4), grad_map), ((5,), adc_map)):
        for column in columns:
            blocks[:, column] = [
                mapping.get(int(value), 0) for value in blocks[:, column]
            ]
    return rf, grad, adc, uses, spectra, flips, channels, integrals


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
