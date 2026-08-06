"""Moving a PyPulseq-shaped sequence into the C++ :class:`pulseq::Sequence`.

The C++ library owns the file formats, deduplication and the block table; what
it does not own is the vocabulary a design script builds a sequence out of.
This module is the seam: it takes anything shaped like a ``pp.Sequence`` --
event libraries keyed by id, a block table, a definitions dict -- and loads it
into the C++ object one library at a time.

Two shapes are accepted for the block table and the extension chains, because a
composed scan holds them as arrays and an ordinary sequence holds them as
dicts. Arrays are handed straight over; dicts are gathered first. That is the
whole reason the transfer is written in terms of columns: on a large 3D
protocol the block table is millions of rows, and rebuilding it row by row on
the way across would cost more than everything else here put together.
"""

from __future__ import annotations

__all__ = ["to_native"]

import numpy as np

from .._ext import _pulseqpp_wrapper as _cxx

#: Extension section name -> the attribute a PyPulseq sequence keeps it under.
_SPECIFICATIONS = {
    "TRIGGERS": "trigger_library",
    "LABELSET": "label_set_library",
    "LABELINC": "label_inc_library",
    "RF_SHIMS": "rf_shim_library",
    "ROTATIONS": "rotation_library",
    "DELAYS": "soft_delay_library",
}


def _rows(library, width: int) -> np.ndarray:
    """A library's rows as one ``(N, width)`` float64 array, in id order.

    ``rows_array`` is what deduplication already produced, so the usual case is
    a reference rather than a copy; anything else is gathered from the dict.

    A library may be *wider* than the file format, and one is: PyPulseq appends
    the system's ADC dead time as a ninth column when it reads an ADC event.
    That column comes from the system rather than from the file and is never
    written back, so the leading ``width`` columns are what crosses. Anything
    narrower is an error worth hearing about.
    """
    cached = getattr(library, "rows_array", None)
    data = getattr(library, "data", {})
    if cached is not None and len(cached) == len(data):
        table = np.ascontiguousarray(cached, dtype=np.float64)
    elif not data:
        return np.zeros((0, width), dtype=np.float64)
    else:
        table = np.asarray(list(data.values()), dtype=np.float64)

    table = table.reshape(len(data), -1) if len(data) else table.reshape(0, width)
    if table.shape[1] < width:
        raise ValueError(f"library rows are {table.shape[1]} wide; {width} are needed")
    return np.ascontiguousarray(table[:, :width])


def _int_rows(library, width: int) -> np.ndarray:
    """A library's rows as one ``(N, width)`` int32 array, in id order."""
    return np.rint(_rows(library, width)).astype(np.int32, copy=False)


def _block_table(seq) -> tuple[np.ndarray, np.ndarray]:
    """``(events, durations)``: the block table as columns.

    A sequence a scan loop composed still holds these as the arrays
    deduplication produced and hands them over untouched. Anything else keeps
    PyPulseq's dicts, and they are gathered here -- once, rather than per block.
    """
    deferred = getattr(seq, "_deferred", None)
    if deferred is not None:
        table, durations = deferred
        table = np.ascontiguousarray(table)
    else:
        events = seq.block_events
        count = len(events)
        table = np.stack(list(events.values())) if count else np.zeros((0, 7), dtype=np.int64)
        durations = np.fromiter(seq.block_durations.values(), dtype=np.float64, count=count)

    # Handed over as it stands, seven columns and all. Column 0 is Pulseq's
    # legacy delay id and is not written, but dropping it here would mean a
    # NumPy pass over sixty megabytes to produce a table the C++ side is about
    # to copy anyway -- so it takes the wide one and skips the column in that
    # copy instead.
    return table, np.ascontiguousarray(durations, dtype=np.float64)


def _gradients(library) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split one gradient library into trapezoids, arbitrary waveforms and slots.

    The file format numbers both kinds in a single sequence, so the split is
    internal and the numbering is not: ``slots`` maps each shared id onto a row
    of one table or the other, positive for a trapezoid and negative for an
    arbitrary waveform, both 1-based. Walking the ids in order is what keeps the
    written file numbered the way this sequence numbered it.
    """
    data = getattr(library, "data", {})
    kinds = getattr(library, "type", {})

    traps: list = []
    arbs: list = []
    slots = np.zeros(len(data), dtype=np.int32)

    for position, key in enumerate(data):
        row = data[key]
        if kinds.get(key) == "g":
            arbs.append(row)
            slots[position] = -len(arbs)
        else:
            traps.append(row)
            slots[position] = len(traps)

    trap_table = (
        np.ascontiguousarray(traps, dtype=np.float64).reshape(-1, 5)
        if traps
        else np.zeros((0, 5), dtype=np.float64)
    )
    arb_table = (
        np.ascontiguousarray(arbs, dtype=np.float64).reshape(-1, 6)
        if arbs
        else np.zeros((0, 6), dtype=np.float64)
    )
    return trap_table, arb_table, slots


def _shapes(library) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The shape library as ``(uncompressed lengths, row starts, samples)``.

    Shapes are the one library whose rows genuinely differ in length, so they
    cross as a flat sample array plus the offsets that cut it into rows -- which
    is how both sides store them anyway.
    """
    data = getattr(library, "data", {})
    lengths = np.empty(len(data), dtype=np.int32)
    starts = np.zeros(len(data) + 1, dtype=np.int32)
    pieces: list = []

    for position, key in enumerate(data):
        shape = data[key]
        samples = np.ascontiguousarray(shape[1:], dtype=np.float64)
        lengths[position] = round(float(shape[0]))
        starts[position + 1] = starts[position] + samples.size
        pieces.append(samples)

    samples = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float64)
    return lengths, starts, samples


def _ragged(library) -> tuple[np.ndarray, np.ndarray]:
    """A ragged float library as ``(row starts, values)``. Used for RF shims."""
    data = getattr(library, "data", {})
    starts = np.zeros(len(data) + 1, dtype=np.int32)
    pieces: list = []
    for position, key in enumerate(data):
        values = np.ascontiguousarray(data[key], dtype=np.float64).ravel()
        starts[position + 1] = starts[position] + values.size
        pieces.append(values)
    values = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float64)
    return starts, values


def _set_definition(native, key: str, value) -> None:
    """Record one ``[DEFINITIONS]`` entry, classified before it crosses.

    Three kinds have to be told apart, because the text writer formats each
    differently and the binary format tags them: text, whole numbers and reals.
    The classification happens here rather than in the binding -- it has to know
    about ``str``, Python ints, NumPy scalars, NumPy arrays and the empty case,
    and that is a lot of type sniffing to do in C++ for no benefit.

    An empty value is tagged as whole numbers. That is not a judgement -- there
    are no values to tag -- but it is what the reference writer emits, and the
    byte comparison against it is worth more than the tidier answer.
    """
    if isinstance(value, str):
        native.set_definition_text(key, value)
        return

    values = np.atleast_1d(np.asarray(value))
    if values.dtype.kind in "SU" or values.dtype == object:
        native.set_definition_text(key, " ".join(str(item) for item in values.ravel()))
        return

    # `int` here means the *Python* type, which is what says the value was
    # meant as a count rather than a measurement; NumPy floats are reals even
    # when they hold a whole number.
    items = value if isinstance(value, (list, tuple)) else values.ravel().tolist()
    integers = all(isinstance(item, (int, np.integer)) and not isinstance(item, bool) for item in items)
    native.set_definition_numbers(key, [float(item) for item in values.ravel()], integers)


def _chains(seq) -> np.ndarray:
    """Extension chains as an ``(N, 3)`` int32 array of ``(type, ref, next)``."""
    deferred = getattr(seq, "_deferred_chains", None)
    if deferred is not None:
        return np.ascontiguousarray(deferred).astype(np.int32, copy=False)

    data = getattr(seq.extensions_library, "data", {})
    if not data:
        return np.zeros((0, 3), dtype=np.int32)
    return np.rint(np.asarray(list(data.values()), dtype=np.float64)).astype(np.int32)


def to_native(seq, *, rotation_library=None, rf_shim_library=None, label_names=None):
    """Load ``seq`` into a C++ :class:`pulseq::Sequence` and return it.

    Parameters
    ----------
    seq
        Anything shaped like :class:`pypulseq.Sequence`: ``definitions``, the
        event libraries, and either a block-table dict or the pair of arrays a
        composed scan defers.
    rotation_library, rf_shim_library : optional
        The two Pulseq 1.5.1 libraries upstream has nowhere to keep. Passed
        separately when the sequence holds them beside itself rather than on
        itself.
    label_names : dict[int, str], optional
        What this sequence's label ids are called. Only names cross into the
        C++ sequence, which numbers them itself -- see :func:`_label_name`.

    Returns
    -------
    The C++ sequence, ready to write, deduplicate or index.
    """
    native = _cxx.Sequence()

    # A composed scan arrives wrapped: the wrapper holds the block table as the
    # arrays deduplication produced, and *delegates every other attribute* to a
    # PyPulseq sequence underneath -- building the dictionaries as it goes,
    # because it cannot know that the caller only wanted the RF library.  So
    # the libraries are read from that inner sequence directly.  Reading them
    # through the wrapper would cost two seconds and two gigabytes on a scan
    # this size, to produce dictionaries nothing here looks at.
    source = getattr(seq, "_seq", seq) if getattr(seq, "_deferred", None) is not None else seq

    # The label registry, resolved once.  It is the same answer for every row.
    registry = label_names
    if registry is None:
        registry = getattr(source, "_label_registry_inv", None)

    native.set_version(
        int(source.version_major), int(source.version_minor), int(source.version_revision)
    )
    native.set_rasters(
        float(source.rf_raster_time),
        float(source.grad_raster_time),
        float(source.adc_raster_time),
        float(source.block_duration_raster),
    )

    for key, value in source.definitions.items():
        _set_definition(native, key, value)

    # RF, with the use tag that rides alongside each row rather than in it.
    rf = _rows(source.rf_library, 10)
    uses = getattr(source.rf_library, "type", {})
    native.set_rf(rf, "".join(str(uses.get(key, "u"))[:1] or "u" for key in source.rf_library.data))

    traps, arbs, slots = _gradients(source.grad_library)
    native.set_gradients(traps, arbs, slots)

    native.set_adc(_rows(source.adc_library, 8))
    native.set_shapes(*_shapes(source.shape_library))
    native.set_extensions(_chains(seq))

    # Each extension section keeps the numeric id the sequence gave it, because
    # the chains refer to it by that number.
    numeric_of = dict(
        zip(
            getattr(source, "extension_string_idx", []),
            getattr(source, "extension_numeric_idx", []),
            strict=False,
        )
    )

    libraries = {
        "ROTATIONS": rotation_library
        if rotation_library is not None
        else getattr(source, "rotation_library", None),
        "RF_SHIMS": rf_shim_library
        if rf_shim_library is not None
        else getattr(source, "rf_shim_library", None),
    }

    for name, attribute in _SPECIFICATIONS.items():
        library = libraries.get(name) or getattr(source, attribute, None)
        if library is None or not getattr(library, "data", None):
            continue
        if name in numeric_of:
            native.set_extension_type_id(name, int(numeric_of[name]))

        if name == "TRIGGERS":
            native.set_triggers(_rows(library, 4))
        elif name == "ROTATIONS":
            native.set_rotations(_rows(library, 4))
        elif name == "RF_SHIMS":
            native.set_rf_shims(*_ragged(library))
        elif name == "DELAYS":
            rows = list(library.data.values())
            native.set_soft_delays(
                [int(row[0]) for row in rows],
                [float(row[1]) for row in rows],
                [float(row[2]) for row in rows],
                [str(row[3]) for row in rows],
            )
        else:
            # A label row is a value and a label id, and the id is only this
            # sequence's index into its own names -- so the *name* is what
            # crosses, and the C++ side assigns its own number for it.
            table = _int_rows(library, 2)
            renamed = table.copy()
            for position in range(len(library.data)):
                renamed[position, 1] = native.label_id(
                    _label_name(int(table[position, 1]), registry)
                )
            if name == "LABELSET":
                native.set_label_set(renamed)
            else:
                native.set_label_inc(renamed)

    native.set_blocks(*_block_table(seq))
    return native


def _label_name(label_id: int, registry=None) -> str:
    """The name ``label_id`` goes by, per ``registry``.

    A sequence built through :mod:`pulserver.pypulseq` carries its own registry
    (custom labels are the point of it); anything else is using Pulseq's
    built-in list, where the id is a position in it.  The registry is resolved
    once by the caller rather than looked up per row: on a wrapper that
    delegates, an attribute that is not there is not free.
    """
    if registry is not None and label_id in registry:
        return registry[label_id]

    from .._ext._pulseqpp_wrapper import Sequence as _NativeSequence

    builtin = _NativeSequence.builtin_labels()
    if 1 <= label_id <= len(builtin):
        return builtin[label_id - 1]
    return f"CUSTOM_{label_id}"
