"""Moving a sequence between the C++ :class:`pulseq::Sequence` and PyPulseq.

The C++ library owns the file formats, deduplication and the block table; what
it does not own is the vocabulary a design script builds a sequence out of, nor
the analyses PyPulseq already implements. This module is the seam in both
directions.

:func:`to_native` takes anything shaped like a ``pp.Sequence`` -- event
libraries keyed by id, a block table, a definitions dict -- and loads it into
the C++ object one library at a time. Two shapes are accepted for the block
table and the extension chains, because a composed scan holds them as arrays
and an ordinary sequence holds them as dicts. Arrays are handed straight over;
dicts are gathered first. That is the whole reason the transfer is written in
terms of columns: on a large 3D protocol the block table is millions of rows,
and rebuilding it row by row on the way across would cost more than everything
else here put together.

:func:`to_upstream` goes the other way, over a range of blocks, and is what the
analysis methods are built on. It is a bulk copy rather than a replay because
the two sides store the same numbers: a Pulseq 1.5 library row *is* what
PyPulseq keeps in ``rf_library.data[id]``, so the rows cross as they stand and
their ids cross with them.
"""

from __future__ import annotations

__all__ = ["RestoringSequence", "to_native", "to_upstream"]

import warnings
from collections.abc import Mapping

import numpy as np
import pypulseq as pp
from pypulseq import eps
from pypulseq.utils.cumsum import cumsum

from .._ext import _pulseqpp_wrapper as _cxx
from ._shapes import restore_additional_shape_samples

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


# %% the other direction: a range of blocks as a PyPulseq sequence


#: The extension sections PyPulseq's ``get_block`` can decode. A node of any
#: other kind makes it raise, and two of ours are of another kind: ``ROTATIONS``
#: and ``RF_SHIMS`` postdate what upstream reads. Chains are rebuilt without
#: them, and what they describe is resolved into the events instead.
_UPSTREAM_SECTIONS = ("TRIGGERS", "LABELSET", "LABELINC", "DELAYS")


def to_upstream(seq, *, first: int = 1, last: int | None = None, numbers=None, lead_in=None):
    """Blocks ``first..last`` of ``seq`` as a real :class:`pypulseq.Sequence`.

    Parameters
    ----------
    seq : pulserver.pypulseq.Sequence
        The sequence to read. Its rotations and RF shims are *not* resolved
        here -- a window carrying either should be materialised first.
    first, last : int
        The block range, 1-based inclusive. ``last`` defaults to the last block.
    numbers : sequence of int, optional
        What to call the blocks in the sequence that comes back. Defaults to
        their own indices, which is what makes a window report the block
        numbers the caller would recognise. A materialised window passes the
        numbers of the blocks it stands for.
    lead_in : float, optional
        Seconds of silence to put in front, as a single delay block numbered
        ``0``. Defaults to the time elapsed before ``first``, which is what
        places the window at its true time on a plot's axis.

    Returns
    -------
    pypulseq.Sequence
        Holding the same rows, under the same library ids.

    Notes
    -----
    The libraries cross as they stand: a Pulseq 1.5 row is what PyPulseq keeps
    in ``data[id]``, and its libraries are id-keyed dicts, so nothing has to be
    renumbered and nothing is decoded on the way. Only the rows the window
    actually names are copied.
    """
    native = seq._native
    total = native.num_blocks()
    first = max(int(first), 1)
    last = total if last is None else min(int(last), total)

    events = np.asarray(native.block_events())
    durations = np.asarray(native.block_durations())
    window = events[first - 1 : last]
    spans = durations[first - 1 : last]

    if numbers is None:
        numbers = np.arange(first, last + 1)
    if lead_in is None:
        lead_in = float(durations[: first - 1].sum())

    upstream = RestoringSequence(system=seq.system)
    # The rasters the blocks were laid out on, which after a read are the
    # file's rather than the system's.
    upstream.rf_raster_time = native.rf_raster_time
    upstream.grad_raster_time = native.grad_raster_time
    upstream.adc_raster_time = native.adc_raster_time
    upstream.block_duration_raster = native.block_duration_raster
    upstream.definitions.update(seq.definitions)

    shapes: set[int] = set()
    _take_rf(native, upstream, _referenced(window[:, 0]), shapes)
    _take_gradients(native, upstream, _referenced(window[:, 1:4]), shapes)
    _take_adc(native, upstream, _referenced(window[:, 4]), shapes)
    _take_shapes(native, upstream, shapes)
    heads = _take_extensions(native, upstream, _referenced(window[:, 5]))

    # PyPulseq's block row is seven wide: its leading column is Pulseq's legacy
    # delay id, which a 1.4-or-later sequence leaves at zero.
    table = np.zeros((len(window), 7), dtype=np.int64)
    table[:, 1:] = window
    table[:, 6] = _remapped(window[:, 5], heads)

    numbers = np.asarray(numbers, dtype=np.int64)
    if lead_in > 0:
        # A block numbered zero, holding the time before the window as a bare
        # delay. Upstream accumulates its time axis out of block durations, so
        # this is what puts a window at the time it is really played at.
        numbers = np.concatenate(([0], numbers))
        table = np.concatenate((np.zeros((1, 7), dtype=table.dtype), table))
        spans = np.concatenate(([float(lead_in)], spans))

    upstream.block_events = _ByBlock(numbers, table)
    upstream.block_durations = _ByBlock(numbers, np.ascontiguousarray(spans, dtype=np.float64))
    upstream.next_free_block_ID = (int(numbers[-1]) + 1) if numbers.size else 1
    return upstream


class _ByBlock(Mapping):
    """A block-numbered view on one array, with no dictionary behind it.

    PyPulseq keeps the block table and the block durations as dicts of a
    million entries each, holding a million one-row arrays and a million
    boxed floats. Building those from the two arrays the C++ sequence already
    has costs about a microsecond a block and half a gigabyte on a large 3D
    protocol -- to be read once, in order, by whichever analysis asked.

    So the arrays stay as they are and this maps block numbers onto rows of
    them. Everything upstream does with these -- index one, iterate them,
    ``len``, ``in``, ``.values()``, ``sorted(.items())`` -- is what a
    :class:`~collections.abc.Mapping` provides.

    Read-only, deliberately: it belongs to a window built for analysis, and an
    assignment to it would be a caller thinking they were editing a sequence.
    """

    __slots__ = ("_at", "_base", "_numbers", "_values")

    def __init__(self, numbers: np.ndarray, values: np.ndarray) -> None:
        self._numbers = numbers
        self._values = values
        # Block numbers run consecutively unless a lead-in delay was put in
        # front, and consecutive means a subtraction rather than a lookup --
        # which is the whole point on the sequence-sized case, the one with no
        # lead-in. Anything else builds the index it needs, once.
        consecutive = numbers.size > 0 and int(numbers[-1]) - int(numbers[0]) + 1 == numbers.size
        self._base = int(numbers[0]) if consecutive else 0
        self._at = None if consecutive else {int(n): i for i, n in enumerate(numbers.tolist())}

    def _position(self, key: object) -> int:
        try:
            number = int(key)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            raise KeyError(key) from None
        if self._at is not None:
            position = self._at.get(number, -1)
        else:
            position = number - self._base
        if not 0 <= position < self._numbers.size:
            raise KeyError(key)
        return position

    def __getitem__(self, key: object):
        return self._values[self._position(key)]

    def __iter__(self):
        return iter(self._numbers.tolist())

    def __len__(self) -> int:
        return int(self._numbers.size)

    def __contains__(self, key: object) -> bool:
        try:
            self._position(key)
        except KeyError:
            return False
        return True


def _referenced(column) -> np.ndarray:
    """The distinct non-zero ids a block-table column names."""
    ids = np.unique(np.asarray(column))
    return ids[ids > 0]


def _remapped(heads: np.ndarray, rebuilt: dict[int, int]) -> np.ndarray:
    """``heads`` with each chain head replaced by the one rebuilt for it."""
    if heads.size == 0 or not heads.any():
        return np.zeros(heads.size, dtype=np.int64)
    lookup = np.zeros(int(heads.max()) + 1, dtype=np.int64)
    for source, target in rebuilt.items():
        lookup[source] = target
    return lookup[heads]


def _finish(library) -> None:
    """Point a filled library at the id after its last."""
    library.next_free_ID = (max(library.data) + 1) if library.data else 1


def _take_rf(native, upstream, ids: np.ndarray, shapes: set[int]) -> None:
    """Copy the RF rows ``ids`` names, noting the shapes they refer to."""
    library = upstream.rf_library
    for rf_id in ids.tolist():
        row = native.rf_row(rf_id)
        library.data[rf_id] = row
        library.type[rf_id] = native.rf_use(rf_id)
        shapes.update(int(row[column]) for column in (1, 2, 3) if row[column] > 0)
    _finish(library)


def _take_gradients(native, upstream, ids: np.ndarray, shapes: set[int]) -> None:
    """Copy the gradient rows ``ids`` names, trapezoid or arbitrary.

    The two kinds live in separate tables here and in one library upstream,
    told apart by the ``'t'``/``'g'`` tag beside each row.
    """
    library = upstream.grad_library
    for grad_id in ids.tolist():
        row = native.grad_row(grad_id)
        trapezoid = native.grad_kind(grad_id) == "trap"
        library.data[grad_id] = row
        library.type[grad_id] = "t" if trapezoid else "g"
        if not trapezoid:
            shapes.update(int(row[column]) for column in (3, 4) if row[column] > 0)
    _finish(library)


def _take_adc(native, upstream, ids: np.ndarray, shapes: set[int]) -> None:
    """Copy the ADC rows ``ids`` names, noting any phase-modulation shape."""
    library = upstream.adc_library
    for adc_id in ids.tolist():
        row = native.adc_row(adc_id)
        library.data[adc_id] = row
        if row[7] > 0:
            shapes.add(int(row[7]))
    _finish(library)


def _take_shapes(native, upstream, ids: set[int]) -> None:
    """Copy the shapes ``ids`` names, in the form upstream decompresses.

    A row is its uncompressed length followed by its stored samples --
    ``decompress_shape`` short-circuits when the two lengths agree, so a shape
    that was never compressed crosses unchanged.
    """
    library = upstream.shape_library
    for shape_id in sorted(ids):
        num_samples, samples = native.shape_row(shape_id)
        library.data[shape_id] = np.concatenate(([float(num_samples)], samples))
    _finish(library)


def _take_extensions(native, upstream, heads: np.ndarray) -> dict[int, int]:
    """Rebuild the chains ``heads`` starts, dropping what upstream cannot read.

    Returns the map from each original head onto the one rebuilt for it, which
    is what the block table is rewritten through. Chains are shared -- after
    deduplication a whole scan may run on a handful of them -- so each is
    rebuilt once however many blocks name it.
    """
    rebuilt: dict[int, int] = {}
    if heads.size == 0:
        return rebuilt

    sections: dict[str, int] = {}
    for head in heads.tolist():
        kept: list[tuple[str, int]] = []
        node = head
        while node:
            type_id, reference, node = native.extension_row(node)
            name = native.extension_type_name(type_id)
            if name not in _UPSTREAM_SECTIONS:
                continue
            if not _take_extension_row(native, upstream, name, reference):
                continue
            if name not in sections:
                sections[name] = len(sections) + 1
                upstream.set_extension_string_ID(name, sections[name])
            kept.append((name, reference))

        # Written back to front, because each node has to know the id of the
        # one after it.
        following = 0
        for name, reference in reversed(kept):
            node_id = len(upstream.extensions_library.data) + 1
            upstream.extensions_library.data[node_id] = np.array(
                [sections[name], reference, following], dtype=float
            )
            following = node_id
        rebuilt[head] = following

    _finish(upstream.extensions_library)
    for library in (
        upstream.trigger_library,
        upstream.label_set_library,
        upstream.label_inc_library,
        upstream.soft_delay_library,
    ):
        _finish(library)
    return rebuilt


def _take_extension_row(native, upstream, name: str, reference: int) -> bool:
    """Copy one extension node's payload; say whether upstream can hold it.

    The one it cannot is a label this project defined and Pulseq did not:
    upstream resolves a label id by indexing its own fixed list of names, so a
    custom id would come back as some unrelated label or as an ``IndexError``.
    Dropping the node is the honest answer -- the label is a console
    instruction, and nothing the window feeds reads it.
    """
    if name == "TRIGGERS":
        upstream.trigger_library.data[reference] = native.trigger_row(reference)
        return True
    if name == "DELAYS":
        number, offset, factor, hint = native.soft_delay_row(reference)
        upstream.soft_delay_library.data[reference] = [number, offset, factor, hint]
        return True

    read = native.label_set_row if name == "LABELSET" else native.label_inc_row
    value, label_id = read(reference)
    if native.is_custom_label(label_id):
        return False
    library = upstream.label_set_library if name == "LABELSET" else upstream.label_inc_library
    library.data[reference] = np.array([value, label_id], dtype=float)
    return True


class RestoringSequence(pp.Sequence):
    """A PyPulseq sequence whose ``waveforms()`` restores raster-edge samples.

    Upstream's :meth:`pypulseq.Sequence.waveforms` carries a
    ``TODO: Implement restoreAdditionalShapeSamples`` where MATLAB reconstructs
    the samples on the gradient raster *edges* of an arbitrary gradient stored
    on the centres raster. Without that reconstruction a trapezoid that was
    converted to a shape comes back with its corners rounded over one raster
    interval, and the peak slew rate read off it is lower than the one the
    sequence asks for.

    Overriding ``waveforms()`` and nothing else is deliberate, and follows
    :class:`~._safety.TRSequence`: every upstream analysis path reaches the
    gradients through ``waveforms()`` -> ``get_gradients()`` -> ``PPoly``, so
    ``calculate_pns`` and ``calculate_gradient_spectrum`` inherit the
    reconstruction without knowing this class exists.

    On a shape the reconstruction does not describe -- a spiral, most often --
    :func:`~._shapes.restore_additional_shape_samples` falls back to the
    waveform as stored, which is what upstream returns unconditionally. So this
    agrees with upstream everywhere except on the shapes upstream approximates.
    """

    def waveforms(self, append_RF: bool = False, time_range=None) -> tuple[np.ndarray, ...]:
        """Upstream's signature and upstream's return, on MATLAB's assembler."""
        pieces: list[list[np.ndarray]] = [[] for _ in range(4 if append_RF else 3)]

        # A time_range starts the clock at its first block's start, not at zero,
        # so the times returned stay absolute.
        elapsed = self._elapsed_before(time_range)
        for number in self._blocks_in(time_range):
            block = self.get_block(number)
            for axis, channel in enumerate(("gx", "gy", "gz")):
                gradient = getattr(block, channel)
                if gradient is not None:
                    piece = self._gradient_piece(gradient, elapsed, number)
                    if piece is not None:
                        pieces[axis].append(piece)

            if append_RF and block.rf is not None:
                pieces[-1].append(self._rf_piece(block.rf, elapsed))

            elapsed += self.block_durations[number]

        return [self._join(piece) for piece in pieces]

    # -- internals -------------------------------------------------------

    def _blocks_in(self, time_range) -> list:
        """The block numbers overlapping ``time_range``, upstream's way.

        Reproduced rather than called because upstream inlines this identically
        into ``waveforms``, ``rf_times`` and ``adc_times``, and the three have
        to agree about which blocks they are looking at or their outputs do not
        line up with each other.
        """
        if time_range is None:
            return list(self.block_events)
        if len(time_range) != 2:
            raise ValueError('Time range must be list of two elements')
        if time_range[0] > time_range[1]:
            raise ValueError('End time of time_range must be after begin time')

        spans = np.array(list(self.block_durations.values()))
        ends = np.cumsum(spans)
        first = np.searchsorted(ends, time_range[0])
        last = np.searchsorted(ends - spans, time_range[1], side='right')
        return list(self.block_durations.keys())[first:last]

    def _elapsed_before(self, time_range) -> float:
        """The time in front of the first block ``_blocks_in`` returns."""
        if time_range is None:
            return 0.0
        spans = np.array(list(self.block_durations.values()))
        ends = np.cumsum(spans)
        first = np.searchsorted(ends, time_range[0])
        if first >= spans.size:
            return 0.0
        return float(ends[first] - spans[first])

    def _gradient_piece(self, gradient, elapsed: float, number: int):
        """One gradient's ``(2, n)`` contribution to its channel."""
        if gradient.type == 'grad':
            on_centres = gradient.tt / self.grad_raster_time + 0.5
            if np.all(np.abs(on_centres - np.arange(1, len(on_centres) + 1)) < eps):
                # Arbitrary gradient on the centres raster: the case upstream
                # approximates and MATLAB reconstructs.
                times, amplitudes = restore_additional_shape_samples(
                    gradient.tt,
                    gradient.waveform,
                    gradient.first,
                    gradient.last,
                    self.grad_raster_time,
                    number,
                )
                return np.array([elapsed + gradient.delay + times, amplitudes])

            if abs(on_centres[0] - 1.0) < eps:
                # An oversampled rasterised gradient: its first sample sits on
                # the half raster, so the stored shape is missing the endpoints.
                # MATLAB adds them back; upstream's else-branch does not.
                return np.array(
                    [
                        elapsed
                        + gradient.delay
                        + np.concatenate(([0.0], gradient.tt, [gradient.shape_dur])),
                        np.concatenate(([gradient.first], gradient.waveform, [gradient.last])),
                    ]
                )

            # An extended trapezoid, already on the raster edges.
            return np.array([elapsed + gradient.delay + gradient.tt, gradient.waveform])

        if abs(gradient.flat_time) > eps:
            return np.vstack(
                (
                    cumsum(
                        elapsed + gradient.delay,
                        gradient.rise_time,
                        gradient.flat_time,
                        gradient.fall_time,
                    ),
                    gradient.amplitude * np.array([0, 1, 1, 0]),
                )
            )

        if abs(gradient.rise_time) > eps and abs(gradient.fall_time) > eps:
            return np.vstack(
                (
                    cumsum(elapsed + gradient.delay, gradient.rise_time, gradient.fall_time),
                    gradient.amplitude * np.array([0, 1, 0]),
                )
            )

        if abs(gradient.amplitude) > eps:
            warnings.warn(
                f"'empty' gradient with non-zero magnitude in block {number}",
                stacklevel=2,
            )
        return None

    def _rf_piece(self, rf, elapsed: float) -> np.ndarray:
        """One RF pulse as complex samples, zero-padded at both ends."""
        gamma_b0 = self.system.gamma * self.system.B0
        frequency = rf.freq_offset + rf.freq_ppm * 1e-6 * gamma_b0
        phase = rf.phase_offset + rf.phase_ppm * 1e-6 * gamma_b0

        piece = np.array(
            [
                elapsed + rf.delay + rf.t,
                rf.signal * np.exp(1j * (phase + 2 * np.pi * frequency * rf.t)),
            ]
        )
        # A pulse that starts or ends away from zero would otherwise be drawn,
        # and integrated, as though it stepped there instantaneously.
        if abs(rf.signal[0]) > 0:
            lead = np.array([[piece[0, 0] - 0.1 * self.system.rf_raster_time], [0]])
            piece = np.hstack((lead, piece))
        if abs(rf.signal[-1]) > 0:
            tail = np.array([[piece[0, -1] + 0.1 * self.system.rf_raster_time], [0]])
            piece = np.hstack((piece, tail))
        return piece

    @staticmethod
    def _join(pieces: list[np.ndarray]) -> np.ndarray:
        """Concatenate one channel's pieces, dropping duplicated join samples."""
        if not pieces:
            return np.zeros((2, 0))
        joined = [pieces[0]] + [
            current if previous[0, -1] + eps < current[0, 0] else current[:, 1:]
            for previous, current in zip(pieces[:-1], pieces[1:])
        ]
        return np.concatenate(joined, axis=1)
