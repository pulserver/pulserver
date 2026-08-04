"""Fast sequence helpers for production bridge execution."""

from __future__ import annotations

__all__ = ["Segment", "Sequence"]

import copy
import contextvars
import itertools
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pypulseq as pp
from pypulseq.block_to_events import block_to_events
from pypulseq.compress_shape import compress_shape
from pypulseq.event_lib import EventLibrary
from pypulseq.supported_labels_rf_use import get_supported_labels

from .._core._module import _GETTERS, _peak
from ._safety import bands_to_resonances, chronaxie_pns, read_asc_bands, read_esp_bands
from ._views import build_views, slice_view, time_bounds

_RF_USE_CHAR_TO_CODE = {
    "u": 0,
    "e": 1,
    "r": 2,
    "i": 3,
    "s": 4,
    "p": 5,
    "o": 6,
}
_RF_USE_CODE_TO_CHAR = {v: k for k, v in _RF_USE_CHAR_TO_CODE.items()}

#: Gradient channel -> its column in a ``block_events`` row.
_CHANNEL_SLOT = {"x": 2, "y": 3, "z": 4}

#: Gradient channel -> its key in a module payload.
_CHANNEL_KEY = {"x": "gx", "y": "gy", "z": "gz"}
#: Stand-in for a payload with no arbitrary gradients, so the waveform check is
#: one dict lookup that misses rather than a branch on every block.
_NO_SHAPES: dict = {}

#: Set only while :meth:`Sequence._unstructured` is constructing one, which is
#: the single case allowed to build a sequence without declaring its structure.
#: A ContextVar rather than a plain flag so a parse cannot leak the permission
#: into anything running alongside it.
_parsing: contextvars.ContextVar[bool] = contextvars.ContextVar("_parsing", default=False)


#: How each event kind yields the values its payload tuple would have carried.
#: The same order and the same arithmetic ``_block_payload`` uses -- it is the
#: same numbers, reached without building the dict that would hold them.
_EVENT_VALUES = {
    "rf": lambda event, peaks: (_peak(event.signal, peaks), *_GETTERS["rf"](event)),
    "trap": lambda event, peaks: (event.amplitude,),  # noqa: ARG005 - one signature for all kinds
    "grad": lambda event, peaks: (_peak(event.waveform, peaks), *_GETTERS["grad"](event)),
    "adc": lambda event, peaks: _GETTERS["adc"](event),  # noqa: ARG005
    "labelset": lambda event, peaks: (event.value,),  # noqa: ARG005
    "labelinc": lambda event, peaks: (event.value,),  # noqa: ARG005
    "trigger": lambda event, peaks: _GETTERS["trigger"](event),  # noqa: ARG005
    "output": lambda event, peaks: _GETTERS["output"](event),  # noqa: ARG005
    "rot3D": lambda event, peaks: tuple(  # noqa: ARG005
        event.rot_quaternion.as_quat(canonical=True, scalar_first=True).tolist()
    ),
    # Magnitude/phase interleaved per channel, matching ``_row_rf_shim``.
    "rf_shim": lambda event, peaks: tuple(  # noqa: ARG005
        np.stack((np.abs(event.shim_vector), np.angle(event.shim_vector)), axis=-1).ravel().tolist()
    ),
}


def _same_waveform(samples, expected) -> bool:
    """Whether two gradient sample arrays describe the same shape.

    Only reached when they are not the same object, which for a module that
    replays its designed waveform is never. A module that rebuilds an equal
    array every shot is wasteful but correct, and this is what tells the two
    apart from a module that re-solved the waveform into something else.
    """
    if samples is None:
        return False
    samples, expected = np.asarray(samples), np.asarray(expected)
    return samples.shape == expected.shape and bool(np.array_equal(samples, expected))

#: Rows buffered across the whole template before IDs are settled. The rounded
#: index is vectorised, so it wants a batch; buffering the whole scan instead
#: would cost more than the sequence. Flushing cannot change the emitted file --
#: batches go out in visit order either way.
_TEMPLATE_FLUSH_ROWS = 1 << 18

#: Libraries the fast path only ever appends to. ``shape_library`` and
#: ``extensions_library`` are absent on purpose: both are looked up by value
#: while building (``find_or_insert`` / ``find``), so both need their keymap.
_APPEND_ONLY_LIBRARIES = (
    "arb_library",
    "trap_library",
    "grad_library",
    "rf_library",
    "adc_library",
    "trigger_library",
    "label_set_library",
    "label_inc_library",
    "rotation_library",
    "rf_shim_library",
)


#: Significant-digit rounding profile per library, one entry per payload column.
#: A positive entry is significant digits, a non-positive one decimal places.
#:
#: These are the tolerances at which two events count as the same event. They
#: are applied twice -- as a range is registered, and again over the whole
#: sequence at write time -- so they live here rather than at either call site:
#: the two passes must agree, or the second would re-split what the first
#: collapsed.
_ROUNDING_PROFILES = {
    "arb_library": (6, -6, -6, -6, -6, -6),
    "trap_library": (6, -6, -6, -6, -6),
    "grad_library": (0,),
    "rf_library": (6, 0, 0, 0, 6, 6, 6, 6, 6, 6),
    "adc_library": (0, -9, -6, 6, 6, 6, 6, 6, 6),
    "trigger_library": (0, 0, 9, 9),
    "label_set_library": (0, 0),
    "label_inc_library": (0, 0),
    "rotation_library": 9,
    "rf_shim_library": 9,
}

#: Libraries a *block row* points at, whose IDs are read by position -- column 1
#: for RF, 2..4 for the gradient axes, 5 for the ADC. Nothing is ordered by
#: those IDs, so write-time deduplication may renumber them freely.
#:
#: The five extension-referenced libraries are deliberately absent, and the
#: template claims their rows one per block per pass for that reason. A block
#: orders its extension chain by reference ID (``_fast_set_block`` sorts on it,
#: mirroring upstream), so strictly increasing references make one pass's chain
#: order every pass's; sharing a row between passes would leave each holding an
#: arbitrary set of small IDs whose sort order varies pass to pass -- which is a
#: different chain, and a different file.
_BLOCK_ROW_LIBRARIES = ("rf_library", "arb_library", "trap_library", "grad_library", "adc_library")


#: RF ``use`` strings pypulseq abbreviates to their initial.
_RF_USES = ("excitation", "refocusing", "inversion", "saturation", "preparation")

#: Event type -> the ``Sequence`` method that builds its library row.
_ROW_BUILDERS = {
    "rf": "_row_rf",
    "trap": "_row_trap",
    "grad": "_row_grad",
    "adc": "_row_adc",
    "labelset": "_row_label",
    "labelinc": "_row_label",
    "output": "_row_control",
    "trigger": "_row_control",
    "rot3D": "_row_rotation",
    "rf_shim": "_row_rf_shim",
}


#: Per event type: the library it registers into, and how the payload's tuple
#: maps onto that library's row -- ``(row index, payload index)`` pairs.
#:
#: Both orders are fixed and have to agree: the row layouts are the ones the
#: ``_row_*`` builders produce, and the payload order is
#: ``pulserver._core._module._DYNAMIC_FIELDS``. Pairs of integers rather than
#: field names on purpose -- the template already knows which event type sits
#: at each slot, so looking a name up per event per shot would re-learn
#: something settled once.
_TEMPLATE_ROW_FIELDS: dict[str, tuple[str, tuple[tuple[int, int], ...]]] = {
    # row: amplitude, mag, phase, time, center, delay, freq_ppm, phase_ppm, freq_off, phase_off
    # payload: amplitude, freq_offset, phase_offset, freq_ppm, phase_ppm
    "rf": ("rf_library", ((0, 0), (6, 3), (7, 4), (8, 1), (9, 2))),
    # row: amplitude, rise, flat, fall, delay        payload: amplitude
    "trap": ("trap_library", ((0, 0),)),
    # row: amplitude, first, last, shape, time, delay
    # payload: amplitude, first, last, delay
    "grad": ("arb_library", ((0, 0), (1, 1), (2, 2), (5, 3))),
    # row: num, dwell, delay, freq_ppm, phase_ppm, freq_off, phase_off, shape, dead
    # payload: freq_offset, phase_offset, freq_ppm, phase_ppm
    "adc": ("adc_library", ((3, 2), (4, 3), (5, 0), (6, 1))),
    "labelset": ("label_set_library", ((0, 0),)),
    "labelinc": ("label_inc_library", ((0, 0),)),
    "rot3D": ("rotation_library", ((0, 0), (1, 1), (2, 2), (3, 3))),
    # row: kind, channel, delay, duration   payload: delay, duration
    # Kind and channel are structural -- which trigger this is -- so only the
    # timing moves. A volume-start trigger plays once per frame and owns a
    # library row, so it has to be able to live *inside* a template: a block
    # outside one may own nothing.
    "trigger": ("trigger_library", ((2, 0), (3, 1))),
    "output": ("trigger_library", ((2, 0), (3, 1))),
    # A transmit shim's whole vector moves, and its width is the channel count,
    # so the mapping is built per event rather than written out here -- see
    # ``_template_slot``. Like a rotation, only the numbers move: *whether* a
    # block carries a shim is structural.
    "rf_shim": ("rf_shim_library", ()),
}

#: Entry type each event registers under, without rendering its row. ``rf`` is
#: the exception -- its type is the ``use`` code, read off the event.
_TEMPLATE_ENTRY_TYPES = {
    "rf": 0, "trap": "t", "grad": "g", "adc": "",
    "labelset": "", "labelinc": "", "rot3D": "",
    "trigger": "", "output": "", "rf_shim": "",
}

#: Event types a template block may hold and still take the fast path. A block
#: holding anything else is marked slow and its payload is refused, loudly --
#: silently falling back would be worse, because a payload cannot be turned
#: back into events (it carries amplitudes, not shapes).
_TEMPLATE_FAST_TYPES = frozenset(_TEMPLATE_ROW_FIELDS) | {"delay", "soft_delay"}


class _ShapeInventory:
    """Every waveform one template slot may play, and the shape IDs they take.

    A module publishes its complete candidate set at construction, so nothing
    here has to be discovered while the scan is filled: a payload names an
    index, and this answers with the ``(shape id, time shape id)`` pair the row
    should point at.

    Entries are compressed **the first time a shot names one**, not all at once
    up front. That is not laziness for its own sake -- ``shape_library``'s
    deduplication compacts in *registration* order, so registering the whole
    inventory at construction would number the shapes differently from a
    block-by-block build and change the emitted file. Registering on first
    reference gives exactly the block-by-block order, and costs one compression
    per distinct waveform for the whole scan rather than one per shot.
    """

    __slots__ = ("by_identity", "ids", "samples", "times")

    def __init__(self, samples, times):
        if len(samples) != len(times):
            raise ValueError(
                f"waveform inventory has {len(samples)} sample arrays and {len(times)} time arrays"
            )
        self.samples = samples
        self.times = times
        #: Registered ``(shape id, time id)`` per index; ``None`` until named.
        self.ids: list = [None] * len(samples)
        #: ``id(samples[k]) -> k``, so a payload carrying the array rather than
        #: the index still resolves without comparing samples.
        self.by_identity = {id(entry): index for index, entry in enumerate(samples)}

    def resolve(self, sequence, index: int):
        """The ``(shape id, time id)`` for candidate ``index``, registering it once."""
        pair = self.ids[index]
        if pair is None:
            _amplitude, shape_ids = sequence._compress_grad_cached(
                self.samples[index], self.times[index]
            )
            pair = self.ids[index] = tuple(shape_ids)
        return pair


@dataclass(eq=False)
class _TemplateSlot:
    """One event of one template block: where its row goes and what moves in it.

    Compared by identity: nearly every field is a numpy array or a view into
    one, so a generated ``__eq__`` would try to broadcast rows of different
    widths against each other and raise rather than answer.

    ``row`` is the whole library row as the template rendered it, so a payload
    that moves nothing still produces a correct row. ``dynamic`` names the
    positions a payload overwrites, resolved once here rather than looked up
    per shot.
    """

    library: str
    column: int | None
    extension: str | None
    data_type: str | int
    #: The event this slot was built from, kept so its row can be rendered on
    #: first use rather than here. Rendering registers the event's *shapes*,
    #: and a shape ID is handed out in visit order -- so rendering the whole
    #: template up front would number the template's shapes ahead of any
    #: preparation block that precedes the first TR, and change the file.
    event: Any
    row: tuple | None
    dynamic: tuple[tuple[int, str], ...]
    #: Payload key this slot reads -- "rf"/"gx"/"gy"/"gz"/"adc", or "" for an
    #: extension, which is matched positionally against the payload's ``ext``.
    key: str
    #: True for the ``grad_library`` entry whose payload is a waveform's ID, and
    #: which therefore cannot be filled until that waveform has one.
    is_reference: bool = False
    #: Template block this slot belongs to. With the TR length that is the
    #: block-table stride, so no row has to record where its ID goes.
    block: int = 0
    #: The waveform slot a gradient reference names.
    source: Any = None
    #: Whether a payload can move anything in this slot's row. A slot nothing
    #: moves is one library row for the whole scan; one that moves is a row per
    #: TR, which write-time deduplication collapses back down.
    varies: bool = False
    #: This slot's row for the first pass, and for every later pass, as ``(k, W)``
    #: *views into the library's own storage*. Writing a payload is therefore a
    #: write into the library -- there is nothing held anywhere else, and nothing
    #: to register afterwards.
    first_rows: Any = None
    later_rows: Any = None
    #: The entry ID of the row for pass 0, and for passes 1.. as an array.
    first_id: int = 0
    later_ids: Any = None
    #: Position of this slot among the block's extensions. A payload lists its
    #: extensions in event order rather than keyed, so this is how the two are
    #: matched -- resolved here, not searched for per block.
    ext_ordinal: int = -1
    #: For an *arbitrary* gradient, the sample array the template registered a
    #: shape for -- the shape this slot's claimed rows already point at.
    #: ``None`` for trapezoids, which have no shape to move. A shot whose
    #: samples differ registers its own shape and repoints its own row's
    #: ``shape_columns``; this is only the fast "nothing moved" answer, and the
    #: TR is still the template either way -- the samples are data, like an
    #: amplitude or a phase.
    waveform: Any = None
    #: This slot's :class:`_ShapeInventory`, when its module published one, or
    #: ``None`` when the samples are fixed for the whole scan. A slot with an
    #: inventory can point its row at any candidate; a slot without one and a
    #: shot that moved the samples is a refusal.
    shapes: Any = None
    #: Row columns holding ``(shape id, time shape id)``. ``_row_grad`` lays the
    #: row out as (amplitude, first, last, shape, time, delay).
    shape_columns: tuple[int, ...] = (3, 4)
    #: ``(event, peak_cache) -> values``, the payload tuple this slot would have
    #: read, taken straight off the event instead. Bound once here so a block
    #: given as events costs no dict; see ``_EVENT_VALUES``.
    event_values: Any = None


@dataclass(eq=False)
class _TemplateBlock:
    """One block position of the TR template; compared by identity, as its slots are."""

    slots: tuple[_TemplateSlot, ...]
    #: Slots carrying an extension, in the order the block's events were seen.
    ext_slots: tuple[int, ...]
    #: Longest event duration in the block; only bare delays can move it.
    static_duration: float
    fast: bool
    reason: str = ""
    #: The block's events as ``tr_struct`` gave them. A module keeps its layout
    #: across states -- that is what makes a re-state cheap -- so a later shot
    #: hands back these *same objects*, which is how ``add_block(*block)``
    #: recognises a template block without the caller having to say so.
    events: tuple = ()


class _AppendOnlyLibrary(EventLibrary):
    """An event library that is only ever appended to, stored as rows not a dict.

    The fast builder's libraries are write-only: nothing reads an entry back
    until deduplication, which wants the whole library as a matrix anyway, and
    writing only ever sees the *deduplicated* result. Keeping them as
    ``{id: payload}`` therefore buys nothing and costs a dict slot per event --
    three million of them for a large 3D protocol -- so entries live in plain
    rows and the ID is the position.

    Entries are kept as ordered runs, which may be Python lists or matrices:
    :meth:`append` grows a list one payload at a time, while :meth:`reserve`
    claims a whole matrix at once, with no Python step per event. IDs are
    positions either way, so a run boundary is invisible to every reader.

    ``data`` and ``type`` still answer as dicts for anything that asks, built on
    demand, so the class stays a drop-in :class:`EventLibrary`.

    Nothing here maintains ``keymap``, the by-value reverse index
    ``EventLibrary.insert`` keeps, and nothing here collapses duplicates.
    Nothing looks an entry up by value one call at a time, so carrying an index
    would cost a tuple and a dict slot per event and never be read; the one
    thing that does look entries up -- write-time deduplication -- compares the
    whole library as a matrix of rounded rows in a single pass, and reaches the
    same answer whether or not anything was collapsed on the way in.
    """

    def __init__(self, numpy_data: bool = False):
        # Deliberately not EventLibrary.__init__: it assigns the very names
        # this class publishes as computed views.
        self.keymap: dict = {}
        self.numpy_data = numpy_data
        #: Ordered runs of entries: a Python list of payload tuples, or a matrix.
        self._runs: list = [[]]
        #: Per-run types: a list per Python run, one value or array per matrix.
        self._type_runs: list = [[]]
        self._tail: list = self._runs[0]
        self._typed = False
        self._count = 0
        self._data_cache: dict | None = None
        self._type_cache: dict | None = None
        self._run_starts_cache: tuple | None = None

    # -- building ------------------------------------------------------

    def append(self, data, data_type: str | int = "") -> int:
        """Append one payload and return its ID.

        This runs once per event of the whole scan -- some three million times
        for a large 3D protocol -- so it stays deliberately thin: a list
        append, a counter, and one branch untyped libraries never take. The
        dict views invalidate themselves by length rather than being cleared
        from here.
        """
        tail = self._tail
        if tail is None:
            tail = self._tail = []
            self._runs.append(tail)
            self._type_runs.append([])
        tail.append(data)
        self._count = count = self._count + 1
        if data_type != "" or self._typed:
            self._push_type(data_type)
        return count

    def _push_type(self, data_type: str | int) -> None:
        """Record one per-entry type, backfilling if this is the first typed entry."""
        tail_types = self._type_runs[-1]
        if not self._typed:
            self._typed = True
            tail_types.extend([""] * (len(self._tail) - 1))
        tail_types.append(data_type)

    def reserve(self, count: int, width: int, data_type=""):
        """Claim ``count`` consecutive entries and return ``(slab, first_id)``.

        The slab is a zero-filled ``(count, width)`` matrix the caller fills in
        place. IDs are positions, so the caller knows every ID it is about to
        use before it has written a single row -- which is what lets a
        gradient reference be resolved against a waveform claimed later.
        """
        slab = np.zeros((int(count), int(width)), dtype=float)
        first_id = self._count + 1
        self._runs.append(slab)
        self._type_runs.append(data_type)
        if isinstance(data_type, np.ndarray) or data_type != "":
            self._typed = True
        # No fresh tail here: append() makes one when it is next called, so a
        # scan that only ever reserves does not leave an empty run behind every
        # slab for each reader to walk past.
        self._tail = None
        self._count += len(slab)
        return slab, first_id

    def truncate(self, count: int) -> None:
        """Give back every entry past ``count``, which must be a whole run tail.

        Rows are claimed before a scan runs and a scan may play less than it
        claimed, so the unused tail has to go: an entry nothing points at is
        still an entry in the emitted file.
        """
        count = int(count)
        if count >= self._count:
            return
        while self._count > count:
            run, types = self._runs[-1], self._type_runs[-1]
            keep = len(run) - (self._count - count)
            if keep > 0:
                self._runs[-1] = run[:keep]
                if isinstance(types, list | np.ndarray) and len(types) == len(run):
                    self._type_runs[-1] = types[:keep]
                self._count = count
            else:
                self._runs.pop()
                self._type_runs.pop()
                self._count -= len(run)
        self._tail = None
        self._data_cache = None
        self._type_cache = None
        self._run_starts_cache = None

    # -- reading -------------------------------------------------------

    def matrix(self) -> np.ndarray:
        """Every payload as one ``(n_entries, width)`` float matrix."""
        if not self._count:
            return np.empty((0, 0))
        parts = [run if isinstance(run, np.ndarray) else np.array(run, dtype=float) for run in self._runs if len(run)]
        return parts[0] if len(parts) == 1 else np.concatenate(parts)

    def type_codes(self) -> np.ndarray | None:
        """Per-entry type as a numeric code, or ``None`` when the library is untyped."""
        if not self._typed:
            return None
        codes: dict = {}

        def code(value):
            try:
                return codes.setdefault(value, len(codes) + 1)
            except TypeError:  # unhashable, e.g. a numpy scalar array
                return codes.setdefault(str(value), len(codes) + 1)

        out = np.empty(self._count, dtype=float)
        cursor = 0
        for run, run_types in zip(self._runs, self._type_runs, strict=True):
            span = len(run)
            if not span:
                continue
            if isinstance(run, np.ndarray):
                if isinstance(run_types, np.ndarray):
                    out[cursor : cursor + span] = [code(value) for value in run_types.tolist()]
                else:
                    out[cursor : cursor + span] = code(run_types)
            else:
                out[cursor : cursor + span] = [code(value) for value in run_types]
            cursor += span
        return out

    def type_mask(self, value: str | int) -> np.ndarray:
        """Boolean mask of the entries whose type is ``value``.

        Deduplication needs to know which gradient entries are arbitrary and
        which are trapezoids -- two questions about a library with millions of
        entries. Asking :attr:`type` would build a dict that size to answer
        them; a run carries one type for a whole reserved slab, so the mask
        can be filled a slice at a time instead.
        """
        out = np.zeros(self._count, dtype=bool)
        if not self._typed:
            return out
        cursor = 0
        for run, run_types in zip(self._runs, self._type_runs, strict=True):
            span = len(run)
            if not span:
                continue
            if isinstance(run_types, np.ndarray):
                out[cursor : cursor + span] = run_types == value
            elif isinstance(run_types, list):
                # A run that predates the library's first typed entry can have
                # fewer types than entries; the entries beyond it are untyped.
                if run_types:
                    out[cursor : cursor + len(run_types)] = np.asarray(run_types, dtype=object) == value
            else:
                # A reserved slab carries one type for every entry in it.
                out[cursor : cursor + span] = run_types == value
            cursor += span
        return out

    def type_at(self, index: int) -> str | int:
        """Type of the entry at 0-based position ``index``.

        Deduplication needs this only for the handful of entries it keeps, so
        it walks the runs rather than materialising the whole ``type`` dict.
        """
        if not self._typed:
            return ""
        # A library holds one run per claim and one per stretch of appends --
        # dozens of them -- and deduplication asks this once per distinct
        # payload, so the run a position falls in is found by bisection rather
        # than by walking from the start every time.
        starts = self._run_starts()
        run_index = int(np.searchsorted(starts, index, side="right")) - 1
        if run_index < 0 or index >= self._count:
            raise IndexError(index)
        run, run_types = self._runs[run_index], self._type_runs[run_index]
        local = index - int(starts[run_index])
        if not isinstance(run, np.ndarray):
            return run_types[local]
        return run_types[local] if isinstance(run_types, np.ndarray) else run_types

    def _run_starts(self) -> np.ndarray:
        """First entry position of each run, cached and invalidated by length."""
        cache = self._run_starts_cache
        if cache is not None and cache[0] == self._count and cache[1] == len(self._runs):
            return cache[2]
        starts = np.zeros(len(self._runs), dtype=np.int64)
        if len(self._runs) > 1:
            np.cumsum([len(run) for run in self._runs[:-1]], out=starts[1:])
        self._run_starts_cache = (self._count, len(self._runs), starts)
        return starts

    def _iter_rows(self):
        for run in self._runs:
            if isinstance(run, np.ndarray):
                yield from (tuple(row) for row in run.tolist())
            else:
                yield from run

    @property
    def data(self) -> dict:
        """Payloads as ``{id: payload}``, materialised on first request."""
        if self._data_cache is None or len(self._data_cache) != self._count:
            self._data_cache = dict(enumerate(self._iter_rows(), start=1))
        return self._data_cache

    @property
    def type(self) -> dict:
        """Per-entry types as ``{id: type}``, matching ``EventLibrary`` semantics."""
        cache = self._type_cache
        if cache is not None and cache[0] == self._count:
            return cache[1]
        out: dict = {}
        if self._typed:
            cursor = 0
            for run, run_types in zip(self._runs, self._type_runs, strict=True):
                span = len(run)
                if not span:
                    continue
                if isinstance(run, np.ndarray) and not isinstance(run_types, np.ndarray):
                    if run_types != "":
                        out.update(dict.fromkeys(range(cursor + 1, cursor + span + 1), run_types))
                else:
                    for offset, value in enumerate(run_types):
                        if value != "":
                            out[cursor + offset + 1] = value
                cursor += span
        self._type_cache = (self._count, out)
        return out

    @property
    def next_free_ID(self) -> int:
        """IDs are positions, so the next one is always one past the count."""
        return self._count + 1

    def insert(self, key_id: int, new_data, data_type: str | int = "") -> int:
        """Append ``new_data``; ``key_id`` must be the ID it is about to get.

        The upstream signature lets a caller choose the key. Here the key *is*
        the position, so the only key that can be honoured is the next free one
        -- and a caller passing anything else expects random-access semantics
        this library does not have.
        """
        key_id = int(key_id)
        if key_id not in (0, self._count + 1):
            raise ValueError(
                f"{type(self).__name__} assigns IDs by position: expected key {self._count + 1}, got {key_id}. "
                "Use append() to add an entry, or pypulseq.EventLibrary if entries must be keyed freely."
            )
        return self.append(new_data if isinstance(new_data, np.ndarray) else tuple(new_data), data_type)

    def __len__(self) -> int:
        return self._count


@dataclass(frozen=True)
class Segment:
    """One virtual segment of a sequence, resolved to its max-energy instance.

    A segment definition is played many times with different gradient
    amplitudes. The instance described here is the one carrying the most
    gradient energy — the one worth looking at, and the one the safety
    backend reasons about.

    Attributes
    ----------
    index : int
        Segment index within the sequence.
    first_block, last_block : int
        Inclusive 1-based Pulseq block indices of this instance.
    duration : float
        Segment duration in seconds.
    pure_delay, is_nav, has_trigger : bool
        Segment classification flags from the C backend.
    from_max_energy_instance : bool
        ``False`` when the scan table was unavailable and the segment
        definition's own first instance was used instead.
    """

    index: int
    first_block: int
    last_block: int
    duration: float
    pure_delay: bool
    is_nav: bool
    has_trigger: bool
    from_max_energy_instance: bool
    _parent: Sequence = None

    @property
    def num_blocks(self) -> int:
        """Number of blocks in the segment."""
        return self.last_block - self.first_block + 1

    @property
    def time_range(self) -> tuple[float, float]:
        """Start and end time of this instance within the parent sequence, in seconds."""
        return time_bounds(self._parent._seqplot, self.first_block, self.last_block)

    def to_sequence(self) -> pp.Sequence:
        """Build a standalone :class:`pypulseq.Sequence` holding just this segment.

        The result shares the parent's :class:`pypulseq.Opts`, so raster times
        and limits are identical.
        """
        return slice_view(self._parent._seqplot, self.first_block, self.last_block)

    def plot(self, **kwargs):
        """Plot this segment, delegating to the parent's plotting view."""
        return self._parent._seqplot.plot(time_range=self.time_range, **kwargs)


class Sequence(pp.Sequence):
    """Sequence container tuned for append-only generation, with extensions.

    A drop-in subclass of :class:`pypulseq.Sequence` for the case a plugin
    actually has: blocks are emitted strictly in order and never revisited. On
    that assumption the per-block deduplication and gradient-continuity checks
    are skipped and each ``add_block`` becomes a direct library insertion —
    which is what keeps generation tractable for sequences with hundreds of
    thousands of blocks. Deduplication still happens, once, at ``write``.

    It also accepts the extension events upstream does not: user-defined
    labels (:func:`make_label`), block rotations (:func:`make_rotation`), and
    pTx shim vectors (:func:`make_rf_shim`).

    Because blocks are never re-checked, this class assumes you append in
    order. Use :class:`pypulseq.Sequence` if you need to modify blocks after
    adding them.

    Parameters
    ----------
    system : pypulseq.Opts, optional
        System limits recorded in the ``.seq`` header.
    use_block_cache : bool, optional
        Kept for upstream compatibility; off by default.

    Examples
    --------
    >>> import pulserver.pypulseq as pp
    >>> seq = pp.Sequence(pp.Opts())
    >>> seq.add_block(pp.make_delay(1e-3), pp.make_label("LIN", "SET", 0))
    >>> len(seq.block_events)
    1

    Append modules rather than events, and write at the end::

        for ky in range(ny):
            excitation(seq)
            readout(seq, pe_idx=ky)
        seq.write(output_path)

    See Also
    --------
    make_label, make_rotation, make_rf_shim : the supported extension events.
    """

    def __init__(
        self,
        system: pp.Opts | None = None,
        loop_size: int | None = None,
        *tr_struct,
        use_block_cache: bool = False,
    ):
        super().__init__(system=system, use_block_cache=use_block_cache)
        # Every library the fast path appends to, in the order write() emits
        # them. They are write-only until deduplication -- see
        # _AppendOnlyLibrary -- so none of them is a dict.
        for name in _APPEND_ONLY_LIBRARIES:
            setattr(self, name, _AppendOnlyLibrary())
        # Bidirectional label registry; extended automatically for custom labels.
        _builtin = get_supported_labels()
        self._label_registry: dict[str, int] = {lbl: i + 1 for i, lbl in enumerate(_builtin)}
        self._label_registry_inv: dict[int, str] = {i + 1: lbl for i, lbl in enumerate(_builtin)}
        # Lazily built analysis state; see _views(). Keyed by block count so
        # appending after an inspection transparently invalidates it, without
        # costing anything in the add_block hot loop.
        self._view_cache: dict | None = None
        self._collection_cache: dict | None = None
        self._segment_cache: dict | None = None
        # Sample-array -> registered shape IDs, keyed by object identity. A
        # module replays one template thousands of times with different
        # offsets; the sample arrays are the same objects every time, so
        # compressing and hashing them once instead of once per shot is what
        # keeps generation linear in blocks rather than in samples. Each entry
        # keeps a reference to the arrays it was computed from: that both
        # pins the id() against reuse and makes the identity check exact.
        # Valid only because events never have their arrays written into --
        # see pulserver.design._system.copy_event.
        self._rf_shape_cache: dict[int, tuple] = {}
        self._grad_shape_cache: dict[int, tuple] = {}
        # Block table, held as ordered runs the same way the libraries are:
        # Python rows from a block added on its own, whole matrices claimed by
        # the TR template. The dict views and the matrix are built on demand.
        self._block_runs: list = [[]]
        self._duration_runs: list = [[]]
        self._block_tail: list = self._block_runs[0]
        self._duration_tail: list = self._duration_runs[0]
        self._n_blocks = 0
        self._block_dict: dict | None = None
        self._duration_dict: dict | None = None
        self._block_rows: np.ndarray | None = None
        # get_extension_type_ID() is a string lookup with a fallback that can
        # allocate; the four names used here are fixed, so resolve on demand
        # and remember.
        self._ext_type_ids: dict[str, int] = {}
        # TR template state; see _build_template. All inert without tr_struct,
        # so a sequence built the ordinary way pays one None check per block.
        self._template: tuple | None = None
        self._template_cursor = 0
        self._template_pass = 0
        self._template_gap = False
        self._template_finished = False
        self._template_written = 0
        self._template_run_start = 0
        self._template_rows: np.ndarray | None = None
        self._template_block_durations: np.ndarray | None = None
        #: How to give back rows a short scan claimed but never filled.
        self._template_claims: list = []
        self._template_chain_levels: tuple | None = None
        self._loop_size = int(loop_size) if loop_size is not None else None
        #: Waveform peaks by sample-array identity, for blocks handed over as
        #: events. A module replays the same arrays, so this stays tiny.
        self._peak_cache: dict = {}
        if tr_struct:
            self._build_template(tr_struct)
        elif not _parsing.get():
            # Building a sequence means declaring its repeating structure. Every
            # library row is claimed from that structure up front, which is what
            # makes ``add_block`` a handful of scalar writes -- so there is no
            # such thing as a sequence without one, and the block-at-a-time
            # builder is not a public path any more.
            #
            # The period need not be the TR: it is whatever actually repeats, and
            # a scan with a once-only lead-in or tail is one pass over the whole
            # thing. Reading a ``.seq`` back is the one case with no structure to
            # declare, and it comes in through ``_for_parsing``.
            raise TypeError(
                "Sequence() needs the structure that repeats and how many times it does: "
                "Sequence(system, num_passes, *modules_or_blocks). Every library row is "
                "claimed from it up front. If you are reading a file back, use "
                "pulserver.io.read()."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_block(self, *args: SimpleNamespace | float | dict) -> None:
        """Append a block assuming strictly sequential insertion.

        Takes the block's events, exactly as upstream does::

            readout.set_state(lin_idx=ky)
            for block in readout:
                seq.add_block(*block)

        On a sequence built with a ``tr_struct`` that loop is also the *fast*
        path, and nothing at the call site says so. A module keeps its event
        objects across states — re-stating rewrites numbers into the events the
        layout already owns — so a block whose events are the ones the template
        recorded **is** that template block, and is taken through it. A
        preparation, a recovery delay or a hand-built block is made of different
        objects, so it takes the ordinary path and the template is untouched.

        A single **payload dict** from :meth:`pulserver.SequenceModule.payloads`
        is also accepted, and is the cheapest form: it skips reading the block's
        events at all, which for a module that computes its payloads directly is
        the difference between rewriting a few floats and re-deriving them.
        """
        if len(args) == 1 and isinstance(args[0], dict):
            payload = args[0]
            if self._template is not None:
                self._template_set_block(payload)
                self.next_free_block_ID += 1
                return
            # No TR structure was declared, so the numbers alone are not enough
            # to build the block -- but a payload carries the events it came
            # from, which are. The loop at the call site is the same either way.
            args = getattr(payload, "events", None)
            if args is None:  # pragma: no cover - guarded by Payload.events
                raise TypeError(
                    "add_block() was given a payload dict, but this sequence has no tr_struct "
                    "and the payload carries no events to build the block from."
                )
        elif self._template is not None and (self._template_events(args) or self._template_delay(args)):
            self.next_free_block_ID += 1
            return
        self._template_require_boundary(args)
        self._fast_set_block(*args)
        self.next_free_block_ID += 1

    def _template_events(self, args) -> bool:
        """Take a block given as events against the template, if it is one.

        Recognition is by *identity*, which is exact rather than a guess: the
        template holds the very event objects ``tr_struct`` was built from, and
        a module hands the same ones back every shot. Two blocks that merely
        look alike — an inversion pulse against the excitation it precedes —
        cannot be confused, which a structural match would risk.

        The numbers are then read off those events, so this costs a walk over
        the block that a payload would have skipped. It is still the template
        path: no shapes are re-registered, no row is rebuilt from scratch, and
        the libraries stay the size of the TR rather than the size of the scan.
        """
        template = self._template[self._template_cursor]
        expected = template.events
        if len(args) == len(expected):
            # The whole check, for the shape a module actually hands over: same
            # count, same objects, in order. Nothing is allocated and nothing is
            # inspected -- a miss costs one comparison.
            for event, want in zip(args, expected, strict=True):
                if event is not want:
                    return False
        elif not expected or len(args) < len(expected):
            return False
        else:
            # A bare delay written alongside the events pads the block out.
            index = 0
            for event in args:
                if isinstance(event, int | float):
                    continue
                if index >= len(expected) or event is not expected[index]:
                    return False
                index += 1
            if index != len(expected):
                return False

        self._template_set_block(None, args)
        return True

    def _template_delay(self, args) -> bool:
        """Take a bare delay against the template, if that is what sits there.

        A TE or TR delay is part of the TR — it belongs in ``tr_struct`` — but it
        has no payload to speak of, so writing ``add_block({})`` for it at the
        call site would be noise. A block the template says holds nothing but
        time can be given as the delay event itself: there are no library rows
        either way, so the two forms cannot diverge.
        """
        template = self._template[self._template_cursor]
        if template.slots:
            return False
        duration = _delay_duration(args)
        if duration is None:
            return False  # anything real goes through the ordinary path
        self._template_set_block({"duration": duration} if duration else {})
        return True

    # ------------------------------------------------------------------
    # TR template
    # ------------------------------------------------------------------

    def _build_template(self, tr_struct) -> None:
        """Record one TR's structure: which rows each block owns, and what moves.

        Walked once, with the same dispatch ``_fast_set_block`` uses, so a
        template row is built by the code that already works. Nothing is
        registered into a library here, deliberately: the modules a template is
        built from are left at full amplitude by their factories, so a state no
        shot actually plays would leave rows that are numbered but unreferenced,
        and the emitted file would change.
        """
        # A module may be given whole, and it is the better way: the sequence
        # then knows *which* module owns each block, so a block handed back by
        # ``for block in module`` can be registered from that module's own
        # prepared payload instead of by reading its events back out. Modules
        # are expanded here so everything below sees plain blocks.
        # Each expanded block keeps the inventory its owner published for it, so
        # a slot whose samples move knows the complete set it may play. Given as
        # events rather than as a module, there is no owner to ask and a moving
        # waveform has nowhere to point -- which is refused, loudly, at fill time.
        expanded: list = []
        for entry in tr_struct:
            if hasattr(entry, "payloads") and hasattr(entry, "blocks"):
                owned = entry.blocks
                inventory = entry.waveform_inventory() or ((),) * len(owned)
                if len(inventory) != len(owned):
                    raise ValueError(
                        f"{type(entry).__name__}.waveform_inventory() returned "
                        f"{len(inventory)} entries for {len(owned)} blocks."
                    )
                expanded.extend(zip(owned, inventory, strict=True))
            else:
                expanded.append((entry, ()))
        blocks = []
        for block, inventory in expanded:
            slots: list[_TemplateSlot] = []
            ext_slots: list[int] = []
            duration = 0.0
            fast, reason = True, ""

            # A template entry is a block: either the tuple of events a module
            # publishes, or a single event written inline -- a TE or TR delay is
            # naturally spelled the second way.
            events = block_to_events(*block) if isinstance(block, tuple | list) else block_to_events(block)
            for event in events:
                if isinstance(event, float):
                    duration = max(duration, float(event))
                    continue
                kind = event.type
                if kind in ("delay", "soft_delay"):
                    duration = max(duration, float(getattr(event, "delay", 0.0)))
                    continue
                if kind not in _TEMPLATE_ROW_FIELDS:
                    fast, reason = False, f"{kind} events have no payload mapping"
                    continue

                built = self._template_slot(event, inventory)
                for slot in built:
                    slot.block = len(blocks)
                    if not slot.is_reference:
                        slot.event_values = _EVENT_VALUES[kind]
                    if slot.extension is not None:
                        slot.ext_ordinal = len(ext_slots)
                        ext_slots.append(len(slots))
                    slots.append(slot)
                if len(built) == 2:
                    built[1].source = built[0]
                duration = max(duration, self._event_duration(event))

            blocks.append(
                _TemplateBlock(
                    tuple(slots), tuple(ext_slots), duration, fast, reason,
                    events=tuple(event for event in events if not isinstance(event, float)),
                )
            )

        self._template = tuple(blocks)
        self._allocate_template()

    def _allocate_template(self) -> None:
        """Give the whole scan its library rows and its block table, up front.

        The TR's structure is known, and with ``loop_size`` so is how many times
        it plays -- so every row every block will ever point at can be claimed
        here, and ``add_block`` becomes a write into a row that already exists
        at an ID that is already settled. Nothing is buffered, nothing is
        registered later, and no ID is ever decided while the scan is being
        filled.

        What makes this small rather than merely early is the split a payload
        already draws: a slot no payload can move holds *one* row for the whole
        scan, and only a slot something moves gets a row per TR. A Cartesian GRE
        moves an RF phase and two phase-encode amplitudes; its readout,
        prewinder, spoiler and ADC are one row each, however many million blocks
        play them.

        The row order is the one the per-block path would have produced, which
        is what keeps the file identical: pass 0 claims every slot in visit
        order, and each later pass claims only the slots that move. The rows
        that would have been re-registered per TR are exactly the ones write-time
        deduplication collapses back onto pass 0's, so the first occurrences --
        which is all the renumbering depends on -- come out in the same order.
        """
        length = len(self._template)
        count = self._loop_size
        if count is None:
            raise ValueError(
                "a Sequence given a TR structure also needs its loop size: "
                "Sequence(system, loop_size, *tr_struct)."
            )

        by_library: dict[str, list] = {}
        for template_block in self._template:
            for slot in template_block.slots:
                slot.varies = slot.source.varies if slot.is_reference else bool(slot.dynamic)
                by_library.setdefault(slot.library, []).append(slot)

        # A gradient reference's row *is* its waveform's ID, so the library
        # holding the waveform has to have been given its IDs first. One round
        # of dependency, never two.
        order = [name for name in by_library if name != "grad_library"]
        if "grad_library" in by_library:
            order.append("grad_library")
        for name in order:
            if name in _BLOCK_ROW_LIBRARIES:
                self._claim_collapsed_rows(name, by_library[name], count)
            else:
                self._claim_row_per_block(name, by_library[name], count)

        self._template_rows = np.zeros((count * length, 7), dtype=np.int64)
        self._template_block_durations = np.tile(
            np.array([block.static_duration for block in self._template], dtype=float), count
        )
        for index, template_block in enumerate(self._template):
            for slot in template_block.slots:
                if slot.column is None:
                    continue
                column = self._template_rows[index::length, slot.column]
                column[:] = slot.first_id
                if slot.later_ids is not None:
                    column[1:] = slot.later_ids
        self._chain_template_extensions(count, length)
        #: Blocks of _template_rows filled so far, and where the run that has
        #: not yet joined the block table begins.
        self._template_written = 0
        self._template_run_start = 0

    def _claim_collapsed_rows(self, name: str, slots: list, count: int) -> None:
        """Claim one library's rows: one per slot, plus one per TR for the movers."""
        library = getattr(self, name)
        width = len(slots[0].row)
        movers = [slot for slot in slots if slot.varies]

        self._template_claims.append(("collapsed", name, library._count + 1, slots, movers))
        first_rows, first_id = library.reserve(len(slots), width, _row_types(slots))
        for position, slot in enumerate(slots):
            first_rows[position] = slot.row
            slot.first_rows = first_rows[position : position + 1]
            slot.first_id = first_id + position

        if movers and count > 1:
            stride = len(movers)
            later, later_id = library.reserve(
                (count - 1) * stride, width, _row_types(movers, count - 1)
            )
            for position, slot in enumerate(movers):
                rows = later[position::stride]
                rows[:] = slot.row
                slot.later_rows = rows
                slot.later_ids = later_id + position + stride * np.arange(count - 1)

        for slot in slots:
            if not slot.is_reference:
                continue
            # The reference's whole payload is an ID it now knows, so it is
            # written here in full and no shot ever touches it again.
            slot.first_rows[0, 0] = slot.source.first_id
            if slot.later_rows is not None:
                slot.later_rows[:, 0] = slot.source.later_ids

    def _claim_row_per_block(self, name: str, slots: list, count: int) -> None:
        """Claim an extension-referenced library's rows: one per block, per TR.

        These are never deduplicated -- their IDs are written into the file as
        they stand -- so collapsing a label whose value never moves would
        renumber every extension after it. One row per block it is, exactly as
        the per-block path would have appended them.
        """
        library = getattr(self, name)
        width = len(slots[0].row)
        stride = len(slots)
        self._template_claims.append(("per_block", name, library._count + 1, slots, ()))
        rows, first_id = library.reserve(count * stride, width, _row_types(slots, count))
        for position, slot in enumerate(slots):
            block_rows = rows[position::stride]
            block_rows[:] = slot.row
            slot.first_rows = block_rows[:1]
            slot.later_rows = block_rows[1:]
            slot.first_id = first_id + position
            slot.later_ids = first_id + position + stride * np.arange(1, count)

    def _chain_template_extensions(self, count: int, length: int) -> None:
        """Build every block's extension chain, for every pass, up front.

        A chain node is ``(type, reference, next)`` and every reference here is
        this block's own, so no two blocks can share a node and the IDs simply
        run on. The references are already known, so the whole column is settled
        before a single payload arrives.
        """
        # Chain nodes per pass, cumulated over the template's blocks, so the
        # nodes a partial last pass never used can be counted back off.
        levels = [0]
        for block in self._template:
            levels.append(levels[-1] + len(block.ext_slots))
        chained = [
            (index, block) for index, block in enumerate(self._template) if block.ext_slots
        ]
        if not chained:
            return
        self._template_chain_levels = (levels, self.extensions_library.next_free_ID)
        library = self.extensions_library
        data, keymap = library.data, library.keymap
        node = library.next_free_ID
        plans = []
        for block_index, block in chained:
            slots = [block.slots[position] for position in block.ext_slots]
            types = [self._ext_type_id(slot.extension) for slot in slots]
            # Sorted by reference, as the per-block path sorts them. The order
            # is taken once: a slot's references advance by the same stride
            # every pass, so a pair that swapped would have to cross, which is
            # checked rather than assumed.
            first = sorted(range(len(slots)), key=lambda i: slots[i].first_id)
            last = sorted(range(len(slots)), key=lambda i: _slot_id(slots[i], count - 1))
            if first != last:
                raise NotImplementedError(
                    "the extensions of one template block change their reference order "
                    "between passes, which would renumber the extension chain."
                )
            plans.append((block_index, [(types[i], slots[i]) for i in first]))

        for pass_index in range(count):
            base = pass_index * length
            for block_index, entries in plans:
                previous = 0
                for ext_type, slot in entries:
                    entry = (ext_type, int(_slot_id(slot, pass_index)), previous)
                    data[node] = entry
                    keymap[entry] = node
                    previous = node
                    node += 1
                self._template_rows[base + block_index, 6] = previous
        library.next_free_ID = node

    def _template_slot(self, event, inventory=()) -> list[_TemplateSlot]:
        """The template slots one event owns, with its row already rendered."""
        kind = event.type
        library, dynamic = _TEMPLATE_ROW_FIELDS[kind]
        # Rendered here, which also registers the event's shapes -- and shape IDs
        # are handed out in visit order, so this is the one place they can be
        # numbered as the per-block path would have numbered them. It is also why
        # a block outside the template may not own library rows.
        row, data_type, _ = self._event_row(event)
        if kind in ("labelset", "labelinc"):
            return [_TemplateSlot(library, None, kind.upper(), data_type, event, row, dynamic, "")]
        if kind == "rot3D":
            return [_TemplateSlot(library, None, "ROTATIONS", data_type, event, row, dynamic, "")]
        if kind in ("trigger", "output"):
            return [_TemplateSlot(library, None, "TRIGGERS", data_type, event, row, dynamic, "")]
        if kind == "rf_shim":
            # The whole row is the vector, so every column of it moves. The width
            # is the transmit channel count, fixed for a given pulse.
            dynamic = tuple((column, column) for column in range(len(row)))
            return [_TemplateSlot(library, None, "RF_SHIMS", data_type, event, row, dynamic, "")]
        if kind == "rf":
            return [_TemplateSlot(library, 1, None, data_type, event, row, dynamic, "rf")]
        if kind == "adc":
            return [_TemplateSlot(library, 5, None, data_type, event, row, dynamic, "adc")]

        # A gradient costs two rows: the waveform, then the reference the block
        # column actually points at.
        key = _CHANNEL_KEY[event.channel]
        candidates = inventory.get(key) if kind == "grad" and inventory else None
        waveform = _TemplateSlot(
            library, None, None, data_type, event, row, dynamic, key,
            waveform=getattr(event, "waveform", None) if kind == "grad" else None,
            shapes=_ShapeInventory(*candidates) if candidates else None,
        )
        reference = _TemplateSlot(
            "grad_library", _CHANNEL_SLOT[event.channel], None, data_type, None, (0.0,), (), "",
            is_reference=True,
        )
        return [waveform, reference]

    def _event_duration(self, event) -> float:
        """How long one event claims -- the static part of a block's duration."""
        kind = event.type
        if kind == "rf":
            return event.shape_dur + event.delay + event.ringdown_time
        if kind == "trap":
            return event.delay + event.rise_time + event.flat_time + event.fall_time
        if kind == "grad":
            return event.delay + math.ceil(event.tt[-1] / self.grad_raster_time - 1e-10) * self.grad_raster_time
        if kind == "adc":
            return event.delay + event.num_samples * event.dwell + event.dead_time
        if kind in ("trigger", "output"):
            # A trigger alone in a block is what sets that block's duration, so
            # this is not cosmetic: without it the block is emitted with
            # duration zero. Spelled as ``_row_control`` spells it.
            return event.delay + event.duration
        return 0.0

    def _template_require_boundary(self, args=()) -> None:
        """A block outside the template may own no library row, and must not split a pass.

        Every row the template will ever use was claimed when the sequence was
        built, so a block that registered one of its own now would take an ID
        the template has already handed out and shift every ID after it. A block
        that owns no rows -- a pure delay -- cannot, so that is what may sit
        between TRs, and nothing else.

        It may also end a pass early, which is what a segmented acquisition
        needs when its train length does not divide the view count: the last,
        short segment still gets its recovery delay. Only the *last* pass may be
        short, though -- the rows a half-played pass did not use are trimmed off
        the end, and there is no trimming a hole out of the middle.
        """
        if self._template is None:
            return
        if not _is_pure_delay(args):
            raise RuntimeError(
                "add_block() was given a block of its own on a sequence built with a TR "
                "structure. Every library row is claimed up front, so a block outside the "
                "template may hold nothing but a delay; put this one in tr_struct."
            )
        # This block goes into the table *now*, so the template rows written
        # since the last boundary have to go in ahead of it. Closing the run
        # only when a pass is cut short is not enough: a recovery delay
        # between whole passes is the ordinary case, and skipping it there
        # leaves every such delay sitting in front of the template's rows.
        self._close_template_run()
        if self._template_cursor:
            # The pass stops here. Its unused rows go when the sequence is
            # finished; a later template block would land on top of them.
            self._template_gap = True
            self._template_cursor = 0

    def _template_set_block(self, payload: dict | None, events=None) -> None:
        """Fill in one block of the template: write what moved, where it goes.

        There is no registering left to do. The row exists, its ID is already in
        the block table, and the only question is which of its columns this pass
        moves -- so this is a handful of scalar writes into a matrix the library
        is already holding, and nothing else at all.

        ``events`` is the block as ``add_block(*block)`` received it, already
        confirmed to be the template's own objects; each slot then reads its own
        event rather than a payload entry.
        """
        if self._template_finished:
            # Reading the block table gives back every row the scan did not
            # reach -- the whole point of claiming them up front is that the
            # unused ones can be counted off the end. A block added now would
            # write into a row that no longer exists, at an ID something else
            # has since been given.
            raise RuntimeError(
                "the TR template was finished when this sequence's block table was read, and "
                "its unused rows given back; it cannot be added to afterwards. Build the "
                "whole sequence before writing or plotting it."
            )
        cursor = self._template_cursor
        template = self._template[cursor]
        if not template.fast:
            raise NotImplementedError(
                f"TR template block {cursor} cannot be built from a payload: {template.reason}. "
                "Add this block with its events instead."
            )
        if cursor == 0:
            if self._template_gap:
                raise RuntimeError(
                    "a TR template pass was ended early by a block outside it, and another pass "
                    "has started. Only the last pass may be short."
                )
            # Counted in template *passes*, not in blocks: a delay the plugin
            # plays between passes is a block of the sequence but not a block of
            # the template, so measuring this against the block table would
            # charge the loop for them and refuse a loop that fits.
            if self._template_pass >= self._loop_size:
                raise RuntimeError(
                    f"TR template loop_size={self._loop_size} allows {self._loop_size} passes over "
                    f"the TR; pass {self._template_pass + 1} exceeds it."
                )

        index = self._template_pass
        peaks = self._peak_cache
        if events is None:
            extensions = payload.get("ext", ())
            duration = payload.get("duration")
            if duration is not None and duration > template.static_duration:
                self._template_block_durations[index * len(self._template) + cursor] = duration
        else:
            # A bare delay or a `make_delay` alongside the events sets the
            # block's duration, exactly as the payload's "duration" key does.
            extensions = ()
            duration = _delay_duration(events) or 0.0
            if duration > template.static_duration:
                self._template_block_durations[index * len(self._template) + cursor] = duration

        for slot in template.slots:
            if slot.waveform is not None:
                chosen = (
                    slot.event.waveform if events is not None
                    else payload.get("shape", _NO_SHAPES).get(slot.key)
                )
                # An index is the fast, intended form: the module says *which*
                # of its candidates this shot plays and nothing is compared at
                # all. Otherwise identity, which is the answer for a module
                # replaying one designed waveform -- a spiral interleaf under a
                # rotation pays nothing here -- then value, for a module that
                # rebuilds an equal array every shot.
                if isinstance(chosen, (int, np.integer)):
                    position = int(chosen)
                elif chosen is slot.waveform or _same_waveform(chosen, slot.waveform):
                    position = None
                elif slot.shapes is not None:
                    position = slot.shapes.by_identity.get(id(chosen))
                    if position is None:
                        raise NotImplementedError(
                            f"TR template block {cursor}: the arbitrary gradient on {slot.key} "
                            "plays samples that are not in the inventory its module published. "
                            "Every waveform a shot can play has to be built at construction."
                        )
                else:
                    # The samples moved and there is no inventory, so there is
                    # no registered shape to point at. Compressing them here
                    # would re-derive an ID the module could simply have named,
                    # and emitting the template's shape under this shot's peak
                    # is a wrong file. So: refused.
                    raise NotImplementedError(
                        f"TR template block {cursor}: the arbitrary gradient on {slot.key} has "
                        "samples this shot changed, but the module published no waveform "
                        "inventory for that slot, so there is no registered shape to point at. "
                        "Hand every shot's waveform to the module's constructor."
                    )
                if position is not None:
                    rows, at = (slot.later_rows, index - 1) if index else (slot.first_rows, 0)
                    for column, shape_id in zip(
                        slot.shape_columns, slot.shapes.resolve(self, position), strict=True
                    ):
                        rows[at, column] = shape_id
            if not slot.dynamic:
                # Nothing a payload can move: the row was written once, when the
                # sequence was built, and every pass points back at it.
                continue
            values = (
                slot.event_values(slot.event, peaks)
                if events is not None
                else payload.get(slot.key)
                if slot.key
                else _ext_values(extensions, slot.ext_ordinal)
            )
            if values is None:
                continue
            if index:
                rows, at = slot.later_rows, index - 1
            else:
                rows, at = slot.first_rows, 0
            for column, position in slot.dynamic:
                rows[at, column] = values[position]

        self._template_written += 1
        self._n_blocks += 1
        self._block_rows = None
        cursor += 1
        if cursor == len(self._template):
            cursor = 0
            self._template_pass = index + 1
        self._template_cursor = cursor

    def _close_template_run(self) -> None:
        """Hand the block table the template rows filled since it last looked.

        The rows were written when the sequence was built, so this is a slice,
        not a copy -- what it marks is where a block from outside the template
        interrupts, so that the two land in the table in the order they were
        added.
        """
        start, stop = self._template_run_start, self._template_written
        if stop <= start:
            return
        self._block_runs.append(self._template_rows[start:stop])
        self._duration_runs.append(self._template_block_durations[start:stop])
        self._block_tail = None
        self._duration_tail = None
        self._block_rows = None
        self._template_run_start = stop

    def _finish_template(self) -> None:
        """Close the template off: hand over its rows, give back what it did not use.

        A scan that plays fewer passes than it claimed -- or whose last pass is
        short -- leaves rows nothing points at, and an unreferenced row is still
        a row in the file. They are always a *tail*: rows are claimed in the
        order the blocks fill them, so giving them back is a truncation.
        """
        if self._template is None or self._template_finished:
            return
        self._template_finished = True
        self._close_template_run()
        length = len(self._template)
        played = self._template_written
        # Blocks reached in the first pass, and in the last one.
        first = min(played, length)
        full, remainder = divmod(played, length)
        for kind, name, first_id, slots, movers in self._template_claims:
            if kind == "collapsed":
                keep = sum(1 for slot in slots if slot.block < first)
                if full:
                    keep += (full - 1) * len(movers)
                    keep += sum(1 for slot in movers if slot.block < remainder)
            else:
                keep = full * len(slots) + sum(1 for slot in slots if slot.block < remainder)
            getattr(self, name).truncate(first_id - 1 + keep)
        if self._template_chain_levels:
            levels, first_node = self._template_chain_levels
            keep = full * levels[-1] + levels[min(remainder, length)]
            library = self.extensions_library
            for node in range(first_node + keep, library.next_free_ID):
                del library.keymap[library.data.pop(node)]
            library.next_free_ID = first_node + keep

    def _chain_extensions(self, extensions) -> int:
        """Canonical extension chain for one block; the ``_fast_set_block`` logic."""
        ext_lib = self.extensions_library
        ext_find = ext_lib.find

        all_found = True
        extension_id = 0
        for ext_type, ext_ref in extensions:
            extension_id, found = ext_find((ext_type, ext_ref, extension_id))
            if not found:
                all_found = False
                break
        if all_found:
            return extension_id

        extension_id = 0
        for ext_type, ext_ref in extensions:
            data = (ext_type, ext_ref, extension_id)
            extension_id, found = ext_find(data)
            if not found:
                ext_lib.insert(extension_id, data)
        return extension_id

    def remove_duplicates(self, in_place: bool = False) -> Sequence:
        """Remove duplicates with hardcoded rounded profiles per library.

        Unlike upstream pypulseq, this also compacts extension-related
        libraries and canonicalizes extension linked lists so identical chains
        are shared across blocks.

        This is the pass that decides the file, and the only one that collapses
        anything: nothing is deduplicated on the way in, so a sequence built
        through :meth:`add_block`
        arrives here with one entry per event and comes out the same way as one
        that arrived pre-collapsed.

        Parameters
        ----------
        in_place:
            If ``True``, deduplicate current instance; otherwise return a copy.
        """
        self._finish_template()
        seq_copy = self if in_place else self._clone_for_dedup()

        seq_copy.shape_library, shape_map = seq_copy.shape_library.remove_duplicates(9)
        shape_lut = _index_lookup(shape_map)

        # Every library that embeds shape IDs has them renumbered as part of
        # its own deduplication pass; see _dedup_library_approx.
        seq_copy.arb_library, arb_map = _dedup_library_approx(
            seq_copy.arb_library, _ROUNDING_PROFILES["arb_library"], remap={3: shape_lut, 4: shape_lut}
        )
        seq_copy.trap_library, trap_map = _dedup_library_approx(seq_copy.trap_library, _ROUNDING_PROFILES["trap_library"])

        # A gradient entry is one reference into either the arbitrary or the
        # trapezoid library, and those number independently -- so which lookup
        # applies is decided by the entry's own type, and the type has to stay
        # part of the deduplication key.
        #
        # This is the largest library in the sequence -- one entry per driven
        # axis per block, near five million for a large 3D protocol -- so it is
        # read as two arrays rather than as ``data`` and ``type``. Those two
        # dicts alone used to be half of the whole deduplication pass.
        grad_lib = seq_copy.grad_library
        if len(grad_lib):
            if isinstance(grad_lib, _AppendOnlyLibrary):
                refs = grad_lib.matrix()[:, 0].astype(np.int64)
                kinds = grad_lib.type_mask("g")
                traps = grad_lib.type_mask("t")
            else:
                grad_ids = sorted(grad_lib.data)
                grad_data, grad_types = grad_lib.data, grad_lib.type
                refs = np.fromiter((grad_data[i][0] for i in grad_ids), dtype=np.int64, count=len(grad_ids))
                kinds = np.fromiter((grad_types.get(i, "") == "g" for i in grad_ids), dtype=bool, count=len(grad_ids))
                traps = np.fromiter((grad_types.get(i, "") == "t" for i in grad_ids), dtype=bool, count=len(grad_ids))
            new_refs = refs.copy()
            new_refs[kinds] = arb_map[refs[kinds]]
            new_refs[traps] = trap_map[refs[traps]]
            # 1 for arbitrary, 2 for trapezoid, 0 for anything else -- the same
            # three-way split the per-entry type string gave.
            type_codes = np.where(kinds, 1.0, np.where(traps, 2.0, 0.0))
            seq_copy.grad_library, grad_map = _dedup_library_approx(
                grad_lib,
                _ROUNDING_PROFILES["grad_library"],
                values=new_refs.reshape(-1, 1).astype(float),
                type_codes=type_codes,
            )
        else:
            seq_copy.grad_library, grad_map = _dedup_library_approx(grad_lib, _ROUNDING_PROFILES["grad_library"])

        seq_copy.rf_library, rf_map = _dedup_library_approx(
            seq_copy.rf_library,
            _ROUNDING_PROFILES["rf_library"],
            remap={1: shape_lut, 2: shape_lut, 3: shape_lut},
        )
        seq_copy.adc_library, adc_map = _dedup_library_approx(
            seq_copy.adc_library, _ROUNDING_PROFILES["adc_library"], remap={7: shape_lut}
        )

        # The block table is the one structure whose size is the scan's rather
        # than the event vocabulary's, so it is renumbered as five column
        # lookups on one integer matrix instead of five dict reads per block.
        rows = seq_copy._block_row_matrix()
        if rows.size:
            rows[:, 1] = rf_map[rows[:, 1]]
            rows[:, 2] = grad_map[rows[:, 2]]
            rows[:, 3] = grad_map[rows[:, 3]]
            rows[:, 4] = grad_map[rows[:, 4]]
            rows[:, 5] = adc_map[rows[:, 5]]

        seq_copy.trigger_library, trig_map = _dedup_library_approx(seq_copy.trigger_library, _ROUNDING_PROFILES["trigger_library"])
        seq_copy.label_set_library, label_set_map = _dedup_library_approx(seq_copy.label_set_library, _ROUNDING_PROFILES["label_set_library"])
        seq_copy.label_inc_library, label_inc_map = _dedup_library_approx(seq_copy.label_inc_library, _ROUNDING_PROFILES["label_inc_library"])
        seq_copy.rotation_library, rotation_map = _dedup_library_approx(seq_copy.rotation_library, _ROUNDING_PROFILES["rotation_library"])

        if seq_copy.rf_shim_library.data:
            widths = {len(v) for v in seq_copy.rf_shim_library.data.values()}
            if len(widths) != 1:
                raise RuntimeError("rf_shim_library has mixed payload widths; cannot apply fixed rounded dedup profile")
            rf_shim_digits = tuple([_ROUNDING_PROFILES["rf_shim_library"]] * next(iter(widths)))
            seq_copy.rf_shim_library, rf_shim_map = _dedup_library_approx(seq_copy.rf_shim_library, rf_shim_digits)
        else:
            seq_copy.rf_shim_library, rf_shim_map = _dedup_library_approx(seq_copy.rf_shim_library, _ROUNDING_PROFILES["rf_shim_library"])

        old_ext_lib = seq_copy.extensions_library
        new_ext_lib = EventLibrary()
        node_cache: dict[tuple[int, int, int], int] = {}

        ref_maps = {
            "TRIGGERS": trig_map,
            "LABELSET": label_set_map,
            "LABELINC": label_inc_map,
            "ROTATIONS": rotation_map,
            "RF_SHIMS": rf_shim_map,
        }

        # An extension chain is canonicalised once per *node*, not once per
        # block. The chain hanging off a node is determined by that node alone,
        # so a scan whose every TR carries labels canonicalises its handful of
        # distinct chains instead of repeating the walk for each of hundreds of
        # thousands of blocks; the blocks then follow with one column lookup.
        #
        # A node's ``next`` is always an earlier node -- chains are built
        # tail-first -- so ascending ID order visits every tail before its head,
        # and it is also the order the old per-block walk first reached each
        # node. New IDs therefore come out in the same order as before.
        # Only the ``tail`` of a chain has to be resolved one node at a time;
        # everything else about a node -- its type, and what its reference
        # renumbers to -- depends on nothing but the node, so both are settled
        # for the whole library first and the walk left with a dict lookup.
        if old_ext_lib.data:
            old_data = old_ext_lib.data
            n_nodes = len(old_data)
            node_ids = np.fromiter(old_data.keys(), dtype=np.int64, count=n_nodes)
            # Nodes are numbered as they are created, so the dict is already in
            # ascending key order; that is checked rather than assumed, and
            # far cheaper than sorting a million keys to find it out.
            if n_nodes > 1 and not bool(np.all(node_ids[:-1] < node_ids[1:])):
                node_ids = np.sort(node_ids)
                payloads = np.fromiter(
                    itertools.chain.from_iterable(old_data[node_id][:3] for node_id in node_ids.tolist()),
                    dtype=np.int64,
                    count=3 * n_nodes,
                )
            else:
                payloads = np.fromiter(
                    itertools.chain.from_iterable(old_data.values()), dtype=np.int64, count=3 * n_nodes
                )
            payloads = payloads.reshape(n_nodes, 3)

            # One remap per extension type instead of one per node.
            ext_types, refs = payloads[:, 0], payloads[:, 1]
            remapped_refs = refs.copy()
            for ext_type_id in np.unique(ext_types).tolist():
                type_name = seq_copy.get_extension_type_string(ext_type_id)
                selected = ext_types == ext_type_id
                if type_name == "DELAYS":
                    remapped_refs[selected] = 0
                else:
                    ref_map = ref_maps.get(type_name)
                    if ref_map is not None:
                        remapped_refs[selected] = ref_map[refs[selected]]

            # Handed to the walk as three flat lists rather than one list of
            # rows: a row per node would allocate a million short lists, and
            # the walk reads the three fields separately anyway.
            type_list = ext_types.tolist()
            ref_list = remapped_refs.tolist()
            next_list = payloads[:, 2].tolist()

            # A plain list, not an array: this is indexed one node at a time.
            new_head = [0] * (int(node_ids[-1]) + 1)
            new_nodes: list[tuple[int, int, int]] = []
            next_new_id = 1
            for node_id, ext_type_id, remapped_ref, next_id in zip(
                node_ids.tolist(), type_list, ref_list, next_list, strict=True
            ):
                tail = new_head[next_id]
                if remapped_ref == 0:
                    new_head[node_id] = tail  # dropped: the block inherits the rest of the chain
                    continue
                key = (ext_type_id, remapped_ref, tail)
                node = node_cache.get(key)
                if node is None:
                    node = next_new_id
                    next_new_id += 1
                    node_cache[key] = node
                    new_nodes.append(key)
                new_head[node_id] = node

            # ``node_cache`` is already {payload: id}, which is exactly the
            # reverse index EventLibrary.insert would have built one call at a
            # time -- half a million of them for a scan that labels every TR.
            new_ext_lib.data.update(zip(range(1, next_new_id), new_nodes, strict=True))
            new_ext_lib.keymap = node_cache
            new_ext_lib.next_free_ID = next_new_id

            if rows.size:
                rows[:, 6] = np.asarray(new_head, dtype=np.int64)[rows[:, 6]]

        seq_copy.extensions_library = new_ext_lib
        seq_copy.block_cache.clear()
        return seq_copy

    @property
    def block_events(self) -> dict[int, list]:
        """Block table as ``{block_id: [duration, rf, gx, gy, gz, adc, ext]}``.

        The table is kept as ordered runs -- Python rows from a block added on
        its own, whole matrices claimed by the TR template -- and this dict is
        built from them only when something asks. Neither writing nor deduplication does:
        both read the matrix, and rebuilding six hundred thousand seven-element
        lists costs more than the work they came to do.
        """
        if self._block_dict is None or len(self._block_dict) != self._n_blocks:
            rows = self._block_row_matrix()
            self._block_dict = dict(enumerate(rows.tolist(), start=1))
        return self._block_dict

    @block_events.setter
    def block_events(self, value: dict[int, list]) -> None:
        # Upstream's __init__ assigns an empty dict; anything else replaces the
        # table wholesale, which only a caller building one by hand would do.
        self._block_runs = [list(value.values())] if value else [[]]
        self._block_tail = self._block_runs[0]
        self._n_blocks = len(value)
        self._block_dict = dict(value) if value else None
        self._block_rows = None

    @property
    def block_durations(self) -> dict[int, float]:
        """Block durations in seconds, keyed by block ID; built on demand."""
        if self._duration_dict is None or len(self._duration_dict) != self._n_blocks:
            self._duration_dict = dict(enumerate(self._block_duration_array().tolist(), start=1))
        return self._duration_dict

    @block_durations.setter
    def block_durations(self, value: dict[int, float]) -> None:
        self._duration_runs = [list(value.values())] if value else [[]]
        self._duration_tail = self._duration_runs[0]
        self._duration_dict = dict(value) if value else None

    def _block_row_matrix(self) -> np.ndarray:
        """The block table as one editable ``(n_blocks, 7)`` integer matrix.

        Deduplication renumbers five of those seven columns and writing reads
        all of them, so the table is materialised once as an array rather than
        row by row. It is cached, and invalidated by block count.

        Any rows the template path is still holding are registered first: their
        IDs are the very columns a reader has come for.
        """
        self._finish_template()
        return self._raw_block_row_matrix()

    def _raw_block_row_matrix(self) -> np.ndarray:
        """The block table as it stands, without settling buffered event IDs."""
        rows = self._block_rows
        if rows is None or len(rows) != self._n_blocks:
            parts = [
                run if isinstance(run, np.ndarray) else np.array(run, dtype=np.int64).reshape(-1, 7)
                for run in self._block_runs
                if len(run)
            ]
            if not parts:
                rows = np.zeros((0, 7), dtype=np.int64)
            else:
                rows = parts[0].astype(np.int64, copy=True) if len(parts) == 1 else np.concatenate(parts)
                if rows.dtype != np.int64:
                    rows = rows.astype(np.int64)
            self._block_rows = rows
        return rows

    def _block_duration_array(self) -> np.ndarray:
        """Block durations as one ``(n_blocks,)`` float array."""
        self._finish_template()
        parts = [
            run if isinstance(run, np.ndarray) else np.array(run, dtype=float)
            for run in self._duration_runs
            if len(run)
        ]
        if not parts:
            return np.zeros(0)
        return parts[0] if len(parts) == 1 else np.concatenate(parts)

    def _reserve_blocks(self, count: int):
        """Claim ``count`` consecutive blocks; returns ``(rows, durations, first_id)``.

        Both slabs are the caller's to fill in place. IDs are positions, so the
        caller knows every block ID it is about to use before filling anything.
        """
        rows = np.zeros((int(count), 7), dtype=np.int64)
        durations = np.zeros(int(count), dtype=float)
        first_id = self._n_blocks + 1
        self._block_runs.append(rows)
        self._duration_runs.append(durations)
        self._block_tail = None
        self._duration_tail = None
        self._n_blocks += int(count)
        self.next_free_block_ID = self._n_blocks + 1
        self._block_rows = None
        return rows, durations, first_id

    def _clone_for_dedup(self) -> Sequence:
        """A private copy holding only the members deduplication rewrites.

        Deduplication replaces every event library outright and renumbers the
        event IDs on each block row, and the caller then records
        ``TotalDuration`` in the definitions. Nothing else changes, so nothing
        else is worth copying: a ``deepcopy`` of the whole sequence would walk
        every tuple in every library -- some three million of them for a large
        3D scan -- to produce copies that the very next step discards.

        Nothing is rewritten *in place* any more: each library is rebuilt by
        ``_dedup_library_approx`` and the block table is renumbered in a fresh
        matrix, so the original sequence stays usable and no dict here needs
        duplicating either.
        """
        clone = copy.copy(self)
        clone.definitions = dict(self.definitions)
        clone.block_cache = {}
        # The block runs are shared, but the matrix built from them is not:
        # deduplication renumbers it in place, and the original must survive.
        clone._block_rows = None
        clone._block_dict = None
        clone._duration_dict = None
        clone._view_cache = None
        clone._collection_cache = None
        clone._segment_cache = None
        clone._rf_shape_cache = {}
        clone._grad_shape_cache = {}
        return clone

    @property
    def custom_labels(self) -> dict[str, int]:
        """Labels auto-registered beyond the built-in ``get_supported_labels()`` set.

        Maps ``label_string -> int_idx`` for every custom label encountered via
        :meth:`add_block`.  The custom write helper uses this to serialise them.
        """
        n_builtin = len(get_supported_labels())
        return {name: idx for idx, name in self._label_registry_inv.items() if idx > n_builtin}

    def rf_from_lib_data(self, lib_data: list, use: str | int = "") -> SimpleNamespace:
        """Decode RF use from numeric code (fast path) or legacy char."""
        if isinstance(use, int | np.integer):
            use = _RF_USE_CODE_TO_CHAR.get(int(use), "u")
        return super().rf_from_lib_data(lib_data, use)

    # ------------------------------------------------------------------
    # Inspection views
    # ------------------------------------------------------------------

    def payload(self) -> bytes:
        """Serialise this sequence to the ``.seq`` payload the views are built from.

        Deduplicated, exactly as :func:`pulserver.io.write` would emit it —
        the C backend requires canonical event IDs to recognise repeated RF
        shims and gradients across TRs, and the views should show what is
        actually going to be written.
        """
        from pulserver.io import write

        return write(self, output=None, remove_duplicates=True, check_timing=False)

    def _views(self) -> dict:
        """Build (or reuse) the upstream views of the current block list."""
        n_blocks = self.next_free_block_ID - 1
        cache = self._view_cache
        if cache is None or cache["n_blocks"] != n_blocks:
            plain, plotting, extensions = build_views(self, self.payload())
            cache = {
                "n_blocks": n_blocks,
                "plain": plain,
                "plot": plotting,
                "extensions": extensions,
            }
            self._view_cache = cache
        return cache

    @property
    def _seq(self) -> pp.Sequence:
        """Plain upstream view: RF, gradients and ADC, extensions dropped.

        This is what the file literally stores. Built on first access and
        reused until more blocks are appended.
        """
        return self._views()["plain"]

    @property
    def _seqplot(self) -> pp.Sequence:
        """Plotting view: rotations applied and RF shims expanded.

        Every ``ROTATIONS`` extension has been folded into the block's
        gradients and every ``RF_SHIMS`` extension into a multi-channel RF
        waveform, so this view shows what the scanner actually plays. It is
        the same object as :attr:`_seq` when the sequence uses neither.
        """
        return self._views()["plot"]

    @property
    def extensions(self) -> dict:
        """Block extensions, keyed by 1-based block index.

        See :class:`pulserver.pypulseq._extensions.BlockExtensions`.
        """
        return self._views()["extensions"]

    def _collection(self):
        """Build (or reuse) the C-backend collection for structural queries."""
        n_blocks = self.next_free_block_ID - 1
        cache = self._collection_cache
        if cache is None or cache["n_blocks"] != n_blocks:
            from pulserver._ext._pulseg_wrapper import _PulseqCollection

            system = self.system
            collection = _PulseqCollection(
                [self.payload()],
                float(system.gamma),
                float(system.B0),
                float(system.max_grad),
                float(system.max_slew),
                float(system.rf_raster_time),
                float(system.grad_raster_time),
                float(system.adc_raster_time),
                float(system.block_duration_raster),
                True,
                1,
            )
            cache = {"n_blocks": n_blocks, "collection": collection}
            self._collection_cache = cache
        return cache["collection"]

    # ------------------------------------------------------------------
    # TR and segment structure (C backend)
    # ------------------------------------------------------------------

    @property
    def tr_info(self) -> dict:
        """TR structure of the sequence as detected by the C backend.

        Returns
        -------
        dict
            Keys include ``tr_size`` (blocks per TR), ``num_trs``,
            ``num_prep_blocks``, ``tr_duration_us`` and ``num_canonical_trs``.
        """
        from pulserver._ext._pulseg_wrapper import _find_tr

        return _find_tr(self._collection(), subsequence_idx=0)

    @property
    def num_trs(self) -> int:
        """Number of TRs detected in the sequence."""
        return int(self.tr_info["num_trs"])

    @property
    def tr_duration(self) -> float:
        """Detected TR duration in seconds."""
        return float(self.tr_info["tr_duration_us"]) * 1e-6

    def tr_block_range(self, tr_index: int) -> tuple[int, int]:
        """Inclusive 1-based block range of TR ``tr_index``.

        The TR layout comes from the C library, so a plugin does not have to
        declare it — whatever structure the blocks actually have is what gets
        indexed.
        """
        info = self.tr_info
        num_trs = int(info["num_trs"])
        if num_trs <= 0:
            raise ValueError("No TR structure was detected in this sequence.")
        if not -num_trs <= tr_index < num_trs:
            raise IndexError(f"tr_index {tr_index} out of range for {num_trs} TRs")
        if tr_index < 0:
            tr_index += num_trs

        tr_size = int(info["tr_size"])
        first = int(info["num_prep_blocks"]) + tr_index * tr_size + 1
        return first, first + tr_size - 1

    @property
    def segments(self) -> tuple[Segment, ...]:
        """Every virtual segment, resolved to its max-energy instance.

        Segmentation is the C library's, so this is the same partitioning the
        scanner executes and the safety backend analyses.
        """
        n_blocks = self.next_free_block_ID - 1
        cache = self._segment_cache
        if cache is None or cache["n_blocks"] != n_blocks:
            try:
                from pulserver._ext._pulseg_wrapper import _get_segments
            except ImportError as exc:  # pragma: no cover - stale build artifact
                raise RuntimeError(
                    "The pulseg extension module predates segment queries. Rebuild it: "
                    "cmake --build build --target _pulseg_wrapper"
                ) from exc

            segments = []
            for row in _get_segments(self._collection(), 0):
                indices = list(row["block_indices"])
                if not indices:
                    continue
                segments.append(
                    Segment(
                        index=int(row["index"]),
                        first_block=min(indices) + 1,
                        last_block=max(indices) + 1,
                        duration=float(row["duration_us"]) * 1e-6,
                        pure_delay=bool(row["pure_delay"]),
                        is_nav=bool(row["is_nav"]),
                        has_trigger=bool(row["has_trigger"]),
                        from_max_energy_instance=bool(row["from_max_energy_instance"]),
                        _parent=self,
                    )
                )
            cache = {"n_blocks": n_blocks, "segments": tuple(segments)}
            self._segment_cache = cache
        return cache["segments"]

    def segment(self, index: int) -> Segment:
        """Return one :class:`Segment` by index."""
        return self.segments[index]

    # ------------------------------------------------------------------
    # Scope resolution
    # ------------------------------------------------------------------

    def _scope(
        self,
        tr_index: int | None = None,
        segment: int | None = None,
    ) -> tuple[int, int] | None:
        """Resolve ``tr_index`` / ``segment`` to an inclusive block range."""
        if tr_index is not None and segment is not None:
            raise ValueError("Pass either tr_index or segment, not both.")
        if tr_index is not None:
            return self.tr_block_range(tr_index)
        if segment is not None:
            seg = self.segment(segment)
            return seg.first_block, seg.last_block
        return None

    def _time_range(
        self,
        tr_index: int | None,
        segment: int | None,
        default=(0, np.inf),
    ):
        """Resolve a scope to an absolute time range in seconds."""
        block_range = self._scope(tr_index, segment)
        if block_range is None:
            return default
        return time_bounds(self._seqplot, *block_range)

    def _scoped_view(self, tr_index: int | None, segment: int | None) -> pp.Sequence:
        """Return the plotting view, sliced when a scope is given."""
        block_range = self._scope(tr_index, segment)
        if block_range is None:
            return self._seqplot
        return slice_view(self._seqplot, *block_range)

    # ------------------------------------------------------------------
    # Visualisation and analysis
    # ------------------------------------------------------------------

    def plot(self, *, tr_index: int | None = None, segment: int | None = None, **kwargs):
        """Plot the sequence as the scanner plays it.

        Rotations and RF shims are resolved into the waveforms (see
        :attr:`_seqplot`), so a rotated readout is drawn on the axes it will
        actually run on and a pTx pulse shows one trace per transmit channel.

        Parameters
        ----------
        tr_index : int, optional
            Restrict the plot to one TR. Mutually exclusive with ``segment``.
        segment : int, optional
            Restrict the plot to one segment's max-energy instance.
        **kwargs
            Passed through to :meth:`pypulseq.Sequence.plot`; ``time_range``
            is set for you when a scope is given.
        """
        kwargs.setdefault("time_range", self._time_range(tr_index, segment))
        return self._seqplot.plot(**kwargs)

    def check_timing(self, print_errors: bool = False):
        """Run upstream's timing check against the plotting view."""
        return self._seqplot.check_timing(print_errors=print_errors)

    def duration(self):
        """Total duration, block count and event count of the sequence."""
        return self._seqplot.duration()

    def test_report(self) -> str:
        """Upstream's textual sequence report."""
        return self._seqplot.test_report()

    def waveforms(self, *, tr_index: int | None = None, segment: int | None = None, append_RF: bool = False):
        """Uniformly sampled waveforms, optionally restricted to a TR or segment."""
        time_range = self._time_range(tr_index, segment, default=None)
        return self._seqplot.waveforms(append_RF=append_RF, time_range=time_range)

    def waveforms_and_times(self, append_RF: bool = False):
        """Event waveforms with their time stamps, from the plotting view."""
        return self._seqplot.waveforms_and_times(append_RF=append_RF)

    def calculate_kspace(self, *, tr_index: int | None = None, segment: int | None = None, **kwargs):
        """k-space trajectory of the sequence, or of one TR / segment.

        Upstream's trajectory calculation has no time-range argument, so a
        scoped call runs against a sliced copy of the plotting view.
        """
        return self._scoped_view(tr_index, segment).calculate_kspace(**kwargs)

    def plot_kspace(
        self,
        *,
        tr_index: int | None = None,
        segment: int | None = None,
        plot_now: bool = True,
        **kwargs,
    ):
        """Plot the k-space trajectory, with ADC sample locations marked.

        Parameters
        ----------
        tr_index, segment : int, optional
            Restrict to one TR or segment.
        plot_now : bool, default True
            Show the figure immediately.
        **kwargs
            Passed to :meth:`calculate_kspace`.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        k_traj_adc, k_traj, *_ = self.calculate_kspace(tr_index=tr_index, segment=segment, **kwargs)

        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        axes[0].plot(k_traj.T)
        axes[0].set_xlabel("Sample")
        axes[0].set_ylabel("k (1/m)")
        axes[0].set_title("Trajectory components")
        axes[0].legend(["kx", "ky", "kz"], loc="upper right")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(k_traj[0], k_traj[1], color="0.6", linewidth=0.8)
        axes[1].plot(k_traj_adc[0], k_traj_adc[1], ".", markersize=2)
        axes[1].set_xlabel("kx (1/m)")
        axes[1].set_ylabel("ky (1/m)")
        axes[1].set_title("kx-ky (ADC samples marked)")
        axes[1].set_aspect("equal", adjustable="datalim")
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        if plot_now:
            plt.show()
        return fig

    def grad_spectrum(
        self,
        *,
        tr_index: int | None = None,
        segment: int | None = None,
        bands: list | None = None,
        esp_file=None,
        asc_file=None,
        window_width: float | None = None,
        max_frequency: float = 2000.0,
        plot: bool = True,
        **kwargs,
    ):
        """Gradient spectrum, with mechanical resonance bands overlaid.

        This is a visualisation only — whether a sequence clears a forbidden
        band is decided by the C safety core at predownload, not here.

        Parameters
        ----------
        tr_index, segment : int, optional
            Restrict the analysis to one TR or segment. When ``tr_index`` is
            given and ``window_width`` is not, the window is set to the whole
            TR, so the result is one spectrum rather than a spectrogram.
        bands : list of tuple, optional
            Forbidden bands as
            ``(freq_min_hz, freq_max_hz, max_amplitude_mT_per_m[, channel])``.
            Defaults to ``system.forbidden_bands`` when the sequence's
            :class:`~pypulseq.Opts` carries them.
        esp_file : str or pathlib.Path, optional
            Read the bands from a GE ESP lockout table instead. See
            :func:`~pulserver.pypulseq._safety.read_esp_bands`.
        asc_file : str or pathlib.Path, optional
            Read the bands from a Siemens ``.asc`` file instead. See
            :func:`~pulserver.pypulseq._safety.read_asc_bands`.
        window_width : float, optional
            Analysis window in seconds.
        max_frequency : float, default 2000.0
            Upper frequency limit of the spectrum.
        plot : bool, default True
            Draw the spectrogram.

        Returns
        -------
        tuple
            ``(spectrograms, spectrogram_sos, frequencies, times)``, as
            :meth:`pypulseq.Sequence.calculate_gradient_spectrum` returns.
        """
        if esp_file is not None and asc_file is not None:
            raise ValueError("Pass either esp_file or asc_file, not both.")
        if esp_file is not None:
            bands = read_esp_bands(esp_file)
        elif asc_file is not None:
            bands = read_asc_bands(asc_file)
        elif bands is None:
            bands = list(getattr(self.system, "forbidden_bands", []) or [])

        time_range = self._time_range(tr_index, segment, default=None)
        if window_width is None:
            window_width = (time_range[1] - time_range[0]) if time_range is not None else 0.05

        # Upstream samples only as far as the last gradient in the range, so a
        # window covering the scope's dead time would exceed the signal and
        # scipy would reject the overlap. Clamp to what will actually be there.
        _, sampled_t = self._sample_gradients(tr_index, segment, self.system.grad_raster_time)
        window_width = min(window_width, len(sampled_t) * self.system.grad_raster_time)

        return self._seqplot.calculate_gradient_spectrum(
            max_frequency=max_frequency,
            window_width=window_width,
            time_range=list(time_range) if time_range is not None else None,
            plot=plot,
            acoustic_resonances=bands_to_resonances(bands),
            **kwargs,
        )

    def mech_resonances(
        self,
        *,
        canonical_tr_idx: int = 0,
        bands: list | None = None,
        esp_file=None,
        asc_file=None,
        max_frequency: float | None = None,
        resolution: float | None = None,
        compress_trains: bool = True,
        plot: bool = True,
    ):
        """A_eq mechanical-resonance lines, straight from the C safety core.

        Unlike :meth:`grad_spectrum`, which draws pypulseq's own spectrogram
        with the bands shaded on top, this runs the compiled structural
        analysis and returns the very lines the headless predownload gate
        decides on — ``candidate_freqs``, ``candidate_grad_amps`` and
        ``candidate_violations`` are the arrays `pulseg_check_safety`
        thresholds. There is no separate Python re-implementation.

        Parameters
        ----------
        canonical_tr_idx : int, default 0
            Which canonical TR to analyse.
        bands : list of tuple, optional
            Forbidden bands as
            ``(freq_min_hz, freq_max_hz, max_amplitude_mT_per_m[, channel])``.
            Defaults to ``system.forbidden_bands``.
        esp_file : str or pathlib.Path, optional
            Read the bands from a GE ESP lockout table instead.
        asc_file : str or pathlib.Path, optional
            Read the bands from a Siemens ``.asc`` file instead.
        max_frequency, resolution : float, optional
            Analysis window. Both default to the same rule the headless gate
            applies, so the plot matches what the scanner computes.
        compress_trains : bool, default True
            Evaluate equally-spaced occurrence trains in compressed form,
            exactly as the headless gate does. Set ``False`` to get the
            uncompressed reference evaluation of the same math — a debugging
            aid, and the only mode in which a ``component_*`` term maps to a
            single materialised occurrence rather than to a whole train.
        plot : bool, default True
            Draw the spectrum, the analytic envelope, the TR-harmonic stems
            and the candidate markers.

        Returns
        -------
        dict
            The full spectra record: ``spectrum_full_g*`` (kissfft continuous
            spectrum), ``envelope_*`` (dense analytic A_eq), ``analytical_peak_*``
            (A_eq at every TR harmonic), ``candidate_*`` (the verdict lines) and
            ``component_*`` (per-term breakdown).
        """
        from pulserver._ext._pulseg_wrapper import _calc_mech_resonances

        if esp_file is not None and asc_file is not None:
            raise ValueError("Pass either esp_file or asc_file, not both.")
        if esp_file is not None:
            bands = read_esp_bands(esp_file)
        elif asc_file is not None:
            bands = read_asc_bands(asc_file)
        elif bands is None:
            bands = list(getattr(self.system, "forbidden_bands", []) or [])
        if not bands:
            raise ValueError(
                "No forbidden bands. Pass bands=, esp_file= or asc_file=, or set forbidden_bands "
                "on the sequence's pulserver.pypulseq.Opts."
            )

        # mT/m -> Hz/m, the unit the C core works in.
        to_hz_per_m = float(self.system.gamma) * 1e-3
        c_bands = [
            (float(b[0]), float(b[1]), float(b[2]) * to_hz_per_m if float(b[2]) > 0 else float(b[2]))
            for b in bands
        ]

        # Mirror pulseg_check_safety's own window rule so the displayed
        # spectrum has the resolution the gate ran at. The candidate lines
        # themselves do not depend on either value -- they are the TR
        # harmonics inside each guarded band -- so they match regardless.
        widths = [hi - lo for lo, hi, *_ in c_bands if hi > lo]
        if resolution is None:
            resolution = min(max(min(widths) / 4.0, 1.0), 5.0) if widths else 1.0
        if max_frequency is None:
            max_frequency = max(1.2 * max(hi for _, hi, *_ in c_bands), 3000.0)

        record = _calc_mech_resonances(
            self._collection(),
            subsequence_idx=0,
            canonical_tr_idx=canonical_tr_idx,
            target_resolution_hz=float(resolution),
            max_freq_hz=float(max_frequency),
            forbidden_bands=c_bands,
            peak_log10_threshold=None,
            peak_norm_scale=None,
            peak_eps=None,
            peak_prominence=None,
            compress_trains=bool(compress_trains),
        )
        if plot:
            # eps per band, mirroring sa_eps_for_band(): a nonzero vendor
            # limit is trusted as-is, a zero-tolerance band falls back to
            # SA_AEQ_K_GMAX * G_max.
            eps = [b[2] if b[2] > 0 else 0.08 * float(self.system.max_grad) for b in c_bands]
            self._plot_mech_resonances(record, c_bands, eps)
        return record

    @staticmethod
    def _plot_mech_resonances(record: dict, bands, eps) -> None:
        """Continuous spectrum, analytic envelope, TR-harmonic stems and the
        candidate lines the headless gate thresholds."""
        import matplotlib.pyplot as plt

        axes = ("gx", "gy", "gz")
        cand_f = np.asarray(record.get("candidate_freqs") or [], dtype=float)
        viol = np.asarray(record.get("candidate_violations") or [], dtype=int)
        cand_a = np.asarray(record.get("candidate_grad_amps") or [], dtype=float)

        fig, axs = plt.subplots(4, 1, figsize=(11, 11), sharex=True)

        # Row 0 -- the verdict itself. candidate_grad_amps is the quantity the
        # gate thresholds: the max over axes AND over the finite-outer-repeat
        # sidelobe probes around each harmonic. It is therefore >= any single
        # axis's coarse value below, which is why the pass/fail marking lives
        # here and not on the per-axis rows.
        vax = axs[0]
        for (lo, hi, *_), lim in zip(bands, eps):
            vax.axvspan(lo, hi, color="C3", alpha=0.10, linewidth=0)
            vax.hlines(lim, lo, hi, color="C3", linestyle="--", linewidth=1.4, label="_eps")
        if cand_f.size:
            bad = (viol != 0) if viol.size == cand_f.size else np.zeros(cand_f.shape, bool)
            vax.vlines(cand_f, 0.0, cand_a, color="0.75", linewidth=0.8)
            vax.plot(cand_f[~bad], cand_a[~bad], "o", ms=4.0, color="C2", label="candidate (pass)")
            if bad.any():
                vax.plot(cand_f[bad], cand_a[bad], "o", ms=7.0, color="C3", label="VIOLATION")
        vax.set_ylabel("verdict\n$A_{eq}$ (Hz/m)")
        vax.grid(True, alpha=0.3)
        vax.legend(loc="upper right", fontsize=8)

        for row, name in enumerate(axes):
            ax = axs[row + 1]
            for (lo, hi, *_), lim in zip(bands, eps):
                ax.axvspan(lo, hi, color="C3", alpha=0.10, linewidth=0)
                ax.hlines(lim, lo, hi, color="C3", linestyle="--", linewidth=1.2)

            env_f = np.asarray(record.get("envelope_freqs_hz") or [], dtype=float)
            env_a = np.asarray(record.get(f"envelope_amp_{name}") or [], dtype=float)
            if env_f.size and env_a.size:
                ax.plot(env_f, env_a, color="0.55", linewidth=1.0, label="analytic envelope")

            peak_f = np.asarray(record.get("analytical_peak_freqs") or [], dtype=float)
            peak_a = np.asarray(record.get(f"analytical_peak_amp_{name}") or [], dtype=float)
            if peak_f.size and peak_a.size:
                ax.vlines(peak_f, 0.0, peak_a, color="C0", linewidth=0.6, alpha=0.55, label="TR harmonics")

            # Coarse per-axis A_eq at the in-band harmonics: context for WHICH
            # axis drives row 0, not a verdict in itself (it excludes the
            # sidelobe probes that row 0's maximum may come from).
            axis_a = np.asarray(record.get(f"candidate_grad_amps_{name}") or [], dtype=float)
            if cand_f.size and axis_a.size:
                ax.plot(cand_f, axis_a, "o", ms=3.5, color="C2", label="in-band harmonic")

            ax.set_ylabel(f"{name}   $A_{{eq}}$ (Hz/m)")
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.legend(loc="upper right", fontsize=8)

        axs[-1].set_xlabel("Frequency (Hz)")
        # Candidates are enumerated per band, so a harmonic inside two
        # overlapping bands (a vendor ESP table repeating the same row for X
        # and Y is the usual cause) appears more than once. The duplicates
        # carry identical values, so counting DISTINCT lines is what the
        # caption should say.
        uniq = np.unique(np.round(cand_f, 6)) if cand_f.size else cand_f
        n_viol_uniq = (
            np.unique(np.round(cand_f[viol != 0], 6)).size if viol.size == cand_f.size else 0
        )
        peak = float(cand_a.max()) if cand_a.size else 0.0
        axs[0].set_title(
            f"Mechanical resonance — {uniq.size} candidate lines, "
            f"{n_viol_uniq} violating, max $A_{{eq}}$ {peak:.4g} Hz/m"
        )
        fig.tight_layout()

    def pns(
        self,
        *,
        tr_index: int | None = None,
        segment: int | None = None,
        model: str = "chronaxie",
        chronaxie_us: float | None = None,
        rheobase: float | None = None,
        alpha: float | None = None,
        hardware=None,
        thresholds=(80.0, 100.0),
        plot: bool = True,
    ):
        """Peripheral nerve stimulation response, for inspection only.

        Two nerve models are available. ``'chronaxie'`` is the Irnich
        rheobase/chronaxie form GE uses, and is the same arithmetic the
        interpreter runs at predownload. ``'safe'`` delegates to upstream's
        SAFE implementation, which needs Siemens hardware parameters.

        No verdict is returned: the threshold lines are drawn so the margin is
        visible, and the pass/fail decision stays with the C safety core.

        Parameters
        ----------
        tr_index, segment : int, optional
            Restrict the analysis to one TR or segment.
        model : {'chronaxie', 'safe'}, default 'chronaxie'
            Nerve model to use.
        chronaxie_us, rheobase, alpha : float, optional
            Irnich model parameters. Default to the matching attributes of
            the sequence's :class:`~pypulseq.Opts` when it carries them
            (:class:`pulserver.pypulseq.Opts` does).
        hardware : optional
            SAFE hardware description or Siemens ``.asc`` path, for
            ``model='safe'``. Defaults to ``system.pns_hardware`` when the
            sequence's :class:`~pypulseq.Opts` carries one.
        thresholds : sequence of float, default (80.0, 100.0)
            Percentage lines to draw.
        plot : bool, default True
            Draw the result.

        Returns
        -------
        tuple
            ``(pns_percent, t)`` where ``pns_percent`` has shape ``(N, 3)``.
        """
        if model == "safe":
            hardware = hardware if hardware is not None else getattr(self.system, "pns_hardware", None)
            if hardware is None:
                raise ValueError(
                    "model='safe' needs SAFE hardware parameters or a Siemens .asc path. Pass them "
                    "as hardware=, or set pns_hardware on the sequence's pulserver.pypulseq.Opts."
                )
            time_range = self._time_range(tr_index, segment, default=None)
            _, _, pns_components, t = self._scoped_view(tr_index, segment).calculate_pns(
                hardware, time_range=time_range, do_plots=plot
            )
            return 100.0 * pns_components, t

        if model != "chronaxie":
            raise ValueError(f"Unknown PNS model {model!r}; expected 'chronaxie' or 'safe'.")

        system = self.system
        chronaxie_us = chronaxie_us if chronaxie_us is not None else getattr(system, "chronaxie_us", None)
        rheobase = rheobase if rheobase is not None else getattr(system, "rheobase", None)
        alpha = alpha if alpha is not None else getattr(system, "alpha", None)
        if chronaxie_us is None or rheobase is None:
            raise ValueError(
                "chronaxie_us and rheobase are required. Pass them explicitly, or set them "
                "on the sequence's pulserver.pypulseq.Opts."
            )

        dt = float(self.system.grad_raster_time)
        gradients, t = self._sample_gradients(tr_index, segment, dt)
        pns_percent = chronaxie_pns(
            gradients,
            dt,
            chronaxie_us=float(chronaxie_us),
            rheobase=float(rheobase),
            alpha=float(alpha) if alpha is not None else 1.0,
        )

        if plot:
            self._plot_pns(pns_percent, t, thresholds)
        return pns_percent, t

    def _sample_gradients(self, tr_index, segment, dt) -> tuple[np.ndarray, np.ndarray]:
        """Sample the gradient waveforms on a uniform raster, in Hz/m."""
        time_range = self._time_range(tr_index, segment, default=None)
        gradients = self._seqplot.get_gradients(time_range=time_range)
        max_t = max(g.x[-1] for g in gradients if g is not None) - 1e-10

        if time_range is None:
            t = (np.arange(int(np.ceil(max_t / dt))) + 0.5) * dt
        else:
            span = min(time_range[1], max_t) - max(time_range[0], 0)
            t = max(time_range[0], 0) + (np.arange(int(np.ceil(span / dt))) + 0.5) * dt

        sampled = np.zeros((t.shape[0], 3))
        for axis, grad in enumerate(gradients[:3]):
            if grad is not None:
                sampled[:, axis] = grad(t)
        return sampled, t

    @staticmethod
    def _plot_pns(pns_percent: np.ndarray, t: np.ndarray, thresholds) -> None:
        """Draw per-axis and combined PNS against the requested threshold lines."""
        import matplotlib.pyplot as plt

        combined = np.sqrt((pns_percent**2).sum(axis=1))
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(t * 1e3, combined, color="C3", linewidth=1.8, label="combined")
        for axis, name in enumerate(("x", "y", "z")):
            ax.plot(t * 1e3, pns_percent[:, axis], linewidth=1.0, label=name)
        for threshold in sorted(float(v) for v in thresholds):
            ax.axhline(threshold, color="0.3", linestyle=":", linewidth=1.4)
            ax.annotate(f"{threshold:g}%", (t[0] * 1e3, threshold), va="bottom", fontsize=8, color="0.3")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("PNS (% of threshold)")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
        fig.tight_layout()

    # ------------------------------------------------------------------
    # Unsupported pypulseq API in fast-builder mode
    # ------------------------------------------------------------------

    def write(
        self,
        name,
        create_signature: bool = False,
        remove_duplicates: bool = False,
        check_timing: bool = False,
        v141_compat: bool = False,
    ):
        """Fast builder does not support writing sequence files."""
        del name, create_signature, remove_duplicates, check_timing, v141_compat
        raise NotImplementedError("pulserver.pypulseq.Sequence is a fast builder and does not implement write().")

    @classmethod
    def _unstructured(cls, system=None) -> "Sequence":
        """A sequence with no declared structure: the block-at-a-time builder.

        Private, and it has exactly two honest uses.

        **Parsing.** Reading a ``.seq`` back is the one case with nothing to
        declare: the blocks arrive one at a time in whatever order the file
        lists them, and there is no period to claim rows from.
        ``pulserver.io.read`` is the only caller in the package.

        **The reference implementation.** This path registers each event as it
        is reached, which is what defines the emitted file -- IDs are handed out
        in visit order and deduplication picks representatives by first
        occurrence. The template has to match it byte for byte, so the tests
        that prove it build the same sequence both ways and diff the bytes. That
        comparison needs this builder, and it is why the code stays rather than
        being deleted outright.

        What it is *not* is a way to author a sequence. Declaring the structure
        is what lets every library row be claimed up front, and that is the
        public contract.
        """
        token = _parsing.set(True)
        try:
            return cls(system=system)
        finally:
            _parsing.reset(token)

    def read(self, *_args, **_kwargs) -> None:
        """Fast builder does not support reading/parsing external sequence files."""
        raise NotImplementedError("pulserver.pypulseq.Sequence is a fast builder and does not implement read().")

    def set_block(self, _block_index: int, *args: SimpleNamespace | float) -> None:
        """Disable positional insertion/update in fast mode."""
        raise NotImplementedError(
            "pulserver.pypulseq.Sequence only supports sequential add_block(). "
            "Use pypulseq.Sequence for random block updates."
        )

    def get_block(self, block_index: int) -> SimpleNamespace:
        """Decode one block through the plain view.

        The block libraries of the fast builder are write-only, so this goes
        through :attr:`_seq`. Extensions are not part of the returned block;
        read them from :attr:`extensions`.
        """
        return self._seq.get_block(block_index)

    # ------------------------------------------------------------------
    # Internal fast block/event registration helpers
    # ------------------------------------------------------------------

    def _fast_set_block(self, *args: SimpleNamespace | float) -> None:
        """Direct-insert block registration: no dedup, no continuity checks, no trace.

        The block's ID is its position in the table, so nothing here needs to
        be told what it is.
        """
        events = block_to_events(*args)
        # A plain list, not an int32 array: this runs once per block, so the
        # array header would be the largest single allocation of the hot loop.
        # Everything that reads a row -- write(), remove_duplicates() -- indexes
        # or slices it; the views that need an upstream Sequence re-parse the
        # payload and build their own.
        new_block = [0, 0, 0, 0, 0, 0, 0]
        duration = 0
        extensions = []

        for event in events:
            if isinstance(event, float):
                if event > duration:
                    duration = event
                continue

            event_type = event.type
            if event_type == "rf":
                rf_id, _ = self._fast_register_rf(event)
                new_block[1] = rf_id
                event_duration = event.shape_dur + event.delay + event.ringdown_time

            elif event_type == "grad":
                grad_id, _ = self._fast_register_grad(event)
                new_block[_CHANNEL_SLOT[event.channel]] = grad_id
                event_duration = (
                    event.delay + math.ceil(event.tt[-1] / self.grad_raster_time - 1e-10) * self.grad_raster_time
                )

            elif event_type == "trap":
                new_block[_CHANNEL_SLOT[event.channel]] = self._fast_register_trap(event)
                event_duration = event.delay + event.rise_time + event.flat_time + event.fall_time

            elif event_type == "adc":
                adc_id, _ = self._fast_register_adc(event)
                new_block[5] = adc_id
                event_duration = event.delay + event.num_samples * event.dwell + event.dead_time

            elif event_type == "delay":
                event_duration = event.delay

            elif event_type in ("output", "trigger"):
                event_id = self._fast_register_control(event)
                extensions.append((self._ext_type_id("TRIGGERS"), event_id))
                event_duration = event.delay + event.duration

            elif event_type in ("labelset", "labelinc"):
                label_id = self._fast_register_label(event)
                extensions.append((self._ext_type_id(event_type.upper()), label_id))
                event_duration = 0

            elif event_type == "soft_delay":
                # Soft delays are intentionally ignored in this fast on-scanner path.
                continue

            elif event_type == "rf_shim":
                rf_shim_id = self._fast_register_rf_shim(event)
                extensions.append((self._ext_type_id("RF_SHIMS"), rf_shim_id))
                event_duration = 0

            elif event_type == "rot3D":
                rot_id = self._fast_register_rotation(event)
                extensions.append((self._ext_type_id("ROTATIONS"), rot_id))
                event_duration = 0

            else:
                raise ValueError(f"Unknown event type {event_type} passed to pulserver.pypulseq.Sequence.add_block().")

            if event_duration > duration:
                duration = event_duration

        if extensions:
            # A block carries a handful of extensions at most, so ordering them
            # by reference is a list sort, not a numpy round trip.
            extensions.sort(key=_extension_ref)
            ext_lib = self.extensions_library
            ext_find = ext_lib.find

            all_found = True
            extension_id = 0
            for ext_type, ext_ref in extensions:
                extension_id, found = ext_find((ext_type, ext_ref, extension_id))
                if not found:
                    all_found = False
                    break

            if not all_found:
                extension_id = 0
                for ext_type, ext_ref in extensions:
                    data = (ext_type, ext_ref, extension_id)
                    extension_id, found = ext_find(data)
                    if not found:
                        ext_lib.insert(extension_id, data)
            new_block[6] = extension_id

        tail = self._block_tail
        if tail is None:
            tail = self._block_tail = []
            self._duration_tail = []
            self._block_runs.append(tail)
            self._duration_runs.append(self._duration_tail)
        tail.append(new_block)
        self._duration_tail.append(float(duration))
        self._n_blocks += 1

    # -- per-event library rows -------------------------------------------
    #
    # Each builder answers ``(row, entry type, duration claimed)`` for one
    # event. They are the single place a payload is built from an event object,
    # shared by the first shot's slot construction and by every later shot's
    # cursor write, so a range cannot register a row the per-shot path would
    # have built differently. Shapes go through the id-keyed caches, which is
    # what keeps a re-rendered shot from recompressing a waveform it shares
    # with the first one -- and what makes a genuinely new waveform register a
    # new shape, in the same visit order the per-shot path would have.
    #
    # The cursor binds one of these to each slot at shot 0 (``_BulkSlot.build``)
    # rather than re-deciding per event: for a large protocol the type dispatch
    # alone runs into the tens of millions.

    def _row_rf(self, event):
        amplitude, shape_ids = self._compress_rf_cached(event)
        row = (amplitude, *shape_ids, event.center, event.delay, event.freq_ppm,
               event.phase_ppm, event.freq_offset, event.phase_offset)
        use = event.use[0] if event.use in _RF_USES else "u"
        duration = event.shape_dur + event.delay + event.ringdown_time
        return row, _RF_USE_CHAR_TO_CODE.get(use, 0), duration

    def _row_trap(self, event):
        row = (event.amplitude, event.rise_time, event.flat_time, event.fall_time, event.delay)
        return row, "t", event.delay + event.rise_time + event.flat_time + event.fall_time

    def _row_grad(self, event):
        amplitude, shape_ids = self._compress_grad_cached(event.waveform, event.tt)
        row = (amplitude, event.first, event.last, *shape_ids, event.delay)
        duration = event.delay + math.ceil(event.tt[-1] / self.grad_raster_time - 1e-10) * self.grad_raster_time
        return row, "g", duration

    def _row_adc(self, event):
        return self._adc_payload(event), "", event.delay + event.num_samples * event.dwell + event.dead_time

    def _row_label(self, event):
        return (event.value, self._get_label_idx(event.label)), "", 0.0

    def _row_control(self, event):
        event_type = ["output", "trigger"].index(event.type)
        channel = (["osc0", "osc1", "ext1"] if event_type == 0 else ["physio1", "physio2"]).index(event.channel)
        return (event_type + 1, channel + 1, event.delay, event.duration), "", event.delay + event.duration

    def _row_rotation(self, event):
        return tuple(event.rot_quaternion.as_quat(canonical=True, scalar_first=True).tolist()), "", 0.0

    def _row_rf_shim(self, event):
        values = np.stack((np.abs(event.shim_vector), np.angle(event.shim_vector)), axis=-1).ravel()
        return tuple(values.tolist()), "", 0.0

    def _row_builder(self, kind: str):
        """The row builder for one event type, bound to this sequence."""
        try:
            return getattr(self, _ROW_BUILDERS[kind])
        except KeyError:
            raise ValueError(
                f"Unknown event type {kind} passed to pulserver.pypulseq.Sequence.add_block()."
            ) from None

    def _event_row(self, event) -> tuple[tuple, str | int, float]:
        """``(row, entry type, duration)`` for one event, dispatched by type."""
        return self._row_builder(event.type)(event)

    # ------------------------------------------------------------------
    # Private direct-insert helpers (no find_or_insert, no dedup)
    # ------------------------------------------------------------------

    def _ext_type_id(self, name: str) -> int:
        """Numeric ID of extension *name*, resolved once per sequence."""
        type_id = self._ext_type_ids.get(name)
        if type_id is None:
            type_id = self.get_extension_type_ID(name)
            self._ext_type_ids[name] = type_id
        return type_id

    def _compress_rf_cached(self, event: SimpleNamespace):
        """``(amplitude, shape_ids)`` for an RF event, compressed at most once.

        Shared with the bulk path so a range of shots cannot register a
        payload the per-shot path would have built differently.

        A module that scales a pulse publishes the envelope it scaled and the
        factor, through ``shape_source``. That is worth honouring rather than
        compressing the scaled array: ``_compress_rf`` normalises by the
        envelope's own peak, so a flip-angle schedule moves the payload's
        amplitude and nothing else -- one registered magnitude shape for a
        whole variable-flip train instead of one per distinct flip angle. The
        two differ in the last bits (``max(|s * k|)`` is not exactly
        ``max(|s|) * k``), which the rounded write-time pass collapses, so the
        emitted file is the same either way.
        """
        signal = event.signal
        times = event.t
        source = getattr(event, "shape_source", None)
        scale = 1.0
        if source is not None:
            # A zero flip normalises to an all-zero envelope, which is a
            # genuinely different shape rather than a smaller number.
            signal, scale = source
            if scale == 0.0:
                signal, scale = event.signal, 1.0

        cached = self._rf_shape_cache.get(id(signal))
        if cached is not None and cached[0] is signal and cached[1] is times:
            return cached[2] * scale, cached[3]
        amplitude, shape_IDs = self._compress_rf(signal, times)
        self._rf_shape_cache[id(signal)] = (signal, times, amplitude, shape_IDs)
        return amplitude * scale, shape_IDs

    def _compress_grad_cached(self, waveform, times):
        """``(amplitude, shape_ids)`` for an arbitrary gradient; see above.

        Takes the samples rather than the event: a slot whose waveform moves
        resolves its shape from a :class:`_ShapeInventory` entry, which is a
        ``(samples, times)`` pair its module built at construction and never an
        event.
        """
        cached = self._grad_shape_cache.get(id(waveform))
        if cached is not None and cached[0] is waveform and cached[1] is times:
            return cached[2], cached[3]
        amplitude, shape_IDs = self._compress_grad(waveform, times)
        self._grad_shape_cache[id(waveform)] = (waveform, times, amplitude, shape_IDs)
        return amplitude, shape_IDs

    def _fast_register_rf(self, event: SimpleNamespace):
        amplitude, shape_IDs = self._compress_rf_cached(event)

        if not hasattr(event, "use"):
            raise ValueError('Parameter "use" is not optional since v1.5.0')
        use = event.use[0] if event.use in _RF_USES else "u"
        use_code = _RF_USE_CHAR_TO_CODE.get(use, 0)

        data = (
            amplitude,
            *shape_IDs,
            event.center,
            event.delay,
            event.freq_ppm,
            event.phase_ppm,
            event.freq_offset,
            event.phase_offset,
        )
        rf_id = self.rf_library.append(data, use_code)
        return rf_id, shape_IDs

    def _fast_register_grad(self, event: SimpleNamespace):
        amplitude, shape_IDs = self._compress_grad_cached(event.waveform, event.tt)
        data = (amplitude, event.first, event.last, *shape_IDs, event.delay)
        arb_id = self.arb_library.append(data)
        grad_id = self.grad_library.append((arb_id,), "g")
        return grad_id, shape_IDs

    # -- shape compression, the part that is worth caching -----------------

    def _compress_rf(self, signal, times):
        """``(amplitude, shape_ids)`` for one RF sample array; see ``_rf_shape_cache``."""
        mag = np.abs(signal)
        amplitude = np.max(mag)
        mag = mag / amplitude
        mag[np.isnan(mag)] = 0
        phase = np.angle(signal)
        phase[phase < 0] += 2 * np.pi
        phase /= 2 * np.pi

        shape_IDs = [0, 0, 0]
        mag_shape = compress_shape(mag)
        shape_IDs[0], _ = self.shape_library.find_or_insert(np.concatenate(([mag_shape.num_samples], mag_shape.data)))
        phase_shape = compress_shape(phase)
        shape_IDs[1], _ = self.shape_library.find_or_insert(
            np.concatenate(([phase_shape.num_samples], phase_shape.data))
        )
        if not (np.floor(times / self.rf_raster_time) == np.arange(len(times))).all():
            time_shape = compress_shape(times / self.rf_raster_time)
            shape_IDs[2], _ = self.shape_library.find_or_insert([time_shape.num_samples, *time_shape.data])
        return amplitude, shape_IDs

    def _compress_grad(self, waveform, times):
        """``(amplitude, shape_ids)`` for one gradient waveform; see ``_grad_shape_cache``."""
        amplitude = np.max(np.abs(waveform))
        if amplitude > 0:
            fnz = waveform[np.nonzero(waveform)[0][0]]
            amplitude *= np.sign(fnz) if fnz != 0 else 1

        shape_IDs = [0, 0]
        g = waveform / amplitude if amplitude != 0 else waveform
        c_shape = compress_shape(g)
        shape_IDs[0], _ = self.shape_library.find_or_insert(np.concatenate(([c_shape.num_samples], c_shape.data)))

        c_time = compress_shape(times / self.grad_raster_time)
        t_data = np.concatenate(([c_time.num_samples], c_time.data))
        if len(c_time.data) == 4 and np.allclose(c_time.data, [0.5, 1, 1, c_time.num_samples - 3]):
            pass  # standard raster, shape_IDs[1] stays 0
        elif len(c_time.data) == 3 and np.allclose(c_time.data, [0.5, 0.5, c_time.num_samples - 2]):
            shape_IDs[1] = -1
        else:
            shape_IDs[1], _ = self.shape_library.find_or_insert(t_data)
        return amplitude, shape_IDs

    def _fast_register_trap(self, event: SimpleNamespace) -> int:
        data = (event.amplitude, event.rise_time, event.flat_time, event.fall_time, event.delay)
        trap_id = self.trap_library.append(data)
        return self.grad_library.append((trap_id,), "t")

    def _adc_payload(self, event: SimpleNamespace) -> tuple:
        """The ADC library row for one event; shared with the bulk path."""
        shape_id = 0
        if (
            hasattr(event, "phase_modulation")
            and event.phase_modulation is not None
            and len(event.phase_modulation) > 0
        ):
            phase_shape = compress_shape(np.asarray(event.phase_modulation).flatten())
            shape_data = np.concatenate(([phase_shape.num_samples], phase_shape.data))
            shape_id, _ = self.shape_library.find_or_insert(shape_data)

        return (
            event.num_samples,
            event.dwell,
            max(event.delay, event.dead_time),
            event.freq_ppm,
            event.phase_ppm,
            event.freq_offset,
            event.phase_offset,
            shape_id,
            event.dead_time,
        )

    def _fast_register_adc(self, event: SimpleNamespace):
        data = self._adc_payload(event)
        adc_id = self.adc_library.append(data)
        return adc_id, int(data[7])

    def _fast_register_control(self, event: SimpleNamespace) -> int:
        event_type = ["output", "trigger"].index(event.type)
        event_channel = (["osc0", "osc1", "ext1"] if event_type == 0 else ["physio1", "physio2"]).index(event.channel)
        data = (event_type + 1, event_channel + 1, event.delay, event.duration)
        return self.trigger_library.append(data)

    def _get_label_idx(self, label: str) -> int:
        """Return 1-based int for *label*, auto-registering unknown strings."""
        if label not in self._label_registry:
            new_idx = max(self._label_registry_inv) + 1
            self._label_registry[label] = new_idx
            self._label_registry_inv[new_idx] = label
        return self._label_registry[label]

    def _fast_register_label(self, event: SimpleNamespace) -> int:
        data = (event.value, self._get_label_idx(event.label))
        lib = self.label_set_library if event.type == "labelset" else self.label_inc_library
        return lib.append(data)

    def _fast_register_rf_shim(self, event: SimpleNamespace) -> int:
        data = (np.abs(event.shim_vector), np.angle(event.shim_vector))
        data = np.stack(data, axis=-1).ravel()
        return self.rf_shim_library.append(tuple(data.tolist()))

    def _fast_register_rotation(self, event: SimpleNamespace) -> int:
        data = tuple(event.rot_quaternion.as_quat(canonical=True, scalar_first=True).tolist())
        return self.rotation_library.append(data)


def _extension_ref(extension: tuple[int, int]) -> int:
    """Reference ID of an ``(extension_type, reference)`` pair."""
    return extension[1]


#: Payload tuple order per extension type, mirroring
#: ``pulserver._core._module._DYNAMIC_FIELDS``.
_EXT_PAYLOAD_FIELDS = {
    "labelset": ("value",),
    "labelinc": ("value",),
    "rot3D": ("0", "1", "2", "3"),
}


def _ext_values(extensions, ordinal: int):
    """The payload entry belonging to one extension slot: its values tuple.

    A payload lists its extensions in block-event order rather than keyed, and
    the template recorded each slot's position in that order, so the two line up
    without searching.
    """
    try:
        return extensions[ordinal][1]
    except IndexError:
        return None


def _row_types(slots, tile: int = 1):
    """Per-row entry types for a claimed run, or ``""`` when nothing is typed."""
    types = [slot.data_type for slot in slots]
    if not any(types):
        return ""
    return np.array(types * tile, dtype=object)


def _slot_id(slot, pass_index: int) -> int:
    """The entry ID this slot's row has on one pass."""
    return slot.first_id if pass_index == 0 else int(slot.later_ids[pass_index - 1])


def _delay_duration(args) -> float | None:
    """How long a block of nothing but time lasts, or ``None`` if it holds more.

    A block written as a bare number, a ``make_delay`` or both owns no library
    row, which is what lets the template treat it differently from every other
    block: it can neither be misnumbered nor renumber anything else.
    """
    duration = 0.0
    for event in args:
        if isinstance(event, int | float):
            duration = max(duration, float(event))
        elif getattr(event, "type", None) in ("delay", "soft_delay"):
            duration = max(duration, float(getattr(event, "delay", 0.0)))
        else:
            return None
    return duration


def _is_pure_delay(args) -> bool:
    """Whether a block is nothing but time; see :func:`_delay_duration`."""
    return _delay_duration(args) is not None


def _index_lookup(mapping: dict[int, int]) -> np.ndarray:
    """``mapping`` as an array, so it can be applied to a whole column at once."""
    if not mapping:
        return np.zeros(1, dtype=np.int64)
    lookup = np.zeros(max(mapping) + 1, dtype=np.int64)
    lookup[list(mapping)] = list(mapping.values())
    return lookup


def _dedup_library_approx(
    lib: EventLibrary,
    digits: int | tuple[int, ...],
    *,
    remap: dict[int, np.ndarray] | None = None,
    values: np.ndarray | None = None,
    type_codes: np.ndarray | None = None,
) -> tuple[EventLibrary, np.ndarray]:
    """Rounded deduplication using hardcoded per-library rounding profiles.

    The ID mapping comes back as an array indexed by old ID, not a dict. Every
    consumer applies it to a whole column of IDs at once, so a dict would be
    built entry by entry -- ten million of them across the libraries of a large
    3D protocol -- only to be converted straight back into this array.

    ``remap`` renumbers whole columns — the shape and waveform references an
    entry holds — as part of the same pass. Those references used to be
    rewritten first, one ``EventLibrary.update`` per entry, and the library
    scanned afterwards; for a large 3D protocol that was a million-iteration
    Python loop over data this function was about to convert to a matrix
    anyway. Fusing them is exact rather than approximate: every rounding
    profile treats a reference column as an integer, so remapping before the
    rounding cannot change what the rounding produces.

    ``values`` and ``type_codes`` let a caller that has already built those
    arrays hand them straight over instead of having them rebuilt.
    """
    new_lib = EventLibrary(numpy_data=lib.numpy_data)

    append_only = isinstance(lib, _AppendOnlyLibrary)
    # An append-only library numbers its entries by position, so its IDs are
    # already 1..n in order and asking for the dict would build one for nothing.
    n_entries = len(lib) if append_only else len(lib.data)
    if not n_entries:
        return new_lib, np.zeros(1, dtype=np.int64)
    ids = range(1, n_entries + 1) if append_only else sorted(lib.data)

    if values is not None:
        matrix = values
    elif append_only:
        matrix = lib.matrix()
    else:
        # One conversion of the whole library into a float matrix. It doubles as
        # the validation the loop below used to do element by element: a ragged or
        # non-numeric library cannot become a 2-D float array, so numpy raises
        # exactly where a hand-rolled check would have.
        try:
            matrix = np.array([lib.data[old_id] for old_id in ids], dtype=float)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("_dedup_library_approx requires uniform, fully numeric payload rows") from exc
    if matrix.ndim != 2:
        raise RuntimeError("_dedup_library_approx requires uniform, fully numeric payload rows")

    if remap:
        matrix = matrix.copy()
        for column, lookup in remap.items():
            matrix[:, column] = lookup[matrix[:, column].astype(np.int64)]

    width = matrix.shape[1]
    if isinstance(digits, int):
        digits_tuple = tuple([digits] * width)
    else:
        if len(digits) < width:
            raise ValueError(f"Rounding profile length {len(digits)} is shorter than payload width {width}")
        digits_tuple = tuple(digits[:width])

    rounded = _round_sig_matrix(matrix, digits_tuple)

    if type_codes is None and append_only:
        type_codes = lib.type_codes()
    if type_codes is not None:
        type_ids = type_codes
    else:
        # Asked for only when it is the thing that answers: ``type`` is a dict
        # the size of the library, and the caller that most needs it -- the
        # gradient library, with an entry per driven axis per block -- supplies
        # its type codes directly and must not pay for one.
        lib_types = lib.type
        if lib_types:
            type_code: dict[str | int, int] = {}
            type_ids = np.asarray(
                [type_code.setdefault(lib_types.get(old_id, ""), len(type_code) + 1) for old_id in ids], dtype=float
            )
        else:
            # No per-entry types: every row would share one type code, and a
            # constant column cannot separate two rows, so the payload alone
            # is the key and no wider matrix is built.
            type_ids = None

    key_matrix = rounded if type_ids is None else np.column_stack([type_ids, rounded])
    first_idx, inverse = _group_identical_rows(key_matrix)
    order = np.argsort(first_idx)

    uniq_to_new_id = np.zeros(len(first_idx), dtype=np.int32)
    for new_id, uniq_idx in enumerate(order, start=1):
        row_idx = int(first_idx[uniq_idx])
        type_key = lib.type_at(row_idx) if append_only else lib.type.get(ids[row_idx], "")
        if lib.numpy_data:
            arr = rounded[row_idx].copy()
            arr.flags.writeable = False
            insert_data = arr
        else:
            insert_data = tuple(rounded[row_idx].tolist())
        new_lib.insert(new_id, insert_data, type_key)
        uniq_to_new_id[uniq_idx] = new_id

    # ID 0 is "no event" and must stay 0, which a zero-filled array gives for
    # free; a library with gaps in its IDs leaves those slots at 0 too.
    if append_only:
        lookup = np.zeros(n_entries + 1, dtype=np.int64)
        lookup[1:] = uniq_to_new_id[inverse]
    else:
        id_array = np.fromiter(ids, dtype=np.int64, count=n_entries)
        lookup = np.zeros(int(id_array.max()) + 1, dtype=np.int64)
        lookup[id_array] = uniq_to_new_id[inverse]
    return new_lib, lookup


def _row_hashes(rows: np.ndarray) -> np.ndarray:
    """One 64-bit FNV hash per row, mixed over the rows' raw bits.

    Hashing the bits rather than the values is what makes ``-0.0`` and ``0.0``
    hash apart, which they must: they are distinct payloads and the emitted file
    must not merge them.
    """
    bits = np.ascontiguousarray(rows).view(np.uint64)
    hashes = np.full(len(bits), np.uint64(0xCBF29CE484222325))
    prime = np.uint64(0x100000001B3)
    for column in range(bits.shape[1]):
        hashes ^= bits[:, column]
        hashes *= prime
    return hashes


def _rows_identical(rows: np.ndarray, others: np.ndarray) -> np.ndarray:
    """Per-row bitwise equality of two equally shaped float matrices."""
    return (np.ascontiguousarray(rows).view(np.uint64) == np.ascontiguousarray(others).view(np.uint64)).all(axis=1)


def _group_identical_rows(key_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Partition rows into bit-identical groups: ``(first_row_of_group, group_of_row)``.

    The obvious spelling is ``np.unique`` over a void view of the rows, which
    sorts n keys with a memcmp comparator and additionally argsorts them to
    produce the inverse. Both are wasted here: a library holds millions of
    entries drawn from a couple of thousand distinct payloads, so the rows are
    instead mixed down to one 64-bit hash each, the (tiny) set of distinct
    hashes is sorted, and every row is placed with a ``searchsorted``.

    Grouping by a hash is only a *guess*, so it is checked: every row is
    compared, bit for bit, against its group's representative, and a collision
    falls back to the exact void sort. Bitwise is the right comparison rather
    than a lenient one -- ``-0.0`` and ``0.0`` are distinct payloads to the
    void sort this replaces, and the emitted file must not change.
    """
    rows = np.ascontiguousarray(key_matrix)
    bits = rows.view(np.uint64)
    n, width = bits.shape

    if n:
        hashes = _row_hashes(rows)
        distinct = np.unique(hashes)
        inverse = np.searchsorted(distinct, hashes)
        # Scattering in reverse row order leaves each group holding its
        # earliest row, which is the representative deduplication must keep.
        first_idx = np.empty(len(distinct), dtype=np.int64)
        first_idx[inverse[::-1]] = np.arange(n - 1, -1, -1)

        representative = first_idx[inverse]
        for column in range(width):
            values = bits[:, column]
            if not np.array_equal(values, values[representative]):
                break
        else:
            return first_idx, inverse

    key_bytes = rows.view(np.dtype((np.void, rows.dtype.itemsize * width))).ravel()
    _, first_idx, inverse = np.unique(key_bytes, return_index=True, return_inverse=True)
    return first_idx, inverse


def _round_sig_matrix(matrix: np.ndarray, digits: tuple[int, ...]) -> np.ndarray:
    """Vectorized significant-digit rounding for 2D numeric matrices."""
    if matrix.ndim != 2:
        raise ValueError("_round_sig_matrix expects a 2D matrix")
    if len(digits) != matrix.shape[1]:
        raise ValueError(f"Rounding profile length {len(digits)} does not match payload width {matrix.shape[1]}")

    # Each column is rounded by exactly one of the two profiles, so each is
    # applied to its own columns rather than to the whole matrix and masked
    # afterwards. The significant-digit branch costs a log10 over everything it
    # touches, and most libraries want it on a single column out of five or ten.
    d = np.asarray(digits, dtype=float)
    significant = np.flatnonzero(d > 0)
    decimal = np.flatnonzero(d <= 0)

    def to_significant(block, exponents):
        mags = np.power(10.0, exponents - np.ceil(np.log10(np.abs(block) + 1e-12)))
        return np.round(block * mags) / mags

    def to_decimal(block, exponents):
        mags = np.power(10.0, -exponents)
        return np.round(block * mags) / mags

    # A profile that is all one kind -- the label and rotation libraries are --
    # rounds the matrix whole, with no column indexing to copy it first.
    if not decimal.size:
        return to_significant(matrix, d)
    if not significant.size:
        return to_decimal(matrix, d)

    out = np.empty(matrix.shape, dtype=float)
    out[:, significant] = to_significant(matrix[:, significant], d[significant])
    out[:, decimal] = to_decimal(matrix[:, decimal], d[decimal])

    return out
