"""Fast sequence helpers for production bridge execution."""

from __future__ import annotations

__all__ = ["Segment", "Sequence"]

import copy
import itertools
import math
from operator import itemgetter
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pypulseq as pp
from pypulseq.block_to_events import block_to_events
from pypulseq.compress_shape import compress_shape
from pypulseq.event_lib import EventLibrary
from pypulseq.supported_labels_rf_use import get_supported_labels

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

#: Libraries collapsed as a range is registered, rather than only at write time.
#:
#: These are exactly the libraries a *block row* points at. Their IDs are read
#: by position -- column 1 for RF, 2..4 for the gradient axes, 5 for the ADC --
#: so collapsing them renumbers nothing that is ordered by ID, and the emitted
#: file is unchanged.
#:
#: The five extension-referenced libraries are deliberately absent. A block
#: orders its extension chain by reference ID (``_fast_set_block`` sorts on it,
#: mirroring upstream), so today's strictly increasing references make one shot's
#: chain order every shot's. Collapsing them would leave each shot holding an
#: arbitrary set of small IDs whose sort order varies shot to shot -- which is a
#: different chain, and a different file. They stay one entry per event and are
#: collapsed once, at write time, where the whole scan is visible at once.
_INSERT_DEDUP_LIBRARIES = ("rf_library", "arb_library", "trap_library", "grad_library", "adc_library")


#: Shortest range worth batching. Below this the per-call numpy overhead
#: exceeds what it saves, and the extra runs it leaves in each library make the
#: later deduplication slower; the measured break-even is around eight shots.
_MIN_BULK_SHOTS = 8

#: Roughly how many library rows a range holds as Python tuples before
#: converting them to matrices and committing. The cursor buys its speed by
#: leaving each shot's payload in a list, and the whole point of a range is
#: protocols with millions of blocks -- so the buffer is bounded by rows rather
#: than by shots, and a group with more events per shot commits more often.
#: Chunking cannot change the emitted file: IDs are positions, so a shot lands
#: at the same index whether its chunk started at 0 or at 900k.
_BULK_CHUNK_SHOTS = 1 << 18

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


@dataclass
class _BulkSlot:
    """One event's entries in one library, across a whole range of shots.

    ``rows`` collects one payload tuple per shot as the cursor walks them --
    a list append, not a matrix write, because a row assignment into numpy
    costs an order of magnitude more than appending to a Python list and this
    runs once per event of the whole range. The list becomes ``payload``, one
    ``(n, width)`` matrix, in a single conversion at commit time.

    ``ids`` is filled in once the library has taken the rows and answered with
    an ID for each -- which is not necessarily one ID per shot, since a library
    that collapses duplicates hands the same ID back for every shot that
    repeated a payload. ``column`` names the block-row column that points at
    this entry (``None`` for a waveform an entry elsewhere refers to), and
    ``source`` names the slot whose IDs are this one's payload -- how a
    ``grad_library`` reference finds its trapezoid or arbitrary waveform.
    """

    library: str
    width: int
    rows: list = field(default_factory=list)
    column: int | None = None
    data_type: str | int = ""
    extension: str | None = None
    source: "_BulkSlot | None" = None
    block: int = 0
    kind: str = ""
    build: Any = None
    ids: np.ndarray | None = None
    payload: np.ndarray | None = None


#: Per event type: the library it registers into, and where each *dynamic*
#: payload field lands in that library's row. Everything not named here is
#: structure -- shape IDs, sample counts, dwell, delays, rise/flat/fall -- and
#: is read once off the template rather than off an event per shot.
#:
#: The row layouts are the ones the ``_row_*`` builders produce, and the field
#: names are the ones ``pulserver._core._module._block_payload`` publishes. The
#: two have to agree; :func:`_template_slot` is where that is checked.
_TEMPLATE_ROW_FIELDS: dict[str, tuple[str, dict[str, int]]] = {
    "rf": ("rf_library", {"amplitude": 0, "freq_ppm": 6, "phase_ppm": 7, "freq_offset": 8, "phase_offset": 9}),
    "trap": ("trap_library", {"amplitude": 0}),
    "grad": ("arb_library", {"amplitude": 0, "first": 1, "last": 2}),
    "adc": ("adc_library", {"freq_ppm": 3, "phase_ppm": 4, "freq_offset": 5, "phase_offset": 6}),
    "labelset": ("label_set_library", {"value": 0}),
    "labelinc": ("label_inc_library", {"value": 0}),
    "rot3D": ("rotation_library", {}),
}

#: Entry type each event registers under, without rendering its row. ``rf`` is
#: the exception -- its type is the ``use`` code, read off the event.
_TEMPLATE_ENTRY_TYPES = {
    "rf": 0, "trap": "t", "grad": "g", "adc": "",
    "labelset": "", "labelinc": "", "rot3D": "",
}

#: Event types a template block may hold and still take the fast path. A block
#: holding anything else is marked slow and its payload is refused, loudly --
#: silently falling back would be worse, because a payload cannot be turned
#: back into events (it carries amplitudes, not shapes).
_TEMPLATE_FAST_TYPES = frozenset(_TEMPLATE_ROW_FIELDS) | {"delay", "soft_delay"}


@dataclass
class _TemplateSlot:
    """One event of one template block: where its row goes and what moves in it.

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
    #: Where this slot's per-iteration values are collected.
    buffer: int = -1
    #: ``operator.itemgetter`` over the dynamic field names, so one C-level
    #: call lifts a whole iteration's values out of the payload.
    getter: Any = None
    #: The waveform slot a gradient reference names.
    source: Any = None
    #: IDs this slot's rows were given by the last flush.
    ids: Any = None


@dataclass
class _TemplateBlock:
    """One block position of the TR template."""

    slots: tuple[_TemplateSlot, ...]
    #: Slots carrying an extension, in the order the block's events were seen.
    ext_slots: tuple[int, ...]
    #: Longest event duration in the block; only bare delays can move it.
    static_duration: float
    fast: bool
    reason: str = ""


class _AppendOnlyLibrary(EventLibrary):
    """An event library that is only ever appended to, stored as rows not a dict.

    The fast builder's libraries are write-only: nothing reads an entry back
    until deduplication, which wants the whole library as a matrix anyway, and
    writing only ever sees the *deduplicated* result. Keeping them as
    ``{id: payload}`` therefore buys nothing and costs a dict slot per event --
    three million of them for a large 3D protocol -- so entries live in plain
    rows and the ID is the position.

    Entries are kept as ordered runs, which may be Python lists or matrices:
    :meth:`append` grows a list one payload at a time, while :meth:`reserve` and
    :meth:`extend` take a whole range of shots in one matrix, with no Python
    step per event. IDs are positions either way, so a run boundary is invisible
    to every reader.

    ``data`` and ``type`` still answer as dicts for anything that asks, built on
    demand, so the class stays a drop-in :class:`EventLibrary`.

    Nothing here maintains ``keymap``, the by-value reverse index
    ``EventLibrary.insert`` keeps. Nothing looks an entry up by value one call at
    a time, so carrying it would cost a tuple and a dict slot per event and never
    be read; what does look entries up -- :meth:`extend` and write-time
    deduplication -- compares a whole batch of rounded rows at once instead.

    Given a rounding profile, :meth:`extend` additionally collapses duplicates
    as they arrive, which is what keeps the libraries small rather than merely
    cheap: a 512-cubed GRE registers 4.5 million entries drawn from some four
    thousand distinct payloads, and holding all 4.5 million costs about half a
    gigabyte before write-time deduplication ever runs. Collapsing early is safe
    because it is a *refinement* of what write time does anyway -- the same
    rounding profile, the same bitwise tie-break, the same first-occurrence
    representative -- so the write pass sees fewer rows and reaches the same
    answer. Two details make that true and are easy to get wrong:

    * The entry *type* is part of the key. ``grad_library`` holds a one-field
      reference into either ``trap_library`` or ``arb_library`` and those number
      independently, so ``(3,)`` is ambiguous and keying on the payload alone
      would point blocks at the wrong waveform.
    * What is stored is the first occurrence's **original** payload, not its
      rounded key. Rounding is not quite idempotent across a decade boundary
      (``9.999...`` rounds up to ``10.0``, which has a different exponent), so
      storing the rounded value would let the write pass round a second time and
      land somewhere the single-pass baseline never would.

    :meth:`append`, the per-block path, deliberately does not do this: its
    callers register one row at a time, and rounding a row in Python costs more
    than the write-time pass saves. The two paths need not agree -- collapsing
    any subset of the entries leaves the write pass's grouping and its choice of
    representative unchanged.
    """

    def __init__(self, numpy_data: bool = False, dedup_profile: int | tuple[int, ...] | None = None):
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
        #: Rounding profile for insert-time collapsing, or None to only append.
        self._dedup_profile = dedup_profile
        #: Distinct keys registered so far, as rounded rows ordered by hash, with
        #: the hashes themselves and the entry ID each one resolves to.
        self._key_rows: np.ndarray | None = None
        self._key_hashes: np.ndarray | None = None
        self._key_ids: np.ndarray | None = None
        #: Entry type -> the numeric code standing in for it in a key row.
        self._key_type_codes: dict = {}
        #: Set when a hash collision or a ragged payload makes the index
        #: untrustworthy; the library then only appends, and write-time
        #: deduplication -- which never relied on this -- still collapses it.
        self._dedup_off = False

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

    def type_key_codes(self, values) -> np.ndarray:
        """Intern entry types as the numeric codes a key row carries.

        Any injective numbering does: a key is only ever compared against
        another key of this same library, and it has to agree with write-time
        deduplication only in which rows it keeps *apart*.
        """
        codes = self._key_type_codes
        return np.array([codes.setdefault(value, len(codes) + 1) for value in values], dtype=float)

    def extend(
        self,
        matrix: np.ndarray,
        data_type="",
        key_codes: np.ndarray | None = None,
        tile: int = 1,
    ) -> np.ndarray:
        """Register ``matrix``'s rows in order and return one ID per row.

        Rows that repeat one already registered -- at this library's rounding
        profile, and counting the entry type as part of the payload -- are not
        stored again; their ID is the first occurrence's. Without a profile, or
        once the index has been switched off, every row is stored and the IDs
        run consecutively, exactly as :meth:`reserve` would have given them.

        ``data_type`` is one type for the whole batch, or a sequence of types
        repeated ``tile`` times over it -- which is the shape a range of shots
        has, and which lets the types of the handful of rows actually stored be
        picked out without ever materialising one entry per row.
        ``key_codes`` is the per-row types already interned by
        :meth:`type_key_codes`; a caller whose types repeat on a fixed stride
        can tile that far more cheaply than a row-by-row pass can build it.
        """
        count = len(matrix)
        if not count:
            return np.zeros(0, dtype=np.int64)

        pattern = None if isinstance(data_type, str | int) else np.asarray(data_type, dtype=object)
        if pattern is not None and len(pattern) * tile != count:
            raise ValueError(f"{len(pattern)} types tiled {tile}x cannot cover {count} rows")

        if self._dedup_profile is None or self._dedup_off:
            slab, first_id = self.reserve(
                count, matrix.shape[1], data_type if pattern is None else np.tile(pattern, tile)
            )
            slab[:] = matrix
            return first_id + np.arange(count, dtype=np.int64)

        if key_codes is None and pattern is not None:
            key_codes = np.tile(self.type_key_codes(pattern.tolist()), tile)
        key = self._dedup_keys(matrix, key_codes)
        hashes = _row_hashes(key)
        ids = self._known_ids(key, hashes)
        if self._dedup_off:  # the lookup found a collision and gave up
            return self.extend(matrix, data_type, tile=tile)

        new = ids < 0
        if new.any():
            (new_rows,) = np.nonzero(new)
            new_key, new_hashes = key[new_rows], hashes[new_rows]
            # Group the newcomers among themselves, numbering the groups in the
            # order they first appear rather than in hash order: the ID a row
            # gets has to be the one a plain append-everything pass would have
            # left its first occurrence with.
            _, first, inverse = np.unique(new_hashes, return_index=True, return_inverse=True)
            inverse = inverse.reshape(-1)
            order = np.argsort(first)
            rank = np.empty(len(first), dtype=np.int64)
            rank[order] = np.arange(len(first), dtype=np.int64)

            if not _rows_identical(new_key, new_key[first[inverse]]).all():
                self._dedup_off = True
                return self.extend(matrix, data_type, tile=tile)

            keep = new_rows[first[order]]
            stored = data_type if pattern is None else pattern[keep % len(pattern)]
            slab, first_id = self.reserve(len(keep), matrix.shape[1], stored)
            slab[:] = matrix[keep]
            ids[new_rows] = first_id + rank[inverse]
            self._register_keys(
                key[keep], hashes[keep], first_id + np.arange(len(keep), dtype=np.int64)
            )
        return ids

    def _dedup_keys(self, matrix: np.ndarray, key_codes: np.ndarray | None) -> np.ndarray:
        """The rows' deduplication keys: the rounded payload, entry type first."""
        width = matrix.shape[1]
        profile = self._dedup_profile
        profile = tuple([profile] * width) if isinstance(profile, int) else tuple(profile[:width])
        rounded = _round_sig_matrix(np.ascontiguousarray(matrix, dtype=float), profile)
        if key_codes is None:
            return rounded
        return np.column_stack([key_codes, rounded])

    def _known_ids(self, key: np.ndarray, hashes: np.ndarray) -> np.ndarray:
        """ID of each row already registered; ``-1`` for one that is new."""
        out = np.full(len(key), -1, dtype=np.int64)
        known = self._key_hashes
        if known is None:
            return out
        if self._key_rows.shape[1] != key.shape[1]:
            # A library whose payload width changes cannot share one index; the
            # rf_shim library is the only one that can, and it is not indexed.
            self._dedup_off = True
            return out
        position = np.searchsorted(known, hashes)
        np.clip(position, 0, len(known) - 1, out=position)
        hit = known[position] == hashes
        if not hit.any():
            return out
        rows = self._key_rows[position[hit]]
        if not _rows_identical(key[hit], rows).all():
            self._dedup_off = True
            return out
        out[hit] = self._key_ids[position[hit]]
        return out

    def _register_keys(self, key_rows: np.ndarray, hashes: np.ndarray, ids: np.ndarray) -> None:
        """Add newly registered keys to the index, kept sorted for searchsorted."""
        if self._key_hashes is None:
            all_rows, all_hashes, all_ids = key_rows, hashes, ids
        else:
            all_rows = np.concatenate([self._key_rows, key_rows])
            all_hashes = np.concatenate([self._key_hashes, hashes])
            all_ids = np.concatenate([self._key_ids, ids])
        order = np.argsort(all_hashes, kind="stable")
        self._key_rows = np.ascontiguousarray(all_rows[order])
        self._key_hashes = all_hashes[order]
        self._key_ids = all_ids[order]

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
        # A library filled by add_range holds one run per shot -- hundreds of
        # them -- and deduplication asks this once per distinct payload, so the
        # run a position falls in is found by bisection rather than by walking
        # from the start every time.
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
        use_block_cache: bool = False,
        tr_struct=None,
        loop_size: int | None = None,
    ):
        super().__init__(system=system, use_block_cache=use_block_cache)
        # Every library the fast path appends to, in the order write() emits
        # them. They are write-only until deduplication -- see
        # _AppendOnlyLibrary -- so none of them is a dict.
        for name in _APPEND_ONLY_LIBRARIES:
            profile = _ROUNDING_PROFILES[name] if name in _INSERT_DEDUP_LIBRARIES else None
            setattr(self, name, _AppendOnlyLibrary(dedup_profile=profile))
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
        # Python rows from add_block, whole matrices from add_range. The dict
        # views and the matrix are built from these on demand.
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
        self._loop_size = int(loop_size) if loop_size is not None else None
        self._slot_rows: list[list] = []
        self._slot_blocks: list[list] = []
        self._buffered = 0
        if tr_struct is not None:
            self._build_template(tr_struct)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_block(self, *args: SimpleNamespace | float | dict) -> None:
        """Append a block assuming strictly sequential insertion.

        Two forms. The ordinary one takes the block's events, exactly as
        upstream does. The other takes a single **payload dict** from
        :meth:`pulserver.SequenceModule.payloads` — the numbers this shot moves,
        keyed the way a pulseq block is — which is only accepted when the
        sequence was given a ``tr_struct``: the template supplies everything the
        payload leaves out, so the block costs a few dict reads instead of a
        walk over its events.
        """
        if self._template is not None and len(args) == 1 and isinstance(args[0], dict):
            self._template_set_block(args[0])
        else:
            self._template_require_boundary()
            self._fast_set_block(*args)
        self.next_free_block_ID += 1

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
        blocks = []
        for block in tr_struct:
            slots: list[_TemplateSlot] = []
            ext_slots: list[int] = []
            duration = 0.0
            fast, reason = True, ""

            for event in block_to_events(*block):
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

                built = self._template_slot(event)
                for slot in built:
                    slot.block = len(blocks)
                    if slot.dynamic:
                        slot.getter = itemgetter(*[name for _, name in slot.dynamic])
                    if slot.extension is not None:
                        ext_slots.append(len(slots))
                    slots.append(slot)
                if len(built) == 2:
                    built[1].source = built[0]
                duration = max(duration, self._event_duration(event))

            blocks.append(
                _TemplateBlock(tuple(slots), tuple(ext_slots), duration, fast, reason)
            )

        self._template = tuple(blocks)
        self._template_cursor = 0
        # Slots grouped by library, in the order the template visits them --
        # which is the order the per-block path would have registered them in,
        # and so the order that decides which occurrence of a repeated payload
        # is the first one. Every slot that lands in a collapsed library gets a
        # buffer; the rest are registered the moment they are seen.
        self._library_slots: dict[str, list] = {}
        collected = 0
        for template_block in blocks:
            for slot in template_block.slots:
                if slot.library not in _INSERT_DEDUP_LIBRARIES:
                    continue
                self._library_slots.setdefault(slot.library, []).append(slot)
                if not slot.is_reference:
                    slot.buffer = collected
                    collected += 1
        self._slot_values: list[list] = [[] for _ in range(collected)]
        #: Block index the buffered run starts at, and how many TRs it holds.
        self._flush_base = 0
        self._flush_trs = 0
        # One buffer per *library*, not per slot. Rows land in it in call
        # order, which is the order the per-block path would have registered
        # them in -- and therefore the order that decides which occurrence of a
        # repeated payload is the first one. Buffering per slot instead would
        # hand each library its rows slot by slot and renumber the whole file.
        self._buffered = 0

    def _template_slot(self, event) -> list[_TemplateSlot]:
        """The template slots one event owns, with its row already rendered."""
        kind = event.type
        library, fields = _TEMPLATE_ROW_FIELDS[kind]
        # The entry type is read off the event rather than off a rendered row:
        # rendering registers the event's *shapes*, and shape IDs are handed out
        # in visit order, so doing it here would number the template's shapes
        # ahead of any preparation block preceding the first TR.
        data_type = _TEMPLATE_ENTRY_TYPES[kind]
        if kind == "rf":
            use = event.use[0] if getattr(event, "use", None) in _RF_USES else "u"
            data_type = _RF_USE_CHAR_TO_CODE.get(use, 0)
        dynamic = tuple(sorted((index, name) for name, index in fields.items()))
        if kind == "rot3D":
            # A quaternion is dynamic end to end; there is no static part.
            dynamic = tuple((index, str(index)) for index in range(4))

        if kind in ("labelset", "labelinc"):
            return [_TemplateSlot(library, None, kind.upper(), data_type, event, None, dynamic, "")]
        if kind == "rot3D":
            return [_TemplateSlot(library, None, "ROTATIONS", data_type, event, None, dynamic, "")]
        if kind == "rf":
            return [_TemplateSlot(library, 1, None, data_type, event, None, dynamic, "rf")]
        if kind == "adc":
            return [_TemplateSlot(library, 5, None, data_type, event, None, dynamic, "adc")]

        # A gradient costs two rows: the waveform, then the reference the block
        # column actually points at.
        key = _CHANNEL_KEY[event.channel]
        waveform = _TemplateSlot(library, None, None, data_type, event, None, dynamic, key)
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
        return 0.0

    def _template_require_boundary(self) -> None:
        """A non-payload block may only land where a template pass has finished.

        A payload cannot be turned back into events -- it carries amplitudes,
        not shapes -- so there is no recovering from a cursor that has drifted.
        Refusing loudly at the boundary is the only safe answer; a preparation
        or a recovery delay between TRs is exactly where one belongs anyway.
        """
        if self._template is None:
            return
        if self._template_cursor:
            raise RuntimeError(
                f"add_block() with events landed {self._template_cursor} block(s) into the TR "
                f"template (of {len(self._template)}). Blocks outside the template may only be "
                "added between complete passes over it; pass this block's payload instead, or "
                "include it in tr_struct."
            )
        # Events register into the libraries immediately, so anything still
        # buffered has to be registered first or the IDs interleave in the wrong
        # order -- a preparation block would take an ID belonging to the TR
        # before it, and every event ID downstream would shift.
        if self._buffered:
            self._flush_template()

    def _template_set_block(self, payload: dict) -> None:
        """Register one block from its payload, against the template's structure."""
        position = self._template_cursor
        template = self._template[position]
        if not template.fast:
            raise NotImplementedError(
                f"TR template block {position} cannot be built from a payload: {template.reason}. "
                "Add this block with its events instead."
            )

        block_index = self._n_blocks
        if not self._buffered and position == 0:
            # Where this buffered run starts. Taken here rather than at the end
            # of the previous flush, because blocks outside the template land
            # in between and move the table on.
            self._flush_base = block_index
        if self._loop_size is not None and block_index >= self._loop_size * len(self._template):
            raise RuntimeError(
                f"TR template loop_size={self._loop_size} allows "
                f"{self._loop_size * len(self._template)} blocks; block {block_index + 1} exceeds it."
            )

        new_block = [0, 0, 0, 0, 0, 0, 0]
        duration = template.static_duration
        if "duration" in payload:
            duration = max(duration, float(payload["duration"]))
        extensions = payload.get("ext", ())
        ext_refs: list[tuple[int, int]] = []

        for index, slot in enumerate(template.slots):
            if slot.buffer >= 0:
                # Only the numbers this iteration moves. The rest of the row is
                # fixed by the template and tiled back on at flush; carrying it
                # per block would be one constant copied a million times.
                values = payload.get(slot.key) if slot.key else _ext_values(extensions, template, index)
                self._slot_values[slot.buffer].append(slot.getter(values))
                self._buffered += 1
            elif slot.is_reference:
                # A gradient reference's payload *is* the waveform's ID, so
                # there is nothing per-iteration to collect at all.
                self._buffered += 1
            else:
                # Extension-referenced libraries are never collapsed on insert,
                # so their IDs are settled the moment the row is appended --
                # which is what lets the chain be built here, exactly as
                # _fast_set_block builds it.
                if slot.row is None:
                    slot.row = self._row_builder(slot.event.type)(slot.event)[0]
                row = list(slot.row)
                values = _ext_values(extensions, template, index)
                if values is not None:
                    for position_in_row, name in slot.dynamic:
                        if name in values:
                            row[position_in_row] = values[name]
                ref = getattr(self, slot.library).append(tuple(row), slot.data_type)
                ext_refs.append((self._ext_type_id(slot.extension), ref))

        if ext_refs:
            ext_refs.sort(key=_extension_ref)
            new_block[6] = self._chain_extensions(ext_refs)

        tail = self._block_tail
        if tail is None:
            tail = self._block_tail = []
            self._duration_tail = []
            self._block_runs.append(tail)
            self._duration_runs.append(self._duration_tail)
        tail.append(new_block)
        self._duration_tail.append(float(duration))
        self._n_blocks += 1
        self._block_rows = None

        self._template_cursor = (position + 1) % len(self._template)
        if self._template_cursor == 0:
            self._flush_trs += 1
            # Only ever at a TR boundary: the buffered run has to be a whole
            # number of passes for a slot's block positions to be one stride.
            if self._buffered >= _TEMPLATE_FLUSH_ROWS:
                self._flush_template()

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

    def _flush_template(self) -> None:
        """Register every buffered row and patch the block table with its ID.

        Buffering is what lets a row be *collapsed* rather than appended: the
        rounded index is vectorised and wants a batch. IDs stay in visit order
        because a buffer is filled in call order and the batches go out in
        order, so the write-time pass still picks the same representatives.

        Waveforms settle before the gradient references that name them -- a
        reference's payload *is* an ID -- which is the same single round of
        dependency ``add_range`` had.
        """
        if not self._buffered:
            return
        rows = self._raw_block_row_matrix()
        length = len(self._template)
        count = self._flush_trs
        base = self._flush_base

        # Waveforms settle before the references that name them: a reference's
        # payload is the waveform's ID. One round of dependency, never two.
        order = [name for name in self._library_slots if name != "grad_library"]
        if "grad_library" in self._library_slots:
            order.append("grad_library")

        for name in order:
            slots = self._library_slots[name]
            stride = len(slots)
            matrices = [self._slot_matrix(slot, count) for slot in slots]
            width = matrices[0].shape[1]
            batch = matrices[0] if stride == 1 else np.stack(matrices, axis=1).reshape(-1, width)

            library = getattr(self, name)
            types = [slot.data_type for slot in slots]
            if any(types):
                pattern = np.array(types, dtype=object)
                codes = np.tile(library.type_key_codes(types), count)
            else:
                pattern, codes = "", None
            ids = library.extend(batch, pattern, key_codes=codes, tile=count)

            for position, slot in enumerate(slots):
                slot_ids = ids[position::stride]
                slot.ids = slot_ids
                if slot.column is not None:
                    # Every iteration puts this slot's block at the same offset
                    # inside the TR, so where the IDs go is a stride, not a list
                    # of positions somebody had to record per row.
                    rows[base + slot.block : base + count * length : length, slot.column] = slot_ids

        # The patched matrix becomes the block table.
        self._block_runs = [rows]
        self._block_tail = None
        self._block_rows = rows

        for held in self._slot_values:
            held.clear()
        self._buffered = 0
        self._flush_trs = 0

    def _slot_matrix(self, slot: _TemplateSlot, count: int) -> np.ndarray:
        """One slot's rows for the buffered run: the static row, tiled, retuned.

        This is where knowing the structure up front pays. Everything the
        template fixed -- shape IDs, dwell, delays, rise/flat/fall -- is one row
        broadcast over the whole run, and only the columns a payload actually
        moved are written. A gradient reference has no static part at all: its
        column is the waveform's IDs.
        """
        if slot.is_reference:
            return slot.source.ids.reshape(-1, 1).astype(float)

        if slot.row is None:
            slot.row = self._row_builder(slot.event.type)(slot.event)[0]
        matrix = np.tile(np.asarray(slot.row, dtype=float), (count, 1))
        if slot.buffer >= 0 and slot.dynamic:
            values = np.asarray(self._slot_values[slot.buffer], dtype=float)
            if len(slot.dynamic) == 1:
                matrix[:, slot.dynamic[0][0]] = values
            else:
                for position, (column, _) in enumerate(slot.dynamic):
                    matrix[:, column] = values[:, position]
        return matrix

    def add_range(self, *items, **states) -> Sequence:
        """Append a repeating group of blocks for a whole range of shots at once.

        The per-shot loop a plugin writes is a fixed chronology with a few
        numbers changing::

            for ky in range(ny):
                excitation.set_state(phase_offset_rad=phases[ky]).add_to(seq)
                seq.add_block(te_delay)
                readout.set_state(lin_idx=ky, phase_offset_rad=phases[ky]).add_to(seq)
                seq.add_block(tr_delay)

        which is the same thing written as one call, with arrays where the
        scalars were::

            seq.add_range(
                (excitation, {"phase_offset_rad": phases}),
                te_delay,
                (readout, {"lin_idx": np.arange(ny), "phase_offset_rad": phases}),
                tr_delay,
            )

        The group is the unit, not the module: events are registered in visit
        order, so all shots of one module cannot be emitted before the next
        module's. Order is preserved exactly, and so is the emitted file.

        Parameters
        ----------
        *items
            The chronology of one shot. Each item is a
            :class:`~pulserver.SequenceModule`, a ``(module, states)`` pair
            whose ``states`` dict is forwarded to ``set_state``, a plain event
            or tuple of events, or ``None`` (skipped, so an optional delay
            needs no branch at the call site).
        **states
            Shot-varying values shared by every module item, for the common
            case of one module in the group.

        Returns
        -------
        Sequence
            ``self``.

        Notes
        -----
        The first shot is rendered in full and *is* the group's structure: how
        many blocks, which events, which library each entry lands in. Every
        later shot is rendered too, but only its numbers are kept — they are
        collected as plain rows, with no library insertion, no extension-chain
        lookup and no per-block Python list. A shot whose structure differs from
        the first one's abandons the batch and replays the whole group block by
        block, so the call is always correct and never slower than writing it
        out by hand.

        The rows a range registers are also collapsed as they go: a scan whose
        every shot plays the same readout at a different phase-encode amplitude
        leaves one library entry per distinct amplitude, not one per shot. The
        tolerances are the ones :meth:`remove_duplicates` would have applied at
        write time, so this changes how much memory the sequence occupies while
        it is being built and nothing at all about the file.

        See Also
        --------
        add_block : the single-block form this batches.
        """
        from ._bulk import expand_states, infer_length

        entries = []
        for item in items:
            if item is None:
                continue
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], dict):
                entries.append((item[0], dict(item[1])))
            elif hasattr(item, "set_state"):
                entries.append((item, dict(states)))
            else:
                entries.append((item, None))

        length = None
        for _, item_states in entries:
            if item_states:
                length = infer_length(item_states)
                if length is not None:
                    break
        if length is None:
            length = infer_length(states) or 1

        if length < _MIN_BULK_SHOTS:
            return self._add_range_by_shot(entries, length)

        expanded = [
            (module, None if item_states is None else expand_states(length, item_states))
            for module, item_states in entries
        ]
        return self._add_range_bulk(entries, expanded, length)

    def _add_range_by_shot(self, entries, length: int, start: int = 0) -> Sequence:
        """Fallback: replay the group one shot at a time, exactly as a loop would.

        ``start`` skips shots a bulk chunk already committed, so a group that
        turns out to move its structure part-way through does not have to undo
        what was correct.
        """
        from ._bulk import expand_states

        expanded = [
            (module, None if item_states is None else expand_states(length, item_states))
            for module, item_states in entries
        ]
        for index in range(start, length):
            for module, item_states in expanded:
                if item_states is None:
                    events = module if isinstance(module, tuple) else (module,)
                    self.add_block(*events)
                    continue
                module.set_state(**{name: values[index] for name, values in item_states.items()})
                for block in module:
                    self.add_block(*block)
        return self

    def remove_duplicates(self, in_place: bool = False) -> Sequence:
        """Remove duplicates with hardcoded rounded profiles per library.

        Unlike upstream pypulseq, this also compacts extension-related
        libraries and canonicalizes extension linked lists so identical chains
        are shared across blocks.

        This is the pass that decides the file. :meth:`add_range` has usually
        collapsed the block-referenced libraries already, at the same tolerances
        (:data:`_ROUNDING_PROFILES`), but that is an optimisation and this does
        not assume it: a sequence built entirely through :meth:`add_block`
        arrives here with one entry per event and comes out the same way as one
        that arrived pre-collapsed.

        Parameters
        ----------
        in_place:
            If ``True``, deduplicate current instance; otherwise return a copy.
        """
        if self._buffered:
            self._flush_template()
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

        The table is kept as ordered runs -- Python rows from ``add_block``,
        whole matrices from :meth:`add_range` -- and this dict is built from
        them only when something asks. Neither writing nor deduplication does:
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
        if self._buffered:
            self._flush_template()
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

    # ------------------------------------------------------------------
    # Bulk registration
    # ------------------------------------------------------------------

    def _add_range_bulk(self, entries, expanded, length: int) -> Sequence:
        """Register ``length`` shots of a group through the cursor.

        The first shot is rendered and walked by exactly the dispatch
        :meth:`_fast_set_block` uses, so every payload is built by the code that
        already works. That walk *is* the structure: it fixes the block count,
        the event order and which library row each event owns, and it claims one
        row per shot in each library it touches. Every later shot is rendered too
        -- there is no declaration of what varies and no module has to publish
        one -- but the walk over it only appends numbers to those rows.

        The range is committed in chunks of :data:`_BULK_CHUNK_SHOTS`, which is
        what keeps a 3.7-million-block protocol inside memory: a chunk's rows
        are Python tuples until it commits, and holding the whole scan's worth
        of them at once would cost more than the sequence itself. Chunking is
        invisible in the emitted file because IDs are positions either way --
        shot ``i`` of a group using ``k`` entries of a library lands at
        ``base + i * k + j`` whether ``base`` advanced once or a hundred times.
        """
        shot = self._render_shot(expanded, 0)
        if not shot:
            return self

        blocks_per_shot = len(shot)
        template = list(self._bulk_slots(shot))
        chunk_size = max(1, _BULK_CHUNK_SHOTS // max(1, len(template)))

        start = 0
        while start < length:
            stop = min(start + chunk_size, length)
            # A waveform slot is always yielded before the reference pointing at
            # it, so each chunk's copy of a reference can be re-pointed at this
            # chunk's copy of its waveform as the list is built.
            fresh: dict[int, _BulkSlot] = {}
            slots: list[_BulkSlot] = []
            for original in template:
                slot = replace(
                    original, rows=[], ids=None, payload=None,
                    source=None if original.source is None else fresh[id(original.source)],
                )
                fresh[id(original)] = slot
                slots.append(slot)
            # A gradient's block column points at a grad_library entry whose own
            # payload is the ID of the waveform entry; only the waveform carries
            # a per-shot row, so only those slots are on the cursor.
            writers = [slot for slot in slots if slot.source is None]

            durations: list[float] = []
            for index in range(start, stop):
                rendered = shot if index == 0 else self._render_shot(expanded, index)
                if not self._cursor_write(writers, rendered, blocks_per_shot, durations):
                    # Nothing of this chunk is committed yet, so the shots it
                    # covers -- and every shot after them -- can still go in
                    # one at a time, which is where a module that moves its
                    # structure belongs.
                    return self._add_range_by_shot(entries, length, start=start)
            self._commit_bulk(slots, durations, blocks_per_shot, stop - start)
            start = stop
        return self

    def _render_shot(self, expanded, index: int) -> list[tuple]:
        """One shot's blocks: each module re-stated at ``index``, in group order.

        Goes to ``_rendered_blocks`` rather than iterating the module, which is
        the same snapshot without the iterator protocol in between -- worth it
        only because this runs once per module per shot.
        """
        blocks: list[tuple] = []
        for module, item_states in expanded:
            if item_states is None:
                blocks.append(tuple(module) if isinstance(module, tuple) else (module,))
                continue
            module.set_state(**{name: values[index] for name, values in item_states.items()})
            blocks.extend(module._rendered_blocks())
        return blocks

    def _cursor_write(self, writers, blocks, blocks_per_shot: int, durations: list) -> bool:
        """Collect one shot's payloads into the slots the first shot's walk laid out.

        This is the whole per-shot cost of a range: one pass over the events,
        one list append each. No library insertion (every row of the chunk is
        registered together, once, by :meth:`_commit_bulk`), no extension-chain
        lookup (the chain is structural), no per-block row list, no block-ID
        bookkeeping.

        Returns ``False`` the moment the shot stops matching the first one --
        a different block count, a different event order, a different RF
        ``use`` or a payload of a different width -- because a cursor can only
        move numbers, never structure. Durations are recomputed rather than
        assumed: they cost one comparison per event here, and assuming them
        would be the one way a structural change could pass unnoticed.
        """
        if len(blocks) != blocks_per_shot:
            return False
        position = 0
        n_writers = len(writers)
        append_duration = durations.append
        for block in blocks:
            duration = 0.0
            # block_to_events only ever rewrites a one-argument block that is a
            # whole namespace; anything else it hands straight back, and this
            # runs once per block of the range.
            events = block_to_events(*block) if len(block) == 1 else block
            for event in events:
                kind = getattr(event, "type", None)
                if kind is None:  # a bare float delay
                    if event > duration:
                        duration = event
                    continue
                if kind == "delay":
                    if event.delay > duration:
                        duration = event.delay
                    continue
                if kind == "soft_delay":
                    continue
                if position >= n_writers:
                    return False
                slot = writers[position]
                position += 1
                if slot.kind != kind:
                    return False
                row, data_type, event_duration = slot.build(event)
                if data_type != slot.data_type or len(row) != slot.width:
                    return False
                slot.rows.append(row)
                if event_duration > duration:
                    duration = event_duration
            append_duration(duration)
        return position == n_writers

    def _commit_bulk(self, slots, durations, blocks_per_shot: int, length: int) -> None:
        """Register every library's entries and write the finished payloads in.

        Entries are handed to each library one batch per library, interleaved so
        that a group using ``k`` entries of a library offers slot ``j`` of shot
        ``i`` as row ``i * k + j`` -- the order the per-shot loop would have
        registered them in, which is what fixes the IDs that come back.

        A gradient reference's payload *is* an ID, so the library holding the
        waveform it names has to have spoken first. That is one round of
        dependency, never two -- a waveform names a shape, and shapes are
        registered while the shot is being rendered -- so the batches go out in
        two passes rather than through a general sort.
        """
        per_library: dict[str, list[_BulkSlot]] = {}
        for slot in slots:
            # One conversion per slot, not one row assignment per shot: the
            # cursor left every shot's row in a Python list on purpose.
            slot.payload = (
                np.zeros((length, 1)) if slot.source is not None else np.array(slot.rows, dtype=float)
            )
            per_library.setdefault(slot.library, []).append(slot)

        deferred: dict[str, list[_BulkSlot]] = {}
        for name, library_slots in per_library.items():
            if any(slot.source is not None for slot in library_slots):
                deferred[name] = library_slots
            else:
                self._register_slots(name, library_slots, length)
        for name, library_slots in deferred.items():
            for slot in library_slots:
                if slot.source is None:
                    raise RuntimeError(f"{name} mixes referring and self-contained entries in one range")
                if slot.source.ids is None:
                    raise RuntimeError(f"{name} refers to {slot.source.library}, which was registered after it")
                slot.payload[:, 0] = slot.source.ids
            self._register_slots(name, library_slots, length)

        rows, block_durations, _ = self._reserve_blocks(length * blocks_per_shot)
        for slot in slots:
            if slot.column is not None:
                rows[slot.block::blocks_per_shot, slot.column] = slot.ids
        block_durations[:] = durations

        self._bulk_extensions(slots, rows, blocks_per_shot, length)

    def _register_slots(self, name: str, library_slots, length: int) -> None:
        """Hand one library every row this chunk owes it, and hand back the IDs.

        The rows go out interleaved -- shot 0's entries in slot order, then shot
        1's -- because that is the order the per-shot path would have registered
        them in, and therefore the order that decides which occurrence of a
        repeated payload is the first one.
        """
        library = getattr(self, name)
        stride = len(library_slots)
        if stride == 1:
            batch = library_slots[0].payload
        else:
            width = library_slots[0].payload.shape[1]
            batch = np.stack([slot.payload for slot in library_slots], axis=1).reshape(-1, width)

        types = [slot.data_type for slot in library_slots]
        if any(types):
            # One shot's types are every shot's, so both the stored types and
            # the key's type column are that one pattern tiled, never a value
            # looked up per row.
            pattern = np.array(types, dtype=object)
            codes = np.tile(library.type_key_codes(types), length)
        else:
            pattern, codes = "", None

        ids = library.extend(batch, pattern, key_codes=codes, tile=length)
        for offset, slot in enumerate(library_slots):
            slot.ids = ids[offset::stride]

    def _bulk_slots(self, blocks):
        """Yield one slot per registered event, in the order add_block visits them.

        The first shot is walked here for its *shape* -- the row width and the
        entry type each event needs -- which is also what pins the shapes into
        the writer's caches before any shot is written.
        """
        for block_index, block in enumerate(blocks):
            for event in block_to_events(*block):
                if isinstance(event, float) or event.type in ("delay", "soft_delay"):
                    continue
                build = self._row_builder(event.type)
                row, data_type, _ = build(event)
                for slot in self._bulk_event(event, len(row), data_type):
                    slot.block = block_index
                    slot.build = build
                    yield slot

    def _bulk_event(self, event, width: int, data_type) -> list[_BulkSlot]:
        """One event -> the (empty) library entries it will register per shot.

        Structure only: which library, how wide a row, which block column or
        extension points at it. Every shot's actual row, the first one's
        included, arrives later through :meth:`_cursor_write`.
        """
        kind = event.type

        if kind == "rf":
            return [_BulkSlot("rf_library", width, column=1, data_type=data_type, kind=kind)]

        if kind in ("trap", "grad"):
            waveform = _BulkSlot(
                "trap_library" if kind == "trap" else "arb_library", width, data_type=data_type, kind=kind
            )
            # A gradient costs two entries: the waveform, then the reference to
            # it that the block row actually points at.
            reference = _BulkSlot("grad_library", 1, column=_CHANNEL_SLOT[event.channel],
                                  data_type=data_type, source=waveform, kind=kind)
            return [waveform, reference]

        if kind == "adc":
            return [_BulkSlot("adc_library", width, column=5, kind=kind)]

        if kind in ("labelset", "labelinc"):
            name = "label_set_library" if kind == "labelset" else "label_inc_library"
            return [_BulkSlot(name, width, extension=kind.upper(), kind=kind)]

        if kind in ("output", "trigger"):
            return [_BulkSlot("trigger_library", width, extension="TRIGGERS", kind=kind)]

        if kind == "rot3D":
            return [_BulkSlot("rotation_library", width, extension="ROTATIONS", kind=kind)]

        return [_BulkSlot("rf_shim_library", width, extension="RF_SHIMS", kind=kind)]

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
        amplitude, shape_ids = self._compress_grad_cached(event)
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
                f"Unknown event type {kind} passed to pulserver.pypulseq.Sequence.add_range()."
            ) from None

    def _event_row(self, event) -> tuple[tuple, str | int, float]:
        """``(row, entry type, duration)`` for one event, dispatched by type."""
        return self._row_builder(event.type)(event)

    def _bulk_extensions(self, slots, rows, blocks_per_shot: int, length: int) -> None:
        """Build every shot's extension chains, in the order add_block would.

        A chain node is ``(type, reference, next)``, and the reference is a
        fresh library entry for every shot -- this path never collapses
        identical events -- so no two shots can share a node and the IDs run
        consecutively without a lookup. They do *not* run consecutively per
        level, though: a shot finishes all its chains before the next shot
        starts, so with ``levels`` nodes per shot the node for shot ``i`` at
        chain position ``offset`` is ``first + i * levels + offset``. Getting
        that stride wrong renumbers the extension column and changes the file.
        """
        per_block: dict[int, list[_BulkSlot]] = {}
        for slot in slots:
            if slot.extension is not None:
                per_block.setdefault(slot.block, []).append(slot)
        if not per_block:
            return

        library = self.extensions_library
        first_id = library.next_free_ID
        shot_index = np.arange(length, dtype=np.int64)
        # Ordering a block's extensions by reference is what add_block does;
        # every slot advances by the same stride, so one shot's order is every
        # shot's order.
        chains = [(block, sorted(block_slots, key=lambda slot: int(slot.ids[0])))
                  for block, block_slots in sorted(per_block.items())]
        levels = sum(len(block_slots) for _, block_slots in chains)

        payloads: list[list[tuple]] = [[] for _ in range(length)]
        offset = 0
        for block_index, block_slots in chains:
            chain = np.zeros(length, dtype=np.int64)
            for slot in block_slots:
                node_ids = first_id + shot_index * levels + offset
                type_id = float(self._ext_type_id(slot.extension))
                for shot, (ref, nxt) in enumerate(zip(slot.ids.tolist(), chain.tolist(), strict=True)):
                    payloads[shot].append((type_id, float(ref), float(nxt)))
                chain = node_ids
                offset += 1
            rows[block_index::blocks_per_shot, 6] = chain

        ordered = [row for shot_rows in payloads for row in shot_rows]
        keys = range(first_id, first_id + len(ordered))
        library.data.update(zip(keys, ordered, strict=True))
        library.keymap.update(zip(ordered, keys, strict=True))
        library.next_free_ID = first_id + len(ordered)

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

    def _compress_grad_cached(self, event: SimpleNamespace):
        """``(amplitude, shape_ids)`` for an arbitrary gradient; see above."""
        waveform = event.waveform
        times = event.tt
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
        amplitude, shape_IDs = self._compress_grad_cached(event)
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


def _ext_values(extensions, template: _TemplateBlock, slot_index: int) -> dict | None:
    """The payload entry belonging to one extension slot, as ``{name: value}``.

    A payload lists its extensions in block-event order rather than keyed, and
    the template's ``ext_slots`` were recorded in that same order, so the two
    line up by position.
    """
    try:
        ordinal = template.ext_slots.index(slot_index)
        kind, values = extensions[ordinal]
    except (ValueError, IndexError):
        return None
    names = _EXT_PAYLOAD_FIELDS.get(kind)
    if names is None or len(names) != len(values):
        return None
    return dict(zip(names, values, strict=True))


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
