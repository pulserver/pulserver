"""A Pulseq sequence whose libraries, block table and file formats live in C++.

A thin face on ``pulseq::Sequence``: blocks go in through
:meth:`Sequence.add_block` and come back through :meth:`Sequence.get_block`,
and everything between -- event libraries, deduplication, shape compression,
the readers and writers -- is C++.

Three things about it are not obvious from the method list.

**Nothing is deduplicated on the way in.** A block appends a row per event and
a shape per unseen waveform; :meth:`Sequence.remove_duplicates` collapses the
result once the sequence is finished. Searching the libraries per event would
make building a scan quadratic in its largest dimension. A consequence is that
``register_*_event`` returns no library row id -- rows are renumbered by
deduplication, so no id would still be true when the file is written.

**Analysis decodes a window, not a scan.** Upstream PyPulseq's plotting,
k-space and waveform code is given a real :class:`pypulseq.Sequence` built
from the blocks asked about, with rotations resolved into its gradients and RF
shims spread across transmit channels -- so it describes what the scanner
plays, not the base waveform the file stores.

**``tr=`` asks a different question from ``time_range=``.** The timeline views
analyse a window of blocks played once from rest; under ``tr`` the C safety
core builds one canonical TR whose amplitudes are the per-sample maximum over
every instance of it -- a waveform that appears nowhere on the timeline -- and
evaluates it periodically. ``tr=None`` is upstream PyPulseq to the bit, so the
other answer has to be asked for by name.
"""

from __future__ import annotations

__all__ = ["Sequence"]

import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pypulseq as pp
from scipy.spatial.transform import Rotation

from .._ext import pulseqpp as _cxx
from . import _results, _safety
from ._analysis import AnalysisMixin
from ._common import _span
from ._make_label import make_label as _make_label
from ._opts import default_system
from ._block import (
    _adc_event,
    _decode_extensions,
    _grad_event,
    _rf_event,
    _shape,
    _times,
)
from ._plot import _is_merged, _merge_columns, _split_columns
from ._pulseqpp import to_upstream
from ._safety_views import SafetyViewsMixin
from ._soft_delay import SoftDelayMixin
from ._structure import _Structure, _worst_window
from ._rotate3d import rotate3D

#: Definitions written for every sequence, taken from the system it was built
#: with. A caller's own value for one of these wins.
_RASTER_DEFINITIONS = (
    ("AdcRasterTime", "adc_raster_time"),
    ("BlockDurationRaster", "block_duration_raster"),
    ("GradientRasterTime", "grad_raster_time"),
    ("RadiofrequencyRasterTime", "rf_raster_time"),
)

#: What every method whose implementation isn't ported yet says. The
#: signature above each of these already matches upstream PyPulseq's; only
#: the body is missing.


#: Blocks past which looking at a whole sequence is said out loud rather than
#: simply attempted. Drawing this many is minutes of Matplotlib, and the caller
#: who did not pass a ``time_range`` almost certainly did not mean to.
_LOUD_ABOVE = 50_000


class Sequence(AnalysisMixin, SafetyViewsMixin, SoftDelayMixin):
    """A Pulseq sequence, built block by block.

    Parameters
    ----------
    system : pypulseq.Opts, optional
        Hardware limits and raster times. Defaults to
        :attr:`pypulseq.Opts.default`.

    Attributes
    ----------
    system : pypulseq.Opts
        The system the sequence was built against.

    Examples
    --------
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    >>> seq = pp.Sequence(system=system)
    >>> gx = pp.make_trapezoid(channel="x", area=1000, system=system)
    >>> index = seq.add_block(gx)
    >>> deduplicated = seq.remove_duplicates(in_place=True)
    >>> seq.write("gradient.seq")  # doctest: +SKIP

    Notes
    -----
    The events :mod:`pulserver.pypulseq` hands back keep their fields in slots
    rather than in a dictionary, and carry the shape ids this sequence issued
    for them, so a waveform reused across a scan is registered once. Plain
    :class:`types.SimpleNamespace` events from upstream PyPulseq work too, at
    the cost of a dictionary lookup per field.
    """

    def __init__(self, system: pp.Opts | None = None) -> None:
        self.system = default_system(system)
        self._native = _cxx.Sequence()
        self._native.set_rasters(
            float(self.system.rf_raster_time),
            float(self.system.grad_raster_time),
            float(self.system.adc_raster_time),
            float(self.system.block_duration_raster),
        )
        self._definitions: dict[str, object] = {}
        self._trid_ids: dict[str, int] = {}
        # Scan structure -- the repetition time and the C collection the
        # safety analyses run on -- is derived on demand and thrown away
        # whenever the sequence changes underneath it. See _structure().
        self._structure: _Structure | None = None
        self._revision = 0

    def _touch(self) -> None:
        """Note that the blocks or libraries changed.

        Anything derived from the sequence as a whole -- currently the scan
        structure behind :meth:`calculate_pns` and
        :meth:`calculate_gradient_spectrum` -- is keyed by this counter, so a
        cache built before the change is never mistaken for one built after
        it. Cheaper than invalidating explicitly from a dozen call sites and,
        more to the point, impossible to half-do.
        """
        self._revision += 1

    # -- version -----------------------------------------------------------

    @property
    def version_major(self) -> int:
        """int : The Pulseq file version's major number."""
        return self._native.version_major

    @property
    def version_minor(self) -> int:
        """int : The Pulseq file version's minor number."""
        return self._native.version_minor

    @property
    def version_revision(self) -> int:
        """int : The Pulseq file version's revision number."""
        return self._native.version_revision

    # -- what is in it ---------------------------------------------------

    def __str__(self) -> str:
        """A one-look summary: library sizes, rasters, block count."""
        native = self._native
        rows = [
            ("shapes", native.num_shapes()),
            ("rf", native.num_rf()),
            ("gradients", native.num_gradients()),
            ("adc", native.num_adc()),
            ("extensions", native.num_extensions()),
        ]
        lines = ["Sequence:"]
        lines += [f"{name}_library: {count}" for name, count in rows]
        lines.append(f"rf_raster_time: {self.system.rf_raster_time}")
        lines.append(f"grad_raster_time: {self.system.grad_raster_time}")
        lines.append(f"block_events: {self.num_blocks}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return self._native.num_blocks()

    @property
    def num_blocks(self) -> int:
        """int : How many blocks the sequence holds."""
        return self._native.num_blocks()

    @property
    def block_durations(self) -> np.ndarray:
        """numpy.ndarray : Every block's duration, in seconds, in play order."""
        return self._native.block_durations()

    @property
    def block_events(self) -> np.ndarray:
        """numpy.ndarray : The block table, as an ``(N, 6)`` array of library ids.

        The columns are RF, the three gradient axes, ADC, and the head of the
        extension chain. Pulseq's legacy delay column is not among them: it is
        always zero in a 1.4-or-later file and is written as such.
        """
        return self._native.block_events()

    # -- the scan structure the C core recovers ---------------------------
    #
    # These three read the same cached `_Structure` every TR-based analysis
    # works from, so asking for them beside `plot(tr=...)` or `calculate_pns`
    # costs nothing extra. They are derived, never declared: a `.seq` written
    # anywhere states no TR and no segmentation, and what is reported here is
    # what `pulseg_find_tr` recovered from the blocks themselves.

    @property
    def num_trs(self) -> int:
        """int : How many structural TRs the repeating region holds.

        The *structural* repeat, which is not the number of TRs the scanner
        plays -- averages multiply it, and prep and cooldown TRs sit outside
        it. The bound on what ``tr=<int>`` can name is
        :attr:`~._sequence._Structure.num_instances`, reached through
        ``plot(tr=...)``, not this.
        """
        return self._structure_for("num_trs").num_trs

    @property
    def tr_size(self) -> int:
        """int : How many blocks one TR holds."""
        return int(self._structure_for("tr_size").tr["tr_size"])

    @property
    def num_segments(self) -> int:
        """int : How many distinct segments the sequence was decomposed into.

        The interpreter's unit of playout. Index one with
        ``plot(segment_idx=...)``.
        """
        return len(self._structure_for("num_segments").segments)

    @property
    def definitions(self) -> dict[str, object]:
        """dict : The ``[DEFINITIONS]`` entries set so far.

        Raster times and the total duration are not among them: they come from
        :attr:`system` and are added when the sequence is written.
        """
        return dict(self._definitions)

    def set_definition(
        self, key: str, value: str | float | int | list | np.ndarray
    ) -> None:
        """Record a ``[DEFINITIONS]`` entry.

        Parameters
        ----------
        key : str
            The definition's name.
        value : str or float or int or list or numpy.ndarray
            Text, one number, or several. Python ints are written as whole
            numbers; floats, NumPy ones included, are written as reals even
            when they hold a whole number.
        """
        self._touch()
        self._definitions[key] = value

    def declare_tr(self) -> int | None:
        """Record the structural TR as the ``TRSize`` definition.

        Runs the safety core's TR detection and writes the number of blocks
        in one structural TR into ``[DEFINITIONS]``. Called by :meth:`write`
        and :meth:`write_binary`, so every written file carries the
        declaration; calling it directly only matters for a sequence handed
        on in memory. Downstream consumers may use it or ignore it: a
        reconstruction derives its sequence description only when the
        definition is present, and an interpreter may take it as a verified
        hint for pattern detection while keeping full detection as the
        check. Each file of a ``NextSequence`` chain declares its own.

        Returns
        -------
        int or None
            The declared blocks-per-TR, or ``None`` when detection finds no
            repeating structure (nothing is written then).
        """
        try:
            tr_size = int(self._structure_for("declare_tr").tr["tr_size"])
        except Exception:
            return None
        if tr_size <= 0:
            return None
        self.set_definition("TRSize", tr_size)
        return tr_size

    def get_definition(self, key: str) -> object | None:
        """The value recorded for ``key``, or ``None`` if there is none.

        Parameters
        ----------
        key : str
            The definition's name.

        Returns
        -------
        object or None
            Whatever was recorded.
        """
        return self._definitions.get(key)

    def duration(self) -> tuple[float, int, np.ndarray]:
        """Total play time, block count, and how many blocks carry each event.

        Returns
        -------
        duration : float
            Seconds.
        num_blocks : int
            How many blocks were counted.
        event_count : numpy.ndarray
            Non-zero ids per block-table column, in the order
            :attr:`block_events` lists them.
        """
        events = self._native.block_events()
        counted = (
            np.count_nonzero(events, axis=0)
            if events.size
            else np.zeros(events.shape[1] or 6, dtype=np.int64)
        )
        return self._native.duration(), self._native.num_blocks(), counted

    # -- building --------------------------------------------------------

    def add_block(self, *events: object) -> int:
        """Append one block playing ``events``.

        Parameters
        ----------
        *events
            Any mixture of RF, gradient, ADC, delay, label, trigger, rotation,
            RF shim and soft delay events. A bare delay sets a floor on the
            block's duration.

        Returns
        -------
        int
            The new block's 1-based index.

        Notes
        -----
        The block lasts as long as the longest thing in it, rounded up onto
        the block duration raster -- which is what PyPulseq computes, and what
        a caller passing a bare delay is asking for directly.
        """
        self._revision += 1
        return self._native.add_block_events(*events)

    def add_trid(self, label_name: str) -> int:
        """Open a TR group here, naming it rather than numbering it.

        Adds a block carrying ``SET TRID <id>``, allocating the id on the
        name's first use. TRID marks where one repetition time -- one
        sequence segment -- begins, which is what an interpreter groups by to
        evaluate SAR and RF amplitude per contrast rather than over the whole
        run. Naming the groups (``"fat_suppression"``, ``"readout"``) is
        MATLAB Pulseq's idiom and keeps the numbering out of the design
        script.

        Parameters
        ----------
        label_name : str
            The group's name. The same name always gets the same id.

        Returns
        -------
        int
            The numeric TRID.
        """
        trid = self.get_or_create_trid_id(label_name)
        self.add_block(_make_label(type="SET", label="TRID", value=trid))
        return trid

    def get_or_create_trid_id(self, label_name: str) -> int:
        """The numeric TRID for ``label_name``, allocating it on first use."""
        if not isinstance(label_name, str) or not label_name:
            raise ValueError("TRID label_name must be a non-empty string")
        if label_name not in self._trid_ids:
            self._trid_ids[label_name] = len(self._trid_ids) + 1
        return self._trid_ids[label_name]

    @property
    def trid_names(self) -> dict[str, int]:
        """The TR-group names seen so far, and the id each was given."""
        return dict(self._trid_ids)

    def set_block(self, block_index: int, *events: object) -> None:
        """Replace block ``block_index`` with one playing ``events``.

        Parameters
        ----------
        block_index : int
            1-based block index, which must already exist.
        *events
            As for :meth:`add_block`.

        Notes
        -----
        Rows the old block referred to stay in their libraries. They may be
        shared with other blocks, and what survives is
        :meth:`remove_duplicates`' decision rather than this one's.
        """
        self._touch()
        self._native.set_block_events(block_index, *events)

    def get_block(self, block_index: int) -> SimpleNamespace:
        """Block ``block_index``, decoded back into PyPulseq events.

        Parameters
        ----------
        block_index : int
            1-based block index.

        Returns
        -------
        types.SimpleNamespace
            ``block_duration``, plus ``rf``, ``gx``, ``gy``, ``gz``, ``adc``,
            ``rotation``, ``rf_shim`` and ``soft_delay`` -- ``None`` when the
            block does not carry them -- and the list-valued ``triggers``,
            ``label_sets``, ``label_incs`` and ``labels``. The last is the
            other two interleaved as ``(kind, name, value)`` in play order,
            which is the only form that says whether a ``SET`` preceded an
            ``INC`` of the same counter.

        Notes
        -----
        Full double precision: the rows are read out of the libraries, not
        parsed back from a written file.
        """
        rf_id, gx_id, gy_id, gz_id, adc_id, ext_id, duration = self._native.get_block(
            block_index
        )
        block = SimpleNamespace(
            block_duration=duration,
            rf=_rf_event(self._native, rf_id) if rf_id else None,
            gx=_grad_event(self._native, gx_id, "x") if gx_id else None,
            gy=_grad_event(self._native, gy_id, "y") if gy_id else None,
            gz=_grad_event(self._native, gz_id, "z") if gz_id else None,
            adc=_adc_event(self._native, adc_id) if adc_id else None,
            triggers=[],
            label_sets=[],
            label_incs=[],
            # The same label operations again, interleaved in the order the
            # block plays them. `label_sets` and `label_incs` cannot say
            # whether a SET came before an INC of the same counter; label
            # evaluation has to know.
            labels=[],
            rotation=None,
            rf_shim=None,
            soft_delay=None,
        )
        _decode_extensions(self._native, ext_id, block)
        return block

    # -- pre-registration --------------------------------------------------
    #
    # PyPulseq's `register_*_event` protocol, so a script written against
    # upstream runs here unchanged. Upstream needs these because its
    # `add_block` would otherwise re-hash a waveform on every use; this class
    # never does that -- an event carries the shape ids this sequence issued
    # for it, and a waveform reused across a scan is registered once whether
    # or not anything calls these.
    #
    # So they are not a no-op, but what they do is move that first
    # registration off whichever loop iteration happened to come first, which
    # is the intent of the upstream call. The shape ids they return are real.
    # An event-library *row* id is not returned, because there is none to
    # give: rows are appended per block and renumbered by
    # :meth:`remove_duplicates`, so any number handed out here would be a
    # different row by the time the file is written. `0` stands in it, and
    # nothing here reads it back.

    def register_rf_event(self, event: object) -> tuple[int, list[int]]:
        """Register ``event``'s shapes ahead of the blocks that play it.

        Returns
        -------
        tuple
            ``(0, shape_ids)`` -- the magnitude, phase and time shape ids.
        """
        self._touch()
        return 0, list(self._native.warm_event(event))

    def register_grad_event(self, event: object) -> int | tuple[int, list[int]]:
        """Register ``event``'s shapes ahead of the blocks that play it.

        Returns
        -------
        int or tuple
            ``0`` for a trapezoid, which has no shape; ``(0, shape_ids)`` for
            an arbitrary gradient, matching upstream.
        """
        self._touch()
        shapes = list(self._native.warm_event(event))
        return (0, shapes) if shapes else 0

    def register_adc_event(self, event: object) -> tuple[int, int]:
        """Register ``event``'s phase-modulation shape, if it has one.

        Returns
        -------
        tuple
            ``(0, shape_id)``; the shape id is ``0`` when there is no
            modulation to store.
        """
        self._touch()
        shapes = list(self._native.warm_event(event))
        return 0, (shapes[0] if shapes else 0)

    def register_label_event(self, event: object) -> int:  # noqa: ARG002 - upstream's signature
        """Accepted for upstream compatibility; a label carries no shape."""
        return 0

    def register_soft_delay_event(self, event: object) -> int:  # noqa: ARG002 - upstream's signature
        """Accepted for upstream compatibility; a soft delay carries no shape."""
        return 0

    def register_control_event(self, event: object) -> int:  # noqa: ARG002 - MATLAB's signature
        """Accepted for symmetry; a trigger or digital output carries no shape.

        MATLAB's ``registerControlEvent``. Upstream PyPulseq has no equivalent,
        so this exists to complete the registration idiom rather than to
        satisfy a drop-in call.
        """
        return 0

    def register_rotation_event(self, event: object) -> int:
        """Register a rotation extension ahead of the blocks that carry it.

        MATLAB's ``registerRotationEvent``; PyPulseq has no equivalent, because
        it cannot read the extension at all.

        Returns
        -------
        int
            The rotation library id holding this orientation.

        Notes
        -----
        **Registering the same rotation twice returns the same id**, because
        this looks for an equal row before appending one. That is not true of
        ``add_block``, which appends a row per block and lets
        :meth:`remove_duplicates` collapse them afterwards -- the right
        trade for the hot path, where searching the library per block would
        make building a scan quadratic in the thing that is already largest.

        So this does not save the scan any work; it gives a caller a stable
        name for an orientation. Do not hold the id across
        :meth:`remove_duplicates`, which renumbers every library.
        """
        rotation = getattr(event, "rot_quaternion", event)
        # The same call ``add_block`` makes, in the same convention -- scalar
        # first, canonical sign. Reading the quaternion any other way would
        # register a row that the identical event played through a block would
        # not match, and the whole point of registering is that it matches.
        if hasattr(rotation, "as_quat"):
            quaternion = np.asarray(
                rotation.as_quat(canonical=True, scalar_first=True), dtype=float
            ).ravel()
        else:
            quaternion = np.asarray(rotation, dtype=float).ravel()
        if quaternion.size != 4:
            raise ValueError(f"a rotation is four numbers, got {quaternion.size}")
        if not np.isclose(np.linalg.norm(quaternion), 1.0, atol=1e-6):
            raise ValueError("rotation quaternion is not a unit quaternion")

        existing = self._find_row(
            self._native.num_rotations, self._native.rotation_row, quaternion
        )
        if existing:
            return existing
        self._touch()
        return int(self._native.register_rotation(quaternion.tolist()))

    def register_rf_shim_event(self, event: object) -> int:
        """Register an RF-shim extension ahead of the blocks that carry it.

        MATLAB's ``registerRfShimEvent``; PyPulseq has no equivalent. A shim is
        one complex number per transmit channel and is normally shared across a
        whole scan, so registering it once is the point.
        """
        shim = np.asarray(getattr(event, "shim_vector", event), dtype=complex).ravel()
        # The library stores magnitude and phase interleaved, which is how the
        # file writes them.
        values: list[float] = []
        for channel in shim:
            values.extend((float(np.abs(channel)), float(np.angle(channel))))

        existing = self._find_row(
            self._native.num_rf_shims, self._native.rf_shim_row, np.asarray(values)
        )
        if existing:
            return existing
        self._touch()
        return int(self._native.register_rf_shim(values))

    @staticmethod
    def _find_row(count, read, wanted: np.ndarray) -> int:
        """The id of an existing row equal to ``wanted``, or 0 for none.

        A linear scan, which is right here and wrong in ``add_block``: this is
        called once per distinct orientation or shim, outside the loop, when
        the library is still small. Doing the same per block is what would make
        building a scan quadratic.
        """
        for identifier in range(1, int(count()) + 1):
            row = np.asarray(read(identifier), dtype=float).ravel()
            if row.size == wanted.size and np.allclose(row, wanted, rtol=0, atol=1e-12):
                return identifier
        return 0

    def remove_duplicates(self, in_place: bool = False) -> Sequence:
        """Collapse every library to its distinct rows and renumber the blocks.

        Rows are compared at the precision the file writes them, so two events
        that would serialise identically become one. Idempotent.

        Parameters
        ----------
        in_place : bool, default False
            Deduplicate this sequence rather than a copy. The default returns
            a copy, which is upstream's.

        Returns
        -------
        Sequence
            The deduplicated sequence -- ``self`` when ``in_place``.
        """
        target = self if in_place else self._clone()
        target._touch()
        target._native.remove_duplicates()
        return target

    def _writes_once(self) -> bool:
        """Whether any block sets the ``ONCE`` flag.

        Reads the LABELSET library rather than walking the blocks, in C++
        because that library holds a row per use until deduplication runs.
        """
        once = self._native.find_label_id("ONCE")
        return once >= 0 and self._native.label_set_writes(once)

    def _expand_repeats(
        self,
        repeats: int,
        *,
        label: str = "AVG",
        strip_once: bool = True,
    ) -> dict[str, int]:
        """Play the sequence ``repeats`` times, written into the block table.

        Private: :func:`pulserver.pypulseq.tile` is the way in, because the
        pass wants a deduplicated sequence and reads much better on one --
        see there. Upstream PyPulseq has no equivalent of either.

        A ``.seq`` describes one pass; playing it several times is normally
        left to the interpreter, which takes the count from outside the file
        (on a GE scanner, ``opnex``). The ``ONCE`` flag says what belongs to
        a single pass:

        =========  ======================================================
        ``ONCE``   plays
        =========  ======================================================
        ``1``      on the first repetition only — preparation, a startup
                   transient
        ``0``      on every repetition — the body of the scan
        ``2``      on the last repetition only — cooldown, a ramp-down
        =========  ======================================================

        This resolves that here instead. Afterwards the block table *is* the
        scan: every repetition is present in the order it plays, and nothing
        downstream has to be told how many times to read it.

        Only the block table grows. A repetition plays the *same* events, so
        every library is untouched and deduplication has nothing left to find
        — a 100 000-block scan repeated three times is 300 000 rows of six
        integers and a duration, and not one extra gradient.


        Parameters
        ----------
        repeats : int
            How many times the body plays, at least 1. ``1`` is not a no-op:
            it resolves the flags, leaving a file whose block table is the
            whole of what plays.
        label : str, default "AVG"
            Counter stamped with the repetition index, or ``""`` for none.
            ``AVG`` because the repetition an interpreter adds is a signal
            average — the same acquisition, sampled again. ``REP`` is the
            frame counter of a dynamic series and means something else. Set
            where it changes, so it costs one extension per repetition.
        strip_once : bool, default True
            Drop the ``ONCE`` labels once resolved — they now describe a
            table they no longer fit. Pulserver's own interpreter gives
            ``ONCE`` no structural meaning, so nothing is lost by dropping
            them.

        Returns
        -------
        dict
            ``repeats``, ``blocks_before``, ``blocks_after`` and the per-pass
            ``prep_blocks``/``body_blocks``/``cooldown_blocks`` counts.

        Raises
        ------
        ValueError
            If ``repeats`` is below 1.
        RuntimeError
            If a block carries an ``ONCE`` value outside ``{0, 1, 2}``, if
            ``repeats`` exceeds 1 with every block marked ``ONCE`` (there is
            no body to repeat), or if the sequence already writes ``label``
            — two meanings on one counter is not resolved quietly.

        Examples
        --------
        >>> import pulserver.pypulseq as pp
        >>> seq = pp.Sequence(pp.Opts())
        >>> seq.add_block(pp.make_delay(1e-3), pp.make_label("ONCE", "SET", 1))
        1
        >>> seq.add_block(pp.make_delay(2e-3), pp.make_label("ONCE", "SET", 0))
        2
        >>> report = seq._expand_repeats(3)
        >>> report["blocks_before"], report["blocks_after"]
        (2, 4)
        >>> report["prep_blocks"], report["body_blocks"]
        (1, 1)

        See Also
        --------
        remove_duplicates : the other whole-sequence pass, run before writing.
        auto_label : the encoding counters, derived rather than materialised.
        """
        if repeats < 1:
            raise ValueError(f"tile(): reps must be at least 1, got {repeats}")
        self._touch()
        if repeats == 1 and not self._writes_once():
            # One repetition of a sequence that flags nothing plays exactly the
            # blocks it already holds: there is no order to resolve and no flag
            # to strip, so the table is left alone. On a million-block scan
            # that is the difference between a rebuild and nothing.
            blocks = self._native.num_blocks()
            report = {
                "repeats": 1,
                "blocks_before": blocks,
                "blocks_after": blocks,
                "prep_blocks": 0,
                "body_blocks": blocks,
                "cooldown_blocks": 0,
            }
        else:
            report = self._native.expand_repeats(int(repeats), label, strip_once)
        return report

    # -- FOV positioning --------------------------------------------------

    # -- files -----------------------------------------------------------

    def write(
        self,
        name: str | Path,
        create_signature: bool = True,
        remove_duplicates: bool = True,
        check_timing: bool = True,
        v141_compat: bool = False,
        *,
        check_gradients: bool = True,
    ) -> str | None:
        """Write the sequence as a ``.seq`` file.

        Parameters
        ----------
        name : str or pathlib.Path
            Where to write.
        create_signature : bool, default True
            Append the ``[SIGNATURE]`` section, and return its hash.
        remove_duplicates : bool, default True
            Deduplicate before writing. The sequence itself is left alone --
            a copy is written -- so the signature returned belongs to the
            deduplicated file rather than to this object, which is upstream's
            behaviour too.
        check_timing : bool, default True
            Warn if the sequence has timing errors.
        v141_compat : bool, default False
            Not implemented -- see Raises.
        check_gradients : bool, default True
            Warn if a gradient exceeds the system's amplitude or slew limit,
            or jumps across a block boundary. Building the sequence checks
            none of these, so writing is the natural place; pass False in a
            server-mode plugin, where the scanner checks them at predownload
            against its own rasters and limits.

        Returns
        -------
        str or None
            The signature, when one was created.

        Raises
        ------
        NotImplementedError
            If ``v141_compat`` is set. The 1.4.1 writer is a second
            serialiser for a superseded revision; nothing here reads it back,
            so it would ship untested.

        See Also
        --------
        write_binary : the same sequence in the binary Pulseq format.

        Notes
        -----
        Text, to a file, always -- the reference toolbox's ``write``. The
        binary format has its own method, and it is the only one of the two
        that will write anywhere but a file.

        Writing declares the structural TR: :meth:`declare_tr` runs first,
        so the file carries ``TRSize`` whenever detection succeeds.
        """
        if v141_compat:
            raise NotImplementedError(
                "Sequence.write(v141_compat=True) is not implemented: Pulseq 1.4.1 is a "
                "superseded revision and nothing here reads it back. Write 1.5 and convert "
                "with MATLAB's write_v141 if a 1.4.1 file is really needed."
            )
        target = self._prepared_for_write(
            check_timing=check_timing,
            check_gradients=check_gradients,
            remove_duplicates=remove_duplicates,
        )
        payload = target._to_text(create_signature=create_signature)
        Path(name).write_bytes(payload)
        return _signature_of(payload) if create_signature else None

    def _prepared_for_write(
        self,
        *,
        check_timing: bool = True,
        check_gradients: bool = True,
        remove_duplicates: bool = True,
    ) -> Sequence:
        """The sequence a writer should serialise, warnings already raised.

        Returns
        -------
        Sequence
            ``self``, or a deduplicated copy of it.
        """
        if check_timing:
            is_ok, error_report = self.check_timing()
            if not is_ok:
                warnings.warn(
                    f"write(): {len(error_report)} timing errors found in the sequence",
                    stacklevel=3,
                )
        if check_gradients:
            for is_ok, message in (
                self.check_hardware_limits(),
                self.check_gradient_continuity(),
            ):
                if not is_ok:
                    warnings.warn(f"write(): {message}", stacklevel=3)
        self.declare_tr()

        # A sequence that has not been touched since its last deduplication
        # has nothing here to find, and saying so skips a copy of the whole
        # thing as well as the pass -- which on a protocol-scale scan is the
        # larger half.
        if remove_duplicates and not self._native.deduplicated():
            return self.remove_duplicates()
        return self

    def write_binary(self, target: str | Path | object) -> None:
        """Write the sequence in the binary Pulseq format.

        Parameters
        ----------
        target : str or pathlib.Path or file object
            A path to write, or an already-open binary stream to write into.
            The reference toolbox takes only a filename; a stream is allowed
            here because the binary format is the one worth handing to another
            process without going through the filesystem first.

        See Also
        --------
        write : the same sequence as ``.seq`` text.
        """
        self.declare_tr()
        payload = self._to_binary()
        writer = getattr(target, "write", None)
        if callable(writer):
            writer(payload)
            return
        Path(target).write_bytes(payload)

    def _to_text(self, *, create_signature: bool = True) -> bytes:
        """The sequence as a ``.seq`` file, in memory.

        Notes
        -----
        Shapes are compressed here rather than as they are registered, so the
        codec runs over the waveforms that survived deduplication instead of
        over every use of them.
        """
        self._native.compress_shapes()
        self._publish_definitions()
        return self._native.write_text(create_signature)

    def _to_binary(self) -> bytes:
        """The sequence as a binary Pulseq file, in memory.

        The form to hand anything that is going to parse it back rather than
        read it: the writer skips number formatting and the reader skips
        ``sscanf``, so a round trip through this costs a fraction of what the
        text format does and loses none of the precision the text format
        rounds away. It is what :meth:`_structure_for` feeds the C safety
        library.
        """
        self._native.compress_shapes()
        self._publish_definitions()
        return self._native.write_binary()

    def read(
        self,
        file_path: str | Path,
        detect_rf_use: bool = False,
        remove_duplicates: bool = True,
    ) -> None:
        """Replace the sequence's contents with the file at ``file_path``.

        Parameters
        ----------
        file_path : str or pathlib.Path
            A ``.seq`` text file or a binary Pulseq file. Which one it is is
            decided by the leading bytes, not by the name.
        detect_rf_use : bool, default False
            Not implemented -- see Raises.
        remove_duplicates : bool, default True
            Deduplicate the libraries after reading.

        Raises
        ------
        NotImplementedError
            If ``detect_rf_use`` is set. Upstream guesses a pulse's ``use``
            from its flip angle for files written before Pulseq 1.5, which
            had no ``use`` column. Every ``use`` this reader reports is one
            the file stated.

        Notes
        -----
        :attr:`system` is left alone. The raster times the file records are
        adopted by the C++ sequence, since they are what its blocks were laid
        out on.
        """
        if detect_rf_use:
            raise NotImplementedError(
                "Sequence.read(detect_rf_use=True) is not implemented: it guesses a pulse's "
                "`use` from its flip angle, which Pulseq 1.5 files state outright. Set the "
                "`use` on the events instead of inferring it."
            )
        self._touch()
        self._native = _cxx.read_file(str(Path(file_path)))
        self._definitions = dict(self._native.definitions())
        if remove_duplicates:
            self._native.remove_duplicates()

    # -- looking at it -----------------------------------------------------
    #
    # Everything below takes upstream PyPulseq's arguments, in upstream's
    # order, with upstream's defaults -- a test asserts it. The implemented
    # ones hand the work to upstream's own code over a window of blocks; the
    # rest still raise NotImplementedError.
    #
    # calculate_pns and calculate_gradient_spectrum take more than upstream
    # does, keyword-only and after every one of upstream's, because they can
    # also answer about a repetition time. See the module docstring.

    def plot(
        self,
        label: str = "",
        show_blocks: bool = False,
        save: bool = False,
        time_range=(0, np.inf),
        time_disp: str = "s",
        grad_disp: str = "kHz/m",
        plot_now: bool = True,
        clear: bool = True,
        overlay: object = None,
        stacked: bool = False,
        show_guides: bool = False,
        *,
        tr: str | int | None = None,
        rf_channel: int = 0,
        segment_idx: int | None = None,
    ):
        """Draw the sequence, one canonical TR, or one segment.

        Parameters
        ----------
        label : str, default ""
            Labels whose value to mark at each ADC, as a comma-separated list --
            ``"LIN,REP"``, say. Not available under ``tr``.
        show_blocks : bool, default False
            Tick the axes at the block boundaries.
        save : bool, default False
            Write the figure out as a JPEG beside the working directory.
        time_range : tuple of float, default (0, inf)
            The seconds to draw. Only the blocks this touches are decoded, so
            asking for a tenth of a second of a long protocol costs a tenth of a
            second of blocks -- and the axis still reads in time from the start
            of the sequence. Under ``tr`` it is a zoom into the TR, whose clock
            starts at zero.
        time_disp : {"s", "ms", "us"}, default "s"
            The time unit on the axis.
        grad_disp : {"kHz/m", "mT/m"}, default "kHz/m"
            The gradient unit.
        plot_now : bool, default True
            Show the figure before returning, blocking until it is closed. When
            False, it is drawn but left for ``plt.show()``.
        clear : bool, default True
            Clear the figure first rather than drawing over what is on it.
        overlay : SeqPlot, optional
            Draw on the figure of an earlier plot, to compare two sequences.
        stacked : bool, default False
            Put all six channels in one column, PyPulseq's stacked style.
        show_guides : bool, default False
            Follow the cursor with a vertical hairline across every panel.
            Needs ``mplcursors``; ignored when it is not installed.
        tr : {"worst_case", "zero_variable"} or int, optional
            Draw one canonical TR rather than the timeline.

            ``None``, the default, is upstream PyPulseq exactly: the blocks
            ``time_range`` covers, as the file stores them.

            ``"worst_case"`` is the TR ``pulseg_check_safety`` judges -- **not**
            any TR the scanner plays, but the one whose gradient at each
            position is the largest any instance reaches. Where a block's
            gradient varies between instances, this is what the PNS,
            mechanical-resonance and gradient-heating checks are run against.

            ``"zero_variable"`` is the same canonical TR with those varying
            gradients zeroed instead, leaving the structure the constant ones
            make -- the skeleton the k-space and timing analyses work from.

            An integer is one instance as it really plays, signed amplitudes
            and all. Its use is checking the claim ``"worst_case"`` rests on:
            that no instance exceeds the envelope.
        rf_channel : int, default 0
            For a pTx TR, which transmit channel upstream draws. The rest are
            added to the same two panels afterwards. Read only under ``tr``.
        segment_idx : int, optional
            Draw one segment instead, as its **highest-energy instance** plays.

            A segment is the interpreter's unit of playout, and which
            repetition of it carries the most gradient energy is what the C
            core already tracks (``max_energy_start_block``) in order to run
            the safety checks against it. This draws exactly those blocks, on
            the sequence's own clock, so what is on screen is the instance the
            checks were run on rather than the first one that happened to be
            played. :attr:`num_segments` says how many there are.

            Not combinable with ``tr``: a TR is the core's reconstructed
            canonical waveform and a segment is a run of real blocks, so a
            call naming both is asking for two different pictures.

        Returns
        -------
        pypulseq.utils.seq_plot.SeqPlot
            The plot, whose ``fig1``/``fig2`` and ``ax1``/``ax2`` are upstream's.

        Raises
        ------
        ValueError
            If ``label`` is asked for under ``tr``. A canonical TR is built by
            the C core out of waveforms, and the core does not carry label
            values through it, so there is nothing truthful to mark.
        ValueError
            If ``tr`` and ``segment_idx`` are given together, or if
            ``segment_idx`` is out of range.

        Notes
        -----
        Drawing is upstream PyPulseq's either way; what changes is the sequence
        it is handed.

        Under ``tr=None`` that is a :class:`pypulseq.Sequence` holding the
        blocks asked for, with rotations resolved into the gradients and RF
        shims spread across the transmit channels first, so what is drawn is
        what the scanner plays rather than the base waveform the file stores.

        Under ``tr`` it is the TR the **C safety core** built --- one
        ``pulseg_get_tr_waveforms`` call, giving gradients, RF magnitude and
        phase, ADC events and block boundaries together --- wrapped in a
        :class:`~._safety.TRSequence` so upstream's plotter can walk it. None
        of it is reconstructed here, so the picture is the waveform the
        interpreter's checks were run on, down to which instance's gradient
        won at each position.

        Unstacked, the two panels PyPulseq would open in separate windows are
        laid out as **one figure, three rows by two columns**, sharing their time
        axis; ``fig1`` and ``fig2`` are then the same figure. ``save=True``
        writes that one window as ``seq_plot.jpg`` rather than upstream's pair.
        """
        if tr is not None and segment_idx is not None:
            raise ValueError(
                "plot(): tr draws the canonical TR the safety core reconstructs and "
                "segment_idx draws a run of real blocks -- pass one or the other"
            )

        if segment_idx is not None:
            first, last = self._segment_blocks(segment_idx)
            window = self._upstream_window(first, last)
        elif tr is None:
            first, last = self._blocks_over(*_span(time_range))
            if last - first + 1 > _LOUD_ABOVE:
                warnings.warn(
                    f"Plotting {last - first + 1} blocks. Matplotlib will take a long time over "
                    "this; pass time_range=(start, stop) to draw a part of the sequence.",
                    stacklevel=2,
                )
            window = self._upstream_window(first, last)
        else:
            if label:
                raise ValueError(
                    "plot(): label reads values off the timeline and tr draws the canonical TR "
                    "the safety core builds out of waveforms, which carries none -- pass one "
                    "or the other"
                )
            window = self._structure_for("plot").waveform(tr, rf_channel=rf_channel)

        merging = not stacked
        if merging and _is_merged(overlay):
            _split_columns(overlay)

        plot = window.plot(
            label=label,
            show_blocks=show_blocks,
            save=save and not merging,
            time_range=time_range,
            time_disp=time_disp,
            grad_disp=grad_disp,
            plot_now=False,
            clear=clear,
            overlay=overlay,
            stacked=stacked,
            show_guides=show_guides and not merging,
        )

        if tr is not None:
            # Upstream's block model carries one RF event, so it drew one
            # transmit channel. The rest go on the same two panels, in the
            # display unit it chose.
            _safety.overlay_rf_channels(
                window,
                plot.ax1[1],
                plot.ax1[2],
                getattr(plot.fig1, "_seq_t_factor", 1.0),
            )

        if merging:
            _merge_columns(plot, show_guides=show_guides)
            if save:
                plot.fig1.savefig("seq_plot.jpg")
        if plot_now:
            plot.show()
        return plot

    def _window_for(self, time_range) -> tuple[int, int]:
        """``time_range`` as a 1-based inclusive block range."""
        if time_range is None:
            return 1, self.num_blocks
        if len(time_range) != 2:
            raise ValueError("Time range must be list of two elements")
        if time_range[0] > time_range[1]:
            raise ValueError("End time of time_range must be after begin time")
        return self._blocks_over(float(time_range[0]), float(time_range[1]))

    def calc_moments_btensor(
        self,
        calc_b: bool = True,
        calc_m1: bool = False,
        calc_m2: bool = False,
        calc_m3: bool = False,
        n_dummy: int = 0,
        *,
        time_range: list[float] | None = None,
        compat: bool = True,
    ):
        """The diffusion b-tensor and the gradient moments, one per excitation.

        Parameters
        ----------
        calc_b : bool, default True
            Compute the b-tensor.
        calc_m1, calc_m2, calc_m3 : bool, default False
            Compute the first, second and third gradient moments.
        n_dummy : int, default 0
            Leading excitations to skip.
        time_range : list of float, optional
            ``[start, stop]`` in seconds. The whole sequence by default.
        compat : bool, optional
            MATLAB's ``(B, m1, m2, m3)`` by default -- ``B`` shaped
            ``(R, 3, 3)`` in s/m^2, the moments ``(R, 3)``. ``False`` returns a
            :class:`~._results.BTensor`, which adds the b-values, b-vectors and
            per-shot tensors in the units and shapes a diffusion pipeline
            takes, the deduplicated table, and the split by what the console's
            prescription rotates.

        Returns
        -------
        tuple or BTensor

        Notes
        -----
        **Where the echo is, is the whole difference from MATLAB.**
        ``calcMomentsBtensor`` takes the echo to be ``2*t_refocusing -
        t_excitation`` and carries its own ``TODO: fixme for double-refocused
        sequences``, because that formula reads exactly one refocusing pulse
        per excitation and is wrong the moment there are two -- which is what
        every twice-refocused diffusion sequence has. Here the echo is the ADC
        sample the C core found nearest k-space zero, walking the real
        trajectory; the number is the same one the reconstruction and the
        scanner use. So a twice-refocused, an oscillating-gradient or a
        free-waveform sequence is handled by the same code as a Stejskal-Tanner
        one, with no special case. The two-pulse formula is still the fallback
        for a design that has no readout yet, and only when it lands inside the
        shot's own interval.

        The integration is exact rather than sampled: a Pulseq gradient is
        piecewise linear, so ``q`` is piecewise quadratic and ``q_i q_j``
        piecewise quartic, and each piece is integrated in closed form. An
        arbitrary gradient stored on the centres raster has its raster-edge
        samples restored first, as :meth:`waveforms` does.

        The arithmetic is :func:`pulseq::calc_moments`, in ``src/cpp/pulseq``, so
        LiveSDK computes the same numbers from the same ``.seq`` and a
        diffusion gradient table can reach the MRD stream. It is **not** in the
        C89 core: the interpreter plays gradients and never needs a b-value,
        and only the reconstruction does.

        See Also
        --------
        Sequence.calculate_kspace : the trajectory the echo positions come from.
        Sequence.write_diffusion_definitions : put the table in the ``.seq``.
        """
        first, last = self._window_for(time_range)
        result = self._native.calc_moments(
            bool(calc_b),
            bool(calc_m1),
            bool(calc_m2),
            bool(calc_m3),
            int(n_dummy),
            first,
            last,
        )

        shots = result["t_excitation"].size
        if shots == 0:
            raise ValueError(
                "calc_moments_btensor(): the window holds no excitation pulse, so there "
                "is nothing to integrate from. A b-tensor is per excitation."
            )

        def _tensor(name: str) -> np.ndarray:
            value = result[name]
            return value if value.size else np.zeros((shots, 3, 3))

        def _vector(name: str) -> np.ndarray:
            value = result[name]
            return value if value.size else np.zeros((shots, 3))

        fixed, rotatable, cross = (
            _tensor("b_fixed"),
            _tensor("b_rotatable"),
            _tensor("b_cross"),
        )
        b_tensor = fixed + rotatable + cross + np.swapaxes(cross, -1, -2)
        m1, m2, m3 = _vector("m1"), _vector("m2"), _vector("m3")

        if compat:
            return (b_tensor, m1, m2, m3)

        table_cross = result["table_cross"]
        table = (
            result["table_fixed"]
            + result["table_rotatable"]
            + table_cross
            + np.swapaxes(table_cross, -1, -2)
        )
        return _results.BTensor.of(
            B=b_tensor,
            m1=m1,
            m2=m2,
            m3=m3,
            excitation_times=result["t_excitation"],
            echo_times=result["t_echo"],
            b_fixed=fixed,
            b_rotatable=rotatable,
            b_cross=cross,
            b_tensor_table=table,
            table_index=result["table_index"],
        )

    def check_timing(
        self,
        print_errors: bool = False,
        *,
        time_range: list[float] | None = None,
    ) -> tuple[bool, list]:
        """Check every block's timing against the rasters and dead times.

        Raster alignment, RF and ADC dead times, ringdown, block-duration
        consistency, and soft-delay agreement, judged against ``system`` --
        its four rasters and its three dead times, not the ones the file
        happens to record. The report is upstream PyPulseq's, entry for entry;
        the arithmetic is compiled and reads the event libraries, so what a
        raster costs is one decision per distinct event rather than one per
        block.

        The scanner runs its own version of this at predownload against its
        real rasters, which is the authoritative one; a design script can only
        check against the rasters it was given. Both are worth having, and
        this is the one available before the sequence leaves the bench.

        Parameters
        ----------
        print_errors : bool, default False
            Also print the report.
        time_range : list of float, optional
            Restrict to the blocks in this window, in seconds.

        Returns
        -------
        is_ok : bool
        error_report : list
            One entry per problem, in upstream's form.

        See Also
        --------
        check_hardware_limits, check_gradient_continuity : the gradient checks
            :meth:`write` runs.
        """
        first, last = self._window_for(time_range)
        error_report = [
            _timing_entry(finding)
            for finding in self._native.check_timing(
                float(self.system.rf_raster_time),
                float(self.system.grad_raster_time),
                float(self.system.adc_raster_time),
                float(self.system.block_duration_raster),
                float(self.system.rf_dead_time),
                float(self.system.rf_ringdown_time),
                float(self.system.adc_dead_time),
                first,
                last,
            )
        ]
        if print_errors:
            for entry in error_report:
                print(entry)
        return not error_report, error_report

    def check_gradient_continuity(self) -> tuple[bool, str]:
        """Check that gradients join across every block boundary.

        Building a sequence does not check this -- the cost would fall on
        every :meth:`add_block` -- so it is done here instead, and by
        :meth:`write` unless told not to. The arithmetic is the C safety
        core's, the same code the scanner runs at predownload.

        Endpoints are those of the shape each block instance actually plays,
        scaled by its own amplitude, and **both sides are rotated before they
        are compared**, so a block carrying a ``ROTATIONS`` extension is
        judged in the physical frame -- the frame the amplifiers slew in --
        rather than in its own logical one.

        Returns
        -------
        is_ok : bool
        message : str
            Names the axis and block of the first discontinuity; empty when
            there is none.
        """
        from .._ext.pulseg import _check_grad_continuity

        message = _check_grad_continuity(
            self._structure_for("check_gradient_continuity").collection
        )
        return not message, message

    def check_hardware_limits(self) -> tuple[bool, str]:
        """Check every gradient against the system's amplitude and slew limits.

        Over the gradient library rather than the block table, so it costs one
        pass per distinct waveform however many times the scan plays it.

        Judged per axis, in the frame the gradients are stored in. A block
        carrying a ``ROTATIONS`` extension plays a combination of them, and a
        combination can exceed on one physical axis what each logical axis
        respects; the scanner's predownload check is the one that sees that,
        and is authoritative either way. What this catches is the design
        asking for more gradient than the system has.

        Returns
        -------
        is_ok : bool
        message : str
            Names the peak and the limit it passed; empty when there is none.

        See Also
        --------
        check_gradient_continuity : the other gradient check :meth:`write` runs.
        """
        from ._block import _grad_event

        peak_grad = 0.0
        peak_slew = 0.0
        for grad_id in range(1, self._native.num_gradients() + 1):
            event = _grad_event(self._native, grad_id, "x")
            if event.type == "trap":
                amplitude = abs(event.amplitude)
                ramps = [event.rise_time, event.fall_time]
                slew = max(amplitude / ramp for ramp in ramps if ramp > 0.0)
            else:
                amplitude = (
                    float(np.max(np.abs(event.waveform)))
                    if event.waveform.size
                    else 0.0
                )
                steps = np.diff(event.tt)
                slew = (
                    float(np.max(np.abs(np.diff(event.waveform) / steps)))
                    if steps.size and np.all(steps > 0.0)
                    else 0.0
                )
            peak_grad = max(peak_grad, amplitude)
            peak_slew = max(peak_slew, slew)

        for name, peak, limit, unit in (
            ("gradient", peak_grad, self.system.max_grad, "Hz/m"),
            ("slew rate", peak_slew, self.system.max_slew, "Hz/m/s"),
        ):
            if limit and peak > limit * (1 + 1e-6):
                return False, (
                    f"peak {name} {peak:.4g} {unit} exceeds the system limit "
                    f"{limit:.4g} {unit}"
                )
        return True, ""

    def test_report(self) -> str:
        """Upstream PyPulseq's text report on the sequence.

        Timings, gradient and slew extrema, the k-space extent and the
        resolution they imply, and upstream's own timing check -- run by
        upstream's :func:`~pypulseq.Sequence.ext_test_report.ext_test_report`
        on an upstream sequence built from these blocks, so the report is the
        one a PyPulseq user recognises rather than a second dialect of it.

        Notes
        -----
        The sequence handed over has rotations resolved into its gradients and
        RF shims spread across the transmit channels, so the extrema reported
        are the ones the scanner plays. The trajectory inside the report is
        therefore upstream's; :meth:`calculate_kspace` is the C core's and is
        the one to quote.
        """
        return self._upstream_window(1, self.num_blocks).test_report()

    #: The ``[DEFINITIONS]`` keys :meth:`Sequence.write_diffusion_definitions`
    #: writes, and the ones the reconstruction reads back out of the MRD
    #: header. Named once so the two sides cannot drift.
    DIFFUSION_DEFINITIONS = (
        "bTensorFixed",
        "bTensorRotatable",
        "bTensorCross",
        "bTensorAxis",
    )

    def diffusion_definitions(
        self,
        *,
        axis: str,
        n_dummy: int = 0,
        time_range: list[float] | None = None,
    ) -> dict[str, object]:
        """The diffusion gradient table as ``[DEFINITIONS]`` entries.

        Parameters
        ----------
        axis : str
            The label counter whose value selects a row -- ``"SET"``,
            ``"ECO"``, whichever the design's
            :class:`~pulserver.ScanLoop` declared for its diffusion dimension.
        n_dummy : int, default 0
            Leading excitations to skip.
        time_range : list of float, optional
            ``[start, stop]`` in seconds. The whole sequence by default.

        Returns
        -------
        dict
            ``{key: value}`` for :meth:`set_definition`. ``bTensorRotatable``
            and ``bTensorCross`` are omitted when identically zero, which is
            the case for a preparation played entirely under ``NOROT``.

        Raises
        ------
        ValueError
            When ``axis`` does not index the tensors: its values are not
            ``0..N-1``, or two shots that share a value do not share a tensor.
            Both mean the table would be indexed wrongly by whoever reads it,
            and a wrong b-vector is not something a pipeline can notice.

        Notes
        -----
        Three matrices rather than one, because the console's FOV rotation is
        not in the ``.seq`` -- see :class:`~._results.BTensor`. They are
        written verbatim into the MRD header by
        ``mrdserver::add_diffusion_parameters`` and composed with the
        acquisition's direction cosines on the reconstruction side, so the
        scanner never has to know a rotation convention.

        See Also
        --------
        Sequence.write_diffusion_definitions : this, stored on the sequence.
        Sequence.calc_moments_btensor : where the tensors come from.
        """
        result = self.calc_moments_btensor(
            n_dummy=n_dummy, time_range=time_range, compat=False
        )
        counters = self._counter_at(axis, result.echo_times, result.excitation_times)
        order = self._diffusion_rows(axis, counters, result)

        definitions: dict[str, object] = {"bTensorAxis": axis}
        for key, part in (
            ("bTensorFixed", result.b_fixed),
            ("bTensorRotatable", result.b_rotatable),
            ("bTensorCross", result.b_cross),
        ):
            rows = np.asarray(part, dtype=float)[order]
            if key != "bTensorFixed" and not np.any(rows):
                continue
            definitions[key] = rows.reshape(-1).tolist()
        return definitions

    def write_diffusion_definitions(
        self, *, axis: str, n_dummy: int = 0
    ) -> _results.DiffusionTable:
        """Compute the diffusion table, check it, and store it on the sequence.

        Parameters
        ----------
        axis : str
            The label counter whose value selects a row.
        n_dummy : int, default 0
            Leading excitations to skip.

        Returns
        -------
        DiffusionTable
            The table as written, in the units a diffusion pipeline takes.

        See Also
        --------
        Sequence.diffusion_definitions : the entries, without storing them.
        """
        definitions = self.diffusion_definitions(axis=axis, n_dummy=n_dummy)
        for key, value in definitions.items():
            self.set_definition(key, value)
        return _results.DiffusionTable.from_definitions(definitions)

    def _counter_at(
        self, axis: str, echoes: np.ndarray, excitations: np.ndarray
    ) -> np.ndarray:
        """The value of label ``axis`` in force at each shot's echo.

        The echo rather than the excitation, because a counter is normally set
        just before the readout it labels; a shot whose echo was not found
        falls back to its excitation, which is the best that is left.
        """
        evolution = self.evaluate_labels(evolution="blocks").get(axis)
        if evolution is None:
            raise ValueError(
                f"diffusion_definitions(): the sequence never sets a {axis!r} label, so "
                f"there is nothing for a reconstruction to index the table by. Give the "
                f"ScanLoop an axis for the diffusion dimension, or name the one it has."
            )

        ends = np.cumsum(np.asarray(self.block_durations, dtype=float))
        times = np.where(np.isfinite(echoes), echoes, excitations)
        blocks = np.clip(
            np.searchsorted(ends, times, side="left"), 0, len(evolution) - 1
        )
        return np.asarray(evolution, dtype=int)[blocks]

    @staticmethod
    def _diffusion_rows(axis: str, counters: np.ndarray, result) -> np.ndarray:
        """One shot index per counter value, having checked that is well posed."""
        values = np.unique(counters)
        expected = np.arange(values.size)
        if not np.array_equal(values, expected):
            raise ValueError(
                f"diffusion_definitions(): {axis} takes values {values.tolist()}, which do "
                f"not index a table -- a consumer reads row {axis} of it, so the values "
                f"have to be 0..{values.size - 1}. Renumber the axis, or name the counter "
                f"that really varies with the diffusion encoding."
            )

        parts = np.concatenate(
            (
                np.asarray(result.b_fixed, dtype=float).reshape(len(counters), 9),
                np.asarray(result.b_rotatable, dtype=float).reshape(len(counters), 9),
                np.asarray(result.b_cross, dtype=float).reshape(len(counters), 9),
            ),
            axis=1,
        )
        scale = max(float(np.abs(parts).max()), np.finfo(float).tiny)

        first = np.zeros(values.size, dtype=int)
        for value in values.tolist():
            shots = np.flatnonzero(counters == value)
            first[value] = shots[0]
            spread = float(np.abs(parts[shots] - parts[shots[0]]).max()) / scale
            if spread > 1e-6:
                raise ValueError(
                    f"diffusion_definitions(): shots {shots.tolist()} all carry {axis}="
                    f"{value} but their b-tensors differ by {spread:.3g} of the largest "
                    f"element, so one row cannot describe them. {axis} is not the axis the "
                    f"diffusion encoding varies along."
                )
        return first

    def calc_rf_power(
        self,
        time_range: list[float] | None = None,
        *,
        window_duration: float | None = None,
        compat: bool = True,
    ):
        """RF power, energy and RMS amplitude, in Pulseq's relative units.

        Parameters
        ----------
        time_range : list of float, optional
            ``[start, stop]`` in seconds. The whole sequence by default.
        window_duration : float, optional
            Report the worst sliding window of this many seconds rather than
            the average over the whole range -- 10 s and 6 min are the two
            windows SAR is regulated over. The window is rounded up to whole
            blocks, so a short window on long blocks over-reports.
        compat : bool, optional
            MATLAB's ``(mean_pwr, peak_pwr, rf_rms, total_energy)`` by
            default. ``False`` returns an :class:`~._results.RfPower`.

        Returns
        -------
        tuple or RfPower
            ``mean_pwr`` in Hz^2, ``peak_pwr`` in Hz^2, ``rf_rms`` in Hz,
            ``total_energy`` in Hz^2 s -- which is Hz.

        Notes
        -----
        **These numbers are relative and are not SAR.** They are amplitudes in
        Hz, so dividing by ``gamma`` gives Tesla and by ``gamma**2`` gives
        mT^2 s; the step from there to watts per kilogram runs through the
        electric field, which depends on the transmit coil and on the subject.
        MATLAB's ``calcRfPower`` says the same thing at more length. The
        number the scanner gates on comes from the interpreter's own SAR
        model, not from here.

        MATLAB integrates each pulse on a fixed 1 us raster, per block. Here
        the raster is the sequence's own and the integral is done **once per
        distinct waveform**, not per block and not even per library row:
        ``|B1|**2`` scales as the square of the row's amplitude, so a
        variable-flip-angle train of thousands of rows over one shape
        decompresses that shape once and multiplies. Spreading the result over
        the blocks is one fancy-index. On a 100 000-block scan that is ~6 ms
        deduplicated and ~140 ms not, against ~2.9 s for the naive walk.

        **The TR structure is deliberately not used**, though every other
        analysis here can be asked about one. It would not help: recovering it
        costs ~100 ms on that same scan -- more than this whole computation --
        because it serialises the sequence and parses it back, while the RF
        library is *already* deduplicated across the whole scan, which is the
        same saving TR folding would offer and a stronger one, since it holds
        across TRs as well as within them.

        There is one deliberate difference from MATLAB beyond that. MATLAB's
        sliding window indexes its per-block bookkeeping with an offset on the
        way in and without one on the way out, so a ``blockRange`` that does
        not start at block 1 reads the wrong blocks back out of the window.
        The window here is computed from cumulative sums instead, which has no
        such index to get wrong.
        """
        first, last = self._window_for(time_range)
        durations = np.asarray(self.block_durations, dtype=float)[first - 1 : last]
        rf_ids = np.asarray(self.block_events, dtype=np.int64)[first - 1 : last, 0]

        # One integral per distinct *shape*, not per row and certainly not per
        # block. |B1|**2 scales as the square of the row's amplitude, so a
        # variable-flip-angle train -- thousands of rows over one waveform --
        # decompresses that waveform once and multiplies. The lookup itself is
        # a fancy-index, so nothing walks the blocks in Python.
        unit: dict[tuple[int, int, int], tuple[float, float]] = {}
        table = np.zeros((self._native.num_rf() + 1, 2))
        for rf_id in np.unique(rf_ids[rf_ids > 0]).tolist():
            row = self._native.rf_row(int(rf_id))
            key = (int(row[1]), int(row[2]), int(row[3]))
            if key not in unit:
                unit[key] = self._rf_power_of(*key)
            table[rf_id] = np.asarray(unit[key]) * row[0] ** 2

        energies, peaks = table[rf_ids, 0], table[rf_ids, 1]

        total_energy = float(energies.sum())
        peak_power = float(peaks.max()) if peaks.size else 0.0
        span = float(durations.sum())

        if window_duration is None:
            reference, worst_energy = span, total_energy
        else:
            reference = float(window_duration)
            if reference <= 0:
                raise ValueError(
                    f"window_duration must be positive, not {window_duration!r}"
                )
            worst_energy = _worst_window(durations, energies, reference)

        # MATLAB keeps a second accumulator for the RMS, but its summand is
        # `rms**2 * shape_dur`, which *is* the pulse's energy -- so the two
        # accumulators are the same number and rf_rms is sqrt(mean_pwr).
        mean_power = worst_energy / reference if reference > 0 else 0.0
        rf_rms = float(np.sqrt(mean_power))

        if compat:
            return mean_power, peak_power, rf_rms, total_energy
        return _results.RfPower(
            mean_power=mean_power,
            peak_power=peak_power,
            rf_rms=rf_rms,
            total_energy=total_energy,
            duration=span,
            window_duration=window_duration,
        )

    def _rf_power_of(
        self, magnitude_id: int, phase_id: int, time_id: int
    ) -> tuple[float, float]:
        """``(energy, peak power)`` for one shape triple at unit amplitude.

        MATLAB's ``mr.calcRfPower``: resample onto the RF raster at sample
        centres, then sum ``|B1|**2`` times the raster. The complex signal is
        what gets resampled, not its magnitude, because that is what MATLAB
        interpolates and the two differ once a pulse carries phase on a
        non-uniform time raster.

        The row's amplitude does not appear: both outputs scale as its square,
        which is what lets one shape serve every row that plays it.

        The raster is the **sequence's**, not ``system``'s. A sequence read
        from a file carries the raster its ``[DEFINITIONS]`` state, which need
        not be the one this ``Opts`` was built with -- and taking the wrong one
        stretches or squashes the pulse, so the energy comes out scaled by the
        ratio of the two.
        """
        raster = float(self._native.rf_raster_time)
        magnitude = _shape(self._native, magnitude_id)
        count = magnitude.size
        if count <= 0:
            return 0.0, 0.0

        signal = magnitude * np.exp(2j * np.pi * _shape(self._native, phase_id))
        times = _times(self._native, time_id, count, raster)
        centres = (np.arange(count) + 0.5) * raster
        squared = np.abs(np.interp(centres, times, signal, left=0.0, right=0.0)) ** 2
        return float(squared.sum()) * raster, float(squared.max())

    def _structure_for(self, what: str) -> _Structure:
        """The scan structure, derived once and reused until the sequence changes.

        Parameters
        ----------
        what : str
            The calling method, named in the error when there is nothing to
            analyse.
        """
        if self.num_blocks == 0:
            raise ValueError(f"{what}(): the sequence holds no blocks")
        if self._structure is None or self._structure.revision != self._revision:
            self._structure = _Structure(self)
        return self._structure

    def evaluate_labels(
        self,
        init: dict | None = None,
        evolution: str = "none",
        *,
        time_range: list[float] | None = None,
    ) -> dict:
        """Play the label counters through the sequence and report where they get to.

        Parameters
        ----------
        init : dict, optional
            Starting values, for evaluating a sequence a window at a time.
            Not modified.
        evolution : {"none", "blocks", "adc", "label"}, optional
            What to record. ``"none"``, the default, keeps only the final
            value of each label. ``"blocks"`` records after every block,
            ``"adc"`` after every block holding an ADC, and ``"label"`` after
            every block that touches a label.
        time_range : list of float, optional
            ``[start, stop]`` in seconds, to evaluate part of the sequence.
            MATLAB's ``evalLabels`` takes a block range here and PyPulseq
            carries a ``TODO`` about wanting one; a time window is what every
            other method on this class takes.

        Returns
        -------
        dict
            Label name to its final value, or to an array of its values under
            an ``evolution``. Only labels the sequence actually uses appear.

        Notes
        -----
        Upstream's semantics, on this class's blocks -- including that a label
        first seen by an ``INC`` starts from zero, and that a label evolution
        reports ``0`` for blocks before the label was first touched.

        The one place this can differ from upstream is a block carrying both
        a ``SET`` and an ``INC`` of the *same* counter: the operations are
        applied in the order the block plays them, read off the extension
        chain, rather than sets-then-increments.
        """
        if evolution not in ("none", "blocks", "adc", "label"):
            raise ValueError(
                "evaluate_labels(): evolution must be 'none', 'blocks', 'adc' or 'label', "
                f"not {evolution!r}"
            )

        first, last = self._window_for(time_range)
        recorded = self._native.label_evolution(first, last, evolution)
        if not init:
            return recorded

        # ``init`` seeds the running state, for evaluating a sequence a window
        # at a time. The walk itself starts from zero, so seeded labels are
        # offset here rather than inside it.
        seeded = dict(recorded)
        for name, value in init.items():
            if name in seeded:
                seeded[name] = (
                    seeded[name] + value
                    if evolution == "none"
                    else np.asarray(seeded[name]) + value
                )
            else:
                seeded[name] = value
        return seeded

    def flip_grad_axis(self, axis: str) -> None:
        """Invert every gradient on ``axis``, in place.

        ``mod_grad_axis(axis, -1)``, which is what upstream's own implementation
        does too.
        """
        self.mod_grad_axis(axis, -1.0)

    def mod_grad_axis(self, axis: str, modifier: float) -> None:
        """Scale every gradient on ``axis`` by ``modifier``, in place.

        Runs on the same C++ path :class:`~pulserver.pypulseq.TransformFOV`
        uses for its ``scale``: a gradient is stored as a normalised shape
        beside a scalar amplitude, so scaling an axis multiplies one number per
        row and touches no waveform. The cost is the number of distinct
        gradients, not the length of the scan.

        Unlike upstream's, this does not refuse a sequence whose gradients are
        shared between axes -- there is no such sharing here, because an
        amplitude belongs to the row rather than to the shape.
        """
        if axis not in ("x", "y", "z"):
            raise ValueError(f"axis must be 'x', 'y' or 'z', not {axis!r}")
        factors = [1.0, 1.0, 1.0]
        factors["xyz".index(axis)] = float(modifier)
        self._native.apply_fov_scale(*factors)
        self._touch()

    def copy_definitions(self, other: Sequence) -> None:
        """Replace this sequence's definitions with another's.

        MATLAB's ``copyDefinitions``; PyPulseq has no equivalent.
        """
        self._definitions = dict(other.definitions)
        self._touch()

    def sound(self, *args: object, **kwargs: object):
        """Play the sequence through the speaker. Not ported, deliberately.

        MATLAB's ``sound()`` renders the gradient waveforms as audio so a human
        can listen for something wrong. The rigorous version of that question
        is :meth:`calculate_gradient_spectrum` with ``resonance_lines=True``,
        which answers it against the scanner's own forbidden bands instead of
        against an ear.
        """
        raise NotImplementedError(
            "Sequence.sound is MATLAB's acoustic preview and is not ported -- use "
            "calculate_gradient_spectrum(tr=..., resonance_lines=True) for the "
            "measured version of the same question."
        )

    def find_block_by_time(self, t: float) -> int | None:
        """The 1-based index of the block being played at time ``t`` seconds.

        ``None`` when ``t`` falls outside the sequence, which is what upstream
        documents.

        The public face of the search every ``time_range`` argument already
        runs -- :meth:`_blocks_over` is the same ``searchsorted`` over the
        cumulative block durations, and is what the analysis methods use.

        Notes
        -----
        **This deliberately does not reproduce upstream's answer, which is
        wrong.** ``pypulseq.Sequence.find_block_by_time`` returns the result of
        a zero-based ``searchsorted`` and then indexes a one-based dictionary
        with it, so it is off by one everywhere and raises ``KeyError: 0`` for
        any time inside the *first* block. There is no ``compat`` flag here
        because there is nothing worth being compatible with.
        """
        moment = float(t)
        edges = np.concatenate(([0.0], np.cumsum(self._native.block_durations())))
        if edges.size < 2 or moment < 0.0 or moment >= edges[-1]:
            return None
        return int(np.searchsorted(edges, moment, side="right"))

    def install(
        self, target: str | None = None, clear_cache: bool = False, **kwargs
    ) -> None:
        """Copy the sequence to a scanner. Not ported; see upstream
        :meth:`pypulseq.Sequence.sequence.Sequence.install`.

        Deliberately not implemented rather than pending: upstream's installer
        speaks to Siemens scanners over their own transports, and this project
        reaches its scanner through the interpreter and the PSD instead. Write
        the file and take it there.
        """
        raise NotImplementedError(
            "Sequence.install is upstream PyPulseq's Siemens scanner transfer and has "
            "no meaning here -- write() the sequence and deploy it through the "
            "interpreter instead."
        )

    def paper_plot(
        self,
        time_range: tuple[float] = (0, np.inf),
        line_width: float = 1.2,
        axes_color: tuple[float] = (0.5, 0.5, 0.5),
        rf_color: str = "black",
        gx_color: str = "blue",
        gy_color: str = "red",
        gz_color: tuple[float] = (0, 0.5, 0.3),
        rf_plot: str = "abs",
    ):
        """Upstream's publication-style pulse-diagram plot, over a window.

        Drawn by upstream from the same decoded window :meth:`plot` uses, so
        rotations and RF shims are resolved into what it shows.
        """
        start, stop = _span(time_range)
        first, last = self._blocks_over(start, stop)
        return self._upstream_window(first, last).paper_plot(
            time_range=time_range,
            line_width=line_width,
            axes_color=axes_color,
            rf_color=rf_color,
            gx_color=gx_color,
            gy_color=gy_color,
            gz_color=gz_color,
            rf_plot=rf_plot,
        )

    def get_extension_type_ID(self, extension_string: str) -> int:
        """The numeric id for an extension name, assigning one if it is new."""
        return int(self._native.extension_type_id(str(extension_string)))

    def get_extension_type_string(self, extension_id: int) -> str:
        """The extension name for a numeric id."""
        name = self._native.extension_type_name(int(extension_id))
        if not name:
            raise ValueError(f"Extension for the given ID {extension_id} is unknown")
        return name

    def set_extension_string_ID(self, extension_str: str, extension_id: int) -> None:
        """Bind an extension name to a numeric id."""
        native = self._native
        if native.find_extension_type_id(
            str(extension_str)
        ) or native.extension_type_name(int(extension_id)):
            raise ValueError("Numeric or string ID has already been used")
        native.set_extension_type_id(str(extension_str), int(extension_id))
        self._touch()

    def get_raw_block_content_IDs(self, block_index: int) -> SimpleNamespace:
        """One block's raw library ids, without decoding any of them.

        The block table already *is* this, so it is read straight out rather
        than assembled: the row is ``(rf, gx, gy, gz, adc, extension)``, and
        upstream's leading legacy delay id is always zero from Pulseq 1.4 on.
        """
        if not 1 <= int(block_index) <= self.num_blocks:
            raise IndexError(
                f"no block {block_index} in a sequence of {self.num_blocks}"
            )
        row = self._native.block_events()[int(block_index) - 1]
        return SimpleNamespace(
            block_duration=float(self._native.block_durations()[int(block_index) - 1]),
            rf=int(row[0]),
            gx=int(row[1]),
            gy=int(row[2]),
            gz=int(row[3]),
            adc=int(row[4]),
            ext=int(row[5]),
        )

    @property
    def sequence_descriptor(self):
        """The state-machine description of one repetition time.

        Returns a :class:`~pulserver.recon.simulation.SequenceDescription`: one
        event per block over the canonical pass, each carrying what a Bloch or
        EPG simulation needs and nothing it does not -- RF use, flip
        amplitude, phase and frequency for the pulses; role, k-space-zero
        timing and echo flag for the readouts.

        Notes
        -----
        **This is the same object the reconstruction receives**, not a
        parallel description of it.
        :func:`~pulserver.recon.simulation.decode_sequence_description` builds one
        from the SEQDESC waveforms the scanner streams; this builds one from a
        sequence that has not been written yet, let alone run. Both come out of
        ``pulseg_get_sequence_description`` in the C core, so a simulation
        driven from a design script and a simulation driven from a scan see
        the same numbers -- which is the only reason comparing them means
        anything.

        **Gradient waveforms are never decompressed here.** A simulation wants
        flip angles, repetition times, phases and ADC roles; making it pay for
        the waveform decompression it would immediately discard is the one
        performance mistake this method could make.

        The scan structure it is built from is cached against the sequence's
        revision, so asking repeatedly costs one detection.
        """
        from ..recon._seqdesc import (
            EventType,
            RfDefinition,
            RfShape,
            SequenceDescription,
            SequenceEvent,
        )
        from .._ext.pulseg import (
            _get_rf_definitions,
            _get_sequence_description,
        )

        structure = self._structure_for("sequence_descriptor")
        raw = _get_sequence_description(structure.collection, 0)

        events = tuple(
            SequenceEvent(
                type=EventType(int(kind)),
                timestamp_us=float(stamp),
                params=tuple(float(value) for value in row),
            )
            for kind, stamp, row in zip(
                raw["type"], raw["timestamp_us"], raw["params"], strict=False
            )
        )

        definitions = {}
        for entry in _get_rf_definitions(structure.collection, 0):
            channels = max(int(entry["num_channels"]), 1)
            magnitude = np.asarray(entry["magnitude"], dtype=np.float32)
            # Channel-major; a simulation reads the first transmit channel,
            # and a shim is a per-channel scaling on top of it.
            samples = magnitude.size // channels
            phase = np.asarray(entry["phase_turns"], dtype=np.float32)
            times = np.asarray(entry["time_us"], dtype=np.float32)

            definitions[int(entry["rf_def_id"])] = RfDefinition(
                id=int(entry["rf_def_id"]),
                # Bandwidth is a cache-section field, derived when the
                # description is written for recon rather than held on the
                # collection. Nothing a simulation does needs it -- it
                # classifies slice selectivity -- so it is left at zero rather
                # than guessed at.
                bandwidth_hz=0.0,
                num_bands=1,
                # Eight slots, all zero -- the width the SEQDESC wire format
                # carries (PULSEG_MAX_BANDS), so a description built here and
                # one decoded from a scanner stream compare equal instead of
                # differing over a field neither of them uses.
                band_frequency_offsets_hz=(0.0,) * 8,
                band_bandwidth_hz=0.0,
                total_b1sq_power=0.0,
                magnitude=RfShape(samples, magnitude[:samples]),
                phase=RfShape(samples, phase[:samples]) if phase.size else None,
                time=RfShape(times.size, times) if times.size else None,
            )

        return SequenceDescription(
            subsequence_index=int(raw["subseq_idx"]),
            tr_duration_us=float(raw["tr_duration_us"]),
            events=events,
            rf_definitions=definitions,
            shim_definitions={},
        )

    def sequence_parameters(self) -> dict:
        """Scan-global timing and flip-angle extremes, from the C core.

        ``min_te_us``, ``min_tr_us``, ``max_tr_us``, ``max_flip_angle_deg`` and
        ``total_scan_time_us``, aggregated over every subsequence.
        """
        from .._ext.pulseg import _get_sequence_parameters

        return _get_sequence_parameters(
            self._structure_for("sequence_parameters").collection
        )

    def rf_from_lib_data(self, lib_data: list, use: str = "") -> SimpleNamespace:
        """Decode one raw RF library row into an event, upstream's way.

        The row layout is shared, so this is upstream's own decoder run against
        this sequence's rasters rather than a second implementation of it.
        """
        carrier = pp.Sequence(system=self.system)
        carrier.rf_raster_time = self._native.rf_raster_time
        carrier.shape_library = self._upstream_window(1, self.num_blocks).shape_library
        return carrier.rf_from_lib_data(lib_data, use)

    # -- internals -------------------------------------------------------

    def _segment_blocks(self, segment_idx: int) -> tuple[int, int]:
        """Segment ``segment_idx`` as a 1-based inclusive block range.

        The blocks the C core reports are its **highest-energy** instance --
        ``pulseg_get_subseq_segment_block_indices`` resolves them through the
        execution stream when it can -- so this names the repetition the safety
        checks were run against, not the first one played.
        """
        segments = self._structure_for("plot").segments
        index = int(segment_idx)
        if not 0 <= index < len(segments):
            raise ValueError(
                f"segment_idx={index} is out of range; the sequence holds "
                f"{len(segments)} segments"
            )

        indices = [int(value) for value in segments[index]["block_indices"]]
        if not indices:
            raise ValueError(f"segment_idx={index} reports no blocks")
        # The core hands back a run, and a plot is a window over a timeline;
        # a hypothetical gapped segment would have to be drawn as pieces, and
        # silently drawing its hull instead would be a lie about the picture.
        if indices != list(range(indices[0], indices[0] + len(indices))):
            raise NotImplementedError(
                f"segment {index} is not a contiguous run of blocks ({indices[:8]}...), "
                "which plot() has no way to draw as one window"
            )
        return indices[0] + 1, indices[-1] + 1

    def _blocks_over(self, start: float, stop: float) -> tuple[int, int]:
        """The blocks overlapping ``start..stop`` seconds, 1-based inclusive."""
        edges = np.concatenate(([0.0], np.cumsum(self._native.block_durations())))
        count = edges.size - 1
        if count == 0:
            return 1, 0
        first = min(
            max(int(np.searchsorted(edges, start, side="right")) - 1, 0), count - 1
        )
        last = min(
            max(int(np.searchsorted(edges, stop, side="left")) - 1, first), count - 1
        )
        return first + 1, last + 1

    def _upstream_window(self, first: int, last: int) -> pp.Sequence:
        """Blocks ``first..last`` as a :class:`pypulseq.Sequence` upstream can read.

        The transfer itself is :func:`~._pulseqpp.to_upstream`, a bulk copy of
        the rows the window names. What happens here first is the part upstream
        has no vocabulary for: a block carrying a rotation or an RF shim is
        replayed with that resolved into its events, because PyPulseq's
        ``get_block`` raises on an extension it does not know, and because the
        resolved form is the one worth looking at.

        Materialising costs a decode and a re-add per block, so it is done only
        for a window that needs it -- and into a sequence of *this* class, whose
        ``add_block`` is the C++ one, rather than through PyPulseq's.
        """
        if not self._rotated_or_shimmed(first, last):
            return to_upstream(self, first=first, last=last)

        edges = np.cumsum(self._native.block_durations())
        played = Sequence(system=self.system)
        for index in range(first, last + 1):
            block = self.get_block(index)
            events = _playable(block, self.system)
            if block.block_duration > 0:
                events.append(pp.make_delay(block.block_duration))
            played.add_block(*events)

        return to_upstream(
            played,
            numbers=np.arange(first, last + 1),
            lead_in=float(edges[first - 2]) if first > 1 else 0.0,
        )

    def _rotated_or_shimmed(self, first: int, last: int) -> bool:
        """Whether blocks ``first..last`` carry a rotation or an RF shim.

        The libraries answer for the whole sequence in one look, and for almost
        every sequence they answer no. Only when one of them holds something
        does this walk the window's own chains, and then only the distinct ones:
        deduplication leaves a scan sharing a handful of them.
        """
        if not (self._native.num_rotations() or self._native.num_rf_shims()):
            return False

        heads = np.unique(self._native.block_events()[first - 1 : last, 5])
        for head in heads[heads > 0].tolist():
            node = head
            while node:
                type_id, _, node = self._native.extension_row(node)
                if self._native.extension_type_name(type_id) in (
                    "ROTATIONS",
                    "RF_SHIMS",
                ):
                    return True
        return False

    def _clone(self) -> Sequence:
        """An independent copy: same blocks and libraries, nothing shared.

        The libraries are copied in C++, where they are flat arrays, so this
        costs one allocation and one memcpy each rather than a walk over
        millions of Python objects. What the copy does *not* inherit is the
        event path's shape-identity cache -- see the binding -- so an event
        reused against both sequences registers its waveform once in each.
        """
        made = Sequence.__new__(Sequence)
        made.system = self.system
        made._native = self._native.copy()
        made._definitions = dict(self._definitions)
        # Not inherited: a structure cache belongs to the sequence it was
        # derived from, and the copy is free to diverge from it immediately.
        made._structure = None
        made._revision = 0
        return made

    def _publish_definitions(self) -> None:
        """Hand the definitions across, rasters and total duration included."""
        entries: dict[str, object] = {
            key: float(getattr(self.system, attribute))
            for key, attribute in _RASTER_DEFINITIONS
        }
        entries["TotalDuration"] = self._native.duration()
        entries.update(self._definitions)

        for key, value in entries.items():
            if isinstance(value, str):
                self._native.set_definition_text(key, value)
                continue
            numbers = np.atleast_1d(np.asarray(value)).ravel()
            items = value if isinstance(value, (list, tuple)) else numbers.tolist()
            # A Python int is what says the value was meant as a count; a NumPy
            # float holding a whole number is still a measurement.
            whole = all(
                isinstance(item, (int, np.integer)) and not isinstance(item, bool)
                for item in np.atleast_1d(items).tolist()
            )
            self._native.set_definition_numbers(
                key, [float(item) for item in numbers], whole
            )


# %% local subroutines


def _signature_of(payload: bytes) -> str | None:
    """The hash the writer put in ``[SIGNATURE]``, read back off the bytes."""
    marker = b"\nHash "
    at = payload.rfind(marker)
    return None if at < 0 else payload[at + len(marker) :].split()[0].decode("ascii")


#: Which fields each kind of timing finding reports, beyond the four every
#: one carries. The sets are upstream's: its message templates format exactly
#: these names, and its printer formats `vars(entry)`.
_TIMING_FIELDS = {
    "RASTER": ("value", "value_rounded", "error", "raster"),
    "NEGATIVE_DELAY": ("value",),
    "BLOCK_DURATION_MISMATCH": ("value", "duration"),
    "RF_DEAD_TIME": ("value", "dead_time"),
    "RF_RINGDOWN_TIME": ("value", "duration", "ringdown_time"),
    "ADC_DEAD_TIME": ("value", "dead_time"),
    "POST_ADC_DEAD_TIME": ("value", "duration", "dead_time"),
    "SOFT_DELAY_FACTOR": ("value", "hint", "numID"),
    "SOFT_DELAY_DUR_INCONSISTENCY": ("value", "hint", "numID"),
}


def _timing_entry(finding: dict) -> SimpleNamespace:
    """One compiled timing finding as the report entry upstream would write."""
    entry = {
        "block": finding["block"],
        "event": finding["event"],
        "field": finding["field"],
        "error_type": finding["error_type"],
    }
    for name in _TIMING_FIELDS.get(finding["error_type"], ()):
        entry[name] = finding[name]
    return SimpleNamespace(**entry)


def _playable(block: SimpleNamespace, system: pp.Opts) -> list:
    """A decoded block's events, with rotation and RF shim resolved into them.

    What comes back is what the scanner plays rather than what the file stores:
    the gradients are rotated, and the RF is spread across the transmit channels
    its shim weights.

    Labels and soft delays are left out. They describe how a console drives the
    sequence, not what it emits, and the block keeps its duration either way.
    """
    events = []
    if block.rf is not None:
        events.append(
            _shimmed(block.rf, block.rf_shim) if block.rf_shim is not None else block.rf
        )

    gradients = [axis for axis in (block.gx, block.gy, block.gz) if axis is not None]
    if gradients and block.rotation is not None:
        matrix = Rotation.from_quat(block.rotation, scalar_first=True).as_matrix()
        gradients = rotate3D(*gradients, rotation_matrix=matrix, system=system)
    events.extend(gradients)

    if block.adc is not None:
        events.append(block.adc)
    events.extend(block.triggers)
    return events


def _shimmed(rf: SimpleNamespace, shim: np.ndarray) -> SimpleNamespace:
    """``rf`` replicated across transmit channels, weighted by ``shim``.

    The per-channel time axes are concatenated rather than offset, so every
    channel is drawn over the same interval and upstream needs no pTx awareness
    to plot it.
    """
    weights = np.asarray(shim, dtype=complex).ravel()
    if weights.size == 0:
        return rf
    signal = np.asarray(rf.signal).ravel()
    spread = dict(vars(rf))
    spread["signal"] = (weights[:, None] * signal[None, :]).ravel()
    spread["t"] = np.tile(np.asarray(rf.t, dtype=float).ravel(), weights.size)
    return SimpleNamespace(**spread)
