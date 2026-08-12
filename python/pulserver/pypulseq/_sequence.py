"""A Pulseq sequence whose libraries, block table and file formats live in C++.

The class here is a thin, documented face on ``pulseq::Sequence``. Blocks go in
through :meth:`Sequence.add_block` exactly as they would with PyPulseq's own
sequence, and come back out through :meth:`Sequence.get_block`; everything
between -- the event libraries, deduplication, shape compression, the text and
binary writers, the reader -- is C++.

Two things follow from that, and they are the point of the design.

**Nothing is checked against a library on the way in.** A block registers a row
per event and a shape per waveform it has not seen before, and
:meth:`Sequence.remove_duplicates` collapses the result when the sequence is
finished. Searching the libraries per event would make building a scan
quadratic in the thing that is already the largest.

**Analysis decodes a window, not a scan.** PyPulseq's plotting, k-space and
waveform code is upstream's and worth keeping, so it is given a real
:class:`pypulseq.Sequence` -- built out of the blocks the caller actually asked
about. ``seq.plot(time_range=(0.2, 0.3))`` costs a tenth of a second of blocks,
not the whole protocol. The window has rotations resolved into its gradients
and RF shims expanded across transmit channels, so what it describes is what
the scanner plays rather than the base waveform the file stores.

Building that window is a **bulk copy, not a replay**. A Pulseq 1.5 library row
is exactly what PyPulseq keeps in ``rf_library.data[id]``, so the rows cross as
they stand, under the ids they already have -- see
:func:`~._pulseqpp.to_upstream`. Only a block carrying a rotation or an RF shim
is decoded and re-added one at a time, because that is the one case where the
row is not what should be looked at.

**Registering an event by hand is not needed here, but it works.** Upstream's
``seq.register_grad_event(g)`` / ``seq.register_rf_event(rf)`` exist so a
caller can pre-register a large, repeatedly-used waveform once and reuse its
id -- without that, every ``add_block`` would re-hash and re-store the
samples. This class never does that re-hashing: an event carries the shape ids
this sequence issued for it, and a waveform is remembered by the identity of
the array behind it rather than by its contents, so building a gradient once
and passing the same object to :meth:`Sequence.add_block` a million times
registers it on the first call only.

The ``register_*_event`` methods are still here, and still do something --
they move that first registration off whichever loop iteration happened to
come first, which is what the upstream call is for -- so a PyPulseq script
runs unchanged, ``event.id = seq.register_grad_event(event)`` included. What
they do not hand back is an event-library row id: rows are appended per block
and renumbered by :meth:`Sequence.remove_duplicates`, so there is no number
that would still be true by the time the file is written.

**The scan structure -- repetition times, segments -- is derived on demand,
not carried.** :meth:`Sequence.calculate_pns`,
:meth:`Sequence.calculate_gradient_spectrum` and :meth:`Sequence.plot` can be
asked about a repetition time rather than about a stretch of the timeline,
and when they are, the C safety library recovers the TR from the serialised
sequence and the answer comes from the same code the scanner runs at
predownload. That structure is cached behind a revision counter and rebuilt
whenever the sequence changes, so it can never describe blocks that are gone.

The two questions are genuinely different and the class does not blur them.
Upstream's methods analyse the timeline: a window of blocks, played once,
from rest. The safety core analyses one canonical TR whose amplitudes are the
per-sample maximum over every instance of it -- a waveform that appears
nowhere on the timeline -- and evaluates it periodically. So ``tr=None``, the
default, is upstream PyPulseq to the bit, and the other answer has to be
asked for by name.

``plot`` takes the same ``tr`` for the same reason the other two do, and gets
its picture from the same call: whichever canonical TR the checks were run
against is the one that can be looked at, gradients, RF and ADC together.

What is here matches upstream PyPulseq's
:class:`~pypulseq.Sequence.sequence.Sequence` method for method, including
several whose implementation is not built yet (:meth:`calculate_kspace`,
:meth:`waveforms`, and others below) -- each raises
:exc:`NotImplementedError` with upstream's exact signature, so the class's
*shape* already matches what it will grow into.
"""

from __future__ import annotations

__all__ = ["Sequence"]

import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pypulseq as pp
from scipy.spatial.transform import Rotation

from .._ext import _pulseqpp_wrapper as _cxx
from . import _results, _safety
from ._pulseqpp import to_upstream
from ._results import RF_USES, AdcTimes, RfTimes, Waveforms, WaveformsAndTimes
from ._rotate3d import rotate3D
from ._transform_fov import TransformFOV

#: Definitions written for every sequence, taken from the system it was built
#: with. A caller's own value for one of these wins.
_RASTER_DEFINITIONS = (
    ("AdcRasterTime", "adc_raster_time"),
    ("BlockDurationRaster", "block_duration_raster"),
    ("GradientRasterTime", "grad_raster_time"),
    ("RadiofrequencyRasterTime", "rf_raster_time"),
)

#: Pulseq's trigger numbering, as ``kind -> channel number -> name``.
_TRIGGER_CHANNELS = {
    1.0: {1.0: "osc0", 2.0: "osc1", 3.0: "ext1"},
    2.0: {1.0: "physio1", 2.0: "physio2"},
}

#: What every method whose implementation isn't ported yet says. The
#: signature above each of these already matches upstream PyPulseq's; only
#: the body is missing.
#: Upstream's default gradient-spectrum window, in seconds. Needed only to
#: tell a caller who chose a window from one who left it alone, since under
#: ``tr`` the window is the repetition time and nothing else.
_UPSTREAM_WINDOW_WIDTH = 0.05

def _per_axis(value, name: str) -> list[float]:
    """One number or three, always as three.

    Upstream takes either for ``trajectory_delay`` and ``gradient_offset``,
    and the C core takes three, so the widening happens once here rather than
    at each call site.
    """
    array = np.atleast_1d(np.asarray(value, dtype=float)).ravel()
    if array.size == 1:
        return [float(array[0])] * 3
    if array.size == 3:
        return [float(v) for v in array]
    raise ValueError(f"{name} must be one value or three, got {array.size}")


#: Blocks past which looking at a whole sequence is said out loud rather than
#: simply attempted. Drawing this many is minutes of Matplotlib, and the caller
#: who did not pass a ``time_range`` almost certainly did not mean to.
_LOUD_ABOVE = 50_000


class Sequence:
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
    >>> seq.remove_duplicates()
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
        self.system = system if system is not None else pp.Opts.default
        self._native = _cxx.Sequence()
        self._native.set_rasters(
            float(self.system.rf_raster_time),
            float(self.system.grad_raster_time),
            float(self.system.adc_raster_time),
            float(self.system.block_duration_raster),
        )
        self._definitions: dict[str, object] = {}
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

    def set_definition(self, key: str, value: str | float | int | list | np.ndarray) -> None:
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
            np.count_nonzero(events, axis=0) if events.size else np.zeros(events.shape[1] or 6, dtype=np.int64)
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

    def set_block(self, index: int, *events: object) -> None:
        """Replace block ``index`` with one playing ``events``.

        Parameters
        ----------
        index : int
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
        self._native.set_block_events(index, *events)

    def get_block(self, index: int) -> SimpleNamespace:
        """Block ``index``, decoded back into PyPulseq events.

        Parameters
        ----------
        index : int
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
        rf_id, gx_id, gy_id, gz_id, adc_id, ext_id, duration = self._native.get_block(index)
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

        existing = self._find_row(self._native.num_rotations, self._native.rotation_row, quaternion)
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
        shim = np.asarray(
            getattr(event, "shim_vector", event), dtype=complex
        ).ravel()
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

    def remove_duplicates(self) -> None:
        """Collapse every library to its distinct rows and renumber the blocks.

        Rows are compared at the precision the file writes them, so two events
        that would serialise identically become one. Idempotent.
        """
        self._touch()
        self._native.remove_duplicates()

    def expand_repeats(
        self,
        repeats: int,
        *,
        label: str = "AVG",
        strip_once: bool = True,
        ignore_averages: bool = True,
    ) -> dict[str, int]:
        """Play the sequence ``repeats`` times, written into the block table.

        A ``.seq`` describes one pass; playing it several times is normally
        left to the interpreter, which takes the count from outside the file
        (on a GE scanner, ``opnex``) and uses the ``ONCE`` flag to decide what
        belongs to a single pass:

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

        Call it like :meth:`remove_duplicates`: once, on a finished sequence,
        before writing.

        Parameters
        ----------
        repeats : int
            How many times the body plays, at least 1. ``1`` is not a no-op:
            it resolves the flags and writes ``IgnoreAverages``, leaving a
            file that says a single pass is all there is.
        label : str, default "AVG"
            Counter stamped with the repetition index, or ``""`` for none.
            ``AVG`` because the repetition an interpreter adds is a signal
            average — the same acquisition, sampled again. ``REP`` is the
            frame counter of a dynamic series and means something else. Set
            where it changes, so it costs one extension per repetition.
        strip_once : bool, default True
            Drop the ``ONCE`` labels once resolved — they now describe a
            table they no longer fit, which is what a foreign interpreter
            needs. Against Pulserver's own interpreter, keeping them costs
            nothing at playback and preserves the preparation/cooldown split
            that ``pulseg`` hands to its TR descriptor.
        ignore_averages : bool, default True
            Write ``IgnoreAverages 1`` into ``[DEFINITIONS]``, so a Pulserver
            interpreter does not repeat what is already written out.

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
        >>> report = seq.expand_repeats(3)
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
            raise ValueError(f"expand_repeats(): repeats must be at least 1, got {repeats}")
        self._touch()
        report = self._native.expand_repeats(int(repeats), label, strip_once, ignore_averages)
        if ignore_averages:
            # Mirrored, not duplicated work: the C++ side writes it so a
            # non-Python caller gets it too, and this is what makes it show up
            # in `definitions` rather than only in the written file.
            self._definitions["IgnoreAverages"] = 1
        return report

    # -- FOV positioning --------------------------------------------------

    def transform_fov(
        self,
        offset_mm: tuple[float, float, float],
        *,
        mode: str = "native",
    ) -> None:
        """Move the field of view by ``offset_mm``, in **logical** coordinates.

        A shift is a phase, ``dr . k``. Written in the same frame as the
        gradients it is invariant under any rotation applied to both, so it
        needs no knowledge of the prescribed orientation, of rotation
        extensions, or of ``NOROT``. That invariance is why this belongs here
        and not in an interpreter: computing the same phase in the physical
        frame forces every rotation to be undone first.

        Parameters
        ----------
        offset_mm : tuple of float
            ``(dx, dy, dz)`` along the logical readout, phase and slice axes.
            Millimetres, matching how a prescription states them.
        mode : {"native", "server"}, default "native"
            ``"native"`` bakes the shift into both RF and ADC, so the file
            needs nothing downstream -- what to write when sharing a ``.seq``
            with another toolbox. ``"server"`` bakes only the RF and instead
            stores each readout's base k-space trajectory, leaving the ADC
            side to a consumer of ours; that keeps one shape per distinct
            trajectory rather than one per readout, which is what makes a
            large non-Cartesian scan affordable.

        Notes
        -----
        Applied where it stands: call it once, on a finished sequence, before
        writing. Calling it twice applies the shift twice.

        See Also
        --------
        TransformFOV : the whole transformation -- scale and rotation too,
            over a block range, honouring ``NOSCL``/``NOPOS``/``NOROT``. This
            is the shorthand for its commonest use.
        """
        if mode not in ("native", "server"):
            raise ValueError(f"transform_fov(): mode must be 'native' or 'server', got {mode!r}")

        self._touch()
        TransformFOV(translation=offset_mm, server_mode=(mode == "server")).apply_to_sequence(
            self, in_place=True
        )

    # -- files -----------------------------------------------------------

    def write(self, path: str | Path, create_signature: bool = True) -> None:
        """Write the sequence as a ``.seq`` file.

        Parameters
        ----------
        path : str or pathlib.Path
            Where to write.
        create_signature : bool, default True
            Append the ``[SIGNATURE]`` section.

        See Also
        --------
        write_binary : the same sequence in the binary Pulseq format.

        Notes
        -----
        Text, to a file, always -- the reference toolbox's ``write``. The
        binary format has its own method, and it is the only one of the two
        that will write anywhere but a file.
        """
        Path(path).write_bytes(self._to_text(create_signature=create_signature))

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

    def read(self, path: str | Path) -> None:
        """Replace the sequence's contents with the file at ``path``.

        Parameters
        ----------
        path : str or pathlib.Path
            A ``.seq`` text file or a binary Pulseq file. Which one it is is
            decided by the leading bytes, not by the name.

        Notes
        -----
        :attr:`system` is left alone. The raster times the file records are
        adopted by the C++ sequence, since they are what its blocks were laid
        out on.
        """
        self._touch()
        self._native = _cxx.read_file(str(Path(path)))
        self._definitions = dict(self._native.definitions())

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

    def calculate_kspace(
        self,
        trajectory_delay: float | list[float] | np.ndarray = 0.0,
        gradient_offset: float | list[float] | np.ndarray = 0.0,
        *,
        time_range: list[float] | None = None,
        frame: str = "physical",
        sample_window_average: bool = False,
        dense: bool = True,
        compat: bool = True,
    ):
        """Where every ADC sample sits in k-space.

        Returns upstream's five-tuple, ``(k_traj_adc, k_traj, t_excitation,
        t_refocusing, t_adc)``, with ``k_traj_adc`` and ``k_traj`` shaped
        ``(3, n)`` in 1/m and the times in seconds.

        The arithmetic is the C library's -- ``csrc/src/pulseq/pulseq_ktraj.c``,
        the same code the interpreter links -- rather than a second
        implementation in Python. It integrates each distinct gradient shape
        once instead of each block, so the cost follows the number of distinct
        gradients rather than the length of the scan, and it memoizes on a
        per-readout repeat key so a scan that plays one readout a hundred
        thousand times pays for it once.

        Parameters
        ----------
        trajectory_delay : float or array-like, optional
            Gradient timing compensation in seconds, one value or one per
            axis. Shifts the gradient time base only, never the ADC or RF
            times -- those are synchronised with each other by construction.
        gradient_offset : float or array-like, optional
            A constant background gradient in Hz/m, one value or one per axis.
        time_range : list of float, optional
            ``[start, stop]`` in seconds, to analyse part of the sequence.
            The whole of it by default. The blocks this touches are the ones
            decoded, and the times come back on the sequence's own clock.
        frame : {'physical', 'logical'}, optional
            Whether to resolve rotation extensions into the answer.
            ``'physical'``, the default, is what upstream PyPulseq and Pulseq's
            MATLAB ``calculateKspacePP`` both return and what a reconstruction
            needs. ``'logical'`` leaves the rotation out, which is the frame
            :class:`~pulserver.pypulseq.TransformFOV` works in.
        sample_window_average : bool, optional
            Give each sample the k averaged over its dwell rather than k at
            the window's midpoint. An ADC sample integrates for a whole dwell,
            so the average is the coordinate it physically belongs to; the two
            differ by ``dwell**2 / 24 * dg/dt`` and so agree exactly wherever
            the gradient is flat. Off by default, because PyPulseq, MRpro and
            mri-nufft all sample at the midpoint and matching them is what
            makes the answer comparable. Turn it on for a gridder.
        dense : bool, optional
            Also build ``k_traj``. It is the one output whose size grows with
            the duration of the scan rather than with the acquisition, and the
            only one computed by upstream rather than here -- see the note
            below. Pass ``False`` to skip it and get an empty array back.

        Returns
        -------
        tuple
            ``(k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc)``.

        Notes
        -----
        **``k_traj`` comes from upstream PyPulseq, the other four from the C
        core.** The dense trajectory is a picture of the sequence -- what a
        plot draws -- and being able to hand it to code written against
        upstream matters more than computing it quickly. The C core's own
        answer is on the gradient *breakpoint* grid, which describes the same
        curve in five to ten times fewer points; that is the better
        representation and the wrong one to return from a function whose
        contract is upstream's. It is still available through
        :meth:`_kspace`, together with its time base ``t_ktraj``, which this
        tuple has nowhere to put.

        Nothing a reconstruction needs is on that path: ``k_traj_adc`` is
        where the samples actually are, and it is the C core's, agreeing with
        upstream to 2e-13 on a GRE and 2e-12 on an EPI.

        Sequences upstream cannot read -- anything carrying rotation or RF-shim
        extensions -- have no ``k_traj``, and ask for one raises rather than
        quietly returning the breakpoint grid in its place.

        See Also
        --------
        auto_label : the encoding counters derived from this trajectory.
        """
        blocks = self._window_for(time_range)
        result = self._kspace(
            trajectory_delay=trajectory_delay,
            gradient_offset=gradient_offset,
            blocks=blocks,
            frame=frame,
            sample_window_average=sample_window_average,
            # The breakpoint-grid trajectory has nowhere to go in upstream's
            # tuple, so it is only asked for when there is somewhere to put it.
            dense=not compat,
        )

        k_traj = np.zeros((3, 0))
        if dense:
            # to_upstream does not resolve rotation or RF-shim extensions, so
            # for a sequence carrying either it would hand back a logical-frame
            # k_traj beside a physical-frame k_traj_adc -- two frames in one
            # tuple, and nothing to say which is which. Refuse instead.
            if self._native.num_rotations() > 0 or self._native.num_rf_shims() > 0:
                raise NotImplementedError(
                    "calculate_kspace: k_traj comes from upstream PyPulseq, which cannot "
                    "read the rotation or RF-shim extensions this sequence carries -- it "
                    "would come back in the logical frame beside a physical-frame "
                    "k_traj_adc. Use dense=False for the ADC samples, or _kspace() for "
                    "the breakpoint-grid trajectory, which does resolve them."
                )
            first, last = blocks
            upstream = to_upstream(self, first=first, last=(None if last == 0 else last))
            k_traj = upstream.calculate_kspace(
                trajectory_delay=trajectory_delay,
                gradient_offset=gradient_offset,
            )[1]

        if compat:
            return (
                result["k_adc"],
                k_traj,
                result["t_excitation"],
                result["t_refocusing"],
                result["t_adc"],
            )
        return _results.KSpace(
            k_traj_adc=result["k_adc"],
            k_traj=k_traj,
            t_excitation=result["t_excitation"],
            t_refocusing=result["t_refocusing"],
            t_adc=result["t_adc"],
            k_traj_breakpoints=result["k_traj"],
            t_breakpoints=result["t_ktraj"],
            k_center=result["k_center"],
            readout_center_sample=result["readout_center_sample"],
        )

    def _kspace(
        self,
        *,
        trajectory_delay=0.0,
        gradient_offset=0.0,
        blocks=None,
        frame="physical",
        sample_window_average=False,
        dense=True,
    ) -> dict:
        """Everything the C core reports, not just upstream's five entries.

        Kept separate because :meth:`calculate_kspace` has to return exactly
        upstream's tuple, and the derived echo positions, the k-space centre
        and the repeat-key statistics have nowhere in it to go.

        ``blocks`` is the 1-based inclusive pair the public methods resolve
        their ``time_range`` into, not a window of its own.
        """
        if frame not in ("physical", "logical"):
            raise ValueError(f"frame must be 'physical' or 'logical', not {frame!r}")

        first, last = (1, 0) if blocks is None else blocks
        return self._native.calculate_kspace(
            _per_axis(trajectory_delay, "trajectory_delay"),
            _per_axis(gradient_offset, "gradient_offset"),
            first,
            last,
            frame == "physical",
            bool(sample_window_average),
            bool(dense),
        )

    def auto_label(
        self,
        *,
        # -- MATLAB Pulseq's autoLabel parameters, under Python names --------
        time_range: list[float] | None = None,
        use_labels: dict | None = None,
        use_aux: dict | None = None,
        skip_apply: bool = False,
        mirror_fourier: bool = False,
        reflect: list[int] | None = None,
        reorder: list[int] | None = None,
        sort_slices: str = "ascending",
        no_plots: bool = True,
        # -- Pulserver's own, on top -----------------------------------------
        trajectory_delay: float | list[float] | np.ndarray = 0.0,
        repeat_dims: list[str | tuple[str, int]] | None = None,
        skip: list[str] | None = None,
    ) -> tuple[dict, dict]:
        """Recover the encoding counters from the sequence's own trajectory.

        A ``.seq`` written elsewhere carries no ``LABELSET`` extensions, so
        nothing downstream knows which line, partition, slice or repetition an
        acquisition belongs to. It is all still there, written into where the
        readouts sit in k-space, and this reads it back out.

        Same labels as Pulseq's MATLAB ``autoLabel``, by a cheaper route: that
        one walks every ADC sample three times over, and here the echo search
        is memoized per distinct readout and the rest reduces to one point per
        readout, so nothing scales with the number of samples.

        Every ``autoLabel`` parameter is accepted, under the Python spelling
        of its name and in its own order; Pulserver's additions come after
        them. Two defaults differ, and both are called out below --
        ``sort_slices`` and ``no_plots``.

        Parameters
        ----------
        time_range : list of float, optional
            ``[start, stop]`` in seconds. MATLAB's ``blockRange`` in the same
            position, in the unit PyPulseq windows everything else by.
        use_labels : dict, optional
            Skip detection and apply these counters instead -- the labels
            half of a previous call's return value, or a set computed some
            other way. Keys are counter names, values one entry per ADC in
            acquisition order.

            For applying one detection to several variants of a sequence, and
            for correcting a counter by hand without recomputing the rest.
            Cannot be combined with ``reflect``, ``reorder`` or
            ``mirror_fourier``, which only affect detection -- MATLAB refuses
            that combination too, and it would silently do nothing.
        use_aux : dict, optional
            The definitions to write, in the same spirit: the ``aux`` half of
            a previous return. Usable on its own or alongside ``use_labels``.
        skip_apply : bool, optional
            Return the counters without writing them onto the sequence. By
            default they are written, as ``SET`` label extensions on each ADC
            block where the value changes, and the derived definitions
            (``kSpaceCenterLine``, ``SliceThickness`` and the rest) go into
            ``[DEFINITIONS]``.
        mirror_fourier : bool, optional
            Negate every Fourier-encoding direction at once -- readout, phase
            and partition -- for a reconstruction that inverse-transforms
            where this assumes a forward transform.

            Not the same as ``reflect=[0, 1, 2]``, in the one way that
            matters: the slice positions and slice-select gradients are left
            alone, so slice ordering is unaffected. Applied before
            ``reflect``, and freely combined with it.
        reflect : list of int, optional
            Axes (0, 1, 2) whose k, slice positions and gradients to negate
            before deriving anything. Applied before ``reorder``.
        reorder : list of int, optional
            A permutation of the axes, as source indices: ``[1, 0, 2]`` swaps
            x and y.
        sort_slices : {"ascending", "descending", "acquisition"}, optional
            How ``SLC`` is assigned. ``SlicePositions[SLC]`` is the position
            of slice ``SLC`` under all three, so a reconstruction reading the
            pair together is right either way; what changes is which index a
            slice is given.

            **The default differs from MATLAB's**, which is
            ``"acquisition"``. A geometric index is what makes the slice
            table usable as a stack: an interleaved acquisition (0, 2, 4, 1,
            3) hands the reconstruction a shuffled volume under arrival order
            and an ordered one under ``"ascending"``. Pass ``"acquisition"``
            for MATLAB's numbering exactly. ``"descending"`` is what
            ``autoLabel``'s own notes recommend for a Siemens interpreter.
        no_plots : bool, optional
            **The default differs from MATLAB's**, which is ``False``.
            ``autoLabel`` draws diagnostic figures; nothing here does, so
            there is nothing to suppress and ``True`` is the only truthful
            value. Passing ``False`` raises rather than quietly drawing
            nothing -- it is a request for output that will not appear.
        trajectory_delay : float or array-like, optional
            As for :meth:`calculate_kspace`.
        repeat_dims : sequence of str, optional
            The dimensions the repetition counter is standing in for, named
            by you, **outermost loop first** -- ``["REP", "ECO"]``.

            Where a readout sits in k says which line, partition and slice it
            is. It cannot say which echo of a train, which frame of a time
            series or which saturation state it is, because all of those
            revisit the same k-space position -- so by default they are
            counted together as ``REP``.

            Only the names are needed. How large each dimension is, is
            written in the acquisition order and read back from it: a
            dimension nested inside the k-space loop brings a position back
            after a short gap, one outside it only after a whole pass. Pass
            ``("ECO", 2)`` in place of a name to pin a size, and it is
            checked against what was read rather than believed.

            Repeats that are not a rectangle -- some positions revisited and
            others not, as with an EPI's navigators -- have no nest to read
            and raise. A single name never does: it takes the whole count,
            which is ``REP`` under a name that means something.
        skip : list of str, optional
            Counters to leave alone -- derived neither into the answer nor
            onto the sequence.

            For a sequence that labelled some of its own axes as it was
            built and wants the geometric ones filled in around them.
            Labels this does not derive at all (``ECO``, ``SET``, ``AVG``,
            anything custom) already survive an ``auto_label`` pass
            untouched and need no mention. ``REP`` is the one that does:
            it is derived by default, so a design loop that separated its
            own contrasts or frames should pass ``skip=["REP"]`` or its own
            labelling is overwritten by a bare repeat count.

        Returns
        -------
        labels : dict
            Counter name to an array with one value per ADC, in acquisition
            order. Only the counters that vary are present -- a single-slice
            scan has no ``SLC``.
        aux : dict
            The derived definitions.

        Raises
        ------
        RuntimeError
            If the readouts do not share a direction. These are Cartesian
            encoding counters, and a non-Cartesian trajectory has no honest
            value for them -- which is what MATLAB's ``autoLabel`` also says.
            Also if the repeats do not form a rectangle that ``repeat_dims``
            can name, or if a size you pinned contradicts the acquisition
            order.

        Notes
        -----
        ``SLC`` is a geometric index: slices are ranked by the position their
        excitation's frequency offset puts them at, so ``SlicePositions[SLC]``
        is where slice ``SLC`` sits whatever order the scan visited them in.
        Those offsets are read as authored, and :class:`TransformFOV` scaling
        rewrites the slice-select gradient without touching them -- so label
        first and transform second.
        """
        if not no_plots:
            raise ValueError(
                "auto_label(): no_plots=False asks for the diagnostic figures MATLAB's "
                "autoLabel draws, and nothing here draws any. Leave it at True and plot "
                "from the returned labels if you need a picture."
            )
        if sort_slices not in ("ascending", "descending", "acquisition"):
            raise ValueError(
                f"auto_label(): sort_slices must be 'ascending', 'descending' or "
                f"'acquisition', got {sort_slices!r}"
            )

        first, last = self._window_for(time_range)

        # Detection-only options against a caller who has skipped detection.
        # MATLAB raises on the same combination, and for the same reason: it
        # would look like it did something.
        if (use_labels is not None or use_aux is not None) and (
            reflect or reorder or mirror_fourier
        ):
            raise ValueError(
                "auto_label(): reflect, reorder and mirror_fourier only affect detection, "
                "so they cannot be combined with use_labels or use_aux."
            )

        if use_labels is not None or use_aux is not None:
            labels = dict(use_labels or {})
            aux = dict(use_aux or {})
            if not skip_apply:
                blocks = [
                    index
                    for index in range(first, (last or self._native.num_blocks()) + 1)
                    if self._native.block_events()[index - 1][4] != 0
                ]
                ordered = [
                    (name, [int(v) for v in np.atleast_1d(values)])
                    for name, values in labels.items()
                ]
                for name, values in ordered:
                    if len(values) != len(blocks):
                        raise ValueError(
                            f"auto_label(): use_labels['{name}'] has {len(values)} values "
                            f"for {len(blocks)} ADCs in range"
                        )
                self._native.apply_labels(blocks, ordered, aux)
                for key, value in aux.items():
                    self.set_definition(
                        key, value.tolist() if hasattr(value, "tolist") else value
                    )
                self._touch()
            return labels, aux

        reflect_mask = [False, False, False]
        for axis in reflect or ():
            if axis not in (0, 1, 2):
                raise ValueError(f"reflect axes must be 0, 1 or 2, not {axis!r}")
            reflect_mask[axis] = True

        order = [0, 1, 2]
        if reorder is not None:
            if sorted(reorder) != list(range(len(reorder))) or len(reorder) not in (2, 3):
                raise ValueError(f"reorder must permute the first 2 or 3 axes, got {reorder!r}")
            order[: len(reorder)] = list(reorder)

        dims = []
        for entry in repeat_dims or ():
            # A bare name is the ordinary case; a (name, size) pair pins one
            # down. Strings are iterable, so they have to be caught first --
            # unpacking "AB" would otherwise succeed and mean nothing.
            if isinstance(entry, str):
                dims.append((entry, 0))
                continue
            try:
                name, size = entry
            except (TypeError, ValueError):
                raise ValueError(
                    f"repeat_dims entries are names, or (name, size) pairs to pin a size, "
                    f"got {entry!r}"
                ) from None
            dims.append((str(name), int(size)))

        result = self._native.auto_label(
            first,
            last,
            reflect_mask,
            order,
            _per_axis(trajectory_delay, "trajectory_delay"),
            not skip_apply,
            dims,
            [str(name) for name in (skip or ())],
            bool(mirror_fourier),
            sort_slices,
        )
        aux = result["aux"]
        if not skip_apply:
            # The C++ side wrote these onto the native sequence, which is right
            # for a C++ caller and invisible here: this class keeps its own
            # definitions and pushes them across when the sequence is written,
            # so anything only the native side knows would be overwritten on
            # the way out. Mirroring them is what makes them survive.
            for key, value in aux.items():
                self.set_definition(key, value.tolist() if hasattr(value, "tolist") else value)
            self._touch()
        return result["labels"], aux

    def calculate_kspacePP(
        self,
        # Present only so the signature matches upstream's; nothing reads them.
        trajectory_delay: float | list[float] | np.ndarray = 0,  # noqa: ARG002
        gradient_offset: float | list[float] | np.ndarray = 0,  # noqa: ARG002
    ):
        """Deprecated upstream; raises instead of forwarding."""
        raise DeprecationWarning(
            "Sequence.calculate_kspacePP has been deprecated, use calculate_kspace instead"
        )

    def waveforms(
        self,
        append_RF: bool = False,
        time_range: list[float] | None = None,
        *,
        compat: bool = True,
    ):
        """The gradient waveforms, decompressed onto one time axis per channel.

        Parameters
        ----------
        append_RF : bool, optional
            Append the complex RF waveform as a fourth channel.
        time_range : list of float, optional
            ``[start, stop]`` in seconds. The whole sequence by default.
        compat : bool, optional
            Upstream's return -- a list of ``(2, n)`` arrays -- by default.
            ``False`` returns a :class:`~._results.Waveforms` instead, whose
            channels are named rather than positional.

        Notes
        -----
        An arbitrary gradient stored on the centres raster has its raster-edge
        samples reconstructed here, which upstream leaves as a ``TODO``. A
        trapezoid that was converted to a shape therefore comes back with its
        corners where the sequence put them, rather than rounded over a raster
        interval -- exactly, on a fixture, against upstream's 1.25% at the
        corner, and in four samples rather than eighty-two. See
        :class:`~._pulseqpp.RestoringSequence`.
        """
        first, last = self._window_for(time_range)
        channels = self._upstream_window(first, last).waveforms(append_RF=append_RF)

        if compat:
            return channels
        return Waveforms(
            gx=channels[0],
            gy=channels[1],
            gz=channels[2],
            rf=channels[3] if append_RF and len(channels) > 3 else None,
        )

    def waveforms_and_times(
        self,
        append_RF: bool = False,
        time_range: list[float] | None = None,
        *,
        compat: bool = True,
    ):
        """The waveforms, plus when the RF pulses and ADC samples happen.

        Parameters
        ----------
        append_RF : bool, optional
            Append the complex RF waveform as a fourth gradient channel.
        time_range : list of float, optional
            ``[start, stop]`` in seconds. The whole sequence by default.
        compat : bool, optional
            Upstream's five-tuple ``(wave_data, tfp_excitation, tfp_refocusing,
            t_adc, fp_adc)`` by default. ``False`` returns a
            :class:`~._results.WaveformsAndTimes`.

        Notes
        -----
        **Upstream's tuple cannot say everything the sequence knows**, and
        ``compat=False`` is where the rest of it comes out:

        - *Every* RF use, not two. Upstream sorts RF into excitation and
          refocusing and silently drops inversion, saturation, preparation and
          "other" -- an inversion pulse does not appear in its answer at all.
        - ``pm_adc``, the per-sample ADC phase modulation, which MATLAB returns
          as a sixth output and PyPulseq does not return at all.
        - The echo centres, which neither returns. MATLAB reconstructs
          ``2*t_refocusing - t_excitation`` in ``calcMomentsBtensor`` and
          carries its own ``TODO: fixme for double-refocused sequences``; the
          value here is the ADC sample nearest k-space zero, found by the C
          core walking the real trajectory. It is computed on first read, not
          with the rest, because it needs that trajectory.
        """
        first, last = self._window_for(time_range)
        window = self._upstream_window(first, last)

        channels = window.waveforms(append_RF=append_RF)
        # The window carries the time in front of it as a lead-in block
        # numbered 0, so walking it from zero already gives absolute times.
        rf = self._rf_times_of(window, 0.0)
        adc = self._adc_times_of(window, 0.0, block_span=(first, last))

        if compat:
            return (
                channels,
                rf.of("excitation", "undefined").tfp,
                rf.of("refocusing").tfp,
                adc.t,
                adc.fp,
            )
        return WaveformsAndTimes(
            waveforms=Waveforms(
                gx=channels[0],
                gy=channels[1],
                gz=channels[2],
                rf=channels[3] if append_RF and len(channels) > 3 else None,
            ),
            rf=rf,
            adc=adc,
        )

    def rf_times(self, time_range: list[float] | None = None, *, compat: bool = True):
        """When each RF pulse reaches its centre, and with what phase.

        Parameters
        ----------
        time_range : list of float, optional
            ``[start, stop]`` in seconds. The whole sequence by default.
        compat : bool, optional
            Upstream's ``(t_excitation, fp_excitation, t_refocusing,
            fp_refocusing)`` by default, which describes two of Pulseq's seven
            RF uses and drops the rest. ``False`` returns a
            :class:`~._results.RfTimes` covering all of them.
        """
        first, last = self._window_for(time_range)
        window = self._upstream_window(first, last)
        pulses = self._rf_times_of(window, 0.0)

        if not compat:
            return pulses

        # Upstream counts an untagged pulse as an excitation.
        excitation = pulses.of("excitation", "undefined")
        refocusing = pulses.of("refocusing")
        return (
            list(excitation.t),
            np.vstack((excitation.freq_offset, excitation.phase_offset)),
            list(refocusing.t),
            np.vstack((refocusing.freq_offset, refocusing.phase_offset)),
        )

    def adc_times(self, time_range: list[float] | None = None, *, compat: bool = True):
        """When every ADC sample is taken.

        Parameters
        ----------
        time_range : list of float, optional
            ``[start, stop]`` in seconds. The whole sequence by default.
        compat : bool, optional
            Upstream's ``(t_adc, fp_adc)`` by default, where ``fp_adc`` is one
            row per ADC *event* carrying its raw frequency and phase offsets.
            ``False`` returns an :class:`~._results.AdcTimes`, which adds the
            per-*sample* phase -- ppm terms, phase modulation and accumulated
            ``2*pi*f*t`` folded in, the number a demodulator wants.
        """
        first, last = self._window_for(time_range)
        window = self._upstream_window(first, last)
        samples = self._adc_times_of(window, 0.0, block_span=(first, last))
        return samples if not compat else (samples.t, samples.fp)

    def get_gradients(
        self,
        trajectory_delay: float | list[float] | np.ndarray = 0,
        gradient_offset: float | list[float] | np.ndarray = 0,
        time_range: list[float] | None = None,
    ) -> list:
        """The gradients as :class:`scipy.interpolate.PPoly` piecewise polynomials.

        Upstream's, evaluated on this class's waveforms -- so the raster-edge
        reconstruction :meth:`waveforms` performs is in them, and in everything
        built on them.
        """
        first, last = self._window_for(time_range)
        return self._upstream_window(first, last).get_gradients(
            trajectory_delay=trajectory_delay,
            gradient_offset=gradient_offset,
        )

    # -- the walk behind rf_times / adc_times ----------------------------

    def _window_for(self, time_range) -> tuple[int, int]:
        """``time_range`` as a 1-based inclusive block range."""
        if time_range is None:
            return 1, self.num_blocks
        if len(time_range) != 2:
            raise ValueError("Time range must be list of two elements")
        if time_range[0] > time_range[1]:
            raise ValueError("End time of time_range must be after begin time")
        return self._blocks_over(float(time_range[0]), float(time_range[1]))

    def _rf_times_of(self, window: pp.Sequence, elapsed: float) -> RfTimes:
        """Walk a window's RF pulses into one flat table, use tags kept.

        The centre is :func:`pypulseq.calc_rf_center`'s, and the phase carries
        the ``2*pi*f*t_centre`` term to it -- upstream's convention and
        MATLAB's, so ``compat=True`` reproduces upstream exactly.
        """
        from pypulseq.calc_rf_center import calc_rf_center

        gamma_b0 = self.system.gamma * self.system.B0
        times: list[float] = []
        frequencies: list[float] = []
        phases: list[float] = []
        uses: list[str] = []
        blocks: list[int] = []

        for number in window.block_events:
            block = window.get_block(number)
            rf = getattr(block, "rf", None)
            if rf is not None:
                centre = calc_rf_center(rf)[0]
                frequency = rf.freq_offset + rf.freq_ppm * 1e-6 * gamma_b0
                phase = rf.phase_offset + rf.phase_ppm * 1e-6 * gamma_b0

                times.append(elapsed + rf.delay + centre)
                frequencies.append(frequency)
                phases.append(phase + 2 * np.pi * frequency * centre)
                uses.append(_rf_use(rf))
                blocks.append(number)
            elapsed += window.block_durations[number]

        return RfTimes(
            t=np.asarray(times, dtype=float),
            freq_offset=np.asarray(frequencies, dtype=float),
            phase_offset=np.asarray(phases, dtype=float),
            use=tuple(uses),
            block=np.asarray(blocks, dtype=int),
        )

    def _adc_times_of(
        self, window: pp.Sequence, elapsed: float, *, block_span: tuple[int, int]
    ) -> AdcTimes:
        """Walk a window's ADC events into sample times and per-sample phase."""
        gamma_b0 = self.system.gamma * self.system.B0
        sample_times: list[np.ndarray] = []
        sample_phases: list[np.ndarray] = []
        modulations: list[np.ndarray] = []
        frequencies: list[float] = []
        phases: list[float] = []
        blocks: list[int] = []
        counts: list[int] = []

        for number in window.block_events:
            block = window.get_block(number)
            adc = getattr(block, "adc", None)
            if adc is not None:
                count = int(adc.num_samples)
                # Samples sit half a dwell into their window -- Siemens' and
                # Pulseq's shared convention, not a midpoint approximation.
                within = (np.arange(count) + 0.5) * adc.dwell
                frequency = adc.freq_offset + adc.freq_ppm * 1e-6 * gamma_b0
                phase = adc.phase_offset + adc.phase_ppm * 1e-6 * gamma_b0

                modulation = getattr(adc, "phase_modulation", None)
                if modulation is None or len(modulation) == 0:
                    modulation = np.zeros(count)
                modulation = np.asarray(modulation, dtype=float).ravel()

                sample_times.append(elapsed + adc.delay + within)
                sample_phases.append(phase + modulation + 2 * np.pi * frequency * within)
                modulations.append(modulation)
                # Upstream's fp_adc is the raw event offsets, no ppm folded in.
                frequencies.append(adc.freq_offset)
                phases.append(adc.phase_offset)
                blocks.append(number)
                counts.append(count)
            elapsed += window.block_durations[number]

        def _stack(pieces, width=None):
            if pieces:
                return np.concatenate(pieces)
            return np.zeros(0) if width is None else np.zeros((0, width))

        return AdcTimes(
            t=_stack(sample_times),
            freq_offset=np.asarray(frequencies, dtype=float),
            phase_offset=np.asarray(phases, dtype=float),
            phase_modulation=_stack(modulations),
            sample_phase=_stack(sample_phases),
            block=np.asarray(blocks, dtype=int),
            num_samples=np.asarray(counts, dtype=int),
            _echoes=lambda: self._echo_centers(block_span),
        )

    def _echo_centers(self, blocks: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """``(sample index, time)`` of the k-space centre of each readout.

        The C core already reports which sample of each readout is nearest
        k-space zero -- ``readout_center_sample``, which it derives while
        integrating the trajectory. Reading it back is cheaper and more honest
        than re-deriving it here, and it is the same number the recon and the
        scanner use.
        """
        first, last = blocks
        result = self._kspace(blocks=(first, last), dense=False)

        centers = np.asarray(result["readout_center_sample"], dtype=int)
        counts = np.asarray(result["readout_samples"], dtype=int)
        starts = np.concatenate(([0], np.cumsum(counts)[:-1])).astype(int)
        t_adc = np.asarray(result["t_adc"], dtype=float)

        absolute = starts + centers
        inside = (absolute >= 0) & (absolute < t_adc.size)
        times = np.full(absolute.shape, np.nan)
        times[inside] = t_adc[absolute[inside]]
        return centers, times

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
            per-shot tensors in the units and shapes a diffusion pipeline takes.

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
        one, with no special case.

        MATLAB flips the gradient sign once, at the first refocusing pulse in
        the interval. The sign here flips at *every* refocusing pulse, which is
        the same answer for one and the right one for two.

        The waveforms are this class's, so an arbitrary gradient stored on the
        centres raster has its raster-edge samples restored -- see
        :meth:`waveforms`. The integration is exact rather than sampled: a
        Pulseq gradient is piecewise linear, so ``q`` is piecewise quadratic
        and ``q_i q_j`` piecewise quartic, and each piece is integrated in
        closed form.

        See Also
        --------
        Sequence.calculate_kspace : the trajectory the echo positions come from.
        """
        first, last = self._window_for(time_range)
        window = self._upstream_window(first, last)

        channels = window.waveforms()
        pulses = self._rf_times_of(window, 0.0)
        excitations = np.sort(pulses.of("excitation", "undefined").t)
        refocusings = np.sort(pulses.of("refocusing").t)

        if n_dummy:
            excitations = excitations[int(n_dummy) :]
        if excitations.size == 0:
            raise ValueError(
                "calc_moments_btensor(): the window holds no excitation pulse, so there "
                "is nothing to integrate from. A b-tensor is per excitation."
            )

        _, echoes = self._echo_centers((first, last))
        echoes = np.asarray(echoes, dtype=float)
        echoes = np.sort(echoes[np.isfinite(echoes)])

        ends = np.concatenate((excitations[1:], [self._window_end(channels)]))
        moments = [order for order, wanted in ((1, calc_m1), (2, calc_m2), (3, calc_m3)) if wanted]

        b_tensor = np.zeros((excitations.size, 3, 3))
        gradient_moments = {order: np.zeros((excitations.size, 3)) for order in (1, 2, 3)}
        echo_times = np.full(excitations.size, np.nan)

        for shot, (start, limit) in enumerate(zip(excitations, ends)):
            echo = self._echo_for(start, limit, echoes, refocusings)
            echo_times[shot] = echo
            if not np.isfinite(echo) or echo <= start:
                continue

            pieces = _gradient_pieces(channels, start, echo, refocusings)
            if calc_b:
                b_tensor[shot] = _b_tensor_over(pieces)
            for order in moments:
                gradient_moments[order][shot] = _moment_over(pieces, order)

        if compat:
            return (
                b_tensor,
                gradient_moments[1],
                gradient_moments[2],
                gradient_moments[3],
            )
        return _results.BTensor.of(
            B=b_tensor,
            m1=gradient_moments[1],
            m2=gradient_moments[2],
            m3=gradient_moments[3],
            excitation_times=excitations,
            echo_times=echo_times,
        )

    @staticmethod
    def _window_end(channels: list[np.ndarray]) -> float:
        """The last time any gradient channel carries, seconds."""
        ends = [channel[0, -1] for channel in channels if channel.shape[1]]
        return float(max(ends)) if ends else 0.0

    @staticmethod
    def _echo_for(start: float, limit: float, echoes: np.ndarray, refocusings: np.ndarray) -> float:
        """When this shot's echo happens.

        The first measured echo centre in the interval, and only if there is
        none does this fall back to MATLAB's ``2*t_refocusing - t_excitation``
        -- which is there for a sequence with no ADC at all (a design being
        checked before its readout is attached), not as an equal alternative.
        """
        inside = echoes[(echoes > start) & (echoes <= limit)]
        if inside.size:
            return float(inside[0])
        refocused = refocusings[(refocusings > start) & (refocusings < limit)]
        if refocused.size:
            return float(2.0 * refocused[0] - start)
        return float("nan")

    def check_timing(self, print_errors: bool = False):  # noqa: ARG002 - upstream's signature
        """Check every block's timing against the raster. Not ported, deliberately.

        The interpreter checks this itself, at predownload, against the
        scanner's own rasters -- which are the rasters that matter and which a
        design script can only guess at. Running a second check here would
        either agree with it, in which case it is noise, or disagree with it,
        in which case the one that agrees with the hardware is not this one.

        The timing report upstream's own checker produces is still reachable,
        as part of :meth:`test_report`, which runs upstream's analysis on an
        upstream sequence.
        """
        raise NotImplementedError(
            "Sequence.check_timing is not ported: the interpreter runs this check at "
            "predownload against the scanner's own raster times, and a second answer "
            "here could only be the less authoritative one. test_report() carries "
            "upstream's timing report if that is what you want."
        )

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
                raise ValueError(f"window_duration must be positive, not {window_duration!r}")
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

    def _rf_power_of(self, magnitude_id: int, phase_id: int, time_id: int) -> tuple[float, float]:
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

    def calculate_pns(
        self,
        hardware: object,
        time_range: list[float] | None = None,
        do_plots: bool = True,
        *,
        tr: str | int | None = None,
        compat: bool = True,
    ):
        """Peripheral nerve stimulation over the sequence, or over one TR.

        Parameters
        ----------
        hardware : str or pathlib.Path or types.SimpleNamespace or dict
            Which nerve model to use, and its coefficients. A Siemens ``.asc``
            path or a per-axis namespace of the kind
            :func:`pypulseq.utils.safe_pns_prediction.safe_example_hw`
            returns selects **SAFE**, upstream's model. A mapping carrying
            ``chronaxie`` (seconds) and ``rheobase`` (Hz/m/s) selects the
            **Irnich** rheobase/chronaxie model, which is what the GE gate
            applies.
        time_range : list of float, optional
            Two timepoints in seconds bounding what to look at. Only with
            ``tr=None``.
        do_plots : bool, default True
            Draw the gradient waveform and the PNS response. The response
            panel is marked at 100 % of the stimulation threshold and at the
            80 % margin, whichever nerve model produced it -- upstream draws
            neither, only the peak the sequence happened to reach.
        tr : {"worst_case"} or int, optional
            Analyse one repetition time rather than the timeline.

            ``None``, the default, is upstream PyPulseq exactly: the sequence
            as written, played once from rest, zero-padded.

            ``"worst_case"`` is the waveform ``pulseg_check_safety`` judges --
            **not** any TR the scanner plays, but the per-sample maximum over
            every instance of one -- evaluated periodically, with the history
            wrapped round from the end of the TR so the nerve model is warmed
            up rather than starting from rest. This is the number the
            interpreter gates on.

            An integer is that TR instance as it really plays, signed
            amplitudes and all, evaluated the same periodic way. Its use is
            checking the claim ``"worst_case"`` rests on: that no instance
            exceeds the envelope.

        Returns
        -------
        ok : bool
            Whether peak PNS stays under the threshold everywhere.
        pns_norm : numpy.ndarray
            ``(N,)`` PNS over all axes, normalised to 1.
        pns_components : numpy.ndarray
            ``(N, 3)`` PNS per gradient axis, normalised to 1.
        t_pns : numpy.ndarray
            ``(N,)`` the time axis, in seconds.

        Notes
        -----
        Under ``tr=None`` this is upstream's
        :func:`pypulseq.Sequence.calc_pns.calc_pns` called on the blocks
        asked for, so a PyPulseq script gets PyPulseq's numbers.

        Under ``tr=``, the response comes from the same C code the scanner
        runs, and the returned arrays run past one TR: the circularly wrapped
        history the model needed is reported rather than trimmed away, which
        is what makes ``ok`` here the gate's own verdict.
        """
        if tr is None:
            from pypulseq.Sequence.calc_pns import calc_pns

            first, last = self._blocks_over(*_span(time_range if time_range else (0.0, np.inf)))
            answer = calc_pns(
                self._upstream_window(first, last),
                hardware,
                time_range=time_range,
                do_plots=do_plots,
            )
            if do_plots:
                # Upstream draws through pyplot and leaves its PNS panel
                # current, so the thresholds go on afterwards rather than
                # forking its plotting.
                _safety.overlay_pns_thresholds()
            return answer if compat else _results.Pns(*answer)

        if time_range is not None:
            raise ValueError(
                "calculate_pns(): time_range selects part of the timeline and tr selects a "
                "repetition time, which is not on it -- pass one or the other"
            )

        structure = self._structure_for("calculate_pns")
        mode, index = structure.resolve(tr)
        result = self._native_pns(structure, hardware, mode, index)

        # The C core reports per-axis percentage of threshold; upstream
        # normalises to 1. Same quantity, hundredfold apart.
        components = 0.01 * np.stack(
            [np.asarray(result[f"slew_{axis}"], dtype=float) for axis in "xyz"], axis=-1
        )
        norm = np.sqrt((components**2).sum(axis=1))
        raster = _safety.SAFETY_RASTER_FRACTION * float(self.system.grad_raster_time)
        times = np.arange(components.shape[0]) * raster

        if do_plots:
            self._plot_pns(structure.waveform(tr), components, raster)

        verdict = (bool(np.all(norm < 1)), norm, components, times)
        return verdict if compat else _results.Pns(*verdict)

    @staticmethod
    def _native_pns(structure: _Structure, hardware: object, mode: int, index: int) -> dict:
        """One TR's PNS response, straight out of the C safety core."""
        from .._ext._pulseg_wrapper import _calc_pns, _calc_pns_safe

        if mode != _safety.AMPLITUDE_MODES["actual"]:
            index = 0

        if _safety.is_safe_hardware(hardware):
            gx, gy, gz = _safety.safe_coefficients(hardware)
            return _calc_pns_safe(structure.collection, 0, index, gx, gy, gz)

        chronaxie_us, rheobase, alpha = _safety.irnich_coefficients(hardware)
        return _calc_pns(structure.collection, 0, index, chronaxie_us, rheobase, alpha)

    @staticmethod
    def _plot_pns(waveform: _safety.TRSequence, components: np.ndarray, raster: float) -> None:
        """Upstream's two PNS figures, drawn over a TR waveform.

        The same pair :func:`pypulseq.Sequence.calc_pns.calc_pns` draws, from
        the same two calls -- the gradient trace off the PPoly upstream itself
        built, and ``safe_plot`` on the components -- plus the thresholds,
        which upstream draws in neither mode. ``raster`` is the safety core's,
        half the gradient raster, not the sequence's.
        """
        import matplotlib.pyplot as plt
        from pypulseq.utils.safe_pns_prediction import safe_plot

        plt.figure()
        for gradient in waveform.get_gradients():
            if gradient is not None:
                plt.plot(gradient.x[1:-1], gradient.c[1, :-1])
        plt.title("gradient wave form, in Hz/m")

        plt.figure()
        safe_plot(components * 100, raster)
        _safety.overlay_pns_thresholds()

    def calculate_gradient_spectrum(
        self,
        max_frequency: float = 2000.0,
        window_width: float = 0.05,
        frequency_oversampling: float = 3.0,
        time_range: list[float] | None = None,
        plot: bool = True,
        combine_mode: str = "max",
        use_derivative: bool = False,
        acoustic_resonances: list[dict] | None = None,
        *,
        tr: str | int | None = None,
        resonance_lines: bool = False,
        bands: list | None = None,
        compat: bool = True,
    ):
        """The gradient spectrum of the sequence, or of one TR.

        Parameters
        ----------
        max_frequency : float, default 2000.0
            Highest frequency to report, in Hz.
        window_width : float, default 0.05
            Length of each transformed window, in seconds. **Ignored under
            ``tr``**, where the window is the repetition time, so that what
            comes back is one spectrum rather than a spectrogram. Passing a
            value there warns rather than being quietly overridden.
        frequency_oversampling : float, default 3.0
            Zero-padding factor along frequency; higher is smoother.
        time_range : list of float, optional
            Two timepoints in seconds bounding what to look at. Only with
            ``tr=None``.
        plot : bool, default True
            Draw the spectrograms.
        combine_mode : {"max", "mean", "rss", "none"}, default "max"
            How to collapse the windows into one spectrogram. Under ``tr``
            there is only ever one window, so this decides nothing except
            whether the single column is kept ( ``"none"`` ) or dropped.
        use_derivative : bool, default False
            Transform the slew rate rather than the gradient.
        acoustic_resonances : list of dict, optional
            Resonances to mark, as ``{'frequency': ..., 'bandwidth': ...}``.
            See :func:`~._safety.bands_to_resonances`.
        tr : {"worst_case"} or int, optional
            Analyse one repetition time rather than the timeline. ``None``,
            the default, is upstream PyPulseq exactly. ``"worst_case"`` is the
            per-sample maximum over every TR instance -- the waveform
            ``pulseg_check_safety`` judges. An integer is that instance as it
            really plays. See :meth:`calculate_pns` for the full account.
        resonance_lines : bool, default False
            Also compute the C safety core's acoustic line spectrum, draw it
            over the spectrogram, and return it. Needs ``tr``.
        bands : list of tuple, optional
            Forbidden bands as ``(freq_min_hz, freq_max_hz,
            max_amplitude_hz_per_m)``, which decide which lines count as
            violations. Read only when ``resonance_lines`` is set; defaults
            to whatever ``acoustic_resonances`` describes, or to nothing.

        Returns
        -------
        spectrograms : list of numpy.ndarray
            One per gradient axis.
        spectrogram_rss : numpy.ndarray
            The axes combined in root-sum-square.
        frequencies : numpy.ndarray
            The frequency axis, in Hz.
        times : numpy.ndarray
            The time axis, meaningful only for ``combine_mode="none"``.
        resonances : ~._safety.MechResonances
            **Only when ``resonance_lines`` is set**, appended as a fifth
            element: the line spectrum at the TR harmonics ``k / T_TR``, in
            equivalent-drive units, and which of them violate a band.

        Notes
        -----
        The transform is always upstream PyPulseq's -- under ``tr`` it is
        upstream's own code run over the waveform the C core extracted, so
        what changes is which waveform is transformed and over what window,
        never how.

        **Under ``tr`` this is a spectrum, not a spectrogram.** A repetition
        time is transformed in one window because it is periodic, and a
        periodic waveform has no time-varying spectrum to chart: it has
        energy only at multiples of ``1 / T_TR``. Cutting it into shorter
        windows would only smear neighbouring harmonics into each other.

        Those harmonics are what ``resonance_lines`` draws over the result,
        on **its own vertical axis** -- see
        :class:`~._safety.MechResonances` for why the two scales must not be
        shared.
        """
        from pypulseq.Sequence.calc_grad_spectrum import calculate_gradient_spectrum

        if acoustic_resonances is None:
            acoustic_resonances = []

        if tr is None:
            if resonance_lines:
                raise ValueError(
                    "calculate_gradient_spectrum(): resonance_lines needs tr -- a line "
                    "spectrum exists only for a repetition time, not for a stretch of timeline"
                )
            first, last = self._blocks_over(*_span(time_range if time_range else (0.0, np.inf)))
            spectrum = calculate_gradient_spectrum(
                self._upstream_window(first, last),
                max_frequency=max_frequency,
                window_width=window_width,
                frequency_oversampling=frequency_oversampling,
                time_range=time_range,
                plot=plot,
                combine_mode=combine_mode,
                use_derivative=use_derivative,
                acoustic_resonances=acoustic_resonances,
            )
            return spectrum if compat else _results.GradientSpectrum(*spectrum)

        if time_range is not None:
            raise ValueError(
                "calculate_gradient_spectrum(): time_range selects part of the timeline and "
                "tr selects a repetition time, which is not on it -- pass one or the other"
            )

        structure = self._structure_for("calculate_gradient_spectrum")
        waveform = structure.waveform(tr)

        if window_width != _UPSTREAM_WINDOW_WIDTH:
            warnings.warn(
                f"calculate_gradient_spectrum(): window_width={window_width} is ignored under "
                "tr -- a repetition time is transformed whole, in one window",
                stacklevel=2,
            )

        spectrum = calculate_gradient_spectrum(
            waveform,
            max_frequency=max_frequency,
            # The window IS the TR, so this is one spectrum rather than a
            # spectrogram. A periodic waveform has energy only at multiples
            # of 1/T_TR, and it is the transform over exactly one period that
            # resolves them; a shorter window would smear neighbouring
            # harmonics together and a longer one does not exist to cut.
            window_width=structure.tr_duration,
            frequency_oversampling=frequency_oversampling,
            time_range=None,
            plot=plot,
            combine_mode=combine_mode,
            use_derivative=use_derivative,
            acoustic_resonances=acoustic_resonances,
        )
        if not resonance_lines:
            return spectrum if compat else _results.GradientSpectrum(*spectrum)

        if bands is None:
            bands = [
                (
                    resonance["frequency"] - 0.5 * resonance["bandwidth"],
                    resonance["frequency"] + 0.5 * resonance["bandwidth"],
                    0.0,
                )
                for resonance in acoustic_resonances
            ]
        resonances = self._resonance_lines(structure, tr, max_frequency, bands)

        if plot and combine_mode != "none":
            _safety.overlay_resonance_lines(resonances, max_frequency=max_frequency)

        if not compat:
            return _results.GradientSpectrum(*spectrum, resonance_lines=resonances)
        # Upstream's four-tuple with a fifth element on the end. This is the
        # shape compat=False exists to retire -- it changes the length of the
        # caller's unpack -- and it survives only because it is what
        # resonance_lines already returned before the flag existed.
        return (*spectrum, resonances)

    def _resonance_lines(
        self, structure: _Structure, tr, max_frequency: float, bands: list
    ) -> _safety.MechResonances:
        """The C safety core's acoustic line spectrum for one TR."""
        from .._ext._pulseg_wrapper import _calc_mech_resonances

        _, index = structure.resolve(tr)
        spectra = _calc_mech_resonances(
            structure.collection,
            0,
            index,
            # A resolution fine enough that the harmonic grid the core
            # tabulates reaches max_frequency; the lines themselves land at
            # k / T_TR regardless of what is asked for here.
            target_resolution_hz=1.0 / structure.tr_duration,
            max_freq_hz=float(max_frequency),
            forbidden_bands=[tuple(float(value) for value in band[:3]) for band in bands],
        )
        return _safety.MechResonances.from_spectra(spectra, structure.tr_duration, bands)

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

        The one place this can differ from upstream is a block that both
        ``SET``s and ``INC``s the *same* counter: the operations are applied in
        the order the block plays them, read off the extension chain, rather
        than sets-then-increments.
        """
        if evolution not in ("none", "blocks", "adc", "label"):
            raise ValueError(
                "evaluate_labels(): evolution must be 'none', 'blocks', 'adc' or 'label', "
                f"not {evolution!r}"
            )

        first, last = self._window_for(time_range)
        labels: dict[str, float] = dict(init or {})
        evolutions: list[dict] = []

        for index in range(first, last + 1):
            block = self.get_block(index)
            if block.labels:
                for kind, name, value in block.labels:
                    if kind == "SET":
                        labels[name] = value
                    else:
                        labels[name] = labels.get(name, 0) + value
                if evolution == "label":
                    evolutions.append(dict(labels))

            if evolution == "blocks" or (evolution == "adc" and block.adc is not None):
                evolutions.append(dict(labels))

        if evolutions:
            return {
                name: np.array([step.get(name, 0) for step in evolutions]) for name in labels
            }
        return labels

    def apply_soft_delay(self, **kwargs: float) -> None:
        """Set named soft delays, rewriting the block durations that carry them.

        Each block holding a soft delay named in ``kwargs`` gets the duration
        ``value / factor + offset``, rounded to the block-duration raster.

        Parameters
        ----------
        **kwargs
            Soft-delay hints and their values in seconds -- ``TE=40e-3``. Only
            the delays named are touched; the sequence may hold others.

        Raises
        ------
        ValueError
            If a named hint is not in the sequence, if a hint and a numeric ID
            disagree with an earlier block's pairing of the two, or if a value
            works out to a negative duration.

        Notes
        -----
        Upstream's arithmetic and upstream's diagnostics, on this class's
        blocks. A soft delay is a *design-time* parameter -- the sequence
        arrives at the scanner with its durations already resolved -- so this
        is for a script that builds one sequence and writes several timings of
        it, not for anything on the interpreter's path.

        See Also
        --------
        get_default_soft_delay_values : the values already in the blocks.
        """
        raster = float(self.system.block_duration_raster)
        seen: dict[str, int] = {}
        warned: set[int] = set()

        for index in range(1, self.num_blocks + 1):
            delay = getattr(self.get_block(index), "soft_delay", None)
            if delay is None:
                continue

            self._check_soft_delay_consistency(delay, index, seen)
            if delay.hint not in kwargs:
                continue

            exact = float(kwargs[delay.hint]) / delay.factor + delay.offset
            rounded = round(exact / raster) * raster
            if abs(rounded - exact) > 0.5e-6 and delay.numID not in warned:
                warnings.warn(
                    f"Soft delay '{delay.hint}' in block {index}: duration rounded by "
                    f"{abs(rounded - exact) * 1e6:.1f} us to align with the block raster "
                    f"({raster * 1e6:.1f} us). Reported once per soft delay.",
                    stacklevel=2,
                )
                warned.add(delay.numID)
            if rounded < 0:
                raise ValueError(
                    f"Soft delay '{delay.hint}' in block {index}: the value works out to a "
                    f"negative duration ({rounded * 1e6:.1f} us). Check the offset "
                    f"({delay.offset * 1e6:.1f} us) and factor ({delay.factor})."
                )
            self._native.set_block_duration(index, rounded)

        unknown = [hint for hint in kwargs if hint not in seen]
        if unknown:
            available = sorted(seen) or ["none"]
            raise ValueError(
                f"apply_soft_delay(): {unknown} not in the sequence. "
                f"Available soft delays: {available}"
            )
        self._touch()

    def get_default_soft_delay_values(self) -> tuple[dict, list[str]]:
        """The soft-delay values the sequence was built with, and what is wrong.

        Returns
        -------
        values : dict
            Hint to the delay value in seconds implied by the block durations
            as they stand -- ``(duration - offset) * factor``, the inverse of
            what :meth:`apply_soft_delay` writes. This is what a user
            interface offers as the starting point of each slider.
        report : list of str
            One line per inconsistency found. Empty when the sequence is sound.

        Notes
        -----
        MATLAB's ``getDefaultSoftDelayValues``, which PyPulseq has no
        equivalent of. Two of its three outputs: the ``softDelayState`` cell
        array is MATLAB's intermediate, and its ``min``/``max`` are carried on
        the dict's values here instead --- each is a
        :class:`~._results.SoftDelay`, whose ``value`` is what the mapping
        compares and prints, so the dict reads as MATLAB's ``easyStruct``
        while still carrying the range the delay may be set over.

        The limits come from the duration reaching zero, which is the only
        bound the sequence itself states; a delay that would collide with the
        events inside its block is not caught here, and is not caught by
        MATLAB either.
        """
        report: list[str] = []
        seen: dict[str, int] = {}
        state: dict[int, dict] = {}

        for index in range(1, self.num_blocks + 1):
            delay = getattr(self.get_block(index), "soft_delay", None)
            if delay is None:
                continue

            if delay.numID < 0:
                report.append(f"Block {index}: soft delay '{delay.hint}' has a negative numeric ID")
                continue
            if delay.factor == 0:
                report.append(
                    f"Block {index}: soft delay '{delay.hint}'/{delay.numID} has factor 0"
                )
                continue
            try:
                self._check_soft_delay_consistency(delay, index, seen)
            except ValueError as problem:
                report.append(str(problem))

            default = (float(self.block_durations[index - 1]) - delay.offset) * delay.factor
            # A duration of zero is the only limit the sequence itself states;
            # which side of the range it is depends on the sign of the factor.
            limit = -delay.offset * delay.factor
            held = state.get(delay.numID)
            if held is None:
                state[delay.numID] = {
                    "hint": delay.hint,
                    "value": default,
                    "block": index,
                    "minimum": limit if delay.factor > 0 else 0.0,
                    "maximum": np.inf if delay.factor > 0 else limit,
                }
                continue

            if abs(default - held["value"]) > 1e-7:
                report.append(
                    f"Block {index}: soft delay '{delay.hint}'/{delay.numID} implies "
                    f"{default * 1e6:.1f} us, against {held['value'] * 1e6:.1f} us from block "
                    f"{held['block']}"
                )
            if delay.factor > 0:
                held["minimum"] = max(held["minimum"], limit)
            else:
                held["maximum"] = min(held["maximum"], limit)

        values: dict[str, _results.SoftDelay] = {}
        for numeric in sorted(state):
            held = state[numeric]
            values[held["hint"]] = _results.SoftDelay(
                hint=held["hint"],
                numeric_id=numeric,
                value=held["value"],
                minimum=held["minimum"],
                maximum=held["maximum"],
            )
        return values, report

    @staticmethod
    def _check_soft_delay_consistency(delay, index: int, seen: dict[str, int]) -> None:
        """One hint means one numeric ID and back, across the whole sequence."""
        known = seen.setdefault(delay.hint, delay.numID)
        if known != delay.numID:
            raise ValueError(
                f"Block {index}: soft delay '{delay.hint}' has numeric ID {delay.numID}, "
                f"against {known} where the same hint appeared before"
            )
        for hint, numeric in seen.items():
            if numeric == delay.numID and hint != delay.hint:
                raise ValueError(
                    f"Block {index}: soft delay numeric ID {delay.numID} is called "
                    f"'{delay.hint}' here and '{hint}' earlier"
                )

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

    def install(self, target: str | None = None, clear_cache: bool = False, **kwargs) -> None:
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
        if native.find_extension_type_id(str(extension_str)) or native.extension_type_name(
            int(extension_id)
        ):
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
            raise IndexError(f"no block {block_index} in a sequence of {self.num_blocks}")
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

    def sequence_descriptor(self, *, subsequence: int = 0):
        """The state-machine description of one repetition time.

        Returns a :class:`~pulserver.recon.seqdesc.SequenceDescription`: one
        event per block over the canonical pass, each carrying what a Bloch or
        EPG simulation needs and nothing it does not -- RF use, flip
        amplitude, phase and frequency for the pulses; role, k-space-zero
        timing and echo flag for the readouts.

        Notes
        -----
        **This is the same object the reconstruction receives**, not a
        parallel description of it.
        :func:`~pulserver.recon.seqdesc.decode_sequence_description` builds one
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

        The description is cached against the sequence's revision, so asking
        repeatedly costs one build.
        """
        from ..recon.seqdesc import (
            AdcRole,
            EventType,
            RfDefinition,
            RfShape,
            RfUse,
            SequenceDescription,
            SequenceEvent,
        )
        from .._ext._pulseg_wrapper import _get_rf_definitions, _get_sequence_description

        structure = self._structure_for("sequence_descriptor")
        raw = _get_sequence_description(structure.collection, int(subsequence))

        events = tuple(
            SequenceEvent(
                type=EventType(int(kind)),
                timestamp_us=float(stamp),
                params=tuple(float(value) for value in row),
            )
            for kind, stamp, row in zip(raw["type"], raw["timestamp_us"], raw["params"])
        )

        definitions = {}
        for entry in _get_rf_definitions(structure.collection, int(subsequence)):
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
        from .._ext._pulseg_wrapper import _get_sequence_parameters

        return _get_sequence_parameters(self._structure_for("sequence_parameters").collection)

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
        first = min(max(int(np.searchsorted(edges, start, side="right")) - 1, 0), count - 1)
        last = min(max(int(np.searchsorted(edges, stop, side="left")) - 1, first), count - 1)
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
                if self._native.extension_type_name(type_id) in ("ROTATIONS", "RF_SHIMS"):
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
            key: float(getattr(self.system, attribute)) for key, attribute in _RASTER_DEFINITIONS
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
            self._native.set_definition_numbers(key, [float(item) for item in numbers], whole)


# %% local subroutines


class _Structure:
    """The scan structure of one sequence, as the C safety core sees it.

    Building this is the expensive half of every TR-based analysis: the
    sequence is serialised, parsed back by the C library, and its repetition
    time and segmentation recovered. Everything after that -- extracting a TR
    waveform, the acoustic line spectrum, PNS -- is comparatively cheap and
    runs against the collection held here, so the cost is paid once per
    sequence rather than once per question asked about it.

    Held by :class:`Sequence` behind a revision number, and rebuilt rather
    than updated when the sequence changes. A structure that has silently
    outlived its sequence would answer about blocks that no longer exist.
    """

    def __init__(self, sequence: Sequence) -> None:
        from .._ext._pulseg_wrapper import _find_tr, _PulseqCollection

        self.system = sequence.system
        system = self.system
        # Binary, not text: this is written only to be parsed straight back,
        # so the number formatting and the sscanf that the text format would
        # spend on both ends are pure waste -- and the six significant digits
        # `%12g` rounds a gradient amplitude to would be a needless loss on
        # the way into a safety analysis.
        #
        # Serialising compresses shapes and republishes definitions, which
        # touches the C++ sequence -- so read the revision after it, not
        # before, or the cache is stale the moment it is built.
        written = sequence._to_binary()
        self.revision = sequence._revision
        self.collection = _PulseqCollection(
            [written],
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
        self.tr = _find_tr(self.collection)
        self._segments: list | None = None

    @property
    def tr_duration(self) -> float:
        """float : The canonical TR, in seconds."""
        return float(self.tr["tr_duration_us"]) * 1e-6

    @property
    def num_trs(self) -> int:
        """int : How many structural TRs the repeating region holds."""
        return int(self.tr["num_trs"])

    @property
    def folded_prep_or_cooldown(self) -> bool:
        """bool : Whether a leading or trailing section is *not* just another TR.

        A prep or cooldown region that repeats the TR pattern is folded away
        by the C core -- its blocks are TR instances like any other, merely
        discarded downstream. Only a genuinely different one survives here,
        and when it does the core's unit of analysis stops being the TR and
        becomes the whole pass it sits in.
        """
        return bool(self.tr["num_prep_blocks"] and not self.tr["degenerate_prep"]) or bool(
            self.tr["num_cooldown_blocks"] and not self.tr["degenerate_cooldown"]
        )

    @property
    def num_instances(self) -> int:
        """int : How many things ``tr=<int>`` can name.

        Passes when a non-degenerate section folds prep and cooldown into the
        unit of analysis (see :attr:`folded_prep_or_cooldown`), and otherwise
        every TR the scanner plays -- the prep and cooldown TRs included, and
        counted once per average, which is how the C core indexes them.
        """
        if self.folded_prep_or_cooldown:
            return max(int(self.tr["num_passes"]), 1)
        averages = max(int(self.tr["num_averages"]), 1)
        return (
            int(self.tr["num_prep_trs"])
            + averages * self.num_trs
            + int(self.tr["num_cooldown_trs"])
        )

    @property
    def segments(self) -> list:
        """list : The segment layout, resolved to each segment's max-energy instance.

        Reporting only. Neither the acoustic nor the PNS analysis selects a
        segment instance -- both run on the per-sample maximum over every TR
        instance -- so this says which blocks a reader should look at, not
        which blocks were analysed.
        """
        if self._segments is None:
            from .._ext._pulseg_wrapper import _get_segments

            self._segments = _get_segments(self.collection, 0)
        return self._segments

    def waveform(self, tr, *, rf_channel: int = 0) -> _safety.TRSequence:
        """The TR ``tr`` names, as a sequence upstream can read.

        Everything in it -- gradients, RF magnitude and phase, ADC events,
        block boundaries -- comes out of one ``pulseg_get_tr_waveforms`` call,
        so what is returned is the canonical TR the C core itself constructs
        rather than one rebuilt from the timeline.

        Parameters
        ----------
        tr : {"worst_case", "zero_variable"} or int
            ``"worst_case"`` for the per-sample maximum over every instance --
            the waveform ``pulseg_check_safety`` judges, which is not any one
            TR the scanner plays. ``"zero_variable"`` for the same TR with the
            gradients that vary between instances zeroed, leaving the
            structure the constant ones make. An integer for that instance as
            it really plays, signed amplitudes and all.
        rf_channel : int, default 0
            Which transmit channel the returned blocks report, for a pTx TR.
        """
        from .._ext._pulseg_wrapper import _get_tr_waveforms

        mode, index = self.resolve(tr)
        return _safety.TRSequence.from_c(
            _get_tr_waveforms(self.collection, 0, mode, index),
            self.system,
            rf_channel=rf_channel,
        )

    def resolve(self, tr) -> tuple[int, int]:
        """``tr`` as the ``(amplitude_mode, tr_index)`` pair the binding takes."""
        if isinstance(tr, str):
            if tr not in _safety.AMPLITUDE_MODES:
                raise ValueError(
                    f"tr must be an integer or one of {sorted(_safety.AMPLITUDE_MODES)}, got {tr!r}"
                )
            return _safety.AMPLITUDE_MODES[tr], 0

        index = int(tr)
        # The core's own bound, which is passes when prep or cooldown is
        # folded into the unit of analysis and TR instances otherwise --
        # not num_trs, which counts only the structural repeat.
        if not 0 <= index < self.num_instances:
            named = "passes" if self.folded_prep_or_cooldown else "TR instances"
            raise ValueError(
                f"tr={index} is out of range; the sequence holds {self.num_instances} {named}"
            )
        return _safety.AMPLITUDE_MODES["actual"], index


class _Pieces:
    """One shot's gradients as straight-line pieces, refocusing folded in.

    ``offsets`` and ``widths`` locate each piece relative to the excitation;
    ``start`` and ``slope`` are its gradient in Hz/m and Hz/m/s. Between two
    nodes each channel is a single straight line, which is what lets the
    integrals below be closed-form rather than sampled.

    **The sign belongs to the piece, not to the node.** A refocusing pulse
    inverts the phase accumulated so far, which is the same as inverting every
    gradient after it -- but the node *at* the pulse is the end of one piece
    and the start of the next, and those two want opposite signs. So the flip
    is counted per piece, from its own start time, and the sign multiplies
    both of that piece's endpoints. Every refocusing instant is forced into
    the grid, so no piece ever straddles one.

    The flip happens at *every* refocusing pulse rather than only the first,
    which is the same answer for a Stejskal-Tanner pair and the right one for
    a twice-refocused sequence -- where MATLAB's single flip is wrong, and
    says so in its own ``TODO``.
    """

    __slots__ = ("offsets", "widths", "start", "slope")

    def __init__(self, times: np.ndarray, gradients: np.ndarray, flips: np.ndarray) -> None:
        widths = np.diff(times)
        keep = widths > 0
        self.widths = widths[keep]
        self.offsets = (times[:-1] - times[0])[keep]

        signs = ((-1.0) ** np.searchsorted(flips, times[:-1], side="right"))[keep, None]
        first = gradients[:-1][keep] * signs
        last = gradients[1:][keep] * signs
        self.start = first
        self.slope = np.divide(
            last - first, self.widths[:, None], out=np.zeros_like(first), where=self.widths[:, None] > 0
        )

    def __len__(self) -> int:
        return int(self.widths.size)


def _gradient_pieces(
    channels: list[np.ndarray], start: float, stop: float, refocusings: np.ndarray
) -> _Pieces:
    """The three gradients over ``start..stop`` as :class:`_Pieces`.

    The grid is the union of every channel's own nodes with the interval ends
    and the refocusing instants.
    """
    inside = refocusings[(refocusings > start) & (refocusings < stop)]
    nodes = [np.array([start, stop], dtype=float), inside]
    for channel in channels[:3]:
        if channel.shape[1]:
            nodes.append(channel[0])

    times = np.unique(np.concatenate(nodes))
    times = times[(times >= start) & (times <= stop)]

    gradients = np.zeros((times.size, 3))
    for axis, channel in enumerate(channels[:3]):
        if channel.shape[1]:
            gradients[:, axis] = np.interp(times, channel[0], channel[1], left=0.0, right=0.0)
    return _Pieces(times, gradients, inside)


def _b_tensor_over(pieces: _Pieces) -> np.ndarray:
    """``(3, 3)`` of ``integral q_i q_j dt`` in s/m^2, exactly.

    ``q = 2*pi*integral g dt`` in rad/m. A Pulseq gradient is piecewise
    linear, so ``q`` is piecewise quadratic and the product of two of its
    components is a quartic -- integrated in closed form per piece rather than
    sampled, which is what makes the answer independent of any raster.
    """
    if not len(pieces):
        return np.zeros((3, 3))

    widths = pieces.widths
    # q(tau) = q0 + 2*pi*(a*tau + b*tau**2/2), as coefficients in tau.
    increments = 2 * np.pi * (
        pieces.start * widths[:, None] + 0.5 * pieces.slope * widths[:, None] ** 2
    )
    origins = np.zeros_like(increments)
    origins[1:] = np.cumsum(increments, axis=0)[:-1]
    coefficients = np.stack(
        (origins, 2 * np.pi * pieces.start, np.pi * pieces.slope), axis=-1
    )  # (n, 3, 3)

    # The integral of tau**(p+q) over a piece, as a weight per coefficient pair.
    powers = np.arange(3)
    orders = powers[:, None] + powers[None, :] + 1
    weights = widths[:, None, None] ** orders / orders
    return np.einsum("nip,njq,npq->ij", coefficients, coefficients, weights)


def _moment_over(pieces: _Pieces, order: int) -> np.ndarray:
    """``2*pi*integral g(t) * (t - t0)**order dt`` per axis, exactly.

    Units are rad/m times seconds to the ``order``. The weight is measured
    from the excitation, which is where MATLAB measures it from.
    """
    from math import comb

    if not len(pieces):
        return np.zeros(3)

    total = np.zeros(3)
    for power in range(order + 1):
        # (offset + tau)**order expanded, times the piece's own (a + b*tau).
        weight = comb(order, power) * pieces.offsets ** (order - power)
        for degree, coefficient in ((power, pieces.start), (power + 1, pieces.slope)):
            total += weight * pieces.widths ** (degree + 1) / (degree + 1) @ coefficient
    return 2 * np.pi * total


def _worst_window(durations: np.ndarray, values: np.ndarray, width: float) -> float:
    """The largest sum of ``values`` over any window of at most ``width`` seconds.

    MATLAB grows the window by one block and then trims from the front while
    it is too wide, taking its maximum *before* the trim -- so the window that
    wins may be up to one block wider than asked for, which is its documented
    "rounded up to a certain number of complete blocks". Reproduced here as
    two cumulative sums and a search rather than a loop, which is the same
    answer without MATLAB's index bug (see :meth:`Sequence.calc_rf_power`).
    """
    if values.size == 0:
        return 0.0

    elapsed = np.concatenate(([0.0], np.cumsum(durations)))
    accumulated = np.concatenate(([0.0], np.cumsum(values)))
    # Window entering block i spans blocks `start(i)..i`, where start is the
    # front the trim left after block i-1: the earliest block whose tail is
    # still within `width` of the end of block i-1.
    starts = np.searchsorted(elapsed, elapsed[:-1] - width, side="left")
    return float((accumulated[1:] - accumulated[starts]).max())


def _rf_use(rf: SimpleNamespace) -> str:
    """One decoded RF pulse's use tag, as one of :data:`~._results.RF_USES`.

    A row with no use character reads back as ``"undefined"``, which is a
    distinct thing from ``"other"``: ``"other"`` was chosen by whoever wrote
    the sequence, ``"undefined"`` means nobody said. Upstream folds the latter
    in with excitation, and :meth:`Sequence.rf_times` reproduces that under
    ``compat=True`` -- but the tag itself is kept, so a caller who wants to
    know can find out.
    """
    use = getattr(rf, "use", None)
    if use in RF_USES:
        return use
    return "undefined"


def _span(time_range) -> tuple[float, float]:
    """``time_range`` as a pair of seconds, checked the way upstream checks it."""
    bounds = tuple(time_range)
    if len(bounds) != 2:
        raise ValueError("time_range must hold two elements")
    start, stop = float(bounds[0]), float(bounds[1])
    if start > stop:
        raise ValueError("time_range must end after it begins")
    return start, stop


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
        events.append(_shimmed(block.rf, block.rf_shim) if block.rf_shim is not None else block.rf)

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


def _decompress(num_samples: int, samples: np.ndarray) -> np.ndarray:
    """One shape's samples, run-length decoded if it was compressed.

    Pulseq stores the quantised derivative and lets a run of three or more
    equal values collapse to two of them plus a repeat count. A shape whose
    stored length already equals its sample count was left uncompressed.
    """
    if samples.size == num_samples:
        return np.asarray(samples, dtype=float)

    derivative = np.empty(num_samples, dtype=float)
    written = 0
    position = 0
    while position < samples.size:
        value = samples[position]
        derivative[written] = value
        written += 1
        position += 1
        if position < samples.size and samples[position] == value:
            derivative[written] = value
            written += 1
            position += 1
            if position < samples.size:
                repeats = round(float(samples[position]))
                position += 1
                derivative[written : written + repeats] = value
                written += repeats
    return np.cumsum(derivative[:written])


def _shape(native, shape_id: int) -> np.ndarray:
    """Library shape ``shape_id``, decompressed."""
    num_samples, samples = native.shape_row(shape_id)
    return _decompress(num_samples, samples)


def _times(native, time_shape_id: int, count: int, raster: float) -> np.ndarray:
    """The sample times of a waveform, materialised on its raster.

    Zero means the standard raster, where sample ``i`` sits at the centre of
    its interval; ``-1`` is the half-raster variant a gradient may use;
    anything else is a shape of ticks.
    """
    if time_shape_id > 0:
        return _shape(native, time_shape_id) * raster
    if time_shape_id < 0:
        return 0.5 * raster * np.arange(1, count + 1)
    return (np.arange(count) + 0.5) * raster


def _rf_event(native, rf_id: int) -> SimpleNamespace:
    """RF library row ``rf_id`` as a PyPulseq RF event."""
    row = native.rf_row(rf_id)
    magnitude = _shape(native, int(row[1]))
    phase = _shape(native, int(row[2]))
    raster = native.rf_raster_time
    return SimpleNamespace(
        type="rf",
        signal=row[0] * magnitude * np.exp(2j * np.pi * phase),
        t=_times(native, int(row[3]), magnitude.size, raster),
        shape_dur=magnitude.size * raster,
        center=row[4],
        delay=row[5],
        freq_ppm=row[6],
        phase_ppm=row[7],
        freq_offset=row[8],
        phase_offset=row[9],
        dead_time=0.0,
        ringdown_time=0.0,
        use=native.rf_use(rf_id),
    )


def _grad_event(native, grad_id: int, channel: str) -> SimpleNamespace:
    """Gradient ``grad_id`` as a PyPulseq trapezoid or arbitrary gradient."""
    row = native.grad_row(grad_id)
    if native.grad_kind(grad_id) == "trap":
        amplitude, rise, flat, fall, delay = row
        return SimpleNamespace(
            type="trap",
            channel=channel,
            amplitude=amplitude,
            rise_time=rise,
            flat_time=flat,
            fall_time=fall,
            delay=delay,
            area=amplitude * (flat + rise / 2 + fall / 2),
            flat_area=amplitude * flat,
            first=0.0,
            last=0.0,
        )

    raster = native.grad_raster_time
    waveform = row[0] * _shape(native, int(row[3]))
    return SimpleNamespace(
        type="grad",
        channel=channel,
        waveform=waveform,
        tt=_times(native, int(row[4]), waveform.size, raster),
        shape_dur=waveform.size * raster,
        first=row[1],
        last=row[2],
        delay=row[5],
        area=float(np.sum(waveform) * raster),
    )


def _adc_event(native, adc_id: int) -> SimpleNamespace:
    """ADC library row ``adc_id`` as a PyPulseq ADC event."""
    row = native.adc_row(adc_id)
    return SimpleNamespace(
        type="adc",
        num_samples=round(float(row[0])),
        dwell=row[1],
        delay=row[2],
        freq_ppm=row[3],
        phase_ppm=row[4],
        freq_offset=row[5],
        phase_offset=row[6],
        dead_time=0.0,
        phase_modulation=_shape(native, int(row[7])) if row[7] else None,
    )


def _decode_extensions(native, head: int, block: SimpleNamespace) -> None:
    """Walk a block's extension chain, filling in what each node carries."""
    node = head
    while node:
        type_id, reference, node = native.extension_row(node)
        name = native.extension_type_name(type_id)
        if name == "TRIGGERS":
            row = native.trigger_row(reference)
            block.triggers.append(
                SimpleNamespace(
                    type="trigger" if row[0] == 2.0 else "output",
                    channel=_TRIGGER_CHANNELS.get(row[0], {}).get(row[1], "osc0"),
                    delay=row[2],
                    duration=row[3],
                )
            )
        elif name == "LABELSET":
            value, label_id = native.label_set_row(reference)
            block.label_sets.append((native.label_name(label_id), value))
            block.labels.append(("SET", native.label_name(label_id), value))
        elif name == "LABELINC":
            value, label_id = native.label_inc_row(reference)
            block.label_incs.append((native.label_name(label_id), value))
            block.labels.append(("INC", native.label_name(label_id), value))
        elif name == "ROTATIONS":
            block.rotation = native.rotation_row(reference)
        elif name == "RF_SHIMS":
            channels = native.rf_shim_row(reference).reshape(-1, 2)
            block.rf_shim = channels[:, 0] * np.exp(1j * channels[:, 1])
        elif name == "DELAYS":
            num, offset, factor, hint = native.soft_delay_row(reference)
            block.soft_delay = SimpleNamespace(
                type="soft_delay", numID=num, offset=offset, factor=factor, hint=hint
            )


# %% one window instead of two
#
# Unstacked, upstream draws RF/ADC on one figure and the gradients on another,
# and opens them as two windows. Reading a sequence means reading the two
# together, so they are laid out side by side instead -- by moving the axes
# upstream made onto a figure of ours, which leaves its plotting code untouched
# and its `SeqPlot` handles pointing at real axes.


def _is_merged(plot: object) -> bool:
    """Whether ``plot`` is one of ours, already laid out in two columns."""
    figure = getattr(plot, "fig1", None)
    return figure is not None and figure is getattr(plot, "fig2", None) and not getattr(plot, "stacked", False)


def _adopt(axis, figure, spec) -> None:
    """Move ``axis`` onto ``figure``, at the grid position ``spec``."""
    axis.remove()
    axis.figure = figure
    figure.add_axes(axis)
    axis.set_subplotspec(spec)


def _merge_columns(plot, *, show_guides: bool = False) -> None:
    """Lay ``plot``'s two figures out as one, three rows by two columns.

    Sharing the time axis has to be re-established by hand: moving an axis
    between figures drops it out of the shared group, silently -- panning one
    column would otherwise leave the other where it was.
    """
    from matplotlib import pyplot as plt

    columns = (tuple(plot.ax1), tuple(plot.ax2))
    sources = (plot.fig1, plot.fig2)
    merged = plt.figure(figsize=(14, 7))
    grid = merged.add_gridspec(3, 2)

    for column, axes in enumerate(columns):
        for row, axis in enumerate(axes):
            _adopt(axis, merged, grid[row, column])

    leader = columns[0][0]
    for axis in (*columns[0][1:], *columns[1]):
        axis.sharex(leader)

    for figure in sources:
        if figure is not None:
            plt.close(figure)

    merged._seq_t_factor = getattr(sources[0], "_seq_t_factor", 1.0)
    merged.tight_layout()

    plot.fig1 = plot.fig2 = merged
    plot.ax1, plot.ax2 = columns
    if show_guides:
        _install_guides(plot)


def _split_columns(plot) -> None:
    """Undo :func:`_merge_columns`, so upstream can draw over ``plot`` again.

    An overlay is upstream's way of comparing two sequences, and it reuses the
    figures of an earlier plot by taking the first three axes of each. Handed
    one merged figure it would take the same three twice and draw both panels
    into the left column, so the columns go back to being two figures for the
    length of the call.
    """
    from matplotlib import pyplot as plt

    merged = plot.fig1
    columns = (tuple(plot.ax1), tuple(plot.ax2))
    figures = []
    for axes in columns:
        figure = plt.figure()
        grid = figure.add_gridspec(3, 1)
        for row, axis in enumerate(axes):
            _adopt(axis, figure, grid[row, 0])
        figure._seq_t_factor = getattr(merged, "_seq_t_factor", 1.0)
        figures.append(figure)

    plt.close(merged)
    plot.fig1, plot.fig2 = figures
    plot.ax1, plot.ax2 = columns


def _install_guides(plot) -> None:
    """Follow the cursor with a hairline across every panel of ``plot``.

    Upstream sets these up too, but binds them to the canvases of the figures it
    made -- which this one replaced. Rebinding is cheaper and less surprising
    than persuading it to draw somewhere else.
    """
    try:
        import mplcursors  # noqa: F401
    except ImportError:
        return

    figure = plot.fig1
    axes = list(dict.fromkeys((*plot.ax1, *plot.ax2)))
    lines = {
        axis: axis.axvline(0.0, color="r", linestyle="--", linewidth=1.0, visible=False, zorder=1000)
        for axis in axes
    }

    def _follow(event) -> None:
        inside = event.inaxes in axes and event.xdata is not None
        for line in lines.values():
            line.set_visible(inside)
            if inside:
                line.set_xdata([event.xdata])
        figure.canvas.draw_idle()

    plot._vlines = lines
    plot._show_guides = True
    plot._guide_cids = [(figure.canvas, figure.canvas.mpl_connect("motion_notify_event", _follow))]


