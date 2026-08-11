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
from . import _safety
from ._pulseqpp import to_upstream
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
_NOT_PORTED = (
    "Sequence.{what} has upstream PyPulseq's signature but no implementation "
    "yet. See the module docstring."
)

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
            ``label_sets`` and ``label_incs``.

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
    ):
        """Draw the sequence, or one canonical TR: ADC, RF, and the three gradients.

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
        if tr is None:
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
        block_range: tuple[int, int] | None = None,
        frame: str = "physical",
        sample_window_average: bool = False,
        dense: bool = True,
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
        block_range : tuple of int, optional
            ``(first, last)``, 1-based and inclusive, to analyse part of the
            sequence. The whole of it by default.
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
        result = self._kspace(
            trajectory_delay=trajectory_delay,
            gradient_offset=gradient_offset,
            block_range=block_range,
            frame=frame,
            sample_window_average=sample_window_average,
            dense=False,
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
            first, last = self._block_range(block_range)
            upstream = to_upstream(self, first=first, last=(None if last == 0 else last))
            k_traj = upstream.calculate_kspace(
                trajectory_delay=trajectory_delay,
                gradient_offset=gradient_offset,
            )[1]

        return (
            result["k_adc"],
            k_traj,
            result["t_excitation"],
            result["t_refocusing"],
            result["t_adc"],
        )

    def _kspace(
        self,
        *,
        trajectory_delay=0.0,
        gradient_offset=0.0,
        block_range=None,
        frame="physical",
        sample_window_average=False,
        dense=True,
    ) -> dict:
        """Everything the C core reports, not just upstream's five entries.

        Kept separate because :meth:`calculate_kspace` has to return exactly
        upstream's tuple, and the derived echo positions, the k-space centre
        and the repeat-key statistics have nowhere in it to go.
        """
        if frame not in ("physical", "logical"):
            raise ValueError(f"frame must be 'physical' or 'logical', not {frame!r}")

        first, last = self._block_range(block_range)
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
        block_range: tuple[int, int] | None = None,
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
        block_range : tuple of int, optional
            ``(first, last)``, 1-based and inclusive.
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

        first, last = self._block_range(block_range)

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

    def _block_range(self, block_range) -> tuple[int, int]:
        """Upstream-style ``(first, last)`` as the C core's 1-based pair.

        ``0`` for the last block is the core's "to the end", which is why the
        default is not ``len(self)``: a range that ends at the end must stay
        true if the sequence grows.
        """
        if block_range is None:
            return 1, 0
        first, last = block_range
        first = int(first)
        last = int(last)
        if first < 1:
            raise ValueError(f"block_range starts at block 1, not {first}")
        if last != 0 and last < first:
            raise ValueError(f"block_range {block_range!r} is empty")
        return first, last

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

    def waveforms(self, append_RF: bool = False, time_range: list[float] | None = None):
        """The sequence's gradient waveforms. Not ported yet; see upstream
        :meth:`pypulseq.Sequence.sequence.Sequence.waveforms`."""
        raise NotImplementedError(_NOT_PORTED.format(what="waveforms"))

    def waveforms_and_times(self, append_RF: bool = False, time_range: list[float] | None = None):
        """The sequence's waveforms with their time axes. Not ported yet; see
        upstream
        :meth:`pypulseq.Sequence.sequence.Sequence.waveforms_and_times`."""
        raise NotImplementedError(_NOT_PORTED.format(what="waveforms_and_times"))

    def check_timing(self, print_errors: bool = False):
        """Check every block's timing against the raster. Not ported yet;
        see upstream
        :meth:`pypulseq.Sequence.sequence.Sequence.check_timing`."""
        raise NotImplementedError(_NOT_PORTED.format(what="check_timing"))

    def test_report(self) -> str:
        """A text report on the sequence. Not ported yet; see upstream
        :meth:`pypulseq.Sequence.sequence.Sequence.test_report`."""
        raise NotImplementedError(_NOT_PORTED.format(what="test_report"))

    def calculate_pns(
        self,
        hardware: object,
        time_range: list[float] | None = None,
        do_plots: bool = True,
        *,
        tr: str | int | None = None,
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
            return answer

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

        return bool(np.all(norm < 1)), norm, components, times

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
            return calculate_gradient_spectrum(
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
            return spectrum

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

    def evaluate_labels(self, init: dict | None = None, evolution: str = "none") -> dict:
        """Label values through the sequence. Not ported yet; see upstream
        :meth:`pypulseq.Sequence.sequence.Sequence.evaluate_labels`."""
        raise NotImplementedError(_NOT_PORTED.format(what="evaluate_labels"))

    def apply_soft_delay(self, **kwargs: object) -> None:
        """Apply soft-delay values to block durations. Not ported yet; see
        upstream
        :meth:`pypulseq.Sequence.sequence.Sequence.apply_soft_delay`."""
        raise NotImplementedError(_NOT_PORTED.format(what="apply_soft_delay"))

    def flip_grad_axis(self, axis: str) -> None:
        """Invert every gradient on ``axis``. Not ported yet; see upstream
        :meth:`pypulseq.Sequence.sequence.Sequence.flip_grad_axis`."""
        raise NotImplementedError(_NOT_PORTED.format(what="flip_grad_axis"))

    def mod_grad_axis(self, axis: str, modifier: float) -> None:
        """Scale every gradient on ``axis``. Not ported yet; see upstream
        :meth:`pypulseq.Sequence.sequence.Sequence.mod_grad_axis`."""
        raise NotImplementedError(_NOT_PORTED.format(what="mod_grad_axis"))

    def find_block_by_time(self, t: float) -> int:
        """The index of the block containing time ``t``. Not ported yet; see
        upstream
        :meth:`pypulseq.Sequence.sequence.Sequence.find_block_by_time`."""
        raise NotImplementedError(_NOT_PORTED.format(what="find_block_by_time"))

    # -- internals -------------------------------------------------------

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
        elif name == "LABELINC":
            value, label_id = native.label_inc_row(reference)
            block.label_incs.append((native.label_name(label_id), value))
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


