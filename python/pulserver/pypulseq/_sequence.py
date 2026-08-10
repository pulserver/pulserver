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

**The scan structure -- repetition times, segments, the safety analyses that
rest on them -- is not here yet.** It is the module and scan-loop layer's to
supply, and this class does not stand in for it with placeholders. What *is*
here matches upstream PyPulseq's :class:`~pypulseq.Sequence.sequence.Sequence`
method for method, including several whose implementation is not built yet
(:meth:`calculate_kspace`, :meth:`calculate_pns`, and others below) -- each
raises :exc:`NotImplementedError` with upstream's exact signature, so the
class's *shape* already matches what it will grow into.
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
        return 0, list(self._native.warm_event(event))

    def register_grad_event(self, event: object) -> int | tuple[int, list[int]]:
        """Register ``event``'s shapes ahead of the blocks that play it.

        Returns
        -------
        int or tuple
            ``0`` for a trapezoid, which has no shape; ``(0, shape_ids)`` for
            an arbitrary gradient, matching upstream.
        """
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
        self._native.remove_duplicates()

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

        TransformFOV(translation=offset_mm, server_mode=(mode == "server")).apply_to_sequence(
            self, in_place=True
        )

    # -- files -----------------------------------------------------------

    def write(
        self,
        path: str | Path,
        *,
        binary: bool | None = None,
        create_signature: bool = True,
    ) -> None:
        """Write the sequence to ``path``.

        Parameters
        ----------
        path : str or pathlib.Path
            Where to write.
        binary : bool, optional
            Write the binary Pulseq format rather than ``.seq`` text. Defaults
            to whether ``path`` ends in ``.bin``.
        create_signature : bool, default True
            Append the ``[SIGNATURE]`` section. Text format only.
        """
        path = Path(path)
        if binary is None:
            binary = path.suffix.lower() == ".bin"
        path.write_bytes(self.serialize(binary=binary, create_signature=create_signature))

    def serialize(self, *, binary: bool = False, create_signature: bool = True) -> bytes:
        """The sequence as a Pulseq file, in memory.

        Parameters
        ----------
        binary : bool, default False
            Write the binary format rather than ``.seq`` text.
        create_signature : bool, default True
            Append the ``[SIGNATURE]`` section. Text format only.

        Returns
        -------
        bytes
            The whole file.

        Notes
        -----
        Shapes are compressed here rather than as they are registered, so the
        codec runs over the waveforms that survived deduplication instead of
        over every use of them.
        """
        self._native.compress_shapes()
        self._publish_definitions()
        if binary:
            return self._native.write_binary()
        return self._native.write_text(create_signature)

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
        self._native = _cxx.read_file(str(Path(path)))
        self._definitions = dict(self._native.definitions())

    # -- looking at it -----------------------------------------------------
    #
    # Everything below has upstream PyPulseq's exact signature and raises
    # NotImplementedError. See the module docstring for why: the scan
    # structure these need (or, for plot/calculate_kspace/etc., used to get
    # from decoding a throwaway window) isn't carried by this class yet.

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
    ):
        """Draw the sequence: ADC, RF magnitude and phase, and the three gradients.

        Parameters
        ----------
        label : str, default ""
            Labels whose value to mark at each ADC, as a comma-separated list --
            ``"LIN,REP"``, say.
        show_blocks : bool, default False
            Tick the axes at the block boundaries.
        save : bool, default False
            Write the figure out as a JPEG beside the working directory.
        time_range : tuple of float, default (0, inf)
            The seconds to draw. Only the blocks this touches are decoded, so
            asking for a tenth of a second of a long protocol costs a tenth of a
            second of blocks -- and the axis still reads in time from the start
            of the sequence.
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

        Returns
        -------
        pypulseq.utils.seq_plot.SeqPlot
            The plot, whose ``fig1``/``fig2`` and ``ax1``/``ax2`` are upstream's.

        Notes
        -----
        Drawing is upstream PyPulseq's, over a :class:`pypulseq.Sequence` holding
        the blocks asked for. Rotations are resolved into the gradients and RF
        shims spread across the transmit channels first, so what is drawn is
        what the scanner plays rather than the base waveform the file stores.

        Unstacked, the two panels PyPulseq would open in separate windows are
        laid out as **one figure, three rows by two columns**, sharing their time
        axis; ``fig1`` and ``fig2`` are then the same figure. ``save=True``
        writes that one window as ``seq_plot.jpg`` rather than upstream's pair.
        """
        first, last = self._blocks_over(*_span(time_range))
        if last - first + 1 > _LOUD_ABOVE:
            warnings.warn(
                f"Plotting {last - first + 1} blocks. Matplotlib will take a long time over "
                "this; pass time_range=(start, stop) to draw a part of the sequence.",
                stacklevel=2,
            )

        window = self._upstream_window(first, last)
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
    ):
        """The k-space trajectory of the sequence. Not ported yet; see
        upstream
        :meth:`pypulseq.Sequence.sequence.Sequence.calculate_kspace`."""
        raise NotImplementedError(_NOT_PORTED.format(what="calculate_kspace"))

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

    def calculate_pns(self, hardware: object, time_range: list[float] | None = None, do_plots: bool = True):
        """Peripheral nerve stimulation, SAFE model. Not ported yet; see
        upstream
        :meth:`pypulseq.Sequence.sequence.Sequence.calculate_pns`. This
        project's own PNS visualisation uses a different (Irnich) model --
        see ``pulserver._safety.chronaxie_pns``; the real gate is the C
        safety core, not this class."""
        raise NotImplementedError(_NOT_PORTED.format(what="calculate_pns"))

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
    ):
        """The sequence's gradient spectrum. Not ported yet; see upstream
        :meth:`pypulseq.Sequence.sequence.Sequence.calculate_gradient_spectrum`."""
        raise NotImplementedError(_NOT_PORTED.format(what="calculate_gradient_spectrum"))

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


