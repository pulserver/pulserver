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
about. ``seq.calculate_kspace(time_range=(0.2, 0.3))`` costs a tenth of a
second of blocks, not the whole protocol. The window has rotations resolved
into its gradients and RF shims expanded across transmit channels, so what it
describes is what the scanner plays rather than the base waveform the file
stores.

Not everything the older Pulserver sequence offered is here yet. The scan
structure -- repetition times, segments, the safety analyses that rest on
them -- is declared and raises :exc:`NotImplementedError` until the module and
scan-loop layer above this one is rebuilt on it.
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
from ._rotate3d import rotate3D

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

#: How many decoded windows to keep before the oldest is dropped.
_WINDOW_CACHE = 8

#: What every method waiting on the scan-structure layer says.
_PENDING = (
    "Sequence.{what} needs the scan structure, which this class does not carry yet -- "
    "it is the module and scan-loop layer's to supply. Build the sequence and write "
    "it; the analysis surface returns when that layer is rebuilt on this class."
)


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
        self._windows: dict[tuple[int, int], pp.Sequence] = {}
        self._starts: np.ndarray | None = None

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
        self._touch()
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

    def remove_duplicates(self) -> None:
        """Collapse every library to its distinct rows and renumber the blocks.

        Rows are compared at the precision the file writes them, so two events
        that would serialise identically become one. Idempotent.
        """
        self._touch()
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
        """
        if mode not in ("native", "server"):
            raise ValueError(f"transform_fov(): mode must be 'native' or 'server', got {mode!r}")

        dx, dy, dz = (float(v) * 1e-3 for v in offset_mm)

        self._touch()
        self._native.apply_fov_shift(dx, dy, dz, bake_adc=(mode == "native"))
        if mode == "server":
            self._native.attach_base_trajectory()

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
        self._touch()
        self._native = _cxx.read_file(str(Path(path)))
        self._definitions = dict(self._native.definitions())

    # -- looking at it ---------------------------------------------------

    def plot(self, **kwargs: object):
        """Plot a window of the sequence through PyPulseq.

        Parameters
        ----------
        tr_index, time_range, block_range
            Which blocks to draw; see :meth:`block_range_of`. The whole
            sequence is drawn when none is given.
        **kwargs
            Forwarded to :meth:`pypulseq.Sequence.plot`.
        """
        return self._window(kwargs).plot(**kwargs)

    def calculate_kspace(self, **kwargs: object):
        """The k-space trajectory of a window, through PyPulseq.

        Parameters
        ----------
        tr_index, time_range, block_range
            Which blocks to compute over; see :meth:`block_range_of`.
        **kwargs
            Forwarded to :meth:`pypulseq.Sequence.calculate_kspace`.

        Returns
        -------
        tuple of numpy.ndarray
            Whatever upstream returns: the trajectory at the ADC samples, the
            whole trajectory, and the excitation, refocusing and ADC instants.
        """
        return self._window(kwargs).calculate_kspace(**kwargs)

    def waveforms(self, **kwargs: object):
        """The played waveforms of a window, through PyPulseq.

        Parameters
        ----------
        tr_index, time_range, block_range
            Which blocks to gather; see :meth:`block_range_of`.
        **kwargs
            Forwarded to :meth:`pypulseq.Sequence.waveforms`.
        """
        return self._window(kwargs).waveforms(**kwargs)

    def waveforms_and_times(self, **kwargs: object):
        """The played waveforms of a window, with their time axes.

        Parameters
        ----------
        tr_index, time_range, block_range
            Which blocks to gather; see :meth:`block_range_of`.
        **kwargs
            Forwarded to :meth:`pypulseq.Sequence.waveforms_and_times`.
        """
        return self._window(kwargs).waveforms_and_times(**kwargs)

    def check_timing(self, **kwargs: object):
        """Check a window's block timing against the raster, through PyPulseq.

        Parameters
        ----------
        tr_index, time_range, block_range
            Which blocks to check; see :meth:`block_range_of`.
        **kwargs
            Forwarded to :meth:`pypulseq.Sequence.check_timing`.

        Returns
        -------
        tuple
            ``(passed, report)``.
        """
        return self._window(kwargs).check_timing(**kwargs)

    def test_report(self, **kwargs: object) -> str:
        """PyPulseq's report on a window of the sequence.

        Parameters
        ----------
        tr_index, time_range, block_range
            Which blocks to report on; see :meth:`block_range_of`.
        **kwargs
            Forwarded to :meth:`pypulseq.Sequence.test_report`.

        Returns
        -------
        str
            The report.
        """
        return self._window(kwargs).test_report(**kwargs)

    def block_range_of(
        self,
        *,
        tr_index: int | None = None,
        time_range: tuple[float, float] | None = None,
        block_range: tuple[int, int] | None = None,
    ) -> tuple[int, int]:
        """Resolve a window specification to an inclusive block range.

        Parameters
        ----------
        tr_index : int, optional
            One repetition, counted from zero. Needs the scan structure, which
            this class does not carry yet.
        time_range : tuple of float, optional
            ``(start, stop)`` in seconds. Every block overlapping the interval
            is included, so a window always starts and ends on a block edge.
        block_range : tuple of int, optional
            ``(first, last)``, 1-based and inclusive.

        Returns
        -------
        tuple of int
            ``(first, last)``, 1-based and inclusive; the whole sequence when
            nothing was asked for.

        Raises
        ------
        ValueError
            If more than one of the three was given, or a range falls outside
            the sequence.
        """
        asked = [
            name
            for name, value in (
                ("tr_index", tr_index),
                ("time_range", time_range),
                ("block_range", block_range),
            )
            if value is not None
        ]
        if len(asked) > 1:
            raise ValueError(f"a window is one of tr_index, time_range or block_range; got {asked}")

        count = self._native.num_blocks()
        if tr_index is not None:
            return self.tr_block_range(tr_index)
        if block_range is not None:
            first, last = (int(value) for value in block_range)
            if not 1 <= first <= last <= count:
                raise ValueError(f"block range {(first, last)} is outside 1..{count}")
            return first, last
        if time_range is not None:
            return self._blocks_over(*time_range)
        return (1, count) if count else (1, 0)

    # -- the scan structure, once there is one ---------------------------

    @property
    def payload(self) -> bytes:
        """bytes : The serialised sequence a plugin hands back. Not implemented yet."""
        raise NotImplementedError(_PENDING.format(what="payload"))

    @property
    def _collection(self):
        """The subsequence collection behind a composed scan. Not implemented yet."""
        raise NotImplementedError(_PENDING.format(what="_collection"))

    @property
    def tr_info(self) -> dict:
        """dict : What one repetition is made of. Not implemented yet."""
        raise NotImplementedError(_PENDING.format(what="tr_info"))

    @property
    def num_trs(self) -> int:
        """int : How many repetitions the scan plays. Not implemented yet."""
        raise NotImplementedError(_PENDING.format(what="num_trs"))

    @property
    def tr_duration(self) -> float:
        """float : How long one repetition lasts. Not implemented yet."""
        raise NotImplementedError(_PENDING.format(what="tr_duration"))

    def tr_block_range(self, tr_index: int) -> tuple[int, int]:
        """The blocks one repetition spans. Not implemented yet."""
        raise NotImplementedError(_PENDING.format(what="tr_block_range"))

    @property
    def segments(self) -> tuple:
        """tuple : The scan's segments. Not implemented yet."""
        raise NotImplementedError(_PENDING.format(what="segments"))

    def segment(self, index: int):
        """One segment of the scan. Not implemented yet."""
        raise NotImplementedError(_PENDING.format(what="segment"))

    def pns(self, **kwargs: object):
        """Peripheral nerve stimulation over the scan. Not implemented yet."""
        raise NotImplementedError(_PENDING.format(what="pns"))

    def mech_resonances(self, **kwargs: object):
        """Mechanical resonance exposure over the scan. Not implemented yet."""
        raise NotImplementedError(_PENDING.format(what="mech_resonances"))

    def grad_spectrum(self, **kwargs: object):
        """The gradient spectrum over the scan. Not implemented yet."""
        raise NotImplementedError(_PENDING.format(what="grad_spectrum"))

    def plot_kspace(self, **kwargs: object):
        """Draw the k-space trajectory. Not implemented yet."""
        raise NotImplementedError(_PENDING.format(what="plot_kspace"))

    # -- internals -------------------------------------------------------

    def _touch(self) -> None:
        """Forget what was derived from the blocks, because they just changed."""
        self._windows.clear()
        self._starts = None

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

    def _window(self, kwargs: dict) -> pp.Sequence:
        """The decoded window ``kwargs`` asks for, consuming the window keys."""
        first, last = self.block_range_of(
            tr_index=kwargs.pop("tr_index", None),
            time_range=kwargs.pop("time_range", None),
            block_range=kwargs.pop("block_range", None),
        )
        window = self._windows.get((first, last))
        if window is None:
            window = self._decode_window(first, last)
            if len(self._windows) >= _WINDOW_CACHE:
                self._windows.pop(next(iter(self._windows)))
            self._windows[(first, last)] = window
        return window

    def _decode_window(self, first: int, last: int) -> pp.Sequence:
        """Blocks ``first..last`` as a standalone :class:`pypulseq.Sequence`.

        Built against a system with no gradient or slew ceiling, so any range
        can be looked at: a window starting mid-ramp is a legitimate thing to
        want to see, and upstream can only reject it because it cannot know a
        preceding block exists.
        """
        window = pp.Sequence(system=_unbounded(self.system))
        for index in range(first, last + 1):
            block = self.get_block(index)
            events = _playable(block, window.system)
            if block.block_duration > 0:
                events.append(pp.make_delay(block.block_duration))
            _append(window, events, index)
        return window

    def _blocks_over(self, start: float, stop: float) -> tuple[int, int]:
        """The blocks overlapping ``start..stop`` seconds, 1-based inclusive."""
        if self._starts is None:
            self._starts = np.concatenate(([0.0], np.cumsum(self._native.block_durations())))
        edges = self._starts
        count = edges.size - 1
        if count == 0:
            return 1, 0
        first = min(max(int(np.searchsorted(edges, start, side="right")) - 1, 0), count - 1)
        last = min(max(int(np.searchsorted(edges, stop, side="left")) - 1, first), count - 1)
        return first + 1, last + 1


# %% local subroutines


def _unbounded(system: pp.Opts) -> pp.Opts:
    """``system`` with no gradient or slew ceiling.

    A decoded window holds waveforms that were checked when they were built.
    Checking them again against a limit they were designed for gains nothing,
    and rejects any window that does not happen to start at zero gradient.
    """
    return pp.Opts(
        max_grad=np.inf,
        max_slew=np.inf,
        rf_dead_time=system.rf_dead_time,
        rf_ringdown_time=system.rf_ringdown_time,
        adc_dead_time=system.adc_dead_time,
        rf_raster_time=system.rf_raster_time,
        grad_raster_time=system.grad_raster_time,
        adc_raster_time=system.adc_raster_time,
        block_duration_raster=system.block_duration_raster,
        gamma=system.gamma,
        B0=system.B0,
    )


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


def _playable(block: SimpleNamespace, system: pp.Opts) -> list:
    """A decoded block's events, with rotation and RF shim resolved into them.

    What comes back is what the scanner plays rather than what the file
    stores: the gradients are rotated, and the RF is spread across the
    transmit channels its shim weights.

    Labels and soft delays are left out. They describe how a console drives
    the sequence, not what it emits, and none of the analyses this window
    feeds reads them.
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
    channel is drawn over the same interval and upstream needs no pTx
    awareness to plot it.
    """
    weights = np.asarray(shim, dtype=complex).ravel()
    if weights.size == 0:
        return rf
    signal = np.asarray(rf.signal).ravel()
    spread = dict(vars(rf))
    spread["signal"] = (weights[:, None] * signal[None, :]).ravel()
    spread["t"] = np.tile(np.asarray(rf.t, dtype=float).ravel(), weights.size)
    return SimpleNamespace(**spread)


def _append(window: pp.Sequence, events: list, index: int) -> None:
    """Add one decoded block to ``window``, saying so if its leading edge moved."""
    try:
        window.add_block(*events)
    except (RuntimeError, ValueError) as raised:
        if "first block has to start at 0" not in str(raised):
            raise
        warnings.warn(
            f"Block {index} starts on a non-zero gradient; its leading edge was zeroed so "
            "the window could be built. Amplitudes elsewhere are unaffected.",
            stacklevel=4,
        )
        for event in events:
            if getattr(event, "type", None) == "grad":
                event.first = 0.0
        window.add_block(*events)
