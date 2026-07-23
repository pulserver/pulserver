"""Fast sequence helpers for production bridge execution."""

from __future__ import annotations

__all__ = ["Segment", "Sequence"]

import math
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace

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
    ):
        super().__init__(system=system, use_block_cache=use_block_cache)
        self.arb_library = EventLibrary()
        self.trap_library = EventLibrary()
        self.rotation_library = EventLibrary()
        self.rf_shim_library = EventLibrary()
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_block(self, *args: SimpleNamespace | float) -> None:
        """Append a block assuming strictly sequential insertion."""
        self._fast_set_block(self.next_free_block_ID, *args)
        self.next_free_block_ID += 1

    def remove_duplicates(self, in_place: bool = False) -> Sequence:
        """Remove duplicates with hardcoded rounded profiles per library.

        Unlike upstream pypulseq, this also compacts extension-related
        libraries and canonicalizes extension linked lists so identical chains
        are shared across blocks.

        Parameters
        ----------
        in_place:
            If ``True``, deduplicate current instance; otherwise return a copy.
        """
        if in_place:
            seq_copy = self
        else:
            tmp = (self.block_cache, self._view_cache, self._collection_cache, self._segment_cache)
            self.block_cache = {}
            self._view_cache = self._collection_cache = self._segment_cache = None
            seq_copy = deepcopy(self)
            (self.block_cache, self._view_cache, self._collection_cache, self._segment_cache) = tmp

        seq_copy.shape_library, shape_map = seq_copy.shape_library.remove_duplicates(9)

        for arb_id in list(seq_copy.arb_library.data):
            data = seq_copy.arb_library.data[arb_id]
            new_data = (*data[0:3], shape_map[data[3]], shape_map[data[4]], data[5])
            if data != new_data:
                seq_copy.arb_library.update(arb_id, None, new_data)

        for rf_id in list(seq_copy.rf_library.data):
            data = seq_copy.rf_library.data[rf_id]
            new_data = (data[0], shape_map[data[1]], shape_map[data[2]], shape_map[data[3]], *data[4:])
            if data != new_data:
                seq_copy.rf_library.update(rf_id, None, new_data, seq_copy.rf_library.type.get(rf_id, "u"))

        for adc_id in list(seq_copy.adc_library.data):
            data = seq_copy.adc_library.data[adc_id]
            shape_id = int(data[7])
            new_data = (*data[0:7], shape_map[shape_id], data[8])
            if data != new_data:
                seq_copy.adc_library.update(adc_id, None, new_data)

        seq_copy.arb_library, arb_map = _dedup_library_approx(seq_copy.arb_library, (6, -6, -6, -6, -6, -6))
        seq_copy.trap_library, trap_map = _dedup_library_approx(seq_copy.trap_library, (6, -6, -6, -6, -6))

        for grad_id in list(seq_copy.grad_library.data):
            grad_type = seq_copy.grad_library.type.get(grad_id, "")
            old_ref = int(seq_copy.grad_library.data[grad_id][0])
            if grad_type == "g":
                new_ref = arb_map[old_ref]
            elif grad_type == "t":
                new_ref = trap_map[old_ref]
            else:
                new_ref = old_ref
            seq_copy.grad_library.update(grad_id, None, (new_ref,), grad_type)

        seq_copy.grad_library, grad_map = _dedup_library_approx(seq_copy.grad_library, (0,))
        seq_copy.rf_library, rf_map = _dedup_library_approx(seq_copy.rf_library, (6, 0, 0, 0, 6, 6, 6, 6, 6, 6))
        seq_copy.adc_library, adc_map = _dedup_library_approx(seq_copy.adc_library, (0, -9, -6, 6, 6, 6, 6, 6, 6))

        for block_id in seq_copy.block_events:
            seq_copy.block_events[block_id][2] = grad_map[seq_copy.block_events[block_id][2]]
            seq_copy.block_events[block_id][3] = grad_map[seq_copy.block_events[block_id][3]]
            seq_copy.block_events[block_id][4] = grad_map[seq_copy.block_events[block_id][4]]
            seq_copy.block_events[block_id][1] = rf_map[seq_copy.block_events[block_id][1]]
            seq_copy.block_events[block_id][5] = adc_map[seq_copy.block_events[block_id][5]]

        seq_copy.trigger_library, trig_map = _dedup_library_approx(seq_copy.trigger_library, (0, 0, 9, 9))
        seq_copy.label_set_library, label_set_map = _dedup_library_approx(seq_copy.label_set_library, (0, 0))
        seq_copy.label_inc_library, label_inc_map = _dedup_library_approx(seq_copy.label_inc_library, (0, 0))
        seq_copy.rotation_library, rotation_map = _dedup_library_approx(seq_copy.rotation_library, 9)

        if seq_copy.rf_shim_library.data:
            widths = {len(v) for v in seq_copy.rf_shim_library.data.values()}
            if len(widths) != 1:
                raise RuntimeError("rf_shim_library has mixed payload widths; cannot apply fixed rounded dedup profile")
            rf_shim_digits = tuple([9] * next(iter(widths)))
            seq_copy.rf_shim_library, rf_shim_map = _dedup_library_approx(seq_copy.rf_shim_library, rf_shim_digits)
        else:
            seq_copy.rf_shim_library, rf_shim_map = _dedup_library_approx(seq_copy.rf_shim_library, 9)

        old_ext_lib = seq_copy.extensions_library
        new_ext_lib = EventLibrary()
        node_cache: dict[tuple[int, int, int], int] = {}

        def remap_ext_ref(ext_type_name: str, old_ref: int) -> int:
            if ext_type_name == "TRIGGERS":
                return trig_map[old_ref]
            if ext_type_name == "LABELSET":
                return label_set_map[old_ref]
            if ext_type_name == "LABELINC":
                return label_inc_map[old_ref]
            if ext_type_name == "DELAYS":
                return 0
            if ext_type_name == "ROTATIONS":
                return rotation_map[old_ref]
            if ext_type_name == "RF_SHIMS":
                return rf_shim_map[old_ref]
            return old_ref

        for block_id, row in seq_copy.block_events.items():
            head_id = int(row[6])
            if head_id == 0:
                continue

            chain: list[tuple[int, int]] = []
            cursor = head_id
            while cursor != 0:
                ext_data = old_ext_lib.data[cursor]
                ext_type_id = int(ext_data[0])
                ext_type_name = seq_copy.get_extension_type_string(ext_type_id)
                remapped_ref = remap_ext_ref(ext_type_name, int(ext_data[1]))
                if remapped_ref != 0:
                    chain.append((ext_type_id, remapped_ref))
                cursor = int(ext_data[2])

            new_next = 0
            for ext_type_id, remapped_ref in reversed(chain):
                key = (ext_type_id, remapped_ref, new_next)
                if key in node_cache:
                    node_id = node_cache[key]
                else:
                    node_id = new_ext_lib.insert(0, key)
                    node_cache[key] = node_id
                new_next = node_id

            seq_copy.block_events[block_id][6] = new_next

        seq_copy.extensions_library = new_ext_lib
        seq_copy.block_cache.clear()
        return seq_copy

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

    def set_block(self, _block_index: int, *args: SimpleNamespace | float) -> None:  # noqa: ARG002
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

    def _fast_set_block(self, block_index: int, *args: SimpleNamespace | float) -> None:
        """Direct-insert block registration: no dedup, no continuity checks, no trace."""
        events = block_to_events(*args)
        new_block = np.zeros(7, dtype=np.int32)
        duration = 0
        extensions = []

        for event in events:
            if isinstance(event, float):
                duration = max(duration, event)
                continue

            if event.type == "rf":
                rf_id, _ = self._fast_register_rf(event)
                new_block[1] = rf_id
                duration = max(duration, event.shape_dur + event.delay + event.ringdown_time)

            elif event.type == "grad":
                channel_num = ["x", "y", "z"].index(event.channel)
                grad_id, _ = self._fast_register_grad(event)
                new_block[2 + channel_num] = grad_id
                grad_duration = (
                    event.delay + math.ceil(event.tt[-1] / self.grad_raster_time - 1e-10) * self.grad_raster_time
                )
                duration = max(duration, grad_duration)

            elif event.type == "trap":
                channel_num = ["x", "y", "z"].index(event.channel)
                new_block[2 + channel_num] = self._fast_register_trap(event)
                duration = max(duration, event.delay + event.rise_time + event.flat_time + event.fall_time)

            elif event.type == "adc":
                adc_id, _ = self._fast_register_adc(event)
                new_block[5] = adc_id
                duration = max(duration, event.delay + event.num_samples * event.dwell + event.dead_time)

            elif event.type == "delay":
                duration = max(duration, event.delay)

            elif event.type in ("output", "trigger"):
                event_id = self._fast_register_control(event)
                extensions.append({"type": self.get_extension_type_ID("TRIGGERS"), "ref": event_id})
                duration = max(duration, event.delay + event.duration)

            elif event.type in ("labelset", "labelinc"):
                label_id = self._fast_register_label(event)
                extensions.append({"type": self.get_extension_type_ID(event.type.upper()), "ref": label_id})

            elif event.type == "soft_delay":
                # Soft delays are intentionally ignored in this fast on-scanner path.
                continue

            elif event.type == "rf_shim":
                rf_shim_id = self._fast_register_rf_shim(event)
                extensions.append({"type": self.get_extension_type_ID("RF_SHIMS"), "ref": rf_shim_id})

            elif event.type == "rot3D":
                rot_id = self._fast_register_rotation(event)
                extensions.append({"type": self.get_extension_type_ID("ROTATIONS"), "ref": rot_id})

            else:
                raise ValueError(f"Unknown event type {event.type} passed to pulserver.pypulseq.Sequence.add_block().")

        if extensions:
            sort_idx = np.argsort([e["ref"] for e in extensions])
            extensions = np.take(extensions, sort_idx)

            all_found = True
            extension_id = 0
            for ext in extensions:
                data = (ext["type"], ext["ref"], extension_id)
                extension_id, found = self.extensions_library.find(data)
                all_found = all_found and found
                if not found:
                    break

            if not all_found:
                extension_id = 0
                for ext in extensions:
                    data = (ext["type"], ext["ref"], extension_id)
                    extension_id, found = self.extensions_library.find(data)
                    if not found:
                        self.extensions_library.insert(extension_id, data)
            new_block[6] = extension_id

        self.block_events[block_index] = new_block
        self.block_durations[block_index] = float(duration)

    # ------------------------------------------------------------------
    # Private direct-insert helpers (no find_or_insert, no dedup)
    # ------------------------------------------------------------------

    def _fast_register_rf(self, event: SimpleNamespace):
        mag = np.abs(event.signal)
        amplitude = np.max(mag)
        mag = mag / amplitude
        mag[np.isnan(mag)] = 0
        phase = np.angle(event.signal)
        phase[phase < 0] += 2 * np.pi
        phase /= 2 * np.pi

        shape_IDs = [0, 0, 0]
        mag_shape = compress_shape(mag)
        shape_IDs[0], _ = self.shape_library.find_or_insert(np.concatenate(([mag_shape.num_samples], mag_shape.data)))
        phase_shape = compress_shape(phase)
        shape_IDs[1], _ = self.shape_library.find_or_insert(
            np.concatenate(([phase_shape.num_samples], phase_shape.data))
        )
        if not (np.floor(event.t / self.rf_raster_time) == np.arange(len(event.t))).all():
            time_shape = compress_shape(event.t / self.rf_raster_time)
            shape_IDs[2], _ = self.shape_library.find_or_insert([time_shape.num_samples, *time_shape.data])

        if not hasattr(event, "use"):
            raise ValueError('Parameter "use" is not optional since v1.5.0')
        use = (
            event.use[0] if event.use in ("excitation", "refocusing", "inversion", "saturation", "preparation") else "u"
        )
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
        rf_id = self.rf_library.insert(0, data, use_code)
        return rf_id, shape_IDs

    def _fast_register_grad(self, event: SimpleNamespace):
        amplitude = np.max(np.abs(event.waveform))
        if amplitude > 0:
            fnz = event.waveform[np.nonzero(event.waveform)[0][0]]
            amplitude *= np.sign(fnz) if fnz != 0 else 1

        shape_IDs = [0, 0]
        g = event.waveform / amplitude if amplitude != 0 else event.waveform
        c_shape = compress_shape(g)
        shape_IDs[0], _ = self.shape_library.find_or_insert(np.concatenate(([c_shape.num_samples], c_shape.data)))

        c_time = compress_shape(event.tt / self.grad_raster_time)
        t_data = np.concatenate(([c_time.num_samples], c_time.data))
        if len(c_time.data) == 4 and np.allclose(c_time.data, [0.5, 1, 1, c_time.num_samples - 3]):
            pass  # standard raster, shape_IDs[1] stays 0
        elif len(c_time.data) == 3 and np.allclose(c_time.data, [0.5, 0.5, c_time.num_samples - 2]):
            shape_IDs[1] = -1
        else:
            shape_IDs[1], _ = self.shape_library.find_or_insert(t_data)

        data = (amplitude, event.first, event.last, *shape_IDs, event.delay)
        arb_id = self.arb_library.insert(0, data)
        grad_id = self.grad_library.insert(0, (arb_id,), "g")
        return grad_id, shape_IDs

    def _fast_register_trap(self, event: SimpleNamespace) -> int:
        data = (event.amplitude, event.rise_time, event.flat_time, event.fall_time, event.delay)
        trap_id = self.trap_library.insert(0, data)
        return self.grad_library.insert(0, (trap_id,), "t")

    def _fast_register_adc(self, event: SimpleNamespace):
        shape_id = 0
        if (
            hasattr(event, "phase_modulation")
            and event.phase_modulation is not None
            and len(event.phase_modulation) > 0
        ):
            phase_shape = compress_shape(np.asarray(event.phase_modulation).flatten())
            shape_data = np.concatenate(([phase_shape.num_samples], phase_shape.data))
            shape_id, _ = self.shape_library.find_or_insert(shape_data)

        data = (
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
        adc_id = self.adc_library.insert(0, data)
        return adc_id, shape_id

    def _fast_register_control(self, event: SimpleNamespace) -> int:
        event_type = ["output", "trigger"].index(event.type)
        event_channel = (["osc0", "osc1", "ext1"] if event_type == 0 else ["physio1", "physio2"]).index(event.channel)
        data = (event_type + 1, event_channel + 1, event.delay, event.duration)
        return self.trigger_library.insert(0, data)

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
        return lib.insert(0, data)

    def _fast_register_rf_shim(self, event: SimpleNamespace) -> int:
        data = (np.abs(event.shim_vector), np.angle(event.shim_vector))
        data = np.stack(data, axis=-1).ravel()
        return self.rf_shim_library.insert(0, tuple(data.tolist()))

    def _fast_register_rotation(self, event: SimpleNamespace) -> int:
        data = tuple(event.rot_quaternion.as_quat(canonical=True, scalar_first=True).tolist())
        return self.rotation_library.insert(0, data)


def _dedup_library_approx(lib: EventLibrary, digits: int | tuple[int, ...]) -> tuple[EventLibrary, dict[int, int]]:
    """Rounded deduplication using hardcoded per-library rounding profiles."""
    new_lib = EventLibrary(numpy_data=lib.numpy_data)
    mapping: dict[int, int] = {0: 0}
    type_code: dict[str | int, int] = {}

    ids = sorted(lib.data)
    if not ids:
        return new_lib, mapping

    if lib.numpy_data:
        rows = [np.asarray(lib.data[old_id], dtype=float).ravel() for old_id in ids]
    else:
        rows = [tuple(lib.data[old_id]) for old_id in ids]

    row_lengths = {len(row) for row in rows}
    all_numeric = all(all(isinstance(v, int | float | np.integer | np.floating) for v in row) for row in rows)

    if len(row_lengths) != 1 or not all_numeric:
        raise RuntimeError("_dedup_library_approx requires uniform, fully numeric payload rows")

    width = next(iter(row_lengths))
    if isinstance(digits, int):
        digits_tuple = tuple([digits] * width)
    else:
        if len(digits) < width:
            raise ValueError(f"Rounding profile length {len(digits)} is shorter than payload width {width}")
        digits_tuple = tuple(digits[:width])

    matrix = np.asarray(rows, dtype=float)
    rounded = _round_sig_matrix(matrix, digits_tuple)

    type_ids = np.asarray([type_code.setdefault(lib.type.get(old_id, ""), len(type_code) + 1) for old_id in ids])
    key_matrix = np.column_stack([type_ids.astype(float), rounded])
    key_bytes = (
        np.ascontiguousarray(key_matrix)
        .view(np.dtype((np.void, key_matrix.dtype.itemsize * key_matrix.shape[1])))
        .ravel()
    )

    _, first_idx, inverse = np.unique(key_bytes, return_index=True, return_inverse=True)
    order = np.argsort(first_idx)

    uniq_to_new_id = np.zeros(len(first_idx), dtype=np.int32)
    for new_id, uniq_idx in enumerate(order, start=1):
        row_idx = int(first_idx[uniq_idx])
        old_id = ids[row_idx]
        type_key = lib.type.get(old_id, "")
        if lib.numpy_data:
            arr = rounded[row_idx].copy()
            arr.flags.writeable = False
            insert_data = arr
        else:
            insert_data = tuple(rounded[row_idx].tolist())
        new_lib.insert(new_id, insert_data, type_key)
        uniq_to_new_id[uniq_idx] = new_id

    mapped = uniq_to_new_id[inverse]
    for i, old_id in enumerate(ids):
        mapping[old_id] = int(mapped[i])

    return new_lib, mapping


def _round_sig_matrix(matrix: np.ndarray, digits: tuple[int, ...]) -> np.ndarray:
    """Vectorized significant-digit rounding for 2D numeric matrices."""
    if matrix.ndim != 2:
        raise ValueError("_round_sig_matrix expects a 2D matrix")
    if len(digits) != matrix.shape[1]:
        raise ValueError(f"Rounding profile length {len(digits)} does not match payload width {matrix.shape[1]}")

    d = np.asarray(digits, dtype=float).reshape(1, -1)
    out = matrix.copy()

    pos_mask = d > 0
    if np.any(pos_mask):
        mags = np.power(10.0, d - np.ceil(np.log10(np.abs(matrix) + 1e-12)))
        rounded_pos = np.round(matrix * mags) / mags
        out = np.where(pos_mask, rounded_pos, out)

    nonpos_mask = ~pos_mask
    if np.any(nonpos_mask):
        mags = np.power(10.0, -d)
        rounded_nonpos = np.round(matrix * mags) / mags
        out = np.where(nonpos_mask, rounded_nonpos, out)

    return out
