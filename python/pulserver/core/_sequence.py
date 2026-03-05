"""SequenceCollection class — thin wrapper around pypulseq with C analysis backend."""

__all__ = ['SequenceCollection']

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pypulseq as pp

from ._extension._pulseqlib_wrapper import (
    _PulseqCollection,
    _check_consistency,
    _check_safety,
    _get_num_unique_blocks,
    _get_report,
    _get_unique_block_id,
)
from ._iostream import write_to_stream


def _normalize_block(block):
    """Normalize a pypulseq block in-place.

    * RF: ``freq_offset`` and ``phase_offset`` set to 0; ``signal``
      divided by its peak absolute value.
    * Gradients: arbitrary waveforms divided by peak absolute value;
      trapezoid amplitudes set to ±1.
    * ADC: ``freq_offset`` and ``phase_offset`` set to 0.

    Returns the (mutated) *block* for convenience.
    """
    # ── RF ────────────────────────────────────────────────────
    rf = getattr(block, 'rf', None)
    if rf is not None:
        rf.freq_offset = 0.0
        rf.phase_offset = 0.0
        sig = getattr(rf, 'signal', None)
        if sig is not None:
            peak = np.max(np.abs(sig))
            if peak > 0:
                rf.signal = sig / peak

    # ── Gradients ────────────────────────────────────────────
    for ax in ('gx', 'gy', 'gz'):
        g = getattr(block, ax, None)
        if g is None:
            continue
        wf = getattr(g, 'waveform', None)
        if wf is not None:
            peak = np.max(np.abs(wf))
            if peak > 0:
                g.waveform = wf / peak
        elif hasattr(g, 'amplitude') and g.amplitude != 0:
            g.amplitude = float(np.sign(g.amplitude))

    # ── ADC ──────────────────────────────────────────────────
    adc = getattr(block, 'adc', None)
    if adc is not None:
        if hasattr(adc, 'freq_offset'):
            adc.freq_offset = 0.0
        if hasattr(adc, 'phase_offset'):
            adc.phase_offset = 0.0

    return block


class SequenceCollection(pp.Sequence):
    """
    Extended Sequence that provides TR / segment / safety analysis.

    Wraps one or more :class:`pypulseq.Sequence` objects and creates
    the C-backed ``_PulseqCollection`` used by all analysis functions.

    Parameters
    ----------
    seq : str, Path, pp.Sequence, or list of pp.Sequence
        * ``str`` / ``Path`` — path to the first ``.seq`` file in a
          linked chain (files reference each other via the ``"next"``
          definition key).  The whole chain is loaded automatically.
        * ``pp.Sequence`` — a single sequence (auto-wrapped in a list).
        * ``list[pp.Sequence]`` — an ordered list of sequences.
    parse_labels : bool
        If ``True`` (default), parse label extensions during loading.
    num_averages : int
        Number of averages; influences scan-time calculations.
    """

    def __init__(
        self,
        seq: str | Path | pp.Sequence | list[pp.Sequence],
        parse_labels: bool = True,
        num_averages: int = 1,
    ):
        from ._cache import deserialize

        # ── Normalise input to list[pp.Sequence] ────────────────
        if isinstance(seq, (str, Path)):
            seqs = deserialize(seq)
        elif isinstance(seq, pp.Sequence):
            seqs = [seq]
        elif isinstance(seq, list):
            seqs = seq
        else:
            raise TypeError(
                f"Expected str, Path, pp.Sequence, or list[pp.Sequence]; "
                f"got {type(seq).__name__}"
            )

        # Deep-copy so we never mutate the caller's objects
        seqs = [copy.deepcopy(s) for s in seqs]
        object.__setattr__(self, '_seqs', seqs)

        # Delegate pp.Sequence attribute access to the first sequence
        object.__setattr__(self, '_seq', seqs[0])

        # ── Build the C collection from all sequence blobs ──────
        sys = seqs[0].system
        blobs = [write_to_stream(s) for s in seqs]
        cseq = _PulseqCollection(
            blobs,
            float(sys.gamma),
            float(sys.B0),
            float(sys.max_grad),
            float(sys.max_slew),
            float(sys.rf_raster_time),
            float(sys.grad_raster_time),
            float(sys.adc_raster_time),
            float(sys.block_duration_raster),
            parse_labels,
            num_averages,
        )
        object.__setattr__(self, '_cseq', cseq)

    def __getattribute__(self, name):
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            return getattr(self._seq, name)

    def __setattr__(self, name, value):
        if name in ('_seq', '_cseq', '_seqs'):
            object.__setattr__(self, name, value)
        else:
            setattr(self._seq, name, value)

    # ── Sequence-list accessors ──────────────────────────────

    @property
    def num_sequences(self) -> int:
        """Number of sequences in the collection."""
        return len(self._seqs)

    def get_sequence(self, idx: int) -> pp.Sequence:
        """Return the sequence at *idx* (0-based).

        Parameters
        ----------
        idx : int
            Index into the sequence list.

        Returns
        -------
        pp.Sequence
            A deep copy of the stored sequence.
        """
        if idx < 0 or idx >= len(self._seqs):
            raise IndexError(
                f"idx {idx} out of range (num_sequences={len(self._seqs)})"
            )
        return copy.deepcopy(self._seqs[idx])

    # ── Unique-block accessors ────────────────────────────────

    def num_blocks(self, sequence_idx: int = 0) -> int:
        """Number of unique blocks in the given subsequence.

        Parameters
        ----------
        sequence_idx : int
            Subsequence index (0-based, default 0).
        """
        return _get_num_unique_blocks(self._cseq, sequence_idx)

    def get_block(self, sequence_idx: int, block_idx: int) -> SimpleNamespace:
        """Return the *block_idx*-th unique block, normalised.

        The C library maps *block_idx* to the corresponding 1-based
        ``block_events`` key.  The block is then fetched from the
        pypulseq Sequence and normalised:

        * RF ``freq_offset`` / ``phase_offset`` → 0;
          ``signal`` scaled to unit peak.
        * Gradient waveforms scaled to unit peak; trapezoid amplitudes
          set to ±1.
        * ADC ``freq_offset`` / ``phase_offset`` → 0.

        Parameters
        ----------
        sequence_idx : int
            Subsequence index (0-based).
        block_idx : int
            Unique-block index within *sequence_idx* (0-based).

        Returns
        -------
        SimpleNamespace
            The normalised pypulseq block.
        """
        block_id = _get_unique_block_id(self._cseq, sequence_idx, block_idx)
        block = self._seqs[sequence_idx].get_block(block_id)
        return _normalize_block(block)

    # ── Segment / TR accessors ───────────────────────────────

    def _subseq_report(self, sequence_idx: int) -> dict:
        """Return the report sub-dict for *sequence_idx*, with bounds check."""
        report = _get_report(self._cseq)
        subseqs = report['subsequences']
        if sequence_idx < 0 or sequence_idx >= len(subseqs):
            raise IndexError(
                f"sequence_idx {sequence_idx} out of range "
                f"(num_subsequences={len(subseqs)})"
            )
        return subseqs[sequence_idx]

    def num_segments(self, sequence_idx: int = 0) -> int:
        """Number of unique segments in the given subsequence.

        Parameters
        ----------
        sequence_idx : int
            Subsequence index (0-based, default 0).
        """
        return len(self._subseq_report(sequence_idx)['segments'])

    def segment_size(self, sequence_idx: int, segment_idx: int) -> int:
        """Number of blocks in a segment.

        Parameters
        ----------
        sequence_idx : int
            Subsequence index (0-based).
        segment_idx : int
            Local segment index within *sequence_idx* (0-based).
        """
        ss = self._subseq_report(sequence_idx)
        segments = ss['segments']
        if segment_idx < 0 or segment_idx >= len(segments):
            raise IndexError(
                f"segment_idx {segment_idx} out of range "
                f"(num_segments={len(segments)})"
            )
        return segments[segment_idx]['num_blocks']

    def get_segment(self, sequence_idx: int, segment_idx: int) -> pp.Sequence:
        """Extract a segment as a normalised :class:`pypulseq.Sequence`.

        Uses the C library's *start_block* and *num_blocks* for the
        requested segment to parse the corresponding consecutive blocks
        from the stored pypulseq Sequence, normalises each block, and
        returns a new Sequence built via ``add_block``.

        Parameters
        ----------
        sequence_idx : int
            Subsequence index (0-based).
        segment_idx : int
            Local segment index within *sequence_idx* (0-based).

        Returns
        -------
        pp.Sequence
            A new pypulseq Sequence containing the normalised blocks of
            the requested segment.
        """
        ss = self._subseq_report(sequence_idx)
        segments = ss['segments']
        if segment_idx < 0 or segment_idx >= len(segments):
            raise IndexError(
                f"segment_idx {segment_idx} out of range "
                f"(num_segments={len(segments)})"
            )
        seg = segments[segment_idx]
        start = seg['start_block']   # 0-based
        nblk  = seg['num_blocks']

        src_seq = self._seqs[sequence_idx]
        new_seq = pp.Sequence(system=src_seq.system)
        for i in range(nblk):
            key = start + i + 1          # pypulseq uses 1-based keys
            block = src_seq.get_block(key)
            new_seq.add_block(_normalize_block(block))
        return new_seq

    def tr_size(self, sequence_idx: int = 0) -> int:
        """Number of blocks per TR in the given subsequence.

        Parameters
        ----------
        sequence_idx : int
            Subsequence index (0-based, default 0).
        """
        return self._subseq_report(sequence_idx)['tr_size']

    # ── Report ────────────────────────────────────────────────

    def report(self, *, do_print: bool = False):
        """Structured report of the sequence collection.

        Returns one :class:`~types.SimpleNamespace` per subsequence
        containing:

        - ``num_blocks`` — number of unique blocks.
        - ``segments`` — list of ``(start_block, num_blocks)`` tuples.
        - ``num_prep_blocks`` — preparation blocks before first TR.
        - ``num_cooldown_blocks`` — cooldown blocks after last TR.
        - ``tr_size`` — blocks per TR.
        - ``tr_duration_s`` — TR duration in seconds.
        - ``segment_order`` — ordered segment IDs composing one TR.
        - ``prep_segment_table`` — segment IDs for prep region.
        - ``cooldown_segment_table`` — segment IDs for cooldown.

        Parameters
        ----------
        do_print : bool
            If ``True``, return a formatted string instead.

        Returns
        -------
        list[SimpleNamespace] or str
        """
        raw = _get_report(self._cseq)
        results = []

        for ss_idx, ss_dict in enumerate(raw['subsequences']):
            ns = SimpleNamespace()
            ns.num_blocks = self.num_blocks(ss_idx)
            ns.segments = [
                (seg['start_block'], seg['num_blocks'])
                for seg in ss_dict['segments']
            ]
            ns.num_prep_blocks = ss_dict['num_prep_blocks']
            ns.num_cooldown_blocks = ss_dict['num_cooldown_blocks']
            ns.tr_size = ss_dict['tr_size']
            ns.tr_duration_s = ss_dict['tr_duration_us'] * 1e-6
            ns.segment_order = list(ss_dict['main_segment_table'])
            ns.prep_segment_table = list(ss_dict['prep_segment_table'])
            ns.cooldown_segment_table = list(
                ss_dict['cooldown_segment_table']
            )
            results.append(ns)

        if not do_print:
            return results

        # ── formatted output ──────────────────────────────────
        lines = []
        total_dur_s = raw['total_duration_us'] * 1e-6
        num_ss = raw['num_subsequences']
        lines.append(
            f"Sequence length: {total_dur_s:.6f} s  |  "
            f"Subsequences: {num_ss}  |  "
            f"Total unique segments: {raw['num_segments']}"
        )
        lines.append('')

        for idx, ns in enumerate(results):
            lines.append(f'--- Subsequence {idx} ---')
            lines.append(f'  TR size:            {ns.tr_size} blocks')
            lines.append(
                f'  TR duration:        {ns.tr_duration_s * 1e3:.3f} ms'
            )
            lines.append(f'  Prep blocks:        {ns.num_prep_blocks}')
            lines.append(f'  Cooldown blocks:    {ns.num_cooldown_blocks}')
            lines.append(f'  Unique blocks:      {ns.num_blocks}')
            lines.append(f'  Unique segments:    {len(ns.segments)}')
            lines.append(f'  Segment order (TR): {ns.segment_order}')
            for si, (start, nblk) in enumerate(ns.segments):
                lines.append(
                    f'    seg {si}: start_block={start}, '
                    f'num_blocks={nblk}'
                )
            lines.append('')

        return '\n'.join(lines)

    # ── Safety checks ────────────────────────────────────────

    def check(
        self,
        *,
        stim_threshold: float = 0.0,
        decay_constant_us: float = 0.0,
        forbidden_bands: list[tuple[float, float, float]] | None = None,
        pns_threshold_percent: float = 100.0,
    ) -> None:
        """Run consistency and safety checks.

        Checks performed (in order):

        1. **Consistency** — segment boundaries, RF periodicity, label
           tables.
        2. **Peak gradient** amplitude vs system limit.
        3. **Gradient continuity** across block boundaries.
        4. **Peak slew-rate** vs system limit.
        5. **Acoustic** forbidden-band violations (per segment).
        6. **PNS** threshold check (per segment, if PNS params given).

        Parameters
        ----------
        stim_threshold : float
            PNS stimulation threshold (Hz/m/s).  This equals
            ``rheobase / alpha`` in the SAFE nerve model.  Set > 0
            together with *decay_constant_us* to enable PNS checking.
        decay_constant_us : float
            PNS decay constant / chronaxie (µs).
        forbidden_bands : list of (freq_min, freq_max, max_amplitude)
            Acoustic forbidden-band specifications.  Each tuple gives
            ``(freq_min_Hz, freq_max_Hz, max_allowed_amplitude_Hz_per_m)``.
        pns_threshold_percent : float
            PNS threshold as a percentage (100 = 100 %).

        Raises
        ------
        RuntimeError
            If a consistency or safety violation is detected.
        """
        _check_consistency(self._cseq)

        bands = list(forbidden_bands) if forbidden_bands else []
        skip_pns = stim_threshold <= 0.0 or decay_constant_us <= 0.0

        _check_safety(
            self._cseq,
            forbidden_bands=bands,
            stim_threshold=stim_threshold,
            decay_constant_us=decay_constant_us,
            pns_threshold_percent=pns_threshold_percent,
            skip_pns=skip_pns,
        )

    # ── Plotting ────────────────────────────────────────────

    def pns(
        self,
        *,
        sequence_idx: int = 0,
        stim_threshold: float,
        decay_constant_us: float,
        threshold_percent: float = 80.0,
    ) -> None:
        """Plot convolved PNS waveforms for a representative TR.

        Displays per-axis and combined PNS percentage waveforms
        together with a horizontal threshold line.  No pass/fail
        check is performed — use :meth:`check` for that.

        Parameters
        ----------
        sequence_idx : int
            Subsequence index (0-based, default 0).
        stim_threshold : float
            PNS stimulation threshold (Hz/m/s) = rheobase / alpha.
        decay_constant_us : float
            PNS decay constant / chronaxie (us).
        threshold_percent : float
            Threshold line to draw (default 80 %).
        """
        from ._pns import pns as _pns_impl

        _pns_impl(
            self,
            sequence_idx=sequence_idx,
            stim_threshold=stim_threshold,
            decay_constant_us=decay_constant_us,
            threshold_percent=threshold_percent,
        )

    def grad_spectrum(
        self,
        *,
        sequence_idx: int = 0,
        forbidden_bands: list[tuple[float, float, float]] | None = None,
        window_duration: float = 25.0e-3,
        spectral_resolution: float = 5.0,
        max_frequency: float = 3000.0,
    ) -> None:
        """Plot acoustic spectra for gradient waveforms in a TR.

        Creates a two-row figure: spectrograms on top,
        harmonic spectra with forbidden-band overlays on the bottom.
        No pass/fail check is performed — use :meth:`check` for that.

        Parameters
        ----------
        sequence_idx : int
            Subsequence index (0-based, default 0).
        forbidden_bands : list of (freq_min, freq_max, max_amplitude)
            Forbidden frequency bands (Hz, Hz, Hz/m).
        window_duration : float
            Sliding-window size in seconds (default 25 ms).
        spectral_resolution : float
            Target frequency resolution in Hz (default 5 Hz).
        max_frequency : float
            Upper frequency limit in Hz (default 3000 Hz).
        """
        from ._acoustics import grad_spectrum as _gs_impl

        _gs_impl(
            self,
            sequence_idx=sequence_idx,
            forbidden_bands=forbidden_bands,
            window_duration=window_duration,
            spectral_resolution=spectral_resolution,
            max_frequency=max_frequency,
        )

    def plot(
        self,
        *,
        sequence_idx: int = 0,
        tr_instance: int | str = 'max_pos',
        hide_prep: bool = True,
        hide_cooldown: bool = True,
        collapse_delays: bool = False,
        show_segments: bool = True,
        show_blocks: bool = True,
        show_slew: bool = False,
        show_rf_centers: bool = False,
        show_echoes: bool = False,
        max_grad_mT_per_m: float | None = None,
        max_slew_T_per_m_per_s: float | None = None,
        time_unit: str = 'ms',
        figsize: tuple | None = None,
    ) -> None:
        """Plot (3, 2) TR waveforms with colour-coded segments.

        Displays gradients (Gx, Gy, Gz), RF magnitude / phase, and
        ADC events for a single TR instance of the requested
        subsequence.  Waveforms are obtained from the C library.

        Parameters
        ----------
        sequence_idx : int
            Subsequence index (0-based, default 0).
        tr_instance : int or {'max_pos', 'zero_var'}
            TR instance to display.  Integer values select a specific
            TR (0 … num_averages * num_trs − 1).  Instance 0 may
            include preparation blocks; the last instance may include
            cooldown blocks (controlled by *hide_prep* /
            *hide_cooldown*).

            Special strings:

            - ``'max_pos'`` — position-maximum envelope (default).
            - ``'zero_var'`` — zero variable grads, keep constant (k-space view).
        hide_prep : bool
            Hide preparation blocks (default ``True``).  Only relevant
            when *tr_instance* is 0 and the subsequence has a
            non-degenerate prep section.
        hide_cooldown : bool
            Hide cooldown blocks (default ``True``).  Only relevant
            when *tr_instance* is the last TR and the subsequence has
            a non-degenerate cooldown section.
        collapse_delays : bool
            Shrink pure-delay blocks at C level (default ``False``).
        show_segments : bool
            Colour-code gradient waveforms by segment (default ``True``).
        show_blocks : bool
            Draw vertical dotted lines at block boundaries.
        show_slew : bool
            Overlay slew-rate waveforms on gradient panels.
        show_rf_centers : bool
            Mark RF iso-centres on the RF magnitude subplot.
        show_echoes : bool
            Mark echo (ADC centre) on the ADC panel.
        max_grad_mT_per_m : float, optional
            Horizontal reference line for maximum gradient amplitude.
        max_slew_T_per_m_per_s : float, optional
            Horizontal reference for maximum slew rate.
        time_unit : str
            ``'ms'`` (default) or ``'us'``.
        figsize : tuple, optional
            Figure size.  Default ``(14, 8)``.
        """
        from ._plot import plot as _plot_impl

        _plot_impl(
            self,
            subsequence_idx=sequence_idx,
            tr_idx=tr_instance,
            hide_prep=hide_prep,
            hide_cooldown=hide_cooldown,
            collapse_delays=collapse_delays,
            show_segments=show_segments,
            show_blocks=show_blocks,
            show_slew=show_slew,
            show_rf_centers=show_rf_centers,
            show_echoes=show_echoes,
            max_grad_mT_per_m=max_grad_mT_per_m,
            max_slew_T_per_m_per_s=max_slew_T_per_m_per_s,
            time_unit=time_unit,
            figsize=figsize,
        )

    # ── Validation ───────────────────────────────────────────

    def validate(
        self,
        *,
        sequence_idx: int = 0,
        xml_path: str | Path | None = None,
        do_plot: bool = False,
        tr_range: tuple[int, int] = (0, 1),
        hide_prep: bool = True,
        hide_cooldown: bool = True,
        show_rf_centers: bool = False,
        show_echoes: bool = False,
        show_segments: bool = True,
        show_blocks: bool = False,
        max_grad_mT_per_m: float | bool | None = True,
        grad_atol: float | None = None,
        rf_rms_percent: float = 10.0,
    ) -> dict:
        """Compare C-backend waveforms against a reference source.

        The reference is either the source pypulseq ``Sequence``
        (default, ``xml_path=None``) or an exported XML waveform file.
        Gradient channels are compared by maximum absolute error; RF
        magnitude is compared by RMS percentage error.

        When *do_plot* is ``True`` the C-backend waveform plot is
        created and the reference is overlaid for visual comparison.

        Parameters
        ----------
        sequence_idx : int
            Subsequence index (0-based, default 0).
        xml_path : str, Path or None
            Path to an XML waveform file.  ``None`` (default) validates
            against the stored pypulseq ``Sequence``.
        do_plot : bool
            Show a comparison overlay plot (default ``False``).
        tr_range : (int, int)
            Half-open TR index range ``[start, stop)`` to validate
            (default ``(0, 1)`` — first TR only).
        hide_prep : bool
            Hide preparation blocks in the plot (default ``True``).
        hide_cooldown : bool
            Hide cooldown blocks in the plot (default ``True``).
        show_rf_centers : bool
            Mark RF iso-centres on the plot (default ``False``).
        show_echoes : bool
            Mark echo (ADC centre) on the plot (default ``False``).
        show_segments : bool
            Colour-code gradients by segment (default ``True``).
        show_blocks : bool
            Block-boundary lines on the plot (default ``False``).
        max_grad_mT_per_m : float, bool or None
            Gradient-limit reference line.  ``True`` (default) derives
            the value from system limits; a float is used directly;
            ``False`` / ``None`` disables the line.
        grad_atol : float or None
            Absolute gradient error tolerance in mT/m.  ``None``
            (default) uses ``3 × max_slew × grad_raster_time``.
        rf_rms_percent : float
            RF magnitude percent-RMS error threshold (default 10).

        Returns
        -------
        dict
            ``'ok'`` : bool — ``True`` if all channels pass.
            ``'errors'`` : dict[str, float] — per-channel max/RMS error.
            ``'messages'`` : list[str] — violation messages.
        """
        from ._validate import validate as _validate_impl

        return _validate_impl(
            self,
            sequence_idx=sequence_idx,
            xml_path=xml_path,
            do_plot=do_plot,
            tr_range=tr_range,
            hide_prep=hide_prep,
            hide_cooldown=hide_cooldown,
            show_rf_centers=show_rf_centers,
            show_echoes=show_echoes,
            show_segments=show_segments,
            show_blocks=show_blocks,
            max_grad_mT_per_m=max_grad_mT_per_m,
            grad_atol=grad_atol,
            rf_rms_percent=rf_rms_percent,
        )

    def __str__(self):
        return str(self._seq)
