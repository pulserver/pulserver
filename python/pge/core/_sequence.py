"""SequenceCollection class — thin wrapper around pypulseq with C analysis backend."""

__all__ = ['SequenceCollection']

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pypulseq as pp

from ._extension._pulseqlib_wrapper import (
    _check_consistency,
    _check_safety,
    _get_num_unique_blocks,
    _get_report,
    _get_sequence_description,
    _get_sequence_parameters,
    _get_unique_block_id,
    _PulseqCollection,
)
from ._iostream import write_to_stream
from ._opts import Opts as PGEOpts


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
        Number of averages; influences scan-time calculations and is used
        by analysis methods such as plotting and validation.
    system : pp.Opts or pge.Opts, optional
        Optional system override used for C-backend initialisation and
        high-level defaults (PNS and forbidden bands). If omitted, uses
        ``seqs[0].system`` from the first sequence.
    """

    def __init__(
        self,
        seq: str | Path | pp.Sequence | list[pp.Sequence],
        parse_labels: bool = True,
        num_averages: int = 1,
        system: pp.Opts | PGEOpts | None = None,
    ):
        from ._cache import deserialize

        # ── Normalise input to list[pp.Sequence] ────────────────
        seq_path: Path | None = None
        raw_blobs: list[bytes] | None = None
        if isinstance(seq, (str, Path)):
            seq_path = Path(seq)
            seqs = deserialize(seq)
            # Collect the file paths in chain order so we can read raw bytes.
            # This avoids a lossy pypulseq round-trip for MATLAB-generated files.
            chain: list[Path] = [seq_path]
            for s in seqs[:-1]:
                next_name = s.definitions.get('NextSequence') or s.definitions.get('next')
                if next_name:
                    chain.append(chain[-1].parent / str(next_name))
            raw_blobs = [p.read_bytes() for p in chain]
        elif isinstance(seq, pp.Sequence):
            seqs = [seq]
        elif isinstance(seq, list):
            seqs = seq
        else:
            raise TypeError(
                f'Expected str, Path, pp.Sequence, or list[pp.Sequence]; '
                f'got {type(seq).__name__}'
            )

        # Deep-copy so we never mutate the caller's objects
        seqs = [copy.deepcopy(s) for s in seqs]
        object.__setattr__(self, '_seqs', seqs)

        # Delegate pp.Sequence attribute access to the first sequence
        object.__setattr__(self, '_seq', seqs[0])
        object.__setattr__(self, '_system_override', system)

        # ── Build the C collection from all sequence blobs ──────
        sys = system if system is not None else seqs[0].system
        # Use raw file bytes when available to avoid a lossy pypulseq round-trip.
        blobs = raw_blobs if raw_blobs is not None else [write_to_stream(s) for s in seqs]
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
        object.__setattr__(self, '_num_averages', int(num_averages))
        object.__setattr__(self, '_seq_path', seq_path)
        object.__setattr__(
            self, '_seq_info_cache', None
        )  # lazy: SequenceDescriptionInfo
        object.__setattr__(
            self, '_traj_info_cache', None
        )  # lazy: TrajectoryInfo | False

    def __getattribute__(self, name):
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            return getattr(self._seq, name)

    def __setattr__(self, name, value):
        if name in (
            '_seq',
            '_cseq',
            '_seqs',
            '_num_averages',
            '_system_override',
            '_seq_path',
            '_seq_info_cache',
            '_traj_info_cache',
        ):
            object.__setattr__(self, name, value)
        elif name == 'system':
            object.__setattr__(self, '_system_override', value)
            setattr(self._seq, name, value)
        else:
            setattr(self._seq, name, value)

    @property
    def system(self):
        """System options in effect for this collection."""
        override = object.__getattribute__(self, '_system_override')
        return override if override is not None else self._seq.system

    def _default_forbidden_bands_hz_per_m(
        self, *, include_channel: bool = False
    ) -> list[tuple[float, float, float] | tuple[float, float, float, str]]:
        """Return default forbidden bands in backend units (Hz, Hz/m)."""
        sys = self.system
        if hasattr(sys, 'forbidden_bands_hz_per_m'):
            try:
                return list(
                    sys.forbidden_bands_hz_per_m(include_channel=include_channel)
                )
            except TypeError:
                return list(sys.forbidden_bands_hz_per_m())
        return []

    @property
    def num_averages(self) -> int:
        """Configured number of averages for this collection."""
        return int(self._num_averages)

    # ── Sequence-list accessors ──────────────────────────────

    @property
    def num_sequences(self) -> int:
        """Total number of sequences in this collection.

        Returns
        -------
        int
            Number of pypulseq Sequence objects loaded during initialization.
            A value of 1 is typical for single-sequence initialization;
            values > 1 indicate linked sequence files (via ``"next"`` definitions).

        See Also
        --------
        get_sequence : Retrieve a sequence by index.
        num_blocks : Number of unique blocks in a subsequence.
        """
        return len(self._seqs)

    def get_sequence(self, idx: int) -> pp.Sequence:
        """Retrieve a deep copy of a sequence by index.

        The returned sequence is a snapshot of the internal sequence
        object and can be modified without affecting the collection.

        Parameters
        ----------
        idx : int
            Index into the sequence list (0-based).

        Returns
        -------
        pp.Sequence
            A deep copy of the stored pypulseq Sequence at position *idx*.

        Raises
        ------
        IndexError
            If *idx* < 0 or *idx* >= ``num_sequences``.

        Examples
        --------
        >>> sc = SequenceCollection('multi_seq_001.seq')  # linked files
        >>> seq1 = sc.get_sequence(1)
        >>> print(f"Sequence 1 has {len(seq1.block_events)} blocks")

        See Also
        --------
        num_sequences : Total number of sequences in the collection.
        """
        if idx < 0 or idx >= len(self._seqs):
            raise IndexError(
                f'idx {idx} out of range (num_sequences={len(self._seqs)})'
            )
        return copy.deepcopy(self._seqs[idx])

    # ── Unique-block accessors ────────────────────────────────

    def num_blocks(self, sequence_idx: int = 0) -> int:
        """Number of unique blocks in a subsequence.

        A "unique block" is a distinct block content identified by the C
        backend's deduplication logic. Identical blocks appearing multiple
        times in the sequence are counted only once.

        Parameters
        ----------
        sequence_idx : int, default 0
            Subsequence index (0-based).

        Returns
        -------
        int
            Number of unique blocks in the subsequence.

        Raises
        ------
        IndexError
            If *sequence_idx* is out of range.

        See Also
        --------
        get_block : Retrieve a normalized unique block by index.
        num_segments : Number of unique segments (collections of blocks).
        """
        return _get_num_unique_blocks(self._cseq, sequence_idx)

    def get_block(self, sequence_idx: int, block_idx: int) -> SimpleNamespace:
        """Retrieve a unique block by index, normalized for comparison.

        The C library maps *block_idx* to the corresponding 1-based
        ``block_events`` key in pypulseq. The block is then fetched,
        normalized, and returned. Normalization:

        - **RF**: ``freq_offset`` and ``phase_offset`` → 0;
          ``signal`` scaled to unit peak amplitude.
        - **Gradients**: waveforms scaled to unit peak; trapezoid
          amplitudes set to ±1.
        - **ADC**: ``freq_offset`` and ``phase_offset`` → 0.

        Parameters
        ----------
        sequence_idx : int
            Subsequence index (0-based).
        block_idx : int
            Unique-block index within *sequence_idx* (0-based).

        Returns
        -------
        SimpleNamespace
            A pypulseq block object (normalized, modified in-place).
            Attributes include ``rf``, ``gx``, ``gy``, ``gz``, ``adc``.

        Raises
        ------
        IndexError
            If *sequence_idx* or *block_idx* is out of range.

        Examples
        --------
        >>> sc = SequenceCollection('path/to/sequence.seq')
        >>> block = sc.get_block(sequence_idx=0, block_idx=5)
        >>> if hasattr(block, 'rf') and block.rf is not None:
        ...     print(f"RF pulse peak: {np.max(np.abs(block.rf.signal)):.3f}")

        See Also
        --------
        num_blocks : Total unique blocks in a subsequence.
        get_segment : Extract an entire segment as a pypulseq Sequence.
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
                f'sequence_idx {sequence_idx} out of range '
                f'(num_subsequences={len(subseqs)})'
            )
        return subseqs[sequence_idx]

    def num_segments(self, sequence_idx: int = 0) -> int:
        """Number of unique segments in a subsequence.

        A "segment" is a collection of blocks that compose a coherent
        imaging unit (e.g., excitation, readout, spoiling). Multiple
        segments are combined to form one TR.

        Parameters
        ----------
        sequence_idx : int, default 0
            Subsequence index (0-based).

        Returns
        -------
        int
            Number of unique segments in the subsequence.

        Raises
        ------
        IndexError
            If *sequence_idx* is out of range.

        See Also
        --------
        get_segment : Extract a segment as a pypulseq Sequence.
        segment_size : Number of blocks in a segment.
        tr_size : Number of blocks per TR.
        """
        return len(self._subseq_report(sequence_idx)['segments'])

    def segment_size(self, sequence_idx: int, segment_idx: int) -> int:
        """Number of blocks in a segment.

        A segment is identified by its position within a subsequence.

        Parameters
        ----------
        sequence_idx : int
            Subsequence index (0-based).
        segment_idx : int
            Local segment index within *sequence_idx* (0-based).

        Returns
        -------
        int
            Number of blocks composing the segment.

        Raises
        ------
        IndexError
            If *sequence_idx* or *segment_idx* is out of range.

        Examples
        --------
        >>> sc = SequenceCollection('path/to/sequence.seq')
        >>> for seg_idx in range(sc.num_segments()):
        ...     size = sc.segment_size(0, seg_idx)
        ...     print(f"Segment {seg_idx}: {size} blocks")

        See Also
        --------
        get_segment : Extract a segment as a pypulseq Sequence.
        num_segments : Total segments in a subsequence.
        """
        ss = self._subseq_report(sequence_idx)
        segments = ss['segments']
        if segment_idx < 0 or segment_idx >= len(segments):
            raise IndexError(
                f'segment_idx {segment_idx} out of range '
                f'(num_segments={len(segments)})'
            )
        return segments[segment_idx]['num_blocks']

    def get_segment(self, sequence_idx: int, segment_idx: int) -> pp.Sequence:
        """Extract a segment as a normalized pypulseq Sequence.

        Uses the C library's *start_block* and *num_blocks* for the
        requested segment to parse the corresponding consecutive blocks
        from the stored pypulseq Sequence. Each block is normalized
        (see :meth:`get_block`), and a new Sequence is constructed.

        Parameters
        ----------
        sequence_idx : int
            Subsequence index (0-based).
        segment_idx : int
            Local segment index within *sequence_idx* (0-based).

        Returns
        -------
        pp.Sequence
            A new pypulseq Sequence containing the normalized blocks of
            the requested segment.

        Raises
        ------
        IndexError
            If *sequence_idx* or *segment_idx* is out of range.

        Examples
        --------
        >>> sc = SequenceCollection('path/to/sequence.seq')
        >>> seg = sc.get_segment(sequence_idx=0, segment_idx=2)
        >>> print(f"Segment has {len(seg.block_events)} blocks")

        Notes
        -----
        The returned Sequence uses the same system parameters as the
        original sequence(s).

        See Also
        --------
        get_block : Retrieve a single normalized block.
        num_segments : Total segments in a subsequence.
        segment_size : Blocks in a segment.
        """
        ss = self._subseq_report(sequence_idx)
        segments = ss['segments']
        if segment_idx < 0 or segment_idx >= len(segments):
            raise IndexError(
                f'segment_idx {segment_idx} out of range '
                f'(num_segments={len(segments)})'
            )
        seg = segments[segment_idx]
        start = seg['start_block']  # 0-based
        nblk = seg['num_blocks']

        src_seq = self._seqs[sequence_idx]
        new_seq = pp.Sequence(system=src_seq.system)
        for i in range(nblk):
            key = start + i + 1  # pypulseq uses 1-based keys
            block = src_seq.get_block(key)
            new_seq.add_block(_normalize_block(block))
        return new_seq

    def tr_size(self, sequence_idx: int = 0) -> int:
        """Number of blocks per TR (repetition time) in a subsequence.

        A TR is composed of multiple segments (e.g., preparation, readout,
        spoiling). This method returns the total number of blocks required
        to form one complete TR.

        Parameters
        ----------
        sequence_idx : int, default 0
            Subsequence index (0-based).

        Returns
        -------
        int
            Number of blocks per TR in the subsequence.

        Raises
        ------
        IndexError
            If *sequence_idx* is out of range.

        Examples
        --------
        >>> sc = SequenceCollection('path/to/sequence.seq')
        >>> blocks_per_tr = sc.tr_size(0)
        >>> num_segs = sc.num_segments(0)
        >>> print(f"TR composed of {num_segs} segments, {blocks_per_tr} blocks total")

        See Also
        --------
        num_segments : Segments per TR.
        segment_size : Blocks per segment.
        """
        return self._subseq_report(sequence_idx)['tr_size']

    # ── Report ────────────────────────────────────────────────

    def report(self, *, do_print: bool = False):
        """Structured report of the sequence collection organization.

        Returns metadata about each subsequence including unique blocks,
        segments, TR timing, and segment ordering. Useful for understanding
        sequence structure and debugging layout issues.

        Returns one :class:`~types.SimpleNamespace` per subsequence containing:

        - ``num_blocks`` (int) — number of unique blocks in the subsequence.
        - ``segments`` (list of tuple) — ``(start_block, num_blocks)`` tuples
          for each segment.
        - ``num_prep_blocks`` (int) — preparation blocks before first TR.
        - ``num_cooldown_blocks`` (int) — cooldown blocks after last TR.
        - ``tr_size`` (int) — number of blocks per TR.
        - ``tr_duration_s`` (float) — TR duration in seconds.
        - ``segment_order`` (list) — ordered segment IDs composing one TR.
        - ``prep_segment_table`` (list) — segment IDs for prep region.
        - ``cooldown_segment_table`` (list) — segment IDs for cooldown region.

        Parameters
        ----------
        do_print : bool, default False
            If ``False`` (default), return structured data objects.
            If ``True``, return a formatted human-readable string.

        Returns
        -------
        list[SimpleNamespace] or str
            If ``do_print=False``: list of :class:`types.SimpleNamespace` objects,
            one per subsequence, containing the fields listed above.
            If ``do_print=True``: formatted text summary suitable for console output.

        Examples
        --------
        Get structured metadata for all subsequences:

        >>> sc = SequenceCollection('path/to/sequence.seq')
        >>> reports = sc.report()
        >>> for idx, rep in enumerate(reports):
        ...     print(f"Subseq {idx}: {rep.num_blocks} blocks, TR={rep.tr_duration_s*1e3:.1f}ms")

        Print human-readable summary:

        >>> print(sc.report(do_print=True))
        Sequence length: 1.200000 s  |  Subsequences: 2  |  Total unique segments: 5
        --- Subsequence 0 ---
          TR size:            64 blocks
          TR duration:        12.345 ms
          Prep blocks:        8
          Cooldown blocks:    4
          ...

        See Also
        --------
        get_sequence : Retrieve a specific sequence by index.
        get_block : Retrieve a normalized unique block.
        get_segment : Extract a segment as a new pypulseq Sequence.
        """
        raw = _get_report(self._cseq)
        results = []

        for ss_idx, ss_dict in enumerate(raw['subsequences']):
            ns = SimpleNamespace()
            ns.num_blocks = self.num_blocks(ss_idx)
            ns.segments = [
                (seg['start_block'], seg['num_blocks']) for seg in ss_dict['segments']
            ]
            ns.num_prep_blocks = ss_dict['num_prep_blocks']
            ns.num_cooldown_blocks = ss_dict['num_cooldown_blocks']
            ns.tr_size = ss_dict['tr_size']
            ns.tr_duration_s = ss_dict['tr_duration_us'] * 1e-6
            ns.segment_order = list(ss_dict['main_segment_table'])
            ns.prep_segment_table = list(ss_dict['prep_segment_table'])
            ns.cooldown_segment_table = list(ss_dict['cooldown_segment_table'])
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
            lines.append(f'  TR duration:        {ns.tr_duration_s * 1e3:.3f} ms')
            lines.append(f'  Prep blocks:        {ns.num_prep_blocks}')
            lines.append(f'  Cooldown blocks:    {ns.num_cooldown_blocks}')
            lines.append(f'  Unique blocks:      {ns.num_blocks}')
            lines.append(f'  Unique segments:    {len(ns.segments)}')
            lines.append(f'  Segment order (TR): {ns.segment_order}')
            for si, (start, nblk) in enumerate(ns.segments):
                lines.append(
                    f'    seg {si}: start_block={start}, ' f'num_blocks={nblk}'
                )
            lines.append('')

        # ── Optional section-6 / section-5 summary lines ────────
        try:
            si = self.info()
            sp = si.seq_params
            te_ms = sp.min_te_us * 1e-3
            tr_ms = sp.min_tr_us * 1e-3
            fa = sp.max_flip_angle_deg
            t_s = sp.total_scan_time_us * 1e-6
            lines.append(
                f'Sequence description: '
                f'min_TE={te_ms:.3f} ms  min_TR={tr_ms:.3f} ms  '
                f'max_FA={fa:.1f} deg  total={t_s:.3f} s'
            )
        except Exception:  # noqa: S110
            pass

        try:
            ti = self.trajectory_info()
            if ti is not None:
                lines.append(
                    f'Trajectory: {len(ti.kshots)} kshots  '
                    f'{len(ti.encoding_spaces)} encoding space(s)  '
                    f'{len(ti.table)} ADC entries'
                )
        except Exception:  # noqa: S110
            pass

        return '\n'.join(lines)

    # ── Sequence description & trajectory ────────────────────

    def info(self):
        """Return the sequence description for all subsequences.

        Builds :class:`SequenceDescriptionInfo` from the loaded collection
        using the C backend.  Always available — no ``.pge`` cache file
        required.

        Returns
        -------
        SequenceDescriptionInfo
            Global sequence parameters plus per-subsequence event lists,
            RF shape tuples, shim definitions, and composite RF groups.
        """
        from ._cache_sections import (
            build_sequence_description_info,
        )

        cached = object.__getattribute__(self, '_seq_info_cache')
        if cached is not None:
            return cached

        params_dict = _get_sequence_parameters(self._cseq)
        n = params_dict['num_subseqs']
        desc_dicts = [_get_sequence_description(self._cseq, i) for i in range(n)]
        result = build_sequence_description_info(params_dict, desc_dicts)
        object.__setattr__(self, '_seq_info_cache', result)
        return result

    def trajectory_info(self):
        """Return trajectory data from the binary cache, if present.

        Reads the TRAJECTORY section (id 6) from the ``.pge`` cache file
        written alongside the ``.seq`` file at load time.  Returns
        ``None`` for Cartesian sequences or when no ``.pge`` file is present.

        Returns
        -------
        TrajectoryInfo or None
            k-shot library, encoding spaces, and per-ADC trajectory table;
            or ``None`` if unavailable.
        """
        from ._cache_sections import read_trajectory_info

        cached = object.__getattribute__(self, '_traj_info_cache')
        if cached is not None:
            # False sentinel means "already checked — not available"
            return None if cached is False else cached

        seq_path = object.__getattribute__(self, '_seq_path')
        result = read_trajectory_info(seq_path) if seq_path is not None else None
        object.__setattr__(
            self, '_traj_info_cache', result if result is not None else False
        )
        return result

    def describe(self, *, do_print: bool = True) -> str:
        """Return a detailed multi-line summary of sequence description and trajectory.

        Combines the sequence description (SEQDESC section) and, when available,
        the trajectory (TRAJECTORY section) into a human-readable report.

        Parameters
        ----------
        do_print : bool, default True
            When ``True`` (default), the summary is also printed to stdout.

        Returns
        -------
        str
            Formatted multi-line string.
        """
        lines = []

        # ── Sequence parameters ──────────────────────────────
        si = self.info()
        sp = si.seq_params
        lines.append('=== Sequence Description ===')
        lines.append(f'  Subsequences:       {sp.num_subseqs}')
        lines.append(f'  Min TE:             {sp.min_te_us * 1e-3:.3f} ms')
        lines.append(f'  Min TR:             {sp.min_tr_us * 1e-3:.3f} ms')
        lines.append(f'  Max TR:             {sp.max_tr_us * 1e-3:.3f} ms')
        lines.append(f'  Max flip angle:     {sp.max_flip_angle_deg:.1f} deg')
        lines.append(f'  Total scan time:    {sp.total_scan_time_us * 1e-6:.3f} s')
        lines.append('')

        # ── Per-subsequence summary ──────────────────────────
        for sd in si.subseqs:
            lines.append(f'  --- Subsequence {sd.subseq_idx} ---')
            lines.append(f'    TR duration:      {sd.tr_duration_us * 1e-3:.3f} ms')
            num_rf = sd.num_rf_events
            num_adc = sd.num_adc_events
            lines.append(
                f'    Events:           {len(sd.events)} total '
                f'({num_rf} RF, {num_adc} ADC)'
            )
            if sd.flip_angles_deg:
                fa_str = ', '.join(f'{f:.1f}' for f in sd.flip_angles_deg[:6])
                suffix = ', ...' if len(sd.flip_angles_deg) > 6 else ''
                lines.append(f'    Flip angles (deg):{fa_str}{suffix}')
            lines.append('')

        # ── Trajectory section ───────────────────────────────
        ti = self.trajectory_info()
        if ti is not None:
            lines.append('=== Trajectory ===')
            lines.append(f'  k-shots:            {len(ti.kshots)}')
            lines.append(f'  Encoding spaces:    {len(ti.encoding_spaces)}')
            lines.append(f'  Table entries:      {len(ti.table)}')
            for ei, es in enumerate(ti.encoding_spaces):
                # Stage 1.5c: FOV/matrix are no longer duplicated on the
                # encoding space -- read the owning subsequence's own
                # pypulseq [DEFINITIONS] (FOV/Matrix, or NavFOV/NavMatrix
                # when geometry_tag == 1). Pulseq FOV is in METERS.
                fov_mm = mat = None
                if 0 <= es.subseq_idx < len(self._seqs):
                    defs = self._seqs[es.subseq_idx].definitions
                    fov_key = 'NavFOV' if es.geometry_tag == 1 else 'FOV'
                    mat_key = 'NavMatrix' if es.geometry_tag == 1 else 'Matrix'
                    fov_raw = defs.get(fov_key)
                    mat_raw = defs.get(mat_key)
                    if fov_raw is not None:
                        fov_mm = tuple(round(float(v) * 1e3, 1) for v in fov_raw)
                    if mat_raw is not None:
                        mat = tuple(int(float(v)) for v in mat_raw)
                lines.append(f'  ES {ei}: FOV={fov_mm} mm  matrix={mat}')
            lines.append('')
        else:
            lines.append('=== Trajectory: not available ===')
            lines.append('')

        text = '\n'.join(lines)
        if do_print:
            print(text)
        return text

    # ── Safety checks ────────────────────────────────────────

    def check(
        self,
        *,
        stim_threshold: float = 0.0,
        decay_constant_us: float = 0.0,
        forbidden_bands: list[tuple[float, float, float]] | None = None,
        pns_threshold_percent: float = 80.0,
    ) -> None:
        """Run consistency and safety checks on the sequence collection.

        Performs a series of automated checks in order:

        1. **Consistency** — segment boundaries, RF periodicity, label tables.
        2. **Peak gradient** amplitude vs system limit.
        3. **Gradient continuity** across block boundaries.
        4. **Peak slew-rate** vs system limit.
        5. **Acoustic spectra** — forbidden-band violations (per segment, if bands given).
        6. **PNS** — peripheral nerve stimulation risk (per segment, if PNS params given).

        Parameters
        ----------
        stim_threshold : float, default 0.0
            PNS stimulation threshold in Hz/m/s. This equals ``rheobase / alpha``
            in the SAFE nerve model. Set > 0 together with *decay_constant_us*
            to enable PNS safety checking. Set to 0 to skip PNS checks.
        decay_constant_us : float, default 0.0
            PNS decay constant / chronaxie in microseconds. Set > 0 together
            with *stim_threshold* to enable PNS checking.
        forbidden_bands : list of (freq_min, freq_max, max_amplitude), optional
            Mechanical resonance forbidden-band specifications for gradient spectrum analysis.
            Each tuple gives ``(freq_min_Hz, freq_max_Hz, max_allowed_amplitude_Hz_per_m)``.
            If ``None``, use defaults from ``self.system`` when available
            (e.g., :class:`pge.Opts`); otherwise no mechanical resonance checks are performed.
        pns_threshold_percent : float, default 80.0
            PNS threshold as a percentage of maximum stimulation (100.0 = 100 %).
            Used only if both *stim_threshold* and *decay_constant_us* are > 0.

        Raises
        ------
        RuntimeError
            If any consistency or safety violation is detected. The exception
            message contains details about the specific failure location
            (subsequence, block, segment, etc.).

        Notes
        -----
        If *stim_threshold* or *decay_constant_us* is ≤ 0, PNS checking is
        automatically skipped regardless of the other parameter's value.

        All checks are performed on all subsequences in the collection.

        Examples
        --------
        Check consistency and default safety limits:

        >>> sc = SequenceCollection('path/to/sequence.seq')
        >>> sc.check()  # Raises RuntimeError on failure

        Check with custom PNS parameters (Siebold et al. 2015):

        >>> sc.check(
        ...     stim_threshold=120.0,
        ...     decay_constant_us=360.0,
        ...     pns_threshold_percent=80.0,
        ... )

        Check with forbidden mechanical resonance bands:

        >>> bands = [
        ...     (100, 500, 1.0),    # Band 1: 100-500 Hz, max 1.0 Hz/m
        ...     (2000, 3000, 0.5),  # Band 2: 2-3 kHz, max 0.5 Hz/m
        ... ]
        >>> sc.check(forbidden_bands=bands)

        See Also
        --------
        validate : Validate waveforms against reference implementation.
        pns : Plot PNS waveforms (visualization only, no pass/fail).
        grad_spectrum : Plot gradient spectrum (visualization only, no pass/fail).
        """
        _check_consistency(self._cseq)

        if forbidden_bands is None:
            bands = self._default_forbidden_bands_hz_per_m()
        else:
            bands = list(forbidden_bands)
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
        sequence_idx: int | None = None,
        stim_threshold: float | None = None,
        decay_constant_us: float | None = None,
        threshold_percent: float | list[float] | tuple[float, ...] | None = None,
    ) -> None:
        """Plot peripheral nerve stimulation (PNS) waveforms for a TR.

        Displays per-axis (X, Y, Z) and combined PNS percentage waveforms
        together with horizontal threshold guide line(s). No pass/fail check
        is performed — use :meth:`check` for safety validation.

        Canonical-TR selection follows the C-backend logic per subsequence.
        One figure is produced for each canonical TR.

        Parameters
        ----------
        sequence_idx : int, optional
            Subsequence index (0-based) to analyze. If ``None`` (default),
            all subsequences are analyzed.
        stim_threshold : float, optional
            PNS stimulation threshold in Hz/m/s. This equals
            ``rheobase / alpha`` in the SAFE nerve model (Siebold et al. 2015).
            If ``None``, attempts to use ``self.system.default_stim_threshold()``.
        decay_constant_us : float, optional
            PNS decay constant / chronaxie in microseconds. Typical values
            are 330-360 us for standard stimulation models. If ``None``,
            attempts to use ``self.system.chronaxie_us``.
        threshold_percent : float or sequence of float, default [80.0, 100.0]
            Threshold line(s) to draw on the plot, as percentage of maximum
            stimulation (100.0 = 100 %). Accepts:

            - Single ``float``: e.g., ``80.0`` draws one line at 80 %.
            - Sequence (list/tuple): e.g., ``[80.0, 100.0]`` draws two lines.

        Raises
        ------
        ValueError
            If *sequence_idx* is out of range, or if required PNS parameters
            are unavailable from both call arguments and ``self.system``.

        Notes
        -----
        This is a visualization function only. To enforce PNS safety limits,
        use :meth:`check` with *stim_threshold* and *decay_constant_us*.

        The waveforms are computed using the C backend's convolution-based
        approach, which models nerve response dynamics accurately.

        Examples
        --------
        Plot PNS waveforms with 80% and 100% threshold lines:

        >>> sc = SequenceCollection('path/to/sequence.seq')
        >>> sc.pns(stim_threshold=120.0, decay_constant_us=360.0)

        Plot with only a single 80% threshold line:

        >>> sc.pns(
        ...     sequence_idx=0,
        ...     stim_threshold=120.0,
        ...     decay_constant_us=360.0,
        ...     threshold_percent=[80.0],
        ... )

        See Also
        --------
        check : Run safety checks including PNS validation.
        grad_spectrum : Plot gradient spectrum (gradient-based).
        """
        from ._pns import pns as _pns_impl

        if threshold_percent is None:
            threshold_percent = [80.0, 100.0]
        if stim_threshold is None and hasattr(self.system, 'default_stim_threshold'):
            stim_threshold = self.system.default_stim_threshold()
        if decay_constant_us is None and hasattr(self.system, 'chronaxie_us'):
            decay_constant_us = self.system.chronaxie_us

        if stim_threshold is None or decay_constant_us is None:
            raise ValueError(
                'stim_threshold and decay_constant_us are required unless provided by system'
            )

        _pns_impl(
            self,
            sequence_idx=sequence_idx,
            stim_threshold=float(stim_threshold),
            decay_constant_us=float(decay_constant_us),
            threshold_percent=threshold_percent,
        )

    def calculate_pns(self, **kwargs) -> None:
        """Alias for :meth:`pns`."""
        self.pns(**kwargs)

    def grad_spectrum(
        self,
        *,
        sequence_idx: int | None = None,
        forbidden_bands: (
            list[tuple[float, float, float] | tuple[float, float, float, str]] | None
        ) = None,
        spectral_resolution: float = 5.0,
        max_frequency: float = 3000.0,
        peak_log10_threshold: float | None = None,
        peak_norm_scale: float | None = None,
        peak_eps: float | None = None,
        peak_prominence: float | None = None,
    ) -> None:
        """Plot gradient spectrum (mechanical resonance candidates) in the canonical TR.

        Creates a single-panel figure with three overlaid spectra (Gx, Gy, Gz) showing
        surviving peak candidates as vertical lines, forbidden bands as shaded regions
        with max-allowed amplitude labels (mT/m), and per-candidate max gradient
        amplitude annotations (mT/m). Two frequency axes are shown: frequency in Hz
        (bottom) and the corresponding echo spacing in ms (top). The vertical axis
        is in arbitrary units (a.u.).

        No pass/fail check is performed — use :meth:`check` for that.

        Canonical-TR selection follows the C-backend logic per subsequence.
        One figure is produced for each canonical TR.

        Parameters
        ----------
        sequence_idx : int, optional
            Subsequence index (0-based) to analyze. If ``None`` (default),
            all subsequences are analyzed.
        forbidden_bands : list of (freq_min, freq_max, max_amplitude), optional
            Forbidden-band specifications.  Each tuple gives
            ``(freq_min_Hz, freq_max_Hz, max_allowed_amplitude_Hz_per_m)``.
            Drawn as shaded regions with max-allowed amplitude labelled in mT/m.
            If ``None``, defaults are used from ``self.system`` when available.
        spectral_resolution : float, default 5.0
            Target frequency resolution for FFT in Hz (default 5 Hz).
        max_frequency : float, default 3000.0
            Upper frequency limit in Hz (default 3 kHz).
        peak_log10_threshold : float, optional
            Resonance detector threshold in log10 space. Higher values detect
            fewer peaks.
        peak_norm_scale : float, optional
            Normalization scale used before log transform in resonance detection.
        peak_eps : float, optional
            Positive epsilon added before log transform for numerical stability.
        peak_prominence : float, optional
            Minimum prominence (in log10 units) for a detected peak.

        Raises
        ------
        ValueError
            If *sequence_idx* is out of range.

        Notes
        -----
        This is a visualization function only. To enforce mechanical resonance
        safety limits, use :meth:`check` with *forbidden_bands*.

        Examples
        --------
        Plot gradient spectrum with forbidden bands:

        >>> sc = SequenceCollection('path/to/sequence.seq')
        >>> bands = [
        ...     (100, 500, 1.0),    # 100-500 Hz: max 1.0 Hz/m
        ...     (2000, 3000, 0.5),  # 2-3 kHz: max 0.5 Hz/m
        ... ]
        >>> sc.grad_spectrum(forbidden_bands=bands)

        See Also
        --------
        check : Run mechanical-resonance safety checks.
        pns : Plot peripheral nerve stimulation waveforms.
        """
        from ._mechresonance import grad_spectrum as _gs_impl

        if forbidden_bands is None:
            forbidden_bands = self._default_forbidden_bands_hz_per_m(
                include_channel=True
            )

        _gs_impl(
            self,
            sequence_idx=sequence_idx,
            forbidden_bands=forbidden_bands,
            spectral_resolution=spectral_resolution,
            max_frequency=max_frequency,
            peak_log10_threshold=peak_log10_threshold,
            peak_norm_scale=peak_norm_scale,
            peak_eps=peak_eps,
            peak_prominence=peak_prominence,
        )

    def calculate_gradient_spectrum(self, **kwargs) -> None:
        """Alias for :meth:`grad_spectrum`."""
        self.grad_spectrum(**kwargs)

    def plot(
        self,
        *,
        subsequence_idx: int | None = None,
        tr_instance: int | str = 'max_pos',
        collapse_delays: bool = True,
        show_segments: bool = True,
        show_blocks: bool = True,
        show_centers: bool = False,
        figsize: tuple | None = None,
    ) -> None:
        """Plot TR waveforms with colour-coded segments and optional overlays.

        Displays a (3, 2) figure showing gradients (Gx, Gy, Gz), RF magnitude / phase,
        and ADC events for a single TR instance or set of canonical TRs of the requested
        subsequence. Waveforms are obtained from the C-backed analysis library using native timing.

        **Canonical TR display** (when ``tr_instance`` is a string): All canonical TRs
        (one per unique shot-ID pattern) are overlaid on the same figure. The first canonical
        TR (ID 0) is displayed with full opacity; subsequent TRs (ID > 0) use reduced opacity
        or dashing to distinguish multi-shot patterns.

        **Actual instance display** (when ``tr_instance`` is an integer): A specific TR
        instance is plotted in actual amplitude mode.

        Parameters
        ----------
        subsequence_idx : int or None, default None
            Subsequence index (0-based). If ``None`` and there is exactly one subsequence,
            defaults to 0. If ``None`` and there are multiple subsequences, raises ``ValueError``.
        tr_instance : int or str, default 'max_pos'
            TR instance selector. Accepts:

            - ``'max_pos'`` (default) — structural canonical TR(s) showing position maximum
              across all shots; multi-shot patterns overlaid with transparency.
            - ``'zero_var'`` — canonical TR(s) with zero-variable gradients (k-space view);
              multi-shot patterns overlaid with transparency.
            - Non-negative integer: 0-based index into the sequence in actual amplitude mode.
            - Negative integer: ``-1`` for last instance, ``-2`` for second-to-last, etc.

        collapse_delays : bool, default True
            If ``True``, remove zero-duration delay blocks from the timeline.
        show_segments : bool, default True
            If ``True``, colour-code waveforms by segment index and draw segment boundaries.
        show_blocks : bool, default False
            If ``True``, draw vertical lines at block boundaries.
        show_centers : bool, default False
            If ``True``, mark RF pulse envelope peaks and readout echoes with vertical markers.
        figsize : tuple, optional
            Figure size ``(width, height)`` in inches. If ``None``, uses matplotlib's
            default sizing (typically 10 x 6 inches).

        Raises
        ------
        ValueError
            If ``subsequence_idx=None`` and there are multiple subsequences.
            If ``tr_instance`` (integer) is out of range for the selected subsequence.

        Notes
        -----
        Native timing reveals ADC sampling patterns and spectral artifacts masked by
        uniform resampling.

        Multi-canonical-TR overlay (when ``tr_instance`` is a string and multiple shot
        patterns exist): Canonical TR 0 is displayed solid/opaque; TRs with ID > 0 use
        reduced opacity or dashing to indicate multi-shot behavior.

        Examples
        --------
        Plot canonical TRs with default settings:

        >>> sc = SequenceCollection('path/to/sequence.seq', num_averages=3)
        >>> sc.plot(subsequence_idx=0)  # Shows all canonical TRs overlaid

        Plot with annotations:

        >>> sc.plot(
        ...     subsequence_idx=0,
        ...     tr_instance='max_pos',
        ...     show_segments=True,
        ...     show_centers=True
        ... )

        Plot a specific actual instance:

        >>> sc.plot(subsequence_idx=0, tr_instance=5)  # 5th instance, actual amp

        See Also
        --------
        validate : Validate waveforms against reference.
        check : Run consistency and safety checks.
        """
        from ._plot import plot as _plot_impl

        _plot_impl(
            self,
            subsequence_idx=subsequence_idx,
            tr_instance=tr_instance,
            collapse_delays=collapse_delays,
            show_segments=show_segments,
            show_blocks=show_blocks,
            show_slew=False,
            show_rf_centers=show_centers,
            show_echoes=show_centers,
            max_grad_mT_per_m=None,
            max_slew_T_per_m_per_s=None,
            time_unit='ms',
            figsize=figsize,
        )

    # ── Validation ───────────────────────────────────────────

    def validate(
        self,
        *,
        do_plot: bool = False,
        subsequence_idx: int | None = None,
        tr_instance: int | None = None,
        grad_atol: float | None = None,
        rf_rms_percent: float = 10.0,
    ) -> bool:
        """Validate scan-table waveforms against a reference implementation.

        Compares C-backend waveforms against pypulseq reference blocks,
        reporting pass/fail status and RMS errors per channel. Scope selection
        follows ``None``-driven semantics:

        - **do_plot=False** (default): ``None`` defaults iterate *all* selected
          dimensions. ``subsequence_idx=None`` → all subsequences;
          ``tr_instance=None`` → all TRs per subsequence.

        - **do_plot=True**: ``None`` defaults are not allowed; caller must
          specify explicit targets. Single-candidate auto-select is permitted
          (i.e., if collection has exactly one subsequence, ``subsequence_idx=None``
          auto-selects to 0).

        Parameters
        ----------
        do_plot : bool, default False
            If ``True``, visually compare waveforms; enforces explicit targeting
            semantics when ``do_plot=False`` allows traversal.
        subsequence_idx : int or None, default None
            Subsequence index (0-based) to validate. If ``None``:

            - With ``do_plot=False``: validate all subsequences.
            - With ``do_plot=True``: auto-select 0 only if
              ``num_sequences == 1``, else raise.

        tr_instance : int or None, default None
            TR instance selector (0-based index or label). If ``None``:

            - With ``do_plot=False``: validate all TRs in selected subsequence(s).
            - With ``do_plot=True``: auto-select 0 only if exactly one TR exists
              in the selected subsequence, else raise.

        grad_atol : float, optional
            Absolute tolerance (mT/m or mT/m per unit) for gradient waveforms.
            If ``None``, uses a sensible default derived from sequence data.
        rf_rms_percent : float, default 10.0
            RMS error threshold as a percentage of RF peak magnitude.

        Returns
        -------
        bool
            ``True`` if validation passed (or completed for ``do_plot=True``);
            ``False`` if any waveform comparison failed or vis mismatches found.

        Raises
        ------
        ValueError
            If ``do_plot=True`` and scope cannot be auto-resolved (e.g.,
            ``subsequence_idx=None`` with ``num_sequences > 1``).
        IndexError
            If specified indices are out of bounds.

        Notes
        -----
        The ``num_averages`` parameter is read from the constructor state
        (``self.num_averages``), not passed at call time. This ensures
        consistency across all analysis methods.

        Validation uses the C-backend's canonical TR selection logic: when
        multiple TRs are present, the first canonical instance (determined by
        shot-ID grouping and amplitude filtering) is selected.

        Examples
        --------
        Validate a single-subsequence collection (default all TRs):

        >>> sc = SequenceCollection('path/to/sequence.seq', num_averages=3)
        >>> result = sc.validate()
        >>> print(f"Validation {'passed' if result else 'failed'}")

        Validate with visual feedback on a specific TR:

        >>> result = sc.validate(do_plot=True, tr_instance=5)

        Validate all subsequences in a multi-subsequence collection:

        >>> sc = SequenceCollection(seqs_list)  # list[Sequence]
        >>> result = sc.validate()  # Iterates all subseqs, all TRs

        See Also
        --------
        check : Run consistency and safety checks (no visual feedback).
        plot : Plot waveforms for a single TR.
        """
        from ._validate import validate as _validate_impl

        return _validate_impl(
            self,
            do_plot=do_plot,
            subsequence_idx=subsequence_idx,
            tr_instance=tr_instance,
            grad_atol=grad_atol,
            rf_rms_percent=rf_rms_percent,
        )

    def __str__(self):
        return str(self._seq)
