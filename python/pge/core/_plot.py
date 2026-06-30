"""TR waveform plotting for SequenceCollection."""

__all__ = ['plot']


import numpy as np

from ._sequence import SequenceCollection
from ._waveforms import ChannelWaveform, get_tr_waveforms

# Matplotlib is imported lazily to avoid hard dependency at import time.


def _wrap_phase(x):
    """Wrap phase to (-pi, pi], snapping values near -pi to +pi."""
    w = (np.asarray(x, dtype=float) + np.pi) % (2 * np.pi) - np.pi
    w[np.isclose(w, -np.pi)] = np.pi
    return w


def _insert_nan_gaps(t, *arrs, gap_factor=5.0):
    """Insert NaN rows where consecutive time samples jump by > gap_factor * median dt.

    Returns (t_out, *arrs_out) with NaN sentinels so matplotlib breaks the line.
    """
    t = np.asarray(t, dtype=float)
    if len(t) < 2:
        return (t, *(np.asarray(a, dtype=float) for a in arrs))
    dt = np.diff(t)
    median_dt = np.median(dt[dt > 0]) if np.any(dt > 0) else 1.0
    gap_idx = np.where(dt > gap_factor * median_dt)[0]
    if len(gap_idx) == 0:
        return (t, *(np.asarray(a, dtype=float) for a in arrs))
    # Build new arrays with NaN inserted after each gap
    n_new = len(t) + len(gap_idx)
    t_out = np.empty(n_new)
    arr_outs = [np.empty(n_new) for _ in arrs]
    src = 0
    dst = 0
    for gi in gap_idx:
        length = gi + 1 - src
        t_out[dst : dst + length] = t[src : src + length]
        for ao, a in zip(arr_outs, arrs, strict=False):
            ao[dst : dst + length] = np.asarray(a, dtype=float)[src : src + length]
        dst += length
        t_out[dst] = np.nan
        for ao in arr_outs:
            ao[dst] = np.nan
        dst += 1
        src = gi + 1
    # Copy remainder
    length = len(t) - src
    t_out[dst : dst + length] = t[src : src + length]
    for ao, a in zip(arr_outs, arrs, strict=False):
        ao[dst : dst + length] = np.asarray(a, dtype=float)[src : src + length]
    return (t_out, *arr_outs)


# ── MATLAB lines-like palette for style parity ───────────────────────
_MATLAB_LINES = [
    '#0072BD',
    '#D95319',
    '#EDB120',
    '#7E2F8E',
    '#77AC30',
    '#4DBEEE',
    '#A2142F',
]

_MAIN_LINEWIDTH = 1.1
_OVERLAY_LINEWIDTH = 1.0

# Physical-unit conversion constants
_GAMMA_DEFAULT = 42.576e6  # Hz/T


def _seg_color(idx: int) -> str:
    """Return a colour for segment index *idx* (cycling)."""
    if idx < 0:
        return '#aaaaaa'  # unmapped
    return _MATLAB_LINES[idx % len(_MATLAB_LINES)]


def _segment_idx_for_time(blocks, t_us: float) -> int:
    """Return segment index for a time sample based on block spans."""
    for blk in blocks:
        blk_start = blk.start_us
        blk_end = blk_start + blk.duration_us
        if (t_us >= blk_start - 0.5) and (t_us <= blk_end + 0.5):
            return blk.segment_idx
    return -1


def _fix_block_segment_indices(wf, source, subsequence_idx):
    """Re-derive segment_idx for every block using segment-order tables.

    The C backend's ``find_segment_for_block_pos`` only matches the first
    occurrence of each segment archetype.  When a TR contains *repeated*
    segments (e.g. MPRAGE: [seg0, seg1, seg2, seg2, seg1]), blocks beyond
    the first occurrence incorrectly get ``segment_idx == -1``.

    This function rebuilds the mapping from the segment ORDER tables and
    per-segment block counts reported by the collection, then patches
    ``wf.blocks[i].segment_idx`` in place.
    """
    rpt = source._subseq_report(subsequence_idx)
    segs = rpt['segments']

    # Valid segment sizes (skip placeholder entries with start_block == -1)
    seg_sizes = {}
    for i, seg in enumerate(segs):
        sb = seg.get('start_block', -1)
        nb = seg.get('num_blocks', 0)
        if sb >= 0 and nb > 0:
            seg_sizes[i] = nb

    # Build one-TR block-position → segment_idx from main_segment_table
    main_table = rpt.get('main_segment_table', [])
    tr_map = []
    for seg_id in main_table:
        tr_map.extend([seg_id] * seg_sizes.get(seg_id, 0))

    if not tr_map:
        return

    # Prep / cooldown segment maps
    prep_map = []
    for seg_id in rpt.get('prep_segment_table', []):
        prep_map.extend([seg_id] * seg_sizes.get(seg_id, 0))
    cool_map = []
    for seg_id in rpt.get('cooldown_segment_table', []):
        cool_map.extend([seg_id] * seg_sizes.get(seg_id, 0))

    num_prep = rpt.get('num_prep_blocks', 0)
    num_cool = rpt.get('num_cooldown_blocks', 0)
    nblk = len(wf.blocks)
    main_blocks = nblk - num_prep - num_cool
    tr_len = len(tr_map)

    for i, blk in enumerate(wf.blocks):
        if i < num_prep:
            # Prep region
            if i < len(prep_map):
                blk.segment_idx = prep_map[i]
            else:
                # Fallback: cycle through main segment map (covers non-degenerate
                # sequences like bSSFP where the whole pass is a single segment
                # and prep/cooldown segment tables are empty)
                blk.segment_idx = (
                    tr_map[i % tr_len] if tr_len > 0 else _find_seg_by_def(segs, i)
                )
        elif i < num_prep + main_blocks:
            # Main imaging region — cycle through tr_map
            pos = (i - num_prep) % tr_len if tr_len > 0 else -1
            blk.segment_idx = tr_map[pos] if 0 <= pos < tr_len else -1
        else:
            # Cooldown region
            cool_pos = i - (num_prep + main_blocks)
            if cool_pos < len(cool_map):
                blk.segment_idx = cool_map[cool_pos]
            else:
                blk.segment_idx = (
                    tr_map[cool_pos % tr_len]
                    if tr_len > 0
                    else _find_seg_by_def(segs, i)
                )


def _find_seg_by_def(segs, block_pos):
    """Fallback: check whether *block_pos* falls within any segment definition."""
    for i, seg in enumerate(segs):
        sb = seg.get('start_block', -1)
        nb = seg.get('num_blocks', 0)
        if sb >= 0 and nb > 0 and sb <= block_pos < sb + nb:
            return i
    return -1


def _slew_rate(ch: ChannelWaveform) -> ChannelWaveform:
    """Compute numerical derivative dA/dt -> T/m/s from mT/m over us."""
    if ch.time_us.size < 2:
        return ChannelWaveform(
            time_us=np.empty(0, dtype=np.float32),
            amplitude=np.empty(0, dtype=np.float32),
        )
    dt = np.diff(ch.time_us) * 1e-3  # us -> ms
    da = np.diff(ch.amplitude)
    dt_safe = np.where(np.abs(dt) < 1e-12, 1e-12, dt)
    slew = da / dt_safe  # mT/m per us = 1e3 T/m/s
    slew_T_per_m_per_s = slew * 1e3
    mid_t = 0.5 * (ch.time_us[:-1] + ch.time_us[1:])
    return ChannelWaveform(
        time_us=mid_t.astype(np.float32),
        amplitude=slew_T_per_m_per_s.astype(np.float32),
    )


class PlotHandle:
    """Lightweight handle returned by :func:`plot` carrying figure + metadata.

    Attributes
    ----------
    fig : matplotlib.figure.Figure
    axes : dict[str, matplotlib.axes.Axes]
    tr_duration_us : float
    num_trs : int
    first_tr_start_us : float
    _tr_start_abs_s : float
        Absolute start time of the displayed TR within the full
        pypulseq sequence (seconds).  Used by Branch 2 overlay to
        compute the correct ``time_range`` for
        ``waveforms_and_times``.
    """

    def __init__(
        self,
        fig,
        axes,
        tr_duration_us,
        num_trs,
        first_tr_start_us,
        *,
        _tr_start_abs_s=0.0,
        _tr_index=0,
        _num_averages=0,
        _subsequence_idx=0,
    ):
        self.fig = fig
        self.axes = axes
        self.tr_duration_us = tr_duration_us
        self.num_trs = num_trs
        self.first_tr_start_us = first_tr_start_us
        self._tr_start_abs_s = _tr_start_abs_s
        self._tr_index = _tr_index
        self._num_averages = _num_averages
        self._subsequence_idx = _subsequence_idx


def plot(
    seq: SequenceCollection,
    *,
    subsequence_idx: int | None = None,
    tr_instance: int | str = 'max_pos',
    collapse_delays: bool = True,
    show_segments: bool = True,
    show_blocks: bool = False,
    show_slew: bool = False,
    show_rf_centers: bool = False,
    show_echoes: bool = False,
    max_grad_mT_per_m: float | None = None,
    max_slew_T_per_m_per_s: float | None = None,
    time_unit: str = 'ms',
    figsize: tuple | None = None,
) -> PlotHandle:
    """Plot native-timing TR waveforms with optional annotations.

    Creates a (3, 2) figure showing RF (magnitude/phase), ADC readout, and
    gradient (Gx/Gy/Gz) waveforms using native microsecond timing from the
    C library with optional annotations (segment colors, block boundaries,
    slew rate, etc.).

    **Canonical TR display**: When ``tr_instance`` is a string (``'max_pos'``
    or ``'zero_var'``), all canonical TRs (one per unique shot-ID pattern) are
    overlaid. The first canonical TR (ID 0) is displayed with full opacity;
    subsequent canonical TRs (ID > 0) are displayed with reduced opacity to
    distinguish multi-shot behavior.

    **Actual instance display**: When ``tr_instance`` is an integer, the specific
    TR instance is plotted in actual amplitude mode.

    Parameters
    ----------
    seq : SequenceCollection
        The sequence to plot.
    subsequence_idx : int or None, default None
        Subsequence index (0-based). If ``None`` and the collection has exactly one
        subsequence, defaults to 0. Otherwise raises ``ValueError``.
    tr_instance : int or str, default 'max_pos'
        TR selector:

        - ``'max_pos'`` (default) — display all canonical TRs (structural
          position maximum across shots); multi-shot patterns overlaid with
          transparency.
        - ``'zero_var'`` — display all canonical TRs (zero-variable gradient view,
          k-space interpretation); multi-shot patterns overlaid with transparency.
        - **int (≥ 0)** — 0-based TR instance index in actual amplitude mode.
        - **int (< 0)** — reverse indexing; ``-1`` selects the last instance.

    collapse_delays : bool, default True
        If ``True``, collapse pure-delay blocks (no RF, grad, ADC) to 0.1 ms
        for visual clarity. Timing information is preserved.
    show_segments : bool, default True
        If ``True``, colour-code gradient waveforms by segment index.
        Segment boundaries are marked.
    show_blocks : bool, default False
        If ``True``, draw vertical dotted lines at block boundaries.
    show_slew : bool, default False
        If ``True``, overlay numerical slew-rate (dG/dt in T/m/s) on
        each gradient panel.
    show_rf_centers : bool, default False
        If ``True``, mark RF pulse envelope peaks with vertical markers.
    show_echoes : bool, default False
        If ``True``, shade ADC sampling windows.
    max_grad_mT_per_m : float, optional
        Gradient axis upper limit (mT/m). If ``None``, auto-scaled.
    max_slew_T_per_m_per_s : float, optional
        Slew-rate axis upper limit (T/m/s, only if ``show_slew=True``).
        If ``None``, auto-scaled.
    time_unit : str, default 'ms'
        Time axis unit: ``'ms'`` (milliseconds) or ``'us'`` (microseconds).
    figsize : tuple, optional
        Figure size ``(width, height)`` in inches. Defaults to sensible layout
        (typically 14 x 8 inches).

    Raises
    ------
    ValueError
        If ``subsequence_idx=None`` and the collection has multiple subsequences.
        If ``tr_instance`` (integer) is out of range for the selected subsequence.

    Notes
    -----
    Native timing preserves ADC sampling patterns and reveals aliasing artifacts
    that uniform resampling would mask.

    Colour palette for segments uses MATLAB line colors (7-color cycle);
    prep/cooldown regions are shown in gray.

    Multi-canonical-TR display (when multiple shot-ID patterns exist):
    - Canonical TR 0: solid, full opacity
    - Canonical TR > 0: reduced opacity (alpha ≈ 0.5) or dashed line style
      to visually distinguish shots

    Examples
    --------
    Plot canonical TR waveforms (default, showing all shot patterns):

    >>> from python.pge.core import SequenceCollection
    >>> sc = SequenceCollection('path/to/sequence.seq')
    >>> sc.plot(subsequence_idx=0)  # 'max_pos' by default

    Plot with segment coloring and slew-rate overlay:

    >>> sc.plot(subsequence_idx=0, show_segments=True, show_slew=True)

    Plot a specific TR instance in actual amplitude mode:

    >>> sc.plot(subsequence_idx=0, tr_instance=5)  # 5th actual instance

    Use zero-variable gradient view:

    >>> sc.plot(subsequence_idx=0, tr_instance='zero_var')

    See Also
    --------
    SequenceCollection.plot : Wrapper method (preferred interface).
    get_tr_waveforms : Extract native-timing waveforms programmatically.
    """
    import matplotlib.pyplot as plt

    source = seq

    # plot() only accepts SequenceCollection.  Validation overlays belong in validate().
    if not isinstance(source, SequenceCollection):
        raise TypeError(
            f'plot() expects a SequenceCollection; got {type(source).__name__}. '
            'For reference overlays use validate(do_plot=True).'
        )

    # Auto-resolve subsequence_idx: None -> 0 if only one subsequence
    if subsequence_idx is None:
        if len(source._seqs) == 1:
            subsequence_idx = 0
        else:
            raise ValueError(
                f'subsequence_idx=None is only valid for single subsequence; '
                f'got {len(source._seqs)} subsequences. '
                f'Please specify subsequence_idx explicitly (0..{len(source._seqs) - 1}).'
            )

    # Validate subsequence_idx range
    if not 0 <= subsequence_idx < len(source._seqs):
        raise ValueError(
            f'subsequence_idx={subsequence_idx} out of range for {len(source._seqs)} subsequences '
            f'(valid: 0..{len(source._seqs) - 1})'
        )

    # Parse tr_instance: can be string ('max_pos', 'zero_var') or integer (actual instance)
    amplitude_mode = None
    tr_index = 0
    if isinstance(tr_instance, str):
        valid_modes = ('max_pos', 'zero_var')
        if tr_instance not in valid_modes:
            raise ValueError(
                f'tr_instance={tr_instance!r} invalid. '
                f'Valid modes: {valid_modes}. '
                f'Use integer >= 0 for specific TR instance.'
            )
        amplitude_mode = tr_instance
    else:
        # Integer mode: specific TR instance
        if not isinstance(tr_instance, int):
            raise TypeError(
                f'tr_instance must be int or str, got {type(tr_instance).__name__}'
            )
        # Will determine num_trs below; store for now
        from ._extension._pulseqlib_wrapper import _find_tr

        tr_info = _find_tr(source._cseq, subsequence_idx=subsequence_idx)
        from ._validate import _total_addressable_trs

        num_trs = _total_addressable_trs(tr_info)
        idx = int(tr_instance)
        if num_trs is not None:
            if idx < 0:
                idx = num_trs + idx
            if idx < 0 or idx >= num_trs:
                raise ValueError(
                    f'tr_instance={tr_instance} out of range for {num_trs} TRs '
                    f'(valid: 0..{num_trs - 1} or -{num_trs}..-1)'
                )
        amplitude_mode = 'actual'
        tr_index = idx

    # ── Time helpers ──
    t_scale = 1e-3 if time_unit == 'ms' else 1.0
    t_label = 'Time (ms)' if time_unit == 'ms' else 'Time (us)'

    # ── Translate public amplitude_mode names to internal ones ──
    _mode_map = {'max_pos': 'max_pos', 'zero_var': 'zero_var', 'actual': 'actual'}
    internal_amplitude_mode = _mode_map[amplitude_mode]

    # ── Fetch waveforms ──
    from ._extension._pulseqlib_wrapper import _find_tr

    tr_info = _find_tr(source._cseq, subsequence_idx=subsequence_idx)
    num_canonical = tr_info.get('num_canonical_trs', 1)

    wf = get_tr_waveforms(
        source,
        subsequence_idx=subsequence_idx,
        amplitude_mode=internal_amplitude_mode,
        tr_index=tr_index,
        collapse_delays=collapse_delays,
    )

    # Fetch additional canonical TRs for overlay (max_pos / zero_var only)
    extra_wfs = []
    if amplitude_mode in ('max_pos', 'zero_var') and num_canonical > 1:
        for ci in range(num_canonical):
            if ci == tr_index:
                continue
            extra = get_tr_waveforms(
                source,
                subsequence_idx=subsequence_idx,
                amplitude_mode=internal_amplitude_mode,
                tr_index=ci,
                collapse_delays=collapse_delays,
            )
            extra_wfs.append(extra)

    # Fix segment_idx: C's find_segment_for_block_pos misses repeated segments.
    _fix_block_segment_indices(wf, source, subsequence_idx)

    tr_dur = tr_info['tr_duration_us']
    from ._validate import _total_addressable_trs

    num_trs = _total_addressable_trs(tr_info)
    # Compute actual start time of first TR from waveform blocks
    first_tr_start_us = 0.0
    for blk in wf.blocks:
        if blk.segment_idx >= 0:
            first_tr_start_us = blk.start_us
            break

    # Absolute start time of displayed TR within the pypulseq seq
    from ._validate import _abs_block_start_s

    _abs_s = _abs_block_start_s(
        source._seqs[subsequence_idx], tr_index * tr_info['tr_size']
    )

    if figsize is None:
        figsize = (14, 8)

    fig_obj, axes_grid = plt.subplots(3, 2, figsize=figsize, sharex=True)
    fig_obj.patch.set_facecolor('white')
    axes = {
        'rf_mag': axes_grid[0, 0],
        'rf_phase': axes_grid[1, 0],
        'adc': axes_grid[2, 0],
        'gx': axes_grid[0, 1],
        'gy': axes_grid[1, 1],
        'gz': axes_grid[2, 1],
    }

    def _t(t_us):
        return np.asarray(t_us) * t_scale

    # ── Segment background shading (all axes) ──
    if show_segments and len(wf.blocks) > 0:
        for blk in wf.blocks:
            t0_blk = _t(np.array([blk.start_us]))[0]
            t1_blk = _t(np.array([blk.start_us + blk.duration_us]))[0]
            color = _seg_color(blk.segment_idx)
            for ax_item in axes.values():
                ax_item.axvspan(t0_blk, t1_blk, color=color, alpha=0.07, linewidth=0)

    # ── RF magnitude ──
    ax = axes['rf_mag']
    ch = wf.rf_mag
    if ch.time_us.size > 0:
        if show_segments and len(wf.blocks) > 0:
            _plot_segmented(ax, _t, ch, wf.blocks, linewidth=_MAIN_LINEWIDTH, alpha=1.0)
        else:
            t_rf, a_rf = _insert_nan_gaps(ch.time_us, ch.amplitude)
            ax.plot(
                _t(t_rf),
                a_rf,
                color='gray',
                linewidth=_MAIN_LINEWIDTH,
                linestyle='-',
                label='pulserver',
            )
    ax.set_ylabel('|RF| (uT)')
    # Per-channel RF magnitude overlay (pTx)
    if wf.rf_mag_channels:
        for cidx, ch_c in enumerate(wf.rf_mag_channels):
            if ch_c.time_us.size > 0:
                ax.plot(
                    _t(ch_c.time_us),
                    ch_c.amplitude,
                    color=_MATLAB_LINES[cidx % len(_MATLAB_LINES)],
                    linewidth=_MAIN_LINEWIDTH,
                    linestyle='-',
                    alpha=0.75,
                    label=f'ch{cidx}',
                )

    # ── RF phase ──
    ax = axes['rf_phase']
    ch = wf.rf_phase
    # Plot RF phase directly from C backend (already includes all offsets)
    if ch.time_us.size > 0:
        # Wrap phase for display, but keep original timing for segment lookup
        phase_ch = ChannelWaveform(
            time_us=ch.time_us,
            amplitude=_wrap_phase(ch.amplitude).astype(np.float32),
        )
        if show_segments and len(wf.blocks) > 0:
            _plot_segmented(
                ax, _t, phase_ch, wf.blocks, linewidth=_MAIN_LINEWIDTH, alpha=1.0
            )
        else:
            t_rf_ph, a_rf_ph = _insert_nan_gaps(phase_ch.time_us, phase_ch.amplitude)
            ax.plot(
                _t(t_rf_ph),
                a_rf_ph,
                color='gray',
                linewidth=_MAIN_LINEWIDTH,
                linestyle='-',
                label='pulserver',
            )
    ax.set_ylabel('RF phase (rad)')
    ax.set_yticks([-np.pi, 0, np.pi])
    ax.set_yticklabels(['-pi', '0', 'pi'])
    # Per-channel RF phase overlay (pTx)
    if wf.rf_phase_channels:
        for cidx, ch_c in enumerate(wf.rf_phase_channels):
            if ch_c.time_us.size > 0:
                ax.plot(
                    _t(ch_c.time_us),
                    _wrap_phase(ch_c.amplitude),
                    color=_MATLAB_LINES[cidx % len(_MATLAB_LINES)],
                    linewidth=_MAIN_LINEWIDTH,
                    linestyle='-',
                    alpha=0.75,
                    label=f'ch{cidx}',
                )

    # ── ADC phase envelope ──
    ax = axes['adc']
    # Only plot for real ADC events (not dummy TRs)
    adc_labeled = False
    for adc in wf.adc_events:
        if adc.num_samples > 0 and adc.duration_us > 0:
            dwell_time = adc.duration_us / adc.num_samples  # us
            t_adc = adc.onset_us + np.arange(adc.num_samples) * dwell_time
            t_adc_s = t_adc * 1e-6
            adc_phase = adc.phase_offset_rad + 2 * np.pi * adc.freq_offset_hz * t_adc_s
            # Wrap phase to [-pi, pi]
            adc_phase_wrapped = (adc_phase + np.pi) % (2 * np.pi) - np.pi
            # Determine segment color for this ADC event
            if show_segments and len(wf.blocks) > 0:
                adc_seg = _segment_idx_for_time(wf.blocks, adc.onset_us)
                adc_color = _seg_color(adc_seg)
            else:
                adc_color = 'purple'
            lbl = 'pulserver' if not adc_labeled else None
            ax.plot(
                _t(t_adc),
                adc_phase_wrapped,
                color=adc_color,
                linewidth=_MAIN_LINEWIDTH,
                label=lbl,
            )
            # Shadowed region
            ax.fill_between(
                _t(t_adc),
                0,
                np.maximum(adc_phase_wrapped, 1.0),
                color=adc_color,
                alpha=0.15,
            )
            adc_labeled = True
    ax.set_ylabel('ADC phase (rad)')
    ax.set_yticks([-np.pi, 0, np.pi])
    ax.set_yticklabels(['-pi', '0', 'pi'])

    # ── Gradients (Gx, Gy, Gz) ──
    for gname in ('gx', 'gy', 'gz'):
        ax = axes[gname]
        ch = getattr(wf, gname)
        if ch.time_us.size > 0:
            if show_segments and len(wf.blocks) > 0:
                _plot_segmented(
                    ax, _t, ch, wf.blocks, linewidth=_MAIN_LINEWIDTH, alpha=1.0
                )
            else:
                ax.plot(
                    _t(ch.time_us),
                    ch.amplitude,
                    color='gray',
                    linewidth=_MAIN_LINEWIDTH,
                    linestyle='-',
                    label='pulserver',
                )

        ax.set_ylabel(f'{gname.upper()} (mT/m)')

        if max_grad_mT_per_m is not None:
            ax.axhline(max_grad_mT_per_m, color='gray', ls='--', lw=0.6, alpha=0.6)
            ax.axhline(-max_grad_mT_per_m, color='gray', ls='--', lw=0.6, alpha=0.6)

        # Slew rate overlay
        if show_slew and ch.time_us.size > 1:
            slew = _slew_rate(ch)
            ax2 = ax.twinx()
            ax2.plot(
                _t(slew.time_us),
                slew.amplitude,
                color='orange',
                linewidth=0.5,
                alpha=0.5,
            )
            ax2.set_ylabel('Slew (T/m/s)', color='orange', fontsize=8)
            ax2.tick_params(axis='y', labelcolor='orange', labelsize=7)
            if max_slew_T_per_m_per_s is not None:
                ax2.axhline(
                    max_slew_T_per_m_per_s,
                    color='orange',
                    ls=':',
                    lw=0.5,
                    alpha=0.5,
                )
                ax2.axhline(
                    -max_slew_T_per_m_per_s,
                    color='orange',
                    ls=':',
                    lw=0.5,
                    alpha=0.5,
                )

    # ── Overlay additional canonical TRs (dashed, reduced alpha) ──
    _OVERLAY_ALPHA = 0.45
    _OVERLAY_LW = _MAIN_LINEWIDTH * 0.8
    for extra in extra_wfs:
        # RF magnitude
        ch = extra.rf_mag
        if ch.time_us.size > 0:
            t_rf, a_rf = _insert_nan_gaps(ch.time_us, ch.amplitude)
            axes['rf_mag'].plot(
                _t(t_rf),
                a_rf,
                color='gray',
                linewidth=_OVERLAY_LW,
                linestyle='--',
                alpha=_OVERLAY_ALPHA,
            )
        # RF phase
        ch = extra.rf_phase
        if ch.time_us.size > 0:
            t_rf_ph, a_rf_ph = _insert_nan_gaps(ch.time_us, _wrap_phase(ch.amplitude))
            axes['rf_phase'].plot(
                _t(t_rf_ph),
                a_rf_ph,
                color='gray',
                linewidth=_OVERLAY_LW,
                linestyle='--',
                alpha=_OVERLAY_ALPHA,
            )
        # ADC
        for adc in extra.adc_events:
            if adc.num_samples > 0 and adc.duration_us > 0:
                dwell = adc.duration_us / adc.num_samples
                t_adc = adc.onset_us + np.arange(adc.num_samples) * dwell
                t_adc_s = t_adc * 1e-6
                adc_phase = (
                    adc.phase_offset_rad + 2 * np.pi * adc.freq_offset_hz * t_adc_s
                )
                adc_phase_wrapped = (adc_phase + np.pi) % (2 * np.pi) - np.pi
                axes['adc'].plot(
                    _t(t_adc),
                    adc_phase_wrapped,
                    color='gray',
                    linewidth=_OVERLAY_LW,
                    linestyle='--',
                    alpha=_OVERLAY_ALPHA,
                )
        # Gradients
        for gname in ('gx', 'gy', 'gz'):
            ch = getattr(extra, gname)
            if ch.time_us.size > 0:
                axes[gname].plot(
                    _t(ch.time_us),
                    ch.amplitude,
                    color='gray',
                    linewidth=_OVERLAY_LW,
                    linestyle='--',
                    alpha=_OVERLAY_ALPHA,
                )

    # ── Block boundaries ──
    if show_blocks and len(wf.blocks) > 0:
        for blk in wf.blocks:
            t_start = _t(np.array([blk.start_us]))[0]
            for ax_item in axes.values():
                ax_item.axvline(t_start, color='k', ls=':', lw=1.5)
        # Also mark end of the last block
        last = wf.blocks[-1]
        t_end = _t(np.array([last.start_us + last.duration_us]))[0]
        for ax_item in axes.values():
            ax_item.axvline(t_end, color='k', ls=':', lw=1.5)

    # ── RF centres ──
    if show_rf_centers and wf.rf_mag.time_us.size > 0:
        _plot_rf_centers(axes['rf_mag'], wf, _t)

    # ── Echo markers ──
    if show_echoes:
        _plot_echo_markers(axes['adc'], wf, _t)

    # ── Finalise ──
    axes['adc'].set_xlabel(t_label)
    axes['gz'].set_xlabel(t_label)
    for ax_item in axes.values():
        ax_item.grid(True)

    # ── Segment legend (single figure-level, below all axes) ──
    if show_segments and len(wf.blocks) > 0:
        from matplotlib.patches import Patch

        seen_segs = sorted(
            {blk.segment_idx for blk in wf.blocks if blk.segment_idx >= 0}
        )
        legend_handles = []
        for si in seen_segs:
            legend_handles.append(
                Patch(facecolor=_seg_color(si), label=f'Segment {si}')
            )
        if legend_handles:
            fig_obj.legend(
                handles=legend_handles,
                loc='lower center',
                ncol=len(legend_handles),
                fontsize=8,
                framealpha=0.8,
                bbox_to_anchor=(0.5, 0.0),
            )
            fig_obj.tight_layout(rect=[0, 0.04, 1, 1])
        else:
            fig_obj.tight_layout()
    else:
        fig_obj.tight_layout()

    return PlotHandle(
        fig=fig_obj,
        axes=axes,
        tr_duration_us=tr_dur,
        num_trs=num_trs if num_trs is not None else 1,
        first_tr_start_us=first_tr_start_us,
        _tr_start_abs_s=_abs_s,
        _tr_index=tr_index,
        _subsequence_idx=subsequence_idx,
    )


def _overlay_xml(handle, xml_path, *, t_scale, label):
    """Overlay waveforms from an XML file onto an existing plot."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    # G/cm -> mT/m: 1 G/cm = 10 mT/m
    # G -> uT: 1 G = 100 uT
    g_per_cm_to_mT_per_m = 10.0
    g_to_uT = 100.0

    alpha = 0.7

    for ch_name in ('gx', 'gy', 'gz'):
        elem = root.find(f'.//{ch_name}')
        if elem is not None and elem.text:
            data = np.fromstring(elem.text, sep=' ')
            if data.size >= 2:
                n = data.size // 2
                t_us = data[:n]
                amp = data[n:] * g_per_cm_to_mT_per_m
                handle.axes[ch_name].plot(
                    t_us * t_scale,
                    amp,
                    linewidth=_OVERLAY_LINEWIDTH,
                    alpha=alpha,
                    label=label,
                )

    rf_elem = root.find('.//rf')
    if rf_elem is not None and rf_elem.text:
        data = np.fromstring(rf_elem.text, sep=' ')
        if data.size >= 2:
            n = data.size // 2
            t_us = data[:n]
            amp = data[n:] * g_to_uT
            handle.axes['rf_mag'].plot(
                t_us * t_scale,
                np.abs(amp),
                linewidth=_OVERLAY_LINEWIDTH,
                alpha=alpha,
                label=label,
            )

    if label:
        handle.axes['rf_mag'].legend(fontsize=8, loc='upper right')


def _new_empty_plot_handle(*, figsize=None):
    """Create an empty plot handle suitable for direct XML overlays."""
    import matplotlib.pyplot as plt

    if figsize is None:
        figsize = (14, 8)

    fig_obj, axes_grid = plt.subplots(3, 2, figsize=figsize, sharex=True)
    fig_obj.patch.set_facecolor('white')
    axes = {
        'rf_mag': axes_grid[0, 0],
        'rf_phase': axes_grid[1, 0],
        'adc': axes_grid[2, 0],
        'gx': axes_grid[0, 1],
        'gy': axes_grid[1, 1],
        'gz': axes_grid[2, 1],
    }

    axes['rf_mag'].set_ylabel('|RF| (uT)')
    axes['rf_phase'].set_ylabel('RF phase (rad)')
    axes['rf_phase'].set_yticks([-np.pi, 0, np.pi])
    axes['rf_phase'].set_yticklabels(['-pi', '0', 'pi'])
    axes['adc'].set_ylabel('ADC')
    axes['adc'].set_ylim(0, 1)
    axes['adc'].set_yticks([])
    axes['gx'].set_ylabel('GX (mT/m)')
    axes['gy'].set_ylabel('GY (mT/m)')
    axes['gz'].set_ylabel('GZ (mT/m)')
    axes['adc'].set_xlabel('Time (ms)')
    axes['gz'].set_xlabel('Time (ms)')
    for ax in axes.values():
        ax.grid(True)

    fig_obj.tight_layout()
    return PlotHandle(
        fig=fig_obj,
        axes=axes,
        tr_duration_us=0.0,
        num_trs=1,
        first_tr_start_us=0.0,
        _tr_start_abs_s=0.0,
    )


def _plot_rf_centers(ax, wf, t_fn):
    """Plot RF iso-centre markers from C library block descriptors."""
    if wf.rf_mag.time_us.size == 0:
        return
    from matplotlib.lines import Line2D

    a = np.abs(wf.rf_mag.amplitude)
    plotted = False
    for blk in wf.blocks:
        if blk.rf_isocenter_us < 0:
            continue
        center_display = t_fn(np.array([blk.rf_isocenter_us]))[0]
        # Find RF peak amplitude within this block for marker placement
        mask = (wf.rf_mag.time_us >= blk.start_us - 0.5) & (
            wf.rf_mag.time_us <= blk.start_us + blk.duration_us + 0.5
        )
        peak = np.max(a[mask]) if np.any(mask) else 0.0
        ax.axvline(center_display, color='r', ls='--', lw=1.0, alpha=0.7)
        ax.plot(center_display, peak * 1.05, 'rv', markersize=6, alpha=0.7)
        plotted = True
    if plotted:
        proxy = Line2D(
            [0],
            [0],
            color='r',
            ls='--',
            lw=1.0,
            marker='v',
            markersize=6,
            alpha=0.7,
            label='RF isocenter',
        )
        ax.legend(handles=[proxy], loc='upper right', fontsize=7, framealpha=0.7)


def _plot_echo_markers(ax, wf, t_fn):
    """Plot echo (ADC k=0) markers from C library block descriptors."""
    from matplotlib.lines import Line2D

    plotted = False
    for blk in wf.blocks:
        if blk.adc_kzero_us < 0:
            continue
        kzero_display = t_fn(np.array([blk.adc_kzero_us]))[0]
        ax.axvline(kzero_display, color='b', ls='--', lw=1.0, alpha=0.7)
        ax.plot(kzero_display, 0.5, 'b^', markersize=6, alpha=0.7)
        plotted = True
    if plotted:
        proxy = Line2D(
            [0],
            [0],
            color='b',
            ls='--',
            lw=1.0,
            marker='^',
            markersize=6,
            alpha=0.7,
            label='k=0 (echo)',
        )
        ax.legend(handles=[proxy], loc='upper right', fontsize=7, framealpha=0.7)


def _plot_segmented(ax, t_fn, ch, blocks, **kwargs):
    """Plot a channel waveform with colour-coded segments."""
    t = ch.time_us
    a = ch.amplitude

    # Build per-sample segment index
    seg_idx = np.full(len(t), -1, dtype=int)
    for blk in blocks:
        blk_start = blk.start_us
        blk_end = blk_start + blk.duration_us
        mask = (t >= blk_start - 0.5) & (t <= blk_end + 0.5)
        seg_idx[mask] = blk.segment_idx

    if len(t) == 0:
        return
    first = True
    run_start = 0
    for i in range(1, len(t)):
        if seg_idx[i] != seg_idx[run_start]:
            # Extend run by one sample so the boundary is visually connected
            end = min(i + 1, len(t))
            kw = dict(**kwargs)
            if first:
                kw['label'] = 'pulserver'
                first = False
            ax.plot(
                t_fn(t[run_start:end]),
                a[run_start:end],
                color=_seg_color(seg_idx[run_start]),
                **kw,
            )
            run_start = i
    kw = dict(**kwargs)
    if first:
        kw['label'] = 'pulserver'
    ax.plot(
        t_fn(t[run_start:]),
        a[run_start:],
        color=_seg_color(seg_idx[run_start]),
        **kw,
    )
