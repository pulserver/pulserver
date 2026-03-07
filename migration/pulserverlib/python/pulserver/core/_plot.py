"""TR waveform plotting for SequenceCollection."""

__all__ = ['plot']

from typing import Sequence as SequenceType, Union
from pathlib import Path

import numpy as np

from ._waveforms import TrWaveforms, ChannelWaveform, get_tr_waveforms
from ._sequence import SequenceCollection

# Matplotlib is imported lazily to avoid hard dependency at import time.

# ── Colour palette for segments ──────────────────────────────────────
_SEG_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
    '#bcbd22', '#17becf',
]

# Physical-unit conversion constants
_GAMMA_DEFAULT = 42.576e6  # Hz/T


def _seg_color(idx: int) -> str:
    """Return a colour for segment index *idx* (cycling)."""
    if idx < 0:
        return '#aaaaaa'  # prep/cooldown
    return _SEG_COLORS[idx % len(_SEG_COLORS)]


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

    def __init__(self, fig, axes, tr_duration_us, num_trs,
                 first_tr_start_us, *, _tr_start_abs_s=0.0):
        self.fig = fig
        self.axes = axes
        self.tr_duration_us = tr_duration_us
        self.num_trs = num_trs
        self.first_tr_start_us = first_tr_start_us
        self._tr_start_abs_s = _tr_start_abs_s


def plot(
    source,
    *,
    subsequence_idx: int = 0,
    tr_idx=0,
    hide_prep: bool = True,
    hide_cooldown: bool = True,
    collapse_delays: bool = True,
    show_segments: bool = True,
    show_blocks: bool = False,
    show_slew: bool = False,
    show_rf_centers: bool = False,
    show_echoes: bool = False,
    max_grad_mT_per_m=None,
    max_slew_T_per_m_per_s=None,
    time_unit: str = 'ms',
    figsize=None,
    fig=None,
    label=None,
):
    """Plot native-timing TR waveforms.

    Layout is (3, 2): first column has RF magnitude (top), RF phase
    (middle), ADC mask (bottom); second column has Gx, Gy, Gz.

    Parameters
    ----------
    source : SequenceCollection or pp.Sequence or str/Path
        Primary data source.  A :class:`SequenceCollection` creates a
        fresh figure.  A pypulseq ``Sequence`` or XML file path
        overlays onto an existing figure (``fig`` must be provided).
    subsequence_idx : int
        Subsequence index (default 0).
    tr_idx : int or {'max_pos', 'zero_var'}
        TR index (0-based) or amplitude-mode string.
    hide_prep : bool
        Hide preparation blocks (default True).
    hide_cooldown : bool
        Hide cooldown blocks (default True).
    collapse_delays : bool
        Shrink pure-delay blocks to 0.1 ms at C level (default True).
    show_segments : bool
        Colour-code gradient waveforms by segment index.
    show_blocks : bool
        Draw vertical dotted lines at block boundaries (default False).
    show_slew : bool
        Overlay slew rate on gradient panels.
    show_rf_centers : bool
        Mark RF iso-centres on the RF magnitude subplot (default False).
    show_echoes : bool
        Mark echo (ADC centre) on the ADC panel (default False).
    max_grad_mT_per_m : float or None
        Horizontal reference line for max gradient amplitude.
    max_slew_T_per_m_per_s : float or None
        Horizontal reference for max slew rate (only if show_slew).
    time_unit : str
        ``'ms'`` (default) or ``'us'``.
    figsize : tuple or None
        Figure size.  Default ``(14, 8)``.
    fig : PlotHandle or None
        Existing plot handle for overlay.  **Required** when *source*
        is a pypulseq Sequence or XML path; **forbidden** when
        *source* is a SequenceCollection.
    label : str or None
        Legend label for overlay traces.

    Returns
    -------
    PlotHandle
        Handle containing figure, axes, and TR metadata.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    is_collection = isinstance(source, SequenceCollection)

    # Lazy check for pypulseq
    try:
        import pypulseq as pp
        is_pulseq = isinstance(source, pp.Sequence)
    except ImportError:
        is_pulseq = False

    is_xml = isinstance(source, (str, Path))

    # ── Validate fig argument ──
    if is_collection and fig is not None:
        raise ValueError(
            "Cannot pass 'fig' handle when source is a SequenceCollection. "
            "Overlay is only supported for pypulseq Sequence or XML sources."
        )
    if (is_pulseq or is_xml) and fig is None:
        raise ValueError(
            "Must pass 'fig' handle when overlaying a pypulseq Sequence "
            "or XML file.  Create a base plot from a SequenceCollection first."
        )

    # ── Determine amplitude mode and tr_index ──
    if isinstance(tr_idx, str):
        amplitude_mode = tr_idx  # 'max_pos' or 'zero_var'
        tr_index = 0
    else:
        amplitude_mode = 'actual' if tr_idx > 0 else 'max_pos'
        tr_index = int(tr_idx)

    # ── Time helpers ──
    t_scale = 1e-3 if time_unit == 'ms' else 1.0
    t_label = 'Time (ms)' if time_unit == 'ms' else 'Time (us)'

    # ================================================================
    #  Branch 1: SequenceCollection -> fresh figure
    # ================================================================
    if is_collection:
        include_prep = not hide_prep
        include_cooldown = not hide_cooldown

        wf = get_tr_waveforms(
            source,
            subsequence_idx=subsequence_idx,
            amplitude_mode=amplitude_mode,
            tr_index=tr_index,
            include_prep=include_prep,
            include_cooldown=include_cooldown,
            collapse_delays=collapse_delays,
        )

        # TR metadata from C library
        from ._extension._pulseqlib_wrapper import _find_tr
        tr_info = _find_tr(source._cseq, subsequence_idx=subsequence_idx)
        tr_dur = tr_info['tr_duration_us']
        num_trs = tr_info['num_trs']
        # Compute actual start time of first TR from waveform blocks
        first_tr_start_us = 0.0
        for blk in wf.blocks:
            if blk.segment_idx >= 0:
                first_tr_start_us = blk.start_us
                break

        # Absolute start time of displayed TR within the pypulseq seq
        # (needed for Branch 2 overlay alignment).
        from ._validate import _abs_tr_start_s
        _abs_s = _abs_tr_start_s(
            source._seqs[subsequence_idx],
            tr_info['num_prep_blocks'], tr_index,
            tr_info['tr_duration_us'],
        )

        if figsize is None:
            figsize = (14, 8)

        fig_obj, axes_grid = plt.subplots(3, 2, figsize=figsize, sharex=True)
        axes = {
            'rf_mag':   axes_grid[0, 0],
            'rf_phase': axes_grid[1, 0],
            'adc':      axes_grid[2, 0],
            'gx':       axes_grid[0, 1],
            'gy':       axes_grid[1, 1],
            'gz':       axes_grid[2, 1],
        }

        def _t(t_us):
            return t_us * t_scale

        # ── RF magnitude ──
        ax = axes['rf_mag']
        ch = wf.rf_mag
        if ch.time_us.size > 0:
            ax.plot(_t(ch.time_us), ch.amplitude,
                    color='k', linewidth=0.8)
        ax.set_ylabel('|RF| (uT)')

        # ── RF phase ──
        ax = axes['rf_phase']
        ch = wf.rf_phase
        if ch.time_us.size > 0:
            ax.plot(_t(ch.time_us), ch.amplitude,
                    color='k', linewidth=0.8)
        ax.set_ylabel('RF phase (rad)')
        ax.set_yticks([-np.pi, 0, np.pi])
        ax.set_yticklabels(['-pi', '0', 'pi'])

        # ── ADC ──
        ax = axes['adc']
        for adc in wf.adc_events:
            t_start = _t(np.array([adc.onset_us]))[0]
            t_end = _t(np.array([adc.onset_us + adc.duration_us]))[0]
            rect = Rectangle(
                (t_start, 0.1), t_end - t_start, 0.8,
                facecolor='#ff7f0e', alpha=0.5, edgecolor='k',
                linewidth=0.5,
            )
            ax.add_patch(rect)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel('ADC')

        # ── Gradients (Gx, Gy, Gz) ──
        axis_color = {'gx': '#1f77b4', 'gy': '#2ca02c', 'gz': '#d62728'}
        for gname in ('gx', 'gy', 'gz'):
            ax = axes[gname]
            ch = getattr(wf, gname)
            if ch.time_us.size > 0:
                if show_segments and len(wf.blocks) > 0:
                    _plot_segmented(ax, _t, ch, wf.blocks,
                                    linewidth=0.8, alpha=1.0)
                else:
                    ax.plot(_t(ch.time_us), ch.amplitude,
                            color=axis_color[gname], linewidth=0.8)

            ax.set_ylabel(f'{gname.upper()} (mT/m)')

            if max_grad_mT_per_m is not None:
                ax.axhline(max_grad_mT_per_m, color='gray',
                           ls='--', lw=0.6, alpha=0.6)
                ax.axhline(-max_grad_mT_per_m, color='gray',
                           ls='--', lw=0.6, alpha=0.6)

            # Slew rate overlay
            if show_slew and ch.time_us.size > 1:
                slew = _slew_rate(ch)
                ax2 = ax.twinx()
                ax2.plot(_t(slew.time_us), slew.amplitude,
                         color='orange', linewidth=0.5, alpha=0.5)
                ax2.set_ylabel('Slew (T/m/s)', color='orange', fontsize=8)
                ax2.tick_params(axis='y', labelcolor='orange', labelsize=7)
                if max_slew_T_per_m_per_s is not None:
                    ax2.axhline(max_slew_T_per_m_per_s, color='orange',
                                ls=':', lw=0.5, alpha=0.5)
                    ax2.axhline(-max_slew_T_per_m_per_s, color='orange',
                                ls=':', lw=0.5, alpha=0.5)

        # ── Block boundaries ──
        if show_blocks and len(wf.blocks) > 0:
            for blk in wf.blocks:
                t_start = _t(np.array([blk.start_us]))[0]
                for ax_item in axes.values():
                    ax_item.axvline(t_start, color='k', ls=':', lw=0.3, alpha=0.3)

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
            ax_item.grid(True, alpha=0.2)

        fig_obj.tight_layout()

        return PlotHandle(
            fig=fig_obj, axes=axes,
            tr_duration_us=tr_dur,
            num_trs=num_trs,
            first_tr_start_us=first_tr_start_us,
            _tr_start_abs_s=_abs_s,
        )

    # ================================================================
    #  Branch 2: pypulseq Sequence overlay
    # ================================================================
    if is_pulseq:
        assert fig is not None
        seq = source
        gamma = seq.system.gamma if hasattr(seq.system, 'gamma') else _GAMMA_DEFAULT

        # Absolute time range of the displayed TR in the pypulseq seq.
        tr_dur_s = fig.tr_duration_us * 1e-6
        abs_t0 = fig._tr_start_abs_s
        abs_t1 = abs_t0 + tr_dur_s

        # waveforms_and_times returns
        #   (wave_data, tfp_exc, tfp_ref, t_adc, fp_adc)
        # wave_data is a list of (2, N) ndarrays per channel.
        result = seq.waveforms_and_times(append_RF=True,
                                          time_range=[abs_t0, abs_t1])
        channels = result[0]  # list of (2, N) ndarrays

        hz_per_m_to_mT_per_m = 1.0 / (gamma * 1e-3)
        hz_to_uT = 1e6 / gamma

        channel_map = {
            'gx': (0, hz_per_m_to_mT_per_m),
            'gy': (1, hz_per_m_to_mT_per_m),
            'gz': (2, hz_per_m_to_mT_per_m),
        }

        alpha = 0.7
        for ch_name, (ch_idx, scale) in channel_map.items():
            if ch_idx < len(channels):
                arr = channels[ch_idx]
                if arr.shape[1] > 0:
                    t_us = (arr[0] - abs_t0) * 1e6  # relative µs
                    t_us += fig.first_tr_start_us    # align with C base
                    fig.axes[ch_name].plot(
                        t_us * t_scale, arr[1] * scale,
                        linewidth=0.7, alpha=alpha, label=label)

        # RF magnitude & phase
        if len(channels) > 3:
            arr = channels[3]
            if arr.shape[1] > 0:
                t_us = (arr[0] - abs_t0) * 1e6 + fig.first_tr_start_us
                rf_mag = np.abs(arr[1]) * hz_to_uT
                rf_phase = np.angle(arr[1])
                fig.axes['rf_mag'].plot(
                    t_us * t_scale, rf_mag,
                    linewidth=0.7, alpha=alpha, label=label)
                fig.axes['rf_phase'].plot(
                    t_us * t_scale, rf_phase,
                    linewidth=0.7, alpha=alpha, label=label)

        if label:
            fig.axes['rf_mag'].legend(fontsize=8, loc='upper right')

        return fig

    # ================================================================
    #  Branch 3: XML file overlay
    # ================================================================
    if is_xml:
        assert fig is not None
        _overlay_xml(fig, source, t_scale=t_scale, label=label)
        return fig

    raise TypeError(f"Unsupported source type: {type(source)}")


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
                    t_us * t_scale, amp,
                    linewidth=0.7, alpha=alpha, label=label)

    rf_elem = root.find('.//rf')
    if rf_elem is not None and rf_elem.text:
        data = np.fromstring(rf_elem.text, sep=' ')
        if data.size >= 2:
            n = data.size // 2
            t_us = data[:n]
            amp = data[n:] * g_to_uT
            handle.axes['rf_mag'].plot(
                t_us * t_scale, np.abs(amp),
                linewidth=0.7, alpha=alpha, label=label)

    if label:
        handle.axes['rf_mag'].legend(fontsize=8, loc='upper right')


def _plot_rf_centers(ax, wf, t_fn):
    """Plot RF iso-centre markers on the RF magnitude panel.

    Approximates RF centre as the amplitude-weighted mean time for each
    contiguous RF burst.
    """
    if wf.rf_mag.time_us.size == 0:
        return
    t = wf.rf_mag.time_us
    a = np.abs(wf.rf_mag.amplitude)

    # Split into contiguous bursts (gaps > 10 us)
    if len(t) < 2:
        return
    gaps = np.where(np.diff(t) > 10.0)[0]
    starts = np.concatenate([[0], gaps + 1])
    ends = np.concatenate([gaps + 1, [len(t)]])

    for s, e in zip(starts, ends):
        seg_t = t[s:e]
        seg_a = a[s:e]
        total = np.sum(seg_a)
        if total > 0:
            center_us = np.sum(seg_t * seg_a) / total
            center_display = t_fn(np.array([center_us]))[0]
            ax.axvline(center_display, color='r', ls='--', lw=1.0, alpha=0.7)
            ax.plot(center_display, np.max(seg_a) * 1.05, 'rv',
                    markersize=6, alpha=0.7)


def _plot_echo_markers(ax, wf, t_fn):
    """Plot echo (ADC midpoint) markers on the ADC panel."""
    for adc in wf.adc_events:
        mid_us = adc.onset_us + adc.duration_us / 2.0
        mid_display = t_fn(np.array([mid_us]))[0]
        ax.axvline(mid_display, color='b', ls='--', lw=1.0, alpha=0.7)
        ax.plot(mid_display, 0.5, 'b^', markersize=6, alpha=0.7)


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
    run_start = 0
    for i in range(1, len(t)):
        if seg_idx[i] != seg_idx[run_start]:
            ax.plot(t_fn(t[run_start:i]), a[run_start:i],
                    color=_seg_color(seg_idx[run_start]), **kwargs)
            run_start = i
    ax.plot(t_fn(t[run_start:]), a[run_start:],
            color=_seg_color(seg_idx[run_start]), **kwargs)
