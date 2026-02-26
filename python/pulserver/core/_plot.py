"""TR waveform plotting for SequenceCollection."""

__all__ = ['plot']

from typing import Sequence as SequenceType

import numpy as np

from ._waveforms import TrWaveforms, ChannelWaveform

# Matplotlib is imported lazily to avoid hard dependency at import time.

# ── Colour palette for segments ──────────────────────────────────────
_SEG_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
    '#bcbd22', '#17becf',
]


def _seg_color(idx: int) -> str:
    """Return a colour for segment index *idx* (cycling)."""
    if idx < 0:
        return '#aaaaaa'  # prep/cooldown
    return _SEG_COLORS[idx % len(_SEG_COLORS)]


def _collapse_delays(
    waveforms: TrWaveforms,
    threshold_us: float = 1000.0,
    collapsed_us: float = 100.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build time→display-time mapping that shrinks pure-delay blocks.

    Returns
    -------
    breaks : np.ndarray
        (N, 3) with columns [original_start_us, original_end_us, display_start_us].
    total_display_us : float
    """
    pieces = []
    display_t = 0.0
    for blk in waveforms.blocks:
        start = blk.start_us
        end = start + blk.duration_us
        is_delay = True
        # A block is a "pure delay" if it has no gradient/RF samples in its range.
        # Simple heuristic: check if any grad or RF sample falls in [start, end).
        for ch in (waveforms.gx, waveforms.gy, waveforms.gz,
                   waveforms.rf_mag):
            if ch.time_us.size > 0:
                mask = (ch.time_us >= start) & (ch.time_us < end)
                if np.any(mask):
                    is_delay = False
                    break
        # Also check ADC
        for adc in waveforms.adc_events:
            if adc.onset_us >= start and adc.onset_us < end:
                is_delay = False
                break

        if is_delay and blk.duration_us > threshold_us:
            pieces.append((start, end, display_t, collapsed_us))
            display_t += collapsed_us
        else:
            pieces.append((start, end, display_t, blk.duration_us))
            display_t += blk.duration_us

    breaks = np.array(
        [(s, e, d, dur) for s, e, d, dur in pieces], dtype=np.float64
    )
    return breaks, display_t


def _remap_time(t_us: np.ndarray, breaks: np.ndarray) -> np.ndarray:
    """Map original time values through the collapsed-delay mapping."""
    out = np.empty_like(t_us, dtype=np.float64)
    for i, t in enumerate(t_us):
        # find which piece this time falls in
        for s, e, d, dur in breaks:
            if t >= s and t <= e:
                frac = (t - s) / max(e - s, 1e-12)
                out[i] = d + frac * dur
                break
        else:
            out[i] = t  # fallback
    return out


def _slew_rate(ch: ChannelWaveform) -> ChannelWaveform:
    """Compute numerical derivative dA/dt (units/ms → units/ms)."""
    if ch.time_us.size < 2:
        return ChannelWaveform(
            time_us=np.empty(0, dtype=np.float32),
            amplitude=np.empty(0, dtype=np.float32),
        )
    dt = np.diff(ch.time_us) * 1e-3  # ms → s?  Actually µs → ms = /1e3
    # We want T/m/s from mT/m over µs:
    # d(mT/m)/d(µs) = 1e-3 T/m / 1e-6 s = 1e3 T/m/s
    da = np.diff(ch.amplitude)
    dt_safe = np.where(np.abs(dt) < 1e-12, 1e-12, dt)
    slew = da / dt_safe  # mT/m per µs = 1e3 T/m/s
    slew_T_per_m_per_s = slew * 1e3  # T/m/s
    mid_t = 0.5 * (ch.time_us[:-1] + ch.time_us[1:])
    return ChannelWaveform(
        time_us=mid_t.astype(np.float32),
        amplitude=slew_T_per_m_per_s.astype(np.float32),
    )


def _first_moment(ch: ChannelWaveform) -> ChannelWaveform:
    """Compute cumulative trapezoidal integral (mT/m·µs)."""
    if ch.time_us.size < 2:
        return ChannelWaveform(
            time_us=np.empty(0, dtype=np.float32),
            amplitude=np.empty(0, dtype=np.float32),
        )
    dt = np.diff(ch.time_us)
    cum = np.zeros(len(ch.time_us), dtype=np.float64)
    cum[1:] = np.cumsum(0.5 * (ch.amplitude[:-1] + ch.amplitude[1:]) * dt)
    return ChannelWaveform(
        time_us=ch.time_us.copy(),
        amplitude=cum.astype(np.float32),
    )


def plot(
    waveforms: TrWaveforms | SequenceType[TrWaveforms],
    *,
    labels: SequenceType[str] | None = None,
    collapse_delays: bool = False,
    delay_threshold_us: float = 1000.0,
    collapsed_duration_us: float = 100.0,
    show_segments: bool = True,
    show_blocks: bool = True,
    show_slew: bool = False,
    show_moment: bool = False,
    max_grad_mT_per_m: float | None = None,
    max_slew_T_per_m_per_s: float | None = None,
    time_unit: str = 'ms',
    figsize: tuple | None = None,
):
    """Plot native-timing TR waveforms.

    Panels (top to bottom): RF magnitude (µT), RF phase (rad),
    Gx (mT/m), Gy (mT/m), Gz (mT/m), ADC.

    Parameters
    ----------
    waveforms : TrWaveforms or list[TrWaveforms]
        One or more waveform sets to overlay.  Each can come from
        :func:`get_tr_waveforms` (segmented representation),
        a pypulseq extraction helper, or an XML parser.
    labels : list[str] or None
        Legend labels for each waveform set.
    collapse_delays : bool
        Shrink pure-delay blocks to a small virtual duration.
    delay_threshold_us : float
        Blocks longer than this with no events are collapsed.
    collapsed_duration_us : float
        Display duration (µs) for collapsed delay blocks.  Default 100
        (i.e. 0.1 ms).
    show_segments : bool
        Colour-code gradient waveforms by segment index.
    show_blocks : bool
        Draw light vertical dotted lines at block boundaries.
    show_slew : bool
        Overlay slew rate (dG/dt) on gradient panels (right y-axis).
    show_moment : bool
        Overlay gradient first moment (integral) on gradient panels.
    max_grad_mT_per_m : float or None
        Draw horizontal reference line for max gradient amplitude.
    max_slew_T_per_m_per_s : float or None
        Draw horizontal reference line for max slew rate (only if
        ``show_slew=True``).
    time_unit : str
        ``'ms'`` (default) or ``'us'``.
    figsize : tuple or None
        Figure size.  Default is ``(14, 10)``.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object (for further customisation or saving).
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if isinstance(waveforms, TrWaveforms):
        waveforms = [waveforms]
    if labels is None:
        labels = [None] * len(waveforms)

    n_sources = len(waveforms)
    has_rf = any(w.rf_mag.time_us.size > 0 for w in waveforms)
    has_adc = any(len(w.adc_events) > 0 for w in waveforms)

    # Determine panels
    panels = []
    if has_rf:
        panels.append('rf_mag')
        panels.append('rf_phase')
    panels.extend(['gx', 'gy', 'gz'])
    if has_adc:
        panels.append('adc')
    n_panels = len(panels)

    if figsize is None:
        figsize = (14, 2.0 * n_panels)

    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True)
    if n_panels == 1:
        axes = [axes]

    t_scale = 1e-3 if time_unit == 'ms' else 1.0
    t_label = 'Time (ms)' if time_unit == 'ms' else 'Time (µs)'

    # Delay collapsing (use first waveform's blocks for the mapping)
    breaks = None
    if collapse_delays and len(waveforms[0].blocks) > 0:
        breaks, _ = _collapse_delays(
            waveforms[0], threshold_us=delay_threshold_us,
            collapsed_us=collapsed_duration_us,
        )

    def _t(t_us):
        """Map time array to display coordinates."""
        if breaks is not None:
            return _remap_time(t_us, breaks) * t_scale
        return t_us * t_scale

    # ── Draw each source ──
    source_alpha = 0.85 if n_sources > 1 else 1.0

    for src_idx, (wf, lbl) in enumerate(zip(waveforms, labels)):
        for panel_idx, panel_name in enumerate(panels):
            ax = axes[panel_idx]

            if panel_name == 'rf_mag':
                ch = wf.rf_mag
                if ch.time_us.size > 0:
                    ax.plot(_t(ch.time_us), ch.amplitude,
                            color='k' if n_sources == 1 else None,
                            linewidth=0.8, alpha=source_alpha, label=lbl)
                if src_idx == 0:
                    ax.set_ylabel('|RF| (µT)')

            elif panel_name == 'rf_phase':
                ch = wf.rf_phase
                if ch.time_us.size > 0:
                    ax.plot(_t(ch.time_us), ch.amplitude,
                            color='k' if n_sources == 1 else None,
                            linewidth=0.8, alpha=source_alpha, label=lbl)
                if src_idx == 0:
                    ax.set_ylabel('∠RF (rad)')
                    ax.set_yticks([-np.pi, 0, np.pi])
                    ax.set_yticklabels(['-π', '0', 'π'])

            elif panel_name in ('gx', 'gy', 'gz'):
                ch = getattr(wf, panel_name)
                axis_color = {'gx': '#1f77b4', 'gy': '#2ca02c', 'gz': '#d62728'}

                if ch.time_us.size > 0:
                    if show_segments and n_sources == 1 and len(wf.blocks) > 0:
                        # Colour-code by segment
                        _plot_segmented(ax, _t, ch, wf.blocks,
                                        linewidth=0.8, alpha=source_alpha)
                    else:
                        ax.plot(_t(ch.time_us), ch.amplitude,
                                color=axis_color[panel_name] if n_sources == 1 else None,
                                linewidth=0.8, alpha=source_alpha, label=lbl)

                if src_idx == 0:
                    ax.set_ylabel(f'{panel_name.upper()} (mT/m)')

                    if max_grad_mT_per_m is not None:
                        ax.axhline(max_grad_mT_per_m, color='gray',
                                   ls='--', lw=0.6, alpha=0.6)
                        ax.axhline(-max_grad_mT_per_m, color='gray',
                                   ls='--', lw=0.6, alpha=0.6)

                # Slew rate overlay
                if show_slew and src_idx == 0 and ch.time_us.size > 1:
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

                # First-moment overlay
                if show_moment and src_idx == 0 and ch.time_us.size > 1:
                    mom = _first_moment(ch)
                    ax_m = ax.twinx()
                    if show_slew:
                        # offset the spine
                        ax_m.spines['right'].set_position(('axes', 1.12))
                    ax_m.plot(_t(mom.time_us), mom.amplitude,
                              color='purple', linewidth=0.5, alpha=0.5)
                    ax_m.set_ylabel('M1 (mT/m·µs)', color='purple', fontsize=8)
                    ax_m.tick_params(axis='y', labelcolor='purple', labelsize=7)

            elif panel_name == 'adc':
                if src_idx == 0:
                    ax.set_ylabel('ADC')
                    ax.set_yticks([])
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

    # ── Block boundaries (from first source only) ──
    wf0 = waveforms[0]
    if show_blocks and len(wf0.blocks) > 0:
        for blk in wf0.blocks:
            t_start = _t(np.array([blk.start_us]))[0]
            for ax in axes:
                ax.axvline(t_start, color='k', ls=':', lw=0.3, alpha=0.3)

    # ── Finalise ──
    axes[-1].set_xlabel(t_label)
    for ax in axes:
        ax.grid(True, alpha=0.2)

    if n_sources > 1 and any(l is not None for l in labels):
        axes[0].legend(fontsize=8, loc='upper right')

    fig.tight_layout()
    return fig


def _plot_segmented(ax, t_fn, ch: ChannelWaveform, blocks, **kwargs):
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

    # Plot contiguous runs of same segment
    if len(t) == 0:
        return
    run_start = 0
    for i in range(1, len(t)):
        if seg_idx[i] != seg_idx[run_start]:
            _s = slice(run_start, i + 1)  # overlap by 1 for continuity
            if i < len(t):
                _s = slice(run_start, min(i + 1, len(t)))
            ax.plot(t_fn(t[run_start:i]), a[run_start:i],
                    color=_seg_color(seg_idx[run_start]), **kwargs)
            run_start = i
    # last run
    ax.plot(t_fn(t[run_start:]), a[run_start:],
            color=_seg_color(seg_idx[run_start]), **kwargs)
