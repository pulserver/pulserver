"""Gradient spectrum (mechanical resonances) visualisation for SequenceCollection (plotting only, no pass/fail)."""

__all__ = ['grad_spectrum']

import warnings

import numpy as np

from ._extension._pulseqlib_wrapper import _calc_mech_resonances, _find_tr
from ._helpers import _add_echo_spacing_axis
from ._sequence import SequenceCollection


def _parse_band_axis(tag: object) -> str | None:
    """Normalize a user/system channel tag to one of gx/gy/gz."""
    if isinstance(tag, str):
        t = tag.strip().lower()
        if t in ('gx', 'x'):
            return 'gx'
        if t in ('gy', 'y'):
            return 'gy'
        if t in ('gz', 'z'):
            return 'gz'
    if isinstance(tag, (int, np.integer)):
        idx = int(tag)
        if idx == 0:
            return 'gx'
        if idx == 1:
            return 'gy'
        if idx == 2:
            return 'gz'
    return None


def _plot_grad_spectrum_for_subsequence(
    seq: SequenceCollection,
    *,
    subsequence_idx: int,
    canonical_tr_indices: list[int],
    forbidden_bands: list[tuple[float, float, float] | tuple[float, float, float, str]],
    spectral_resolution: float,
    max_frequency: float,
    peak_log10_threshold: float | None,
    peak_norm_scale: float | None,
    peak_eps: float | None,
    peak_prominence: float | None,
) -> None:
    """Plot gradient spectra for all canonical TRs of one subsequence.

    Each panel shows the analytical power spectrum comb (TR harmonics weighted
    by |H(f)|²) for the corresponding gradient axis.  All CTR combs are
    normalised to the global peak power across all axes and all CTRs so that
    relative channel levels are directly comparable; y-limits are fixed at
    (0, 1.1).  The first canonical TR is drawn solid; additional CTRs are
    dashed overlays at decreasing opacity.

    Forbidden bands are shaded in every panel regardless of their associated
    channel.  Structural candidates appear as dotted vertical lines that span
    the full panel height, shown only in the panel(s) where they carry
    significant spectral weight (≥ 25 % of the cross-axis maximum).  The
    effective gradient amplitude in mT/m is annotated beside the line only
    when the candidate violates a forbidden band.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    gamma_hz_per_t = float(seq.system.gamma)
    hz_per_m_to_mT_per_m = 1e3 / gamma_hz_per_t
    n_ctrs = len(canonical_tr_indices)
    axis_names = ('gx', 'gy', 'gz')
    axis_labels = {'gx': 'Gx', 'gy': 'Gy', 'gz': 'Gz'}
    colors = {'gx': 'C0', 'gy': 'C1', 'gz': 'C2'}

    # --- Parse forbidden bands ---
    plot_bands: list[tuple[float, float, float, str]] = []
    c_forbidden_bands: list[tuple[float, float, float]] = []
    for bi, band in enumerate(forbidden_bands):
        if len(band) == 4:
            b0, b1, b2, b3 = band
            b_axis = _parse_band_axis(b3)
            if b_axis is None:
                b_axis = axis_names[bi % 3]
        else:
            b0, b1, b2 = band
            b_axis = axis_names[bi % 3]
        c_forbidden_bands.append((float(b0), float(b1), float(b2)))
        plot_bands.append((float(b0), float(b1), float(b2), b_axis))

    # --- Collect per-CTR result dicts ---
    all_rd: list[dict] = []
    for ctr_idx in canonical_tr_indices:
        rd = _calc_mech_resonances(
            seq._cseq,
            subsequence_idx=subsequence_idx,
            canonical_tr_idx=ctr_idx,
            target_resolution_hz=spectral_resolution,
            max_freq_hz=max_frequency,
            forbidden_bands=c_forbidden_bands,
            peak_log10_threshold=peak_log10_threshold,
            peak_norm_scale=peak_norm_scale,
            peak_eps=peak_eps,
            peak_prominence=peak_prominence,
        )
        all_rd.append(rd)

    freq_min = 0.0
    freq_max = max_frequency

    # --- Build TR FFT power spectra (dense uniform grid) ---
    # spectrum_full_gx/gy/gz[k] = |FFT(canonical_TR_waveform)| on the dense
    # uniform frequency grid (freq_min_hz + k * freq_spacing_hz).  Squaring
    # gives the power spectrum — the continuous envelope of the Dirichlet comb.
    # Global normalisation preserves relative scale across axes and CTRs.
    comb_data: list[dict[str, tuple[np.ndarray, np.ndarray]]] = []
    global_peak_power = 0.0
    for rd in all_rd:
        n_bins = int(rd.get('num_freq_bins', 0))
        f_min = float(rd.get('freq_min_hz', 0.0))
        f_spacing = float(rd.get('freq_spacing_hz', 1.0))
        per_axis: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for ax_name in axis_names:
            if n_bins > 0:
                amps = np.asarray(
                    rd.get(f'spectrum_full_{ax_name}', []), dtype=np.float64
                )
                n = min(n_bins, len(amps))
                f = f_min + np.arange(n, dtype=np.float64) * f_spacing
                pw = amps[:n] ** 2
            else:
                f = np.empty(0, dtype=np.float64)
                pw = np.empty(0, dtype=np.float64)
            per_axis[ax_name] = (f, pw)
            if pw.size > 0 and np.any(np.isfinite(pw)):
                p = float(np.nanmax(pw))
                if p > global_peak_power:
                    global_peak_power = p
        comb_data.append(per_axis)
    if global_peak_power <= 0.0:
        global_peak_power = 1.0

    # --- Figure ---
    title_suffix = (
        f'CTR{canonical_tr_indices[0]}-CTR{canonical_tr_indices[-1]}'
        if n_ctrs > 1
        else f'CTR{canonical_tr_indices[0]}'
    )
    title = f'[SS{subsequence_idx}, {title_suffix}] Mechanical Resonances'
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    for panel_idx, ax_name in enumerate(axis_names):
        ax = axes[panel_idx]
        color = colors[ax_name]

        # 1 — TR FFT power spectrum envelope (underlay, continuous line)
        for k, per_axis in enumerate(comb_data):
            f, pw = per_axis[ax_name]
            if len(f) == 0:
                continue
            pw_norm = pw / global_peak_power
            if k == 0:
                ax.plot(
                    f,
                    pw_norm,
                    color=color,
                    linewidth=1.2,
                    alpha=0.85,
                    linestyle='solid',
                    zorder=1,
                )
            else:
                alpha_k = max(0.20, 0.50 - 0.12 * k)
                ax.plot(
                    f,
                    pw_norm,
                    color=color,
                    linewidth=0.7,
                    alpha=alpha_k,
                    linestyle='dashed',
                    zorder=1,
                )

        # 2 — Forbidden bands shaded across ALL panels
        for blo, bhi, _blim, b_ax in plot_bands:
            ax.axvspan(blo, bhi, alpha=0.10, color=colors[b_ax], zorder=0)

        # 3 — Structural candidate dotted vertical lines
        # Each candidate is drawn only in panel(s) where its per-axis G_eff
        # is nonzero (i.e. the axis was flagged as a candidate at that
        # frequency).  The effective gradient amplitude (mT/m) is annotated
        # beside the line only when the candidate falls within a forbidden
        # band.
        for rd in all_rd:
            cf_arr = np.asarray(rd.get('candidate_freqs', []), dtype=np.float64)
            {
                n: np.asarray(rd.get(f'candidate_amps_{n}', []), dtype=np.float64)
                for n in axis_names
            }
            grad_amp_per_ax = {
                n: np.asarray(rd.get(f'candidate_grad_amps_{n}', []), dtype=np.float64)
                for n in axis_names
            }
            # Within each forbidden band, suppress all but the highest Geff,ax
            # candidate so that only the worst-case peak is displayed per band.
            best_in_any_band: set = set()
            in_any_band: set = set()
            for blo, bhi, _blim, _ in plot_bands:
                in_band = [
                    c
                    for c in range(len(cf_arr))
                    if blo <= float(cf_arr[c]) <= bhi
                    and c < len(grad_amp_per_ax[ax_name])
                    and float(grad_amp_per_ax[ax_name][c]) > 0.0
                ]
                if not in_band:
                    continue
                in_any_band.update(in_band)
                best = max(in_band, key=lambda c: float(grad_amp_per_ax[ax_name][c]))
                best_in_any_band.add(best)
            suppressed = in_any_band - best_in_any_band
            for ci in range(len(cf_arr)):
                cf = float(cf_arr[ci])
                if cf < freq_min or cf > freq_max:
                    continue
                # Skip this panel if the axis was not a candidate
                gamp_hz = (
                    float(grad_amp_per_ax[ax_name][ci])
                    if ci < len(grad_amp_per_ax[ax_name])
                    else 0.0
                )
                if gamp_hz <= 0.0:
                    continue
                # Full-height dotted line
                ax.axvline(
                    cf,
                    color=color,
                    linestyle='dotted',
                    linewidth=1.0,
                    alpha=0.85,
                    zorder=3,
                )
                # Annotate effective amplitude only if this axis violates a band,
                # and annotate only the dominant candidate per band.
                if ci not in suppressed and any(
                    blo <= cf <= bhi and gamp_hz > blim
                    for blo, bhi, blim, _ in plot_bands
                ):
                    gamp_mT = gamp_hz * hz_per_m_to_mT_per_m
                    ax.annotate(
                        f'{gamp_mT:.1f}',
                        xy=(cf, 1.04),
                        xycoords=('data', 'axes fraction'),
                        ha='center',
                        va='bottom',
                        fontsize=6,
                        fontweight='bold',
                        color=color,
                        alpha=0.9,
                        clip_on=False,
                    )

        ax.set_xlim(freq_min, freq_max)
        ax.set_ylim(0.0, 1.1)
        ax.set_ylabel(f'{axis_labels[ax_name]} power (norm.)')
        ax.grid(True, alpha=0.3)

    axes[0].set_title(title)
    axes[2].set_xlabel('Frequency (Hz)')

    # --- Legend ---
    legend_handles: list = []
    for ax_name in axis_names:
        lbl = f'{axis_labels[ax_name]} spectrum'
        if n_ctrs > 1:
            lbl += ' (CTR0, solid)'
        legend_handles.append(
            Line2D(
                [0], [0], color=colors[ax_name], lw=1.4, linestyle='solid', label=lbl
            )
        )
    if n_ctrs > 1:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color='gray',
                lw=0.9,
                linestyle='dashed',
                alpha=0.5,
                label='Other CTRs (dashed)',
            )
        )
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color='gray',
            lw=1.0,
            linestyle='dotted',
            label='Structural candidates',
        )
    )
    for _bi, (blo, bhi, blim, b_ax) in enumerate(plot_bands):
        blim_mT = blim * hz_per_m_to_mT_per_m
        legend_handles.append(
            Patch(
                facecolor=colors[b_ax],
                alpha=0.25,
                edgecolor=colors[b_ax],
                label=f'{axis_labels[b_ax]} band [{blo:.0f} - {bhi:.0f} Hz, ≤{blim_mT:.1f} mT/m]',
            )
        )
    n_cols = min(4, max(1, len(legend_handles)))
    fig.legend(
        handles=legend_handles,
        loc='lower center',
        ncol=n_cols,
        frameon=True,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.00),
    )

    _add_echo_spacing_axis(axes[0], freq_min, freq_max)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fig.tight_layout(rect=(0.0, 0.10, 1.0, 1.0))


def grad_spectrum(
    seq: SequenceCollection,
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
    """Plot gradient spectrum (mechanical resonance candidates) for the canonical TR.

    Creates a single-panel figure with three overlaid spectra (Gx, Gy, Gz) showing
    surviving peak candidates as vertical lines, forbidden bands as shaded regions,
    and max gradient amplitude annotations (mT/m) per candidate and per band.

    Two frequency axes are provided: frequency in Hz (bottom) and the corresponding
    echo spacing in ms (top). The vertical axis is in arbitrary units (a.u.).

    No pass/fail check is performed — use
    :meth:`SequenceCollection.check` for that.

    Parameters
    ----------
    seq : SequenceCollection
        The sequence to analyse.
    sequence_idx : int, optional
        Subsequence index (0-based, default all).
    forbidden_bands : list of (freq_min, freq_max, max_amplitude), optional
        Forbidden frequency bands.  Each tuple gives
        ``(freq_min_Hz, freq_max_Hz, max_allowed_amplitude_Hz_per_m)``.
        Drawn as shaded regions; the max allowed amplitude is labelled in mT/m.
    spectral_resolution : float
        Target frequency resolution in Hz (default 5 Hz).
    max_frequency : float
        Upper frequency limit in Hz (default 3000 Hz).
    peak_log10_threshold : float, optional
        Resonance detector threshold in log10 space.
    peak_norm_scale : float, optional
        Normalization scale used before the log transform in resonance detection.
    peak_eps : float, optional
        Positive epsilon added before log transform for numerical stability.
    peak_prominence : float, optional
        Minimum prominence (in log10 units) for a peak to be retained.
    """
    if forbidden_bands is None:
        forbidden_bands = []

    if sequence_idx is None:
        subsequence_indices = list(range(seq.num_sequences))
    else:
        if sequence_idx < 0 or sequence_idx >= seq.num_sequences:
            raise ValueError(
                f'sequence_idx={sequence_idx} out of range for {seq.num_sequences} subsequences'
            )
        subsequence_indices = [int(sequence_idx)]

    for ss_idx in subsequence_indices:
        tr_info = _find_tr(seq._cseq, subsequence_idx=ss_idx)
        num_canonical = int(tr_info.get('num_canonical_trs', 1))
        _plot_grad_spectrum_for_subsequence(
            seq,
            subsequence_idx=ss_idx,
            canonical_tr_indices=list(range(num_canonical)),
            forbidden_bands=forbidden_bands,
            spectral_resolution=spectral_resolution,
            max_frequency=max_frequency,
            peak_log10_threshold=peak_log10_threshold,
            peak_norm_scale=peak_norm_scale,
            peak_eps=peak_eps,
            peak_prominence=peak_prominence,
        )


# Alias
mechanical_resonances = grad_spectrum
