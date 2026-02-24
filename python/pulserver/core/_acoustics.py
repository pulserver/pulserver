"""Acoustic spectra analysis for PulserverSequence."""

__all__ = ['get_tr_acoustic_spectra']

import warnings
from types import SimpleNamespace

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from ._extension._pulseqlib_wrapper import _calc_acoustic_spectra
from ._helpers import _add_echo_spacing_axis
from ._sequence import PulserverSequence
from ._analysis import get_tr_gradient_waveforms


def get_tr_acoustic_spectra(
    seq: PulserverSequence,
    window_duration: float = 25.0e-3,
    spectral_resolution: float = 5.0,
    max_frequency: float = 3000.0,
    combined: bool = False,
    forbidden_bands: list[dict] | None = None,
    do_plot: bool = False,
) -> SimpleNamespace:
    """
    Compute acoustic spectra for gradient waveforms in a TR.

    Performs sliding-window FFT analysis on each gradient axis to identify
    potential acoustic resonance frequencies, plus full TR and N-TR sequence spectra.

    Parameters
    ----------
    seq : PulserverSequence
        The sequence to analyze.
    window_duration : float
        Target window size in seconds for sliding window analysis.
        Default is 0.025 s.
    spectral_resolution : float
        Target frequency resolution in Hz.  Default is 5.0 Hz.
        FFT size is automatically chosen via zero-padding to achieve
        approximately this resolution.
    max_frequency : float
        Maximum frequency to include in output (Hz).  Default is 3000.0 Hz.
        If ``None``, the full spectrum up to Nyquist is returned.
    combined : bool
        If ``True``, return pointwise maximum across all windows (1-D arrays).
        If ``False``, stack all windows (2-D arrays).  Default is ``False``.
    forbidden_bands : list[dict] | None
        Optional list of forbidden frequency bands for acoustic resonance
        check.  Each dict should contain:

        - ``freq_min_hz`` : float — minimum frequency of the band (Hz)
        - ``freq_max_hz`` : float — maximum frequency of the band (Hz)
        - ``max_amplitude`` : float — maximum allowed gradient amplitude (Hz/m)

        If ``None`` or empty list, no acoustic check is performed.
    do_plot : bool
        If ``True``, plot the computed spectra.  Default is ``False``.

    Returns
    -------
    SimpleNamespace
        Result containing:

        *Sliding window spectra:*

        - ``frequencies`` : np.ndarray of shape ``(num_freq_bins,)``
        - ``spectra_gx`` / ``spectra_gy`` / ``spectra_gz`` : np.ndarray —
          shape ``(num_windows, num_freq_bins)`` if ``combined=False``,
          or ``(num_freq_bins,)`` if ``combined=True``
        - ``max_envelope_gx`` / ``_gy`` / ``_gz`` : np.ndarray (mT/m)
        - ``peaks_gx`` / ``_gy`` / ``_gz`` : np.ndarray or ``None`` (only
          when ``combined=False``)

        *Full TR spectrum:*

        - ``spectrum_full_gx`` / ``_gy`` / ``_gz`` : np.ndarray
        - ``peaks_full_gx`` / ``_gy`` / ``_gz`` : np.ndarray

        *N-TR sequence spectra (only when num_trs > 1):*

        - ``frequencies_seq`` : np.ndarray
        - ``spectrum_gx_seq`` / ``_gy_seq`` / ``_gz_seq`` : np.ndarray
        - ``peaks_gx_seq`` / ``_gy_seq`` / ``_gz_seq`` : np.ndarray

    Notes
    -----
    Three types of spectral analysis are performed:

    1. **Sliding window spectra** — temporal evolution of frequencies
       (50 % overlap, zero-padded, cosine tapered, real FFT).
    2. **Full TR spectrum** — single FFT of the full TR waveform.
    3. **N-TR sequence spectrum** — spectral lines at harmonics of 1/TR
       (only computed when num_trs > 1).

    When ``combined=True`` the sliding-window output is the pointwise
    maximum magnitude across all windows, useful for worst-case analysis.
    """
    if max_frequency is None:
        max_frequency = -1.0

    if forbidden_bands is None:
        forbidden_bands = []

    gamma = seq.system.gamma
    grad_raster_time = seq.system.grad_raster_time

    # Window size in half-raster samples
    target_window_size = int(2.0 * window_duration / grad_raster_time)

    # Call C++ extension
    rd = _calc_acoustic_spectra(
        seq._cseq,
        target_window_size,
        spectral_resolution,
        max_frequency,
        forbidden_bands,
    )

    num_windows = rd["num_windows"]
    num_freq_bins = rd["num_freq_bins"]

    # Reconstruct frequency axis
    frequencies = rd["freq_min_hz"] + np.arange(num_freq_bins) * rd["freq_spacing_hz"]

    # Reshape spectrograms  (flat → 2-D)
    spectrogram_gx = np.asarray(rd["spectrogram_gx"], dtype=np.float32).reshape(
        num_windows, num_freq_bins
    )
    spectrogram_gy = np.asarray(rd["spectrogram_gy"], dtype=np.float32).reshape(
        num_windows, num_freq_bins
    )
    spectrogram_gz = np.asarray(rd["spectrogram_gz"], dtype=np.float32).reshape(
        num_windows, num_freq_bins
    )

    peaks_gx = np.asarray(rd["peaks_gx"], dtype=np.int32).reshape(
        num_windows, num_freq_bins
    )
    peaks_gy = np.asarray(rd["peaks_gy"], dtype=np.int32).reshape(
        num_windows, num_freq_bins
    )
    peaks_gz = np.asarray(rd["peaks_gz"], dtype=np.int32).reshape(
        num_windows, num_freq_bins
    )

    # Max-envelope over windows (mT/m)
    max_envelope_gx = spectrogram_gx.max(axis=0) / gamma * 1000.0
    max_envelope_gy = spectrogram_gy.max(axis=0) / gamma * 1000.0
    max_envelope_gz = spectrogram_gz.max(axis=0) / gamma * 1000.0

    if combined:
        spectra_gx = spectrogram_gx.max(axis=0)
        spectra_gy = spectrogram_gy.max(axis=0)
        spectra_gz = spectrogram_gz.max(axis=0)
        out_peaks_gx = None
        out_peaks_gy = None
        out_peaks_gz = None
    else:
        spectra_gx = spectrogram_gx
        spectra_gy = spectrogram_gy
        spectra_gz = spectrogram_gz
        out_peaks_gx = peaks_gx
        out_peaks_gy = peaks_gy
        out_peaks_gz = peaks_gz

    result = SimpleNamespace(
        frequencies=frequencies.astype(np.float32),
        spectra_gx=spectra_gx,
        spectra_gy=spectra_gy,
        spectra_gz=spectra_gz,
        max_envelope_gx=max_envelope_gx,
        max_envelope_gy=max_envelope_gy,
        max_envelope_gz=max_envelope_gz,
        peaks_gx=out_peaks_gx,
        peaks_gy=out_peaks_gy,
        peaks_gz=out_peaks_gz,
    )

    # Full-TR spectrum
    result.spectrum_full_gx = np.asarray(rd["spectrum_full_gx"], dtype=np.float32)
    result.spectrum_full_gy = np.asarray(rd["spectrum_full_gy"], dtype=np.float32)
    result.spectrum_full_gz = np.asarray(rd["spectrum_full_gz"], dtype=np.float32)
    result.peaks_full_gx = np.asarray(rd["peaks_full_gx"], dtype=np.int32)
    result.peaks_full_gy = np.asarray(rd["peaks_full_gy"], dtype=np.int32)
    result.peaks_full_gz = np.asarray(rd["peaks_full_gz"], dtype=np.int32)

    # Sequence spectra (only when numTRs > 1)
    num_seq_bins = rd.get("num_freq_bins_seq", 0)
    if num_seq_bins > 0 and "spectrum_seq_gx" in rd:
        freq_spacing_seq = rd["freq_spacing_seq_hz"]
        result.frequencies_seq = (
            np.arange(num_seq_bins) * freq_spacing_seq
        ).astype(np.float32)
        result.spectrum_gx_seq = np.asarray(rd["spectrum_seq_gx"], dtype=np.float32)
        result.spectrum_gy_seq = np.asarray(rd["spectrum_seq_gy"], dtype=np.float32)
        result.spectrum_gz_seq = np.asarray(rd["spectrum_seq_gz"], dtype=np.float32)
        result.max_envelope_gx_seq = result.spectrum_gx_seq / gamma * 1000.0
        result.max_envelope_gy_seq = result.spectrum_gy_seq / gamma * 1000.0
        result.max_envelope_gz_seq = result.spectrum_gz_seq / gamma * 1000.0
        result.peaks_gx_seq = np.asarray(rd["peaks_seq_gx"], dtype=np.int32)
        result.peaks_gy_seq = np.asarray(rd["peaks_seq_gy"], dtype=np.int32)
        result.peaks_gz_seq = np.asarray(rd["peaks_seq_gz"], dtype=np.int32)

    if do_plot:
        _plot_acoustic_spectra(result, seq=seq, forbidden_bands=forbidden_bands)

    # Acoustic check
    if forbidden_bands:
        _check_acoustic_forbidden_bands(result, seq, forbidden_bands)

    return result


# ── plotting ─────────────────────────────────────────────────────────

def _plot_acoustic_spectra(
    spectra: SimpleNamespace,
    seq: PulserverSequence | None = None,
    forbidden_bands: list[dict] | None = None,
) -> tuple:
    """
    Plot acoustic spectra with waveforms and sliding windows.

    Creates a comprehensive visualisation of acoustic spectra including:

    - Full sequence spectrum with waveforms
    - Sliding window spectra matrix (if combined=False)
    - Detected peaks and forbidden frequency bands

    Parameters
    ----------
    spectra : SimpleNamespace
        Output from :func:`get_tr_acoustic_spectra` with ``combined=False``.
    seq : PulserverSequence | None
        The sequence object (needed to get waveforms and system parameters).
        If ``None``, waveform panel is skipped.
    forbidden_bands : list[dict] | None
        List of forbidden frequency bands.

    Returns
    -------
    tuple
        ``(fig, axes)`` for further customisation.
    """
    if not hasattr(spectra, 'spectra_gx') or not hasattr(spectra, 'peaks_gx'):
        raise ValueError("spectra must be from get_tr_acoustic_spectra() with combined=False")

    if spectra.peaks_gx is None:
        raise ValueError("Spectra must have peaks detected (combined=False)")

    if spectra.spectra_gx.ndim != 2:
        num_windows = 1
    else:
        num_windows = spectra.spectra_gx.shape[0]

    # Waveforms
    waveforms = None
    if seq is not None:
        try:
            waveforms = get_tr_gradient_waveforms(seq)
            gamma = seq.system.gamma
        except Exception as e:
            print(f"Warning: Could not get waveforms: {e}")

    fig = plt.figure()

    freq_min = 0.0
    freq_max = spectra.frequencies[-1]

    # ── Panel 1: Full Sequence + Waveforms ──
    ax_seq_spec = plt.subplot(2, 1, 1)

    colors = {'x': 'C0', 'y': 'C1', 'z': 'C2'}
    labels = {'x': 'Gx', 'y': 'Gy', 'z': 'Gz'}

    for axis_name, color in colors.items():
        spec_attr = f'spectra_g{axis_name}'
        peaks_attr = f'peaks_g{axis_name}'

        spectra_axis = getattr(spectra, spec_attr)
        peaks_axis = getattr(spectra, peaks_attr, None)

        spec_max = spectra_axis.max(axis=0)

        ax_seq_spec.plot(
            spectra.frequencies, spec_max, color=color, linewidth=2,
            label=labels[axis_name],
        )

        if peaks_axis is not None:
            peaks_any = peaks_axis.max(axis=0) > 0
            peak_freqs = spectra.frequencies[peaks_any]
            peak_mags = spec_max[peaks_any]
            ax_seq_spec.plot(
                peak_freqs, peak_mags,
                marker='o', color=color, linestyle='none',
                markersize=6.5, markerfacecolor=color,
                markeredgecolor=color, markeredgewidth=0,
            )

    if forbidden_bands:
        for band in forbidden_bands:
            ax_seq_spec.axvline(band['freq_min_hz'], color='black',
                                linestyle='--', linewidth=1.5, alpha=0.7)
            ax_seq_spec.axvline(band['freq_max_hz'], color='black',
                                linestyle='--', linewidth=1.5, alpha=0.7)

    ax_seq_spec.set_xlim(freq_min, freq_max)
    ax_seq_spec.set_xlabel('Frequency (Hz)', fontsize=12)
    ax_seq_spec.set_ylabel('Magnitude [a.u.]', fontsize=12)
    ax_seq_spec.set_title('Full Sequence Acoustic Spectrum', fontsize=14,
                          fontweight='bold')
    ax_seq_spec.legend(loc='upper right', fontsize=11)
    ax_seq_spec.grid(True, alpha=0.3)
    _add_echo_spacing_axis(ax_seq_spec, freq_min, freq_max)

    # ── Panel 2: Waveforms ──
    if waveforms is not None:
        ax_wf = plt.subplot(2, 1, 2)

        wf_gx_mtpm = waveforms.waveform_gx / gamma * 1000.0
        wf_gy_mtpm = waveforms.waveform_gy / gamma * 1000.0
        wf_gz_mtpm = waveforms.waveform_gz / gamma * 1000.0
        time_ms = waveforms.time / 1000.0

        ax_wf.plot(time_ms, wf_gx_mtpm, color=colors['x'], linewidth=2,
                   label='Gx')
        ax_wf.plot(time_ms, wf_gy_mtpm, color=colors['y'], linewidth=2,
                   label='Gy')
        ax_wf.plot(time_ms, wf_gz_mtpm, color=colors['z'], linewidth=2,
                   label='Gz')

        ax_wf.set_xlabel('Time (ms)', fontsize=12)
        ax_wf.set_ylabel('Gradient Amplitude (mT/m)', fontsize=12)
        ax_wf.set_title('TR Gradient Waveforms', fontsize=14, fontweight='bold')
        ax_wf.legend(loc='upper right', fontsize=11)
        ax_wf.grid(True, alpha=0.3)

    plt.figure(fig.number)
    plt.tight_layout(rect=[0, 0, 1, 1])

    # ── Panel 3: Sliding Window Spectrograms ──
    fig2 = None
    if num_windows > 1:
        fig2 = plt.figure()
        im = None

        for idx, (axis_name, color) in enumerate(colors.items()):
            ax = plt.subplot(1, 3, 1 + idx)

            spectra_matrix = getattr(spectra, f'spectra_g{axis_name}')
            peaks_matrix = getattr(spectra, f'peaks_g{axis_name}', None)

            im = ax.pcolormesh(
                spectra.frequencies, np.arange(num_windows),
                spectra_matrix, cmap='viridis', shading='auto',
                norm=Normalize(vmin=spectra_matrix.min(),
                               vmax=spectra_matrix.max()),
            )

            if peaks_matrix is not None:
                peak_coords = np.where(peaks_matrix)
                if len(peak_coords[0]) > 0:
                    ax.plot(
                        spectra.frequencies[peak_coords[1]], peak_coords[0],
                        marker='*', color='red', linestyle='none',
                        markersize=16, markerfacecolor='red',
                        markeredgecolor='red', markeredgewidth=0,
                    )

            if forbidden_bands:
                for band in forbidden_bands:
                    ax.axvline(band['freq_min_hz'], color='white',
                               linestyle='--', linewidth=1.5, alpha=0.8)
                    ax.axvline(band['freq_max_hz'], color='white',
                               linestyle='--', linewidth=1.5, alpha=0.8)

            ax.set_xlim(freq_min, freq_max)
            ax.set_xlabel('Frequency (Hz)', fontsize=11)
            ax.set_ylabel('Window Index', fontsize=11)
            ax.set_title(f'Sliding Window - {labels[axis_name]}',
                         fontsize=12, fontweight='bold')
            ax.set_ylim(-0.5, num_windows - 0.5)

            max_ticks = 10
            tick_stride = max(1, int(np.ceil(num_windows / max_ticks)))
            yticks = np.arange(0, num_windows, tick_stride)
            ax.set_yticks(yticks)
            ax.set_yticklabels([str(int(i)) for i in yticks], fontsize=9)

            for win_idx in np.arange(0.5, num_windows, 1):
                ax.axhline(win_idx, color='white', linestyle='-',
                           linewidth=0.5, alpha=0.6)

            _add_echo_spacing_axis(ax, freq_min, freq_max)

        if im is not None:
            cbar_ax = fig2.add_axes([0.15, 0.05, 0.7, 0.02])
            cbar = fig2.colorbar(im, cax=cbar_ax, orientation='horizontal')
            cbar.set_label('Magnitude [a.u.]', fontsize=11)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fig2.tight_layout(rect=[0, 0.08, 1, 1])

    return fig, fig.axes


# ── acoustic band check ──────────────────────────────────────────────

def _check_acoustic_forbidden_bands(
    spectra: SimpleNamespace,
    seq: PulserverSequence,
    forbidden_bands: list[dict],
) -> None:
    """
    Check acoustic spectra against forbidden frequency bands and warn.
    """
    if not forbidden_bands:
        return

    gamma = seq.system.gamma
    axes = {'gx': 'X', 'gy': 'Y', 'gz': 'Z'}

    for band in forbidden_bands:
        freq_min = band['freq_min_hz']
        freq_max = band['freq_max_hz']
        max_allowed_hzm = band.get('max_amplitude', float('inf'))
        max_allowed_mtpm = max_allowed_hzm / gamma * 1000.0

        for axis_short, axis_name in axes.items():
            peaks_attr = f'peaks_{axis_short}'
            spec_attr = f'spectra_{axis_short}'
            envelope_attr = f'max_envelope_{axis_short}'

            if not hasattr(spectra, peaks_attr) or not hasattr(spectra, spec_attr):
                continue

            peaks = getattr(spectra, peaks_attr)
            specs = getattr(spectra, spec_attr)
            envelope = getattr(spectra, envelope_attr)

            freq_indices = (
                (spectra.frequencies >= freq_min) & (spectra.frequencies <= freq_max)
            )

            if peaks is None:
                if specs.ndim == 1:
                    has_peaks_in_band = np.any(specs[freq_indices] > 0)
                else:
                    has_peaks_in_band = np.any(peaks[:, freq_indices] > 0)
            else:
                has_peaks_in_band = (
                    np.any(peaks[:, freq_indices] > 0)
                    if peaks.ndim == 2
                    else np.any(peaks[freq_indices] > 0)
                )

            if has_peaks_in_band:
                max_envelope_in_band = np.max(envelope[freq_indices])
                if max_envelope_in_band > max_allowed_mtpm:
                    warnings.warn(
                        f"Acoustic forbidden band violation: {axis_name} "
                        f"({freq_min:.1f}\u2013{freq_max:.1f} Hz) "
                        f"has peaks with envelope {max_envelope_in_band:.2f} mT/m "
                        f"(allowed: {max_allowed_mtpm:.2f} mT/m)",
                        UserWarning,
                    )

            # Sequence-harmonic check
            spec_seq = getattr(spectra, f'spectrum_{axis_short}_seq', None)
            peaks_seq = getattr(spectra, f'peaks_{axis_short}_seq', None)
            envelope_seq = getattr(spectra, f'max_envelope_{axis_short}_seq', None)

            if spec_seq is not None and peaks_seq is not None:
                freq_idx_seq = (
                    (spectra.frequencies_seq >= freq_min)
                    & (spectra.frequencies_seq <= freq_max)
                )
                has_peaks_seq = np.any(peaks_seq[freq_idx_seq] > 0)
                if has_peaks_seq and envelope_seq is not None:
                    max_env_seq = np.max(envelope_seq[freq_idx_seq])
                    if max_env_seq > max_allowed_mtpm:
                        warnings.warn(
                            f"Acoustic forbidden band violation (sequence spectrum): "
                            f"{axis_name} ({freq_min:.1f}\u2013{freq_max:.1f} Hz) "
                            f"has peaks with envelope {max_env_seq:.2f} mT/m "
                            f"(allowed: {max_allowed_mtpm:.2f} mT/m)",
                            UserWarning,
                        )
