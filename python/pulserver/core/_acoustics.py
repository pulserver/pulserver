"""Gradient spectrum visualisation for SequenceCollection (plotting only, no pass/fail)."""

__all__ = ['grad_spectrum']

import warnings

import numpy as np
from matplotlib.colors import Normalize

from ._extension._pulseqlib_wrapper import _calc_acoustic_spectra
from ._helpers import _add_echo_spacing_axis
from ._sequence import SequenceCollection


def grad_spectrum(
    seq: SequenceCollection,
    *,
    sequence_idx: int = 0,
    forbidden_bands: list[tuple[float, float, float]] | None = None,
    window_duration: float = 25.0e-3,
    spectral_resolution: float = 5.0,
    max_frequency: float = 3000.0,
) -> None:
    """Plot acoustic spectra for gradient waveforms in a TR.

    Creates a two-row figure:

    * **Top row** — three sliding-window spectrograms (Gx, Gy, Gz).
    * **Bottom row** — full-TR harmonic spectrum (max envelope across
      windows) with forbidden-band overlays.

    No pass/fail check is performed — use
    :meth:`SequenceCollection.check` for that.

    Parameters
    ----------
    seq : SequenceCollection
        The sequence to analyse.
    sequence_idx : int
        Subsequence index (0-based, default 0).
    forbidden_bands : list of (freq_min, freq_max, max_amplitude), optional
        Forbidden frequency bands.  Each tuple gives
        ``(freq_min_Hz, freq_max_Hz, max_allowed_amplitude_Hz_per_m)``.
        Drawn as shaded regions on the harmonic plot.
    window_duration : float
        Sliding-window size in seconds (default 25 ms).
    spectral_resolution : float
        Target frequency resolution in Hz (default 5 Hz).
    max_frequency : float
        Upper frequency limit in Hz (default 3000 Hz).
    """
    import matplotlib.pyplot as plt

    if forbidden_bands is None:
        forbidden_bands = []

    grad_raster_time = seq.system.grad_raster_time
    target_window_size = int(2.0 * window_duration / grad_raster_time)

    # Call C extension
    rd = _calc_acoustic_spectra(
        seq._cseq,
        subsequence_idx=sequence_idx,
        target_window_size=target_window_size,
        target_resolution_hz=spectral_resolution,
        max_freq_hz=max_frequency,
        forbidden_bands=forbidden_bands,
    )

    num_windows = rd["num_windows"]
    num_freq_bins = rd["num_freq_bins"]
    frequencies = rd["freq_min_hz"] + np.arange(num_freq_bins) * rd["freq_spacing_hz"]

    # Reshape spectrograms (flat -> 2-D)
    spectrograms = {}
    peaks = {}
    for ax_name in ('gx', 'gy', 'gz'):
        spectrograms[ax_name] = np.asarray(
            rd[f"spectrogram_{ax_name}"], dtype=np.float32,
        ).reshape(num_windows, num_freq_bins)
        peaks[ax_name] = np.asarray(
            rd[f"peaks_{ax_name}"], dtype=np.int32,
        ).reshape(num_windows, num_freq_bins)

    # Full-TR spectra
    spectrum_full = {}
    peaks_full = {}
    for ax_name in ('gx', 'gy', 'gz'):
        spectrum_full[ax_name] = np.asarray(
            rd[f"spectrum_full_{ax_name}"], dtype=np.float32,
        )
        peaks_full[ax_name] = np.asarray(
            rd[f"peaks_full_{ax_name}"], dtype=np.int32,
        )

    freq_min = 0.0
    freq_max = float(frequencies[-1])

    axis_labels = {'gx': 'Gx', 'gy': 'Gy', 'gz': 'Gz'}
    colors = {'gx': 'C0', 'gy': 'C1', 'gz': 'C2'}

    # ── Two-row figure: spectrograms (top), harmonics (bottom) ──
    fig, axes = plt.subplots(
        2, 3, figsize=(16, 8),
        gridspec_kw={'height_ratios': [1, 1]},
    )

    # ── Top row: sliding-window spectrograms ──────────────────
    for col, ax_name in enumerate(('gx', 'gy', 'gz')):
        ax = axes[0, col]
        sg = spectrograms[ax_name]
        pk = peaks[ax_name]

        im = ax.pcolormesh(
            frequencies, np.arange(num_windows), sg,
            cmap='viridis', shading='auto',
            norm=Normalize(vmin=sg.min(), vmax=sg.max()),
        )

        # Mark peaks
        peak_coords = np.where(pk > 0)
        if len(peak_coords[0]) > 0:
            ax.plot(
                frequencies[peak_coords[1]], peak_coords[0],
                marker='*', color='red', linestyle='none',
                markersize=10, markeredgewidth=0,
            )

        # Forbidden bands as vertical lines
        for band in forbidden_bands:
            ax.axvline(band[0], color='white', ls='--', lw=1.2, alpha=0.8)
            ax.axvline(band[1], color='white', ls='--', lw=1.2, alpha=0.8)

        ax.set_xlim(freq_min, freq_max)
        ax.set_ylim(-0.5, num_windows - 0.5)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Window Index')
        ax.set_title(f'{axis_labels[ax_name]} Spectrogram')
        _add_echo_spacing_axis(ax, freq_min, freq_max)

    # ── Bottom row: full-TR harmonic spectra + forbidden bands ──
    for col, ax_name in enumerate(('gx', 'gy', 'gz')):
        ax = axes[1, col]
        color = colors[ax_name]

        # Max envelope across windows
        max_env = spectrograms[ax_name].max(axis=0)
        ax.plot(frequencies, max_env, color=color, lw=1.5,
                label='Max envelope')

        # Full-TR spectrum
        ax.plot(frequencies, spectrum_full[ax_name], color=color,
                lw=0.8, alpha=0.5, label='Full TR')

        # Mark full-TR peaks
        peak_mask = peaks_full[ax_name] > 0
        if np.any(peak_mask):
            ax.plot(
                frequencies[peak_mask],
                spectrum_full[ax_name][peak_mask],
                'o', color=color, markersize=5, markeredgewidth=0,
            )

        # Forbidden bands as shaded regions
        for band in forbidden_bands:
            ax.axvspan(
                band[0], band[1],
                alpha=0.15, color='red', zorder=0,
            )
            # Threshold line within band
            band_mask = (frequencies >= band[0]) & (frequencies <= band[1])
            if np.any(band_mask):
                ax.hlines(
                    band[2], band[0], band[1],
                    colors='red', linestyles='--', linewidth=1.2,
                )

        ax.set_xlim(freq_min, freq_max)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude (a.u.)')
        ax.set_title(f'{axis_labels[ax_name]} Harmonic Spectrum')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        _add_echo_spacing_axis(ax, freq_min, freq_max)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fig.tight_layout()
