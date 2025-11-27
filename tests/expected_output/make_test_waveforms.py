"""This scripts generates test waveforms for acoustic and PNS checks.

Contains the following test MR gradient waveforms:
- Balanced SSFP (bSSFP): a short TR sequence consisting of (phase prewinder, readout, rewinder)
- Spoiled Gradient Echo (SPGR): a short TR sequence similar to bSSFP, consisting of (prewinder, readout, rewinder, spoiler)
- Multi-Echo Gradient Echo (MEGRE): a medium TR sequence similar to SPGR, has multiple bipolar readouts: (prewinder, *(Nechoes * [readout, -readout]), rewinder, spoiler)
- Echo Planar imaging (EPI): a medium TR sequence similar to MEGRE, but phase blips are interleaved with readouts: (prewinder, *(Nechoes * [phase blip, readout]), rewinder, spoiler)
- Fast Spin Echo (FSE): a long TR spin-echo sequence with multiple refocusing pulses - similar to EPI, but stronger blips and larger spacing between readouts
- Magnetic Resonance Fingerprinting (MRF): a long TR sequence with variable flip angles and TRs, consisting of multiple segments of (preparation, *(Nshots * [readout, spoiler])) with spiral readout
- MPRAGE: a long TR sequence similar to MRF, but with cartesian reaout and a final spiral navigator shot before next TR: (preparation, *(Nshots * [prewinder, readout, rewinder, spoiler]), *spiral navigator)

"""

import struct
from pathlib import Path
from collections.abc import Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab

def write_gradients(path, gx, gy, gz):
    data = np.stack([gx, gy, gz], axis=1).astype(np.float32)
    samples, channels = data.shape
    with open(path, "wb") as fp:
        fp.write(struct.pack("<4i", samples, channels, 0, 0))
        fp.write(data.tobytes(order="C"))

def save_gradient_snapshot(out_path, gx, gy, gz, dt, n_trs=1, title=None):
    """Save a 3-row subplot (Gx, Gy, Gz) with TR markers spanning all rows."""
    base_gx = gx.copy()
    base_gy = gy.copy()
    base_gz = gz.copy()
    
    n_trs_int = max(1, int(n_trs))

    gx_plot = np.tile(base_gx, n_trs_int)
    gy_plot = np.tile(base_gy, n_trs_int)
    gz_plot = np.tile(base_gz, n_trs_int)

    total_samples = len(gx_plot)
    time = np.arange(total_samples, dtype=np.float32) * float(dt) * 1e3
    tr_duration_ms = total_samples / n_trs_int * float(dt) * 1e3

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 6), constrained_layout=True)
    gradients = [(gx_plot, "Gx"), (gy_plot, "Gy"), (gz_plot, "Gz")]
    for axis, (data, label) in zip(axes, gradients):
        axis.plot(time, data, linewidth=1.0, color="black")
        axis.set_ylabel(f"{label} (mT/m)")
        axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    axes[-1].set_xlabel("Time (ms)")
    if title:
        fig.suptitle(title)

    # Draw TR boundaries (start/end) as hairline red markers spanning all subplots
    for idx in range(n_trs_int):
        start_ms = idx * tr_duration_ms
        end_ms = (idx + 1) * tr_duration_ms
        for axis in axes:
            axis.axvline(start_ms, color="red", linewidth=0.8)
        if idx == n_trs_int - 1:
            for axis in axes:
                axis.axvline(end_ms, color="red", linewidth=0.8)

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_gradient_frequency_view(
    out_path,
    gx,
    gy,
    gz,
    dt,
    tr_samples=None,
    n_trs=1,
    mode="spectrum",
    nfft=2048,
    detrend=True,
    max_freq_hz=None,
    zero_pad_factor=1,
    apply_window=True,
    labels=None,
):
    """Save frequency-domain view for the RSS of gradient spectra.

    In ``mode="spectrum"`` this computes an RSS FFT of the requested TR tilings. A
    scalar ``n_trs`` > 1 automatically overlays the single-TR response with the
    multi-TR response so dominant peaks are easier to compare. In
    ``mode="spectrogram"`` it computes sliding-window spectra per channel, combines
    them via RSS, and plots the per-frequency maximum across windows.
    """
    if tr_samples is None:
        tr_samples = len(gx)
    if tr_samples <= 0:
        raise ValueError("tr_samples must be positive")

    mode = mode.lower()
    if mode not in {"spectrum", "spectrogram"}:
        raise ValueError("mode must be 'spectrum' or 'spectrogram'")

    base_gx = np.asarray(gx[:tr_samples], dtype=np.float32)
    base_gy = np.asarray(gy[:tr_samples], dtype=np.float32)
    base_gz = np.asarray(gz[:tr_samples], dtype=np.float32)

    if detrend:
        base_gx = base_gx - float(np.mean(base_gx))
        base_gy = base_gy - float(np.mean(base_gy))
        base_gz = base_gz - float(np.mean(base_gz))

    if apply_window:
        window = np.hanning(tr_samples).astype(np.float32)
        base_gx = base_gx * window
        base_gy = base_gy * window
        base_gz = base_gz * window

    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    sample_rate = 1.0 / float(dt)
    zero_pad_factor = max(1, int(zero_pad_factor))

    if mode == "spectrum":
        if isinstance(n_trs, Sequence) and not isinstance(n_trs, (str, bytes)):
            tr_counts = [max(1, int(value)) for value in n_trs]
        else:
            tr_scalar = max(1, int(n_trs))
            tr_counts = [1] if tr_scalar == 1 else [1, tr_scalar]

        seen = set()
        filtered_counts = []
        for count in tr_counts:
            if count not in seen:
                filtered_counts.append(count)
                seen.add(count)
        tr_counts = filtered_counts

        pending_labels = labels
        if pending_labels is not None and len(pending_labels) != len(tr_counts):
            raise ValueError("labels length must match number of TR counts")

        for idx, tr_int in enumerate(tr_counts):
            gx_concat = np.tile(base_gx, tr_int)
            gy_concat = np.tile(base_gy, tr_int)
            gz_concat = np.tile(base_gz, tr_int)

            if zero_pad_factor > 1:
                target_len = len(gx_concat) * zero_pad_factor
                pad_len = target_len - len(gx_concat)
                gx_fft = np.pad(gx_concat, (0, pad_len), mode="constant")
                gy_fft = np.pad(gy_concat, (0, pad_len), mode="constant")
                gz_fft = np.pad(gz_concat, (0, pad_len), mode="constant")
            else:
                gx_fft = gx_concat
                gy_fft = gy_concat
                gz_fft = gz_concat

            freq_axis = np.fft.rfftfreq(len(gx_fft), float(dt)) / 1e3  # kHz
            spectrum_x = np.fft.rfft(gx_fft)
            spectrum_y = np.fft.rfft(gy_fft)
            spectrum_z = np.fft.rfft(gz_fft)
            magnitude = np.sqrt(
                np.abs(spectrum_x) ** 2 + np.abs(spectrum_y) ** 2 + np.abs(spectrum_z) ** 2
            )
            if max_freq_hz is not None:
                max_freq_khz = float(max_freq_hz) / 1e3
                valid = freq_axis <= max_freq_khz
                freq_axis = freq_axis[valid]
                magnitude = magnitude[valid]

            if pending_labels is not None:
                label = pending_labels[idx]
            else:
                label = "Single TR" if tr_int == 1 else f"{tr_int} TRs"
            ax.plot(freq_axis, magnitude, linewidth=1.0, label=label)

        ax.set_ylabel("RSS |F| (a.u.)")
        ax.set_xlabel("Frequency (kHz)")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        if len(tr_counts) > 1:
            ax.legend()
    else:
        if isinstance(n_trs, Sequence) and not isinstance(n_trs, (str, bytes)):
            if len(n_trs) != 1:
                raise ValueError("spectrogram mode supports a single TR count")
            tr_value = n_trs[0]
        else:
            tr_value = n_trs
        tr_int = max(1, int(tr_value))
        gx_concat = np.tile(base_gx, tr_int)
        gy_concat = np.tile(base_gy, tr_int)
        gz_concat = np.tile(base_gz, tr_int)
        nfft = int(min(nfft, len(gx_concat)))
        noverlap = max(0, nfft // 2)
        power_x, freqs, _bins = mlab.specgram(
            gx_concat,
            NFFT=nfft,
            Fs=sample_rate,
            noverlap=noverlap,
        )
        power_y, _, _ = mlab.specgram(
            gy_concat,
            NFFT=nfft,
            Fs=sample_rate,
            noverlap=noverlap,
        )
        power_z, _, _ = mlab.specgram(
            gz_concat,
            NFFT=nfft,
            Fs=sample_rate,
            noverlap=noverlap,
        )
        rss_power = power_x + power_y + power_z
        rss_mag = np.sqrt(np.maximum(rss_power, 1e-20))
        rss_max = rss_mag.max(axis=1)
        if max_freq_hz is not None:
            valid = freqs <= float(max_freq_hz)
            freqs = freqs[valid]
            rss_max = rss_max[valid]
        ax.plot(freqs / 1e3, rss_max, linewidth=1.0, color="black")
        ax.set_ylabel("RSS max |F| (a.u.)")
        ax.set_xlabel("Frequency (kHz)")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)


    ax.set_title("Gradient {} (RSS)".format("Spectrum" if mode == "spectrum" else "Spectrogram"))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_bssfp_waveform(export=False):
    """Generate a simple BSSFP-like gradient waveform (single TR)."""
    gdt = 4e-6  # Gradient update time (s)
    rf_dur = 1.0e-3  # RF pulse duration (s)

    # Individual gradient lobes
    samples_phase = int(0.5e-3 / gdt)
    samples_read = int(1.0e-3 / gdt)
    gph = 30.0 * np.ones(samples_phase, dtype=np.float32)  # 0.5 ms phase encoding; 30 mT/m
    gread = 30.0 * np.ones(samples_read, dtype=np.float32)  # 1.0 ms readout; 30 mT/m

    # Build y and z phase encoding and x readout gradients
    gy = np.concatenate((-gph, np.zeros_like(gread), gph))
    gz = gy.copy()
    gx = np.concatenate((-gph, gread, -gph))

    # Pad with zeros to account for RF pulse duration
    rf = np.zeros(int(rf_dur / gdt), dtype=np.float32)
    gx = np.concatenate((rf, gx))
    gy = np.concatenate((rf, gy))
    gz = np.concatenate((rf, gz))

    gx = np.asarray(gx, dtype=np.float32)
    gy = np.asarray(gy, dtype=np.float32)
    gz = np.asarray(gz, dtype=np.float32)

    # Export waveforms and visualizations if requested
    if export:
        export_dir = Path(__file__).resolve().parent
        export_dir.mkdir(parents=True, exist_ok=True)
        snapshot_trs = 3
        multi_tr_spectrum_count = 256*256
        max_freq_hz = 2000.0 # Gz
        spectrum_zero_pad = 4
        write_gradients(export_dir / "bssfp_waveform.dat", gx, gy, gz)
        save_gradient_snapshot(
            export_dir / "bssfp_waveform.png",
            gx,
            gy,
            gz,
            gdt,
            n_trs=snapshot_trs,
            title="BSSFP Gradient Waveform",
        )
        save_gradient_frequency_view(
            export_dir / "bssfp_waveform_spectrum.png",
            gx,
            gy,
            gz,
            gdt,
            mode="spectrum",
            max_freq_hz=max_freq_hz,
            zero_pad_factor=spectrum_zero_pad,
            n_trs=multi_tr_spectrum_count,
        )

    return gx, gy, gz

def make_spgr_waveform(export=False):
    ...
    
def make_megre_waveform(export=False):
    ...
    
def make_epi_waveform(export=False):
    ...
    
def make_fse_waveform(export=False):
    ...
    
def make_mrf_waveform(export=False):
    ...
    
def make_mprage_waveform(export=False):
    ...
    
def main():
    make_bssfp_waveform(export=True)


if __name__ == "__main__":
    main()
