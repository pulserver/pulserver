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

_eps = 1e-20


def detrend_signal(sig):
    return sig - sig.mean()


def filter_signal(sig):
    win = np.hanning(sig.size).astype(np.float32)
    return win * sig


def get_rss_spectrum(sig1, sig2, sig3, dt, fmax, M_desired):
    if M_desired < 2:
        raise ValueError("M_desired must be >= 2 (to include DC and fmax).")
    fs = 1.0 / dt
    nyq = fs / 2.0
    if fmax > nyq:
        raise ValueError(f"fmax must be <= Nyquist ({nyq}).")

    # compute target df assuming inclusive endpoints 0..fmax with M_desired bins
    df_target = fmax / (M_desired - 1)

    # required Nfft to achieve df <= df_target: Nfft >= 1/(dt*df_target)
    Nfft_req = int(np.ceil(1.0 / (dt * df_target)))

    # ensure Nfft at least as large as longest signal
    N_orig = max(len(sig1), len(sig2), len(sig3))
    Nfft = max(Nfft_req, N_orig)

    # actual df and kmax for fmax
    df = 1.0 / (dt * Nfft)
    kmax = int(np.floor(fmax / df))

    # rfft produces Nfft//2 + 1 bins; ensure kmax does not exceed available bins
    max_bin = Nfft // 2
    if kmax > max_bin:
        kmax = max_bin

    # zero-pad (or leave if equal) and rfft
    def rmag(x):
        x = np.asarray(x, dtype=float)
        if x.size < Nfft:
            x = np.concatenate([x, np.zeros(Nfft - x.size, dtype=float)])
        # rfft -> bins 0..Nfft//2
        X = np.fft.rfft(x, n=Nfft)
        return np.abs(X[: kmax + 1])

    S1 = rmag(sig1)
    S2 = rmag(sig2)
    S3 = rmag(sig3)

    # frequency vector for bins 0..kmax
    freqs = np.arange(kmax + 1) * df

    # RSS across the three magnitude spectra (before any PSD normalization/rescaling)
    rss = (S1**2 + S2**2 + S3**2) ** 0.5

    return freqs, rss


def refer_rss_to_delta(rss):
    rss = np.asarray(rss, dtype=float)
    if rss.size == 0:
        return rss.copy(), 0.0

    power = rss**2
    total = power.sum()
    if total <= 0:
        return np.zeros_like(rss)

    p = power / total
    N = p.size
    uniform = 1.0 / N
    denom = 1.0 - uniform if N > 1 else 1.0

    norm_per_bin = (p - uniform) / denom

    # clip to [0,1] so small-than-uniform bins map to 0, perfect-delta -> 1
    norm_per_bin = np.clip(norm_per_bin, 0.0, 1.0)

    return norm_per_bin


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
    threshold=0.3,
    analytic_multi_tr=False,
    title=None,
):
    """Save frequency-domain view for the RSS of gradient spectra.

    In ``mode="spectrum"`` this computes an RSS FFT of the requested TR tilings. A
    scalar ``n_trs`` > 1 automatically overlays the single-TR response with the
    multi-TR response so dominant peaks are easier to compare. In
    ``mode="spectrogram"`` it computes sliding-window spectra per channel, combines
    them via RSS, and plots the per-frequency maximum across windows.

    The first subplot always shows the normalized (peak == 1) RSS magnitude so that
    single- and multi-TR spectra share the same vertical scale. The second subplot
    visualises the guard metric compared against the supplied ``threshold``: single-TR
    traces use the band-energy fraction (relative area), while multi-TR traces use the
    normalized peak amplitude. Points exceeding the threshold are highlighted with
    crosses: black for the single-TR trace, red for multi-TR traces.

    When ``analytic_multi_tr`` is true, spectra for ``tr_count > 1`` are generated by
    reusing the single-TR FFT and applying the analytic replication gain instead of
    explicitly concatenating the waveform.
    """
    mode = mode.lower()
    if mode not in {"spectrum", "spectrogram"}:
        raise ValueError("mode must be 'spectrum' or 'spectrogram'")

    base_gx = gx.astype(np.float32)
    base_gy = gy.astype(np.float32)
    base_gz = gz.astype(np.float32)

    if detrend:
        base_gx = detrend_signal(base_gx)
        base_gy = detrend_signal(base_gy)
        base_gz = detrend_signal(base_gz)

    if apply_window:
        base_gx = filter_signal(base_gx)
        base_gy = filter_signal(base_gy)
        base_gz = filter_signal(base_gz)

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 6), constrained_layout=True)
    amp_ax, ratio_ax = axes
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

        base_len = len(base_gx)
        base_fft_x = np.fft.rfft(base_gx)
        base_fft_y = np.fft.rfft(base_gy)
        base_fft_z = np.fft.rfft(base_gz)
        base_power = (
            np.abs(base_fft_x) ** 2 + np.abs(base_fft_y) ** 2 + np.abs(base_fft_z) ** 2
        )
        total_power = float(np.sum(base_power))

        datasets = []
        color_cycle = ["black", "red", "tab:blue", "tab:green", "tab:orange"]

        for idx, tr_int in enumerate(tr_counts):
            if tr_int > 1 and analytic_multi_tr:
                fft_len = base_len * tr_int
                freq_axis = np.fft.rfftfreq(fft_len, float(dt)) / 1e3
                power = np.zeros_like(freq_axis, dtype=np.float64)
                indices = np.arange(freq_axis.size)
                mask = (indices % tr_int) == 0
                base_indices = indices[mask] // tr_int
                power[mask] = (tr_int * tr_int) * base_power[base_indices]
                magnitude = np.sqrt(power)
                if magnitude.size > 0:
                    magnitude[0] = 0.0
            else:
                gx_concat = np.tile(base_gx, tr_int)
                gy_concat = np.tile(base_gy, tr_int)
                gz_concat = np.tile(base_gz, tr_int)

                if zero_pad_factor > 1 and tr_int == 1:
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
                power = (
                    np.abs(spectrum_x) ** 2
                    + np.abs(spectrum_y) ** 2
                    + np.abs(spectrum_z) ** 2
                )
                magnitude = np.sqrt(power)
                if tr_int > 1 and magnitude.size > 0:
                    magnitude[0] = 0.0
            if max_freq_hz is not None:
                max_freq_khz = float(max_freq_hz) / 1e3
                valid = freq_axis <= max_freq_khz
                freq_axis = freq_axis[valid]
                magnitude = magnitude[valid]
                power = power[valid]

            if pending_labels is not None:
                label = pending_labels[idx]
            else:
                label = "Single TR" if tr_int == 1 else f"{tr_int} TRs"
            color = color_cycle[idx % len(color_cycle)]
            marker_color = "black" if tr_int == 1 else "red"
            datasets.append(
                {
                    "label": label,
                    "freq": freq_axis,
                    "magnitude": magnitude,
                    "power": power,
                    "color": color,
                    "marker": marker_color,
                    "tr_count": tr_int,
                }
            )

        amp_handles = []
        ratio_handles = []
        ratio_labels = []
        for data in datasets:
            freq_axis = data["freq"]
            magnitude = data["magnitude"]
            power = data["power"]
            color = data["color"]
            label = data["label"]
            marker_color = data["marker"]
            tr_count = data["tr_count"]

            dataset_max = float(np.max(magnitude))
            if dataset_max <= 0.0:
                dataset_max = 1.0
            magnitude_norm = magnitude / dataset_max
            (line_handle,) = amp_ax.plot(
                freq_axis, magnitude_norm, linewidth=1.0, color=color, label=label
            )
            amp_handles.append(line_handle)

            if tr_count == 1:
                power_for_guard = power.copy() if power.size > 0 else power
                if total_power <= 0.0:
                    guard_values = np.zeros_like(power_for_guard)
                else:
                    guard_values = power_for_guard / total_power
                guard_label = f"{label} (energy)"
            else:
                guard_values = magnitude_norm
                if guard_values.size > 0:
                    guard_values = guard_values.copy()
                    guard_values[0] = 0.0
                guard_label = f"{label} (peak)"
            (ratio_line,) = ratio_ax.plot(
                freq_axis, guard_values, linewidth=1.0, color=color
            )
            ratio_handles.append(ratio_line)
            ratio_labels.append(guard_label)

            exceed = guard_values > threshold
            if np.any(exceed):
                amp_ax.scatter(
                    freq_axis[exceed],
                    magnitude_norm[exceed],
                    marker="x",
                    color=marker_color,
                    s=36,
                    linewidths=1.1,
                    zorder=3,
                )

        if len(amp_handles) > 1:
            amp_ax.legend(loc="upper right")

        threshold_handle = ratio_ax.axhline(
            threshold,
            color="gray",
            linestyle="--",
            linewidth=1.0,
        )
        ratio_handles.append(threshold_handle)
        ratio_labels.append(f"Threshold {threshold:.2f}")
        ratio_ax.legend(ratio_handles, ratio_labels, loc="upper right")

        amp_ax.set_ylabel("Normalized RSS |F|")
        ratio_ax.set_ylabel("Guard metric")
        ratio_ax.set_xlabel("Frequency (kHz)")
        for axis in axes:
            axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        amp_ax.set_ylim(0.0, 1.05)
        ratio_ax.set_ylim(0.0, 1.0)
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
        freqs_khz = freqs / 1e3
        max_val = float(np.max(rss_max))
        if max_val <= 0.0:
            max_val = 1.0
        rss_norm = rss_max / max_val
        if rss_norm.size > 0:
            rss_norm = rss_norm.copy()
            rss_norm[0] = 0.0
        amp_ax.plot(freqs_khz, rss_norm, linewidth=1.0, color="black", label="Max RSS")

        guard_values = rss_norm
        (ratio_line,) = ratio_ax.plot(
            freqs_khz,
            guard_values,
            linewidth=1.0,
            color="black",
            label="Max RSS (peak)",
        )

        exceed = guard_values > threshold
        if np.any(exceed):
            amp_ax.scatter(
                freqs_khz[exceed],
                rss_norm[exceed],
                marker="x",
                color="black",
                s=36,
                linewidths=1.1,
                zorder=3,
            )

        threshold_handle = ratio_ax.axhline(
            threshold,
            color="gray",
            linestyle="--",
            linewidth=1.0,
        )
        ratio_ax.legend(
            [ratio_line, threshold_handle],
            ["Max RSS (peak)", f"Threshold {threshold:.2f}"],
            loc="upper right",
        )

        amp_ax.set_ylabel("Normalized RSS |F|")
        ratio_ax.set_ylabel("Guard metric")
        ratio_ax.set_xlabel("Frequency (kHz)")
        for axis in axes:
            axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        amp_ax.set_ylim(0.0, 1.05)
        ratio_ax.set_ylim(0.0, 1.0)

    if title is None:
        amp_ax.set_title(
            "Gradient {} (RSS)".format(
                "Spectrum" if mode == "spectrum" else "Spectrogram"
            )
        )
    else:
        amp_ax.set_title(title)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_bssfp_waveform(export=False):
    """Generate a simple bSSFP-like gradient waveform (single TR)."""
    gdt = 4e-6  # Gradient update time (s)
    rf_dur = 1.0e-3  # RF pulse duration (s)

    # Individual gradient lobes
    samples_phase = int(0.5e-3 / gdt)
    samples_read = int(1.0e-3 / gdt)
    gph = 30.0 * np.ones(
        samples_phase, dtype=np.float32
    )  # 0.5 ms phase encoding; 30 mT/m
    gread = 30.0 * np.ones(samples_read, dtype=np.float32)  # 1.0 ms readout; 30 mT/m

    # Build y and z phase encoding and x readout gradients
    gy = np.concatenate((gph, np.zeros_like(gread), -gph))
    gz = gy.copy()
    gx = np.concatenate((-gph, gread, -gph))

    # Pad with zeros to account for RF pulse duration
    rf = np.zeros(int(rf_dur / gdt), dtype=np.float32)
    gx = np.concatenate((rf, gx))
    gy = np.concatenate((rf, gy))
    gz = np.concatenate((rf, gz))

    # Export waveforms and visualizations if requested
    if export:
        export_dir = Path(__file__).resolve().parent
        export_dir.mkdir(parents=True, exist_ok=True)
        snapshot_trs = 3
        multi_tr_spectrum_count = 256**2
        max_freq_hz = 2000.0  # Hz
        spectrum_zero_pad = 4
        write_gradients(export_dir / "bssfp_waveform.dat", gx, gy, gz)
        save_gradient_snapshot(
            export_dir / "bssfp_waveform.png",
            gx,
            gy,
            gz,
            gdt,
            n_trs=snapshot_trs,
            title="bSSFP Gradient Waveform",
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
            analytic_multi_tr=True,
            title="bSSFP Gradient Spectrum (RSS)",
        )

    return gx, gy, gz


def make_spgr_waveform(export=False):
    """Generate a simple SPGR-like gradient waveform (single TR)."""
    gdt = 4e-6  # Gradient update time (s)

    # Spoiler
    samples_spoiler = int(8e-3 // gdt)
    gspoil = 35.0 * np.ones(samples_spoiler, dtype=np.float32)

    # Get base waveform
    gx, gy, gz = make_bssfp_waveform()

    # Concatenate
    gx = np.concatenate((gx, gspoil))
    gy = np.concatenate((gy, gspoil))
    gz = np.concatenate((gz, gspoil))

    # Export waveforms and visualizations if requested
    if export:
        export_dir = Path(__file__).resolve().parent
        export_dir.mkdir(parents=True, exist_ok=True)
        snapshot_trs = 3
        multi_tr_spectrum_count = 256**2
        max_freq_hz = 2000.0  # Hz
        spectrum_zero_pad = 4
        write_gradients(export_dir / "spgr_waveform.dat", gx, gy, gz)
        save_gradient_snapshot(
            export_dir / "spgr_waveform.png",
            gx,
            gy,
            gz,
            gdt,
            n_trs=snapshot_trs,
            title="SPGR Gradient Waveform",
        )
        save_gradient_frequency_view(
            export_dir / "spgr_waveform_spectrum.png",
            gx,
            gy,
            gz,
            gdt,
            mode="spectrum",
            max_freq_hz=max_freq_hz,
            zero_pad_factor=spectrum_zero_pad,
            n_trs=multi_tr_spectrum_count,
            analytic_multi_tr=True,
            title="SPGR Gradient Spectrum (RSS)",
        )

    return gx, gy, gz


def make_megre_waveform(export=False):
    gdt = 4e-6  # Gradient update time (s)
    rf_dur = 1.0e-3  # RF pulse duration (s)
    nechoes = 13

    # Individual gradient lobes
    samples_phase = int(0.5e-3 / gdt)
    samples_read = int(1.0e-3 / gdt)
    samples_spoiler = int(8e-3 // gdt)
    gph = 30.0 * np.ones(
        samples_phase, dtype=np.float32
    )  # 0.5 ms phase encoding; 30 mT/m
    gread = 30.0 * np.ones(samples_read, dtype=np.float32)  # 1.0 ms readout; 30 mT/m
    gspoil = 35.0 * np.ones(samples_spoiler, dtype=np.float32)

    # Build y and z phase encoding and x readout gradients
    gy = np.concatenate((gph, np.zeros(samples_read * nechoes, dtype=np.float32), -gph))
    gz = gy.copy()
    gx = np.concatenate((-gph, *((nechoes - 1) // 2 * [gread, -gread]), gread, -gph))

    # Pad with zeros to account for RF pulse duration
    rf = np.zeros(int(rf_dur / gdt), dtype=np.float32)
    gx = np.concatenate((rf, gx, gspoil))
    gy = np.concatenate((rf, gy, gspoil))
    gz = np.concatenate((rf, gz, gspoil))

    # Export waveforms and visualizations if requested
    if export:
        export_dir = Path(__file__).resolve().parent
        export_dir.mkdir(parents=True, exist_ok=True)
        snapshot_trs = 3
        multi_tr_spectrum_count = 256**2
        max_freq_hz = 2000.0  # Hz
        spectrum_zero_pad = 4
        write_gradients(export_dir / "megre_waveform.dat", gx, gy, gz)
        save_gradient_snapshot(
            export_dir / "megre_waveform.png",
            gx,
            gy,
            gz,
            gdt,
            n_trs=snapshot_trs,
            title="ME-GRE Gradient Waveform",
        )
        save_gradient_frequency_view(
            export_dir / "megre_waveform_spectrum.png",
            gx,
            gy,
            gz,
            gdt,
            mode="spectrum",
            max_freq_hz=max_freq_hz,
            zero_pad_factor=spectrum_zero_pad,
            n_trs=multi_tr_spectrum_count,
            analytic_multi_tr=True,
            title="ME-GRE Gradient Spectrum (RSS)",
        )

    return gx, gy, gz


def make_epi_waveform(export=False): ...


def make_fse_waveform(export=False): ...


def make_mrf_waveform(export=False): ...


def make_mprage_waveform(export=False): ...


def main():
    make_bssfp_waveform(export=True)
    make_spgr_waveform(export=True)
    make_megre_waveform(export=True)


if __name__ == "__main__":
    main()
