"""Compatibility helpers for the staged FSE sequence plugins.

These functions preserve the pre-namespace-refactor procedural API.  New
design code should construct FSE modules through ``make_fse_readout``.
"""

from __future__ import annotations

import math

import numpy as np
import pypulseq as pp

from .._system import quantize_readout_timing, round_to_raster

RF_REFOCUS_TIME_S = 3.0e-3
RF_REFOCUS_APODIZATION = 0.5
RF_REFOCUS_TIME_BW_PRODUCT = 4.0
CPMG_PHASE_OFFSET_RAD = 1.5707963267948966
DEFAULT_REFOCUS_FLIP_DEG = 180.0
EXCITATION_FLIP_DEG = 90.0

_TRAPS_OPT_EX1 = 2.0
_TRAPS_OPT_CA = 0.4
_TRAPS_OPT_CB = 2.0
_TRAPS_OPT_CC = 2.0


def build_refocusing_pulse(opts: pp.Opts, slice_thickness_m: float):
    """Build the reference ``(rf180, gz180)`` CPMG refocusing pulse."""
    rf180, gz180, _ = pp.make_sinc_pulse(
        flip_angle=3.14159265358979,
        duration=RF_REFOCUS_TIME_S,
        slice_thickness=slice_thickness_m,
        apodization=RF_REFOCUS_APODIZATION,
        time_bw_product=RF_REFOCUS_TIME_BW_PRODUCT,
        system=opts,
        use="refocusing",
        return_gz=True,
    )
    rf180.phase_offset = CPMG_PHASE_OFFSET_RAD
    return rf180, gz180


def scale_refocusing_pulse(rf180_ref, flip_deg: float):
    """Return a copy of the reference pulse scaled to ``flip_deg``."""
    rf_scaled = type(rf180_ref)()
    rf_scaled.__dict__.update(rf180_ref.__dict__)
    rf_scaled.signal = rf180_ref.signal * (flip_deg / 180.0)
    return rf_scaled


def build_refocus_flip_schedule(etl: int, refocus_flip_deg: float, variable: bool) -> list[float]:
    """Return the legacy per-echo TRAPS refocusing schedule in degrees."""
    if not variable or etl <= 1:
        return [refocus_flip_deg] * etl

    rf1 = (
        90.0
        + refocus_flip_deg / 2.0
        + _TRAPS_OPT_CA
        * ((_TRAPS_OPT_CC - 1.0) / _TRAPS_OPT_CC) ** _TRAPS_OPT_EX1
        * (90.0 - refocus_flip_deg / 2.0)
    )
    dr = rf1 - refocus_flip_deg
    schedule = [rf1]
    for k in range(2, etl + 1):
        schedule.append(refocus_flip_deg + dr / (_TRAPS_OPT_CB ** (k - 0.5)))
    return schedule


def fse_timing(esp_s, d90_s, c90_s, d180_s, c180_s, crusher_dur_s, half_loop_fixed_s, strict):
    """Return ``(tau_first_s, tau_steady_s)`` for the CPMG train."""
    half_esp_s = esp_s / 2.0
    tau_first_s = round_to_raster(half_esp_s - (d90_s - c90_s) - crusher_dur_s - c180_s)
    tau_steady_s = round_to_raster(half_esp_s - (d180_s - c180_s) - crusher_dur_s - half_loop_fixed_s)
    if (tau_first_s < -1e-9 or tau_steady_s < -1e-9) and strict:
        return None
    return max(tau_first_s, 0.0), max(tau_steady_s, 0.0)


def surgery_ramp_time_s(opts: pp.Opts, amplitudes_hz_m) -> float:
    """Return a raster-ceiled slew-feasible ramp duration."""
    max_amp = max((abs(a) for a in amplitudes_hz_m), default=0.0)
    raster = opts.grad_raster_time
    if max_amp <= 0.0 or opts.max_slew <= 0.0:
        return raster
    dg = max_amp / opts.max_slew
    return math.ceil(dg / raster - 1e-9) * raster


def _fuse(channel: str, times, amplitudes):
    return pp.make_extended_trapezoid(channel=channel, times=np.asarray(times), amplitudes=np.asarray(amplitudes))


def _bridge(opts: pp.Opts, channel: str, area: float, grad_start: float, grad_end: float, target_duration_s: float):
    """Bridge two amplitudes with an exact area and duration."""
    _, times_min, _ = pp.make_extended_trapezoid_area(
        area=area, channel=channel, grad_start=grad_start, grad_end=grad_end, system=opts
    )
    times_min = [float(t) for t in times_min]
    min_duration = times_min[-1]
    if target_duration_s < min_duration - 1e-9:
        raise ValueError(
            f"target duration {target_duration_s * 1e6:.1f} us is shorter than the shortest feasible "
            f"{channel}-bridge ({min_duration * 1e6:.1f} us) for area={area:.3f}"
        )

    if len(times_min) == 3:
        rise_s = times_min[1] - times_min[0]
        fall_s = times_min[2] - times_min[1]
    else:
        rise_s = times_min[1] - times_min[0]
        fall_s = times_min[3] - times_min[2]

    flat_s = max(0.0, target_duration_s - rise_s - fall_s)
    denom = 0.5 * rise_s + flat_s + 0.5 * fall_s
    if denom <= 0.0:
        raise ValueError(f"degenerate {channel}-bridge duration for area={area:.3f}")
    plateau = (area - 0.5 * rise_s * grad_start - 0.5 * fall_s * grad_end) / denom

    if abs(plateau) > opts.max_grad + 1e-6:
        raise ValueError(f"re-solved {channel}-bridge plateau {plateau:.1f} exceeds max_grad")
    if rise_s > 0.0 and abs(plateau - grad_start) / rise_s > opts.max_slew * (1.0 + 1e-6):
        raise ValueError(f"re-solved {channel}-bridge rise leg exceeds max_slew")
    if fall_s > 0.0 and abs(grad_end - plateau) / fall_s > opts.max_slew * (1.0 + 1e-6):
        raise ValueError(f"re-solved {channel}-bridge fall leg exceeds max_slew")

    if flat_s > 0.0:
        times = [0.0, rise_s, rise_s + flat_s, rise_s + flat_s + fall_s]
        amplitudes = [grad_start, plateau, plateau, grad_end]
    else:
        times = [0.0, rise_s, rise_s + fall_s]
        amplitudes = [grad_start, plateau, grad_end]
    return _fuse(channel, times, amplitudes)


def build_slice_surgery_chain(
    opts, gz_ex_amp, gz_ref_amp, crusher_area, t_ex_wd_s, t_ref_wd_s, t_spex_s, t_sp_s, dg_s
):
    """Build the legacy Z-channel slice-select/crusher surgery chain."""
    gs1 = _fuse("z", [0.0, dg_s], [0.0, gz_ex_amp])
    gs2 = _fuse("z", [0.0, t_ex_wd_s], [gz_ex_amp, gz_ex_amp])
    gs2_fall = _fuse("z", [0.0, dg_s], [gz_ex_amp, 0.0])
    gs3 = _bridge(opts, "z", crusher_area, 0.0, gz_ref_amp, t_spex_s)
    gs4 = _fuse("z", [0.0, t_ref_wd_s], [gz_ref_amp, gz_ref_amp])
    gs5 = _bridge(opts, "z", crusher_area, gz_ref_amp, 0.0, t_sp_s)
    gs7 = _bridge(opts, "z", crusher_area, 0.0, gz_ref_amp, t_sp_s)
    return {"GS1": gs1, "GS2": gs2, "GS2_FALL": gs2_fall, "GS3": gs3, "GS4": gs4, "GS5": gs5, "GS7": gs7}


def build_kz_crusher_bridge(opts, gz_ref_amp, crusher_area, kz_area, t_sp_s):
    """Build 3D GS5/GS7 bridges combining crushers and partition encoding."""
    gs5 = _bridge(opts, "z", crusher_area + kz_area, gz_ref_amp, 0.0, t_sp_s)
    gs7 = _bridge(opts, "z", crusher_area - kz_area, 0.0, gz_ref_amp, t_sp_s)
    return gs5, gs7


def build_surgery_readout(opts, ro_axis, nx_ro, fov_ro_m, bandwidth_hz_px):
    """Build the plateau and ADC for the legacy gradient-surgery FSE."""
    delta_k_ro = 1.0 / fov_ro_m
    readout_area = nx_ro * delta_k_ro
    dwell_s, flat_time_s = quantize_readout_timing(
        nx_ro=nx_ro,
        target_bw_hz_px=bandwidth_hz_px,
        grad_raster_s=opts.grad_raster_time,
        adc_raster_s=opts.adc_raster_time,
        min_flat_time_s=abs(readout_area) / (0.95 * opts.max_grad),
    )
    gx_flat_amp = readout_area / flat_time_s
    adc = pp.make_adc(num_samples=nx_ro, dwell=dwell_s, delay=0.0, system=opts)
    return {
        "adc": adc,
        "readout_area": readout_area,
        "gx_flat_amp": gx_flat_amp,
        "flat_time_s": flat_time_s,
        "adc_center_s": 0.5 * flat_time_s,
    }


def build_readout_surgery_chain(opts, ro_axis, readout_area, gx_flat_amp, flat_time_s, t_sp_s):
    """Build the legacy readout prephase/rephase surgery chain."""
    prephase_area = -0.5 * readout_area
    gr5 = _bridge(opts, ro_axis, prephase_area, 0.0, gx_flat_amp, t_sp_s)
    gr6 = _fuse(ro_axis, [0.0, flat_time_s], [gx_flat_amp, gx_flat_amp])
    gr7 = _bridge(opts, ro_axis, prephase_area, gx_flat_amp, 0.0, t_sp_s)
    return {"GR5": gr5, "GR6": gr6, "GR7": gr7}


def surgery_timing(esp_s, t_ex_wd_s, c90_s, dg_s, d_gz_reph_s, t_ref_wd_s, c180_s, adc_center_s, strict):
    """Return fused FSE surgery segment durations."""
    half_esp_s = esp_s / 2.0
    t_spex_s = round_to_raster(half_esp_s - (t_ex_wd_s - c90_s) - dg_s - d_gz_reph_s - c180_s)
    t_sp_s = round_to_raster(half_esp_s - (t_ref_wd_s - c180_s) - adc_center_s)
    if (t_spex_s < -1e-9 or t_sp_s < -1e-9) and strict:
        return None
    return max(t_spex_s, 0.0), max(t_sp_s, 0.0)


__all__ = [
    "RF_REFOCUS_TIME_S",
    "CPMG_PHASE_OFFSET_RAD",
    "DEFAULT_REFOCUS_FLIP_DEG",
    "EXCITATION_FLIP_DEG",
    "build_refocusing_pulse",
    "scale_refocusing_pulse",
    "build_refocus_flip_schedule",
    "fse_timing",
    "surgery_ramp_time_s",
    "build_slice_surgery_chain",
    "build_kz_crusher_bridge",
    "build_surgery_readout",
    "build_readout_surgery_chain",
    "surgery_timing",
]
