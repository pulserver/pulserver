"""Shared building blocks for the 3D/multi-echo GRE plugin family.

Used by ``gre_3d.py``, ``gre_multiecho_2d.py``, and ``gre_multiecho_3d.py``.

Note: ``gre.py`` (the original 2D single-echo example) predates this module
and is intentionally left untouched/self-contained, per the project's
existing "self-contained example" convention for that file — it is not
retrofitted here. This module exists because the *newer* plugins would
otherwise duplicate two genuinely tricky pieces of logic 2-3x: the 3D
partition-encode/Z-channel gradient combination, and the flyback/bipolar
echo-train builder (a plain single-echo GRE is just the ``num_echoes=1``
case of the echo-train builder).
"""

from __future__ import annotations

import numpy as np
import pypulseq as pp

from pulserver import UIParam

RF_TIME_S = 2.00e-3
RF_APODIZATION = 0.5
RF_TIME_BW_PRODUCT = 4.0
DEFAULT_BANDWIDTH_HZ_PX = 125_000.0
MAX_RASTER_SEARCH_STEPS = 50_000
RF_SPOILING_INC_DEG = 117.0
MAX_GRAD_DERATE = 0.9
MAX_SLEW_DERATE = 0.9
SPOIL_FACTOR_X = 4.0
SPOIL_FACTOR_Z = 2.0


# ---------------------------------------------------------------------------
# System / raster helpers
# ---------------------------------------------------------------------------


def apply_system_derates(opts: pp.Opts) -> None:
    if not hasattr(opts, "_pulserver_base_max_grad"):
        opts._pulserver_base_max_grad = float(opts.max_grad)
    if not hasattr(opts, "_pulserver_base_max_slew"):
        opts._pulserver_base_max_slew = float(opts.max_slew)

    opts.max_grad = opts._pulserver_base_max_grad * float(MAX_GRAD_DERATE)
    opts.max_slew = opts._pulserver_base_max_slew * float(MAX_SLEW_DERATE)


def quantize_readout_timing(
    nx_ro: int,
    target_bw_hz_px: float,
    grad_raster_s: float,
    adc_raster_s: float,
    min_flat_time_s: float,
) -> tuple[float, float]:
    target_dwell = 1.0 / target_bw_hz_px
    m_target = max(1, int(round(target_dwell / adc_raster_s)))

    m_min = max(m_target, int(np.ceil(min_flat_time_s / (nx_ro * adc_raster_s))))

    best_m = None
    best_cost = None

    for step in range(MAX_RASTER_SEARCH_STEPS):
        m = m_min + step
        flat = nx_ro * m * adc_raster_s
        ratio = flat / grad_raster_s
        if abs(ratio - round(ratio)) > 1e-9:
            continue
        cost = abs((m * adc_raster_s) - target_dwell)
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_m = m
            if cost == 0.0:
                break

    if best_m is None:
        best_m = m_min

    dwell = best_m * adc_raster_s
    flat_time = nx_ro * dwell
    return dwell, flat_time


def copy_event(event):
    event_copy = type(event)()
    event_copy.__dict__.update(event.__dict__)
    return event_copy


def sampled_lines(n: int, r: int, acs_lines: int) -> list[int]:
    sampled = {i for i in range(n) if (i % r) == 0}
    if acs_lines > 0:
        center = n // 2
        start = max(0, center - acs_lines // 2)
        stop = min(n, start + acs_lines)
        for i in range(start, stop):
            sampled.add(i)
    return sorted(sampled)


# ---------------------------------------------------------------------------
# Protocol parameter getters
# ---------------------------------------------------------------------------


def param_float(protocol: dict, key: UIParam) -> float:
    return float(protocol[str(key)].value)


def param_int(protocol: dict, key: UIParam) -> int:
    return int(protocol[str(key)].value)


def param_float_optional(protocol: dict, key: UIParam | str, default: float) -> float:
    name = str(key)
    if name not in protocol:
        return default
    return float(protocol[name].value)


def param_int_optional(protocol: dict, key: UIParam | str, default: int) -> int:
    name = str(key)
    if name not in protocol:
        return default
    return int(protocol[name].value)


def param_bool_optional(protocol: dict, key: UIParam | str, default: bool) -> bool:
    name = str(key)
    if name not in protocol:
        return default
    return bool(protocol[name].value)


def user_float(protocol: dict, slot: int, default: float) -> float:
    return param_float_optional(protocol, UIParam.user_value(slot), default)


def phase_fov_mm_from_protocol(protocol: dict) -> float:
    fov_mm = param_float(protocol, UIParam.FOV)
    phase_fov = param_float_optional(protocol, UIParam.PHASE_FOV, fov_mm)
    if phase_fov <= 1.5:
        return max(0.0, phase_fov * fov_mm)
    return phase_fov


def acs_lines_from_protocol(protocol: dict, ny_pe: int, slot: int) -> int:
    raw = user_float(protocol, slot, 0.0)
    acs = int(round(raw))
    if acs < 0:
        return 0
    return min(acs, ny_pe)


def resolve_readout_phase_axes(protocol: dict) -> tuple[str, str]:
    ro_axis = "x"
    pe_axis = "y"
    if param_bool_optional(protocol, UIParam.SWAP_PHASE_FREQ, False):
        ro_axis, pe_axis = "y", "x"
    return ro_axis, pe_axis


def set_protocol_value(protocol: dict, key: UIParam | str, value) -> None:
    entry = protocol[str(key)]
    entry["value"] = value
    if entry.get("type") == "stringlist":
        entry["index"] = entry["options"].index(value)


# ---------------------------------------------------------------------------
# RF / slice-or-slab excitation
# ---------------------------------------------------------------------------


def build_rf(opts: pp.Opts, flip_deg: float, thickness_m: float):
    """Build the (slice- or slab-selective) sinc excitation, matching gre.py."""
    return pp.make_sinc_pulse(
        flip_angle=np.deg2rad(flip_deg),
        duration=RF_TIME_S,
        slice_thickness=thickness_m,
        apodization=RF_APODIZATION,
        time_bw_product=RF_TIME_BW_PRODUCT,
        system=opts,
        return_gz=True,
    )


# ---------------------------------------------------------------------------
# Readout / echo-train building blocks (2D and 3D share this verbatim — a
# plain single-echo GRE is simply the num_echoes=1, flyback=True case).
# ---------------------------------------------------------------------------


def compute_readout_and_echo_train(
    opts: pp.Opts,
    ro_axis: str,
    nx_ro: int,
    fov_ro_m: float,
    bandwidth_hz_px: float,
    slice_thickness_m: float,
    num_echoes: int,
    echo_spacing_s: float,
    flyback: bool,
    strict: bool,
):
    """Design the readout trapezoid(s), ADC, prephaser, spoiler, and echo
    spacing padding. Returns ``None`` if infeasible under ``strict``.
    """
    delta_k_ro = 1.0 / fov_ro_m
    readout_area = nx_ro * delta_k_ro
    dwell_s, flat_time_s = quantize_readout_timing(
        nx_ro=nx_ro,
        target_bw_hz_px=bandwidth_hz_px,
        grad_raster_s=opts.grad_raster_time,
        adc_raster_s=opts.adc_raster_time,
        min_flat_time_s=abs(readout_area) / (0.95 * opts.max_grad),
    )

    # Full rise/flat/fall readout trapezoid per echo: every echo already
    # returns to zero gradient at its end, so no split/bridge trick is
    # required to reach the final spoiler (unlike gre.py's single readout).
    gx_echo = pp.make_trapezoid(channel=ro_axis, flat_area=readout_area, flat_time=flat_time_s, system=opts)
    gx_echo_neg = pp.scale_grad(gx_echo, -1.0)
    adc = pp.make_adc(num_samples=nx_ro, dwell=dwell_s, delay=gx_echo.rise_time, system=opts)

    gx_pre = pp.make_trapezoid(channel=ro_axis, area=-0.5 * gx_echo.area, system=opts)
    gx_spoil = pp.make_trapezoid(channel=ro_axis, area=SPOIL_FACTOR_X / slice_thickness_m, system=opts)

    gx_rewind = None
    gx_echo_duration_s = pp.calc_duration(gx_echo)
    if flyback:
        gx_rewind = pp.make_trapezoid(channel=ro_axis, area=-gx_echo.area, system=opts)
        min_echo_spacing_s = gx_echo_duration_s + pp.calc_duration(gx_rewind)
    else:
        min_echo_spacing_s = gx_echo_duration_s

    # Echo spacing is only meaningful with >= 2 echoes; a single-echo GRE
    # has no inter-echo gap to validate/pad.
    echo_pad_s = 0.0
    if num_echoes > 1:
        echo_pad_s = echo_spacing_s - min_echo_spacing_s
        if echo_pad_s < -1e-9 and strict:
            return None
        if echo_pad_s < 0.0:
            echo_pad_s = 0.0

    adc_center_s = adc.delay + 0.5 * gx_echo.flat_time
    echo_train_span_s = max(0, num_echoes - 1) * echo_spacing_s + gx_echo_duration_s

    return {
        "gx_echo": gx_echo,
        "gx_echo_neg": gx_echo_neg,
        "gx_rewind": gx_rewind,
        "adc": adc,
        "gx_pre": gx_pre,
        "gx_spoil": gx_spoil,
        "gx_echo_duration_s": gx_echo_duration_s,
        "echo_pad_s": echo_pad_s,
        "echo_train_span_s": echo_train_span_s,
        "adc_center_s": adc_center_s,
    }


def add_echo_train_blocks(seq, echo: dict, num_echoes: int, flyback: bool, rf_phase_rad: float, emit_labels: bool = True) -> None:
    """Append the per-echo readout/ADC (+ inter-echo rewind/pad) blocks."""
    gx_echo = echo["gx_echo"]
    gx_echo_neg = echo["gx_echo_neg"]
    gx_rewind = echo["gx_rewind"]
    adc = echo["adc"]
    echo_pad = pp.make_delay(echo["echo_pad_s"]) if echo["echo_pad_s"] > 0.0 else None

    for echo_idx in range(num_echoes):
        gx_this = gx_echo if (flyback or echo_idx % 2 == 0) else gx_echo_neg
        adc_this = copy_event(adc)
        adc_this.phase_offset = rf_phase_rad
        if emit_labels:
            label_eco = pp.make_label(type="SET", label="ECO", value=echo_idx)
            seq.add_block(gx_this, adc_this, label_eco)
        else:
            seq.add_block(gx_this, adc_this)

        if echo_idx < num_echoes - 1:
            if flyback:
                seq.add_block(gx_rewind)
            if echo_pad is not None:
                seq.add_block(echo_pad)


# ---------------------------------------------------------------------------
# 3D partition (Z phase-)encode helpers
# ---------------------------------------------------------------------------


def partition_geometry(npar: int, slice_spacing_m: float):
    """Return (par_areas, max_par_area) for ``npar`` partitions.

    ``par_areas`` mirrors the ``phase_areas`` convention used for the Y
    phase encode; ``max_par_area`` is 0.0 when ``npar <= 1`` (no encoding).
    """
    if npar <= 1:
        return np.zeros(max(npar, 1)), 0.0
    slab_extent_m = slice_spacing_m * npar
    delta_k_par = 1.0 / slab_extent_m
    par_areas = (np.arange(npar) - 0.5 * npar) * delta_k_par
    max_par_area = float(np.max(np.abs(par_areas)))
    return par_areas, max_par_area


def z_worst_case_trapezoids(gz_reph, gz_spoil, max_par_area: float, opts: pp.Opts):
    """Conservative (upper-bound duration) combined Z trapezoids, used only
    for TR/TE feasibility estimation — see ``combined_z_gradients`` for the
    actual per-partition gradients used when building the sequence.
    """
    gz_pre_worst = pp.make_trapezoid(channel="z", area=abs(gz_reph.area) + max_par_area, system=opts)
    gz_post_worst = pp.make_trapezoid(channel="z", area=max_par_area + abs(gz_spoil.area), system=opts)
    return gz_pre_worst, gz_post_worst


def combined_z_gradients(z_scale: float, gz_pe_template, gz_reph, gz_spoil, opts: pp.Opts):
    """Combine the (fixed) RF slab-rephase area with the (variable,
    per-partition) Z phase-encode area into single trapezoids — a pypulseq
    block can only carry one gradient per channel, so the two must be
    folded into one Z-channel event per block (pre-block: rephase + encode;
    post-block: un-encode + spoil).
    """
    gz_pe_scaled = pp.scale_grad(gz_pe_template, z_scale)
    gz_pre_combined = pp.make_trapezoid(channel="z", area=gz_reph.area + gz_pe_scaled.area, system=opts)
    gz_post_combined = pp.make_trapezoid(channel="z", area=-gz_pe_scaled.area + gz_spoil.area, system=opts)
    return gz_pre_combined, gz_post_combined
