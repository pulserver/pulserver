"""Shared EPI-train building blocks for the zoo plugin family.

Used by ``epi_2d.py`` and ``epi_3d.py``. Builds the blipped gradient-echo
train (alternating readout polarity + small phase-encode blips between
lines) shared by both single-shot and multi-shot (``ETL < NY``) EPI.

Loaded via file-path (``importlib.util.spec_from_file_location``), same
convention as ``_gre_common.py`` (loaded as a sibling here for the readout
quantization/system-derate helpers).

Ramp sampling: when disabled (default), the ADC only spans the readout
trapezoid's flat top (like the rest of the GRE family). When enabled, the
ADC spans the *entire* trapezoid (rise + flat + fall) at the same dwell,
i.e. it genuinely samples during the gradient ramps — this is the
sequence-timing-level definition of ramp sampling. Correcting the resulting
non-uniform k-space sample spacing (regridding) is a reconstruction concern
and is out of scope here; this module only produces the correct gradient/ADC
*timing*.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pypulseq as pp


def _load_sibling_module(name: str):
    module_path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gc = _load_sibling_module("_gre_common")


def build_epi_readout(
    opts: pp.Opts,
    ro_axis: str,
    nx_ro: int,
    fov_ro_m: float,
    bandwidth_hz_px: float,
    ramp_sample: bool,
):
    """Build the (gx_pos, gx_neg, adc) readout-line trio shared by every EPI
    line — same trapezoid shape regardless of polarity/ramp-sampling."""
    delta_k_ro = 1.0 / fov_ro_m
    readout_area = nx_ro * delta_k_ro
    dwell_s, flat_time_s = gc.quantize_readout_timing(
        nx_ro=nx_ro,
        target_bw_hz_px=bandwidth_hz_px,
        grad_raster_s=opts.grad_raster_time,
        adc_raster_s=opts.adc_raster_time,
        min_flat_time_s=abs(readout_area) / (0.95 * opts.max_grad),
    )
    gx_pos = pp.make_trapezoid(channel=ro_axis, flat_area=readout_area, flat_time=flat_time_s, system=opts)
    gx_neg = pp.scale_grad(gx_pos, -1.0)

    if ramp_sample:
        total_duration_s = pp.calc_duration(gx_pos)
        n_samples = max(1, int(round(total_duration_s / dwell_s)))
        adc = pp.make_adc(num_samples=n_samples, dwell=dwell_s, delay=0.0, system=opts)
    else:
        adc = pp.make_adc(num_samples=nx_ro, dwell=dwell_s, delay=gx_pos.rise_time, system=opts)

    return gx_pos, gx_neg, adc


def build_pe_blip(opts: pp.Opts, pe_axis: str, fov_pe_m: float):
    """Build the small inter-line phase-encode blip (area = 1 delta_k_pe)."""
    delta_k_pe = 1.0 / fov_pe_m
    return pp.make_trapezoid(channel=pe_axis, area=delta_k_pe, system=opts)


def pre_post_180_delays(
    half_te_s: float,
    d90_s: float,
    c90_s: float,
    d180_s: float,
    c180_s: float,
    pre180_fixed_s: float,
    post180_fixed_s: float,
    strict: bool,
):
    """Compute the two spin-echo timing delays (before/after the 180) that
    keep it exactly at ``TE/2`` (a hard physical requirement for correct
    refocusing, not just a convention) — generalizes
    ``_t2prep_common.t2prep_delays_seconds`` to a non-RF reference point
    after the 180 (the first EPI line's ADC center) with extra fixed
    durations (crushers/diffusion gradients/prephasers) inserted in each
    gap. Returns ``(tau1_s, tau2_s)`` or ``None`` if infeasible under
    ``strict``."""
    pc = _load_sibling_module("_prep_common")
    tau1_s = pc.round_to_raster(half_te_s - (d90_s - c90_s) - pre180_fixed_s - c180_s)
    tau2_s = pc.round_to_raster(half_te_s - (d180_s - c180_s) - post180_fixed_s)
    if (tau1_s < -1e-9 or tau2_s < -1e-9) and strict:
        return None
    return max(tau1_s, 0.0), max(tau2_s, 0.0)


def add_epi_train_blocks(
    seq,
    gx_pos,
    gx_neg,
    gy_blip,
    adc,
    n_lines: int,
    start_polarity_positive: bool = True,
    emit_lin_labels: bool = True,
    lin_start: int = 0,
) -> None:
    """Append ``n_lines`` alternating readout/ADC blocks with blips between
    consecutive lines (no blip after the last line)."""
    for i in range(n_lines):
        positive = start_polarity_positive == (i % 2 == 0)
        gx_this = gx_pos if positive else gx_neg
        adc_this = gc.copy_event(adc)
        if emit_lin_labels:
            label_lin = pp.make_label(type="SET", label="LIN", value=lin_start + i)
            seq.add_block(gx_this, adc_this, label_lin)
        else:
            seq.add_block(gx_this, adc_this)
        if i < n_lines - 1:
            seq.add_block(gy_blip)
