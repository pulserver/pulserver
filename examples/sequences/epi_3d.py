"""Standalone 3D single-/multi-shot spin-echo EPI sequence plugin.

This module implements the three mandatory module-level entry points
required by the bridge dispatcher (see ``gre_3d.py`` for the Cartesian 3D
GRE this parallels, ``epi_2d.py`` for the 2D EPI this mirrors, and
``_gre_common.py`` / ``_epi_common.py`` / ``_prep_common.py`` /
``_diffusion_common.py`` for the shared building blocks):

- ``get_default_protocol(opts)``  — return the initial CV/protocol dictionary.
- ``validate_protocol(opts, protocol)``  — validate the current protocol and
  return a ``{"valid": bool, "duration": float | None, "info": str}`` dict.
- ``make_sequence(opts, protocol, output_path)``  — synthesise the sequence
  and write it to *output_path* using :func:`pulserver.io.write`.

Same 90-crusher-[diffusion]-180-crusher-[diffusion]-prephasers-blipped-train
structure as ``epi_2d.py`` (see that file's docstring for the TE/180-timing
rationale), stacked across partitions (accelerated via ``Rz``) like
``gre_3d.py``: a standalone Z phase-encode trapezoid rides alongside the
X/Y prephasers before the train, and is unwound + spoiled in a combined
Z-channel trapezoid after the train (no ``combined_z_gradients`` needed
here since the slab rephase, unlike ``gre_3d.py``, isn't sharing a block
with the partition encode — the rephase happens earlier, symmetric with the
90, before the crushers/180).

Not implemented in this pass (same as ``epi_2d.py``): optional
inversion/T2-prep before the 90.

Usage
-----
Register the plugin in the interpreter tree by symlinking (or copying) this
file to the ``sequences/src/`` directory and creating a numbered alias::

    cp epi_3d.py pulserver-interpreter/package/pulserver/sequences/src/epi_3d.py
    ln -sf src/epi_3d.py pulserver-interpreter/package/pulserver/sequences/sequence14.py
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pypulseq as pp

import pulserver.io as pio
import pulserver.pulseq as ps

from pulserver import (
    PulseqSequence,
    DropdownFloatParam,
    DropdownIntParam,
    TypeinFloatParam,
    UIParam,
    Validate,
    dict_to_protocol,
    make_enum_param,
    protocol_to_dict,
)
from pulserver.core import SequenceType


def _load_sibling_module(name: str):
    """Load a same-directory helper module by file path (see gre_multiecho_2d.py)."""
    module_path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gc = _load_sibling_module("_gre_common")
ec = _load_sibling_module("_epi_common")
pc = _load_sibling_module("_prep_common")
dc = _load_sibling_module("_diffusion_common")

RF_REFOCUS_TIME_S = 3.0e-3
RF_REFOCUS_APODIZATION = 0.5
RF_REFOCUS_TIME_BW_PRODUCT = 4.0

USER_SLOT_RAMP_SAMPLE = 0


class Epi3DPulseqSequence(PulseqSequence):
    """Generate a 3D single-/multi-shot spin-echo EPI sequence."""

    def get_default_protocol(self, opts: pp.Opts) -> dict[str, dict]:
        del opts
        protocol = {
            UIParam.TE: DropdownFloatParam(
                value=60.0, min=20.0, max=200.0, incr=0.1, unit="ms",
                options=[40.0, 60.0, 80.0, 100.0, 150.0], validate=Validate.NONE,
            ),
            UIParam.TR: DropdownFloatParam(
                value=2000.0, min=200.0, max=10000.0, incr=1.0, unit="ms",
                options=[1000.0, 2000.0, 3000.0, 5000.0, 8000.0], validate=Validate.NONE,
            ),
            UIParam.FLIP: DropdownFloatParam(
                value=90.0, min=1.0, max=90.0, incr=1.0, unit="deg",
                options=[45.0, 60.0, 90.0], validate=Validate.NONE,
            ),
            UIParam.FOV: DropdownFloatParam(
                value=220.0, min=80.0, max=500.0, incr=1.0, unit="mm",
                options=[180.0, 220.0, 280.0, 340.0, 500.0], validate=Validate.NONE,
            ),
            UIParam.PHASE_FOV: DropdownFloatParam(
                value=220.0, min=80.0, max=500.0, incr=1.0, unit="mm",
                options=[180.0, 220.0, 280.0, 340.0, 500.0], validate=Validate.NONE,
            ),
            UIParam.SLICE_THICKNESS: DropdownFloatParam(
                value=160.0, min=10.0, max=256.0, incr=1.0, unit="mm",
                options=[100.0, 160.0, 180.0, 200.0, 220.0], validate=Validate.NONE,
            ),
            UIParam.SLICE_SPACING: DropdownFloatParam(
                value=2.0, min=0.5, max=10.0, incr=0.5, unit="mm",
                options=[1.0, 1.5, 2.0, 2.5, 3.0], validate=Validate.NONE,
            ),
            UIParam.NX: DropdownIntParam(
                value=64, min=16, max=512, incr=1, options=[64, 128, 192, 256, 384], validate=Validate.NONE,
            ),
            UIParam.NY: DropdownIntParam(
                value=64, min=8, max=512, incr=1, options=[64, 128, 192, 256, 384], validate=Validate.NONE,
            ),
            UIParam.NSLICES: DropdownIntParam(
                value=8, min=1, max=256, incr=1, options=[8, 16, 32, 64, 128], validate=Validate.NONE,
            ),
            UIParam.ETL: DropdownIntParam(
                value=64, min=1, max=512, incr=1, options=[16, 32, 64, 128, 256], validate=Validate.NONE,
            ),
            UIParam.RZ: TypeinFloatParam(value=1.0, min=1.0, max=8.0, incr=1.0, unit="", validate=Validate.NONE),
            UIParam.BANDWIDTH: TypeinFloatParam(
                value=250_000.0, min=50_000.0, max=1_000_000.0, incr=1_000.0,
                unit="Hz/px", validate=Validate.NONE,
            ),
            UIParam.DIFFUSION_BVALUES: TypeinFloatParam(
                value=0.0, min=0.0, max=5000.0, incr=1.0, unit="s/mm2", validate=Validate.NONE,
            ),
            UIParam.DIFFUSION_DIRECTIONS: DropdownIntParam(
                value=3, min=1, max=32, incr=1, options=[1, 3, 6], validate=Validate.NONE,
            ),
            UIParam.SEQUENCE_TYPE: make_enum_param(UIParam.SEQUENCE_TYPE, SequenceType.SPIN_ECHO),
            UIParam.user_value(USER_SLOT_RAMP_SAMPLE): DropdownFloatParam(
                value=0.0, min=0.0, max=1.0, incr=1.0, unit="", options=[0.0, 1.0], validate=Validate.NONE,
            ),
        }
        return protocol_to_dict(protocol)

    def validate_protocol(self, opts: pp.Opts, protocol: dict[str, dict]) -> dict:
        prot = dict_to_protocol(protocol)
        cfg = _read_protocol(prot)

        if cfg.te_s <= 0.0 or cfg.tr_s <= 0.0:
            return {"valid": False, "duration": None, "info": "TE and TR must be > 0"}
        if cfg.fov_ro_m <= 0.0 or cfg.fov_pe_m <= 0.0 or cfg.slab_thickness_m <= 0.0:
            return {"valid": False, "duration": None, "info": "FOV and slab thickness must be > 0"}
        if not (0.0 < cfg.flip_deg <= 90.0):
            return {"valid": False, "duration": None, "info": "Flip angle must be in (0, 90] deg"}
        if cfg.nx_ro < 1 or cfg.ny_pe < 1 or cfg.npar < 1:
            return {"valid": False, "duration": None, "info": "NX, NY, and NSLICES (partitions) must be >= 1"}
        if cfg.bandwidth_hz_px <= 0.0:
            return {"valid": False, "duration": None, "info": "Bandwidth must be > 0"}
        if cfg.etl < 1:
            return {"valid": False, "duration": None, "info": "ETL must be >= 1"}
        if cfg.b_value_s_mm2 < 0.0:
            return {"valid": False, "duration": None, "info": "Diffusion b-value must be >= 0"}

        timing = _compute_timing(opts=opts, cfg=cfg, strict=True)
        if timing is None:
            return {
                "valid": False,
                "duration": None,
                "info": "TE too short for the 90/180/crushers/diffusion/prephasers, or TR too short",
            }

        n_shots = _n_shots(cfg)
        sampled_par = gc.sampled_lines(cfg.npar, cfg.rz, 0)
        n_dirs = cfg.n_directions if cfg.b_value_s_mm2 > 0.0 else 1
        duration_s = cfg.tr_s * float(n_shots) * float(len(sampled_par)) * float(n_dirs)
        return {"valid": True, "duration": duration_s, "info": f"TA = {duration_s:.2f} s"}

    def make_sequence(self, opts: pp.Opts, protocol: dict[str, dict], output_path: str) -> None:
        prot = dict_to_protocol(protocol)
        cfg = _read_protocol(prot)

        timing = _compute_timing(opts=opts, cfg=cfg, strict=False)

        gz90 = timing["gz90"]
        gz_reph = timing["gz_reph"]
        gx_pre = timing["gx_pre"]
        gy_pre_template = timing["gy_pre_template"]
        gz_pe_template = timing["gz_pe_template"]
        gz_spoil = timing["gz_spoil"]
        gy_blip = timing["gy_blip"]
        gx_pos = timing["gx_pos"]
        gx_neg = timing["gx_neg"]
        adc = timing["adc"]
        tau1_s = timing["tau1_s"]
        tau2_s = timing["tau2_s"]
        tr_delay_s = timing["tr_delay_s"]

        tau1_delay = pp.make_delay(tau1_s) if tau1_s > 0.0 else None
        tau2_delay = pp.make_delay(tau2_s) if tau2_s > 0.0 else None
        tr_delay = pp.make_delay(tr_delay_s) if tr_delay_s > 0.0 else None

        seq = ps.Sequence(opts)

        shot_starts = list(range(0, cfg.ny_pe, cfg.etl))
        par_areas, max_par_area = gc.partition_geometry(cfg.npar, cfg.slice_spacing_m)
        sampled_par = gc.sampled_lines(cfg.npar, cfg.rz, 0)
        n_directions = cfg.n_directions if cfg.b_value_s_mm2 > 0.0 else 1
        directions = dc.diffusion_directions(cfg.n_directions) if cfg.b_value_s_mm2 > 0.0 else [None]

        for direction in directions[:n_directions]:
            diff_grad = (
                dc.build_diffusion_gradients(opts, direction, timing["delta_s"], timing["grad_t_per_m"])
                if direction is not None
                else []
            )

            for par in sampled_par:
                z_scale = par_areas[par] / max_par_area if max_par_area > 0.0 else 0.0
                gz_pe = pp.scale_grad(gz_pe_template, z_scale) if gz_pe_template is not None else None
                gz_post = pp.make_trapezoid(
                    channel="z",
                    area=(-(gz_pe.area) if gz_pe is not None else 0.0) + gz_spoil.area,
                    system=opts,
                )

                for shot_idx, ky_start in enumerate(shot_starts):
                    n_lines = min(cfg.etl, cfg.ny_pe - ky_start)
                    label_par = pp.make_label(type="SET", label="PAR", value=par)
                    label_slc = pp.make_label(type="SET", label="SLC", value=0)

                    rf90 = gc.copy_event(timing["rf90"])
                    rf180 = gc.copy_event(timing["rf180"])

                    seq.add_block(rf90, gz90, label_slc, label_par)
                    seq.add_block(gz_reph)
                    seq.add_block(*timing["crusher_before"])
                    if diff_grad:
                        seq.add_block(*diff_grad)
                    if tau1_delay is not None:
                        seq.add_block(tau1_delay)
                    seq.add_block(rf180, timing["gz180"])
                    seq.add_block(*timing["crusher_after"])
                    if diff_grad:
                        seq.add_block(*diff_grad)
                    if tau2_delay is not None:
                        seq.add_block(tau2_delay)

                    y_start_scale = (2.0 * ky_start / cfg.ny_pe) - 1.0
                    gy_pre = pp.scale_grad(gy_pre_template, y_start_scale)
                    if gz_pe is not None:
                        seq.add_block(gx_pre, gy_pre, gz_pe)
                    else:
                        seq.add_block(gx_pre, gy_pre)

                    ec.add_epi_train_blocks(
                        seq, gx_pos, gx_neg, gy_blip, adc, n_lines,
                        start_polarity_positive=True, emit_lin_labels=True, lin_start=ky_start,
                    )

                    seq.add_block(gz_post)

                    if tr_delay is not None:
                        seq.add_block(tr_delay)

        seq.set_definition("Name", "epi_3d")
        seq.set_definition("FOV", [cfg.fov_ro_m, cfg.fov_pe_m, cfg.slice_spacing_m * cfg.npar])
        seq.set_definition("TE", cfg.te_s)
        seq.set_definition("TR", cfg.tr_s)
        seq.set_definition("Flip", cfg.flip_deg)
        seq.set_definition("ImagingMode", "3d")
        seq.set_definition("ETL", cfg.etl)
        seq.set_definition("NumShots", len(shot_starts))
        seq.set_definition("RampSample", cfg.ramp_sample)
        seq.set_definition("BValue", cfg.b_value_s_mm2)
        seq.set_definition("DiffusionDirections", n_directions)
        seq.set_definition("BandwidthHzPerPx", cfg.bandwidth_hz_px)
        seq.set_definition("Rz", cfg.rz)
        seq.set_definition("Nx", cfg.nx_ro)
        seq.set_definition("Ny", cfg.ny_pe)
        seq.set_definition("NumPartitions", cfg.npar)
        pio.write(seq, output=output_path, remove_duplicates=False, check_timing=False)


def _n_shots(cfg: "_Config") -> int:
    return len(range(0, cfg.ny_pe, cfg.etl))


class _Config:
    __slots__ = (
        "te_s", "tr_s", "flip_deg", "fov_ro_m", "fov_pe_m", "slab_thickness_m", "slice_spacing_m",
        "nx_ro", "ny_pe", "npar", "etl", "rz", "bandwidth_hz_px", "ramp_sample",
        "b_value_s_mm2", "n_directions",
    )


def _read_protocol(prot: dict) -> _Config:
    cfg = _Config()
    cfg.te_s = gc.param_float(prot, UIParam.TE) * 1e-3
    cfg.tr_s = gc.param_float(prot, UIParam.TR) * 1e-3
    cfg.flip_deg = gc.param_float(prot, UIParam.FLIP)
    cfg.fov_ro_m = gc.param_float(prot, UIParam.FOV) * 1e-3
    cfg.fov_pe_m = gc.phase_fov_mm_from_protocol(prot) * 1e-3
    cfg.slab_thickness_m = gc.param_float(prot, UIParam.SLICE_THICKNESS) * 1e-3
    cfg.slice_spacing_m = gc.param_float(prot, UIParam.SLICE_SPACING) * 1e-3
    cfg.nx_ro = gc.param_int(prot, UIParam.NX)
    cfg.ny_pe = gc.param_int(prot, UIParam.NY)
    cfg.npar = gc.param_int(prot, UIParam.NSLICES)
    cfg.etl = gc.param_int_optional(prot, UIParam.ETL, cfg.ny_pe)
    cfg.rz = max(1, int(round(gc.param_float_optional(prot, UIParam.RZ, 1.0))))
    cfg.bandwidth_hz_px = gc.param_float_optional(prot, UIParam.BANDWIDTH, 250_000.0)
    cfg.ramp_sample = gc.user_float(prot, USER_SLOT_RAMP_SAMPLE, 0.0) >= 0.5
    cfg.b_value_s_mm2 = gc.param_float_optional(prot, UIParam.DIFFUSION_BVALUES, 0.0)
    cfg.n_directions = gc.param_int_optional(prot, UIParam.DIFFUSION_DIRECTIONS, 3)
    return cfg


def _compute_timing(opts: pp.Opts, cfg: _Config, strict: bool):
    gc.apply_system_derates(opts)

    rf90, gz90, gz_reph = gc.build_rf(opts, cfg.flip_deg, cfg.slab_thickness_m)
    rf180, gz180, _ = pp.make_sinc_pulse(
        flip_angle=np.pi,
        duration=RF_REFOCUS_TIME_S,
        slice_thickness=cfg.slab_thickness_m,
        apodization=RF_REFOCUS_APODIZATION,
        time_bw_product=RF_REFOCUS_TIME_BW_PRODUCT,
        system=opts,
        use="refocusing",
        return_gz=True,
    )

    gx_pos, gx_neg, adc = ec.build_epi_readout(
        opts, "x", cfg.nx_ro, cfg.fov_ro_m, cfg.bandwidth_hz_px, cfg.ramp_sample
    )
    gy_blip = ec.build_pe_blip(opts, "y", cfg.fov_pe_m)

    gx_pre = pp.make_trapezoid(channel="x", area=-0.5 * gx_pos.area, system=opts)
    max_pe_area = 0.5 * cfg.ny_pe * (1.0 / cfg.fov_pe_m)
    gy_pre_template = pp.make_trapezoid(channel="y", area=max_pe_area, system=opts)

    _, max_par_area = gc.partition_geometry(cfg.npar, cfg.slice_spacing_m)
    gz_pe_template = pp.make_trapezoid(channel="z", area=max_par_area, system=opts) if max_par_area > 0.0 else None
    gz_spoil = pp.make_trapezoid(channel="z", area=gc.SPOIL_FACTOR_Z / cfg.slab_thickness_m, system=opts)

    voxel_size_m = cfg.fov_ro_m / cfg.nx_ro
    crusher_before = pc.build_inversion_spoiler(opts, voxel_size_m)
    crusher_after = pc.build_inversion_spoiler(opts, voxel_size_m)

    delta_s, separation_s = dc.diffusion_timing(cfg.te_s)
    grad_t_per_m = dc.required_gradient_t_per_m(cfg.b_value_s_mm2, delta_s, separation_s, opts.gamma)
    max_grad_t_per_m = opts.max_grad / opts.gamma
    if grad_t_per_m > max_grad_t_per_m:
        if strict:
            return None
        grad_t_per_m = max_grad_t_per_m
    diff_grad_duration_s = delta_s if cfg.b_value_s_mm2 > 0.0 else 0.0

    d90_s = pp.calc_duration(rf90, gz90)
    c90_s = pp.calc_rf_center(rf90)[0]
    d180_s = pp.calc_duration(rf180, gz180)
    c180_s = pp.calc_rf_center(rf180)[0]

    d_gz_reph_s = pp.calc_duration(gz_reph)
    d_crusher_before_s = pp.calc_duration(*crusher_before)
    d_crusher_after_s = pp.calc_duration(*crusher_after)
    gz_pre_worst = pp.make_trapezoid(channel="z", area=max_par_area, system=opts) if max_par_area > 0.0 else None
    d_prephasers_s = (
        pp.calc_duration(gx_pre, gy_pre_template, gz_pre_worst)
        if gz_pre_worst is not None
        else pp.calc_duration(gx_pre, gy_pre_template)
    )

    if cfg.ramp_sample:
        adc_center_s = adc.delay + 0.5 * pp.calc_duration(gx_pos)
    else:
        adc_center_s = adc.delay + 0.5 * gx_pos.flat_time

    pre180_fixed_s = d_gz_reph_s + d_crusher_before_s + diff_grad_duration_s
    post180_fixed_s = d_crusher_after_s + diff_grad_duration_s + d_prephasers_s + adc_center_s

    delays = ec.pre_post_180_delays(
        cfg.te_s / 2.0, d90_s, c90_s, d180_s, c180_s, pre180_fixed_s, post180_fixed_s, strict
    )
    if delays is None:
        return None
    tau1_s, tau2_s = delays

    n_lines_first_shot = min(cfg.etl, cfg.ny_pe)
    line_period_s = pp.calc_duration(gx_pos)
    train_span_s = n_lines_first_shot * line_period_s + max(0, n_lines_first_shot - 1) * pp.calc_duration(gy_blip)
    d_post_s = pp.calc_duration(gz_spoil)

    n_inner = len(gc.sampled_lines(cfg.npar, cfg.rz, 0)) if strict is False else 1

    min_block_s = (
        d90_s + d_gz_reph_s + d_crusher_before_s + diff_grad_duration_s + tau1_s
        + d180_s + d_crusher_after_s + diff_grad_duration_s + tau2_s
        + d_prephasers_s + train_span_s + d_post_s
    )
    tr_delay_s = cfg.tr_s - n_inner * min_block_s
    if tr_delay_s < -1e-9 and strict:
        return None
    if tr_delay_s < 0.0:
        tr_delay_s = 0.0

    return {
        "rf90": rf90,
        "gz90": gz90,
        "gz_reph": gz_reph,
        "rf180": rf180,
        "gz180": gz180,
        "gx_pre": gx_pre,
        "gy_pre_template": gy_pre_template,
        "gz_pe_template": gz_pe_template,
        "gz_spoil": gz_spoil,
        "gy_blip": gy_blip,
        "gx_pos": gx_pos,
        "gx_neg": gx_neg,
        "adc": adc,
        "crusher_before": crusher_before,
        "crusher_after": crusher_after,
        "delta_s": delta_s,
        "grad_t_per_m": grad_t_per_m,
        "tau1_s": tau1_s,
        "tau2_s": tau2_s,
        "tr_delay_s": tr_delay_s,
        "min_block_s": min_block_s,
    }


PLUGIN = Epi3DPulseqSequence()


def get_default_protocol(opts):
    return PLUGIN.get_default_protocol(opts)


def validate_protocol(opts, protocol):
    return PLUGIN.validate_protocol(opts, protocol)


def make_sequence(opts, protocol, output_path):
    return PLUGIN.make_sequence(opts, protocol, output_path)


def makeSeq(opts, protocol, output_path):
    """Offline alias for compatibility with older helper naming."""
    return PLUGIN.make_sequence(opts, protocol, output_path)


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a 3D spin-echo EPI .seq offline.")
    parser.add_argument("-o", "--output", default="epi_3d.seq", help="Output .seq file path")
    parser.add_argument("--te-ms", type=float)
    parser.add_argument("--tr-ms", type=float)
    parser.add_argument("--flip-deg", type=float)
    parser.add_argument("--fov-mm", type=float)
    parser.add_argument("--phase-fov-mm", type=float)
    parser.add_argument("--slab-thickness-mm", type=float, dest="slice_thickness_mm")
    parser.add_argument("--partition-spacing-mm", type=float, dest="slice_spacing_mm")
    parser.add_argument("--nx", type=int)
    parser.add_argument("--ny", type=int)
    parser.add_argument("--npartitions", type=int, dest="nslices")
    parser.add_argument("--etl", type=int)
    parser.add_argument("--rz", type=float)
    parser.add_argument("--bandwidth-hz-px", type=float)
    parser.add_argument("--ramp-sample", action="store_true")
    parser.add_argument("--bvalue", type=float)
    parser.add_argument("--directions", type=int)
    parser.add_argument("--max-grad-mtm", type=float)
    parser.add_argument("--max-slew-tm-s", type=float)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def _cli(argv: list[str]) -> int:
    parser = _build_cli_parser()
    args = parser.parse_args(argv)

    opts_kwargs = {}
    if args.max_grad_mtm is not None:
        opts_kwargs["max_grad"] = args.max_grad_mtm
        opts_kwargs["grad_unit"] = "mT/m"
    if args.max_slew_tm_s is not None:
        opts_kwargs["max_slew"] = args.max_slew_tm_s
        opts_kwargs["slew_unit"] = "T/m/s"
    opts = pp.Opts(**opts_kwargs)

    protocol = PLUGIN.get_default_protocol(opts)

    if args.te_ms is not None:
        gc.set_protocol_value(protocol, UIParam.TE, args.te_ms)
    if args.tr_ms is not None:
        gc.set_protocol_value(protocol, UIParam.TR, args.tr_ms)
    if args.flip_deg is not None:
        gc.set_protocol_value(protocol, UIParam.FLIP, args.flip_deg)
    if args.fov_mm is not None:
        gc.set_protocol_value(protocol, UIParam.FOV, args.fov_mm)
    if args.phase_fov_mm is not None:
        gc.set_protocol_value(protocol, UIParam.PHASE_FOV, args.phase_fov_mm)
    if args.slice_thickness_mm is not None:
        gc.set_protocol_value(protocol, UIParam.SLICE_THICKNESS, args.slice_thickness_mm)
    if args.slice_spacing_mm is not None:
        gc.set_protocol_value(protocol, UIParam.SLICE_SPACING, args.slice_spacing_mm)
    if args.nx is not None:
        gc.set_protocol_value(protocol, UIParam.NX, args.nx)
    if args.ny is not None:
        gc.set_protocol_value(protocol, UIParam.NY, args.ny)
    if args.nslices is not None:
        gc.set_protocol_value(protocol, UIParam.NSLICES, args.nslices)
    if args.etl is not None:
        gc.set_protocol_value(protocol, UIParam.ETL, args.etl)
    if args.rz is not None:
        gc.set_protocol_value(protocol, UIParam.RZ, args.rz)
    if args.bandwidth_hz_px is not None:
        gc.set_protocol_value(protocol, UIParam.BANDWIDTH, args.bandwidth_hz_px)
    if args.ramp_sample:
        gc.set_protocol_value(protocol, UIParam.user_value(USER_SLOT_RAMP_SAMPLE), 1.0)
    if args.bvalue is not None:
        gc.set_protocol_value(protocol, UIParam.DIFFUSION_BVALUES, args.bvalue)
    if args.directions is not None:
        gc.set_protocol_value(protocol, UIParam.DIFFUSION_DIRECTIONS, args.directions)

    result = PLUGIN.validate_protocol(opts, protocol)
    if not result.get("valid", False):
        print(f"ERROR: {result.get('info', 'Protocol invalid')}", file=sys.stderr)
        return 2

    print(result.get("info", "Protocol valid"))
    if args.validate_only:
        return 0

    PLUGIN.make_sequence(opts, protocol, args.output)
    print(f"Wrote sequence: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv[1:]))
