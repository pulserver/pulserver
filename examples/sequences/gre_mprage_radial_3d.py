"""Standalone 3D non-Cartesian ("stack-of-stars") inversion/T2-prepared GRE.

This module implements the three mandatory module-level entry points
required by the bridge dispatcher (see ``mprage_3d.py`` for the Cartesian
inversion-prepared GRE this extends, ``gre_radial_3d.py`` for the
non-prepared stack-of-stars sibling, and ``_gre_common.py`` /
``_prep_common.py`` / ``_t2prep_common.py`` / ``_mt_common.py`` for the
shared building blocks):

- ``get_default_protocol(opts)``  — return the initial CV/protocol dictionary.
- ``validate_protocol(opts, protocol)``  — validate the current protocol and
  return a ``{"valid": bool, "duration": float | None, "info": str}`` dict.
- ``make_sequence(opts, protocol, output_path)``  — synthesise the sequence
  and write it to *output_path* using :func:`pulserver.io.write`.

Structure — one magnetization-prep module gates one "segment" of spokes at
a fixed partition; the segment length is ``ETL`` (chunked if
``ETL < NumShots``), exactly mirroring ``mprage_3d.py``'s PE-line
segmentation with spoke index in place of ``ky``. Partitions are the outer
loop (accelerated via ``Rz``); spokes are the inner, per-segment loop.

``PreparationType`` (native GE enum, already closed to ``inversion``/
``t2_prep`` — no new values added, per the project's hard rule) selects the
magnetization-prep module:

- ``inversion``: non-selective hard/adiabatic 180 + spoiler + TI delay
  (``_prep_common.py``, same as ``mprage_3d.py``).
- ``t2_prep``: composite 90-tau/2-180-tau/2-(-90) + spoiler
  (``_t2prep_common.py``) — ``TE_prep`` sets the T2 weighting directly, so
  (unlike TI) there is no extra post-module centering delay; the segment
  starts right after the spoiler.

An independent, *additive* MT saturation toggle (``_mt_common.py``) can be
stacked on top of either prep — real MT-prepared sequences apply the MT
pulse every TR (steady-state effect), not once per segment, so it is not a
third ``PreparationType`` option (which would also violate the closed-enum
rule: GE's real UI has no generic "prep type = MT" slot).

``TE``/``TR`` retain their per-view meaning; ``ETL``/``Trecovery`` are
reused from the MPRAGE family. TI, ``TE_prep``, the inversion/refocus-mode
toggles, and the MT enable/params all lack a native GE counterpart, so they
are ``opuser`` custom variables. Trajectory is plain full-echo radial
(rotated spokes, ``pulserver.pulseq.make_rotation`` — no ``pulserver.arbgrad``
waveform solver needed) — see ``gre_noncart_3d.py`` for the arbgrad-backed
spiral/rosette sibling; swapping the readout builder there for this file's
prep/segment scheduling is a mechanical follow-up, not implemented here.

Usage
-----
Register the plugin in the interpreter tree by symlinking (or copying) this
file to the ``sequences/src/`` directory and creating a numbered alias::

    cp gre_mprage_radial_3d.py pulserver-interpreter/package/pulserver/sequences/src/gre_mprage_radial_3d.py
    ln -sf src/gre_mprage_radial_3d.py pulserver-interpreter/package/pulserver/sequences/sequence12.py
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pypulseq as pp
from scipy.spatial.transform import Rotation

import pulserver.io as pio
import pulserver.pulseq as ps

from pulserver import (
    PulseqSequence,
    BoolParam,
    Description,
    DropdownFloatParam,
    DropdownIntParam,
    TypeinFloatParam,
    UIParam,
    Validate,
    dict_to_protocol,
    make_enum_param,
    protocol_to_dict,
)
from pulserver import arbgrad
from pulserver.core import PreparationType, SequenceType


def _load_sibling_module(name: str):
    """Load a same-directory helper module by file path (see gre_multiecho_2d.py)."""
    module_path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gc = _load_sibling_module("_gre_common")
pc = _load_sibling_module("_prep_common")
t2p = _load_sibling_module("_t2prep_common")
mtc = _load_sibling_module("_mt_common")

NUM_ECHOES = 1
FLYBACK = True
RO_AXIS = "x"

USER_SLOT_TI = 0
USER_SLOT_INV_MODE = 1
USER_SLOT_TE_PREP = 2
USER_SLOT_REFOCUS_MODE = 3
USER_SLOT_ORDER_MODE = 4
USER_SLOT_MT_ENABLE = 5


def _order_mode_name(code: float) -> str:
    return "golden" if code >= 0.5 else "uniform"


class GreMprageRadial3DPulseqSequence(PulseqSequence):
    """Generate a 3D stack-of-stars inversion/T2-prepared GRE, +/- MT sat."""

    def get_default_protocol(self, opts: pp.Opts) -> dict[str, dict]:
        del opts
        protocol = {
            UIParam.TE: DropdownFloatParam(
                value=3.0, min=1.5, max=80.0, incr=0.1, unit="ms",
                options=[2.0, 3.0, 5.0, 8.0, 15.0], validate=Validate.NONE,
            ),
            UIParam.TR: DropdownFloatParam(
                value=8.0, min=4.0, max=100.0, incr=0.1, unit="ms",
                options=[6.0, 8.0, 10.0, 15.0, 30.0], validate=Validate.NONE,
            ),
            UIParam.TRECOVERY: DropdownFloatParam(
                value=1200.0, min=0.0, max=5000.0, incr=1.0, unit="ms",
                options=[600.0, 1000.0, 1200.0, 1500.0, 2000.0], validate=Validate.NONE,
            ),
            UIParam.ETL: DropdownIntParam(
                value=8, min=1, max=512, incr=1, options=[8, 16, 32, 64, 128], validate=Validate.NONE,
            ),
            UIParam.FLIP: DropdownFloatParam(
                value=8.0, min=1.0, max=90.0, incr=1.0, unit="deg",
                options=[4.0, 8.0, 12.0, 20.0, 30.0], validate=Validate.NONE,
            ),
            UIParam.FOV: DropdownFloatParam(
                value=220.0, min=80.0, max=500.0, incr=1.0, unit="mm",
                options=[180.0, 220.0, 280.0, 340.0, 500.0], validate=Validate.NONE,
            ),
            UIParam.SLICE_THICKNESS: DropdownFloatParam(
                value=160.0, min=10.0, max=256.0, incr=1.0, unit="mm",
                options=[100.0, 160.0, 180.0, 200.0, 220.0], validate=Validate.NONE,
            ),
            UIParam.SLICE_SPACING: DropdownFloatParam(
                value=1.0, min=0.5, max=10.0, incr=0.5, unit="mm",
                options=[0.8, 1.0, 1.2, 1.5, 2.0], validate=Validate.NONE,
            ),
            UIParam.NX: DropdownIntParam(
                value=64, min=16, max=512, incr=1, options=[64, 128, 192, 256, 384], validate=Validate.NONE,
            ),
            UIParam.NSLICES: DropdownIntParam(
                value=8, min=1, max=256, incr=1, options=[8, 16, 32, 64, 128], validate=Validate.NONE,
            ),
            UIParam.NUM_SHOTS: DropdownIntParam(
                value=32, min=8, max=2048, incr=1, options=[16, 32, 64, 128, 256], validate=Validate.NONE,
            ),
            UIParam.RZ: TypeinFloatParam(value=1.0, min=1.0, max=8.0, incr=1.0, unit="", validate=Validate.NONE),
            UIParam.BANDWIDTH: TypeinFloatParam(
                value=gc.DEFAULT_BANDWIDTH_HZ_PX, min=5_000.0, max=500_000.0, incr=100.0,
                unit="Hz/px", validate=Validate.NONE,
            ),
            UIParam.SEQUENCE_TYPE: make_enum_param(UIParam.SEQUENCE_TYPE, SequenceType.GRADIENT_ECHO),
            UIParam.PREPARATION_TYPE: make_enum_param(UIParam.PREPARATION_TYPE, PreparationType.INVERSION),
            UIParam.user_name(USER_SLOT_TI): Description(text="Inversion time (TI)"),
            UIParam.user_value(USER_SLOT_TI): TypeinFloatParam(
                value=900.0, min=0.1, max=5000.0, incr=1.0, unit="ms", validate=Validate.NONE,
            ),
            UIParam.user_name(USER_SLOT_INV_MODE): Description(text="Inversion pulse (0=hard, 1=adiabatic)"),
            UIParam.user_value(USER_SLOT_INV_MODE): DropdownFloatParam(
                value=1.0, min=0.0, max=1.0, incr=1.0, unit="", options=[0.0, 1.0], validate=Validate.NONE,
            ),
            UIParam.user_name(USER_SLOT_TE_PREP): Description(text="T2-prep echo time (TE_prep)"),
            UIParam.user_value(USER_SLOT_TE_PREP): TypeinFloatParam(
                value=50.0, min=1.0, max=200.0, incr=1.0, unit="ms", validate=Validate.NONE,
            ),
            UIParam.user_name(USER_SLOT_REFOCUS_MODE): Description(text="T2-prep refocus (0=hard, 1=adiabatic)"),
            UIParam.user_value(USER_SLOT_REFOCUS_MODE): DropdownFloatParam(
                value=0.0, min=0.0, max=1.0, incr=1.0, unit="", options=[0.0, 1.0], validate=Validate.NONE,
            ),
            UIParam.user_name(USER_SLOT_ORDER_MODE): Description(text="Spoke order (0=uniform, 1=golden)"),
            UIParam.user_value(USER_SLOT_ORDER_MODE): DropdownFloatParam(
                value=1.0, min=0.0, max=1.0, incr=1.0, unit="", options=[0.0, 1.0], validate=Validate.NONE,
            ),
            UIParam.user_name(USER_SLOT_MT_ENABLE): Description(text="MT saturation (0=off, 1=on, every view)"),
            UIParam.user_value(USER_SLOT_MT_ENABLE): DropdownFloatParam(
                value=0.0, min=0.0, max=1.0, incr=1.0, unit="", options=[0.0, 1.0], validate=Validate.NONE,
            ),
        }
        return protocol_to_dict(protocol)

    def validate_protocol(self, opts: pp.Opts, protocol: dict[str, dict]) -> dict:
        prot = dict_to_protocol(protocol)
        cfg = _read_protocol(prot)

        if cfg.te_s <= 0.0 or cfg.tr_s <= 0.0:
            return {"valid": False, "duration": None, "info": "TE and TR must be > 0"}
        if cfg.fov_m <= 0.0 or cfg.slab_thickness_m <= 0.0:
            return {"valid": False, "duration": None, "info": "FOV and slab thickness must be > 0"}
        if not (0.0 < cfg.flip_deg <= 180.0):
            return {"valid": False, "duration": None, "info": "Flip angle must be in (0, 180] deg"}
        if cfg.nx_ro < 1 or cfg.npar < 1:
            return {"valid": False, "duration": None, "info": "NX and NSLICES (partitions) must be >= 1"}
        if cfg.bandwidth_hz_px <= 0.0:
            return {"valid": False, "duration": None, "info": "Bandwidth must be > 0"}
        if cfg.etl < 1:
            return {"valid": False, "duration": None, "info": "ETL must be >= 1"}
        if cfg.num_shots < 1:
            return {"valid": False, "duration": None, "info": "NumShots must be >= 1"}
        if cfg.trecovery_s < 0.0:
            return {"valid": False, "duration": None, "info": "Trecovery must be >= 0"}
        if cfg.prep_type == PreparationType.INVERSION and cfg.ti_s <= 0.0:
            return {"valid": False, "duration": None, "info": "TI must be > 0"}
        if cfg.prep_type == PreparationType.T2_PREP and cfg.te_prep_s <= 0.0:
            return {"valid": False, "duration": None, "info": "TE_prep must be > 0"}

        timing = _compute_timing(opts=opts, cfg=cfg, strict=True)
        if timing is None:
            return {
                "valid": False,
                "duration": None,
                "info": "TE, TR, TI, or TE_prep infeasible for the requested gradients/ETL",
            }

        sampled_par = gc.sampled_lines(cfg.npar, cfg.rz, 0)
        n_segments = len(pc.chunk_indices(list(range(cfg.num_shots)), cfg.etl))
        shot_s = timing["shot_s"]
        duration_s = shot_s * n_segments * len(sampled_par)
        return {"valid": True, "duration": duration_s, "info": f"TA = {duration_s:.2f} s"}

    def make_sequence(self, opts: pp.Opts, protocol: dict[str, dict], output_path: str) -> None:
        prot = dict_to_protocol(protocol)
        cfg = _read_protocol(prot)

        timing = _compute_timing(opts=opts, cfg=cfg, strict=False)

        gz_reph = timing["gz_reph"]
        gz_spoil = timing["gz_spoil"]
        gz_pe_template = timing["gz_pe_template"]
        echo = timing["echo"]
        te_delay_s = timing["te_delay_s"]
        tr_delay_s = timing["tr_delay_s"]

        te_delay = pp.make_delay(te_delay_s) if te_delay_s > 0.0 else None
        tr_delay = pp.make_delay(tr_delay_s) if tr_delay_s > 0.0 else None
        trecovery_delay = pp.make_delay(pc.round_to_raster(cfg.trecovery_s)) if cfg.trecovery_s > 0.0 else None

        seq = ps.Sequence(opts)

        angles = arbgrad.shot_angles(cfg.num_shots, mode=cfg.order_mode)
        segments = pc.chunk_indices(list(range(cfg.num_shots)), cfg.etl)

        par_areas, max_par_area = gc.partition_geometry(cfg.npar, cfg.slice_spacing_m)
        sampled_par = gc.sampled_lines(cfg.npar, cfg.rz, 0)

        for par in sampled_par:
            z_scale = par_areas[par] / max_par_area if max_par_area > 0.0 else 0.0
            label_par = pp.make_label(type="SET", label="PAR", value=par)
            label_slc = pp.make_label(type="SET", label="SLC", value=0)

            for segment in segments:
                segment_duration_s = len(segment) * cfg.tr_s
                _emit_prep_module(seq, timing, cfg, segment_duration_s)

                rf_phase_deg = 0.0
                rf_phase_inc_deg = 0.0

                for shot in segment:
                    angle = float(angles[shot])
                    rotation = ps.make_rotation(Rotation.from_euler("z", angle))
                    label_lin = pp.make_label(type="SET", label="LIN", value=shot)

                    gz_pre_combined, gz_post_combined = gc.combined_z_gradients(
                        z_scale, gz_pe_template, gz_reph, gz_spoil, opts
                    )

                    if cfg.mt_enable:
                        seq.add_block(gc.copy_event(timing["rf_mt"]))
                        seq.add_block(*timing["mt_spoiler"])

                    rf_curr = gc.copy_event(timing["rf"])
                    rf_curr.phase_offset = np.deg2rad(rf_phase_deg)
                    adc_curr = gc.copy_event(echo["adc"])
                    adc_curr.phase_offset = rf_curr.phase_offset

                    seq.add_block(rf_curr, timing["gz"], label_slc, label_par, label_lin)
                    seq.add_block(echo["gx_pre"], gz_pre_combined, rotation)
                    if te_delay is not None:
                        seq.add_block(te_delay)

                    # NUM_ECHOES is pinned to 1 (single-echo GRE view), so the
                    # echo train is just one rotated readout block — emitted
                    # directly (not via gc.add_echo_train_blocks, which has
                    # no extension-passing hook for a non-Cartesian rotation).
                    seq.add_block(echo["gx_echo"], adc_curr, rotation)

                    seq.add_block(echo["gx_spoil"], gz_post_combined, rotation)
                    if tr_delay is not None:
                        seq.add_block(tr_delay)

                    rf_phase_deg = (rf_phase_deg + rf_phase_inc_deg) % 360.0
                    rf_phase_inc_deg = (rf_phase_inc_deg + gc.RF_SPOILING_INC_DEG) % 360.0

                if trecovery_delay is not None:
                    seq.add_block(trecovery_delay)

        seq.set_definition("Name", "gre_mprage_radial_3d")
        seq.set_definition("FOV", [cfg.fov_m, cfg.fov_m, cfg.slice_spacing_m * cfg.npar])
        seq.set_definition("TE", cfg.te_s)
        seq.set_definition("TR", cfg.tr_s)
        seq.set_definition("Trecovery", cfg.trecovery_s)
        seq.set_definition("ETL", cfg.etl)
        seq.set_definition("PreparationType", str(cfg.prep_type))
        seq.set_definition("TI", cfg.ti_s)
        seq.set_definition("TEprep", cfg.te_prep_s)
        seq.set_definition("MTEnabled", cfg.mt_enable)
        seq.set_definition("Flip", cfg.flip_deg)
        seq.set_definition("ImagingMode", "3d")
        seq.set_definition("Trajectory", "radial")
        seq.set_definition("SpokeOrder", cfg.order_mode)
        seq.set_definition("BandwidthHzPerPx", cfg.bandwidth_hz_px)
        seq.set_definition("Rz", cfg.rz)
        seq.set_definition("RfSpoilingIncDeg", gc.RF_SPOILING_INC_DEG)
        seq.set_definition("Nx", cfg.nx_ro)
        seq.set_definition("NumShots", cfg.num_shots)
        seq.set_definition("NumPartitions", cfg.npar)
        pio.write(seq, output=output_path, remove_duplicates=False, check_timing=False)


def _emit_prep_module(seq, timing: dict, cfg: "_Config", segment_duration_s: float) -> None:
    """Emit the (inversion or T2-prep) magnetization-prep blocks for one
    segment. TI feasibility was already checked in ``_compute_timing``; here
    we only recompute the (non-strict, clipped) delay for the actual
    segment length, mirroring ``mprage_3d.py``."""
    if cfg.prep_type == PreparationType.INVERSION:
        rf_inv = timing["rf_inv"]
        spoiler = timing["prep_spoiler"]
        spoiler_duration_s = timing["prep_spoiler_duration_s"]
        ti_delay_s = pc.ti_delay_seconds(cfg.ti_s, rf_inv, spoiler_duration_s, segment_duration_s, strict=False)
        seq.add_block(gc.copy_event(rf_inv))
        seq.add_block(*spoiler)
        if ti_delay_s > 0.0:
            seq.add_block(pp.make_delay(ti_delay_s))
    else:
        rf90_down, rf180, rf90_up = timing["t2prep_pulses"]
        spoiler = timing["prep_spoiler"]
        delay1_s, delay2_s = t2p.t2prep_delays_seconds(cfg.te_prep_s, rf90_down, rf180, rf90_up, strict=False)
        seq.add_block(gc.copy_event(rf90_down))
        if delay1_s > 0.0:
            seq.add_block(pp.make_delay(delay1_s))
        seq.add_block(gc.copy_event(rf180))
        if delay2_s > 0.0:
            seq.add_block(pp.make_delay(delay2_s))
        seq.add_block(gc.copy_event(rf90_up))
        seq.add_block(*spoiler)


class _Config:
    __slots__ = (
        "te_s", "tr_s", "flip_deg", "fov_m", "slab_thickness_m", "slice_spacing_m",
        "nx_ro", "npar", "bandwidth_hz_px", "rz", "num_shots", "order_mode",
        "etl", "trecovery_s", "prep_type", "ti_s", "inv_mode", "te_prep_s", "refocus_mode", "mt_enable",
    )


def _read_protocol(prot: dict) -> _Config:
    cfg = _Config()
    cfg.te_s = gc.param_float(prot, UIParam.TE) * 1e-3
    cfg.tr_s = gc.param_float(prot, UIParam.TR) * 1e-3
    cfg.flip_deg = gc.param_float(prot, UIParam.FLIP)
    cfg.fov_m = gc.param_float(prot, UIParam.FOV) * 1e-3
    cfg.slab_thickness_m = gc.param_float(prot, UIParam.SLICE_THICKNESS) * 1e-3
    cfg.slice_spacing_m = gc.param_float(prot, UIParam.SLICE_SPACING) * 1e-3
    cfg.nx_ro = gc.param_int(prot, UIParam.NX)
    cfg.npar = gc.param_int(prot, UIParam.NSLICES)
    cfg.bandwidth_hz_px = gc.param_float_optional(prot, UIParam.BANDWIDTH, gc.DEFAULT_BANDWIDTH_HZ_PX)
    cfg.rz = max(1, int(round(gc.param_float_optional(prot, UIParam.RZ, 1.0))))
    cfg.num_shots = gc.param_int_optional(prot, UIParam.NUM_SHOTS, 32)
    cfg.order_mode = _order_mode_name(gc.user_float(prot, USER_SLOT_ORDER_MODE, 1.0))
    cfg.etl = gc.param_int_optional(prot, UIParam.ETL, cfg.num_shots)
    cfg.trecovery_s = gc.param_float_optional(prot, UIParam.TRECOVERY, 1200.0) * 1e-3
    prep_value = prot[str(UIParam.PREPARATION_TYPE)].value
    cfg.prep_type = PreparationType(prep_value)
    cfg.ti_s = gc.user_float(prot, USER_SLOT_TI, 900.0) * 1e-3
    cfg.inv_mode = "adiabatic" if gc.user_float(prot, USER_SLOT_INV_MODE, 1.0) >= 0.5 else "hard"
    cfg.te_prep_s = gc.user_float(prot, USER_SLOT_TE_PREP, 50.0) * 1e-3
    cfg.refocus_mode = "adiabatic" if gc.user_float(prot, USER_SLOT_REFOCUS_MODE, 0.0) >= 0.5 else "hard"
    cfg.mt_enable = gc.user_float(prot, USER_SLOT_MT_ENABLE, 0.0) >= 0.5
    return cfg


def _compute_timing(opts: pp.Opts, cfg: _Config, strict: bool):
    gc.apply_system_derates(opts)

    rf, gz, gz_reph = gc.build_rf(opts, cfg.flip_deg, cfg.slab_thickness_m)

    echo = gc.compute_readout_and_echo_train(
        opts=opts,
        ro_axis=RO_AXIS,
        nx_ro=cfg.nx_ro,
        fov_ro_m=cfg.fov_m,
        bandwidth_hz_px=cfg.bandwidth_hz_px,
        slice_thickness_m=cfg.slab_thickness_m,
        num_echoes=NUM_ECHOES,
        echo_spacing_s=0.0,
        flyback=FLYBACK,
        strict=strict,
    )
    if echo is None:
        return None

    gz_spoil = pp.make_trapezoid(channel="z", area=gc.SPOIL_FACTOR_Z / cfg.slab_thickness_m, system=opts)
    _, max_par_area = gc.partition_geometry(cfg.npar, cfg.slice_spacing_m)
    gz_pe_template = pp.make_trapezoid(channel="z", area=max_par_area, system=opts) if max_par_area > 0.0 else None
    gz_pre_worst, gz_post_worst = gc.z_worst_case_trapezoids(gz_reph, gz_spoil, max_par_area, opts)

    d_rf = pp.calc_duration(rf, gz)
    d_pre = pp.calc_duration(echo["gx_pre"], gz_pre_worst)
    d_post = pp.calc_duration(echo["gx_spoil"], gz_post_worst)

    rf_center_s = pp.calc_rf_center(rf)[0]
    min_te_s = (d_rf - rf_center_s) + d_pre + echo["adc_center_s"]
    te_delay_s = cfg.te_s - min_te_s
    if te_delay_s < -1e-9 and strict:
        return None
    if te_delay_s < 0.0:
        te_delay_s = 0.0

    per_view_s = d_rf + d_pre + te_delay_s + echo["echo_train_span_s"] + d_post
    if cfg.mt_enable:
        rf_mt = mtc.build_mt_pulse(opts)
        mt_spoiler = mtc.build_mt_spoiler(opts, cfg.fov_m / cfg.nx_ro)
        per_view_s += pp.calc_duration(rf_mt) + pp.calc_duration(*mt_spoiler)
    else:
        rf_mt = None
        mt_spoiler = None

    min_block_s = per_view_s
    tr_delay_s = cfg.tr_s - min_block_s
    if tr_delay_s < -1e-9 and strict:
        return None
    if tr_delay_s < 0.0:
        tr_delay_s = 0.0

    voxel_size_m = cfg.fov_m / cfg.nx_ro
    worst_segment_len = max(1, min(cfg.etl, cfg.num_shots))
    worst_segment_duration_s = worst_segment_len * cfg.tr_s

    if cfg.prep_type == PreparationType.INVERSION:
        rf_inv = pc.build_inversion_pulse(opts, cfg.inv_mode)
        prep_spoiler = pc.build_inversion_spoiler(opts, voxel_size_m)
        prep_spoiler_duration_s = pp.calc_duration(*prep_spoiler)
        ti_delay_s = pc.ti_delay_seconds(
            cfg.ti_s, rf_inv, prep_spoiler_duration_s, worst_segment_duration_s, strict=strict
        )
        if ti_delay_s is None:
            return None
        prep_duration_s = pp.calc_duration(rf_inv) + prep_spoiler_duration_s + ti_delay_s
        t2prep_pulses = None
    else:
        t2prep_pulses = t2p.build_t2prep_pulses(opts, cfg.refocus_mode)
        rf90_down, rf180, rf90_up = t2prep_pulses
        delays = t2p.t2prep_delays_seconds(cfg.te_prep_s, rf90_down, rf180, rf90_up, strict=strict)
        if delays is None:
            return None
        delay1_s, delay2_s = delays
        prep_spoiler = t2p.build_t2prep_spoiler(opts, voxel_size_m)
        prep_spoiler_duration_s = pp.calc_duration(*prep_spoiler)
        prep_duration_s = (
            pp.calc_duration(rf90_down) + delay1_s + pp.calc_duration(rf180) + delay2_s
            + pp.calc_duration(rf90_up) + prep_spoiler_duration_s
        )
        rf_inv = None

    shot_s = prep_duration_s + worst_segment_duration_s + cfg.trecovery_s

    return {
        "rf": rf,
        "gz": gz,
        "gz_reph": gz_reph,
        "gz_spoil": gz_spoil,
        "gz_pe_template": gz_pe_template,
        "echo": echo,
        "te_delay_s": te_delay_s,
        "tr_delay_s": tr_delay_s,
        "min_block_s": min_block_s,
        "rf_inv": rf_inv,
        "t2prep_pulses": t2prep_pulses,
        "prep_spoiler": prep_spoiler,
        "prep_spoiler_duration_s": prep_spoiler_duration_s,
        "rf_mt": rf_mt,
        "mt_spoiler": mt_spoiler,
        "shot_s": shot_s,
    }


PLUGIN = GreMprageRadial3DPulseqSequence()


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
    parser = argparse.ArgumentParser(description="Generate a 3D stack-of-stars prepared GRE .seq offline.")
    parser.add_argument("-o", "--output", default="gre_mprage_radial_3d.seq", help="Output .seq file path")
    parser.add_argument("--te-ms", type=float)
    parser.add_argument("--tr-ms", type=float)
    parser.add_argument("--trecovery-ms", type=float)
    parser.add_argument("--etl", type=int)
    parser.add_argument("--prep-type", choices=["inversion", "t2_prep"])
    parser.add_argument("--ti-ms", type=float)
    parser.add_argument("--inversion-mode", choices=["hard", "adiabatic"])
    parser.add_argument("--te-prep-ms", type=float)
    parser.add_argument("--refocus-mode", choices=["hard", "adiabatic"])
    parser.add_argument("--mt", action="store_true")
    parser.add_argument("--flip-deg", type=float)
    parser.add_argument("--fov-mm", type=float)
    parser.add_argument("--slab-thickness-mm", type=float, dest="slice_thickness_mm")
    parser.add_argument("--partition-spacing-mm", type=float, dest="slice_spacing_mm")
    parser.add_argument("--nx", type=int)
    parser.add_argument("--npartitions", type=int, dest="nslices")
    parser.add_argument("--num-shots", type=int)
    parser.add_argument("--bandwidth-hz-px", type=float)
    parser.add_argument("--rz", type=float)
    parser.add_argument("--order-mode", choices=["uniform", "golden"])
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
    if args.trecovery_ms is not None:
        gc.set_protocol_value(protocol, UIParam.TRECOVERY, args.trecovery_ms)
    if args.etl is not None:
        gc.set_protocol_value(protocol, UIParam.ETL, args.etl)
    if args.prep_type is not None:
        gc.set_protocol_value(protocol, UIParam.PREPARATION_TYPE, args.prep_type)
    if args.ti_ms is not None:
        gc.set_protocol_value(protocol, UIParam.user_value(USER_SLOT_TI), args.ti_ms)
    if args.inversion_mode is not None:
        gc.set_protocol_value(protocol, UIParam.user_value(USER_SLOT_INV_MODE), 1.0 if args.inversion_mode == "adiabatic" else 0.0)
    if args.te_prep_ms is not None:
        gc.set_protocol_value(protocol, UIParam.user_value(USER_SLOT_TE_PREP), args.te_prep_ms)
    if args.refocus_mode is not None:
        gc.set_protocol_value(protocol, UIParam.user_value(USER_SLOT_REFOCUS_MODE), 1.0 if args.refocus_mode == "adiabatic" else 0.0)
    if args.mt:
        gc.set_protocol_value(protocol, UIParam.user_value(USER_SLOT_MT_ENABLE), 1.0)
    if args.flip_deg is not None:
        gc.set_protocol_value(protocol, UIParam.FLIP, args.flip_deg)
    if args.fov_mm is not None:
        gc.set_protocol_value(protocol, UIParam.FOV, args.fov_mm)
    if args.slice_thickness_mm is not None:
        gc.set_protocol_value(protocol, UIParam.SLICE_THICKNESS, args.slice_thickness_mm)
    if args.slice_spacing_mm is not None:
        gc.set_protocol_value(protocol, UIParam.SLICE_SPACING, args.slice_spacing_mm)
    if args.nx is not None:
        gc.set_protocol_value(protocol, UIParam.NX, args.nx)
    if args.nslices is not None:
        gc.set_protocol_value(protocol, UIParam.NSLICES, args.nslices)
    if args.num_shots is not None:
        gc.set_protocol_value(protocol, UIParam.NUM_SHOTS, args.num_shots)
    if args.bandwidth_hz_px is not None:
        gc.set_protocol_value(protocol, UIParam.BANDWIDTH, args.bandwidth_hz_px)
    if args.rz is not None:
        gc.set_protocol_value(protocol, UIParam.RZ, args.rz)
    if args.order_mode is not None:
        gc.set_protocol_value(protocol, UIParam.user_value(USER_SLOT_ORDER_MODE), 1.0 if args.order_mode == "golden" else 0.0)

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
