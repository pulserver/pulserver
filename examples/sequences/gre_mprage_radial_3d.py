"""Standalone 3D non-Cartesian ("stack-of-stars") inversion/T2-prepared GRE.

This module implements the three mandatory module-level entry points
required by the bridge dispatcher (see ``mprage_3d.py`` for the Cartesian
inversion-prepared GRE this extends, ``gre_radial_3d.py`` for the
non-prepared stack-of-stars sibling, and ``_gre_common.py`` /
``pulserver.preparations`` for the
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
  (``pulserver.preparations``, same as ``mprage_3d.py``).
- ``t2_prep``: composite 90-tau/2-180-tau/2-(-90) + spoiler
  (``pulserver.preparations``) — ``TE_prep`` sets the T2 weighting directly, so
  (unlike TI) there is no extra post-module centering delay; the segment
  starts right after the spoiler.

An independent, *additive* MT saturation toggle (``preparations.mt_saturation``) can be
stacked on top of either prep — real MT-prepared sequences apply the MT
pulse every TR (steady-state effect), not once per segment, so it is not a
third ``PreparationType`` option (which would also violate the closed-enum
rule: GE's real UI has no generic "prep type = MT" slot).

``TE``/``TR`` retain their per-view meaning; ``ETL``/``Trecovery`` are
reused from the MPRAGE family. TI, ``TE_prep``, the inversion/refocus-mode
toggles, and the MT enable/params all lack a native GE counterpart, so they
are ``opuser`` custom variables. Trajectory is plain full-echo radial
(rotated spokes, ``pulserver.pypulseq.make_rotation`` — no ``pulserver.pypulseq.arbgrad``
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

import sys

import numpy as np
import pulserver.io as pio
import pulserver.pypulseq as pp
from pulserver import (
    Description,
    DropdownFloatParam,
    DropdownIntParam,
    PreparationType,
    Sequence,
    SequenceType,
    TypeinFloatParam,
    UIParam,
    Validate,
    dict_to_protocol,
    make_enum_param,
    params,
    protocol_to_dict,
    run_cli,
)
from pulserver.pypulseq import _gradients as encoding
from pulserver.pypulseq import _readout as readout
from pulserver.pypulseq import _sampling as sampling
from pulserver.pypulseq import _system as system
from pulserver.pypulseq import arbgrad
from pulserver.pypulseq._rf import _excitation_helpers as excitation
from pulserver.pypulseq._rf import _preparation_helpers as preparations
from scipy.spatial.transform import Rotation

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


class GreMprageRadial3DPulseqSequence(Sequence):
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
                value=system.DEFAULT_BANDWIDTH_HZ_PX, min=5_000.0, max=500_000.0, incr=100.0,
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

        sampled_par = sampling.sampled_lines(cfg.npar, cfg.rz, 0)
        n_segments = len(sampling.chunk_indices(list(range(cfg.num_shots)), cfg.etl))
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
        trecovery_delay = pp.make_delay(system.round_to_raster(cfg.trecovery_s)) if cfg.trecovery_s > 0.0 else None

        seq = pp.Sequence(opts)

        angles = arbgrad.shot_angles(cfg.num_shots, mode=cfg.order_mode)
        segments = sampling.chunk_indices(list(range(cfg.num_shots)), cfg.etl)

        par_areas, max_par_area = encoding.partition_geometry(cfg.npar, cfg.slice_spacing_m)
        sampled_par = sampling.sampled_lines(cfg.npar, cfg.rz, 0)

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
                    rotation = pp.make_rotation(Rotation.from_euler("z", angle))
                    label_lin = pp.make_label(type="SET", label="LIN", value=shot)

                    gz_pre_combined, gz_post_combined = encoding.combined_z_gradients(
                        z_scale, gz_pe_template, gz_reph, gz_spoil, opts
                    )

                    if cfg.mt_enable:
                        seq.add_block(system.copy_event(timing["rf_mt"]))
                        seq.add_block(*timing["mt_spoiler"])

                    rf_curr = system.copy_event(timing["rf"])
                    rf_curr.phase_offset = np.deg2rad(rf_phase_deg)
                    adc_curr = system.copy_event(echo["adc"])
                    adc_curr.phase_offset = rf_curr.phase_offset

                    seq.add_block(rf_curr, timing["gz"], label_slc, label_par, label_lin)
                    seq.add_block(echo["gx_pre"], gz_pre_combined, rotation)
                    if te_delay is not None:
                        seq.add_block(te_delay)

                    # NUM_ECHOES is pinned to 1 (single-echo GRE view), so the
                    # echo train is just one rotated readout block — emitted
                    # directly (not via readout.add_echo_train_blocks, which has
                    # no extension-passing hook for a non-Cartesian rotation).
                    seq.add_block(echo["gx_echo"], adc_curr, rotation)

                    seq.add_block(echo["gx_spoil"], gz_post_combined, rotation)
                    if tr_delay is not None:
                        seq.add_block(tr_delay)

                    rf_phase_deg = (rf_phase_deg + rf_phase_inc_deg) % 360.0
                    rf_phase_inc_deg = (rf_phase_inc_deg + excitation.RF_SPOILING_INC_DEG) % 360.0

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
        seq.set_definition("RfSpoilingIncDeg", excitation.RF_SPOILING_INC_DEG)
        seq.set_definition("Nx", cfg.nx_ro)
        seq.set_definition("NumShots", cfg.num_shots)
        seq.set_definition("NumPartitions", cfg.npar)
        pio.write(seq, output=output_path, remove_duplicates=False, check_timing=False)


def _emit_prep_module(seq, timing: dict, cfg: _Config, segment_duration_s: float) -> None:
    """Emit the (inversion or T2-prep) magnetization-prep blocks for one
    segment. TI feasibility was already checked in ``_compute_timing``; here
    we only recompute the (non-strict, clipped) delay for the actual
    segment length, mirroring ``mprage_3d.py``."""
    if cfg.prep_type == PreparationType.INVERSION:
        rf_inv = timing["rf_inv"]
        spoiler = timing["prep_spoiler"]
        spoiler_duration_s = timing["prep_spoiler_duration_s"]
        ti_delay_s = preparations.ti_delay_seconds(cfg.ti_s, rf_inv, spoiler_duration_s, segment_duration_s, strict=False)
        seq.add_block(system.copy_event(rf_inv))
        seq.add_block(*spoiler)
        if ti_delay_s > 0.0:
            seq.add_block(pp.make_delay(ti_delay_s))
    else:
        t2prep = timing["t2prep"]
        seq.add_block(system.copy_event(t2prep.rf90_down))
        if t2prep.delay1_s > 0.0:
            seq.add_block(pp.make_delay(t2prep.delay1_s))
        seq.add_block(system.copy_event(t2prep.rf180))
        if t2prep.delay2_s > 0.0:
            seq.add_block(pp.make_delay(t2prep.delay2_s))
        seq.add_block(system.copy_event(t2prep.rf90_up))
        seq.add_block(*t2prep.spoiler)


class _Config:
    __slots__ = (
        "te_s", "tr_s", "flip_deg", "fov_m", "slab_thickness_m", "slice_spacing_m",
        "nx_ro", "npar", "bandwidth_hz_px", "rz", "num_shots", "order_mode",
        "etl", "trecovery_s", "prep_type", "ti_s", "inv_mode", "te_prep_s", "refocus_mode", "mt_enable",
    )


def _read_protocol(prot: dict) -> _Config:
    cfg = _Config()
    cfg.te_s = params.param_float(prot, UIParam.TE) * 1e-3
    cfg.tr_s = params.param_float(prot, UIParam.TR) * 1e-3
    cfg.flip_deg = params.param_float(prot, UIParam.FLIP)
    cfg.fov_m = params.param_float(prot, UIParam.FOV) * 1e-3
    cfg.slab_thickness_m = params.param_float(prot, UIParam.SLICE_THICKNESS) * 1e-3
    cfg.slice_spacing_m = params.param_float(prot, UIParam.SLICE_SPACING) * 1e-3
    cfg.nx_ro = params.param_int(prot, UIParam.NX)
    cfg.npar = params.param_int(prot, UIParam.NSLICES)
    cfg.bandwidth_hz_px = params.param_float_optional(prot, UIParam.BANDWIDTH, system.DEFAULT_BANDWIDTH_HZ_PX)
    cfg.rz = max(1, int(round(params.param_float_optional(prot, UIParam.RZ, 1.0))))
    cfg.num_shots = params.param_int_optional(prot, UIParam.NUM_SHOTS, 32)
    cfg.order_mode = _order_mode_name(params.user_float(prot, USER_SLOT_ORDER_MODE, 1.0))
    cfg.etl = params.param_int_optional(prot, UIParam.ETL, cfg.num_shots)
    cfg.trecovery_s = params.param_float_optional(prot, UIParam.TRECOVERY, 1200.0) * 1e-3
    prep_value = prot[str(UIParam.PREPARATION_TYPE)].value
    cfg.prep_type = PreparationType(prep_value)
    cfg.ti_s = params.user_float(prot, USER_SLOT_TI, 900.0) * 1e-3
    cfg.inv_mode = "adiabatic" if params.user_float(prot, USER_SLOT_INV_MODE, 1.0) >= 0.5 else "hard"
    cfg.te_prep_s = params.user_float(prot, USER_SLOT_TE_PREP, 50.0) * 1e-3
    cfg.refocus_mode = "adiabatic" if params.user_float(prot, USER_SLOT_REFOCUS_MODE, 0.0) >= 0.5 else "hard"
    cfg.mt_enable = params.user_float(prot, USER_SLOT_MT_ENABLE, 0.0) >= 0.5
    return cfg


def _compute_timing(opts: pp.Opts, cfg: _Config, strict: bool):
    system.apply_system_derates(opts)

    rf, gz, gz_reph = excitation.slice_selective(opts, cfg.flip_deg, cfg.slab_thickness_m)

    echo = readout.compute_readout_and_echo_train(
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

    gz_spoil = pp.make_trapezoid(channel="z", area=encoding.SPOIL_FACTOR_Z / cfg.slab_thickness_m, system=opts)
    _, max_par_area = encoding.partition_geometry(cfg.npar, cfg.slice_spacing_m)
    gz_pe_template = pp.make_trapezoid(channel="z", area=max_par_area, system=opts) if max_par_area > 0.0 else None
    gz_pre_worst, gz_post_worst = encoding.z_worst_case_trapezoids(gz_reph, gz_spoil, max_par_area, opts)

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
        mt_module = preparations.mt_saturation(opts, voxel_size_m=cfg.fov_m / cfg.nx_ro)
        rf_mt = mt_module.rf
        mt_spoiler = mt_module.spoiler
        per_view_s += mt_module.duration_s
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
        inv_module = preparations.inversion(opts, cfg.inv_mode, voxel_size_m)
        rf_inv = inv_module.rf
        prep_spoiler = inv_module.spoiler
        prep_spoiler_duration_s = inv_module.spoiler_duration_s
        ti_delay_s = inv_module.ti_delay_s(cfg.ti_s, worst_segment_duration_s, strict)
        if ti_delay_s is None:
            return None
        prep_duration_s = inv_module.rf_duration_s + prep_spoiler_duration_s + ti_delay_s
        t2prep = None
    else:
        t2prep = preparations.t2_prep(
            opts, cfg.te_prep_s, refocus_mode=cfg.refocus_mode, voxel_size_m=voxel_size_m, strict=strict
        )
        if t2prep is None:
            return None
        prep_spoiler = t2prep.spoiler
        prep_spoiler_duration_s = pp.calc_duration(*t2prep.spoiler)
        prep_duration_s = t2prep.duration_s
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
        "t2prep": t2prep,
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


_ARG_MAP = [
    ('--te-ms', UIParam.TE, float, ""),
    ('--tr-ms', UIParam.TR, float, ""),
    ('--trecovery-ms', UIParam.TRECOVERY, float, ""),
    ('--etl', UIParam.ETL, int, ""),
    ('--prep-type', UIParam.PREPARATION_TYPE, {'inversion': 'inversion', 't2_prep': 't2_prep'}, ""),
    ('--ti-ms', UIParam.user_value(USER_SLOT_TI), float, ""),
    ('--inversion-mode', UIParam.user_value(USER_SLOT_INV_MODE), {'hard': 0.0, 'adiabatic': 1.0}, ""),
    ('--te-prep-ms', UIParam.user_value(USER_SLOT_TE_PREP), float, ""),
    ('--refocus-mode', UIParam.user_value(USER_SLOT_REFOCUS_MODE), {'hard': 0.0, 'adiabatic': 1.0}, ""),
    ('--mt', UIParam.user_value(USER_SLOT_MT_ENABLE), ("const", 1.0), ""),
    ('--flip-deg', UIParam.FLIP, float, ""),
    ('--fov-mm', UIParam.FOV, float, ""),
    ('--slab-thickness-mm', UIParam.SLICE_THICKNESS, float, ""),
    ('--partition-spacing-mm', UIParam.SLICE_SPACING, float, ""),
    ('--nx', UIParam.NX, int, ""),
    ('--npartitions', UIParam.NSLICES, int, ""),
    ('--num-shots', UIParam.NUM_SHOTS, int, ""),
    ('--bandwidth-hz-px', UIParam.BANDWIDTH, float, ""),
    ('--rz', UIParam.RZ, float, ""),
    ('--order-mode', UIParam.user_value(USER_SLOT_ORDER_MODE), {'uniform': 0.0, 'golden': 1.0}, ""),
]

if __name__ == "__main__":
    raise SystemExit(
        run_cli(
            PLUGIN,
            sys.argv[1:],
            arg_map=_ARG_MAP,
            description='Generate a 3D stack-of-stars prepared GRE .seq offline.',
            default_output='gre_mprage_radial_3d.seq',
        )
    )
