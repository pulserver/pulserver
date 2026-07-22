"""Standalone 3D inversion-prepared (MPRAGE) sequence plugin for pulserver.

This module implements the three mandatory module-level entry points
required by the bridge dispatcher (see ``gre_3d.py`` for the non-prepared 3D
GRE this extends, and ``_gre_common.py``/``pulserver`` for the shared
low-level building blocks):

- ``get_default_protocol(opts)``  — return the initial CV/protocol dictionary.
- ``validate_protocol(opts, protocol)``  — validate the current protocol and
  return a ``{"valid": bool, "duration": float | None, "info": str}`` dict.
- ``make_sequence(opts, protocol, output_path)``  — synthesise the sequence
  and write it to *output_path* using :func:`pulserver.io.write`.

Sequence structure — one inversion pulse gates one "segment" of PE1 (``NY``)
lines at a fixed partition; the segment length is ``ETL`` (chunked if
``ETL < NY``). Partitions are the outer loop (accelerated via ``Rz``); PE1 is
the inner, per-segment loop (accelerated via ``Ry``):

    for partition in partitions:                     # one inversion each
        non-selective inversion pulse (hard or adiabatic hypsec)
        3-axis non-selective spoiler
        TI delay (measured from the inversion RF center to the temporal
            center of the following segment — see ``pulserver.preparations``)
        for ky_chunk in chunks(PE1 lines, ETL):       # one shot
            for ky in ky_chunk:
                single-echo GRE view (RF spoiled, shares readout/echo-train
                and Z-channel gradient-combination helpers with gre_3d.py)
            recovery delay (``Trecovery``) before the next shot/inversion

``TE``/``TR`` retain the same per-view meaning as in ``gre_3d.py`` (``TR`` is
the inner, per-view period — the demo's "TRinner"). ``ETL`` (``IntKey.ETL``,
a real native GE variable already unused elsewhere) is reused here for
"views per inversion shot". ``Trecovery`` (``FloatKey.TRECOVERY``, native GE
``optrecovery``/``pitrecovery*``) is the post-segment recovery delay. TI and
the inversion-mode toggle (hard vs adiabatic) have no native GE ``pi*``
counterpart, so both are carried as ``opuser`` custom variables.

Usage
-----
Register the plugin in the interpreter tree by symlinking (or copying) this
file to the ``sequences/src/`` directory and creating a numbered alias::

    cp mprage_3d.py pulserver-interpreter/package/pulserver/sequences/src/mprage_3d.py
    ln -sf src/mprage_3d.py pulserver-interpreter/package/pulserver/sequences/sequence6.py
"""

from __future__ import annotations

import sys

import numpy as np
import pulserver.io as pio
import pulserver.pypulseq as pp
from pulserver import (
    BoolParam,
    Description,
    DropdownFloatParam,
    DropdownIntParam,
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

encoding = readout = sampling = system = excitation = preparations = pp

NUM_ECHOES = 1
FLYBACK = True

USER_SLOT_TI = 0
USER_SLOT_INV_MODE = 1


class Mprage3DPulseqSequence(Sequence):
    """Generate a 3D inversion-prepared (MPRAGE) Cartesian GRE sequence."""

    def get_default_protocol(self, opts: pp.Opts) -> dict[str, dict]:
        del opts
        protocol = {
            UIParam.TE: DropdownFloatParam(
                value=3.0, min=1.5, max=80.0, incr=0.1, unit="ms",
                options=[2.0, 3.0, 4.0, 8.0, 15.0], validate=Validate.NONE,
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
            UIParam.PHASE_FOV: DropdownFloatParam(
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
            UIParam.NY: DropdownIntParam(
                value=64, min=8, max=512, incr=1, options=[64, 128, 192, 256, 384], validate=Validate.NONE,
            ),
            UIParam.NSLICES: DropdownIntParam(
                value=8, min=1, max=256, incr=1, options=[8, 16, 32, 64, 128], validate=Validate.NONE,
            ),
            UIParam.RY: TypeinFloatParam(value=1.0, min=1.0, max=8.0, incr=1.0, unit="", validate=Validate.NONE),
            UIParam.RZ: TypeinFloatParam(value=1.0, min=1.0, max=8.0, incr=1.0, unit="", validate=Validate.NONE),
            UIParam.BANDWIDTH: TypeinFloatParam(
                value=125_000.0, min=5_000.0, max=500_000.0, incr=100.0,
                unit="Hz/px", validate=Validate.NONE,
            ),
            UIParam.SWAP_PHASE_FREQ: BoolParam(value=False, validate=Validate.NONE),
            UIParam.SEQUENCE_TYPE: make_enum_param(UIParam.SEQUENCE_TYPE, SequenceType.GRADIENT_ECHO),
            UIParam.user_name(USER_SLOT_TI): Description(text="Inversion time (TI)"),
            UIParam.user_value(USER_SLOT_TI): TypeinFloatParam(
                value=900.0, min=0.1, max=5000.0, incr=1.0, unit="ms", validate=Validate.NONE,
            ),
            UIParam.user_name(USER_SLOT_INV_MODE): Description(text="Inversion pulse (0=hard, 1=adiabatic)"),
            UIParam.user_value(USER_SLOT_INV_MODE): DropdownFloatParam(
                value=1.0, min=0.0, max=1.0, incr=1.0, unit="", options=[0.0, 1.0], validate=Validate.NONE,
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
        if not (0.0 < cfg.flip_deg <= 180.0):
            return {"valid": False, "duration": None, "info": "Flip angle must be in (0, 180] deg"}
        if cfg.nx_ro < 1 or cfg.ny_pe < 1 or cfg.npar < 1:
            return {"valid": False, "duration": None, "info": "NX, NY, and NSLICES (partitions) must be >= 1"}
        if cfg.bandwidth_hz_px <= 0.0:
            return {"valid": False, "duration": None, "info": "Bandwidth must be > 0"}
        if cfg.etl < 1:
            return {"valid": False, "duration": None, "info": "ETL must be >= 1"}
        if cfg.ti_s <= 0.0:
            return {"valid": False, "duration": None, "info": "TI must be > 0"}
        if cfg.trecovery_s < 0.0:
            return {"valid": False, "duration": None, "info": "Trecovery must be >= 0"}

        timing = _compute_public(opts=opts, cfg=cfg, strict=True)
        if timing is None:
            return {
                "valid": False,
                "duration": None,
                "info": "TE, TR, or TI infeasible for the requested gradients/ETL",
            }

        sampled_pe = pp.calc_sampled_lines(cfg.ny_pe, cfg.ry, 0)
        sampled_par = pp.calc_sampled_lines(cfg.npar, cfg.rz, 0)
        n_segments = len(pp.calc_chunk_indices(sampled_pe, cfg.etl))
        shot_s = timing["shot_s"]
        duration_s = shot_s * n_segments * len(sampled_par)
        return {"valid": True, "duration": duration_s, "info": f"TA = {duration_s:.2f} s"}

    def make_sequence(self, opts: pp.Opts, protocol: dict[str, dict], output_path: str) -> None:
        prot = dict_to_protocol(protocol)
        cfg = _read_protocol(prot)
        return _make_public_sequence(opts, cfg, output_path)

        timing = _compute_timing(opts=opts, cfg=cfg, strict=False)

        gz_reph = timing["gz_reph"]
        gz_spoil = timing["gz_spoil"]
        gz_pe_template = timing["gz_pe_template"]
        echo = timing["echo"]
        gy_template = timing["gy_template"]
        te_delay_s = timing["te_delay_s"]
        tr_delay_s = timing["tr_delay_s"]
        rf_inv = timing["rf_inv"]
        gx_inv, gy_inv, gz_inv = timing["inv_spoiler"]
        spoiler_duration_s = timing["spoiler_duration_s"]

        te_delay = pp.make_delay(te_delay_s) if te_delay_s > 0.0 else None
        tr_delay = pp.make_delay(tr_delay_s) if tr_delay_s > 0.0 else None
        trecovery_s_rounded = system.round_to_raster(cfg.trecovery_s)
        trecovery_delay = pp.make_delay(trecovery_s_rounded) if trecovery_s_rounded > 0.0 else None

        seq = pp.Sequence(opts)

        delta_k_pe = 1.0 / cfg.fov_pe_m
        phase_areas = (np.arange(cfg.ny_pe) - 0.5 * cfg.ny_pe) * delta_k_pe
        max_pe_area = float(np.max(np.abs(phase_areas)))
        sampled_pe = sampling.calc_sampled_lines(cfg.ny_pe, cfg.ry, 0)
        segments = sampling.calc_chunk_indices(sampled_pe, cfg.etl)

        par_areas, max_par_area = encoding.partition_geometry(cfg.npar, cfg.slice_spacing_m)
        sampled_par = sampling.calc_sampled_lines(cfg.npar, cfg.rz, 0)

        for par in sampled_par:
            z_scale = par_areas[par] / max_par_area if max_par_area > 0.0 else 0.0
            label_par = pp.make_label(type="SET", label="PAR", value=par)
            label_slc = pp.make_label(type="SET", label="SLC", value=0)

            for segment in segments:
                segment_duration_s = len(segment) * cfg.tr_s
                ti_delay_s = preparations.ti_delay_seconds(
                    cfg.ti_s, rf_inv, spoiler_duration_s, segment_duration_s, strict=False
                )
                ti_delay = pp.make_delay(ti_delay_s) if ti_delay_s > 0.0 else None

                seq.add_block(system.copy_event(rf_inv))
                seq.add_block(gx_inv, gy_inv, gz_inv)
                if ti_delay is not None:
                    seq.add_block(ti_delay)

                rf_phase_deg = 0.0
                rf_phase_inc_deg = 0.0

                for ky in segment:
                    y_scale = phase_areas[ky] / max_pe_area if max_pe_area > 0.0 else 0.0
                    gy_pre = pp.scale_grad(gy_template, y_scale)
                    gy_reph = pp.scale_grad(gy_template, -y_scale)
                    gz_pre_combined, gz_post_combined = encoding.combined_z_gradients(
                        z_scale, gz_pe_template, gz_reph, gz_spoil, opts
                    )

                    rf_curr = system.copy_event(timing["rf"])
                    rf_curr.phase_offset = np.deg2rad(rf_phase_deg)

                    label_lin = pp.make_label(type="SET", label="LIN", value=ky)

                    seq.add_block(rf_curr, timing["gz"], label_slc, label_par, label_lin)
                    seq.add_block(echo["gx_pre"], gy_pre, gz_pre_combined)
                    if te_delay is not None:
                        seq.add_block(te_delay)

                    readout.add_echo_train_blocks(
                        seq, echo, NUM_ECHOES, FLYBACK, rf_curr.phase_offset, emit_labels=False
                    )

                    seq.add_block(echo["gx_spoil"], gy_reph, gz_post_combined)
                    if tr_delay is not None:
                        seq.add_block(tr_delay)

                    rf_phase_deg = (rf_phase_deg + rf_phase_inc_deg) % 360.0
                    rf_phase_inc_deg = (rf_phase_inc_deg + excitation.RF_SPOILING_INC_DEG) % 360.0

                if trecovery_delay is not None:
                    seq.add_block(trecovery_delay)

        seq.set_definition("Name", "mprage_3d")
        seq.set_definition("FOV", [cfg.fov_ro_m, cfg.fov_pe_m, cfg.slice_spacing_m * cfg.npar])
        seq.set_definition("TE", cfg.te_s)
        seq.set_definition("TR", cfg.tr_s)
        seq.set_definition("TI", cfg.ti_s)
        seq.set_definition("Trecovery", cfg.trecovery_s)
        seq.set_definition("ETL", cfg.etl)
        seq.set_definition("InversionMode", cfg.inv_mode)
        seq.set_definition("Flip", cfg.flip_deg)
        seq.set_definition("ImagingMode", "3d")
        seq.set_definition("ReadoutAxis", cfg.ro_axis)
        seq.set_definition("PhaseAxis", cfg.pe_axis)
        seq.set_definition("BandwidthHzPerPx", cfg.bandwidth_hz_px)
        seq.set_definition("Ry", cfg.ry)
        seq.set_definition("Rz", cfg.rz)
        seq.set_definition("RfSpoilingIncDeg", excitation.RF_SPOILING_INC_DEG)
        seq.set_definition("Nx", cfg.nx_ro)
        seq.set_definition("Ny", cfg.ny_pe)
        seq.set_definition("NySampled", len(sampled_pe))
        seq.set_definition("NumPartitions", cfg.npar)
        pio.write(seq, output=output_path, remove_duplicates=False, check_timing=False)


class _Config:
    __slots__ = (
        "te_s", "tr_s", "flip_deg", "fov_ro_m", "fov_pe_m", "slab_thickness_m", "slice_spacing_m",
        "nx_ro", "ny_pe", "npar", "bandwidth_hz_px", "ry", "rz", "ro_axis", "pe_axis",
        "etl", "ti_s", "trecovery_s", "inv_mode",
    )


def _read_protocol(prot: dict) -> _Config:
    cfg = _Config()
    cfg.te_s = params.param_float(prot, UIParam.TE) * 1e-3
    cfg.tr_s = params.param_float(prot, UIParam.TR) * 1e-3
    cfg.flip_deg = params.param_float(prot, UIParam.FLIP)
    cfg.fov_ro_m = params.param_float(prot, UIParam.FOV) * 1e-3
    cfg.fov_pe_m = params.phase_fov_mm_from_protocol(prot) * 1e-3
    cfg.slab_thickness_m = params.param_float(prot, UIParam.SLICE_THICKNESS) * 1e-3
    cfg.slice_spacing_m = params.param_float(prot, UIParam.SLICE_SPACING) * 1e-3
    cfg.nx_ro = params.param_int(prot, UIParam.NX)
    cfg.ny_pe = params.param_int(prot, UIParam.NY)
    cfg.npar = params.param_int(prot, UIParam.NSLICES)
    cfg.bandwidth_hz_px = params.param_float_optional(prot, UIParam.BANDWIDTH, 125_000.0)
    cfg.ry = max(1, int(round(params.param_float_optional(prot, UIParam.RY, 1.0))))
    cfg.rz = max(1, int(round(params.param_float_optional(prot, UIParam.RZ, 1.0))))
    cfg.ro_axis, cfg.pe_axis = params.resolve_readout_phase_axes(prot)
    cfg.etl = params.param_int_optional(prot, UIParam.ETL, cfg.ny_pe)
    cfg.ti_s = params.user_float(prot, USER_SLOT_TI, 900.0) * 1e-3
    cfg.trecovery_s = params.param_float_optional(prot, UIParam.TRECOVERY, 1200.0) * 1e-3
    cfg.inv_mode = "adiabatic" if params.user_float(prot, USER_SLOT_INV_MODE, 1.0) >= 0.5 else "hard"
    return cfg


def _compute_timing(opts: pp.Opts, cfg: _Config, strict: bool):
    system.apply_system_derates(opts)

    rf, gz, gz_reph = excitation.slice_selective(opts, cfg.flip_deg, cfg.slab_thickness_m)

    echo = readout.compute_readout_and_echo_train(
        opts=opts,
        ro_axis=cfg.ro_axis,
        nx_ro=cfg.nx_ro,
        fov_ro_m=cfg.fov_ro_m,
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
    max_pe_area = 0.5 * cfg.ny_pe * (1.0 / cfg.fov_pe_m)
    gy_template = pp.make_trapezoid(channel=cfg.pe_axis, area=max_pe_area, system=opts)

    _, max_par_area = encoding.partition_geometry(cfg.npar, cfg.slice_spacing_m)
    gz_pe_template = pp.make_trapezoid(channel="z", area=max_par_area, system=opts) if max_par_area > 0.0 else None
    gz_pre_worst, gz_post_worst = encoding.z_worst_case_trapezoids(gz_reph, gz_spoil, max_par_area, opts)

    d_rf = pp.calc_duration(rf, gz)
    d_pre = pp.calc_duration(echo["gx_pre"], gy_template, gz_pre_worst)
    d_post = pp.calc_duration(echo["gx_spoil"], gy_template, gz_post_worst)

    rf_center_s = pp.calc_rf_center(rf)[0]
    min_te_s = (d_rf - rf_center_s) + d_pre + echo["adc_center_s"]
    te_delay_s = cfg.te_s - min_te_s
    if te_delay_s < -1e-9 and strict:
        return None
    if te_delay_s < 0.0:
        te_delay_s = 0.0

    min_block_s = d_rf + d_pre + te_delay_s + echo["echo_train_span_s"] + d_post
    tr_delay_s = cfg.tr_s - min_block_s
    if tr_delay_s < -1e-9 and strict:
        return None
    if tr_delay_s < 0.0:
        tr_delay_s = 0.0

    voxel_size_m = cfg.fov_pe_m / cfg.ny_pe
    inv_module = preparations.inversion(opts, cfg.inv_mode, voxel_size_m)
    rf_inv = inv_module.rf
    inv_spoiler = inv_module.spoiler
    spoiler_duration_s = inv_module.spoiler_duration_s

    # Worst-case (largest) segment gates the strict TI feasibility check —
    # every segment except possibly the last is exactly ETL lines long.
    worst_segment_len = max(1, min(cfg.etl, cfg.ny_pe))
    worst_segment_duration_s = worst_segment_len * cfg.tr_s
    ti_delay_s = preparations.ti_delay_seconds(cfg.ti_s, rf_inv, spoiler_duration_s, worst_segment_duration_s, strict=strict)
    if ti_delay_s is None:
        return None

    inv_duration_s = pp.calc_duration(rf_inv)
    shot_s = inv_duration_s + spoiler_duration_s + ti_delay_s + worst_segment_duration_s + cfg.trecovery_s

    return {
        "rf": rf,
        "gz": gz,
        "gz_reph": gz_reph,
        "gz_spoil": gz_spoil,
        "gz_pe_template": gz_pe_template,
        "gy_template": gy_template,
        "echo": echo,
        "te_delay_s": te_delay_s,
        "tr_delay_s": tr_delay_s,
        "min_block_s": min_block_s,
        "rf_inv": rf_inv,
        "inv_spoiler": inv_spoiler,
        "spoiler_duration_s": spoiler_duration_s,
        "shot_s": shot_s,
    }


def _compute_public(opts: pp.Opts, cfg: _Config, strict: bool):
    inversion = pp.make_inversion_pulse(adiabatic=cfg.inv_mode == "adiabatic", system=opts)
    pulse = pp.make_slice_selective_pulse(
        np.deg2rad(cfg.flip_deg), cfg.slab_thickness_m, system=opts
    )
    line = pp.make_line_readout(
        opts,
        (cfg.fov_ro_m, cfg.fov_pe_m, cfg.slice_spacing_m * cfg.npar),
        (cfg.nx_ro, cfg.ny_pe, cfg.npar),
        bandwidth_hz_px=cfg.bandwidth_hz_px,
        spoil_position="post",
        spoil_cycles=1.0,
    )
    d_pulse = sum(pp.calc_duration(*block) for block in pulse)
    rf_center = pp.calc_rf_center(pulse.rf)[0] + pulse.rf.delay
    raster = opts.block_duration_raster
    te_delay_s = round((cfg.te_s - (d_pulse - rf_center) - line.t_first_echo_s) / raster) * raster
    if te_delay_s < -1e-9 and strict:
        return None
    te_delay_s = max(0.0, te_delay_s)
    min_block_s = d_pulse + te_delay_s + line.duration
    tr_delay_s = round((cfg.tr_s - min_block_s) / raster) * raster
    if tr_delay_s < -1e-9 and strict:
        return None
    inversion_duration = sum(pp.calc_duration(*block) for block in inversion)
    ti_delay_s = round((cfg.ti_s - inversion_duration) / raster) * raster
    if ti_delay_s < -1e-9 and strict:
        return None
    ti_delay_s = max(0.0, ti_delay_s)
    shot_s = inversion_duration + ti_delay_s + cfg.etl * cfg.tr_s + cfg.trecovery_s
    return {
        "inversion": inversion,
        "pulse": pulse,
        "line": line,
        "te_delay_s": te_delay_s,
        "tr_delay_s": max(0.0, tr_delay_s),
        "ti_delay_s": ti_delay_s,
        "min_block_s": min_block_s,
        "shot_s": shot_s,
    }


def _make_public_sequence(opts: pp.Opts, cfg: _Config, output_path: str) -> None:
    timing = _compute_public(opts, cfg, strict=False)
    seq = pp.Sequence(opts)
    sampled_pe = pp.calc_sampled_lines(cfg.ny_pe, cfg.ry, 0)
    sampled_par = pp.calc_sampled_lines(cfg.npar, cfg.rz, 0)
    coords = np.asarray([(ky, par) for par in sampled_par for ky in sampled_pe], dtype=int)
    segments = [coords[start : start + cfg.etl] for start in range(0, len(coords), cfg.etl)]
    phases = pp.make_rf_spoiling_schedule(len(coords))
    te_delay = pp.make_delay(timing["te_delay_s"]) if timing["te_delay_s"] > 0 else None
    tr_delay = pp.make_delay(timing["tr_delay_s"]) if timing["tr_delay_s"] > 0 else None
    ti_delay = pp.make_delay(timing["ti_delay_s"]) if timing["ti_delay_s"] > 0 else None
    recovery_s = round(cfg.trecovery_s / opts.block_duration_raster) * opts.block_duration_raster
    recovery = pp.make_delay(recovery_s) if recovery_s > 0 else None
    phase_idx = 0
    for segment in segments:
        for block in timing["inversion"]:
            seq.add_block(*block)
        if ti_delay is not None:
            seq.add_block(ti_delay)
        for ky, par in segment:
            phase = float(phases[phase_idx])
            timing["pulse"].set_state(phase_offset_rad=phase)
            for block in timing["pulse"]:
                seq.add_block(*block)
            if te_delay is not None:
                seq.add_block(te_delay)
            timing["line"].set_state(
                lin_idx=int(ky), par_idx=int(par), adc_phase_rad=phase
            )
            for block in timing["line"]:
                seq.add_block(*block)
            if tr_delay is not None:
                seq.add_block(tr_delay)
            phase_idx += 1
        if recovery is not None:
            seq.add_block(recovery)
    seq.set_definition("Name", "mprage_3d")
    seq.set_definition("FOV", [cfg.fov_ro_m, cfg.fov_pe_m, cfg.slice_spacing_m * cfg.npar])
    seq.set_definition("TE", cfg.te_s)
    seq.set_definition("TR", cfg.tr_s)
    seq.set_definition("TI", cfg.ti_s)
    seq.set_definition("Trecovery", cfg.trecovery_s)
    seq.set_definition("ETL", cfg.etl)
    seq.set_definition("InversionMode", cfg.inv_mode)
    seq.set_definition("ImagingMode", "3d")
    seq.set_definition("ReadoutAxis", "x")
    seq.set_definition("PhaseAxis", "y")
    seq.set_definition("RfSpoilingIncDeg", 117.0)
    pio.write(seq, output=output_path, remove_duplicates=False, check_timing=False)


PLUGIN = Mprage3DPulseqSequence()


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
    ('--ti-ms', UIParam.user_value(USER_SLOT_TI), float, ""),
    ('--trecovery-ms', UIParam.TRECOVERY, float, ""),
    ('--etl', UIParam.ETL, int, ""),
    ('--inversion-mode', UIParam.user_value(USER_SLOT_INV_MODE), {'hard': 0.0, 'adiabatic': 1.0}, ""),
    ('--flip-deg', UIParam.FLIP, float, ""),
    ('--fov-mm', UIParam.FOV, float, ""),
    ('--phase-fov-mm', UIParam.PHASE_FOV, float, ""),
    ('--slab-thickness-mm', UIParam.SLICE_THICKNESS, float, ""),
    ('--partition-spacing-mm', UIParam.SLICE_SPACING, float, ""),
    ('--nx', UIParam.NX, int, ""),
    ('--ny', UIParam.NY, int, ""),
    ('--npartitions', UIParam.NSLICES, int, ""),
    ('--bandwidth-hz-px', UIParam.BANDWIDTH, float, ""),
    ('--ry', UIParam.RY, float, ""),
    ('--rz', UIParam.RZ, float, ""),
    ('--swap-phase-freq', UIParam.SWAP_PHASE_FREQ, ("const", True), ""),
]

if __name__ == "__main__":
    raise SystemExit(
        run_cli(
            PLUGIN,
            sys.argv[1:],
            arg_map=_ARG_MAP,
            description='Generate a 3D inversion-prepared (MPRAGE) .seq offline.',
            default_output='mprage_3d.seq',
        )
    )
