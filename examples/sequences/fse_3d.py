"""Standalone 3D fast/turbo spin-echo (FSE) sequence plugin for pulserver.

This module implements the three mandatory module-level entry points
required by the bridge dispatcher (see ``gre_3d.py`` for the Cartesian 3D
GRE this parallels, ``fse_2d.py`` for the 2D FSE this mirrors, and
``_gre_common.py`` / ``_fse_common.py`` / ``_diffusion_common.py`` for the
shared building blocks):

- ``get_default_protocol(opts)``  — return the initial CV/protocol dictionary.
- ``validate_protocol(opts, protocol)``  — validate the current protocol and
  return a ``{"valid": bool, "duration": float | None, "info": str}`` dict.
- ``make_sequence(opts, protocol, output_path)``  — synthesise the sequence
  and write it to *output_path* using :func:`pulserver.io.write`.

Same 90-crusher-[diffusion]-CPMG-train structure as ``fse_2d.py`` (see that
file's docstring for the TE/ESP/CPMG-timing rationale, the flip-angle role
convention — ``UIParam.FLIP`` drives refocusing, excitation is fixed — and
the net-zero-per-echo-cycle bookkeeping — a real bug there, fixed and
covered by a regression test, applies identically here), stacked across partitions
(accelerated via ``Rz``): kz is FIXED for an entire shot (one partition),
so it gets the exact same symmetric prephase/rephase treatment as ky (a
constant-value pair every echo, net zero per cycle) rather than being
combined into the crusher — unlike ``gre_3d.py``'s ``combined_z_gradients``,
which shares a block with the (fixed) slab rephase; here the slab rephase
happens earlier, symmetric with the 90, before the crushers/180s.

Not implemented in this pass (same as ``fse_2d.py``/``epi_3d.py``):
optional inversion before the 90.

Usage
-----
Register the plugin in the interpreter tree by symlinking (or copying) this
file to the ``sequences/src/`` directory and creating a numbered alias::

    cp fse_3d.py pulserver-interpreter/package/pulserver/sequences/src/fse_3d.py
    ln -sf src/fse_3d.py pulserver-interpreter/package/pulserver/sequences/sequence16.py
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
from pulserver.pypulseq._rf import _excitation_helpers as excitation
from pulserver.pypulseq._rf import _preparation_helpers as preparations

# See fse_2d.py: refocusing scheme (TRAPS on/off) carried as an opuser
# custom variable; the refocusing flip ANGLE reuses UIParam.FLIP directly.
USER_SLOT_REFOCUS_SCHEME = 0


class Fse3DPulseqSequence(Sequence):
    """Generate a 3D turbo/fast spin-echo (CPMG) sequence."""

    def get_default_protocol(self, opts: pp.Opts) -> dict[str, dict]:
        del opts
        protocol = {
            UIParam.TE: DropdownFloatParam(
                value=12.0, min=5.0, max=50.0, incr=0.1, unit="ms",
                options=[8.0, 10.0, 12.0, 16.0, 20.0], validate=Validate.NONE,
            ),
            UIParam.TR: DropdownFloatParam(
                value=1000.0, min=200.0, max=10000.0, incr=1.0, unit="ms",
                options=[500.0, 1000.0, 1500.0, 2000.0, 3000.0], validate=Validate.NONE,
            ),
            # Spin-echo role: FLIP drives the REFOCUSING pulse (the 90
            # excitation is fixed, not user-selectable).
            UIParam.FLIP: DropdownFloatParam(
                value=readout.DEFAULT_REFOCUS_FLIP_DEG, min=90.0, max=180.0, incr=1.0, unit="deg",
                options=[90.0, 120.0, 150.0, 180.0], validate=Validate.NONE,
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
                value=16, min=1, max=256, incr=1, options=[4, 8, 16, 32, 64], validate=Validate.NONE,
            ),
            UIParam.RY: TypeinFloatParam(value=1.0, min=1.0, max=8.0, incr=1.0, unit="", validate=Validate.NONE),
            UIParam.RZ: TypeinFloatParam(value=1.0, min=1.0, max=8.0, incr=1.0, unit="", validate=Validate.NONE),
            UIParam.BANDWIDTH: TypeinFloatParam(
                value=125_000.0, min=5_000.0, max=500_000.0, incr=100.0,
                unit="Hz/px", validate=Validate.NONE,
            ),
            UIParam.DIFFUSION_BVALUES: TypeinFloatParam(
                value=0.0, min=0.0, max=5000.0, incr=1.0, unit="s/mm2", validate=Validate.NONE,
            ),
            UIParam.DIFFUSION_DIRECTIONS: DropdownIntParam(
                value=3, min=1, max=32, incr=1, options=[1, 3, 6], validate=Validate.NONE,
            ),
            UIParam.SEQUENCE_TYPE: make_enum_param(UIParam.SEQUENCE_TYPE, SequenceType.SPIN_ECHO),
            UIParam.user_name(USER_SLOT_REFOCUS_SCHEME): Description(
                text="Refocusing scheme (0=constant, 1=variable/TRAPS)"
            ),
            UIParam.user_value(USER_SLOT_REFOCUS_SCHEME): DropdownFloatParam(
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
        if not (0.0 < cfg.refocus_flip_deg <= 180.0):
            return {"valid": False, "duration": None, "info": "Flip (refocusing) angle must be in (0, 180] deg"}
        if cfg.nx_ro < 1 or cfg.ny_pe < 1 or cfg.npar < 1:
            return {"valid": False, "duration": None, "info": "NX, NY, and NSLICES (partitions) must be >= 1"}
        if cfg.bandwidth_hz_px <= 0.0:
            return {"valid": False, "duration": None, "info": "Bandwidth must be > 0"}
        if cfg.etl < 1:
            return {"valid": False, "duration": None, "info": "ETL must be >= 1"}
        if cfg.b_value_s_mm2 < 0.0:
            return {"valid": False, "duration": None, "info": "Diffusion b-value must be >= 0"}

        compute_timing = _compute_timing_legacy if cfg.b_value_s_mm2 > 0.0 else _compute_timing_surgery
        timing = compute_timing(opts=opts, cfg=cfg, strict=True)
        if timing is None:
            return {
                "valid": False,
                "duration": None,
                "info": "TE (ESP) too short for the 90/180/crushers/diffusion/readout, or TR too short",
            }

        n_shots = _n_shots(cfg)
        sampled_par = sampling.sampled_lines(cfg.npar, cfg.rz, 0)
        duration_s = cfg.tr_s * float(n_shots) * float(len(sampled_par))
        return {"valid": True, "duration": duration_s, "info": f"TA = {duration_s:.2f} s"}

    def make_sequence(self, opts: pp.Opts, protocol: dict[str, dict], output_path: str) -> None:
        prot = dict_to_protocol(protocol)
        cfg = _read_protocol(prot)

        seq = pp.Sequence(opts)
        if cfg.b_value_s_mm2 > 0.0:
            n_shots, n_directions = _build_legacy(seq, opts, cfg)
        else:
            n_shots, n_directions = _build_surgery(seq, opts, cfg)

        seq.set_definition("Name", "fse_3d")
        seq.set_definition("FOV", [cfg.fov_ro_m, cfg.fov_pe_m, cfg.slice_spacing_m * cfg.npar])
        seq.set_definition("TE", cfg.te_s)
        seq.set_definition("TR", cfg.tr_s)
        seq.set_definition("Flip", cfg.refocus_flip_deg)
        seq.set_definition("ExcitationFlip", readout.EXCITATION_FLIP_DEG)
        seq.set_definition("RefocusVariable", cfg.refocus_variable)
        seq.set_definition("ImagingMode", "3d")
        seq.set_definition("ETL", cfg.etl)
        seq.set_definition("NumShots", n_shots)
        seq.set_definition("BValue", cfg.b_value_s_mm2)
        seq.set_definition("DiffusionDirections", n_directions)
        seq.set_definition("BandwidthHzPerPx", cfg.bandwidth_hz_px)
        seq.set_definition("Ry", cfg.ry)
        seq.set_definition("Rz", cfg.rz)
        seq.set_definition("Nx", cfg.nx_ro)
        seq.set_definition("Ny", cfg.ny_pe)
        seq.set_definition("NumPartitions", cfg.npar)
        pio.write(seq, output=output_path, remove_duplicates=False, check_timing=False)


def _n_shots(cfg: _Config) -> int:
    sampled_pe = sampling.sampled_lines(cfg.ny_pe, cfg.ry, 0)
    return len(range(0, len(sampled_pe), cfg.etl))


class _Config:
    __slots__ = (
        "te_s", "tr_s", "refocus_flip_deg", "refocus_variable",
        "fov_ro_m", "fov_pe_m", "slab_thickness_m", "slice_spacing_m",
        "nx_ro", "ny_pe", "npar", "etl", "ry", "rz", "bandwidth_hz_px", "b_value_s_mm2", "n_directions",
    )


def _read_protocol(prot: dict) -> _Config:
    cfg = _Config()
    cfg.te_s = params.param_float(prot, UIParam.TE) * 1e-3
    cfg.tr_s = params.param_float(prot, UIParam.TR) * 1e-3
    # Spin-echo role: FLIP is the refocusing angle (see module docstring).
    cfg.refocus_flip_deg = params.param_float(prot, UIParam.FLIP)
    cfg.refocus_variable = params.user_float(prot, USER_SLOT_REFOCUS_SCHEME, 0.0) >= 0.5
    cfg.fov_ro_m = params.param_float(prot, UIParam.FOV) * 1e-3
    cfg.fov_pe_m = params.phase_fov_mm_from_protocol(prot) * 1e-3
    cfg.slab_thickness_m = params.param_float(prot, UIParam.SLICE_THICKNESS) * 1e-3
    cfg.slice_spacing_m = params.param_float(prot, UIParam.SLICE_SPACING) * 1e-3
    cfg.nx_ro = params.param_int(prot, UIParam.NX)
    cfg.ny_pe = params.param_int(prot, UIParam.NY)
    cfg.npar = params.param_int(prot, UIParam.NSLICES)
    cfg.etl = params.param_int_optional(prot, UIParam.ETL, cfg.ny_pe)
    cfg.ry = max(1, int(round(params.param_float_optional(prot, UIParam.RY, 1.0))))
    cfg.rz = max(1, int(round(params.param_float_optional(prot, UIParam.RZ, 1.0))))
    cfg.bandwidth_hz_px = params.param_float_optional(prot, UIParam.BANDWIDTH, system.DEFAULT_BANDWIDTH_HZ_PX)
    cfg.b_value_s_mm2 = params.param_float_optional(prot, UIParam.DIFFUSION_BVALUES, 0.0)
    cfg.n_directions = params.param_int_optional(prot, UIParam.DIFFUSION_DIRECTIONS, 3)
    return cfg


def _compute_timing_legacy(opts: pp.Opts, cfg: _Config, strict: bool):
    """Legacy (unfused) timing — used only for diffusion-weighted shots
    (``b_value > 0``). See ``_fse_common.py`` module docstring for why
    gradient surgery is scoped to ``b_value == 0``."""
    system.apply_system_derates(opts)

    rf90, gz90, gz_reph = excitation.slice_selective(opts, readout.EXCITATION_FLIP_DEG, cfg.slab_thickness_m)
    rf180, gz180 = readout.build_refocusing_pulse(opts, cfg.slab_thickness_m)

    echo = readout.compute_readout_and_echo_train(
        opts=opts,
        ro_axis="x",
        nx_ro=cfg.nx_ro,
        fov_ro_m=cfg.fov_ro_m,
        bandwidth_hz_px=cfg.bandwidth_hz_px,
        slice_thickness_m=cfg.slab_thickness_m,
        num_echoes=1,
        echo_spacing_s=0.0,
        # See fse_2d.py: FSE has no flyback/bipolar echo-train concept;
        # `flyback` is inert here (num_echoes=1) — polarity/rewind is built
        # manually below per the CPMG net-zero-per-cycle rule.
        flyback=False,
        strict=strict,
    )
    if echo is None:
        return None

    max_pe_area = 0.5 * cfg.ny_pe * (1.0 / cfg.fov_pe_m)
    gy_template = pp.make_trapezoid(channel="y", area=max_pe_area, system=opts)

    _, max_par_area = encoding.partition_geometry(cfg.npar, cfg.slice_spacing_m)
    gz_pe_template = pp.make_trapezoid(channel="z", area=max_par_area, system=opts) if max_par_area > 0.0 else None

    voxel_size_m = cfg.fov_ro_m / cfg.nx_ro
    crusher = readout.build_z_crusher(opts, cfg.slab_thickness_m)

    delta_s, separation_s = preparations.diffusion_timing(cfg.te_s)
    grad_t_per_m = preparations.required_gradient_t_per_m(cfg.b_value_s_mm2, delta_s, separation_s, opts.gamma)
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
    d_crusher_s = pp.calc_duration(crusher)
    d_pre_s = (
        pp.calc_duration(echo["gx_pre"], gy_template, gz_pe_template)
        if gz_pe_template is not None
        else pp.calc_duration(echo["gx_pre"], gy_template)
    )
    adc_center_s = echo["adc_center_s"]

    half_loop_fixed_first_s = d_gz_reph_s + diff_grad_duration_s
    half_loop_fixed_steady_s = diff_grad_duration_s + d_pre_s + adc_center_s

    delays_first = readout.fse_timing(
        cfg.te_s, d90_s, c90_s, d180_s, c180_s, d_crusher_s, half_loop_fixed_first_s, strict
    )
    delays_steady = readout.fse_timing(
        cfg.te_s, d90_s, c90_s, d180_s, c180_s, d_crusher_s, half_loop_fixed_steady_s, strict
    )
    if delays_first is None or delays_steady is None:
        return None
    tau_first_s, _ = delays_first
    _, tau_steady_s = delays_steady

    line_period_s = (
        d180_s + d_crusher_s + tau_steady_s + d_pre_s + pp.calc_duration(echo["gx_echo"], echo["adc"])
        + d_pre_s + tau_steady_s + d_crusher_s
    )
    n_lines_first_shot = min(cfg.etl, cfg.ny_pe)

    min_block_s = d90_s + d_gz_reph_s + d_crusher_s + tau_first_s + n_lines_first_shot * line_period_s
    tr_delay_s = cfg.tr_s - min_block_s
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
        "echo": echo,
        "gy_template": gy_template,
        "gz_pe_template": gz_pe_template,
        "crusher": crusher,
        "delta_s": delta_s,
        "grad_t_per_m": grad_t_per_m,
        "tau_first_s": tau_first_s,
        "tau_steady_s": tau_steady_s,
        "tr_delay_s": tr_delay_s,
        "min_block_s": min_block_s,
    }


def _build_legacy(seq, opts: pp.Opts, cfg: _Config) -> tuple[int, int]:
    """Diffusion-weighted path (``b_value > 0``): unfused design, unchanged
    from before gradient surgery. See ``_fse_common.py`` module docstring
    for scope rationale. Returns ``(n_shots, n_directions)``."""
    timing = _compute_timing_legacy(opts=opts, cfg=cfg, strict=False)

    gz90 = timing["gz90"]
    gz_reph = timing["gz_reph"]
    gz180 = timing["gz180"]
    echo = timing["echo"]
    gy_template = timing["gy_template"]
    gz_pe_template = timing["gz_pe_template"]
    tau_first_s = timing["tau_first_s"]
    tau_steady_s = timing["tau_steady_s"]
    tr_delay_s = timing["tr_delay_s"]

    tau_first_delay = pp.make_delay(tau_first_s) if tau_first_s > 0.0 else None
    tau_steady_delay = pp.make_delay(tau_steady_s) if tau_steady_s > 0.0 else None
    tr_delay = pp.make_delay(tr_delay_s) if tr_delay_s > 0.0 else None

    delta_k_pe = 1.0 / cfg.fov_pe_m
    phase_areas = (np.arange(cfg.ny_pe) - 0.5 * cfg.ny_pe) * delta_k_pe
    max_pe_area = float(np.max(np.abs(phase_areas)))
    sampled_pe = sampling.sampled_lines(cfg.ny_pe, cfg.ry, 0)
    shot_starts = list(range(0, len(sampled_pe), cfg.etl))

    refocus_flip_schedule = readout.build_refocus_flip_schedule(cfg.etl, cfg.refocus_flip_deg, cfg.refocus_variable)

    par_areas, max_par_area = encoding.partition_geometry(cfg.npar, cfg.slice_spacing_m)
    sampled_par = sampling.sampled_lines(cfg.npar, cfg.rz, 0)
    n_directions = cfg.n_directions if cfg.b_value_s_mm2 > 0.0 else 1
    directions = preparations.diffusion_directions(cfg.n_directions) if cfg.b_value_s_mm2 > 0.0 else [None]

    for direction in directions[:n_directions]:
        diff_grad = (
            preparations.build_diffusion_gradients(opts, direction, timing["delta_s"], timing["grad_t_per_m"])
            if direction is not None
            else []
        )

        for par in sampled_par:
            z_scale = par_areas[par] / max_par_area if max_par_area > 0.0 else 0.0
            gz_pe = pp.scale_grad(gz_pe_template, z_scale) if gz_pe_template is not None else None
            gz_pe_neg = pp.scale_grad(gz_pe_template, -z_scale) if gz_pe_template is not None else None

            for shot_start in shot_starts:
                segment = sampled_pe[shot_start : shot_start + cfg.etl]
                label_par = pp.make_label(type="SET", label="PAR", value=par)
                label_slc = pp.make_label(type="SET", label="SLC", value=0)

                rf90 = system.copy_event(timing["rf90"])
                seq.add_block(rf90, gz90, label_slc, label_par)
                seq.add_block(gz_reph)
                seq.add_block(timing["crusher"])
                if diff_grad:
                    seq.add_block(*diff_grad)
                if tau_first_delay is not None:
                    seq.add_block(tau_first_delay)

                for echo_idx, ky in enumerate(segment):
                    rf180 = readout.scale_refocusing_pulse(timing["rf180"], refocus_flip_schedule[echo_idx])
                    seq.add_block(rf180, gz180)
                    seq.add_block(timing["crusher"])
                    if echo_idx == 0 and diff_grad:
                        seq.add_block(*diff_grad)
                    if tau_steady_delay is not None:
                        seq.add_block(tau_steady_delay)

                    y_scale = phase_areas[ky] / max_pe_area if max_pe_area > 0.0 else 0.0
                    gy_pre = pp.scale_grad(gy_template, y_scale)
                    gy_reph = pp.scale_grad(gy_template, -y_scale)
                    label_lin = pp.make_label(type="SET", label="LIN", value=ky)

                    # See fse_2d.py for the net-zero-per-echo-cycle rationale
                    # (a real, regression-tested bug): the same symmetric
                    # (+area before / -area after) treatment that keeps kx
                    # centered also applies to kz here, since kz is FIXED
                    # for the whole shot and must survive every 180's sign
                    # flip unperturbed, exactly like ky.
                    if gz_pe is not None:
                        seq.add_block(echo["gx_pre"], gy_pre, gz_pe)
                    else:
                        seq.add_block(echo["gx_pre"], gy_pre)
                    seq.add_block(echo["gx_echo"], echo["adc"], label_lin)
                    if gz_pe_neg is not None:
                        seq.add_block(echo["gx_pre"], gy_reph, gz_pe_neg)
                    else:
                        seq.add_block(echo["gx_pre"], gy_reph)

                    if tau_steady_delay is not None:
                        seq.add_block(tau_steady_delay)
                    seq.add_block(timing["crusher"])

                if tr_delay is not None:
                    seq.add_block(tr_delay)

    return len(shot_starts), n_directions


def _compute_timing_surgery(opts: pp.Opts, cfg: _Config, strict: bool):
    """Gradient-surgery timing (default, ``b_value == 0``) — see
    ``_fse_common.py`` module docstring. Unlike ``fse_2d.py``, the
    per-echo z-chain (``GS5``/``GS7``) is built PER PARTITION (not shared
    across the whole plugin) since it also carries the kz phase-encode —
    see ``readout.build_kz_crusher_bridge``. This function still computes/
    returns the SHARED pieces (GS1-GS4, x/y chains) plus everything needed
    to build the per-partition GS5/GS7 pair in ``_build_surgery``, and
    validates that the WORST-CASE (largest |kz|) partition is feasible."""
    system.apply_system_derates(opts)

    rf90, gz90_auto, gz_reph = excitation.slice_selective(opts, readout.EXCITATION_FLIP_DEG, cfg.slab_thickness_m)
    rf180, gz180_auto = readout.build_refocusing_pulse(opts, cfg.slab_thickness_m)

    rf90.delay = opts.rf_dead_time
    rf180.delay = opts.rf_dead_time

    gz_ex_amp = gz90_auto.amplitude
    gz_ref_amp = gz180_auto.amplitude

    ro_events = readout.build_surgery_readout(opts, "x", cfg.nx_ro, cfg.fov_ro_m, cfg.bandwidth_hz_px)

    max_pe_area = 0.5 * cfg.ny_pe * (1.0 / cfg.fov_pe_m)
    _, max_par_area = encoding.partition_geometry(cfg.npar, cfg.slice_spacing_m)

    t_ex_wd_s = excitation.DEFAULT_SLICE_RF_DURATION_S + opts.rf_dead_time + opts.rf_ringdown_time
    t_ref_wd_s = readout.RF_REFOCUS_TIME_S + opts.rf_dead_time + opts.rf_ringdown_time

    c90_s = pp.calc_rf_center(rf90)[0]
    c180_s = pp.calc_rf_center(rf180)[0]

    dg_s = readout.surgery_ramp_time_s(opts, [gz_ex_amp])
    d_gz_reph_s = pp.calc_duration(gz_reph)

    delays = readout.surgery_timing(
        cfg.te_s, t_ex_wd_s, c90_s, dg_s, d_gz_reph_s, t_ref_wd_s, c180_s, ro_events["adc_center_s"], strict
    )
    if delays is None:
        return None
    t_spex_s, t_sp_s = delays

    crusher_area = encoding.CRUSHER_CYCLES_Z / cfg.slab_thickness_m

    try:
        slice_chain = readout.build_slice_surgery_chain(
            opts, gz_ex_amp, gz_ref_amp, crusher_area, t_ex_wd_s, t_ref_wd_s, t_spex_s, t_sp_s, dg_s
        )
        readout_chain = readout.build_readout_surgery_chain(
            opts, "x", ro_events["readout_area"], ro_events["gx_flat_amp"], ro_events["flat_time_s"], t_sp_s,
        )
        gy_template = pp.make_trapezoid(channel="y", area=max_pe_area, duration=t_sp_s, system=opts)
        # Worst-case (largest |kz|) partition must also be feasible at this
        # t_sp_s — checked eagerly here so infeasible protocols are caught
        # at validate_protocol time, not mid-`make_sequence`.
        if max_par_area > 0.0:
            readout.build_kz_crusher_bridge(opts, gz_ref_amp, crusher_area, max_par_area, t_sp_s)
    except (ValueError, AssertionError):
        if strict:
            return None
        raise

    line_period_s = t_ref_wd_s + t_sp_s + ro_events["flat_time_s"] + t_sp_s
    n_lines_first_shot = min(cfg.etl, cfg.ny_pe)

    min_block_s = (
        dg_s + t_ex_wd_s + dg_s + d_gz_reph_s + t_spex_s  # GS1, GS2, GS2_FALL, gz_reph, GS3
        + n_lines_first_shot * line_period_s
        + t_ref_wd_s + t_sp_s  # trailing GS4 (no RF) + GS5 close-out
    )
    tr_delay_s = cfg.tr_s - min_block_s
    if tr_delay_s < -1e-9 and strict:
        return None
    if tr_delay_s < 0.0:
        tr_delay_s = 0.0

    return {
        "rf90": rf90,
        "rf180": rf180,
        "gz_reph": gz_reph,
        "gz_ref_amp": gz_ref_amp,
        "readout": ro_events,
        "slice_chain": slice_chain,
        "readout_chain": readout_chain,
        "gy_template": gy_template,
        "crusher_area": crusher_area,
        "t_sp_s": t_sp_s,
        "max_par_area": max_par_area,
        "tr_delay_s": tr_delay_s,
        "min_block_s": min_block_s,
    }


def _build_surgery(seq, opts: pp.Opts, cfg: _Config) -> tuple[int, int]:
    """Default path (``b_value == 0``): gradient-surgery design — see
    ``_fse_common.py`` module docstring. Returns ``(n_shots,
    n_directions)`` (``n_directions`` is always 1 — diffusion uses
    ``_build_legacy``)."""
    timing = _compute_timing_surgery(opts=opts, cfg=cfg, strict=False)

    chain = timing["slice_chain"]
    rchain = timing["readout_chain"]
    ro_events = timing["readout"]
    gy_template = timing["gy_template"]
    tr_delay_s = timing["tr_delay_s"]
    tr_delay = pp.make_delay(tr_delay_s) if tr_delay_s > 0.0 else None

    delta_k_pe = 1.0 / cfg.fov_pe_m
    phase_areas = (np.arange(cfg.ny_pe) - 0.5 * cfg.ny_pe) * delta_k_pe
    max_pe_area = float(np.max(np.abs(phase_areas)))
    sampled_pe = sampling.sampled_lines(cfg.ny_pe, cfg.ry, 0)
    shot_starts = list(range(0, len(sampled_pe), cfg.etl))

    refocus_flip_schedule = readout.build_refocus_flip_schedule(cfg.etl, cfg.refocus_flip_deg, cfg.refocus_variable)

    par_areas, max_par_area = encoding.partition_geometry(cfg.npar, cfg.slice_spacing_m)
    sampled_par = sampling.sampled_lines(cfg.npar, cfg.rz, 0)

    for par in sampled_par:
        kz_area = par_areas[par] if max_par_area > 0.0 else 0.0
        gs5_par, gs7_par = readout.build_kz_crusher_bridge(
            opts, timing["gz_ref_amp"], timing["crusher_area"], kz_area, timing["t_sp_s"]
        )

        for shot_start in shot_starts:
            segment = sampled_pe[shot_start : shot_start + cfg.etl]
            label_par = pp.make_label(type="SET", label="PAR", value=par)
            label_slc = pp.make_label(type="SET", label="SLC", value=0)

            rf90 = system.copy_event(timing["rf90"])
            seq.add_block(chain["GS1"])
            seq.add_block(chain["GS2"], rf90, label_slc, label_par)
            seq.add_block(chain["GS2_FALL"])
            seq.add_block(timing["gz_reph"])
            seq.add_block(chain["GS3"])

            for echo_idx, ky in enumerate(segment):
                rf180 = readout.scale_refocusing_pulse(timing["rf180"], refocus_flip_schedule[echo_idx])
                seq.add_block(chain["GS4"], rf180)

                y_scale = phase_areas[ky] / max_pe_area if max_pe_area > 0.0 else 0.0
                gy_pre = pp.scale_grad(gy_template, y_scale)
                gy_reph = pp.scale_grad(gy_template, -y_scale)
                label_lin = pp.make_label(type="SET", label="LIN", value=ky)

                # gs5_par/gs7_par carry the SAME kz value every echo (kz is
                # fixed for the whole shot/partition) — same net-zero-
                # across-the-cycle rule as ky, folded into the crusher's
                # own area (see readout.build_kz_crusher_bridge).
                seq.add_block(gs5_par, rchain["GR5"], gy_pre)
                seq.add_block(rchain["GR6"], ro_events["adc"], label_lin)
                seq.add_block(gs7_par, rchain["GR7"], gy_reph)

            seq.add_block(chain["GS4"])  # close-out flat hold, no RF (writeTSE.m convention)
            seq.add_block(chain["GS5"])  # ramp back down to 0 (crusher-only — kz already net zero)

            if tr_delay is not None:
                seq.add_block(tr_delay)

    return len(shot_starts), 1


PLUGIN = Fse3DPulseqSequence()


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
    ('--flip-deg', UIParam.FLIP, float, 'Refocusing flip angle (deg); excitation is fixed at 90'),
    ('--refocus-variable', UIParam.user_value(USER_SLOT_REFOCUS_SCHEME), ("const", 1.0), ""),
    ('--fov-mm', UIParam.FOV, float, ""),
    ('--phase-fov-mm', UIParam.PHASE_FOV, float, ""),
    ('--slab-thickness-mm', UIParam.SLICE_THICKNESS, float, ""),
    ('--partition-spacing-mm', UIParam.SLICE_SPACING, float, ""),
    ('--nx', UIParam.NX, int, ""),
    ('--ny', UIParam.NY, int, ""),
    ('--npartitions', UIParam.NSLICES, int, ""),
    ('--etl', UIParam.ETL, int, ""),
    ('--ry', UIParam.RY, float, ""),
    ('--rz', UIParam.RZ, float, ""),
    ('--bandwidth-hz-px', UIParam.BANDWIDTH, float, ""),
    ('--bvalue', UIParam.DIFFUSION_BVALUES, float, ""),
    ('--directions', UIParam.DIFFUSION_DIRECTIONS, int, ""),
]

if __name__ == "__main__":
    raise SystemExit(
        run_cli(
            PLUGIN,
            sys.argv[1:],
            arg_map=_ARG_MAP,
            description='Generate a 3D fast/turbo spin-echo .seq offline.',
            default_output='fse_3d.seq',
        )
    )
