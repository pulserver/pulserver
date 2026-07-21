"""Standalone 2D fast/turbo spin-echo (FSE) sequence plugin for pulserver.

This module implements the three mandatory module-level entry points
required by the bridge dispatcher (see ``gre.py`` for the Cartesian GRE
family this parallels, ``epi_2d.py`` for the other spin-echo-family
sibling, and ``_gre_common.py`` / ``_fse_common.py`` / ``_diffusion_common.py``
for the shared building blocks):

- ``get_default_protocol(opts)``  — return the initial CV/protocol dictionary.
- ``validate_protocol(opts, protocol)``  — validate the current protocol and
  return a ``{"valid": bool, "duration": float | None, "info": str}`` dict.
- ``make_sequence(opts, protocol, output_path)``  — synthesise the sequence
  and write it to *output_path* using :func:`pulserver.io.write`.

Structure: 90 (slice-selective, fixed at ``_fse_common.EXCITATION_FLIP_DEG``,
not user-selectable) - crusher - [diffusion, first 180 only] - CPMG train of
``ETL`` refocusing pulses (constant 90 deg phase offset from the 90 — see
``_fse_common.py``), each flanked by a crusher and followed by its own
independently phase-encoded readout (a different ``ky`` per echo — NOT a
blip train). ``TE`` doubles as both "time to the first echo" and the echo
spacing (ESP): the symmetric crusher/prephase/rephase construction keeps
every inter-180 period identical by design (see ``_fse_common.fse_timing``),
so the two are the same value here — a simplification of the more general
clinical FSE case where phase-encode *reordering* (centric etc.) decouples
"effective TE" from ESP; this plugin always uses simple sequential ky order.

Flip-angle role (GE/TSEplus convention for spin-echo sequences): the
standard flip control (``UIParam.FLIP``) drives the REFOCUSING pulse angle
(default 180 deg, constant across the train unless the TRAPS scheme is
enabled), not the 90 excitation, which is fixed. See
``_fse_common.py``'s module docstring.

``ETL`` (native GE, "echoes per shot") splits ``NY`` into
``ceil(NY / ETL)`` sequential shots, each with its own 90-CPMG-train.
``DIFFUSION_BVALUES``/``DIFFUSION_DIRECTIONS`` (native) flank only the
FIRST 180 in the train (as a b0/DW preparation before the T2-weighted
readout proper) — ``b <= 0`` (default) disables it, same convention as
``epi_2d.py``.

Not implemented in this pass (noted, not silently dropped): optional
inversion before the 90 (IR-FSE / FLAIR-FSE) — same proven pattern as
``gre_mprage_radial_2d.py``/``epi_2d.py``, deferred to reach ZTE.

Usage
-----
Register the plugin in the interpreter tree by symlinking (or copying) this
file to the ``sequences/src/`` directory and creating a numbered alias::

    cp fse_2d.py pulserver-interpreter/package/pulserver/sequences/src/fse_2d.py
    ln -sf src/fse_2d.py pulserver-interpreter/package/pulserver/sequences/sequence15.py
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
from pulserver.pypulseq import _gradients as encoding
from pulserver.pypulseq import _readout as readout
from pulserver.pypulseq import _sampling as sampling
from pulserver.pypulseq import _system as system
from pulserver.pypulseq._rf import _excitation_helpers as excitation
from pulserver.pypulseq._rf import _preparation_helpers as preparations

# Refocusing scheme (TRAPS on/off) has no native GE CV — carried as an
# opuser custom variable, same convention as gre_multiecho_2d.py's
# USER_SLOT_*. The refocusing flip ANGLE itself is not a separate slot: it
# reuses the standard UIParam.FLIP control (see module docstring).
USER_SLOT_REFOCUS_SCHEME = 0


class Fse2DPulseqSequence(Sequence):
    """Generate a 2D turbo/fast spin-echo (CPMG) sequence."""

    def get_default_protocol(self, opts: pp.Opts) -> dict[str, dict]:
        del opts
        protocol = {
            UIParam.TE: DropdownFloatParam(
                value=12.0, min=5.0, max=50.0, incr=0.1, unit="ms",
                options=[8.0, 10.0, 12.0, 16.0, 20.0], validate=Validate.NONE,
            ),
            UIParam.TR: DropdownFloatParam(
                value=3000.0, min=200.0, max=10000.0, incr=1.0, unit="ms",
                options=[1500.0, 2000.0, 3000.0, 5000.0, 8000.0], validate=Validate.NONE,
            ),
            # Spin-echo role: FLIP drives the REFOCUSING pulse (the 90
            # excitation is fixed, not user-selectable — see module
            # docstring / _fse_common.EXCITATION_FLIP_DEG).
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
                value=5.0, min=1.0, max=20.0, incr=0.5, unit="mm",
                options=[1.0, 3.0, 5.0, 8.0, 10.0], validate=Validate.NONE,
            ),
            UIParam.SLICE_SPACING: DropdownFloatParam(
                value=5.0, min=1.0, max=20.0, incr=0.5, unit="mm",
                options=[1.0, 3.0, 5.0, 8.0, 10.0], validate=Validate.NONE,
            ),
            UIParam.NX: DropdownIntParam(
                value=64, min=16, max=512, incr=1, options=[64, 128, 192, 256, 384], validate=Validate.NONE,
            ),
            UIParam.NY: DropdownIntParam(
                value=64, min=8, max=512, incr=1, options=[64, 128, 192, 256, 384], validate=Validate.NONE,
            ),
            UIParam.NSLICES: DropdownIntParam(
                value=1, min=1, max=128, incr=1, options=[1, 5, 10, 20, 40], validate=Validate.NONE,
            ),
            UIParam.ETL: DropdownIntParam(
                value=16, min=1, max=256, incr=1, options=[4, 8, 16, 32, 64], validate=Validate.NONE,
            ),
            UIParam.RY: TypeinFloatParam(value=1.0, min=1.0, max=8.0, incr=1.0, unit="", validate=Validate.NONE),
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
            UIParam.SWAP_PHASE_FREQ: BoolParam(value=False, validate=Validate.NONE),
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
        if cfg.fov_ro_m <= 0.0 or cfg.fov_pe_m <= 0.0 or cfg.slice_thickness_m <= 0.0:
            return {"valid": False, "duration": None, "info": "FOV and slice thickness must be > 0"}
        if not (0.0 < cfg.refocus_flip_deg <= 180.0):
            return {"valid": False, "duration": None, "info": "Flip (refocusing) angle must be in (0, 180] deg"}
        if cfg.nx_ro < 1 or cfg.ny_pe < 1 or cfg.nslices < 1:
            return {"valid": False, "duration": None, "info": "NX, NY, and NSLICES must be >= 1"}
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
        duration_s = cfg.tr_s * float(n_shots)
        return {"valid": True, "duration": duration_s, "info": f"TA = {duration_s:.2f} s"}

    def make_sequence(self, opts: pp.Opts, protocol: dict[str, dict], output_path: str) -> None:
        prot = dict_to_protocol(protocol)
        cfg = _read_protocol(prot)

        seq = pp.Sequence(opts)
        if cfg.b_value_s_mm2 > 0.0:
            n_shots, n_directions = _build_legacy(seq, opts, cfg)
        else:
            n_shots, n_directions = _build_surgery(seq, opts, cfg)

        slice_step_m = cfg.slice_spacing_m if cfg.nslices > 1 else 0.0
        seq.set_definition("Name", "fse_2d")
        seq.set_definition(
            "FOV",
            [cfg.fov_ro_m, cfg.fov_pe_m, slice_step_m * cfg.nslices if cfg.nslices > 1 else cfg.slice_thickness_m],
        )
        seq.set_definition("TE", cfg.te_s)
        seq.set_definition("TR", cfg.tr_s)
        seq.set_definition("Flip", cfg.refocus_flip_deg)
        seq.set_definition("ExcitationFlip", readout.EXCITATION_FLIP_DEG)
        seq.set_definition("RefocusVariable", cfg.refocus_variable)
        seq.set_definition("ImagingMode", "2d")
        seq.set_definition("ReadoutAxis", cfg.ro_axis)
        seq.set_definition("PhaseAxis", cfg.pe_axis)
        seq.set_definition("ETL", cfg.etl)
        seq.set_definition("NumShots", n_shots)
        seq.set_definition("BValue", cfg.b_value_s_mm2)
        seq.set_definition("DiffusionDirections", n_directions)
        seq.set_definition("BandwidthHzPerPx", cfg.bandwidth_hz_px)
        seq.set_definition("Ry", cfg.ry)
        seq.set_definition("Nx", cfg.nx_ro)
        seq.set_definition("Ny", cfg.ny_pe)
        seq.set_definition("NumSlices", cfg.nslices)
        pio.write(seq, output=output_path, remove_duplicates=False, check_timing=False)


def _n_shots(cfg: _Config) -> int:
    sampled_pe = sampling.sampled_lines(cfg.ny_pe, cfg.ry, 0)
    return len(range(0, len(sampled_pe), cfg.etl))


class _Config:
    __slots__ = (
        "te_s", "tr_s", "refocus_flip_deg", "refocus_variable",
        "fov_ro_m", "fov_pe_m", "slice_thickness_m", "slice_spacing_m",
        "nx_ro", "ny_pe", "nslices", "etl", "ry", "bandwidth_hz_px", "b_value_s_mm2", "n_directions",
        "ro_axis", "pe_axis",
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
    cfg.slice_thickness_m = params.param_float(prot, UIParam.SLICE_THICKNESS) * 1e-3
    cfg.slice_spacing_m = params.param_float(prot, UIParam.SLICE_SPACING) * 1e-3
    cfg.nx_ro = params.param_int(prot, UIParam.NX)
    cfg.ny_pe = params.param_int(prot, UIParam.NY)
    cfg.nslices = params.param_int(prot, UIParam.NSLICES)
    cfg.etl = params.param_int_optional(prot, UIParam.ETL, cfg.ny_pe)
    cfg.ry = max(1, int(round(params.param_float_optional(prot, UIParam.RY, 1.0))))
    cfg.bandwidth_hz_px = params.param_float_optional(prot, UIParam.BANDWIDTH, system.DEFAULT_BANDWIDTH_HZ_PX)
    cfg.b_value_s_mm2 = params.param_float_optional(prot, UIParam.DIFFUSION_BVALUES, 0.0)
    cfg.n_directions = params.param_int_optional(prot, UIParam.DIFFUSION_DIRECTIONS, 3)
    cfg.ro_axis, cfg.pe_axis = params.resolve_readout_phase_axes(prot)
    return cfg


def _compute_timing_legacy(opts: pp.Opts, cfg: _Config, strict: bool):
    """Legacy (unfused) timing — used only for diffusion-weighted shots
    (``b_value > 0``). See ``_fse_common.py`` module docstring for why
    gradient surgery is scoped to ``b_value == 0``."""
    system.apply_system_derates(opts)

    rf90, gz90, gz_reph = excitation.slice_selective(opts, readout.EXCITATION_FLIP_DEG, cfg.slice_thickness_m)
    rf180, gz180 = readout.build_refocusing_pulse(opts, cfg.slice_thickness_m)

    echo = readout.compute_readout_and_echo_train(
        opts=opts,
        ro_axis=cfg.ro_axis,
        nx_ro=cfg.nx_ro,
        fov_ro_m=cfg.fov_ro_m,
        bandwidth_hz_px=cfg.bandwidth_hz_px,
        slice_thickness_m=cfg.slice_thickness_m,
        num_echoes=1,
        echo_spacing_s=0.0,
        # FSE has no flyback/bipolar echo-train concept (unlike GRE multi-
        # echo): with num_echoes=1, `flyback` is inert in this helper — the
        # per-echo readout polarity and prephase/rewind here are built
        # manually below, following the CPMG net-zero-per-cycle rule (see
        # make_sequence), not the GRE-style flyback rewind.
        flyback=False,
        strict=strict,
    )
    if echo is None:
        return None

    max_pe_area = 0.5 * cfg.ny_pe * (1.0 / cfg.fov_pe_m)
    gy_template = pp.make_trapezoid(channel=cfg.pe_axis, area=max_pe_area, system=opts)

    voxel_size_m = cfg.fov_ro_m / cfg.nx_ro
    crusher = readout.build_z_crusher(opts, cfg.slice_thickness_m)

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
    d_pre_s = pp.calc_duration(echo["gx_pre"], gy_template)
    adc_center_s = echo["adc_center_s"]

    half_loop_fixed_first_s = d_gz_reph_s + diff_grad_duration_s
    half_loop_fixed_steady_s = diff_grad_duration_s + d_pre_s + adc_center_s

    # First half-period (90 -> first 180) and steady half-period (every
    # subsequent 180-to-readout / readout-to-180 gap) can have different
    # fixed overhead (gz_reph only applies once, right after the 90), so
    # tau_first uses the 90-side overhead and tau_steady uses the
    # readout-side overhead — both target half_esp_s = TE/2.
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

    # The post-readout rewind reuses gx_pre (same -0.5*area trapezoid, same
    # duration) — see the CPMG net-zero-per-cycle rationale in make_sequence.
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
        "crusher": crusher,
        "delta_s": delta_s,
        "grad_t_per_m": grad_t_per_m,
        "tau_first_s": tau_first_s,
        "tau_steady_s": tau_steady_s,
        "tr_delay_s": tr_delay_s,
        "min_block_s": min_block_s,
    }


def _build_legacy(seq, opts: pp.Opts, cfg: _Config) -> tuple[int, int]:
    """Diffusion-weighted path (``b_value > 0``): unfused design (separate
    crusher + padding delay), unchanged from before gradient surgery. See
    ``_fse_common.py`` module docstring for scope rationale. Returns
    ``(n_shots, n_directions)`` for the caller's ``set_definition`` calls."""
    timing = _compute_timing_legacy(opts=opts, cfg=cfg, strict=False)

    gz90 = timing["gz90"]
    gz_reph = timing["gz_reph"]
    gz180 = timing["gz180"]
    echo = timing["echo"]
    gy_template = timing["gy_template"]
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

    slice_step_m = cfg.slice_spacing_m if cfg.nslices > 1 else 0.0
    n_directions = cfg.n_directions if cfg.b_value_s_mm2 > 0.0 else 1
    directions = preparations.diffusion_directions(cfg.n_directions) if cfg.b_value_s_mm2 > 0.0 else [None]

    for direction in directions[:n_directions]:
        diff_grad = (
            preparations.build_diffusion_gradients(opts, direction, timing["delta_s"], timing["grad_t_per_m"])
            if direction is not None
            else []
        )

        for sl in range(cfg.nslices):
            slice_offset_m = (sl - 0.5 * (cfg.nslices - 1)) * slice_step_m

            for shot_start in shot_starts:
                segment = sampled_pe[shot_start : shot_start + cfg.etl]

                rf90 = system.copy_event(timing["rf90"])
                rf90.freq_offset = gz90.amplitude * slice_offset_m
                seq.add_block(rf90, gz90)
                seq.add_block(gz_reph)
                seq.add_block(timing["crusher"])
                if diff_grad:
                    seq.add_block(*diff_grad)
                if tau_first_delay is not None:
                    seq.add_block(tau_first_delay)

                for echo_idx, ky in enumerate(segment):
                    rf180 = readout.scale_refocusing_pulse(timing["rf180"], refocus_flip_schedule[echo_idx])
                    rf180.freq_offset = gz180.amplitude * slice_offset_m
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

                    # Every 180 flips the sign of the accumulated k_x (pypulseq's
                    # calculate_kspace models refocusing as an exact sign flip), so
                    # the post-readout gradient must bring the NET x-moment for the
                    # whole prephase+readout+rewind cycle back to exactly ZERO (not
                    # to the pre-prephase state, i.e. NOT the "-full area" flyback
                    # rewind _gre_common builds for ordinary GRE echo trains, which
                    # assumes no intervening refocusing). -half + full - half = 0,
                    # so the rewind reuses the SAME -0.5*area trapezoid as the
                    # prephaser — applied every echo, unconditionally. Getting this
                    # wrong was caught as a real, verified-via-calculate_kspace
                    # asymmetric-kx bug in an earlier draft (see fse_2d.py history).
                    seq.add_block(echo["gx_pre"], gy_pre)
                    seq.add_block(echo["gx_echo"], echo["adc"], label_lin)
                    seq.add_block(echo["gx_pre"], gy_reph)

                    if tau_steady_delay is not None:
                        seq.add_block(tau_steady_delay)
                    seq.add_block(timing["crusher"])

                if tr_delay is not None:
                    seq.add_block(tr_delay)

    return len(shot_starts), n_directions


def _compute_timing_surgery(opts: pp.Opts, cfg: _Config, strict: bool):
    """Gradient-surgery timing (default, ``b_value == 0``) — see
    ``_fse_common.py`` module docstring."""
    system.apply_system_derates(opts)

    rf90, gz90_auto, gz_reph = excitation.slice_selective(opts, readout.EXCITATION_FLIP_DEG, cfg.slice_thickness_m)
    rf180, gz180_auto = readout.build_refocusing_pulse(opts, cfg.slice_thickness_m)

    # GS2/GS4 are flat-only (the ramp is supplied externally by GS1/GS3/
    # GS5/GS7), so the RF's own delay must be just rf_dead_time — not the
    # gz-rise-time-inflated delay `make_sinc_pulse(..., return_gz=True)`
    # sets by default for a self-contained trapezoid.
    rf90.delay = opts.rf_dead_time
    rf180.delay = opts.rf_dead_time

    gz_ex_amp = gz90_auto.amplitude
    gz_ref_amp = gz180_auto.amplitude

    ro_events = readout.build_surgery_readout(opts, cfg.ro_axis, cfg.nx_ro, cfg.fov_ro_m, cfg.bandwidth_hz_px)

    max_pe_area = 0.5 * cfg.ny_pe * (1.0 / cfg.fov_pe_m)

    t_ex_wd_s = excitation.DEFAULT_SLICE_RF_DURATION_S + opts.rf_dead_time + opts.rf_ringdown_time
    t_ref_wd_s = readout.RF_REFOCUS_TIME_S + opts.rf_dead_time + opts.rf_ringdown_time

    c90_s = pp.calc_rf_center(rf90)[0]
    c180_s = pp.calc_rf_center(rf180)[0]

    # dg_s is only for GS1/GS2_FALL (plain, unconstrained 0<->gz_ex_amp
    # ramps); GS3/GS5/GS7/GR5/GR7 each independently resolve their own leg
    # ramp times via `_bridge` — see `surgery_ramp_time_s`'s docstring for
    # why a single shared dG is unsafe for those.
    dg_s = readout.surgery_ramp_time_s(opts, [gz_ex_amp])
    d_gz_reph_s = pp.calc_duration(gz_reph)

    delays = readout.surgery_timing(
        cfg.te_s, t_ex_wd_s, c90_s, dg_s, d_gz_reph_s, t_ref_wd_s, c180_s, ro_events["adc_center_s"], strict
    )
    if delays is None:
        return None
    t_spex_s, t_sp_s = delays

    crusher_area = encoding.CRUSHER_CYCLES_Z / cfg.slice_thickness_m

    try:
        slice_chain = readout.build_slice_surgery_chain(
            opts, gz_ex_amp, gz_ref_amp, crusher_area, t_ex_wd_s, t_ref_wd_s, t_spex_s, t_sp_s, dg_s
        )
        readout_chain = readout.build_readout_surgery_chain(
            opts, cfg.ro_axis, ro_events["readout_area"], ro_events["gx_flat_amp"], ro_events["flat_time_s"], t_sp_s,
        )
        # No explicit rise_time: pypulseq's own shortest-feasible-ramp solve
        # for a plain 0->amp->0 trapezoid is self-consistent (no cross-
        # channel amplitude coupling the way GR5/GR7's bridges have).
        gy_template = pp.make_trapezoid(channel=cfg.pe_axis, area=max_pe_area, duration=t_sp_s, system=opts)
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
        "gz_ex_amp": gz_ex_amp,
        "gz_ref_amp": gz_ref_amp,
        "readout": ro_events,
        "slice_chain": slice_chain,
        "readout_chain": readout_chain,
        "gy_template": gy_template,
        "tr_delay_s": tr_delay_s,
        "min_block_s": min_block_s,
    }


def _build_surgery(seq, opts: pp.Opts, cfg: _Config) -> tuple[int, int]:
    """Default path (``b_value == 0``): gradient-surgery design — see
    ``_fse_common.py`` module docstring. Returns ``(n_shots,
    n_directions)`` for the caller's ``set_definition`` calls
    (``n_directions`` is always 1 here — diffusion uses ``_build_legacy``)."""
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

    slice_step_m = cfg.slice_spacing_m if cfg.nslices > 1 else 0.0

    for sl in range(cfg.nslices):
        slice_offset_m = (sl - 0.5 * (cfg.nslices - 1)) * slice_step_m

        for shot_start in shot_starts:
            segment = sampled_pe[shot_start : shot_start + cfg.etl]

            rf90 = system.copy_event(timing["rf90"])
            rf90.freq_offset = timing["gz_ex_amp"] * slice_offset_m
            seq.add_block(chain["GS1"])
            seq.add_block(chain["GS2"], rf90)
            seq.add_block(chain["GS2_FALL"])
            seq.add_block(timing["gz_reph"])
            seq.add_block(chain["GS3"])

            for echo_idx, ky in enumerate(segment):
                rf180 = readout.scale_refocusing_pulse(timing["rf180"], refocus_flip_schedule[echo_idx])
                rf180.freq_offset = timing["gz_ref_amp"] * slice_offset_m
                seq.add_block(chain["GS4"], rf180)

                y_scale = phase_areas[ky] / max_pe_area if max_pe_area > 0.0 else 0.0
                gy_pre = pp.scale_grad(gy_template, y_scale)
                gy_reph = pp.scale_grad(gy_template, -y_scale)
                label_lin = pp.make_label(type="SET", label="LIN", value=ky)

                seq.add_block(chain["GS5"], rchain["GR5"], gy_pre)
                seq.add_block(rchain["GR6"], ro_events["adc"], label_lin)
                seq.add_block(chain["GS7"], rchain["GR7"], gy_reph)

            seq.add_block(chain["GS4"])  # close-out flat hold, no RF (writeTSE.m convention)
            seq.add_block(chain["GS5"])  # ramp back down to 0

            if tr_delay is not None:
                seq.add_block(tr_delay)

    return len(shot_starts), 1


PLUGIN = Fse2DPulseqSequence()


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
    ('--slice-thickness-mm', UIParam.SLICE_THICKNESS, float, ""),
    ('--slice-spacing-mm', UIParam.SLICE_SPACING, float, ""),
    ('--nx', UIParam.NX, int, ""),
    ('--ny', UIParam.NY, int, ""),
    ('--nslices', UIParam.NSLICES, int, ""),
    ('--etl', UIParam.ETL, int, ""),
    ('--ry', UIParam.RY, float, ""),
    ('--bandwidth-hz-px', UIParam.BANDWIDTH, float, ""),
    ('--bvalue', UIParam.DIFFUSION_BVALUES, float, ""),
    ('--directions', UIParam.DIFFUSION_DIRECTIONS, int, ""),
    ('--swap-phase-freq', UIParam.SWAP_PHASE_FREQ, ("const", True), ""),
]

if __name__ == "__main__":
    raise SystemExit(
        run_cli(
            PLUGIN,
            sys.argv[1:],
            arg_map=_ARG_MAP,
            description='Generate a 2D fast/turbo spin-echo .seq offline.',
            default_output='fse_2d.seq',
        )
    )
