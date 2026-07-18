"""Standalone 2D single-/multi-shot spin-echo EPI sequence plugin.

This module implements the three mandatory module-level entry points
required by the bridge dispatcher (see ``gre.py`` for the Cartesian GRE
family this parallels, and ``_gre_common.py`` / ``_epi_common.py`` /
``pulserver.design`` / ``_diffusion_common.py`` for the shared building
blocks):

- ``get_default_protocol(opts)``  — return the initial CV/protocol dictionary.
- ``validate_protocol(opts, protocol)``  — validate the current protocol and
  return a ``{"valid": bool, "duration": float | None, "info": str}`` dict.
- ``make_sequence(opts, protocol, output_path)``  — synthesise the sequence
  and write it to *output_path* using :func:`pulserver.io.write`.

Structure: 90 (slice-selective) - crusher - [diffusion gradient] - 180
(slice-selective, same slice) - crusher - [diffusion gradient] -
prephasers - blipped EPI readout train. The 180 is kept at exactly
``TE / 2`` (a hard requirement for correct refocusing, not a convention —
see ``_epi_common.pre_post_180_delays``). ``TE`` is defined as the time
from the 90's center to the FIRST readout line's ADC center — with a
linear (non-centric) blip order starting at one k-space edge, this first
line genuinely is the spin-echo peak, and later lines drift toward
T2*-weighting, which is standard single-shot EPI physics.

``ETL`` (native GE, "lines per shot") splits ``NY`` into
``ceil(NY / ETL)`` shots, each with its own full 90-180-train (sequential,
not interleaved, k-space partitioning). Ramp sampling is an ``opuser``
toggle (no native GE counterpart) — see ``_epi_common.py`` for exactly what
it does and does not correct for. ``DIFFUSION_BVALUES``/
``DIFFUSION_DIRECTIONS`` are native GE variables reused directly;
``b <= 0`` (the default) means no diffusion weighting, so no separate
enable flag is needed.

Not implemented in this pass (noted, not silently dropped): optional
inversion/T2-prep before the 90 — same proven pattern as
``gre_mprage_radial_2d.py``, a mechanical follow-up, deferred to keep pace
with the rest of the non-Cartesian-adjacent zoo backlog (EPI/FSE/ZTE).

Usage
-----
Register the plugin in the interpreter tree by symlinking (or copying) this
file to the ``sequences/src/`` directory and creating a numbered alias::

    cp epi_2d.py pulserver-interpreter/package/pulserver/sequences/src/epi_2d.py
    ln -sf src/epi_2d.py pulserver-interpreter/package/pulserver/sequences/sequence13.py
"""

from __future__ import annotations

import sys

import numpy as np
import pypulseq as pp

import pulserver.io as pio
import pulserver.pulseq as ps

from pulserver import (
    PulseqSequence,
    BoolParam,
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
from pulserver.design import cli, encoding, excitation, params, preparations, readout, sampling, system



RF_REFOCUS_TIME_S = 3.0e-3
RF_REFOCUS_APODIZATION = 0.5
RF_REFOCUS_TIME_BW_PRODUCT = 4.0

USER_SLOT_RAMP_SAMPLE = 0


class Epi2DPulseqSequence(PulseqSequence):
    """Generate a 2D single-/multi-shot spin-echo EPI sequence."""

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
                value=64, min=1, max=512, incr=1, options=[16, 32, 64, 128, 256], validate=Validate.NONE,
            ),
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
            UIParam.SWAP_PHASE_FREQ: BoolParam(value=False, validate=Validate.NONE),
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
        if cfg.fov_ro_m <= 0.0 or cfg.fov_pe_m <= 0.0 or cfg.slice_thickness_m <= 0.0:
            return {"valid": False, "duration": None, "info": "FOV and slice thickness must be > 0"}
        if not (0.0 < cfg.flip_deg <= 90.0):
            return {"valid": False, "duration": None, "info": "Flip angle must be in (0, 90] deg"}
        if cfg.nx_ro < 1 or cfg.ny_pe < 1 or cfg.nslices < 1:
            return {"valid": False, "duration": None, "info": "NX, NY, and NSLICES must be >= 1"}
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
        n_dirs = cfg.n_directions if cfg.b_value_s_mm2 > 0.0 else 1
        duration_s = cfg.tr_s * float(n_shots) * float(cfg.nslices) * float(n_dirs)
        return {"valid": True, "duration": duration_s, "info": f"TA = {duration_s:.2f} s"}

    def make_sequence(self, opts: pp.Opts, protocol: dict[str, dict], output_path: str) -> None:
        prot = dict_to_protocol(protocol)
        cfg = _read_protocol(prot)

        timing = _compute_timing(opts=opts, cfg=cfg, strict=False)

        gz90 = timing["gz90"]
        gz_reph = timing["gz_reph"]
        gx_pre = timing["gx_pre"]
        gy_pre_template = timing["gy_pre_template"]
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

        slice_step_m = cfg.slice_spacing_m if cfg.nslices > 1 else 0.0
        shot_starts = list(range(0, cfg.ny_pe, cfg.etl))
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

                for shot_idx, ky_start in enumerate(shot_starts):
                    n_lines = min(cfg.etl, cfg.ny_pe - ky_start)

                    rf90 = system.copy_event(timing["rf90"])
                    rf90.freq_offset = gz90.amplitude * slice_offset_m
                    rf180 = system.copy_event(timing["rf180"])
                    rf180.freq_offset = timing["gz180"].amplitude * slice_offset_m

                    seq.add_block(rf90, gz90)
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
                    seq.add_block(gx_pre, gy_pre)

                    readout.add_epi_train_blocks(
                        seq, gx_pos, gx_neg, gy_blip, adc, n_lines,
                        start_polarity_positive=True, emit_lin_labels=True, lin_start=ky_start,
                    )

                    if tr_delay is not None:
                        seq.add_block(tr_delay)

        seq.set_definition("Name", "epi_2d")
        seq.set_definition(
            "FOV",
            [cfg.fov_ro_m, cfg.fov_pe_m, slice_step_m * cfg.nslices if cfg.nslices > 1 else cfg.slice_thickness_m],
        )
        seq.set_definition("TE", cfg.te_s)
        seq.set_definition("TR", cfg.tr_s)
        seq.set_definition("Flip", cfg.flip_deg)
        seq.set_definition("ImagingMode", "2d")
        seq.set_definition("ETL", cfg.etl)
        seq.set_definition("NumShots", len(shot_starts))
        seq.set_definition("RampSample", cfg.ramp_sample)
        seq.set_definition("BValue", cfg.b_value_s_mm2)
        seq.set_definition("DiffusionDirections", n_directions)
        seq.set_definition("BandwidthHzPerPx", cfg.bandwidth_hz_px)
        seq.set_definition("Nx", cfg.nx_ro)
        seq.set_definition("Ny", cfg.ny_pe)
        seq.set_definition("NumSlices", cfg.nslices)
        pio.write(seq, output=output_path, remove_duplicates=False, check_timing=False)


def _n_shots(cfg: "_Config") -> int:
    return len(range(0, cfg.ny_pe, cfg.etl))


class _Config:
    __slots__ = (
        "te_s", "tr_s", "flip_deg", "fov_ro_m", "fov_pe_m", "slice_thickness_m", "slice_spacing_m",
        "nx_ro", "ny_pe", "nslices", "etl", "bandwidth_hz_px", "ramp_sample",
        "b_value_s_mm2", "n_directions",
    )


def _read_protocol(prot: dict) -> _Config:
    cfg = _Config()
    cfg.te_s = params.param_float(prot, UIParam.TE) * 1e-3
    cfg.tr_s = params.param_float(prot, UIParam.TR) * 1e-3
    cfg.flip_deg = params.param_float(prot, UIParam.FLIP)
    cfg.fov_ro_m = params.param_float(prot, UIParam.FOV) * 1e-3
    cfg.fov_pe_m = params.phase_fov_mm_from_protocol(prot) * 1e-3
    cfg.slice_thickness_m = params.param_float(prot, UIParam.SLICE_THICKNESS) * 1e-3
    cfg.slice_spacing_m = params.param_float(prot, UIParam.SLICE_SPACING) * 1e-3
    cfg.nx_ro = params.param_int(prot, UIParam.NX)
    cfg.ny_pe = params.param_int(prot, UIParam.NY)
    cfg.nslices = params.param_int(prot, UIParam.NSLICES)
    cfg.etl = params.param_int_optional(prot, UIParam.ETL, cfg.ny_pe)
    cfg.bandwidth_hz_px = params.param_float_optional(prot, UIParam.BANDWIDTH, 250_000.0)
    cfg.ramp_sample = params.user_float(prot, USER_SLOT_RAMP_SAMPLE, 0.0) >= 0.5
    cfg.b_value_s_mm2 = params.param_float_optional(prot, UIParam.DIFFUSION_BVALUES, 0.0)
    cfg.n_directions = params.param_int_optional(prot, UIParam.DIFFUSION_DIRECTIONS, 3)
    return cfg


def _compute_timing(opts: pp.Opts, cfg: _Config, strict: bool):
    system.apply_system_derates(opts)

    rf90, gz90, gz_reph = excitation.slice_selective(opts, cfg.flip_deg, cfg.slice_thickness_m)
    rf180, gz180, _ = pp.make_sinc_pulse(
        flip_angle=np.pi,
        duration=RF_REFOCUS_TIME_S,
        slice_thickness=cfg.slice_thickness_m,
        apodization=RF_REFOCUS_APODIZATION,
        time_bw_product=RF_REFOCUS_TIME_BW_PRODUCT,
        system=opts,
        use="refocusing",
        return_gz=True,
    )

    gx_pos, gx_neg, adc = readout.build_epi_readout(
        opts, "x", cfg.nx_ro, cfg.fov_ro_m, cfg.bandwidth_hz_px, cfg.ramp_sample
    )
    gy_blip = readout.build_pe_blip(opts, "y", cfg.fov_pe_m)

    gx_pre = pp.make_trapezoid(channel="x", area=-0.5 * gx_pos.area, system=opts)
    max_pe_area = 0.5 * cfg.ny_pe * (1.0 / cfg.fov_pe_m)
    gy_pre_template = pp.make_trapezoid(channel="y", area=max_pe_area, system=opts)

    voxel_size_m = cfg.fov_ro_m / cfg.nx_ro
    crusher_before = encoding.spoiler_3axis(opts, voxel_size_m)
    crusher_after = encoding.spoiler_3axis(opts, voxel_size_m)

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
    d_crusher_before_s = pp.calc_duration(*crusher_before)
    d_crusher_after_s = pp.calc_duration(*crusher_after)
    d_prephasers_s = pp.calc_duration(gx_pre, gy_pre_template)

    if cfg.ramp_sample:
        adc_center_s = adc.delay + 0.5 * pp.calc_duration(gx_pos)
    else:
        adc_center_s = adc.delay + 0.5 * gx_pos.flat_time

    pre180_fixed_s = d_gz_reph_s + d_crusher_before_s + diff_grad_duration_s
    post180_fixed_s = d_crusher_after_s + diff_grad_duration_s + d_prephasers_s + adc_center_s

    delays = readout.pre_post_180_delays(
        cfg.te_s / 2.0, d90_s, c90_s, d180_s, c180_s, pre180_fixed_s, post180_fixed_s, strict
    )
    if delays is None:
        return None
    tau1_s, tau2_s = delays

    n_lines_first_shot = min(cfg.etl, cfg.ny_pe)
    line_period_s = pp.calc_duration(gx_pos)
    train_span_s = n_lines_first_shot * line_period_s + max(0, n_lines_first_shot - 1) * pp.calc_duration(gy_blip)

    min_block_s = (
        d90_s + d_gz_reph_s + d_crusher_before_s + diff_grad_duration_s + tau1_s
        + d180_s + d_crusher_after_s + diff_grad_duration_s + tau2_s
        + d_prephasers_s + train_span_s
    )
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
        "gx_pre": gx_pre,
        "gy_pre_template": gy_pre_template,
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


PLUGIN = Epi2DPulseqSequence()


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
    ('--flip-deg', UIParam.FLIP, float, ""),
    ('--fov-mm', UIParam.FOV, float, ""),
    ('--phase-fov-mm', UIParam.PHASE_FOV, float, ""),
    ('--slice-thickness-mm', UIParam.SLICE_THICKNESS, float, ""),
    ('--slice-spacing-mm', UIParam.SLICE_SPACING, float, ""),
    ('--nx', UIParam.NX, int, ""),
    ('--ny', UIParam.NY, int, ""),
    ('--nslices', UIParam.NSLICES, int, ""),
    ('--etl', UIParam.ETL, int, ""),
    ('--bandwidth-hz-px', UIParam.BANDWIDTH, float, ""),
    ('--ramp-sample', UIParam.user_value(USER_SLOT_RAMP_SAMPLE), ("const", 1.0), ""),
    ('--bvalue', UIParam.DIFFUSION_BVALUES, float, ""),
    ('--directions', UIParam.DIFFUSION_DIRECTIONS, int, ""),
    ('--swap-phase-freq', UIParam.SWAP_PHASE_FREQ, ("const", True), ""),
]

if __name__ == "__main__":
    raise SystemExit(
        cli.run_cli(
            PLUGIN,
            sys.argv[1:],
            arg_map=_ARG_MAP,
            description='Generate a 2D spin-echo EPI .seq offline.',
            default_output='epi_2d.seq',
        )
    )
