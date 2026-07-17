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

Structure: 90 (slice-selective) - crusher - [diffusion, first 180 only] -
CPMG train of ``ETL`` 180 refocusing pulses (constant 90 deg phase offset
from the 90 — see ``_fse_common.py``), each flanked by a crusher and
followed by its own independently phase-encoded readout (a different ``ky``
per echo — NOT a blip train). ``TE`` doubles as both "time to the first
echo" and the echo spacing (ESP): the symmetric crusher/prephase/rephase
construction keeps every inter-180 period identical by design (see
``_fse_common.fse_timing``), so the two are the same value here — a
simplification of the more general clinical FSE case where phase-encode
*reordering* (centric etc.) decouples "effective TE" from ESP; this plugin
always uses simple sequential ky order.

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


def _load_sibling_module(name: str):
    """Load a same-directory helper module by file path (see gre_multiecho_2d.py)."""
    module_path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gc = _load_sibling_module("_gre_common")
fc = _load_sibling_module("_fse_common")
dc = _load_sibling_module("_diffusion_common")


class Fse2DPulseqSequence(PulseqSequence):
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
                "info": "TE (ESP) too short for the 90/180/crushers/diffusion/readout, or TR too short",
            }

        n_shots = _n_shots(cfg)
        duration_s = cfg.tr_s * float(n_shots)
        return {"valid": True, "duration": duration_s, "info": f"TA = {duration_s:.2f} s"}

    def make_sequence(self, opts: pp.Opts, protocol: dict[str, dict], output_path: str) -> None:
        prot = dict_to_protocol(protocol)
        cfg = _read_protocol(prot)

        timing = _compute_timing(opts=opts, cfg=cfg, strict=False)

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

        seq = ps.Sequence(opts)

        delta_k_pe = 1.0 / cfg.fov_pe_m
        phase_areas = (np.arange(cfg.ny_pe) - 0.5 * cfg.ny_pe) * delta_k_pe
        max_pe_area = float(np.max(np.abs(phase_areas)))
        sampled_pe = gc.sampled_lines(cfg.ny_pe, cfg.ry, 0)
        shot_starts = list(range(0, len(sampled_pe), cfg.etl))

        slice_step_m = cfg.slice_spacing_m if cfg.nslices > 1 else 0.0
        n_directions = cfg.n_directions if cfg.b_value_s_mm2 > 0.0 else 1
        directions = dc.diffusion_directions(cfg.n_directions) if cfg.b_value_s_mm2 > 0.0 else [None]

        for direction in directions[:n_directions]:
            diff_grad = (
                dc.build_diffusion_gradients(opts, direction, timing["delta_s"], timing["grad_t_per_m"])
                if direction is not None
                else []
            )

            for sl in range(cfg.nslices):
                slice_offset_m = (sl - 0.5 * (cfg.nslices - 1)) * slice_step_m

                for shot_start in shot_starts:
                    segment = sampled_pe[shot_start : shot_start + cfg.etl]

                    rf90 = gc.copy_event(timing["rf90"])
                    rf90.freq_offset = gz90.amplitude * slice_offset_m
                    seq.add_block(rf90, gz90)
                    seq.add_block(gz_reph)
                    seq.add_block(*timing["crusher"])
                    if diff_grad:
                        seq.add_block(*diff_grad)
                    if tau_first_delay is not None:
                        seq.add_block(tau_first_delay)

                    for echo_idx, ky in enumerate(segment):
                        rf180 = gc.copy_event(timing["rf180"])
                        rf180.freq_offset = gz180.amplitude * slice_offset_m
                        seq.add_block(rf180, gz180)
                        seq.add_block(*timing["crusher"])
                        if echo_idx == 0 and diff_grad:
                            seq.add_block(*diff_grad)
                        if tau_steady_delay is not None:
                            seq.add_block(tau_steady_delay)

                        y_scale = phase_areas[ky] / max_pe_area if max_pe_area > 0.0 else 0.0
                        gy_pre = pp.scale_grad(gy_template, y_scale)
                        gy_reph = pp.scale_grad(gy_template, -y_scale)
                        label_lin = pp.make_label(type="SET", label="LIN", value=ky)

                        seq.add_block(echo["gx_pre"], gy_pre)
                        seq.add_block(echo["gx_echo"], echo["adc"], label_lin)
                        seq.add_block(echo["gx_rewind"], gy_reph)

                        if tau_steady_delay is not None:
                            seq.add_block(tau_steady_delay)
                        seq.add_block(*timing["crusher"])

                    if tr_delay is not None:
                        seq.add_block(tr_delay)

        seq.set_definition("Name", "fse_2d")
        seq.set_definition(
            "FOV",
            [cfg.fov_ro_m, cfg.fov_pe_m, slice_step_m * cfg.nslices if cfg.nslices > 1 else cfg.slice_thickness_m],
        )
        seq.set_definition("TE", cfg.te_s)
        seq.set_definition("TR", cfg.tr_s)
        seq.set_definition("Flip", cfg.flip_deg)
        seq.set_definition("ImagingMode", "2d")
        seq.set_definition("ReadoutAxis", cfg.ro_axis)
        seq.set_definition("PhaseAxis", cfg.pe_axis)
        seq.set_definition("ETL", cfg.etl)
        seq.set_definition("NumShots", len(shot_starts))
        seq.set_definition("BValue", cfg.b_value_s_mm2)
        seq.set_definition("DiffusionDirections", n_directions)
        seq.set_definition("BandwidthHzPerPx", cfg.bandwidth_hz_px)
        seq.set_definition("Ry", cfg.ry)
        seq.set_definition("Nx", cfg.nx_ro)
        seq.set_definition("Ny", cfg.ny_pe)
        seq.set_definition("NumSlices", cfg.nslices)
        pio.write(seq, output=output_path, remove_duplicates=False, check_timing=False)


def _n_shots(cfg: "_Config") -> int:
    sampled_pe = gc.sampled_lines(cfg.ny_pe, cfg.ry, 0)
    return len(range(0, len(sampled_pe), cfg.etl))


class _Config:
    __slots__ = (
        "te_s", "tr_s", "flip_deg", "fov_ro_m", "fov_pe_m", "slice_thickness_m", "slice_spacing_m",
        "nx_ro", "ny_pe", "nslices", "etl", "ry", "bandwidth_hz_px", "b_value_s_mm2", "n_directions",
        "ro_axis", "pe_axis",
    )


def _read_protocol(prot: dict) -> _Config:
    cfg = _Config()
    cfg.te_s = gc.param_float(prot, UIParam.TE) * 1e-3
    cfg.tr_s = gc.param_float(prot, UIParam.TR) * 1e-3
    cfg.flip_deg = gc.param_float(prot, UIParam.FLIP)
    cfg.fov_ro_m = gc.param_float(prot, UIParam.FOV) * 1e-3
    cfg.fov_pe_m = gc.phase_fov_mm_from_protocol(prot) * 1e-3
    cfg.slice_thickness_m = gc.param_float(prot, UIParam.SLICE_THICKNESS) * 1e-3
    cfg.slice_spacing_m = gc.param_float(prot, UIParam.SLICE_SPACING) * 1e-3
    cfg.nx_ro = gc.param_int(prot, UIParam.NX)
    cfg.ny_pe = gc.param_int(prot, UIParam.NY)
    cfg.nslices = gc.param_int(prot, UIParam.NSLICES)
    cfg.etl = gc.param_int_optional(prot, UIParam.ETL, cfg.ny_pe)
    cfg.ry = max(1, int(round(gc.param_float_optional(prot, UIParam.RY, 1.0))))
    cfg.bandwidth_hz_px = gc.param_float_optional(prot, UIParam.BANDWIDTH, gc.DEFAULT_BANDWIDTH_HZ_PX)
    cfg.b_value_s_mm2 = gc.param_float_optional(prot, UIParam.DIFFUSION_BVALUES, 0.0)
    cfg.n_directions = gc.param_int_optional(prot, UIParam.DIFFUSION_DIRECTIONS, 3)
    cfg.ro_axis, cfg.pe_axis = gc.resolve_readout_phase_axes(prot)
    return cfg


def _compute_timing(opts: pp.Opts, cfg: _Config, strict: bool):
    gc.apply_system_derates(opts)

    rf90, gz90, gz_reph = gc.build_rf(opts, cfg.flip_deg, cfg.slice_thickness_m)
    rf180, gz180 = fc.build_refocusing_pulse(opts, cfg.slice_thickness_m)

    echo = gc.compute_readout_and_echo_train(
        opts=opts,
        ro_axis=cfg.ro_axis,
        nx_ro=cfg.nx_ro,
        fov_ro_m=cfg.fov_ro_m,
        bandwidth_hz_px=cfg.bandwidth_hz_px,
        slice_thickness_m=cfg.slice_thickness_m,
        num_echoes=1,
        echo_spacing_s=0.0,
        flyback=True,
        strict=strict,
    )
    if echo is None:
        return None

    max_pe_area = 0.5 * cfg.ny_pe * (1.0 / cfg.fov_pe_m)
    gy_template = pp.make_trapezoid(channel=cfg.pe_axis, area=max_pe_area, system=opts)

    voxel_size_m = cfg.fov_ro_m / cfg.nx_ro
    crusher = fc.pc.build_inversion_spoiler(opts, voxel_size_m)

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
    d_crusher_s = pp.calc_duration(*crusher)
    d_pre_s = pp.calc_duration(echo["gx_pre"], gy_template)
    adc_center_s = echo["adc_center_s"]

    half_loop_fixed_first_s = d_gz_reph_s + diff_grad_duration_s
    half_loop_fixed_steady_s = diff_grad_duration_s + d_pre_s + adc_center_s

    # First half-period (90 -> first 180) and steady half-period (every
    # subsequent 180-to-readout / readout-to-180 gap) can have different
    # fixed overhead (gz_reph only applies once, right after the 90), so
    # tau_first uses the 90-side overhead and tau_steady uses the
    # readout-side overhead — both target half_esp_s = TE/2.
    delays_first = fc.fse_timing(
        cfg.te_s, d90_s, c90_s, d180_s, c180_s, d_crusher_s, half_loop_fixed_first_s, strict
    )
    delays_steady = fc.fse_timing(
        cfg.te_s, d90_s, c90_s, d180_s, c180_s, d_crusher_s, half_loop_fixed_steady_s, strict
    )
    if delays_first is None or delays_steady is None:
        return None
    tau_first_s, _ = delays_first
    _, tau_steady_s = delays_steady

    line_period_s = (
        d180_s + d_crusher_s + tau_steady_s + d_pre_s + pp.calc_duration(echo["gx_echo"], echo["adc"])
        + pp.calc_duration(echo["gx_rewind"]) + tau_steady_s + d_crusher_s
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


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a 2D fast/turbo spin-echo .seq offline.")
    parser.add_argument("-o", "--output", default="fse_2d.seq", help="Output .seq file path")
    parser.add_argument("--te-ms", type=float)
    parser.add_argument("--tr-ms", type=float)
    parser.add_argument("--flip-deg", type=float)
    parser.add_argument("--fov-mm", type=float)
    parser.add_argument("--phase-fov-mm", type=float)
    parser.add_argument("--slice-thickness-mm", type=float)
    parser.add_argument("--slice-spacing-mm", type=float)
    parser.add_argument("--nx", type=int)
    parser.add_argument("--ny", type=int)
    parser.add_argument("--nslices", type=int)
    parser.add_argument("--etl", type=int)
    parser.add_argument("--ry", type=float)
    parser.add_argument("--bandwidth-hz-px", type=float)
    parser.add_argument("--bvalue", type=float)
    parser.add_argument("--directions", type=int)
    parser.add_argument("--swap-phase-freq", action="store_true")
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
    if args.ry is not None:
        gc.set_protocol_value(protocol, UIParam.RY, args.ry)
    if args.bandwidth_hz_px is not None:
        gc.set_protocol_value(protocol, UIParam.BANDWIDTH, args.bandwidth_hz_px)
    if args.bvalue is not None:
        gc.set_protocol_value(protocol, UIParam.DIFFUSION_BVALUES, args.bvalue)
    if args.directions is not None:
        gc.set_protocol_value(protocol, UIParam.DIFFUSION_DIRECTIONS, args.directions)
    if args.swap_phase_freq:
        gc.set_protocol_value(protocol, UIParam.SWAP_PHASE_FREQ, True)

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
