"""Standalone 2D non-Cartesian (spiral/rosette) GRE sequence plugin.

This module implements the three mandatory module-level entry points
required by the bridge dispatcher (see ``gre.py`` for the Cartesian 2D GRE
this parallels, ``gre_radial_2d.py`` for the plain-trapezoid non-Cartesian
sibling, and ``_noncart_common.py``/``_gre_common.py`` for the shared
building blocks):

- ``get_default_protocol(opts)``  — return the initial CV/protocol dictionary.
- ``validate_protocol(opts, protocol)``  — validate the current protocol and
  return a ``{"valid": bool, "duration": float | None, "info": str}`` dict.
- ``make_sequence(opts, protocol, output_path)``  — synthesise the sequence
  and write it to *output_path* using :func:`pulserver.io.write`.

A single slew/gradient-limited base waveform (spiral or rosette, chosen via
an ``opuser`` toggle) is designed once via ``pulserver.arbgrad`` and
replayed per shot with a per-shot in-plane rotation (pulseq ``ROTATIONS``
extension, ``pulserver.pulseq.make_rotation``) — the C++ solver never loops
over shots itself, per the project's arbgrad design rule. The trajectory
starts at k-space center (``k0`` ≈ 0) and returns to zero gradient at the
end, so no separate prephaser/rewinder is needed — only the slice-select
rephase (before) and a spoiler (after).

``NumShots`` (``IntKey.NUM_SHOTS``, a real native GE variable) is the number
of shots actually played (free choice, not enforced against arbgrad's
Nyquist estimate). The trajectory-shape and shot-order toggles have no
native GE counterpart, so both are carried as ``opuser`` custom variables.
There is no ``bandwidth`` control here: ADC dwell is pinned to the gradient
raster time (the arbgrad waveform is designed on that raster).

Usage
-----
Register the plugin in the interpreter tree by symlinking (or copying) this
file to the ``sequences/src/`` directory and creating a numbered alias::

    cp gre_noncart_2d.py pulserver-interpreter/package/pulserver/sequences/src/gre_noncart_2d.py
    ln -sf src/gre_noncart_2d.py pulserver-interpreter/package/pulserver/sequences/sequence9.py
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
    Description,
    DropdownFloatParam,
    DropdownIntParam,
    UIParam,
    Validate,
    dict_to_protocol,
    make_enum_param,
    protocol_to_dict,
)
from pulserver import arbgrad
from pulserver.core import SequenceType


def _load_sibling_module(name: str):
    """Load a same-directory helper module by file path (see gre_multiecho_2d.py)."""
    module_path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gc = _load_sibling_module("_gre_common")
nc = _load_sibling_module("_noncart_common")

USER_SLOT_TRAJECTORY = 0
USER_SLOT_ORDER_MODE = 1


class GreNoncart2DPulseqSequence(PulseqSequence):
    """Generate a 2D spiral/rosette non-Cartesian GRE."""

    def get_default_protocol(self, opts: pp.Opts) -> dict[str, dict]:
        del opts
        protocol = {
            UIParam.TE: DropdownFloatParam(
                value=3.0, min=0.5, max=80.0, incr=0.1, unit="ms",
                options=[1.0, 3.0, 5.0, 8.0, 15.0], validate=Validate.NONE,
            ),
            UIParam.TR: DropdownFloatParam(
                value=20.0, min=5.0, max=2000.0, incr=0.1, unit="ms",
                options=[10.0, 20.0, 50.0, 100.0, 200.0], validate=Validate.NONE,
            ),
            UIParam.FLIP: DropdownFloatParam(
                value=12.0, min=1.0, max=90.0, incr=1.0, unit="deg",
                options=[5.0, 12.0, 30.0, 60.0, 90.0], validate=Validate.NONE,
            ),
            UIParam.FOV: DropdownFloatParam(
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
            UIParam.NSLICES: DropdownIntParam(
                value=1, min=1, max=128, incr=1, options=[1, 5, 10, 20, 40], validate=Validate.NONE,
            ),
            UIParam.NUM_SHOTS: DropdownIntParam(
                value=32, min=1, max=2048, incr=1, options=[16, 32, 64, 128, 256], validate=Validate.NONE,
            ),
            UIParam.SEQUENCE_TYPE: make_enum_param(UIParam.SEQUENCE_TYPE, SequenceType.GRADIENT_ECHO),
            UIParam.user_name(USER_SLOT_TRAJECTORY): Description(text="Trajectory (0=spiral, 1=rosette)"),
            UIParam.user_value(USER_SLOT_TRAJECTORY): DropdownFloatParam(
                value=0.0, min=0.0, max=1.0, incr=1.0, unit="", options=[0.0, 1.0], validate=Validate.NONE,
            ),
            UIParam.user_name(USER_SLOT_ORDER_MODE): Description(text="Shot order (0=uniform, 1=golden)"),
            UIParam.user_value(USER_SLOT_ORDER_MODE): DropdownFloatParam(
                value=1.0, min=0.0, max=1.0, incr=1.0, unit="", options=[0.0, 1.0], validate=Validate.NONE,
            ),
        }
        return protocol_to_dict(protocol)

    def validate_protocol(self, opts: pp.Opts, protocol: dict[str, dict]) -> dict:
        prot = dict_to_protocol(protocol)
        cfg = _read_protocol(prot)

        if cfg.te_s <= 0.0 or cfg.tr_s <= 0.0:
            return {"valid": False, "duration": None, "info": "TE and TR must be > 0"}
        if cfg.fov_m <= 0.0 or cfg.slice_thickness_m <= 0.0:
            return {"valid": False, "duration": None, "info": "FOV and slice thickness must be > 0"}
        if not (0.0 < cfg.flip_deg <= 180.0):
            return {"valid": False, "duration": None, "info": "Flip angle must be in (0, 180] deg"}
        if cfg.nx_ro < 1 or cfg.nslices < 1:
            return {"valid": False, "duration": None, "info": "NX and NSLICES must be >= 1"}
        if cfg.num_shots < 1:
            return {"valid": False, "duration": None, "info": "NumShots must be >= 1"}

        timing = _compute_timing(opts=opts, cfg=cfg, strict=True, n_inner=1)
        if timing is None:
            return {"valid": False, "duration": None, "info": "TE or TR too short for the designed waveform"}

        min_block_s = timing["min_block_s"]
        nslices_max = int(cfg.tr_s / min_block_s)
        if cfg.nslices > nslices_max:
            return {
                "valid": False,
                "duration": None,
                "info": (
                    f"TR {cfg.tr_s * 1e3:.1f} ms too short for {cfg.nslices} slice(s) "
                    f"(Tblock = {min_block_s * 1e3:.1f} ms, max {nslices_max} slice(s))"
                ),
            }

        duration_s = cfg.tr_s * float(cfg.num_shots)
        return {"valid": True, "duration": duration_s, "info": f"TA = {duration_s:.2f} s"}

    def make_sequence(self, opts: pp.Opts, protocol: dict[str, dict], output_path: str) -> None:
        prot = dict_to_protocol(protocol)
        cfg = _read_protocol(prot)

        timing = _compute_timing(opts=opts, cfg=cfg, strict=False)

        gz = timing["gz"]
        gz_reph = timing["gz_reph"]
        gz_spoil = timing["gz_spoil"]
        gx_ro = timing["gx_ro"]
        gy_ro = timing["gy_ro"]
        adc = timing["adc"]
        te_delay_s = timing["te_delay_s"]
        tr_delay_s = timing["tr_delay_s"]

        te_delay = pp.make_delay(te_delay_s) if te_delay_s > 0.0 else None
        tr_delay = pp.make_delay(tr_delay_s) if tr_delay_s > 0.0 else None

        seq = ps.Sequence(opts)

        angles = arbgrad.shot_angles(cfg.num_shots, mode=cfg.order_mode)
        slice_step_m = cfg.slice_spacing_m if cfg.nslices > 1 else 0.0
        rf_phase_deg = 0.0
        rf_phase_inc_deg = 0.0

        for shot, angle in enumerate(angles):
            rotation = ps.make_rotation(Rotation.from_euler("z", float(angle)))
            label_lin = pp.make_label(type="SET", label="LIN", value=shot)

            for sl in range(cfg.nslices):
                slice_offset_m = (sl - 0.5 * (cfg.nslices - 1)) * slice_step_m

                rf_curr = gc.copy_event(timing["rf"])
                rf_curr.freq_offset = gz.amplitude * slice_offset_m
                rf_curr.phase_offset = np.deg2rad(rf_phase_deg)
                adc_curr = gc.copy_event(adc)
                adc_curr.phase_offset = rf_curr.phase_offset

                label_slc = pp.make_label(type="SET", label="SLC", value=sl)

                seq.add_block(rf_curr, gz, label_slc, label_lin)
                seq.add_block(gz_reph)
                if te_delay is not None:
                    seq.add_block(te_delay)
                seq.add_block(gx_ro, gy_ro, adc_curr, rotation)
                seq.add_block(gz_spoil)

                rf_phase_deg = (rf_phase_deg + rf_phase_inc_deg) % 360.0
                rf_phase_inc_deg = (rf_phase_inc_deg + gc.RF_SPOILING_INC_DEG) % 360.0

            if tr_delay is not None:
                seq.add_block(tr_delay)

        seq.set_definition("Name", "gre_noncart_2d")
        seq.set_definition("FOV", [cfg.fov_m, cfg.fov_m, slice_step_m * cfg.nslices if cfg.nslices > 1 else cfg.slice_thickness_m])
        seq.set_definition("TE", cfg.te_s)
        seq.set_definition("TR", cfg.tr_s)
        seq.set_definition("Flip", cfg.flip_deg)
        seq.set_definition("ImagingMode", "2d")
        seq.set_definition("Trajectory", cfg.trajectory)
        seq.set_definition("ShotOrder", cfg.order_mode)
        seq.set_definition("RfSpoilingIncDeg", gc.RF_SPOILING_INC_DEG)
        seq.set_definition("Nx", cfg.nx_ro)
        seq.set_definition("NumShots", cfg.num_shots)
        seq.set_definition("NumSlices", cfg.nslices)
        pio.write(seq, output=output_path, remove_duplicates=False, check_timing=False)


class _Config:
    __slots__ = (
        "te_s", "tr_s", "flip_deg", "fov_m", "slice_thickness_m", "slice_spacing_m",
        "nx_ro", "nslices", "num_shots", "trajectory", "order_mode",
    )


def _read_protocol(prot: dict) -> _Config:
    cfg = _Config()
    cfg.te_s = gc.param_float(prot, UIParam.TE) * 1e-3
    cfg.tr_s = gc.param_float(prot, UIParam.TR) * 1e-3
    cfg.flip_deg = gc.param_float(prot, UIParam.FLIP)
    cfg.fov_m = gc.param_float(prot, UIParam.FOV) * 1e-3
    cfg.slice_thickness_m = gc.param_float(prot, UIParam.SLICE_THICKNESS) * 1e-3
    cfg.slice_spacing_m = gc.param_float(prot, UIParam.SLICE_SPACING) * 1e-3
    cfg.nx_ro = gc.param_int(prot, UIParam.NX)
    cfg.nslices = gc.param_int(prot, UIParam.NSLICES)
    cfg.num_shots = gc.param_int_optional(prot, UIParam.NUM_SHOTS, 32)
    cfg.trajectory = nc.trajectory_name(gc.user_float(prot, USER_SLOT_TRAJECTORY, 0.0))
    cfg.order_mode = nc.order_mode_name(gc.user_float(prot, USER_SLOT_ORDER_MODE, 1.0))
    return cfg


def _compute_timing(opts: pp.Opts, cfg: _Config, strict: bool, n_inner: int | None = None):
    if n_inner is None:
        n_inner = cfg.nslices
    gc.apply_system_derates(opts)

    rf, gz, gz_reph = gc.build_rf(opts, cfg.flip_deg, cfg.slice_thickness_m)

    _, grad_si_xy, _ = nc.build_base_waveform(opts, cfg.trajectory, cfg.fov_m, cfg.nx_ro)
    gx_ro, gy_ro = nc.build_readout_gradients(opts, grad_si_xy)
    adc = nc.build_readout_adc(opts, grad_si_xy.shape[0])

    gz_spoil = pp.make_trapezoid(channel="z", area=gc.SPOIL_FACTOR_Z / cfg.slice_thickness_m, system=opts)

    d_rf = pp.calc_duration(rf, gz)
    d_pre = pp.calc_duration(gz_reph)
    d_ro = pp.calc_duration(gx_ro, gy_ro, adc)
    d_spoil = pp.calc_duration(gz_spoil)

    rf_center_s = pp.calc_rf_center(rf)[0]
    # k-space center is the first ADC sample (waveform starts at k0 ~ 0).
    min_te_s = (d_rf - rf_center_s) + d_pre + adc.delay
    te_delay_s = cfg.te_s - min_te_s
    if te_delay_s < -1e-9 and strict:
        return None
    if te_delay_s < 0.0:
        te_delay_s = 0.0

    min_block_s = d_rf + d_pre + te_delay_s + d_ro + d_spoil
    tr_delay_s = cfg.tr_s - n_inner * min_block_s
    if tr_delay_s < -1e-9 and strict:
        return None
    if tr_delay_s < 0.0:
        tr_delay_s = 0.0

    return {
        "rf": rf,
        "gz": gz,
        "gz_reph": gz_reph,
        "gz_spoil": gz_spoil,
        "gx_ro": gx_ro,
        "gy_ro": gy_ro,
        "adc": adc,
        "te_delay_s": te_delay_s,
        "tr_delay_s": tr_delay_s,
        "min_block_s": min_block_s,
    }


PLUGIN = GreNoncart2DPulseqSequence()


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
    parser = argparse.ArgumentParser(description="Generate a 2D spiral/rosette non-Cartesian GRE .seq offline.")
    parser.add_argument("-o", "--output", default="gre_noncart_2d.seq", help="Output .seq file path")
    parser.add_argument("--te-ms", type=float)
    parser.add_argument("--tr-ms", type=float)
    parser.add_argument("--flip-deg", type=float)
    parser.add_argument("--fov-mm", type=float)
    parser.add_argument("--slice-thickness-mm", type=float)
    parser.add_argument("--slice-spacing-mm", type=float)
    parser.add_argument("--nx", type=int)
    parser.add_argument("--nslices", type=int)
    parser.add_argument("--num-shots", type=int)
    parser.add_argument("--trajectory", choices=["spiral", "rosette"])
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
    if args.trajectory is not None:
        gc.set_protocol_value(protocol, UIParam.user_value(USER_SLOT_TRAJECTORY), 1.0 if args.trajectory == "rosette" else 0.0)
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
