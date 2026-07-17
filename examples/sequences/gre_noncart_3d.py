"""Standalone 3D non-Cartesian ("stack-of-spirals"/"stack-of-rosettes") GRE.

This module implements the three mandatory module-level entry points
required by the bridge dispatcher (see ``gre_3d.py`` for the Cartesian 3D
GRE this parallels, ``gre_radial_3d.py``/``gre_noncart_2d.py`` for the
non-Cartesian siblings, and ``_noncart_common.py``/``_gre_common.py`` for
the shared building blocks):

- ``get_default_protocol(opts)``  — return the initial CV/protocol dictionary.
- ``validate_protocol(opts, protocol)``  — validate the current protocol and
  return a ``{"valid": bool, "duration": float | None, "info": str}`` dict.
- ``make_sequence(opts, protocol, output_path)``  — synthesise the sequence
  and write it to *output_path* using :func:`pulserver.io.write`.

Stack-of-spirals/rosettes: the same in-plane arbgrad base waveform (XY) used
by ``gre_noncart_2d.py`` is rotated per shot about the physical Z axis while
the partition (kz) dimension is phase-encoded with a conventional Z-channel
trapezoid, exactly as ``gre_radial_3d.py`` does for plain spokes — a
rotation about Z leaves the Z gradient channel untouched, so the two
compose in the same block without interaction. Shots are the outer loop
(one ``TR`` cycle covers a full partition sweep per shot); partitions are
the inner loop (accelerated via ``Rz``).

``NumShots`` (``IntKey.NUM_SHOTS``, native GE) is the number of shots
actually played. Trajectory shape and shot order have no native GE
counterpart, so both are ``opuser`` custom variables.

Usage
-----
Register the plugin in the interpreter tree by symlinking (or copying) this
file to the ``sequences/src/`` directory and creating a numbered alias::

    cp gre_noncart_3d.py pulserver-interpreter/package/pulserver/sequences/src/gre_noncart_3d.py
    ln -sf src/gre_noncart_3d.py pulserver-interpreter/package/pulserver/sequences/sequence10.py
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
    TypeinFloatParam,
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


class GreNoncart3DPulseqSequence(PulseqSequence):
    """Generate a 3D stack-of-spirals/rosettes non-Cartesian GRE."""

    def get_default_protocol(self, opts: pp.Opts) -> dict[str, dict]:
        del opts
        protocol = {
            UIParam.TE: DropdownFloatParam(
                value=3.0, min=0.5, max=80.0, incr=0.1, unit="ms",
                options=[1.0, 3.0, 5.0, 8.0, 15.0], validate=Validate.NONE,
            ),
            UIParam.TR: DropdownFloatParam(
                value=60.0, min=5.0, max=300.0, incr=0.1, unit="ms",
                options=[20.0, 40.0, 60.0, 100.0, 150.0], validate=Validate.NONE,
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
                value=32, min=1, max=2048, incr=1, options=[16, 32, 64, 128, 256], validate=Validate.NONE,
            ),
            UIParam.RZ: TypeinFloatParam(value=1.0, min=1.0, max=8.0, incr=1.0, unit="", validate=Validate.NONE),
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
        if cfg.fov_m <= 0.0 or cfg.slab_thickness_m <= 0.0:
            return {"valid": False, "duration": None, "info": "FOV and slab thickness must be > 0"}
        if not (0.0 < cfg.flip_deg <= 180.0):
            return {"valid": False, "duration": None, "info": "Flip angle must be in (0, 180] deg"}
        if cfg.nx_ro < 1 or cfg.npar < 1:
            return {"valid": False, "duration": None, "info": "NX and NSLICES (partitions) must be >= 1"}
        if cfg.num_shots < 1:
            return {"valid": False, "duration": None, "info": "NumShots must be >= 1"}

        timing = _compute_timing(opts=opts, cfg=cfg, strict=True, n_inner=1)
        if timing is None:
            return {"valid": False, "duration": None, "info": "TE or TR too short for the designed waveform"}

        min_block_s = timing["min_block_s"]
        npar_max = int(cfg.tr_s / min_block_s)
        if cfg.npar > npar_max:
            return {
                "valid": False,
                "duration": None,
                "info": (
                    f"TR {cfg.tr_s * 1e3:.1f} ms too short for {cfg.npar} partition(s) "
                    f"(Tblock = {min_block_s * 1e3:.1f} ms, max {npar_max} partition(s))"
                ),
            }

        duration_s = cfg.tr_s * float(cfg.num_shots)
        return {"valid": True, "duration": duration_s, "info": f"TA = {duration_s:.2f} s"}

    def make_sequence(self, opts: pp.Opts, protocol: dict[str, dict], output_path: str) -> None:
        prot = dict_to_protocol(protocol)
        cfg = _read_protocol(prot)

        timing = _compute_timing(opts=opts, cfg=cfg, strict=False)

        gz_reph = timing["gz_reph"]
        gz_spoil = timing["gz_spoil"]
        gz_pe_template = timing["gz_pe_template"]
        gx_ro = timing["gx_ro"]
        gy_ro = timing["gy_ro"]
        adc = timing["adc"]
        te_delay_s = timing["te_delay_s"]
        tr_delay_s = timing["tr_delay_s"]

        te_delay = pp.make_delay(te_delay_s) if te_delay_s > 0.0 else None
        tr_delay = pp.make_delay(tr_delay_s) if tr_delay_s > 0.0 else None

        seq = ps.Sequence(opts)

        angles = arbgrad.shot_angles(cfg.num_shots, mode=cfg.order_mode)
        par_areas, max_par_area = gc.partition_geometry(cfg.npar, cfg.slice_spacing_m)
        sampled_par = gc.sampled_lines(cfg.npar, cfg.rz, 0)

        rf_phase_deg = 0.0
        rf_phase_inc_deg = 0.0

        for shot, angle in enumerate(angles):
            rotation = ps.make_rotation(Rotation.from_euler("z", float(angle)))
            label_lin = pp.make_label(type="SET", label="LIN", value=shot)

            for par in sampled_par:
                z_scale = par_areas[par] / max_par_area if max_par_area > 0.0 else 0.0
                gz_pre_combined, gz_post_combined = gc.combined_z_gradients(
                    z_scale, gz_pe_template, gz_reph, gz_spoil, opts
                )

                rf_curr = gc.copy_event(timing["rf"])
                rf_curr.phase_offset = np.deg2rad(rf_phase_deg)
                adc_curr = gc.copy_event(adc)
                adc_curr.phase_offset = rf_curr.phase_offset

                label_par = pp.make_label(type="SET", label="PAR", value=par)
                label_slc = pp.make_label(type="SET", label="SLC", value=0)

                seq.add_block(rf_curr, timing["gz"], label_slc, label_par, label_lin)
                seq.add_block(gz_pre_combined)
                if te_delay is not None:
                    seq.add_block(te_delay)
                seq.add_block(gx_ro, gy_ro, adc_curr, rotation)
                seq.add_block(gz_post_combined)

                rf_phase_deg = (rf_phase_deg + rf_phase_inc_deg) % 360.0
                rf_phase_inc_deg = (rf_phase_inc_deg + gc.RF_SPOILING_INC_DEG) % 360.0

            if tr_delay is not None:
                seq.add_block(tr_delay)

        seq.set_definition("Name", "gre_noncart_3d")
        seq.set_definition("FOV", [cfg.fov_m, cfg.fov_m, cfg.slice_spacing_m * cfg.npar])
        seq.set_definition("TE", cfg.te_s)
        seq.set_definition("TR", cfg.tr_s)
        seq.set_definition("Flip", cfg.flip_deg)
        seq.set_definition("ImagingMode", "3d")
        seq.set_definition("Trajectory", cfg.trajectory)
        seq.set_definition("ShotOrder", cfg.order_mode)
        seq.set_definition("Rz", cfg.rz)
        seq.set_definition("RfSpoilingIncDeg", gc.RF_SPOILING_INC_DEG)
        seq.set_definition("Nx", cfg.nx_ro)
        seq.set_definition("NumShots", cfg.num_shots)
        seq.set_definition("NumPartitions", cfg.npar)
        pio.write(seq, output=output_path, remove_duplicates=False, check_timing=False)


class _Config:
    __slots__ = (
        "te_s", "tr_s", "flip_deg", "fov_m", "slab_thickness_m", "slice_spacing_m",
        "nx_ro", "npar", "rz", "num_shots", "trajectory", "order_mode",
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
    cfg.rz = max(1, int(round(gc.param_float_optional(prot, UIParam.RZ, 1.0))))
    cfg.num_shots = gc.param_int_optional(prot, UIParam.NUM_SHOTS, 32)
    cfg.trajectory = nc.trajectory_name(gc.user_float(prot, USER_SLOT_TRAJECTORY, 0.0))
    cfg.order_mode = nc.order_mode_name(gc.user_float(prot, USER_SLOT_ORDER_MODE, 1.0))
    return cfg


def _compute_timing(opts: pp.Opts, cfg: _Config, strict: bool, n_inner: int | None = None):
    if n_inner is None:
        n_inner = len(gc.sampled_lines(cfg.npar, cfg.rz, 0))
    gc.apply_system_derates(opts)

    rf, gz, gz_reph = gc.build_rf(opts, cfg.flip_deg, cfg.slab_thickness_m)

    _, grad_si_xy, _ = nc.build_base_waveform(opts, cfg.trajectory, cfg.fov_m, cfg.nx_ro)
    gx_ro, gy_ro = nc.build_readout_gradients(opts, grad_si_xy)
    adc = nc.build_readout_adc(opts, grad_si_xy.shape[0])

    gz_spoil = pp.make_trapezoid(channel="z", area=gc.SPOIL_FACTOR_Z / cfg.slab_thickness_m, system=opts)

    _, max_par_area = gc.partition_geometry(cfg.npar, cfg.slice_spacing_m)
    gz_pe_template = pp.make_trapezoid(channel="z", area=max_par_area, system=opts) if max_par_area > 0.0 else None
    gz_pre_worst, gz_post_worst = gc.z_worst_case_trapezoids(gz_reph, gz_spoil, max_par_area, opts)

    d_rf = pp.calc_duration(rf, gz)
    d_pre = pp.calc_duration(gz_pre_worst)
    d_ro = pp.calc_duration(gx_ro, gy_ro, adc)
    d_post = pp.calc_duration(gz_post_worst)

    rf_center_s = pp.calc_rf_center(rf)[0]
    min_te_s = (d_rf - rf_center_s) + d_pre + adc.delay
    te_delay_s = cfg.te_s - min_te_s
    if te_delay_s < -1e-9 and strict:
        return None
    if te_delay_s < 0.0:
        te_delay_s = 0.0

    min_block_s = d_rf + d_pre + te_delay_s + d_ro + d_post
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
        "gz_pe_template": gz_pe_template,
        "gx_ro": gx_ro,
        "gy_ro": gy_ro,
        "adc": adc,
        "te_delay_s": te_delay_s,
        "tr_delay_s": tr_delay_s,
        "min_block_s": min_block_s,
    }


PLUGIN = GreNoncart3DPulseqSequence()


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
    parser = argparse.ArgumentParser(description="Generate a 3D stack-of-spirals/rosettes GRE .seq offline.")
    parser.add_argument("-o", "--output", default="gre_noncart_3d.seq", help="Output .seq file path")
    parser.add_argument("--te-ms", type=float)
    parser.add_argument("--tr-ms", type=float)
    parser.add_argument("--flip-deg", type=float)
    parser.add_argument("--fov-mm", type=float)
    parser.add_argument("--slab-thickness-mm", type=float, dest="slice_thickness_mm")
    parser.add_argument("--partition-spacing-mm", type=float, dest="slice_spacing_mm")
    parser.add_argument("--nx", type=int)
    parser.add_argument("--npartitions", type=int, dest="nslices")
    parser.add_argument("--num-shots", type=int)
    parser.add_argument("--rz", type=float)
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
    if args.rz is not None:
        gc.set_protocol_value(protocol, UIParam.RZ, args.rz)
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
