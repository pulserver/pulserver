"""Standalone 2D radial (non-Cartesian) GRE sequence plugin for pulserver.

This module implements the three mandatory module-level entry points
required by the bridge dispatcher (see ``gre.py`` for the Cartesian 2D GRE
this parallels, and ``_gre_common.py`` for the shared readout/echo-train
helper reused here unmodified):

- ``get_default_protocol(opts)``  — return the initial CV/protocol dictionary.
- ``validate_protocol(opts, protocol)``  — validate the current protocol and
  return a ``{"valid": bool, "duration": float | None, "info": str}`` dict.
- ``make_sequence(opts, protocol, output_path)``  — synthesise the sequence
  and write it to *output_path* using :func:`pulserver.io.write`.

Full-echo (diameter, not half-spoke) radial trajectory: a single readout
trapezoid on the logical X axis (prephased to -kmax, read through the
center to +kmax) is designed ONCE via ``_gre_common``'s standard
single-echo readout/echo-train builder (``ro_axis`` pinned to ``"x"``), then
replayed for every spoke with a per-shot in-plane rotation (pulseq
``ROTATIONS`` extension, ``pulserver.pulseq.make_rotation`` — no separate
gradient waveform per spoke, no ``pulserver.arbgrad`` solver needed, per the
project's design rule that plain spokes don't need slew-limited waveform
design). Spoke angles come from ``pulserver.arbgrad.shot_angles`` (uniform
or golden-angle ordering) — reused here purely as an angle utility, no
arbgrad *waveform* design is involved for radial.

``NumShots`` (``IntKey.NUM_SHOTS``, a real native GE variable) is the number
of radial spokes. The spoke-ordering mode has no native GE counterpart, so
it is carried as an ``opuser`` custom variable.

Usage
-----
Register the plugin in the interpreter tree by symlinking (or copying) this
file to the ``sequences/src/`` directory and creating a numbered alias::

    cp gre_radial_2d.py pulserver-interpreter/package/pulserver/sequences/src/gre_radial_2d.py
    ln -sf src/gre_radial_2d.py pulserver-interpreter/package/pulserver/sequences/sequence7.py
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
from pulserver.core import SequenceType


def _load_sibling_module(name: str):
    """Load a same-directory helper module by file path (see gre_multiecho_2d.py)."""
    module_path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gc = _load_sibling_module("_gre_common")

NUM_ECHOES = 1
FLYBACK = True
RO_AXIS = "x"  # logical readout axis; per-shot rotation supplies the physical angle

USER_SLOT_ORDER_MODE = 0


def _order_mode_name(code: float) -> str:
    return "golden" if code >= 0.5 else "uniform"


class GreRadial2DPulseqSequence(PulseqSequence):
    """Generate a 2D full-echo radial (stack-of-stars-free, single-slice) GRE."""

    def get_default_protocol(self, opts: pp.Opts) -> dict[str, dict]:
        del opts
        protocol = {
            UIParam.TE: DropdownFloatParam(
                value=3.0, min=1.5, max=80.0, incr=0.1, unit="ms",
                options=[2.0, 3.0, 5.0, 8.0, 15.0], validate=Validate.NONE,
            ),
            UIParam.TR: DropdownFloatParam(
                value=10.0, min=5.0, max=2000.0, incr=0.1, unit="ms",
                options=[8.0, 10.0, 20.0, 50.0, 100.0], validate=Validate.NONE,
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
                value=100, min=8, max=2048, incr=1, options=[50, 100, 200, 400, 800], validate=Validate.NONE,
            ),
            UIParam.BANDWIDTH: TypeinFloatParam(
                value=gc.DEFAULT_BANDWIDTH_HZ_PX, min=5_000.0, max=500_000.0, incr=100.0,
                unit="Hz/px", validate=Validate.NONE,
            ),
            UIParam.SEQUENCE_TYPE: make_enum_param(UIParam.SEQUENCE_TYPE, SequenceType.GRADIENT_ECHO),
            UIParam.user_name(USER_SLOT_ORDER_MODE): Description(text="Spoke order (0=uniform, 1=golden)"),
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
        if cfg.bandwidth_hz_px <= 0.0:
            return {"valid": False, "duration": None, "info": "Bandwidth must be > 0"}
        if cfg.num_shots < 1:
            return {"valid": False, "duration": None, "info": "NumShots must be >= 1"}

        timing = _compute_timing(opts=opts, cfg=cfg, strict=True, n_inner=1)
        if timing is None:
            return {"valid": False, "duration": None, "info": "TE or TR too short for gradients and readout timing"}

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
        echo = timing["echo"]
        te_delay_s = timing["te_delay_s"]
        tr_delay_s = timing["tr_delay_s"]

        te_delay = pp.make_delay(te_delay_s) if te_delay_s > 0.0 else None
        tr_delay = pp.make_delay(tr_delay_s) if tr_delay_s > 0.0 else None

        seq = ps.Sequence(opts)

        angles = arbgrad.shot_angles(cfg.num_shots, mode=cfg.order_mode)
        slice_step_m = cfg.slice_spacing_m if cfg.nslices > 1 else 0.0
        rf_phase_deg = 0.0
        rf_phase_inc_deg = 0.0

        for spoke, angle in enumerate(angles):
            rotation = ps.make_rotation(Rotation.from_euler("z", float(angle)))
            label_lin = pp.make_label(type="SET", label="LIN", value=spoke)

            for sl in range(cfg.nslices):
                slice_offset_m = (sl - 0.5 * (cfg.nslices - 1)) * slice_step_m

                rf_curr = gc.copy_event(timing["rf"])
                rf_curr.freq_offset = gz.amplitude * slice_offset_m
                rf_curr.phase_offset = np.deg2rad(rf_phase_deg)
                adc_curr = gc.copy_event(echo["adc"])
                adc_curr.phase_offset = rf_curr.phase_offset

                label_slc = pp.make_label(type="SET", label="SLC", value=sl)

                seq.add_block(rf_curr, gz, label_slc, label_lin)
                seq.add_block(echo["gx_pre"], gz_reph, rotation)
                if te_delay is not None:
                    seq.add_block(te_delay)
                seq.add_block(echo["gx_echo"], adc_curr, rotation)
                seq.add_block(echo["gx_spoil"], gz_spoil, rotation)

                rf_phase_deg = (rf_phase_deg + rf_phase_inc_deg) % 360.0
                rf_phase_inc_deg = (rf_phase_inc_deg + gc.RF_SPOILING_INC_DEG) % 360.0

            if tr_delay is not None:
                seq.add_block(tr_delay)

        seq.set_definition("Name", "gre_radial_2d")
        seq.set_definition("FOV", [cfg.fov_m, cfg.fov_m, slice_step_m * cfg.nslices if cfg.nslices > 1 else cfg.slice_thickness_m])
        seq.set_definition("TE", cfg.te_s)
        seq.set_definition("TR", cfg.tr_s)
        seq.set_definition("Flip", cfg.flip_deg)
        seq.set_definition("ImagingMode", "2d")
        seq.set_definition("Trajectory", "radial")
        seq.set_definition("SpokeOrder", cfg.order_mode)
        seq.set_definition("BandwidthHzPerPx", cfg.bandwidth_hz_px)
        seq.set_definition("RfSpoilingIncDeg", gc.RF_SPOILING_INC_DEG)
        seq.set_definition("Nx", cfg.nx_ro)
        seq.set_definition("NumShots", cfg.num_shots)
        seq.set_definition("NumSlices", cfg.nslices)
        pio.write(seq, output=output_path, remove_duplicates=False, check_timing=False)


class _Config:
    __slots__ = (
        "te_s", "tr_s", "flip_deg", "fov_m", "slice_thickness_m", "slice_spacing_m",
        "nx_ro", "nslices", "bandwidth_hz_px", "num_shots", "order_mode",
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
    cfg.bandwidth_hz_px = gc.param_float_optional(prot, UIParam.BANDWIDTH, gc.DEFAULT_BANDWIDTH_HZ_PX)
    cfg.num_shots = gc.param_int_optional(prot, UIParam.NUM_SHOTS, 100)
    cfg.order_mode = _order_mode_name(gc.user_float(prot, USER_SLOT_ORDER_MODE, 1.0))
    return cfg


def _compute_timing(opts: pp.Opts, cfg: _Config, strict: bool, n_inner: int | None = None):
    if n_inner is None:
        n_inner = cfg.nslices
    gc.apply_system_derates(opts)

    rf, gz, gz_reph = gc.build_rf(opts, cfg.flip_deg, cfg.slice_thickness_m)

    echo = gc.compute_readout_and_echo_train(
        opts=opts,
        ro_axis=RO_AXIS,
        nx_ro=cfg.nx_ro,
        fov_ro_m=cfg.fov_m,
        bandwidth_hz_px=cfg.bandwidth_hz_px,
        slice_thickness_m=cfg.slice_thickness_m,
        num_echoes=NUM_ECHOES,
        echo_spacing_s=0.0,
        flyback=FLYBACK,
        strict=strict,
    )
    if echo is None:
        return None

    gz_spoil = pp.make_trapezoid(channel="z", area=gc.SPOIL_FACTOR_Z / cfg.slice_thickness_m, system=opts)

    d_rf = pp.calc_duration(rf, gz)
    d_pre = pp.calc_duration(echo["gx_pre"], gz_reph)
    d_post = pp.calc_duration(echo["gx_spoil"], gz_spoil)

    rf_center_s = pp.calc_rf_center(rf)[0]
    min_te_s = (d_rf - rf_center_s) + d_pre + echo["adc_center_s"]
    te_delay_s = cfg.te_s - min_te_s
    if te_delay_s < -1e-9 and strict:
        return None
    if te_delay_s < 0.0:
        te_delay_s = 0.0

    min_block_s = d_rf + d_pre + te_delay_s + echo["echo_train_span_s"] + d_post
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
        "echo": echo,
        "te_delay_s": te_delay_s,
        "tr_delay_s": tr_delay_s,
        "min_block_s": min_block_s,
    }


PLUGIN = GreRadial2DPulseqSequence()


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
    parser = argparse.ArgumentParser(description="Generate a 2D full-echo radial GRE .seq offline.")
    parser.add_argument("-o", "--output", default="gre_radial_2d.seq", help="Output .seq file path")
    parser.add_argument("--te-ms", type=float)
    parser.add_argument("--tr-ms", type=float)
    parser.add_argument("--flip-deg", type=float)
    parser.add_argument("--fov-mm", type=float)
    parser.add_argument("--slice-thickness-mm", type=float)
    parser.add_argument("--slice-spacing-mm", type=float)
    parser.add_argument("--nx", type=int)
    parser.add_argument("--nslices", type=int)
    parser.add_argument("--num-shots", type=int)
    parser.add_argument("--bandwidth-hz-px", type=float)
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
    if args.bandwidth_hz_px is not None:
        gc.set_protocol_value(protocol, UIParam.BANDWIDTH, args.bandwidth_hz_px)
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
