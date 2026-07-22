"""Optimized segmented 3D bSSFP/TRUFI sequence plugin."""

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

USER_SLOT_SELECTIVE = 0


class Bssfp3DPulseqSequence(Sequence):
    """Generate segmented slab-selective or nonselective 3D bSSFP."""

    def get_default_protocol(self, opts: pp.Opts) -> dict[str, dict]:
        del opts
        return protocol_to_dict(
            {
                UIParam.FLIP: DropdownFloatParam(
                    value=25.0, min=1.0, max=90.0, incr=1.0, unit="deg",
                    options=[15.0, 25.0, 35.0, 50.0], validate=Validate.NONE,
                ),
                UIParam.FOV: DropdownFloatParam(
                    value=220.0, min=80.0, max=500.0, incr=1.0, unit="mm",
                    options=[180.0, 220.0, 280.0], validate=Validate.NONE,
                ),
                UIParam.PHASE_FOV: DropdownFloatParam(
                    value=220.0, min=80.0, max=500.0, incr=1.0, unit="mm",
                    options=[180.0, 220.0, 280.0], validate=Validate.NONE,
                ),
                UIParam.SLICE_THICKNESS: DropdownFloatParam(
                    value=160.0, min=10.0, max=300.0, incr=1.0, unit="mm",
                    options=[120.0, 160.0, 220.0], validate=Validate.NONE,
                ),
                UIParam.NX: DropdownIntParam(
                    value=64, min=16, max=512, incr=1,
                    options=[64, 128, 192], validate=Validate.NONE,
                ),
                UIParam.NY: DropdownIntParam(
                    value=32, min=4, max=512, incr=1,
                    options=[16, 32, 64], validate=Validate.NONE,
                ),
                UIParam.NSLICES: DropdownIntParam(
                    value=8, min=1, max=256, incr=1,
                    options=[4, 8, 16, 32], validate=Validate.NONE,
                ),
                UIParam.NUM_SHOTS: DropdownIntParam(
                    value=8, min=1, max=256, incr=1,
                    options=[1, 2, 4, 8, 16], validate=Validate.NONE,
                ),
                UIParam.BANDWIDTH: TypeinFloatParam(
                    value=100_000.0, min=5_000.0, max=500_000.0, incr=100.0,
                    unit="Hz/px", validate=Validate.NONE,
                ),
                UIParam.SEQUENCE_TYPE: make_enum_param(
                    UIParam.SEQUENCE_TYPE, SequenceType.GRADIENT_ECHO
                ),
                UIParam.user_name(USER_SLOT_SELECTIVE): Description(
                    text="Excitation (0=nonselective, 1=slab-selective)"
                ),
                UIParam.user_value(USER_SLOT_SELECTIVE): DropdownFloatParam(
                    value=0.0, min=0.0, max=1.0, incr=1.0, unit="",
                    options=[0.0, 1.0], validate=Validate.NONE,
                ),
            }
        )

    def validate_protocol(self, opts: pp.Opts, protocol: dict[str, dict]) -> dict:
        cfg = _read_protocol(dict_to_protocol(protocol))
        try:
            train = _make_train(opts, cfg)
            one_segment = sum(pp.calc_duration(*block) for block in train)
        except (TypeError, ValueError) as error:
            return {"valid": False, "duration": None, "info": str(error)}
        duration = one_segment * train.num_segments
        return {
            "valid": True,
            "duration": duration,
            "info": f"TA = {duration:.2f} s; {train.num_segments} segments; TR = {train.tr_s * 1e3:.2f} ms",
        }

    def make_sequence(self, opts: pp.Opts, protocol: dict[str, dict], output_path: str) -> None:
        cfg = _read_protocol(dict_to_protocol(protocol))
        train = _make_train(opts, cfg)
        seq = pp.Sequence(opts)
        for segment_idx in range(train.num_segments):
            train.set_state(segment_idx=segment_idx, rf_phase_rad=0.0)
            for block in train:
                seq.add_block(*block)
        seq.set_definition("Name", "bssfp_3d")
        seq.set_definition("FOV", [cfg.fov_x_m, cfg.fov_y_m, cfg.fov_z_m])
        seq.set_definition("TE", train.te_s)
        seq.set_definition("TR", train.tr_s)
        seq.set_definition("Flip", cfg.flip_deg)
        seq.set_definition("ImagingMode", "3d")
        seq.set_definition("Segments", cfg.segments)
        seq.set_definition("SlabSelective", cfg.selective)
        pio.write(seq, output=output_path, remove_duplicates=False, check_timing=False)


class _Config:
    pass


def _read_protocol(prot) -> _Config:
    cfg = _Config()
    cfg.flip_deg = params.param_float(prot, UIParam.FLIP)
    cfg.fov_x_m = params.param_float(prot, UIParam.FOV) * 1e-3
    cfg.fov_y_m = params.phase_fov_mm_from_protocol(prot) * 1e-3
    cfg.fov_z_m = params.param_float(prot, UIParam.SLICE_THICKNESS) * 1e-3
    cfg.nx = params.param_int(prot, UIParam.NX)
    cfg.ny = params.param_int(prot, UIParam.NY)
    cfg.nz = params.param_int(prot, UIParam.NSLICES)
    cfg.segments = params.param_int_optional(prot, UIParam.NUM_SHOTS, 8)
    cfg.bandwidth = params.param_float_optional(prot, UIParam.BANDWIDTH, 100_000.0)
    cfg.selective = params.user_float(prot, USER_SLOT_SELECTIVE, 0.0) >= 0.5
    return cfg


def _make_train(opts, cfg):
    if cfg.selective:
        excitation = pp.make_slice_selective_pulse(
            np.deg2rad(cfg.flip_deg), cfg.fov_z_m, duration=0.6e-3, system=opts
        )
    else:
        excitation = pp.make_hard_pulse(
            np.deg2rad(cfg.flip_deg), duration=0.5e-3, system=opts
        )
    return pp.make_bssfp_readout(
        opts,
        (cfg.fov_x_m, cfg.fov_y_m, cfg.fov_z_m),
        (cfg.nx, cfg.ny, cfg.nz),
        excitation,
        segment=cfg.segments,
        bandwidth_hz_px=cfg.bandwidth,
    )


PLUGIN = Bssfp3DPulseqSequence()
get_default_protocol = PLUGIN.get_default_protocol
validate_protocol = PLUGIN.validate_protocol
make_sequence = PLUGIN.make_sequence
makeSeq = PLUGIN.make_sequence

_ARG_MAP = [
    ("--flip-deg", UIParam.FLIP, float, ""),
    ("--fov-mm", UIParam.FOV, float, ""),
    ("--phase-fov-mm", UIParam.PHASE_FOV, float, ""),
    ("--slab-thickness-mm", UIParam.SLICE_THICKNESS, float, ""),
    ("--nx", UIParam.NX, int, ""),
    ("--ny", UIParam.NY, int, ""),
    ("--npartitions", UIParam.NSLICES, int, ""),
    ("--segments", UIParam.NUM_SHOTS, int, ""),
    ("--bandwidth-hz-px", UIParam.BANDWIDTH, float, ""),
    ("--slab-selective", UIParam.user_value(USER_SLOT_SELECTIVE), ("const", 1.0), ""),
]

if __name__ == "__main__":
    raise SystemExit(run_cli(PLUGIN, sys.argv[1:], arg_map=_ARG_MAP, default_output="bssfp_3d.seq"))
