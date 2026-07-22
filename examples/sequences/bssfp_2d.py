"""Optimized slice-selective 2D bSSFP/TRUFI sequence plugin."""

from __future__ import annotations

import sys

import numpy as np
import pulserver.io as pio
import pulserver.pypulseq as pp
from pulserver import (
    DropdownFloatParam,
    DropdownIntParam,
    Sequence,
    SequenceType,
    TriggerType,
    TypeinFloatParam,
    UIParam,
    Validate,
    dict_to_protocol,
    make_enum_param,
    params,
    protocol_to_dict,
    run_cli,
)


class Bssfp2DPulseqSequence(Sequence):
    """Generate one optimized continuous-gradient 2D bSSFP acquisition."""

    def get_default_protocol(self, opts: pp.Opts) -> dict[str, dict]:
        del opts
        return protocol_to_dict(
            {
                UIParam.FLIP: DropdownFloatParam(
                    value=35.0, min=1.0, max=90.0, incr=1.0, unit="deg",
                    options=[15.0, 25.0, 35.0, 50.0, 70.0], validate=Validate.NONE,
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
                    value=5.0, min=1.0, max=20.0, incr=0.5, unit="mm",
                    options=[3.0, 5.0, 8.0], validate=Validate.NONE,
                ),
                UIParam.NX: DropdownIntParam(
                    value=64, min=16, max=512, incr=1,
                    options=[64, 128, 192, 256], validate=Validate.NONE,
                ),
                UIParam.NY: DropdownIntParam(
                    value=64, min=8, max=512, incr=1,
                    options=[32, 64, 128, 192], validate=Validate.NONE,
                ),
                UIParam.BANDWIDTH: TypeinFloatParam(
                    value=100_000.0, min=5_000.0, max=500_000.0, incr=100.0,
                    unit="Hz/px", validate=Validate.NONE,
                ),
                UIParam.NUM_FRAMES: DropdownIntParam(
                    value=1, min=1, max=10000, incr=1, options=[1, 10, 20, 50], validate=Validate.NONE,
                ),
                UIParam.TRIGGER_TYPE: make_enum_param(UIParam.TRIGGER_TYPE, TriggerType.NONE),
                UIParam.SEQUENCE_TYPE: make_enum_param(
                    UIParam.SEQUENCE_TYPE, SequenceType.GRADIENT_ECHO
                ),
            }
        )

    def validate_protocol(self, opts: pp.Opts, protocol: dict[str, dict]) -> dict:
        cfg = _read_protocol(dict_to_protocol(protocol))
        if cfg.num_frames < 1:
            return {"valid": False, "duration": None, "info": "NUM_FRAMES must be >= 1"}
        try:
            train = _make_train(opts, cfg)
            duration = sum(pp.calc_duration(*block) for block in train) * cfg.num_frames
            if cfg.trigger != TriggerType.NONE:
                duration += 1e-3 * cfg.num_frames
        except (TypeError, ValueError) as error:
            return {"valid": False, "duration": None, "info": str(error)}
        return {
            "valid": True,
            "duration": duration,
            "info": f"TA = {duration:.2f} s; TR = {train.tr_s * 1e3:.2f} ms",
        }

    def make_sequence(self, opts: pp.Opts, protocol: dict[str, dict], output_path: str) -> None:
        cfg = _read_protocol(dict_to_protocol(protocol))
        train = _make_train(opts, cfg)
        seq = pp.Sequence(opts)
        for frame in range(cfg.num_frames):
            phase_label = pp.make_label(type="SET", label="PHS", value=frame)
            if cfg.trigger != TriggerType.NONE:
                seq.add_block(pp.make_trigger(cfg.trigger, duration=1e-3, system=opts), phase_label)
                phase_label = None
            train.set_state(rf_phase_rad=0.0)
            for block_idx, block in enumerate(train):
                labels = (phase_label,) if block_idx == 0 and phase_label is not None else ()
                seq.add_block(*block, *labels)
        seq.set_definition("Name", "bssfp_2d")
        seq.set_definition("FOV", [cfg.fov_x_m, cfg.fov_y_m, cfg.slice_thickness_m])
        seq.set_definition("TE", train.te_s)
        seq.set_definition("TR", train.tr_s)
        seq.set_definition("Flip", cfg.flip_deg)
        seq.set_definition("ImagingMode", "2d")
        seq.set_definition("NumFrames", cfg.num_frames)
        seq.set_definition("TriggerType", str(cfg.trigger))
        pio.write(seq, output=output_path, remove_duplicates=False, check_timing=False)


class _Config:
    pass


def _read_protocol(prot) -> _Config:
    cfg = _Config()
    cfg.flip_deg = params.param_float(prot, UIParam.FLIP)
    cfg.fov_x_m = params.param_float(prot, UIParam.FOV) * 1e-3
    cfg.fov_y_m = params.phase_fov_mm_from_protocol(prot) * 1e-3
    cfg.slice_thickness_m = params.param_float(prot, UIParam.SLICE_THICKNESS) * 1e-3
    cfg.nx = params.param_int(prot, UIParam.NX)
    cfg.ny = params.param_int(prot, UIParam.NY)
    cfg.bandwidth = params.param_float_optional(prot, UIParam.BANDWIDTH, 100_000.0)
    cfg.num_frames = params.param_int_optional(prot, UIParam.NUM_FRAMES, 1)
    cfg.trigger = prot[str(UIParam.TRIGGER_TYPE)].value
    return cfg


def _make_train(opts, cfg):
    excitation = pp.make_slice_selective_pulse(
        np.deg2rad(cfg.flip_deg), cfg.slice_thickness_m, duration=0.6e-3, system=opts
    )
    return pp.make_bssfp_readout(
        opts,
        (cfg.fov_x_m, cfg.fov_y_m),
        (cfg.nx, cfg.ny),
        excitation,
        bandwidth_hz_px=cfg.bandwidth,
    )


PLUGIN = Bssfp2DPulseqSequence()
get_default_protocol = PLUGIN.get_default_protocol
validate_protocol = PLUGIN.validate_protocol
make_sequence = PLUGIN.make_sequence
makeSeq = PLUGIN.make_sequence

_ARG_MAP = [
    ("--flip-deg", UIParam.FLIP, float, ""),
    ("--fov-mm", UIParam.FOV, float, ""),
    ("--phase-fov-mm", UIParam.PHASE_FOV, float, ""),
    ("--slice-thickness-mm", UIParam.SLICE_THICKNESS, float, ""),
    ("--nx", UIParam.NX, int, ""),
    ("--ny", UIParam.NY, int, ""),
    ("--bandwidth-hz-px", UIParam.BANDWIDTH, float, ""),
    ("--num-frames", UIParam.NUM_FRAMES, int, ""),
    ("--trigger", UIParam.TRIGGER_TYPE, {"none": "none", "respiratory": "physio1", "cardiac": "physio2"}, ""),
]

if __name__ == "__main__":
    raise SystemExit(run_cli(PLUGIN, sys.argv[1:], arg_map=_ARG_MAP, default_output="bssfp_2d.seq"))
