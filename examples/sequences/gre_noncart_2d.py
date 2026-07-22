"""Standalone 2D non-Cartesian (spiral/rosette) GRE sequence plugin.

This module implements the three mandatory module-level entry points
required by the bridge dispatcher (see ``gre_2d.py`` for the Cartesian 2D GRE
this parallels, ``gre_radial_2d.py`` for the plain-trapezoid non-Cartesian
sibling, and ``_noncart_common.py``/``_gre_common.py`` for the shared
building blocks):

- ``get_default_protocol(opts)``  — return the initial CV/protocol dictionary.
- ``validate_protocol(opts, protocol)``  — validate the current protocol and
  return a ``{"valid": bool, "duration": float | None, "info": str}`` dict.
- ``make_sequence(opts, protocol, output_path)``  — synthesise the sequence
  and write it to *output_path* using :func:`pulserver.io.write`.

A single slew/gradient-limited base waveform (spiral or rosette, chosen via
an ``opuser`` toggle) is designed once via the public readout factories and
replayed per shot with a per-shot in-plane rotation (pulseq ``ROTATIONS``
extension, ``pulserver.pypulseq.make_rotation``) — the C++ solver never loops
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
    TriggerType,
    UIParam,
    Validate,
    dict_to_protocol,
    make_enum_param,
    params,
    protocol_to_dict,
    run_cli,
)

USER_SLOT_TRAJECTORY = 0
USER_SLOT_ORDER_MODE = 1


class GreNoncart2DPulseqSequence(Sequence):
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
            UIParam.NUM_FRAMES: DropdownIntParam(
                value=1, min=1, max=10000, incr=1, options=[1, 10, 20, 50], validate=Validate.NONE,
            ),
            UIParam.TRIGGER_TYPE: make_enum_param(UIParam.TRIGGER_TYPE, TriggerType.NONE),
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
        if cfg.num_frames < 1:
            return {"valid": False, "duration": None, "info": "NUM_FRAMES must be >= 1"}

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

        duration_s = cfg.tr_s * float(cfg.num_shots) * cfg.num_frames
        if cfg.trigger != TriggerType.NONE:
            duration_s += 1e-3 * cfg.num_frames
        return {"valid": True, "duration": duration_s, "info": f"TA = {duration_s:.2f} s"}

    def make_sequence(self, opts: pp.Opts, protocol: dict[str, dict], output_path: str) -> None:
        prot = dict_to_protocol(protocol)
        cfg = _read_protocol(prot)

        timing = _compute_timing(opts=opts, cfg=cfg, strict=False)

        pulse = timing["pulse"]
        readout_module = timing["readout"]
        te_delay_s = timing["te_delay_s"]
        tr_delay_s = timing["tr_delay_s"]

        te_delay = pp.make_delay(te_delay_s) if te_delay_s > 0.0 else None
        tr_delay = pp.make_delay(tr_delay_s) if tr_delay_s > 0.0 else None

        seq = pp.Sequence(opts)

        rotations = pp.make_radial_tilt(cfg.num_shots, scheme=cfg.order_mode).to_rotations()
        slice_step_m = cfg.slice_spacing_m if cfg.nslices > 1 else 0.0
        rf_phases = pp.make_rf_spoiling_schedule(cfg.num_shots * cfg.nslices * cfg.num_frames)
        phase_idx = 0

        for frame in range(cfg.num_frames):
            phase_label = pp.make_label(type="SET", label="PHS", value=frame)
            if cfg.trigger != TriggerType.NONE:
                seq.add_block(pp.make_trigger(cfg.trigger, duration=1e-3, system=opts), phase_label)
            else:
                seq.add_block(phase_label)
            for shot, rotation in enumerate(rotations):
                for sl in range(cfg.nslices):
                    slice_offset_m = (sl - 0.5 * (cfg.nslices - 1)) * slice_step_m
                    phase = float(rf_phases[phase_idx])
                    pulse.set_state(
                        freq_offset_hz=pulse.gradients[0].amplitude * slice_offset_m,
                        phase_offset_rad=phase,
                    )
                    for block_idx, block in enumerate(pulse):
                        labels = (pp.make_label(type="SET", label="SLC", value=sl),) if block_idx == 0 else ()
                        seq.add_block(*block, *labels)
                    if te_delay is not None:
                        seq.add_block(te_delay)
                    readout_module.set_state(
                        lin_idx=shot, adc_phase_rad=phase, rotation=rotation
                    )
                    for block in readout_module:
                        seq.add_block(*block)
                    phase_idx += 1

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
        seq.set_definition("RfSpoilingIncDeg", 117.0)
        seq.set_definition("Nx", cfg.nx_ro)
        seq.set_definition("NumShots", cfg.num_shots)
        seq.set_definition("NumSlices", cfg.nslices)
        seq.set_definition("NumFrames", cfg.num_frames)
        seq.set_definition("TriggerType", str(cfg.trigger))
        pio.write(seq, output=output_path, remove_duplicates=False, check_timing=False)


class _Config:
    __slots__ = (
        "te_s", "tr_s", "flip_deg", "fov_m", "slice_thickness_m", "slice_spacing_m",
        "nx_ro", "nslices", "num_shots", "trajectory", "order_mode", "num_frames", "trigger",
    )


def _read_protocol(prot: dict) -> _Config:
    cfg = _Config()
    cfg.te_s = params.param_float(prot, UIParam.TE) * 1e-3
    cfg.tr_s = params.param_float(prot, UIParam.TR) * 1e-3
    cfg.flip_deg = params.param_float(prot, UIParam.FLIP)
    cfg.fov_m = params.param_float(prot, UIParam.FOV) * 1e-3
    cfg.slice_thickness_m = params.param_float(prot, UIParam.SLICE_THICKNESS) * 1e-3
    cfg.slice_spacing_m = params.param_float(prot, UIParam.SLICE_SPACING) * 1e-3
    cfg.nx_ro = params.param_int(prot, UIParam.NX)
    cfg.nslices = params.param_int(prot, UIParam.NSLICES)
    cfg.num_shots = params.param_int_optional(prot, UIParam.NUM_SHOTS, 32)
    cfg.trajectory = "rosette" if params.user_float(prot, USER_SLOT_TRAJECTORY, 0.0) >= 0.5 else "spiral"
    cfg.order_mode = "golden" if params.user_float(prot, USER_SLOT_ORDER_MODE, 1.0) >= 0.5 else "uniform"
    cfg.num_frames = params.param_int_optional(prot, UIParam.NUM_FRAMES, 1)
    cfg.trigger = prot[str(UIParam.TRIGGER_TYPE)].value
    return cfg


def _compute_timing(opts: pp.Opts, cfg: _Config, strict: bool, n_inner: int | None = None):
    if n_inner is None:
        n_inner = cfg.nslices
    pulse = pp.make_slice_selective_pulse(
        np.deg2rad(cfg.flip_deg), cfg.slice_thickness_m, system=opts
    )
    if cfg.trajectory == "spiral":
        readout_module = pp.make_spiral_readout(
            opts,
            cfg.fov_m,
            cfg.nx_ro,
            cfg.num_shots,
            slice_rephasing=pulse.rephasers[0],
        )
    else:
        readout_module = pp.make_rosette_readout(
            opts, cfg.fov_m, cfg.nx_ro, slice_rephasing=pulse.rephasers[0]
        )

    d_pulse = sum(pp.calc_duration(*block) for block in pulse)
    rf_center_s = pp.calc_rf_center(pulse.rf)[0] + pulse.rf.delay
    min_te_s = d_pulse - rf_center_s + readout_module.t_prephase_s
    raster = opts.block_duration_raster
    te_delay_s = round((cfg.te_s - min_te_s) / raster) * raster
    if te_delay_s < -1e-9 and strict:
        return None
    if te_delay_s < 0.0:
        te_delay_s = 0.0

    min_block_s = d_pulse + te_delay_s + readout_module.duration
    tr_delay_s = round((cfg.tr_s - n_inner * min_block_s) / raster) * raster
    if tr_delay_s < -1e-9 and strict:
        return None
    if tr_delay_s < 0.0:
        tr_delay_s = 0.0

    return {
        "pulse": pulse,
        "readout": readout_module,
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


_ARG_MAP = [
    ('--te-ms', UIParam.TE, float, ""),
    ('--tr-ms', UIParam.TR, float, ""),
    ('--flip-deg', UIParam.FLIP, float, ""),
    ('--fov-mm', UIParam.FOV, float, ""),
    ('--slice-thickness-mm', UIParam.SLICE_THICKNESS, float, ""),
    ('--slice-spacing-mm', UIParam.SLICE_SPACING, float, ""),
    ('--nx', UIParam.NX, int, ""),
    ('--nslices', UIParam.NSLICES, int, ""),
    ('--num-shots', UIParam.NUM_SHOTS, int, ""),
    ('--trajectory', UIParam.user_value(USER_SLOT_TRAJECTORY), {'spiral': 0.0, 'rosette': 1.0}, ""),
    ('--order-mode', UIParam.user_value(USER_SLOT_ORDER_MODE), {'uniform': 0.0, 'golden': 1.0}, ""),
    ('--num-frames', UIParam.NUM_FRAMES, int, ""),
    ('--trigger', UIParam.TRIGGER_TYPE, {'none': 'none', 'respiratory': 'physio1', 'cardiac': 'physio2'}, ""),
]

if __name__ == "__main__":
    raise SystemExit(
        run_cli(
            PLUGIN,
            sys.argv[1:],
            arg_map=_ARG_MAP,
            description='Generate a 2D spiral/rosette non-Cartesian GRE .seq offline.',
            default_output='gre_noncart_2d.seq',
        )
    )
