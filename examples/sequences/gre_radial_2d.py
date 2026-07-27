"""Standalone 2D radial (non-Cartesian) GRE sequence plugin for pulserver.

This module implements the three mandatory module-level entry points
required by the bridge dispatcher (see ``gre_2d.py`` for the Cartesian 2D GRE
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
``ROTATIONS`` extension, ``pulserver.pypulseq.make_rotation`` — no separate
gradient waveform per spoke. The public ``make_radial_tilt`` factory supplies
uniform or golden-angle rotations, while ``make_radial_readout`` owns the
canonical slew-limited spoke.

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

import sys

import numpy as np
import pulserver.design as design
import pulserver.io as pio
import pulserver.pypulseq as pp
from pulserver import (
    Description,
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

DEFAULT_BANDWIDTH_HZ_PX = 125_000.0

USER_SLOT_ORDER_MODE = 0


def _order_mode_name(code: float) -> str:
    return "golden" if code >= 0.5 else "uniform"


class GreRadial2DPulseqSequence(Sequence):
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
            UIParam.NUM_FRAMES: DropdownIntParam(
                value=1, min=1, max=10000, incr=1, options=[1, 10, 20, 50], validate=Validate.NONE,
            ),
            UIParam.TRIGGER_TYPE: make_enum_param(UIParam.TRIGGER_TYPE, TriggerType.NONE),
            UIParam.BANDWIDTH: TypeinFloatParam(
                value=DEFAULT_BANDWIDTH_HZ_PX, min=5_000.0, max=500_000.0, incr=100.0,
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
        if cfg.num_frames < 1:
            return {"valid": False, "duration": None, "info": "NUM_FRAMES must be >= 1"}

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

        duration_s = cfg.tr_s * float(cfg.num_shots) * cfg.num_frames
        if cfg.trigger != TriggerType.NONE:
            duration_s += 1e-3 * cfg.num_frames
        return {"valid": True, "duration": duration_s, "info": f"TA = {duration_s:.2f} s"}

    def make_sequence(self, opts: pp.Opts, protocol: dict[str, dict], output_path: str) -> None:
        prot = dict_to_protocol(protocol)
        cfg = _read_protocol(prot)

        timing = _compute_timing(opts=opts, cfg=cfg, strict=False)

        pulse = timing["pulse"]
        radial = timing["radial"]
        te_delay_s = timing["te_delay_s"]
        tr_delay_s = timing["tr_delay_s"]

        te_delay = pp.make_delay(te_delay_s) if te_delay_s > 0.0 else None
        tr_delay = pp.make_delay(tr_delay_s) if tr_delay_s > 0.0 else None

        seq = pp.Sequence(opts)

        tilts = design.make_noncartesian_2d_sampling(
            (cfg.nx_ro, cfg.nx_ro), views=cfg.num_shots, scheme=cfg.order_mode
        )
        rotations = tilts.to_rotations()
        slice_step_m = cfg.slice_spacing_m if cfg.nslices > 1 else 0.0
        slices = design.make_slice_loop(
            cfg.nslices, slice_step_m or cfg.slice_thickness_m, order="sequential"
        )
        offsets_hz = slices.to_frequencies(pulse.gradients[0].amplitude) if slice_step_m else None
        frames = design.make_counter_loop(cfg.num_frames, label="PHS")
        rf_phases = design.make_rf_spoiling_schedule(cfg.num_shots * len(slices) * len(frames))
        phase_idx = 0

        # The frame counter rides on the first block of every TR rather than on
        # a lead-in block of its own. A block that plays once per frame has no
        # counterpart in the TRs that follow it, so TR-period detection finds no
        # period at all and falls back to "the whole frame is one TR" -- which
        # then costs a waveform buffer proportional to the frame duration.
        # Re-SETting a counter that already holds the right value is free.
        for frame in range(len(frames)):
            (phase_label,) = frames.labels(frame)
            frame_labels = (phase_label,)
            if cfg.trigger != TriggerType.NONE:
                # A volume-start trigger genuinely is a once-per-frame block, so
                # mark it as one: ONCE=1 opens the section and the ONCE=0 that
                # every TR carries closes it, leaving the TRs identical.
                seq.add_block(
                    pp.make_trigger(cfg.trigger, duration=1e-3, system=opts),
                    pp.make_label(type="SET", label="ONCE", value=1),
                )
                frame_labels = (phase_label, pp.make_label(type="SET", label="ONCE", value=0))
            for spoke, rotation in enumerate(rotations):
                for sl, band in enumerate(slices.shots):
                    offset_hz = float(offsets_hz[band[0]]) if offsets_hz is not None else 0.0
                    phase = float(rf_phases[phase_idx])
                    pulse.set_state(freq_offset_hz=offset_hz, phase_offset_rad=phase)
                    pulse.set_labels(*slices.labels(sl), *frame_labels)
                    for block in pulse:
                        seq.add_block(*block)
                    if te_delay is not None:
                        seq.add_block(te_delay)
                    radial.set_state(lin_idx=spoke, phase_offset_rad=phase, rotation=rotation)
                    for block in radial:
                        seq.add_block(*block)
                    phase_idx += 1

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
        seq.set_definition("RfSpoilingIncDeg", 117.0)
        seq.set_definition("Nx", cfg.nx_ro)
        seq.set_definition("NumShots", cfg.num_shots)
        seq.set_definition("NumSlices", cfg.nslices)
        seq.set_definition("NumFrames", cfg.num_frames)
        seq.set_definition("TriggerType", str(cfg.trigger))
        pio.write(seq, output=output_path, check_timing=False)


class _Config:
    __slots__ = (
        "bandwidth_hz_px",
        "flip_deg",
        "fov_m",
        "nslices",
        "num_frames",
        "num_shots",
        "nx_ro",
        "order_mode",
        "slice_spacing_m",
        "slice_thickness_m",
        "te_s",
        "tr_s",
        "trigger",
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
    cfg.bandwidth_hz_px = params.param_float_optional(prot, UIParam.BANDWIDTH, DEFAULT_BANDWIDTH_HZ_PX)
    cfg.num_shots = params.param_int_optional(prot, UIParam.NUM_SHOTS, 100)
    cfg.order_mode = _order_mode_name(params.user_float(prot, USER_SLOT_ORDER_MODE, 1.0))
    cfg.num_frames = params.param_int_optional(prot, UIParam.NUM_FRAMES, 1)
    cfg.trigger = prot[str(UIParam.TRIGGER_TYPE)].value
    return cfg


def _compute_timing(opts: pp.Opts, cfg: _Config, strict: bool, n_inner: int | None = None):
    if n_inner is None:
        n_inner = cfg.nslices
    pulse = design.make_slice_selective_pulse(
        np.deg2rad(cfg.flip_deg), cfg.slice_thickness_m, system=opts
    )
    radial = design.make_radial_readout(
        opts,
        cfg.fov_m,
        cfg.nx_ro,
        bandwidth_hz_px=cfg.bandwidth_hz_px,
        slice_rephasing=pulse.rephasers[0],
    )
    d_pulse = sum(pp.calc_duration(*block) for block in pulse)
    rf_center_s = pp.calc_rf_center(pulse.rf)[0] + pulse.rf.delay
    min_te_s = (d_pulse - rf_center_s) + radial.t_prephase_s + 0.5 * radial.readout.read_duration
    raster = opts.block_duration_raster
    te_delay_s = round((cfg.te_s - min_te_s) / raster) * raster
    if te_delay_s < -1e-9 and strict:
        return None
    if te_delay_s < 0.0:
        te_delay_s = 0.0

    min_block_s = d_pulse + te_delay_s + radial.duration
    tr_delay_s = round((cfg.tr_s - n_inner * min_block_s) / raster) * raster
    if tr_delay_s < -1e-9 and strict:
        return None
    if tr_delay_s < 0.0:
        tr_delay_s = 0.0

    return {
        "pulse": pulse,
        "radial": radial,
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
    ('--bandwidth-hz-px', UIParam.BANDWIDTH, float, ""),
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
            description='Generate a 2D full-echo radial GRE .seq offline.',
            default_output='gre_radial_2d.seq',
        )
    )
