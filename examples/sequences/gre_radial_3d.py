"""Standalone 3D radial ("stack-of-stars") non-Cartesian GRE plugin.

This module implements the three mandatory module-level entry points
required by the bridge dispatcher (see ``gre_3d.py`` for the Cartesian 3D
GRE this parallels, ``gre_radial_2d.py`` for the 2D radial sibling, and
``_gre_common.py`` for the shared readout/echo-train and Z-channel
gradient-combination helpers reused here unmodified):

- ``get_default_protocol(opts)``  — return the initial CV/protocol dictionary.
- ``validate_protocol(opts, protocol)``  — validate the current protocol and
  return a ``{"valid": bool, "duration": float | None, "info": str}`` dict.
- ``make_sequence(opts, protocol, output_path)``  — synthesise the sequence
  and write it to *output_path* using :func:`pulserver.io.write`.

Stack-of-stars: the same in-plane full-echo radial spoke (logical X axis,
designed once via ``_gre_common``) is rotated per shot about the physical Z
axis (``ROTATIONS`` extension, ``pulserver.pypulseq.make_rotation`` — no
arbitrary-trajectory solver needed for straight spokes) while the
partition (kz) dimension is phase-encoded with a conventional Z-channel
trapezoid, exactly as in ``gre_3d.py``. A rotation purely about Z leaves the
Z gradient channel untouched, so the two compose in the same block without
interaction. Spokes are the outer loop (one ``TR`` block covers a full
partition sweep per spoke); partitions are the inner loop (accelerated via
``Rz``).

``NumShots`` (``IntKey.NUM_SHOTS``, a real native GE variable) is the number
of radial spokes. The spoke-ordering mode has no native GE counterpart, so
it is carried as an ``opuser`` custom variable.

Usage
-----
Register the plugin in the interpreter tree by symlinking (or copying) this
file to the ``sequences/src/`` directory and creating a numbered alias::

    cp gre_radial_3d.py pulserver-interpreter/package/pulserver/sequences/src/gre_radial_3d.py
    ln -sf src/gre_radial_3d.py pulserver-interpreter/package/pulserver/sequences/sequence8.py
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


class GreRadial3DPulseqSequence(Sequence):
    """Generate a 3D stack-of-stars (radial in-plane, Cartesian partition) GRE."""

    def get_default_protocol(self, opts: pp.Opts) -> dict[str, dict]:
        del opts
        protocol = {
            UIParam.TE: DropdownFloatParam(
                value=3.0, min=1.5, max=80.0, incr=0.1, unit="ms",
                options=[2.0, 3.0, 5.0, 8.0, 15.0], validate=Validate.NONE,
            ),
            UIParam.TR: DropdownFloatParam(
                value=60.0, min=4.0, max=300.0, incr=0.1, unit="ms",
                options=[15.0, 30.0, 60.0, 100.0, 150.0], validate=Validate.NONE,
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
                value=100, min=8, max=2048, incr=1, options=[50, 100, 200, 400, 800], validate=Validate.NONE,
            ),
            UIParam.NUM_FRAMES: DropdownIntParam(
                value=1, min=1, max=10000, incr=1, options=[1, 10, 20, 50], validate=Validate.NONE,
            ),
            UIParam.TRIGGER_TYPE: make_enum_param(UIParam.TRIGGER_TYPE, TriggerType.NONE),
            UIParam.RZ: TypeinFloatParam(value=1.0, min=1.0, max=8.0, incr=1.0, unit="", validate=Validate.NONE),
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
        if cfg.fov_m <= 0.0 or cfg.slab_thickness_m <= 0.0:
            return {"valid": False, "duration": None, "info": "FOV and slab thickness must be > 0"}
        if not (0.0 < cfg.flip_deg <= 180.0):
            return {"valid": False, "duration": None, "info": "Flip angle must be in (0, 180] deg"}
        if cfg.nx_ro < 1 or cfg.npar < 1:
            return {"valid": False, "duration": None, "info": "NX and NSLICES (partitions) must be >= 1"}
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

        rotations = design.make_noncartesian_2d_sampling(
            (cfg.nx_ro, cfg.nx_ro), views=cfg.num_shots, scheme=cfg.order_mode
        ).to_rotations()
        par_loop = design.make_cartesian_sampling((cfg.nx_ro, cfg.npar), acceleration=cfg.rz)
        rf_phases = design.make_rf_spoiling_schedule(cfg.num_shots * len(par_loop) * cfg.num_frames)
        shot = 0

        frames = design.make_counter_loop(cfg.num_frames, label="PHS")
        for frame in range(len(frames)):
            (phase_label,) = frames.labels(frame)
            if cfg.trigger != TriggerType.NONE:
                seq.add_block(pp.make_trigger(cfg.trigger, duration=1e-3, system=opts), phase_label)
            else:
                seq.add_block(phase_label)
            for spoke, rotation in enumerate(rotations):
                for par_shot in par_loop:
                    phase = float(rf_phases[shot])
                    pulse.set_state(phase_offset_rad=phase)
                    for block_idx, block in enumerate(pulse):
                        labels = (pp.make_label(type="SET", label="SLC", value=0),) if block_idx == 0 else ()
                        seq.add_block(*block, *labels)
                    if te_delay is not None:
                        seq.add_block(te_delay)
                    radial.set_state(
                        lin_idx=spoke,
                        par_idx=int(par_shot[0, 0]),
                        phase_offset_rad=phase,
                        rotation=rotation,
                    )
                    for block in radial:
                        seq.add_block(*block)
                    shot += 1

                if tr_delay is not None:
                    seq.add_block(tr_delay)

        seq.set_definition("Name", "gre_radial_3d")
        seq.set_definition("FOV", [cfg.fov_m, cfg.fov_m, cfg.slice_spacing_m * cfg.npar])
        seq.set_definition("TE", cfg.te_s)
        seq.set_definition("TR", cfg.tr_s)
        seq.set_definition("Flip", cfg.flip_deg)
        seq.set_definition("ImagingMode", "3d")
        seq.set_definition("Trajectory", "radial")
        seq.set_definition("SpokeOrder", cfg.order_mode)
        seq.set_definition("BandwidthHzPerPx", cfg.bandwidth_hz_px)
        seq.set_definition("Rz", cfg.rz)
        seq.set_definition("RfSpoilingIncDeg", 117.0)
        seq.set_definition("Nx", cfg.nx_ro)
        seq.set_definition("NumShots", cfg.num_shots)
        seq.set_definition("NumPartitions", cfg.npar)
        seq.set_definition("NumFrames", cfg.num_frames)
        seq.set_definition("TriggerType", str(cfg.trigger))
        pio.write(seq, output=output_path, remove_duplicates=False, check_timing=False)


class _Config:
    __slots__ = (
        "bandwidth_hz_px",
        "flip_deg",
        "fov_m",
        "npar",
        "num_frames",
        "num_shots",
        "nx_ro",
        "order_mode",
        "rz",
        "slab_thickness_m",
        "slice_spacing_m",
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
    cfg.slab_thickness_m = params.param_float(prot, UIParam.SLICE_THICKNESS) * 1e-3
    cfg.slice_spacing_m = params.param_float(prot, UIParam.SLICE_SPACING) * 1e-3
    cfg.nx_ro = params.param_int(prot, UIParam.NX)
    cfg.npar = params.param_int(prot, UIParam.NSLICES)
    cfg.bandwidth_hz_px = params.param_float_optional(prot, UIParam.BANDWIDTH, DEFAULT_BANDWIDTH_HZ_PX)
    cfg.rz = max(1, round(params.param_float_optional(prot, UIParam.RZ, 1.0)))
    cfg.num_shots = params.param_int_optional(prot, UIParam.NUM_SHOTS, 100)
    cfg.order_mode = _order_mode_name(params.user_float(prot, USER_SLOT_ORDER_MODE, 1.0))
    cfg.num_frames = params.param_int_optional(prot, UIParam.NUM_FRAMES, 1)
    cfg.trigger = prot[str(UIParam.TRIGGER_TYPE)].value
    return cfg


def _compute_timing(opts: pp.Opts, cfg: _Config, strict: bool, n_inner: int | None = None):
    if n_inner is None:
        n_inner = len(design.make_cartesian_sampling((cfg.nx_ro, cfg.npar), acceleration=cfg.rz))
    pulse = design.make_slice_selective_pulse(
        np.deg2rad(cfg.flip_deg), cfg.slab_thickness_m, system=opts
    )
    radial = design.make_radial_stack_readout(
        opts,
        cfg.fov_m,
        cfg.nx_ro,
        cfg.slice_spacing_m * cfg.npar,
        cfg.npar,
        bandwidth_hz_px=cfg.bandwidth_hz_px,
        slice_rephasing=pulse.rephasers,
    )
    # The readout folds the rephaser into its own prewinder, so the
    # excitation must stop playing it -- otherwise the slice is rephased
    # twice and every shot ends with a net selection moment.
    pulse = pulse.without_rephasers()

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


PLUGIN = GreRadial3DPulseqSequence()


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
    ('--slab-thickness-mm', UIParam.SLICE_THICKNESS, float, ""),
    ('--partition-spacing-mm', UIParam.SLICE_SPACING, float, ""),
    ('--nx', UIParam.NX, int, ""),
    ('--npartitions', UIParam.NSLICES, int, ""),
    ('--num-shots', UIParam.NUM_SHOTS, int, ""),
    ('--bandwidth-hz-px', UIParam.BANDWIDTH, float, ""),
    ('--rz', UIParam.RZ, float, ""),
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
            description='Generate a 3D stack-of-stars radial GRE .seq offline.',
            default_output='gre_radial_3d.seq',
        )
    )
