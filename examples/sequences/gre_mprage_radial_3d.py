"""Standalone 3D non-Cartesian ("stack-of-stars") inversion/T2-prepared GRE.

This module implements the three mandatory module-level entry points
required by the bridge dispatcher (see ``mprage_3d.py`` for the Cartesian
inversion-prepared GRE this extends, ``gre_radial_3d.py`` for the
non-prepared stack-of-stars sibling, and ``_gre_common.py`` /
``pulserver.preparations`` for the
shared building blocks):

- ``get_default_protocol(opts)``  — return the initial CV/protocol dictionary.
- ``validate_protocol(opts, protocol)``  — validate the current protocol and
  return a ``{"valid": bool, "duration": float | None, "info": str}`` dict.
- ``make_sequence(opts, protocol, output_path)``  — synthesise the sequence
  and write it to *output_path* using :func:`pulserver.io.write`.

Structure — one magnetization-prep module gates one "segment" of spokes at
a fixed partition; the segment length is ``ETL`` (chunked if
``ETL < NumShots``), exactly mirroring ``mprage_3d.py``'s PE-line
segmentation with spoke index in place of ``ky``. Partitions are the outer
loop (accelerated via ``Rz``); spokes are the inner, per-segment loop.

``PreparationType`` (native GE enum, already closed to ``inversion``/
``t2_prep`` — no new values added, per the project's hard rule) selects the
magnetization-prep module:

- ``inversion``: non-selective hard/adiabatic 180 + spoiler + TI delay
  (``pulserver.preparations``, same as ``mprage_3d.py``).
- ``t2_prep``: composite 90-tau/2-180-tau/2-(-90) + spoiler
  (``pulserver.preparations``) — ``TE_prep`` sets the T2 weighting directly, so
  (unlike TI) there is no extra post-module centering delay; the segment
  starts right after the spoiler.

An independent, *additive* MT saturation toggle (``preparations.mt_saturation``) can be
stacked on top of either prep — real MT-prepared sequences apply the MT
pulse every TR (steady-state effect), not once per segment, so it is not a
third ``PreparationType`` option (which would also violate the closed-enum
rule: GE's real UI has no generic "prep type = MT" slot).

``TE``/``TR`` retain their per-view meaning; ``ETL``/``Trecovery`` are
reused from the MPRAGE family. TI, ``TE_prep``, the inversion/refocus-mode
toggles, and the MT enable/params all lack a native GE counterpart, so they
are ``opuser`` custom variables. Trajectory is plain full-echo radial
(rotated spokes from the public radial tilt factory — no arbitrary-gradient
waveform solver needed) — see ``gre_noncart_3d.py`` for the arbgrad-backed
spiral/rosette sibling; swapping the readout builder there for this file's
prep/segment scheduling is a mechanical follow-up, not implemented here.

Usage
-----
Register the plugin in the interpreter tree by symlinking (or copying) this
file to the ``sequences/src/`` directory and creating a numbered alias::

    cp gre_mprage_radial_3d.py pulserver-interpreter/package/pulserver/sequences/src/gre_mprage_radial_3d.py
    ln -sf src/gre_mprage_radial_3d.py pulserver-interpreter/package/pulserver/sequences/sequence12.py
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
    PreparationType,
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

NUM_ECHOES = 1
FLYBACK = True
RO_AXIS = "x"

USER_SLOT_TI = 0
USER_SLOT_INV_MODE = 1
USER_SLOT_TE_PREP = 2
USER_SLOT_REFOCUS_MODE = 3
USER_SLOT_ORDER_MODE = 4
USER_SLOT_MT_ENABLE = 5


def _order_mode_name(code: float) -> str:
    return "golden" if code >= 0.5 else "uniform"


class GreMprageRadial3DPulseqSequence(Sequence):
    """Generate a 3D stack-of-stars inversion/T2-prepared GRE, +/- MT sat."""

    def get_default_protocol(self, opts: pp.Opts) -> dict[str, dict]:
        del opts
        protocol = {
            UIParam.TE: DropdownFloatParam(
                value=3.0, min=1.5, max=80.0, incr=0.1, unit="ms",
                options=[2.0, 3.0, 5.0, 8.0, 15.0], validate=Validate.NONE,
            ),
            UIParam.TR: DropdownFloatParam(
                value=8.0, min=4.0, max=100.0, incr=0.1, unit="ms",
                options=[6.0, 8.0, 10.0, 15.0, 30.0], validate=Validate.NONE,
            ),
            UIParam.TRECOVERY: DropdownFloatParam(
                value=1200.0, min=0.0, max=5000.0, incr=1.0, unit="ms",
                options=[600.0, 1000.0, 1200.0, 1500.0, 2000.0], validate=Validate.NONE,
            ),
            UIParam.ETL: DropdownIntParam(
                value=8, min=1, max=512, incr=1, options=[8, 16, 32, 64, 128], validate=Validate.NONE,
            ),
            UIParam.FLIP: DropdownFloatParam(
                value=8.0, min=1.0, max=90.0, incr=1.0, unit="deg",
                options=[4.0, 8.0, 12.0, 20.0, 30.0], validate=Validate.NONE,
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
                value=32, min=8, max=2048, incr=1, options=[16, 32, 64, 128, 256], validate=Validate.NONE,
            ),
            UIParam.RZ: TypeinFloatParam(value=1.0, min=1.0, max=8.0, incr=1.0, unit="", validate=Validate.NONE),
            UIParam.BANDWIDTH: TypeinFloatParam(
                value=125_000.0, min=5_000.0, max=500_000.0, incr=100.0,
                unit="Hz/px", validate=Validate.NONE,
            ),
            UIParam.SEQUENCE_TYPE: make_enum_param(UIParam.SEQUENCE_TYPE, SequenceType.GRADIENT_ECHO),
            UIParam.PREPARATION_TYPE: make_enum_param(UIParam.PREPARATION_TYPE, PreparationType.INVERSION),
            UIParam.user_name(USER_SLOT_TI): Description(text="Inversion time (TI)"),
            UIParam.user_value(USER_SLOT_TI): TypeinFloatParam(
                value=900.0, min=0.1, max=5000.0, incr=1.0, unit="ms", validate=Validate.NONE,
            ),
            UIParam.user_name(USER_SLOT_INV_MODE): Description(text="Inversion pulse (0=hard, 1=adiabatic)"),
            UIParam.user_value(USER_SLOT_INV_MODE): DropdownFloatParam(
                value=1.0, min=0.0, max=1.0, incr=1.0, unit="", options=[0.0, 1.0], validate=Validate.NONE,
            ),
            UIParam.user_name(USER_SLOT_TE_PREP): Description(text="T2-prep echo time (TE_prep)"),
            UIParam.user_value(USER_SLOT_TE_PREP): TypeinFloatParam(
                value=50.0, min=1.0, max=200.0, incr=1.0, unit="ms", validate=Validate.NONE,
            ),
            UIParam.user_name(USER_SLOT_REFOCUS_MODE): Description(text="T2-prep refocus (0=hard, 1=adiabatic)"),
            UIParam.user_value(USER_SLOT_REFOCUS_MODE): DropdownFloatParam(
                value=0.0, min=0.0, max=1.0, incr=1.0, unit="", options=[0.0, 1.0], validate=Validate.NONE,
            ),
            UIParam.user_name(USER_SLOT_ORDER_MODE): Description(text="Spoke order (0=uniform, 1=golden)"),
            UIParam.user_value(USER_SLOT_ORDER_MODE): DropdownFloatParam(
                value=1.0, min=0.0, max=1.0, incr=1.0, unit="", options=[0.0, 1.0], validate=Validate.NONE,
            ),
            UIParam.user_name(USER_SLOT_MT_ENABLE): Description(text="MT saturation (0=off, 1=on, every view)"),
            UIParam.user_value(USER_SLOT_MT_ENABLE): DropdownFloatParam(
                value=0.0, min=0.0, max=1.0, incr=1.0, unit="", options=[0.0, 1.0], validate=Validate.NONE,
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
        if cfg.etl < 1:
            return {"valid": False, "duration": None, "info": "ETL must be >= 1"}
        if cfg.num_shots < 1:
            return {"valid": False, "duration": None, "info": "NumShots must be >= 1"}
        if cfg.trecovery_s < 0.0:
            return {"valid": False, "duration": None, "info": "Trecovery must be >= 0"}
        if cfg.prep_type == PreparationType.INVERSION and cfg.ti_s <= 0.0:
            return {"valid": False, "duration": None, "info": "TI must be > 0"}
        if cfg.prep_type == PreparationType.T2_PREP and cfg.te_prep_s <= 0.0:
            return {"valid": False, "duration": None, "info": "TE_prep must be > 0"}

        timing = _compute_public(opts=opts, cfg=cfg, strict=True)
        if timing is None:
            return {
                "valid": False,
                "duration": None,
                "info": "TE, TR, TI, or TE_prep infeasible for the requested gradients/ETL",
            }

        par_loop = design.make_cartesian_sampling((cfg.nx_ro, cfg.npar), acceleration=cfg.rz)
        n_segments = len(_view_loop(cfg))
        shot_s = timing["shot_s"]
        duration_s = shot_s * n_segments * len(par_loop)
        return {"valid": True, "duration": duration_s, "info": f"TA = {duration_s:.2f} s"}

    def make_sequence(self, opts: pp.Opts, protocol: dict[str, dict], output_path: str) -> None:
        prot = dict_to_protocol(protocol)
        cfg = _read_protocol(prot)
        return _make_public_sequence(opts, cfg, output_path)


class _Config:
    __slots__ = (
        "te_s", "tr_s", "flip_deg", "fov_m", "slab_thickness_m", "slice_spacing_m",
        "nx_ro", "npar", "bandwidth_hz_px", "rz", "num_shots", "order_mode",
        "etl", "trecovery_s", "prep_type", "ti_s", "inv_mode", "te_prep_s", "refocus_mode", "mt_enable",
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
    cfg.bandwidth_hz_px = params.param_float_optional(prot, UIParam.BANDWIDTH, 125_000.0)
    cfg.rz = max(1, int(round(params.param_float_optional(prot, UIParam.RZ, 1.0))))
    cfg.num_shots = params.param_int_optional(prot, UIParam.NUM_SHOTS, 32)
    cfg.order_mode = _order_mode_name(params.user_float(prot, USER_SLOT_ORDER_MODE, 1.0))
    cfg.etl = params.param_int_optional(prot, UIParam.ETL, cfg.num_shots)
    cfg.trecovery_s = params.param_float_optional(prot, UIParam.TRECOVERY, 1200.0) * 1e-3
    prep_value = prot[str(UIParam.PREPARATION_TYPE)].value
    cfg.prep_type = PreparationType(prep_value)
    cfg.ti_s = params.user_float(prot, USER_SLOT_TI, 900.0) * 1e-3
    cfg.inv_mode = "adiabatic" if params.user_float(prot, USER_SLOT_INV_MODE, 1.0) >= 0.5 else "hard"
    cfg.te_prep_s = params.user_float(prot, USER_SLOT_TE_PREP, 50.0) * 1e-3
    cfg.refocus_mode = "adiabatic" if params.user_float(prot, USER_SLOT_REFOCUS_MODE, 0.0) >= 0.5 else "hard"
    cfg.mt_enable = params.user_float(prot, USER_SLOT_MT_ENABLE, 0.0) >= 0.5
    return cfg


def _compute_public(opts: pp.Opts, cfg: _Config, strict: bool):
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
    prep = (
        design.make_inversion_pulse(adiabatic=cfg.inv_mode == "adiabatic", system=opts)
        if cfg.prep_type == PreparationType.INVERSION
        else design.make_t2prep_pulse(
            cfg.te_prep_s,
            adiabatic=cfg.refocus_mode == "adiabatic",
            voxel_size=cfg.fov_m / cfg.nx_ro,
            system=opts,
        )
    )
    mt = (
        design.make_mt_pulse(voxel_size=cfg.fov_m / cfg.nx_ro, system=opts)
        if cfg.mt_enable
        else None
    )
    # The readout folds the rephaser into its own prewinder, so the
    # excitation must stop playing it -- otherwise the slice is rephased
    # twice and every shot ends with a net selection moment.
    pulse = pulse.without_rephasers()

    d_pulse = sum(pp.calc_duration(*block) for block in pulse)
    center = pp.calc_rf_center(pulse.rf)[0] + pulse.rf.delay
    raster = opts.block_duration_raster
    te_delay_s = round(
        (cfg.te_s - (d_pulse - center) - radial.t_prephase_s - 0.5 * radial.readout.read_duration)
        / raster
    ) * raster
    if te_delay_s < -1e-9 and strict:
        return None
    te_delay_s = max(0.0, te_delay_s)
    mt_duration = 0.0 if mt is None else sum(pp.calc_duration(*block) for block in mt)
    min_block_s = mt_duration + d_pulse + te_delay_s + radial.duration
    tr_delay_s = round((cfg.tr_s - min_block_s) / raster) * raster
    if tr_delay_s < -1e-9 and strict:
        return None
    prep_duration = sum(pp.calc_duration(*block) for block in prep)
    prep_delay_s = cfg.ti_s - prep_duration if cfg.prep_type == PreparationType.INVERSION else 0.0
    prep_delay_s = round(prep_delay_s / raster) * raster
    if prep_delay_s < -1e-9 and strict:
        return None
    shot_s = prep_duration + max(0.0, prep_delay_s) + cfg.etl * cfg.tr_s + cfg.trecovery_s
    return {
        "pulse": pulse,
        "radial": radial,
        "prep": prep,
        "mt": mt,
        "te_delay_s": te_delay_s,
        "tr_delay_s": max(0.0, tr_delay_s),
        "prep_delay_s": max(0.0, prep_delay_s),
        "min_block_s": min_block_s,
        "shot_s": shot_s,
    }


def _view_loop(cfg: _Config):
    """Radial view chronology, grouped into one shot per prepared segment."""
    return design.make_noncartesian_2d_sampling(
        (cfg.nx_ro, cfg.nx_ro),
        views=cfg.num_shots,
        scheme=cfg.order_mode,
        segment_length=cfg.etl,
    )


def _make_public_sequence(opts: pp.Opts, cfg: _Config, output_path: str) -> None:
    timing = _compute_public(opts, cfg, strict=False)
    seq = pp.Sequence(opts)
    views = _view_loop(cfg)
    segments = views.shots
    rotations = views.to_rotations()
    par_loop = design.make_cartesian_sampling((cfg.nx_ro, cfg.npar), acceleration=cfg.rz)
    phases = design.make_rf_spoiling_schedule(cfg.num_shots * len(par_loop))
    te_delay = pp.make_delay(timing["te_delay_s"]) if timing["te_delay_s"] > 0 else None
    tr_delay = pp.make_delay(timing["tr_delay_s"]) if timing["tr_delay_s"] > 0 else None
    prep_delay = pp.make_delay(timing["prep_delay_s"]) if timing["prep_delay_s"] > 0 else None
    recovery_s = round(cfg.trecovery_s / opts.block_duration_raster) * opts.block_duration_raster
    recovery = pp.make_delay(recovery_s) if recovery_s > 0 else None
    phase_idx = 0
    for par_shot in par_loop:
        for segment in segments:
            for block in timing["prep"]:
                seq.add_block(*block)
            if prep_delay is not None:
                seq.add_block(prep_delay)
            for shot in segment:
                if timing["mt"] is not None:
                    for block in timing["mt"]:
                        seq.add_block(*block)
                phase = float(phases[phase_idx])
                timing["pulse"].set_state(phase_offset_rad=phase)
                for block in timing["pulse"]:
                    seq.add_block(*block)
                if te_delay is not None:
                    seq.add_block(te_delay)
                timing["radial"].set_state(
                    lin_idx=int(shot),
                    par_idx=int(par_shot[0, 0]),
                    phase_offset_rad=phase,
                    rotation=rotations[shot],
                )
                for block in timing["radial"]:
                    seq.add_block(*block)
                if tr_delay is not None:
                    seq.add_block(tr_delay)
                phase_idx += 1
            if recovery is not None:
                seq.add_block(recovery)
    seq.set_definition("Name", "gre_mprage_radial_3d")
    seq.set_definition("FOV", [cfg.fov_m, cfg.fov_m, cfg.slice_spacing_m * cfg.npar])
    seq.set_definition("TE", cfg.te_s)
    seq.set_definition("TR", cfg.tr_s)
    seq.set_definition("PreparationType", str(cfg.prep_type))
    seq.set_definition("Trajectory", "radial")
    seq.set_definition("ImagingMode", "3d")
    seq.set_definition("RfSpoilingIncDeg", 117.0)
    pio.write(seq, output=output_path, remove_duplicates=False, check_timing=False)


PLUGIN = GreMprageRadial3DPulseqSequence()


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
    ('--trecovery-ms', UIParam.TRECOVERY, float, ""),
    ('--etl', UIParam.ETL, int, ""),
    ('--prep-type', UIParam.PREPARATION_TYPE, {'inversion': 'inversion', 't2_prep': 't2_prep'}, ""),
    ('--ti-ms', UIParam.user_value(USER_SLOT_TI), float, ""),
    ('--inversion-mode', UIParam.user_value(USER_SLOT_INV_MODE), {'hard': 0.0, 'adiabatic': 1.0}, ""),
    ('--te-prep-ms', UIParam.user_value(USER_SLOT_TE_PREP), float, ""),
    ('--refocus-mode', UIParam.user_value(USER_SLOT_REFOCUS_MODE), {'hard': 0.0, 'adiabatic': 1.0}, ""),
    ('--mt', UIParam.user_value(USER_SLOT_MT_ENABLE), ("const", 1.0), ""),
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
]

if __name__ == "__main__":
    raise SystemExit(
        run_cli(
            PLUGIN,
            sys.argv[1:],
            arg_map=_ARG_MAP,
            description='Generate a 3D stack-of-stars prepared GRE .seq offline.',
            default_output='gre_mprage_radial_3d.seq',
        )
    )
