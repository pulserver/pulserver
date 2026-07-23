"""Standalone 2D non-Cartesian (radial) inversion/T2-prepared GRE plugin.

This module implements the three mandatory module-level entry points
required by the bridge dispatcher (see ``mprage_2d.py`` for the Cartesian
inversion-prepared GRE this extends, ``gre_radial_2d.py`` for the
non-prepared radial sibling, ``gre_mprage_radial_3d.py`` for the 3D
counterpart this mirrors, and ``_gre_common.py`` / ``pulserver`` /
``pulserver.preparations`` for the shared building blocks):

- ``get_default_protocol(opts)``  — return the initial CV/protocol dictionary.
- ``validate_protocol(opts, protocol)``  — validate the current protocol and
  return a ``{"valid": bool, "duration": float | None, "info": str}`` dict.
- ``make_sequence(opts, protocol, output_path)``  — synthesise the sequence
  and write it to *output_path* using :func:`pulserver.io.write`.

Structure — one magnetization-prep module gates one "segment" of spokes at
a fixed slice; the segment length is ``ETL`` (chunked if
``ETL < NumShots``). Slices are the outer loop (one prep module each,
non-selective, so it re-preps every slice — same accepted simplification as
``mprage_2d.py``); spokes are the inner, per-segment loop, replayed with a
per-shot in-plane rotation (pulseq ``ROTATIONS`` extension) instead of a
Cartesian Y phase encode, same as ``gre_radial_2d.py``.

``PreparationType`` selects inversion vs T2-prep (see
``gre_mprage_radial_3d.py`` docstring for the full rationale, including why
MT saturation is a separate additive ``opuser`` toggle rather than a third
``PreparationType`` value).

Usage
-----
Register the plugin in the interpreter tree by symlinking (or copying) this
file to the ``sequences/src/`` directory and creating a numbered alias::

    cp gre_mprage_radial_2d.py pulserver-interpreter/package/pulserver/sequences/src/gre_mprage_radial_2d.py
    ln -sf src/gre_mprage_radial_2d.py pulserver-interpreter/package/pulserver/sequences/sequence11.py
"""

from __future__ import annotations

import sys

import numpy as np
import pulserver.io as pio
import pulserver.design as design
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


class GreMprageRadial2DPulseqSequence(Sequence):
    """Generate a 2D radial inversion/T2-prepared GRE, +/- MT sat."""

    def get_default_protocol(self, opts: pp.Opts) -> dict[str, dict]:
        del opts
        protocol = {
            UIParam.TE: DropdownFloatParam(
                value=3.0, min=1.5, max=80.0, incr=0.1, unit="ms",
                options=[2.0, 3.0, 5.0, 8.0, 15.0], validate=Validate.NONE,
            ),
            UIParam.TR: DropdownFloatParam(
                value=10.0, min=5.0, max=100.0, incr=0.1, unit="ms",
                options=[8.0, 10.0, 15.0, 20.0, 30.0], validate=Validate.NONE,
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
                value=32, min=8, max=2048, incr=1, options=[16, 32, 64, 128, 256], validate=Validate.NONE,
            ),
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
        if cfg.fov_m <= 0.0 or cfg.slice_thickness_m <= 0.0:
            return {"valid": False, "duration": None, "info": "FOV and slice thickness must be > 0"}
        if not (0.0 < cfg.flip_deg <= 180.0):
            return {"valid": False, "duration": None, "info": "Flip angle must be in (0, 180] deg"}
        if cfg.nx_ro < 1 or cfg.nslices < 1:
            return {"valid": False, "duration": None, "info": "NX and NSLICES must be >= 1"}
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

        n_segments = len(_view_loop(cfg))
        shot_s = timing["shot_s"]
        duration_s = shot_s * n_segments * cfg.nslices
        return {"valid": True, "duration": duration_s, "info": f"TA = {duration_s:.2f} s"}

    def make_sequence(self, opts: pp.Opts, protocol: dict[str, dict], output_path: str) -> None:
        prot = dict_to_protocol(protocol)
        cfg = _read_protocol(prot)
        return _make_public_sequence(opts, cfg, output_path)


class _Config:
    __slots__ = (
        "te_s", "tr_s", "flip_deg", "fov_m", "slice_thickness_m", "slice_spacing_m",
        "nx_ro", "nslices", "bandwidth_hz_px", "num_shots", "order_mode",
        "etl", "trecovery_s", "prep_type", "ti_s", "inv_mode", "te_prep_s", "refocus_mode", "mt_enable",
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
    cfg.bandwidth_hz_px = params.param_float_optional(prot, UIParam.BANDWIDTH, 125_000.0)
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
        np.deg2rad(cfg.flip_deg), cfg.slice_thickness_m, system=opts
    )
    radial = design.make_radial_readout(
        opts,
        cfg.fov_m,
        cfg.nx_ro,
        bandwidth_hz_px=cfg.bandwidth_hz_px,
        slice_rephasing=pulse.rephasers[0],
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
    phases = design.make_rf_spoiling_schedule(cfg.num_shots * cfg.nslices)
    te_delay = pp.make_delay(timing["te_delay_s"]) if timing["te_delay_s"] > 0 else None
    tr_delay = pp.make_delay(timing["tr_delay_s"]) if timing["tr_delay_s"] > 0 else None
    prep_delay = pp.make_delay(timing["prep_delay_s"]) if timing["prep_delay_s"] > 0 else None
    recovery_s = round(cfg.trecovery_s / opts.block_duration_raster) * opts.block_duration_raster
    recovery = pp.make_delay(recovery_s) if recovery_s > 0 else None
    slice_step = cfg.slice_spacing_m if cfg.nslices > 1 else 0.0
    slices = design.make_slice_loop(
        cfg.nslices, slice_step or cfg.slice_thickness_m, order="sequential"
    )
    offsets_hz = slices.to_frequencies(timing["pulse"].gradients[0].amplitude) if slice_step else None
    phase_idx = 0
    for sl, band in enumerate(slices.shots):
        offset_hz = float(offsets_hz[band[0]]) if offsets_hz is not None else 0.0
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
                timing["pulse"].set_state(freq_offset_hz=offset_hz, phase_offset_rad=phase)
                timing["pulse"].set_labels(*slices.labels(sl))
                for block in timing["pulse"]:
                    seq.add_block(*block)
                if te_delay is not None:
                    seq.add_block(te_delay)
                timing["radial"].set_state(
                    lin_idx=int(shot), phase_offset_rad=phase, rotation=rotations[shot]
                )
                for block in timing["radial"]:
                    seq.add_block(*block)
                if tr_delay is not None:
                    seq.add_block(tr_delay)
                phase_idx += 1
            if recovery is not None:
                seq.add_block(recovery)
    seq.set_definition("Name", "gre_mprage_radial_2d")
    seq.set_definition("FOV", [cfg.fov_m, cfg.fov_m, slice_step * cfg.nslices or cfg.slice_thickness_m])
    seq.set_definition("TE", cfg.te_s)
    seq.set_definition("TR", cfg.tr_s)
    seq.set_definition("PreparationType", str(cfg.prep_type))
    seq.set_definition("Trajectory", "radial")
    seq.set_definition("ImagingMode", "2d")
    seq.set_definition("RfSpoilingIncDeg", 117.0)
    pio.write(seq, output=output_path, remove_duplicates=False, check_timing=False)


PLUGIN = GreMprageRadial2DPulseqSequence()


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
    ('--slice-thickness-mm', UIParam.SLICE_THICKNESS, float, ""),
    ('--slice-spacing-mm', UIParam.SLICE_SPACING, float, ""),
    ('--nx', UIParam.NX, int, ""),
    ('--nslices', UIParam.NSLICES, int, ""),
    ('--num-shots', UIParam.NUM_SHOTS, int, ""),
    ('--bandwidth-hz-px', UIParam.BANDWIDTH, float, ""),
    ('--order-mode', UIParam.user_value(USER_SLOT_ORDER_MODE), {'uniform': 0.0, 'golden': 1.0}, ""),
]

if __name__ == "__main__":
    raise SystemExit(
        run_cli(
            PLUGIN,
            sys.argv[1:],
            arg_map=_ARG_MAP,
            description='Generate a 2D radial prepared GRE .seq offline.',
            default_output='gre_mprage_radial_2d.seq',
        )
    )
