"""Standalone 3D inversion-prepared (MPRAGE) sequence plugin for pulserver.

This module implements the three mandatory module-level entry points
required by the bridge dispatcher (see ``gre_3d.py`` for the non-prepared 3D
GRE this extends, and ``_gre_common.py``/``pulserver`` for the shared
low-level building blocks):

- ``get_default_protocol(opts)``  — return the initial CV/protocol dictionary.
- ``validate_protocol(opts, protocol)``  — validate the current protocol and
  return a ``{"valid": bool, "duration": float | None, "info": str}`` dict.
- ``make_sequence(opts, protocol, output_path)``  — synthesise the sequence
  and write it to *output_path* using :func:`pulserver.io.write`.

Sequence structure — one inversion pulse gates one "segment" of PE1 (``NY``)
lines at a fixed partition; the segment length is ``ETL`` (chunked if
``ETL < NY``). Partitions are the outer loop (accelerated via ``Rz``); PE1 is
the inner, per-segment loop (accelerated via ``Ry``):

    for partition in partitions:                     # one inversion each
        non-selective inversion pulse (hard or adiabatic hypsec)
        3-axis non-selective spoiler
        TI delay (measured from the inversion RF center to the temporal
            center of the following segment — see ``pulserver.preparations``)
        for ky_chunk in chunks(PE1 lines, ETL):       # one shot
            for ky in ky_chunk:
                single-echo GRE view (RF spoiled, shares readout/echo-train
                and Z-channel gradient-combination helpers with gre_3d.py)
            recovery delay (``Trecovery``) before the next shot/inversion

``TE``/``TR`` retain the same per-view meaning as in ``gre_3d.py`` (``TR`` is
the inner, per-view period — the demo's "TRinner"). ``ETL`` (``IntKey.ETL``,
a real native GE variable already unused elsewhere) is reused here for
"views per inversion shot". ``Trecovery`` (``FloatKey.TRECOVERY``, native GE
``optrecovery``/``pitrecovery*``) is the post-segment recovery delay. TI and
the inversion-mode toggle (hard vs adiabatic) have no native GE ``pi*``
counterpart, so both are carried as ``opuser`` custom variables.

Usage
-----
Register the plugin in the interpreter tree by symlinking (or copying) this
file to the ``sequences/src/`` directory and creating a numbered alias::

    cp mprage_3d.py pulserver-interpreter/package/pulserver/sequences/src/mprage_3d.py
    ln -sf src/mprage_3d.py pulserver-interpreter/package/pulserver/sequences/sequence6.py
"""

from __future__ import annotations

import sys

import numpy as np
import pulserver.io as pio
import pulserver.design as design
import pulserver.pypulseq as pp
from pulserver import (
    BoolParam,
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

NUM_ECHOES = 1
FLYBACK = True

USER_SLOT_TI = 0
USER_SLOT_INV_MODE = 1
USER_SLOT_SPSP = 2
USER_SLOT_SPSP_BW = 3


class Mprage3DPulseqSequence(Sequence):
    """Generate a 3D inversion-prepared (MPRAGE) Cartesian GRE sequence."""

    def get_default_protocol(self, opts: pp.Opts) -> dict[str, dict]:
        del opts
        protocol = {
            UIParam.TE: DropdownFloatParam(
                value=3.0, min=1.5, max=80.0, incr=0.1, unit="ms",
                options=[2.0, 3.0, 4.0, 8.0, 15.0], validate=Validate.NONE,
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
            UIParam.PHASE_FOV: DropdownFloatParam(
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
            UIParam.NY: DropdownIntParam(
                value=64, min=8, max=512, incr=1, options=[64, 128, 192, 256, 384], validate=Validate.NONE,
            ),
            UIParam.NSLICES: DropdownIntParam(
                value=8, min=1, max=256, incr=1, options=[8, 16, 32, 64, 128], validate=Validate.NONE,
            ),
            UIParam.RY: TypeinFloatParam(value=1.0, min=1.0, max=8.0, incr=1.0, unit="", validate=Validate.NONE),
            UIParam.RZ: TypeinFloatParam(value=1.0, min=1.0, max=8.0, incr=1.0, unit="", validate=Validate.NONE),
            UIParam.BANDWIDTH: TypeinFloatParam(
                value=125_000.0, min=5_000.0, max=500_000.0, incr=100.0,
                unit="Hz/px", validate=Validate.NONE,
            ),
            UIParam.SWAP_PHASE_FREQ: BoolParam(value=False, validate=Validate.NONE),
            UIParam.SEQUENCE_TYPE: make_enum_param(UIParam.SEQUENCE_TYPE, SequenceType.GRADIENT_ECHO),
            UIParam.user_name(USER_SLOT_TI): Description(text="Inversion time (TI)"),
            UIParam.user_value(USER_SLOT_TI): TypeinFloatParam(
                value=900.0, min=0.1, max=5000.0, incr=1.0, unit="ms", validate=Validate.NONE,
            ),
            UIParam.user_name(USER_SLOT_INV_MODE): Description(text="Inversion pulse (0=hard, 1=adiabatic)"),
            UIParam.user_value(USER_SLOT_INV_MODE): DropdownFloatParam(
                value=1.0, min=0.0, max=1.0, incr=1.0, unit="", options=[0.0, 1.0], validate=Validate.NONE,
            ),
            UIParam.user_name(USER_SLOT_SPSP): Description(text="Excitation (0=slab selective, 1=SPSP water selective)"),
            UIParam.user_value(USER_SLOT_SPSP): DropdownFloatParam(
                value=0.0, min=0.0, max=1.0, incr=1.0, unit="", options=[0.0, 1.0], validate=Validate.NONE,
            ),
            UIParam.user_name(USER_SLOT_SPSP_BW): Description(text="SPSP spectral bandwidth"),
            UIParam.user_value(USER_SLOT_SPSP_BW): TypeinFloatParam(
                value=250.0, min=50.0, max=1000.0, incr=10.0, unit="Hz", validate=Validate.NONE,
            ),
        }
        return protocol_to_dict(protocol)

    def validate_protocol(self, opts: pp.Opts, protocol: dict[str, dict]) -> dict:
        prot = dict_to_protocol(protocol)
        cfg = _read_protocol(prot)

        if cfg.te_s <= 0.0 or cfg.tr_s <= 0.0:
            return {"valid": False, "duration": None, "info": "TE and TR must be > 0"}
        if cfg.fov_ro_m <= 0.0 or cfg.fov_pe_m <= 0.0 or cfg.slab_thickness_m <= 0.0:
            return {"valid": False, "duration": None, "info": "FOV and slab thickness must be > 0"}
        if not (0.0 < cfg.flip_deg <= 180.0):
            return {"valid": False, "duration": None, "info": "Flip angle must be in (0, 180] deg"}
        if cfg.nx_ro < 1 or cfg.ny_pe < 1 or cfg.npar < 1:
            return {"valid": False, "duration": None, "info": "NX, NY, and NSLICES (partitions) must be >= 1"}
        if cfg.bandwidth_hz_px <= 0.0:
            return {"valid": False, "duration": None, "info": "Bandwidth must be > 0"}
        if cfg.spsp and cfg.spsp_bandwidth_hz <= 0.0:
            return {"valid": False, "duration": None, "info": "SPSP bandwidth must be > 0"}
        if cfg.etl < 1:
            return {"valid": False, "duration": None, "info": "ETL must be >= 1"}
        if cfg.ti_s <= 0.0:
            return {"valid": False, "duration": None, "info": "TI must be > 0"}
        if cfg.trecovery_s < 0.0:
            return {"valid": False, "duration": None, "info": "Trecovery must be >= 0"}

        timing = _compute_public(opts=opts, cfg=cfg, strict=True)
        if timing is None:
            return {
                "valid": False,
                "duration": None,
                "info": "TE, TR, or TI infeasible for the requested gradients/ETL",
            }

        shot_s = timing["shot_s"]
        duration_s = shot_s * len(_segment_loop(cfg))
        return {"valid": True, "duration": duration_s, "info": f"TA = {duration_s:.2f} s"}

    def make_sequence(self, opts: pp.Opts, protocol: dict[str, dict], output_path: str) -> None:
        prot = dict_to_protocol(protocol)
        cfg = _read_protocol(prot)
        return _make_public_sequence(opts, cfg, output_path)


class _Config:
    __slots__ = (
        "te_s", "tr_s", "flip_deg", "fov_ro_m", "fov_pe_m", "slab_thickness_m", "slice_spacing_m",
        "nx_ro", "ny_pe", "npar", "bandwidth_hz_px", "ry", "rz", "ro_axis", "pe_axis",
        "etl", "ti_s", "trecovery_s", "inv_mode", "spsp", "spsp_bandwidth_hz",
    )


def _read_protocol(prot: dict) -> _Config:
    cfg = _Config()
    cfg.te_s = params.param_float(prot, UIParam.TE) * 1e-3
    cfg.tr_s = params.param_float(prot, UIParam.TR) * 1e-3
    cfg.flip_deg = params.param_float(prot, UIParam.FLIP)
    cfg.fov_ro_m = params.param_float(prot, UIParam.FOV) * 1e-3
    cfg.fov_pe_m = params.phase_fov_mm_from_protocol(prot) * 1e-3
    cfg.slab_thickness_m = params.param_float(prot, UIParam.SLICE_THICKNESS) * 1e-3
    cfg.slice_spacing_m = params.param_float(prot, UIParam.SLICE_SPACING) * 1e-3
    cfg.nx_ro = params.param_int(prot, UIParam.NX)
    cfg.ny_pe = params.param_int(prot, UIParam.NY)
    cfg.npar = params.param_int(prot, UIParam.NSLICES)
    cfg.bandwidth_hz_px = params.param_float_optional(prot, UIParam.BANDWIDTH, 125_000.0)
    cfg.ry = max(1, int(round(params.param_float_optional(prot, UIParam.RY, 1.0))))
    cfg.rz = max(1, int(round(params.param_float_optional(prot, UIParam.RZ, 1.0))))
    cfg.ro_axis, cfg.pe_axis = params.resolve_readout_phase_axes(prot)
    cfg.etl = params.param_int_optional(prot, UIParam.ETL, cfg.ny_pe)
    cfg.ti_s = params.user_float(prot, USER_SLOT_TI, 900.0) * 1e-3
    cfg.trecovery_s = params.param_float_optional(prot, UIParam.TRECOVERY, 1200.0) * 1e-3
    cfg.inv_mode = "adiabatic" if params.user_float(prot, USER_SLOT_INV_MODE, 1.0) >= 0.5 else "hard"
    cfg.spsp = params.user_float(prot, USER_SLOT_SPSP, 0.0) >= 0.5
    cfg.spsp_bandwidth_hz = params.user_float(prot, USER_SLOT_SPSP_BW, 250.0)
    return cfg


def _compute_public(opts: pp.Opts, cfg: _Config, strict: bool):
    inversion = design.make_inversion_pulse(adiabatic=cfg.inv_mode == "adiabatic", system=opts)
    pulse = (
        design.make_spsp_pulse(
            np.deg2rad(cfg.flip_deg), cfg.slab_thickness_m,
            cfg.spsp_bandwidth_hz, system=opts,
        )
        if cfg.spsp
        else design.make_slice_selective_pulse(
            np.deg2rad(cfg.flip_deg), cfg.slab_thickness_m, system=opts
        )
    )
    line = design.make_line_readout(
        opts,
        (cfg.fov_ro_m, cfg.fov_pe_m, cfg.slice_spacing_m * cfg.npar),
        (cfg.nx_ro, cfg.ny_pe, cfg.npar),
        bandwidth_hz_px=cfg.bandwidth_hz_px,
        spoil_position="post",
        spoil_cycles=1.0,
    )
    d_pulse = sum(pp.calc_duration(*block) for block in pulse)
    rf_center = pp.calc_rf_center(pulse.rf)[0] + pulse.rf.delay
    raster = opts.block_duration_raster
    te_delay_s = round((cfg.te_s - (d_pulse - rf_center) - line.t_first_echo_s) / raster) * raster
    if te_delay_s < -1e-9 and strict:
        return None
    te_delay_s = max(0.0, te_delay_s)
    min_block_s = d_pulse + te_delay_s + line.duration
    tr_delay_s = round((cfg.tr_s - min_block_s) / raster) * raster
    if tr_delay_s < -1e-9 and strict:
        return None
    inversion_duration = sum(pp.calc_duration(*block) for block in inversion)
    ti_delay_s = round((cfg.ti_s - inversion_duration) / raster) * raster
    if ti_delay_s < -1e-9 and strict:
        return None
    ti_delay_s = max(0.0, ti_delay_s)
    shot_s = inversion_duration + ti_delay_s + cfg.etl * cfg.tr_s + cfg.trecovery_s
    return {
        "inversion": inversion,
        "pulse": pulse,
        "line": line,
        "te_delay_s": te_delay_s,
        "tr_delay_s": max(0.0, tr_delay_s),
        "ti_delay_s": ti_delay_s,
        "min_block_s": min_block_s,
        "shot_s": shot_s,
    }


def _segment_loop(cfg: _Config):
    """(ky, kz) loop chunked into inversion segments — one shot per segment."""
    return design.make_cartesian_sampling(
        (cfg.nx_ro, cfg.ny_pe, cfg.npar),
        acceleration=(cfg.ry, cfg.rz),
        train_length=cfg.etl,
    )


def _make_public_sequence(opts: pp.Opts, cfg: _Config, output_path: str) -> None:
    timing = _compute_public(opts, cfg, strict=False)
    seq = pp.Sequence(opts)
    segments = _segment_loop(cfg)
    phases = design.make_rf_spoiling_schedule(segments.n_positions)
    te_delay = pp.make_delay(timing["te_delay_s"]) if timing["te_delay_s"] > 0 else None
    tr_delay = pp.make_delay(timing["tr_delay_s"]) if timing["tr_delay_s"] > 0 else None
    ti_delay = pp.make_delay(timing["ti_delay_s"]) if timing["ti_delay_s"] > 0 else None
    recovery_s = round(cfg.trecovery_s / opts.block_duration_raster) * opts.block_duration_raster
    recovery = pp.make_delay(recovery_s) if recovery_s > 0 else None
    phase_idx = 0
    for segment in segments:
        for block in timing["inversion"]:
            seq.add_block(*block)
        if ti_delay is not None:
            seq.add_block(ti_delay)
        for ky, par in segment:
            phase = float(phases[phase_idx])
            timing["pulse"].set_state(phase_offset_rad=phase)
            for block in timing["pulse"]:
                seq.add_block(*block)
            if te_delay is not None:
                seq.add_block(te_delay)
            timing["line"].set_state(
                lin_idx=int(ky), par_idx=int(par), phase_offset_rad=phase
            )
            for block in timing["line"]:
                seq.add_block(*block)
            if tr_delay is not None:
                seq.add_block(tr_delay)
            phase_idx += 1
        if recovery is not None:
            seq.add_block(recovery)
    seq.set_definition("Name", "mprage_3d")
    seq.set_definition("FOV", [cfg.fov_ro_m, cfg.fov_pe_m, cfg.slice_spacing_m * cfg.npar])
    seq.set_definition("TE", cfg.te_s)
    seq.set_definition("TR", cfg.tr_s)
    seq.set_definition("TI", cfg.ti_s)
    seq.set_definition("Trecovery", cfg.trecovery_s)
    seq.set_definition("ETL", cfg.etl)
    seq.set_definition("InversionMode", cfg.inv_mode)
    seq.set_definition("ImagingMode", "3d")
    seq.set_definition("ReadoutAxis", "x")
    seq.set_definition("PhaseAxis", "y")
    seq.set_definition("RfSpoilingIncDeg", 117.0)
    seq.set_definition("SPSPExcitation", cfg.spsp)
    pio.write(seq, output=output_path, remove_duplicates=False, check_timing=False)


PLUGIN = Mprage3DPulseqSequence()


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
    ('--ti-ms', UIParam.user_value(USER_SLOT_TI), float, ""),
    ('--trecovery-ms', UIParam.TRECOVERY, float, ""),
    ('--etl', UIParam.ETL, int, ""),
    ('--inversion-mode', UIParam.user_value(USER_SLOT_INV_MODE), {'hard': 0.0, 'adiabatic': 1.0}, ""),
    ('--flip-deg', UIParam.FLIP, float, ""),
    ('--fov-mm', UIParam.FOV, float, ""),
    ('--phase-fov-mm', UIParam.PHASE_FOV, float, ""),
    ('--slab-thickness-mm', UIParam.SLICE_THICKNESS, float, ""),
    ('--partition-spacing-mm', UIParam.SLICE_SPACING, float, ""),
    ('--nx', UIParam.NX, int, ""),
    ('--ny', UIParam.NY, int, ""),
    ('--npartitions', UIParam.NSLICES, int, ""),
    ('--bandwidth-hz-px', UIParam.BANDWIDTH, float, ""),
    ('--ry', UIParam.RY, float, ""),
    ('--rz', UIParam.RZ, float, ""),
    ('--swap-phase-freq', UIParam.SWAP_PHASE_FREQ, ("const", True), ""),
    ('--spsp', UIParam.user_value(USER_SLOT_SPSP), ("const", 1.0), ""),
    ('--spsp-bandwidth-hz', UIParam.user_value(USER_SLOT_SPSP_BW), float, ""),
]

if __name__ == "__main__":
    raise SystemExit(
        run_cli(
            PLUGIN,
            sys.argv[1:],
            arg_map=_ARG_MAP,
            description='Generate a 3D inversion-prepared (MPRAGE) .seq offline.',
            default_output='mprage_3d.seq',
        )
    )
