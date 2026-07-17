"""Standalone 2D inversion-prepared (MPRAGE) sequence plugin for pulserver.

This module implements the three mandatory module-level entry points
required by the bridge dispatcher (see ``gre_multiecho_2d.py`` for the
non-prepared 2D GRE this extends, and ``_gre_common.py``/``_prep_common.py``
for the shared low-level building blocks):

- ``get_default_protocol(opts)``  — return the initial CV/protocol dictionary.
- ``validate_protocol(opts, protocol)``  — validate the current protocol and
  return a ``{"valid": bool, "duration": float | None, "info": str}`` dict.
- ``make_sequence(opts, protocol, output_path)``  — synthesise the sequence
  and write it to *output_path* using :func:`pulserver.io.write`.

Sequence structure — one inversion pulse gates one "segment" of PE lines at
a fixed slice; the segment length is ``ETL`` (chunked if ``ETL < NY``).
Slices are the outer loop (one inversion each); PE is the inner, per-segment
loop (accelerated via ``Ry``):

    for slice in slices:                              # one inversion each
        non-selective inversion pulse (hard or adiabatic hypsec)
        3-axis non-selective spoiler
        TI delay (measured from the inversion RF center to the temporal
            center of the following segment — see ``_prep_common.py``)
        for ky_chunk in chunks(PE lines, ETL):         # one shot
            for ky in ky_chunk:
                single-echo GRE view (slice-selective sinc excitation with
                per-slice frequency offset, RF spoiled — shares
                readout/echo-train helpers with gre.py/gre_multiecho_2d.py)
            recovery delay (``Trecovery``) before the next shot/inversion

Note: the inversion pulse is non-selective (global), so for multi-slice
acquisitions each slice's inversion also re-inverts the magnetization of
every other slice — this plugin is a straightforward per-slice-independent
simplification (accurate for single-slice use, or when ``Trecovery`` is long
enough that cross-slice history doesn't matter for the intended contrast),
not a true multi-slice-interleaved IR scheme.

``TE``/``TR`` retain the same per-view meaning as in ``gre.py`` (``TR`` is
the inner, per-view period). ``ETL`` (``IntKey.ETL``, a real native GE
variable already unused elsewhere) is reused here for "views per inversion
shot". ``Trecovery`` (``FloatKey.TRECOVERY``, native GE
``optrecovery``/``pitrecovery*``) is the post-segment recovery delay. TI and
the inversion-mode toggle (hard vs adiabatic) have no native GE ``pi*``
counterpart, so both are carried as ``opuser`` custom variables.

Usage
-----
Register the plugin in the interpreter tree by symlinking (or copying) this
file to the ``sequences/src/`` directory and creating a numbered alias::

    cp mprage_2d.py pulserver-interpreter/package/pulserver/sequences/src/mprage_2d.py
    ln -sf src/mprage_2d.py pulserver-interpreter/package/pulserver/sequences/sequence5.py
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pypulseq as pp

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
from pulserver.core import SequenceType


def _load_sibling_module(name: str):
    """Load a same-directory helper module by file path (see gre_multiecho_2d.py)."""
    module_path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gc = _load_sibling_module("_gre_common")
pc = _load_sibling_module("_prep_common")

NUM_ECHOES = 1
FLYBACK = True

USER_SLOT_TI = 0
USER_SLOT_INV_MODE = 1


class Mprage2DPulseqSequence(PulseqSequence):
    """Generate a 2D inversion-prepared (MPRAGE) Cartesian GRE sequence."""

    def get_default_protocol(self, opts: pp.Opts) -> dict[str, dict]:
        del opts
        protocol = {
            UIParam.TE: DropdownFloatParam(
                value=4.0, min=2.0, max=80.0, incr=0.1, unit="ms",
                options=[3.0, 4.0, 8.0, 15.0, 30.0], validate=Validate.NONE,
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
            UIParam.PHASE_FOV: DropdownFloatParam(
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
            UIParam.NY: DropdownIntParam(
                value=64, min=8, max=512, incr=1, options=[64, 128, 192, 256, 384], validate=Validate.NONE,
            ),
            UIParam.NSLICES: DropdownIntParam(
                value=1, min=1, max=128, incr=1, options=[1, 5, 10, 20, 40], validate=Validate.NONE,
            ),
            UIParam.RY: TypeinFloatParam(value=1.0, min=1.0, max=8.0, incr=1.0, unit="", validate=Validate.NONE),
            UIParam.BANDWIDTH: TypeinFloatParam(
                value=gc.DEFAULT_BANDWIDTH_HZ_PX, min=5_000.0, max=500_000.0, incr=100.0,
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
        }
        return protocol_to_dict(protocol)

    def validate_protocol(self, opts: pp.Opts, protocol: dict[str, dict]) -> dict:
        prot = dict_to_protocol(protocol)
        cfg = _read_protocol(prot)

        if cfg.te_s <= 0.0 or cfg.tr_s <= 0.0:
            return {"valid": False, "duration": None, "info": "TE and TR must be > 0"}
        if cfg.fov_ro_m <= 0.0 or cfg.fov_pe_m <= 0.0 or cfg.slice_thickness_m <= 0.0:
            return {"valid": False, "duration": None, "info": "FOV and slice thickness must be > 0"}
        if not (0.0 < cfg.flip_deg <= 180.0):
            return {"valid": False, "duration": None, "info": "Flip angle must be in (0, 180] deg"}
        if cfg.nx_ro < 1 or cfg.ny_pe < 1 or cfg.nslices < 1:
            return {"valid": False, "duration": None, "info": "NX, NY, and NSLICES must be >= 1"}
        if cfg.bandwidth_hz_px <= 0.0:
            return {"valid": False, "duration": None, "info": "Bandwidth must be > 0"}
        if cfg.etl < 1:
            return {"valid": False, "duration": None, "info": "ETL must be >= 1"}
        if cfg.ti_s <= 0.0:
            return {"valid": False, "duration": None, "info": "TI must be > 0"}
        if cfg.trecovery_s < 0.0:
            return {"valid": False, "duration": None, "info": "Trecovery must be >= 0"}

        timing = _compute_timing(opts=opts, cfg=cfg, strict=True)
        if timing is None:
            return {
                "valid": False,
                "duration": None,
                "info": "TE, TR, or TI infeasible for the requested gradients/ETL",
            }

        sampled_pe = gc.sampled_lines(cfg.ny_pe, cfg.ry, 0)
        n_segments = len(pc.chunk_indices(sampled_pe, cfg.etl))
        shot_s = timing["shot_s"]
        duration_s = shot_s * n_segments * cfg.nslices
        return {"valid": True, "duration": duration_s, "info": f"TA = {duration_s:.2f} s"}

    def make_sequence(self, opts: pp.Opts, protocol: dict[str, dict], output_path: str) -> None:
        prot = dict_to_protocol(protocol)
        cfg = _read_protocol(prot)

        timing = _compute_timing(opts=opts, cfg=cfg, strict=False)

        gz = timing["gz"]
        gz_reph = timing["gz_reph"]
        echo = timing["echo"]
        gy_template = timing["gy_template"]
        te_delay_s = timing["te_delay_s"]
        tr_delay_s = timing["tr_delay_s"]
        rf_inv = timing["rf_inv"]
        gx_inv, gy_inv, gz_inv = timing["inv_spoiler"]
        spoiler_duration_s = timing["spoiler_duration_s"]

        te_delay = pp.make_delay(te_delay_s) if te_delay_s > 0.0 else None
        tr_delay = pp.make_delay(tr_delay_s) if tr_delay_s > 0.0 else None
        trecovery_s_rounded = pc.round_to_raster(cfg.trecovery_s)
        trecovery_delay = pp.make_delay(trecovery_s_rounded) if trecovery_s_rounded > 0.0 else None

        seq = ps.Sequence(opts)

        delta_k_pe = 1.0 / cfg.fov_pe_m
        phase_areas = (np.arange(cfg.ny_pe) - 0.5 * cfg.ny_pe) * delta_k_pe
        max_pe_area = float(np.max(np.abs(phase_areas)))
        sampled_pe = gc.sampled_lines(cfg.ny_pe, cfg.ry, 0)
        segments = pc.chunk_indices(sampled_pe, cfg.etl)

        slice_step_m = cfg.slice_spacing_m if cfg.nslices > 1 else 0.0

        for sl in range(cfg.nslices):
            slice_offset_m = (sl - 0.5 * (cfg.nslices - 1)) * slice_step_m
            label_slc = pp.make_label(type="SET", label="SLC", value=sl)

            for segment in segments:
                segment_duration_s = len(segment) * cfg.tr_s
                ti_delay_s = pc.ti_delay_seconds(
                    cfg.ti_s, rf_inv, spoiler_duration_s, segment_duration_s, strict=False
                )
                ti_delay = pp.make_delay(ti_delay_s) if ti_delay_s > 0.0 else None

                seq.add_block(gc.copy_event(rf_inv))
                seq.add_block(gx_inv, gy_inv, gz_inv)
                if ti_delay is not None:
                    seq.add_block(ti_delay)

                rf_phase_deg = 0.0
                rf_phase_inc_deg = 0.0

                for ky in segment:
                    y_scale = phase_areas[ky] / max_pe_area if max_pe_area > 0.0 else 0.0
                    gy_pre = pp.scale_grad(gy_template, y_scale)
                    gy_reph = pp.scale_grad(gy_template, -y_scale)

                    rf_curr = gc.copy_event(timing["rf"])
                    rf_curr.freq_offset = gz.amplitude * slice_offset_m
                    rf_curr.phase_offset = np.deg2rad(rf_phase_deg)

                    label_lin = pp.make_label(type="SET", label="LIN", value=ky)

                    seq.add_block(rf_curr, gz, label_slc, label_lin)
                    seq.add_block(echo["gx_pre"], gy_pre, gz_reph)
                    if te_delay is not None:
                        seq.add_block(te_delay)

                    gc.add_echo_train_blocks(
                        seq, echo, NUM_ECHOES, FLYBACK, rf_curr.phase_offset, emit_labels=False
                    )

                    seq.add_block(echo["gx_spoil"], gy_reph, timing["gz_spoil"])
                    if tr_delay is not None:
                        seq.add_block(tr_delay)

                    rf_phase_deg = (rf_phase_deg + rf_phase_inc_deg) % 360.0
                    rf_phase_inc_deg = (rf_phase_inc_deg + gc.RF_SPOILING_INC_DEG) % 360.0

                if trecovery_delay is not None:
                    seq.add_block(trecovery_delay)

        seq.set_definition("Name", "mprage_2d")
        seq.set_definition(
            "FOV",
            [cfg.fov_ro_m, cfg.fov_pe_m, slice_step_m * cfg.nslices if cfg.nslices > 1 else cfg.slice_thickness_m],
        )
        seq.set_definition("TE", cfg.te_s)
        seq.set_definition("TR", cfg.tr_s)
        seq.set_definition("TI", cfg.ti_s)
        seq.set_definition("Trecovery", cfg.trecovery_s)
        seq.set_definition("ETL", cfg.etl)
        seq.set_definition("InversionMode", cfg.inv_mode)
        seq.set_definition("Flip", cfg.flip_deg)
        seq.set_definition("ImagingMode", "2d")
        seq.set_definition("ReadoutAxis", cfg.ro_axis)
        seq.set_definition("PhaseAxis", cfg.pe_axis)
        seq.set_definition("BandwidthHzPerPx", cfg.bandwidth_hz_px)
        seq.set_definition("Ry", cfg.ry)
        seq.set_definition("RfSpoilingIncDeg", gc.RF_SPOILING_INC_DEG)
        seq.set_definition("Nx", cfg.nx_ro)
        seq.set_definition("Ny", cfg.ny_pe)
        seq.set_definition("NySampled", len(sampled_pe))
        seq.set_definition("NumSlices", cfg.nslices)
        pio.write(seq, output=output_path, remove_duplicates=False, check_timing=False)


class _Config:
    __slots__ = (
        "te_s", "tr_s", "flip_deg", "fov_ro_m", "fov_pe_m", "slice_thickness_m", "slice_spacing_m",
        "nx_ro", "ny_pe", "nslices", "bandwidth_hz_px", "ry", "ro_axis", "pe_axis",
        "etl", "ti_s", "trecovery_s", "inv_mode",
    )


def _read_protocol(prot: dict) -> _Config:
    cfg = _Config()
    cfg.te_s = gc.param_float(prot, UIParam.TE) * 1e-3
    cfg.tr_s = gc.param_float(prot, UIParam.TR) * 1e-3
    cfg.flip_deg = gc.param_float(prot, UIParam.FLIP)
    cfg.fov_ro_m = gc.param_float(prot, UIParam.FOV) * 1e-3
    cfg.fov_pe_m = gc.phase_fov_mm_from_protocol(prot) * 1e-3
    cfg.slice_thickness_m = gc.param_float(prot, UIParam.SLICE_THICKNESS) * 1e-3
    cfg.slice_spacing_m = gc.param_float(prot, UIParam.SLICE_SPACING) * 1e-3
    cfg.nx_ro = gc.param_int(prot, UIParam.NX)
    cfg.ny_pe = gc.param_int(prot, UIParam.NY)
    cfg.nslices = gc.param_int(prot, UIParam.NSLICES)
    cfg.bandwidth_hz_px = gc.param_float_optional(prot, UIParam.BANDWIDTH, gc.DEFAULT_BANDWIDTH_HZ_PX)
    cfg.ry = max(1, int(round(gc.param_float_optional(prot, UIParam.RY, 1.0))))
    cfg.ro_axis, cfg.pe_axis = gc.resolve_readout_phase_axes(prot)
    cfg.etl = gc.param_int_optional(prot, UIParam.ETL, cfg.ny_pe)
    cfg.ti_s = gc.user_float(prot, USER_SLOT_TI, 900.0) * 1e-3
    cfg.trecovery_s = gc.param_float_optional(prot, UIParam.TRECOVERY, 1200.0) * 1e-3
    cfg.inv_mode = "adiabatic" if gc.user_float(prot, USER_SLOT_INV_MODE, 1.0) >= 0.5 else "hard"
    return cfg


def _compute_timing(opts: pp.Opts, cfg: _Config, strict: bool):
    gc.apply_system_derates(opts)

    rf, gz, gz_reph = gc.build_rf(opts, cfg.flip_deg, cfg.slice_thickness_m)

    echo = gc.compute_readout_and_echo_train(
        opts=opts,
        ro_axis=cfg.ro_axis,
        nx_ro=cfg.nx_ro,
        fov_ro_m=cfg.fov_ro_m,
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
    max_pe_area = 0.5 * cfg.ny_pe * (1.0 / cfg.fov_pe_m)
    gy_template = pp.make_trapezoid(channel=cfg.pe_axis, area=max_pe_area, system=opts)

    d_rf = pp.calc_duration(rf, gz)
    d_pre = pp.calc_duration(echo["gx_pre"], gy_template, gz_reph)
    d_post = pp.calc_duration(echo["gx_spoil"], gy_template, gz_spoil)

    rf_center_s = pp.calc_rf_center(rf)[0]
    min_te_s = (d_rf - rf_center_s) + d_pre + echo["adc_center_s"]
    te_delay_s = cfg.te_s - min_te_s
    if te_delay_s < -1e-9 and strict:
        return None
    if te_delay_s < 0.0:
        te_delay_s = 0.0

    min_block_s = d_rf + d_pre + te_delay_s + echo["echo_train_span_s"] + d_post
    tr_delay_s = cfg.tr_s - min_block_s
    if tr_delay_s < -1e-9 and strict:
        return None
    if tr_delay_s < 0.0:
        tr_delay_s = 0.0

    rf_inv = pc.build_inversion_pulse(opts, cfg.inv_mode)
    voxel_size_m = cfg.fov_pe_m / cfg.ny_pe
    inv_spoiler = pc.build_inversion_spoiler(opts, voxel_size_m)
    spoiler_duration_s = pp.calc_duration(*inv_spoiler)

    worst_segment_len = max(1, min(cfg.etl, cfg.ny_pe))
    worst_segment_duration_s = worst_segment_len * cfg.tr_s
    ti_delay_s = pc.ti_delay_seconds(cfg.ti_s, rf_inv, spoiler_duration_s, worst_segment_duration_s, strict=strict)
    if ti_delay_s is None:
        return None

    inv_duration_s = pp.calc_duration(rf_inv)
    shot_s = inv_duration_s + spoiler_duration_s + ti_delay_s + worst_segment_duration_s + cfg.trecovery_s

    return {
        "rf": rf,
        "gz": gz,
        "gz_reph": gz_reph,
        "gz_spoil": gz_spoil,
        "gy_template": gy_template,
        "echo": echo,
        "te_delay_s": te_delay_s,
        "tr_delay_s": tr_delay_s,
        "min_block_s": min_block_s,
        "rf_inv": rf_inv,
        "inv_spoiler": inv_spoiler,
        "spoiler_duration_s": spoiler_duration_s,
        "shot_s": shot_s,
    }


PLUGIN = Mprage2DPulseqSequence()


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
    parser = argparse.ArgumentParser(description="Generate a 2D inversion-prepared (MPRAGE) .seq offline.")
    parser.add_argument("-o", "--output", default="mprage_2d.seq", help="Output .seq file path")
    parser.add_argument("--te-ms", type=float)
    parser.add_argument("--tr-ms", type=float)
    parser.add_argument("--ti-ms", type=float)
    parser.add_argument("--trecovery-ms", type=float)
    parser.add_argument("--etl", type=int)
    parser.add_argument("--inversion-mode", choices=["hard", "adiabatic"])
    parser.add_argument("--flip-deg", type=float)
    parser.add_argument("--fov-mm", type=float)
    parser.add_argument("--phase-fov-mm", type=float)
    parser.add_argument("--slice-thickness-mm", type=float)
    parser.add_argument("--slice-spacing-mm", type=float)
    parser.add_argument("--nx", type=int)
    parser.add_argument("--ny", type=int)
    parser.add_argument("--nslices", type=int)
    parser.add_argument("--bandwidth-hz-px", type=float)
    parser.add_argument("--ry", type=float)
    parser.add_argument("--swap-phase-freq", action="store_true")
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
    if args.ti_ms is not None:
        gc.set_protocol_value(protocol, UIParam.user_value(USER_SLOT_TI), args.ti_ms)
    if args.trecovery_ms is not None:
        gc.set_protocol_value(protocol, UIParam.TRECOVERY, args.trecovery_ms)
    if args.etl is not None:
        gc.set_protocol_value(protocol, UIParam.ETL, args.etl)
    if args.inversion_mode is not None:
        gc.set_protocol_value(protocol, UIParam.user_value(USER_SLOT_INV_MODE), 1.0 if args.inversion_mode == "adiabatic" else 0.0)
    if args.flip_deg is not None:
        gc.set_protocol_value(protocol, UIParam.FLIP, args.flip_deg)
    if args.fov_mm is not None:
        gc.set_protocol_value(protocol, UIParam.FOV, args.fov_mm)
    if args.phase_fov_mm is not None:
        gc.set_protocol_value(protocol, UIParam.PHASE_FOV, args.phase_fov_mm)
    if args.slice_thickness_mm is not None:
        gc.set_protocol_value(protocol, UIParam.SLICE_THICKNESS, args.slice_thickness_mm)
    if args.slice_spacing_mm is not None:
        gc.set_protocol_value(protocol, UIParam.SLICE_SPACING, args.slice_spacing_mm)
    if args.nx is not None:
        gc.set_protocol_value(protocol, UIParam.NX, args.nx)
    if args.ny is not None:
        gc.set_protocol_value(protocol, UIParam.NY, args.ny)
    if args.nslices is not None:
        gc.set_protocol_value(protocol, UIParam.NSLICES, args.nslices)
    if args.bandwidth_hz_px is not None:
        gc.set_protocol_value(protocol, UIParam.BANDWIDTH, args.bandwidth_hz_px)
    if args.ry is not None:
        gc.set_protocol_value(protocol, UIParam.RY, args.ry)
    if args.swap_phase_freq:
        gc.set_protocol_value(protocol, UIParam.SWAP_PHASE_FREQ, True)

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
