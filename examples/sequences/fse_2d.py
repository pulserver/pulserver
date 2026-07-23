"""Standalone 2D fast/turbo spin-echo (FSE) sequence plugin for pulserver.

This module implements the three mandatory module-level entry points
required by the bridge dispatcher (see ``gre_2d.py`` for the Cartesian GRE
family this parallels, ``epi_2d.py`` for the other spin-echo-family
sibling, and ``_gre_common.py`` / ``_fse_common.py`` / ``_diffusion_common.py``
for the shared building blocks):

- ``get_default_protocol(opts)``  — return the initial CV/protocol dictionary.
- ``validate_protocol(opts, protocol)``  — validate the current protocol and
  return a ``{"valid": bool, "duration": float | None, "info": str}`` dict.
- ``make_sequence(opts, protocol, output_path)``  — synthesise the sequence
  and write it to *output_path* using :func:`pulserver.io.write`.

Structure: 90 (slice-selective, fixed at ``_fse_common.EXCITATION_FLIP_DEG``,
not user-selectable) - crusher - [diffusion, first 180 only] - CPMG train of
``ETL`` refocusing pulses (constant 90 deg phase offset from the 90 — see
``_fse_common.py``), each flanked by a crusher and followed by its own
independently phase-encoded readout (a different ``ky`` per echo — NOT a
blip train). ``TE`` doubles as both "time to the first echo" and the echo
spacing (ESP): the symmetric crusher/prephase/rephase construction keeps
every inter-180 period identical by design (see ``_fse_common.fse_timing``),
so the two are the same value here — a simplification of the more general
clinical FSE case where phase-encode *reordering* (centric etc.) decouples
"effective TE" from ESP; this plugin always uses simple sequential ky order.

Flip-angle role (GE/TSEplus convention for spin-echo sequences): the
standard flip control (``UIParam.FLIP``) drives the REFOCUSING pulse angle
(default 180 deg, constant across the train unless the TRAPS scheme is
enabled), not the 90 excitation, which is fixed. See
``_fse_common.py``'s module docstring.

``ETL`` (native GE, "echoes per shot") splits ``NY`` into
``ceil(NY / ETL)`` sequential shots, each with its own 90-CPMG-train.
``DIFFUSION_BVALUES``/``DIFFUSION_DIRECTIONS`` (native) flank only the
FIRST 180 in the train (as a b0/DW preparation before the T2-weighted
readout proper) — ``b <= 0`` (default) disables it, same convention as
``epi_2d.py``.

Not implemented in this pass (noted, not silently dropped): optional
inversion before the 90 (IR-FSE / FLAIR-FSE) — same proven pattern as
``gre_mprage_radial_2d.py``/``epi_2d.py``, deferred to reach ZTE.

Usage
-----
Register the plugin in the interpreter tree by symlinking (or copying) this
file to the ``sequences/src/`` directory and creating a numbered alias::

    cp fse_2d.py pulserver-interpreter/package/pulserver/sequences/src/fse_2d.py
    ln -sf src/fse_2d.py pulserver-interpreter/package/pulserver/sequences/sequence15.py
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

# Refocusing scheme (TRAPS on/off) has no native GE CV — carried as an
# opuser custom variable, same convention as gre_multiecho_2d.py's
# USER_SLOT_*. The refocusing flip ANGLE itself is not a separate slot: it
# reuses the standard UIParam.FLIP control (see module docstring).
USER_SLOT_REFOCUS_SCHEME = 0


class Fse2DPulseqSequence(Sequence):
    """Generate a 2D turbo/fast spin-echo (CPMG) sequence."""

    def get_default_protocol(self, opts: pp.Opts) -> dict[str, dict]:
        del opts
        protocol = {
            UIParam.TE: DropdownFloatParam(
                value=12.0, min=5.0, max=50.0, incr=0.1, unit="ms",
                options=[8.0, 10.0, 12.0, 16.0, 20.0], validate=Validate.NONE,
            ),
            UIParam.TR: DropdownFloatParam(
                value=3000.0, min=200.0, max=10000.0, incr=1.0, unit="ms",
                options=[1500.0, 2000.0, 3000.0, 5000.0, 8000.0], validate=Validate.NONE,
            ),
            # Spin-echo role: FLIP drives the REFOCUSING pulse (the 90
            # excitation is fixed, not user-selectable — see module
            # docstring / _fse_common.EXCITATION_FLIP_DEG).
            UIParam.FLIP: DropdownFloatParam(
                value=180.0, min=90.0, max=180.0, incr=1.0, unit="deg",
                options=[90.0, 120.0, 150.0, 180.0], validate=Validate.NONE,
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
            UIParam.ETL: DropdownIntParam(
                value=16, min=1, max=256, incr=1, options=[4, 8, 16, 32, 64], validate=Validate.NONE,
            ),
            UIParam.RY: TypeinFloatParam(value=1.0, min=1.0, max=8.0, incr=1.0, unit="", validate=Validate.NONE),
            UIParam.BANDWIDTH: TypeinFloatParam(
                value=125_000.0, min=5_000.0, max=500_000.0, incr=100.0,
                unit="Hz/px", validate=Validate.NONE,
            ),
            UIParam.DIFFUSION_BVALUES: TypeinFloatParam(
                value=0.0, min=0.0, max=5000.0, incr=1.0, unit="s/mm2", validate=Validate.NONE,
            ),
            UIParam.DIFFUSION_DIRECTIONS: DropdownIntParam(
                value=3, min=1, max=32, incr=1, options=[1, 3, 6], validate=Validate.NONE,
            ),
            UIParam.SWAP_PHASE_FREQ: BoolParam(value=False, validate=Validate.NONE),
            UIParam.SEQUENCE_TYPE: make_enum_param(UIParam.SEQUENCE_TYPE, SequenceType.SPIN_ECHO),
            UIParam.user_name(USER_SLOT_REFOCUS_SCHEME): Description(
                text="Refocusing scheme (0=constant, 1=variable/TRAPS)"
            ),
            UIParam.user_value(USER_SLOT_REFOCUS_SCHEME): DropdownFloatParam(
                value=0.0, min=0.0, max=1.0, incr=1.0, unit="", options=[0.0, 1.0], validate=Validate.NONE,
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
        if not (0.0 < cfg.refocus_flip_deg <= 180.0):
            return {"valid": False, "duration": None, "info": "Flip (refocusing) angle must be in (0, 180] deg"}
        if cfg.nx_ro < 1 or cfg.ny_pe < 1 or cfg.nslices < 1:
            return {"valid": False, "duration": None, "info": "NX, NY, and NSLICES must be >= 1"}
        if cfg.bandwidth_hz_px <= 0.0:
            return {"valid": False, "duration": None, "info": "Bandwidth must be > 0"}
        if cfg.etl < 1:
            return {"valid": False, "duration": None, "info": "ETL must be >= 1"}
        if cfg.b_value_s_mm2 < 0.0:
            return {"valid": False, "duration": None, "info": "Diffusion b-value must be >= 0"}

        timing = _compute_public(opts, cfg, strict=True)
        if timing is None:
            return {
                "valid": False,
                "duration": None,
                "info": "TE (ESP) too short for the 90/180/crushers/diffusion/readout, or TR too short",
            }

        n_shots = len(_segment_loop(cfg))
        duration_s = cfg.tr_s * float(n_shots)
        return {"valid": True, "duration": duration_s, "info": f"TA = {duration_s:.2f} s"}

    def make_sequence(self, opts: pp.Opts, protocol: dict[str, dict], output_path: str) -> None:
        prot = dict_to_protocol(protocol)
        cfg = _read_protocol(prot)

        seq = pp.Sequence(opts)
        n_shots, n_directions = _build_public(seq, opts, cfg)

        slice_step_m = cfg.slice_spacing_m if cfg.nslices > 1 else 0.0
        seq.set_definition("Name", "fse_2d")
        seq.set_definition(
            "FOV",
            [cfg.fov_ro_m, cfg.fov_pe_m, slice_step_m * cfg.nslices if cfg.nslices > 1 else cfg.slice_thickness_m],
        )
        seq.set_definition("TE", cfg.te_s)
        seq.set_definition("TR", cfg.tr_s)
        seq.set_definition("Flip", cfg.refocus_flip_deg)
        seq.set_definition("ExcitationFlip", 90.0)
        seq.set_definition("RefocusVariable", cfg.refocus_variable)
        seq.set_definition("ImagingMode", "2d")
        seq.set_definition("ReadoutAxis", "x")
        seq.set_definition("PhaseAxis", "y")
        seq.set_definition("ETL", cfg.etl)
        seq.set_definition("NumShots", n_shots)
        seq.set_definition("BValue", cfg.b_value_s_mm2)
        seq.set_definition("DiffusionDirections", n_directions)
        seq.set_definition("BandwidthHzPerPx", cfg.bandwidth_hz_px)
        seq.set_definition("Ry", cfg.ry)
        seq.set_definition("Nx", cfg.nx_ro)
        seq.set_definition("Ny", cfg.ny_pe)
        seq.set_definition("NumSlices", cfg.nslices)
        pio.write(seq, output=output_path, remove_duplicates=False, check_timing=False)


def _segment_loop(cfg: _Config):
    """Phase-encode loop chunked into echo trains — one shot per train."""
    return design.make_cartesian_sampling(
        (cfg.nx_ro, cfg.ny_pe), acceleration=cfg.ry, train_length=cfg.etl
    )


class _Config:
    __slots__ = (
        "te_s", "tr_s", "refocus_flip_deg", "refocus_variable",
        "fov_ro_m", "fov_pe_m", "slice_thickness_m", "slice_spacing_m",
        "nx_ro", "ny_pe", "nslices", "etl", "ry", "bandwidth_hz_px", "b_value_s_mm2", "n_directions",
        "ro_axis", "pe_axis",
    )


def _read_protocol(prot: dict) -> _Config:
    cfg = _Config()
    cfg.te_s = params.param_float(prot, UIParam.TE) * 1e-3
    cfg.tr_s = params.param_float(prot, UIParam.TR) * 1e-3
    # Spin-echo role: FLIP is the refocusing angle (see module docstring).
    cfg.refocus_flip_deg = params.param_float(prot, UIParam.FLIP)
    cfg.refocus_variable = params.user_float(prot, USER_SLOT_REFOCUS_SCHEME, 0.0) >= 0.5
    cfg.fov_ro_m = params.param_float(prot, UIParam.FOV) * 1e-3
    cfg.fov_pe_m = params.phase_fov_mm_from_protocol(prot) * 1e-3
    cfg.slice_thickness_m = params.param_float(prot, UIParam.SLICE_THICKNESS) * 1e-3
    cfg.slice_spacing_m = params.param_float(prot, UIParam.SLICE_SPACING) * 1e-3
    cfg.nx_ro = params.param_int(prot, UIParam.NX)
    cfg.ny_pe = params.param_int(prot, UIParam.NY)
    cfg.nslices = params.param_int(prot, UIParam.NSLICES)
    cfg.etl = params.param_int_optional(prot, UIParam.ETL, cfg.ny_pe)
    cfg.ry = max(1, int(round(params.param_float_optional(prot, UIParam.RY, 1.0))))
    cfg.bandwidth_hz_px = params.param_float_optional(prot, UIParam.BANDWIDTH, 125_000.0)
    cfg.b_value_s_mm2 = params.param_float_optional(prot, UIParam.DIFFUSION_BVALUES, 0.0)
    cfg.n_directions = params.param_int_optional(prot, UIParam.DIFFUSION_DIRECTIONS, 3)
    cfg.ro_axis, cfg.pe_axis = params.resolve_readout_phase_axes(prot)
    return cfg


def _compute_public(opts: pp.Opts, cfg: _Config, strict: bool):
    excitation_pulse = design.make_slice_selective_pulse(
        np.pi / 2.0, cfg.slice_thickness_m, system=opts
    )
    refocusing = design.make_refocusing_pulse(
        slice_thickness=cfg.slice_thickness_m, system=opts
    )
    flips = design.make_traps_schedule(
        cfg.etl, np.deg2rad(cfg.refocus_flip_deg), variable=cfg.refocus_variable
    )
    train = design.make_fse_readout(
        opts,
        (cfg.fov_ro_m, cfg.fov_pe_m),
        (cfg.nx_ro, cfg.ny_pe),
        cfg.etl,
        refocusing,
        bandwidth_hz_px=cfg.bandwidth_hz_px,
        esp_s=cfg.te_s,
        refoc_flip_scale=flips / np.pi,
        spoil_cycles=4.0,
    )
    diffusion = None
    if cfg.b_value_s_mm2 > 0.0:
        try:
            diffusion = design.make_diffusion_prep(
                cfg.b_value_s_mm2,
                voxel_size=cfg.fov_ro_m / cfg.nx_ro,
                system=opts,
            )
        except ValueError:
            if strict:
                return None

    d_exc = sum(pp.calc_duration(*block) for block in excitation_pulse)
    center = pp.calc_rf_center(excitation_pulse.rf)[0] + excitation_pulse.rf.delay
    gap_s = max(0.0, train.t_exc_center_to_train_start - (d_exc - center))
    raster = opts.block_duration_raster
    gap_s = round(gap_s / raster) * raster
    prep_s = 0.0 if diffusion is None else diffusion.duration
    min_block_s = prep_s + d_exc + gap_s + train.duration
    tr_delay_s = round((cfg.tr_s - min_block_s) / raster) * raster
    if tr_delay_s < -1e-9 and strict:
        return None
    return {
        "excitation": excitation_pulse,
        "train": train,
        "diffusion": diffusion,
        "gap_s": gap_s,
        "tr_delay_s": max(0.0, tr_delay_s),
        "min_block_s": min_block_s,
    }


def _build_public(seq, opts: pp.Opts, cfg: _Config) -> tuple[int, int]:
    timing = _compute_public(opts, cfg, strict=False)
    excitation_pulse = timing["excitation"]
    train = timing["train"]
    diffusion = timing["diffusion"]
    gap = pp.make_delay(timing["gap_s"]) if timing["gap_s"] > 0.0 else None
    tr_delay = pp.make_delay(timing["tr_delay_s"]) if timing["tr_delay_s"] > 0.0 else None
    segments = _segment_loop(cfg)
    rotations = (
        design.make_noncartesian_projection_sampling(
            (cfg.nx_ro, cfg.nx_ro, cfg.nx_ro), views=cfg.n_directions
        ).to_rotations()
        if diffusion is not None
        else [None]
    )
    slice_step = cfg.slice_spacing_m if cfg.nslices > 1 else 0.0
    slices = design.make_slice_loop(
        cfg.nslices, slice_step or cfg.slice_thickness_m, order="sequential"
    )
    offsets_hz = slices.to_frequencies(excitation_pulse.gradients[0].amplitude) if slice_step else None
    for rotation in rotations:
        for sl, band in enumerate(slices.shots):
            freq = float(offsets_hz[band[0]]) if offsets_hz is not None else 0.0
            for segment in segments:
                indices = np.resize(segment[:, 0].astype(int), cfg.etl)
                if diffusion is not None:
                    diffusion.set_state(b_value=cfg.b_value_s_mm2, rotation=rotation)
                    for block in diffusion:
                        seq.add_block(*block)
                excitation_pulse.set_state(freq_offset_hz=freq)
                excitation_pulse.set_labels(*slices.labels(sl))
                for block in excitation_pulse:
                    seq.add_block(*block)
                if gap is not None:
                    seq.add_block(gap)
                train.set_state(lin_idx=indices, freq_offset_hz=freq)
                for block in train:
                    seq.add_block(*block)
                if tr_delay is not None:
                    seq.add_block(tr_delay)
    return len(segments), len(rotations)


PLUGIN = Fse2DPulseqSequence()


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
    ('--flip-deg', UIParam.FLIP, float, 'Refocusing flip angle (deg); excitation is fixed at 90'),
    ('--refocus-variable', UIParam.user_value(USER_SLOT_REFOCUS_SCHEME), ("const", 1.0), ""),
    ('--fov-mm', UIParam.FOV, float, ""),
    ('--phase-fov-mm', UIParam.PHASE_FOV, float, ""),
    ('--slice-thickness-mm', UIParam.SLICE_THICKNESS, float, ""),
    ('--slice-spacing-mm', UIParam.SLICE_SPACING, float, ""),
    ('--nx', UIParam.NX, int, ""),
    ('--ny', UIParam.NY, int, ""),
    ('--nslices', UIParam.NSLICES, int, ""),
    ('--etl', UIParam.ETL, int, ""),
    ('--ry', UIParam.RY, float, ""),
    ('--bandwidth-hz-px', UIParam.BANDWIDTH, float, ""),
    ('--bvalue', UIParam.DIFFUSION_BVALUES, float, ""),
    ('--directions', UIParam.DIFFUSION_DIRECTIONS, int, ""),
    ('--swap-phase-freq', UIParam.SWAP_PHASE_FREQ, ("const", True), ""),
]

if __name__ == "__main__":
    raise SystemExit(
        run_cli(
            PLUGIN,
            sys.argv[1:],
            arg_map=_ARG_MAP,
            description='Generate a 2D fast/turbo spin-echo .seq offline.',
            default_output='fse_2d.seq',
        )
    )
