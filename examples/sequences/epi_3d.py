"""Standalone 3D single-/multi-shot spin-echo EPI sequence plugin.

This module implements the three mandatory module-level entry points
required by the bridge dispatcher (see ``gre_3d.py`` for the Cartesian 3D
GRE this parallels, ``epi_2d.py`` for the 2D EPI this mirrors, and
``_gre_common.py`` / ``_epi_common.py`` / ``pulserver`` /
``_diffusion_common.py`` for the shared building blocks):

- ``get_default_protocol(opts)``  — return the initial CV/protocol dictionary.
- ``validate_protocol(opts, protocol)``  — validate the current protocol and
  return a ``{"valid": bool, "duration": float | None, "info": str}`` dict.
- ``make_sequence(opts, protocol, output_path)``  — synthesise the sequence
  and write it to *output_path* using :func:`pulserver.io.write`.

Same 90-crusher-[diffusion]-180-crusher-[diffusion]-prephasers-blipped-train
structure as ``epi_2d.py`` (see that file's docstring for the TE/180-timing
rationale), stacked across partitions (accelerated via ``Rz``) like
``gre_3d.py``: a standalone Z phase-encode trapezoid rides alongside the
X/Y prephasers before the train, and is unwound + spoiled in a combined
Z-channel trapezoid after the train (no ``combined_z_gradients`` needed
here since the slab rephase, unlike ``gre_3d.py``, isn't sharing a block
with the partition encode — the rephase happens earlier, symmetric with the
90, before the crushers/180).

Not implemented in this pass (same as ``epi_2d.py``): optional
inversion/T2-prep before the 90.

Usage
-----
Register the plugin in the interpreter tree by symlinking (or copying) this
file to the ``sequences/src/`` directory and creating a numbered alias::

    cp epi_3d.py pulserver-interpreter/package/pulserver/sequences/src/epi_3d.py
    ln -sf src/epi_3d.py pulserver-interpreter/package/pulserver/sequences/sequence14.py
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
    TypeinFloatParam,
    UIParam,
    Validate,
    dict_to_protocol,
    make_enum_param,
    params,
    protocol_to_dict,
    run_cli,
)

USER_SLOT_RAMP_SAMPLE = 0
USER_SLOT_SPSP = 1
USER_SLOT_SPSP_BW = 2
USER_SLOT_TTL = 3
USER_SLOT_PHASE_CORRECTION = 4
USER_SLOT_REVERSE_PE = 5


class Epi3DPulseqSequence(Sequence):
    """Generate a 3D single-/multi-shot spin-echo EPI sequence."""

    def get_default_protocol(self, opts: pp.Opts) -> dict[str, dict]:
        del opts
        protocol = {
            UIParam.TE: DropdownFloatParam(
                value=60.0, min=20.0, max=200.0, incr=0.1, unit="ms",
                options=[40.0, 60.0, 80.0, 100.0, 150.0], validate=Validate.NONE,
            ),
            UIParam.TR: DropdownFloatParam(
                value=2000.0, min=200.0, max=10000.0, incr=1.0, unit="ms",
                options=[1000.0, 2000.0, 3000.0, 5000.0, 8000.0], validate=Validate.NONE,
            ),
            UIParam.FLIP: DropdownFloatParam(
                value=90.0, min=1.0, max=90.0, incr=1.0, unit="deg",
                options=[45.0, 60.0, 90.0], validate=Validate.NONE,
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
                value=2.0, min=0.5, max=10.0, incr=0.5, unit="mm",
                options=[1.0, 1.5, 2.0, 2.5, 3.0], validate=Validate.NONE,
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
            UIParam.ETL: DropdownIntParam(
                value=64, min=1, max=512, incr=1, options=[16, 32, 64, 128, 256], validate=Validate.NONE,
            ),
            UIParam.RZ: TypeinFloatParam(value=1.0, min=1.0, max=8.0, incr=1.0, unit="", validate=Validate.NONE),
            UIParam.BANDWIDTH: TypeinFloatParam(
                value=250_000.0, min=50_000.0, max=1_000_000.0, incr=1_000.0,
                unit="Hz/px", validate=Validate.NONE,
            ),
            UIParam.DIFFUSION_BVALUES: TypeinFloatParam(
                value=0.0, min=0.0, max=5000.0, incr=1.0, unit="s/mm2", validate=Validate.NONE,
            ),
            UIParam.DIFFUSION_DIRECTIONS: DropdownIntParam(
                value=3, min=1, max=32, incr=1, options=[1, 3, 6], validate=Validate.NONE,
            ),
            UIParam.NUM_FRAMES: DropdownIntParam(
                value=1, min=1, max=10000, incr=1, options=[1, 10, 50, 100], validate=Validate.NONE,
            ),
            UIParam.SEQUENCE_TYPE: make_enum_param(UIParam.SEQUENCE_TYPE, SequenceType.SPIN_ECHO),
            UIParam.user_value(USER_SLOT_RAMP_SAMPLE): DropdownFloatParam(
                value=0.0, min=0.0, max=1.0, incr=1.0, unit="", options=[0.0, 1.0], validate=Validate.NONE,
            ),
            UIParam.user_name(USER_SLOT_SPSP): Description(text="Excitation (0=slab selective, 1=SPSP water selective)"),
            UIParam.user_value(USER_SLOT_SPSP): DropdownFloatParam(
                value=0.0, min=0.0, max=1.0, incr=1.0, unit="", options=[0.0, 1.0], validate=Validate.NONE,
            ),
            UIParam.user_name(USER_SLOT_SPSP_BW): Description(text="SPSP spectral bandwidth"),
            UIParam.user_value(USER_SLOT_SPSP_BW): TypeinFloatParam(
                value=250.0, min=50.0, max=1000.0, incr=10.0, unit="Hz", validate=Validate.NONE,
            ),
            UIParam.user_name(USER_SLOT_TTL): Description(text="TTL output at each volume start"),
            UIParam.user_value(USER_SLOT_TTL): DropdownFloatParam(
                value=0.0, min=0.0, max=1.0, incr=1.0, unit="", options=[0.0, 1.0], validate=Validate.NONE,
            ),
            UIParam.user_name(USER_SLOT_PHASE_CORRECTION): Description(
                text="Acquire blip-nulled EPI navigator for odd/even phase correction"
            ),
            UIParam.user_value(USER_SLOT_PHASE_CORRECTION): DropdownFloatParam(
                value=0.0, min=0.0, max=1.0, incr=1.0, unit="", options=[0.0, 1.0], validate=Validate.NONE,
            ),
            UIParam.user_name(USER_SLOT_REVERSE_PE): Description(
                text="Acquire one b=0 volume with reversed phase-encode polarity"
            ),
            UIParam.user_value(USER_SLOT_REVERSE_PE): DropdownFloatParam(
                value=0.0, min=0.0, max=1.0, incr=1.0, unit="", options=[0.0, 1.0], validate=Validate.NONE,
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
        if not (0.0 < cfg.flip_deg <= 90.0):
            return {"valid": False, "duration": None, "info": "Flip angle must be in (0, 90] deg"}
        if cfg.nx_ro < 1 or cfg.ny_pe < 1 or cfg.npar < 1:
            return {"valid": False, "duration": None, "info": "NX, NY, and NSLICES (partitions) must be >= 1"}
        if cfg.bandwidth_hz_px <= 0.0:
            return {"valid": False, "duration": None, "info": "Bandwidth must be > 0"}
        if cfg.etl < 1:
            return {"valid": False, "duration": None, "info": "ETL must be >= 1"}
        if cfg.b_value_s_mm2 < 0.0:
            return {"valid": False, "duration": None, "info": "Diffusion b-value must be >= 0"}
        if cfg.num_frames < 1:
            return {"valid": False, "duration": None, "info": "NUM_FRAMES must be >= 1"}
        if cfg.spsp and cfg.spsp_bandwidth_hz <= 0.0:
            return {"valid": False, "duration": None, "info": "SPSP bandwidth must be > 0"}

        timing = _compute_timing(opts=opts, cfg=cfg, strict=True)
        if timing is None:
            return {
                "valid": False,
                "duration": None,
                "info": "TE too short for the 90/180/crushers/diffusion/prephasers, or TR too short",
            }

        n_shots = _n_shots(cfg)
        par_loop = design.make_cartesian_sampling((cfg.nx_ro, cfg.npar), acceleration=cfg.rz)
        n_dirs = cfg.n_directions if cfg.b_value_s_mm2 > 0.0 else 1
        duration_s = cfg.tr_s * float(n_shots) * float(len(par_loop)) * float(n_dirs) * cfg.num_frames
        if cfg.phase_correction:
            duration_s += cfg.tr_s
        if cfg.reverse_pe:
            duration_s += cfg.tr_s * float(n_shots) * float(len(par_loop))
        if cfg.ttl_output:
            duration_s += 1e-3 * cfg.num_frames
        return {"valid": True, "duration": duration_s, "info": f"TA = {duration_s:.2f} s"}

    def make_sequence(self, opts: pp.Opts, protocol: dict[str, dict], output_path: str) -> None:
        prot = dict_to_protocol(protocol)
        cfg = _read_protocol(prot)

        timing = _compute_timing(opts=opts, cfg=cfg, strict=False)

        pulse = timing["pulse"]
        epi = timing["epi"]
        navigator = timing["navigator"]
        reverse_epi = timing["reverse_epi"]
        diffusion = timing["diffusion"]
        te_delay_s = timing["te_delay_s"]
        tr_delay_s = timing["tr_delay_s"]

        te_delay = pp.make_delay(te_delay_s) if te_delay_s > 0.0 else None
        tr_delay = pp.make_delay(tr_delay_s) if tr_delay_s > 0.0 else None

        seq = pp.Sequence(opts)

        shot_starts = list(range(0, cfg.ny_pe, cfg.etl))
        par_loop = design.make_cartesian_sampling((cfg.nx_ro, cfg.npar), acceleration=cfg.rz)
        n_directions = cfg.n_directions if cfg.b_value_s_mm2 > 0.0 else 1
        rotations = (
            design.make_noncartesian_projection_sampling(
            (cfg.nx_ro, cfg.nx_ro, cfg.nx_ro), views=cfg.n_directions
        ).to_rotations()
            if diffusion is not None
            else [None]
        )

        ttl = pp.make_digital_output_pulse("ext1", duration=1e-3, system=opts) if cfg.ttl_output else None
        frames = design.make_counter_loop(cfg.num_frames, label="PHS")
        image_labels = (pp.make_label(type="SET", label="SET", value=0),)
        if cfg.phase_correction:
            for block_idx, block in enumerate(pulse):
                labels = (pp.make_label(type="SET", label="SLC", value=0),) if block_idx == 0 else ()
                seq.add_block(*block, *labels)
            if te_delay is not None:
                seq.add_block(te_delay)
            navigator.set_state(
                lin_idx=0,
                par_idx=0,
                adc_labels=(
                    pp.make_label(type="SET", label="NAV", value=1),
                    pp.make_label(type="SET", label="REF", value=1),
                ),
            )
            for block in navigator:
                seq.add_block(*block)
            if tr_delay is not None:
                seq.add_block(tr_delay)
        # The frame counter rides on the first block of every TR rather than on
        # a lead-in block of its own. A block that plays once per frame has no
        # counterpart in the TRs that follow it, so TR-period detection finds no
        # period at all and falls back to "the whole frame is one TR" -- which
        # then costs a waveform buffer proportional to the frame duration.
        # Re-SETting a counter that already holds the right value is free.
        for frame in range(len(frames)):
            (phase_label,) = frames.labels(frame)
            frame_labels = (phase_label,)
            if ttl is not None:
                # A volume-start trigger genuinely is a once-per-frame block, so
                # mark it as one: ONCE=1 opens the section and the ONCE=0 that
                # every TR carries closes it, leaving the TRs identical.
                seq.add_block(ttl, pp.make_label(type="SET", label="ONCE", value=1))
                frame_labels = (phase_label, pp.make_label(type="SET", label="ONCE", value=0))
            tr_labels = (pp.make_label(type="SET", label="SLC", value=0), *frame_labels)
            for rotation in rotations[:n_directions]:
                for par_shot in par_loop:
                    for ky_start in shot_starts:
                        if diffusion is not None:
                            diffusion.set_state(b_value=cfg.b_value_s_mm2, rotation=rotation)
                            for block in diffusion:
                                seq.add_block(*block)
                        for block_idx, block in enumerate(pulse):
                            seq.add_block(*block, *(tr_labels if block_idx == 0 else ()))
                        if te_delay is not None:
                            seq.add_block(te_delay)
                        epi.set_state(
                            lin_idx=ky_start, par_idx=int(par_shot[0, 0]), adc_labels=image_labels
                        )
                        for block in epi:
                            seq.add_block(*block)

                        if tr_delay is not None:
                            seq.add_block(tr_delay)

        if cfg.reverse_pe:
            reverse_labels = (pp.make_label(type="SET", label="SET", value=1),)
            for par_shot in par_loop:
                for ky_start in shot_starts:
                    for block_idx, block in enumerate(pulse):
                        labels = (pp.make_label(type="SET", label="SLC", value=0),) if block_idx == 0 else ()
                        seq.add_block(*block, *labels)
                    if te_delay is not None:
                        seq.add_block(te_delay)
                    reverse_epi.set_state(
                        lin_idx=ky_start, par_idx=int(par_shot[0, 0]), adc_labels=reverse_labels
                    )
                    for block in reverse_epi:
                        seq.add_block(*block)
                    if tr_delay is not None:
                        seq.add_block(tr_delay)

        seq.set_definition("Name", "epi_3d")
        seq.set_definition("FOV", [cfg.fov_ro_m, cfg.fov_pe_m, cfg.slice_spacing_m * cfg.npar])
        seq.set_definition("TE", cfg.te_s)
        seq.set_definition("TR", cfg.tr_s)
        seq.set_definition("Flip", cfg.flip_deg)
        seq.set_definition("ImagingMode", "3d")
        seq.set_definition("ETL", cfg.etl)
        seq.set_definition("NumShots", len(shot_starts))
        seq.set_definition("RampSample", cfg.ramp_sample)
        seq.set_definition("BValue", cfg.b_value_s_mm2)
        seq.set_definition("DiffusionDirections", n_directions)
        seq.set_definition("BandwidthHzPerPx", cfg.bandwidth_hz_px)
        seq.set_definition("Rz", cfg.rz)
        seq.set_definition("Nx", cfg.nx_ro)
        seq.set_definition("Ny", cfg.ny_pe)
        seq.set_definition("NumPartitions", cfg.npar)
        seq.set_definition("NumFrames", cfg.num_frames)
        seq.set_definition("TTLExternalOutput", cfg.ttl_output)
        seq.set_definition("SPSPExcitation", cfg.spsp)
        seq.set_definition("EpiPhaseCorrectionNavigator", cfg.phase_correction)
        seq.set_definition("ReversePhaseEncodeReference", cfg.reverse_pe)
        pio.write(seq, output=output_path, check_timing=False)


def _n_shots(cfg: _Config) -> int:
    return len(range(0, cfg.ny_pe, cfg.etl))


class _Config:
    __slots__ = (
        "b_value_s_mm2",
        "bandwidth_hz_px",
        "etl",
        "flip_deg",
        "fov_pe_m",
        "fov_ro_m",
        "n_directions",
        "npar",
        "num_frames",
        "nx_ro",
        "ny_pe",
        "phase_correction",
        "ramp_sample",
        "reverse_pe",
        "rz",
        "slab_thickness_m",
        "slice_spacing_m",
        "spsp",
        "spsp_bandwidth_hz",
        "te_s",
        "tr_s",
        "ttl_output",
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
    cfg.etl = params.param_int_optional(prot, UIParam.ETL, cfg.ny_pe)
    cfg.rz = max(1, round(params.param_float_optional(prot, UIParam.RZ, 1.0)))
    cfg.bandwidth_hz_px = params.param_float_optional(prot, UIParam.BANDWIDTH, 250_000.0)
    cfg.ramp_sample = params.user_float(prot, USER_SLOT_RAMP_SAMPLE, 0.0) >= 0.5
    cfg.b_value_s_mm2 = params.param_float_optional(prot, UIParam.DIFFUSION_BVALUES, 0.0)
    cfg.n_directions = params.param_int_optional(prot, UIParam.DIFFUSION_DIRECTIONS, 3)
    cfg.num_frames = params.param_int_optional(prot, UIParam.NUM_FRAMES, 1)
    cfg.ttl_output = params.user_float(prot, USER_SLOT_TTL, 0.0) >= 0.5
    cfg.spsp = params.user_float(prot, USER_SLOT_SPSP, 0.0) >= 0.5
    cfg.spsp_bandwidth_hz = params.user_float(prot, USER_SLOT_SPSP_BW, 250.0)
    cfg.phase_correction = params.user_float(prot, USER_SLOT_PHASE_CORRECTION, 0.0) >= 0.5
    cfg.reverse_pe = params.user_float(prot, USER_SLOT_REVERSE_PE, 0.0) >= 0.5
    return cfg


def _compute_timing(opts: pp.Opts, cfg: _Config, strict: bool):
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
    n_lines = min(cfg.etl, cfg.ny_pe)
    mask = np.column_stack((np.arange(n_lines, dtype=int), np.zeros(n_lines, dtype=int)))
    epi = design.make_epi_readout(
        opts,
        (cfg.fov_ro_m, cfg.fov_pe_m, cfg.slice_spacing_m * cfg.npar),
        (cfg.nx_ro, cfg.ny_pe, cfg.npar),
        _n_shots(cfg),
        mask,
        bandwidth_hz_px=cfg.bandwidth_hz_px,
        ramp_sample=cfg.ramp_sample,
    )
    navigator = design.make_epi_readout(
        opts,
        (cfg.fov_ro_m, cfg.fov_pe_m, cfg.slice_spacing_m * cfg.npar),
        (cfg.nx_ro, cfg.ny_pe, cfg.npar),
        1,
        np.zeros((epi.etl, 2), dtype=int),
        bandwidth_hz_px=cfg.bandwidth_hz_px,
        ramp_sample=cfg.ramp_sample,
        blip_duration_s=epi.blip_duration_s,
    )
    reverse_epi = design.make_epi_readout(
        opts,
        (cfg.fov_ro_m, cfg.fov_pe_m, cfg.slice_spacing_m * cfg.npar),
        (cfg.nx_ro, cfg.ny_pe, cfg.npar),
        _n_shots(cfg),
        np.column_stack((np.arange(epi.etl - 1, -1, -1, dtype=int), np.zeros(epi.etl, dtype=int))),
        bandwidth_hz_px=cfg.bandwidth_hz_px,
        ramp_sample=cfg.ramp_sample,
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

    d_pulse = sum(pp.calc_duration(*block) for block in pulse)
    rf_center_s = pp.calc_rf_center(pulse.rf)[0] + pulse.rf.delay
    first_echo_s = epi.duration - (epi.etl - 0.5) * epi.esp
    raster = opts.block_duration_raster
    te_delay_s = round((cfg.te_s - (d_pulse - rf_center_s) - first_echo_s) / raster) * raster
    if te_delay_s < -1e-9 and strict:
        return None
    te_delay_s = max(0.0, te_delay_s)
    prep_duration = 0.0 if diffusion is None else diffusion.duration
    min_block_s = prep_duration + d_pulse + te_delay_s + epi.duration
    n_inner = 1 if strict else len(design.make_cartesian_sampling((cfg.nx_ro, cfg.npar), acceleration=cfg.rz))
    tr_delay_s = round((cfg.tr_s - n_inner * min_block_s) / raster) * raster
    if tr_delay_s < -1e-9 and strict:
        return None
    if tr_delay_s < 0.0:
        tr_delay_s = 0.0

    return {
        "pulse": pulse,
        "epi": epi,
        "navigator": navigator,
        "reverse_epi": reverse_epi,
        "diffusion": diffusion,
        "te_delay_s": te_delay_s,
        "tr_delay_s": tr_delay_s,
        "min_block_s": min_block_s,
    }


PLUGIN = Epi3DPulseqSequence()


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
    ('--phase-fov-mm', UIParam.PHASE_FOV, float, ""),
    ('--slab-thickness-mm', UIParam.SLICE_THICKNESS, float, ""),
    ('--partition-spacing-mm', UIParam.SLICE_SPACING, float, ""),
    ('--nx', UIParam.NX, int, ""),
    ('--ny', UIParam.NY, int, ""),
    ('--npartitions', UIParam.NSLICES, int, ""),
    ('--etl', UIParam.ETL, int, ""),
    ('--rz', UIParam.RZ, float, ""),
    ('--bandwidth-hz-px', UIParam.BANDWIDTH, float, ""),
    ('--ramp-sample', UIParam.user_value(USER_SLOT_RAMP_SAMPLE), ("const", 1.0), ""),
    ('--bvalue', UIParam.DIFFUSION_BVALUES, float, ""),
    ('--directions', UIParam.DIFFUSION_DIRECTIONS, int, ""),
    ('--num-frames', UIParam.NUM_FRAMES, int, ""),
    ('--ttl-output', UIParam.user_value(USER_SLOT_TTL), ("const", 1.0), ""),
    ('--phase-correction', UIParam.user_value(USER_SLOT_PHASE_CORRECTION), ("const", 1.0), ""),
    ('--reverse-pe', UIParam.user_value(USER_SLOT_REVERSE_PE), ("const", 1.0), ""),
    ('--spsp', UIParam.user_value(USER_SLOT_SPSP), ("const", 1.0), ""),
    ('--spsp-bandwidth-hz', UIParam.user_value(USER_SLOT_SPSP_BW), float, ""),
]

if __name__ == "__main__":
    raise SystemExit(
        run_cli(
            PLUGIN,
            sys.argv[1:],
            arg_map=_ARG_MAP,
            description='Generate a 3D spin-echo EPI .seq offline.',
            default_output='epi_3d.seq',
        )
    )
