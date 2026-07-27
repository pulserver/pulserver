"""Standalone 2D single-/multi-shot spin-echo EPI sequence plugin.

This module implements the three mandatory module-level entry points
required by the bridge dispatcher (see ``gre_2d.py`` for the Cartesian GRE
family this parallels, and ``_gre_common.py`` / ``_epi_common.py`` /
``pulserver`` / ``_diffusion_common.py`` for the shared building
blocks):

- ``get_default_protocol(opts)``  — return the initial CV/protocol dictionary.
- ``validate_protocol(opts, protocol)``  — validate the current protocol and
  return a ``{"valid": bool, "duration": float | None, "info": str}`` dict.
- ``make_sequence(opts, protocol, output_path)``  — synthesise the sequence
  and write it to *output_path* using :func:`pulserver.io.write`.

Structure: 90 (slice-selective) - crusher - [diffusion gradient] - 180
(slice-selective, same slice) - crusher - [diffusion gradient] -
prephasers - blipped EPI readout train. The 180 is kept at exactly
``TE / 2`` (a hard requirement for correct refocusing, not a convention —
see ``_epi_common.pre_post_180_delays``). ``TE`` is defined as the time
from the 90's center to the FIRST readout line's ADC center — with a
linear (non-centric) blip order starting at one k-space edge, this first
line genuinely is the spin-echo peak, and later lines drift toward
T2*-weighting, which is standard single-shot EPI physics.

``ETL`` (native GE, "lines per shot") splits ``NY`` into
``ceil(NY / ETL)`` shots, each with its own full 90-180-train (sequential,
not interleaved, k-space partitioning). Ramp sampling is an ``opuser``
toggle (no native GE counterpart) — see ``_epi_common.py`` for exactly what
it does and does not correct for. ``DIFFUSION_BVALUES``/
``DIFFUSION_DIRECTIONS`` are native GE variables reused directly;
``b <= 0`` (the default) means no diffusion weighting, so no separate
enable flag is needed.

Not implemented in this pass (noted, not silently dropped): optional
inversion/T2-prep before the 90 — same proven pattern as
``gre_mprage_radial_2d.py``, a mechanical follow-up, deferred to keep pace
with the rest of the non-Cartesian-adjacent zoo backlog (EPI/FSE/ZTE).

Usage
-----
Register the plugin in the interpreter tree by symlinking (or copying) this
file to the ``sequences/src/`` directory and creating a numbered alias::

    cp epi_2d.py pulserver-interpreter/package/pulserver/sequences/src/epi_2d.py
    ln -sf src/epi_2d.py pulserver-interpreter/package/pulserver/sequences/sequence13.py
"""

from __future__ import annotations

import sys

import numpy as np
import pulserver.design as design
import pulserver.io as pio
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

USER_SLOT_RAMP_SAMPLE = 0
USER_SLOT_FAT_SAT = 1
USER_SLOT_B0 = 2
USER_SLOT_TTL = 3
USER_SLOT_PHASE_CORRECTION = 4
USER_SLOT_REVERSE_PE = 5
USER_SLOT_SMS_REFERENCE = 6


class Epi2DPulseqSequence(Sequence):
    """Generate a 2D single-/multi-shot spin-echo EPI sequence."""

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
            UIParam.MULTIBAND: TypeinFloatParam(
                value=1.0, min=1.0, max=16.0, incr=1.0, unit="", validate=Validate.NONE,
            ),
            UIParam.NUM_FRAMES: DropdownIntParam(
                value=1, min=1, max=10000, incr=1, options=[1, 10, 50, 100], validate=Validate.NONE,
            ),
            UIParam.ETL: DropdownIntParam(
                value=64, min=1, max=512, incr=1, options=[16, 32, 64, 128, 256], validate=Validate.NONE,
            ),
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
            UIParam.SWAP_PHASE_FREQ: BoolParam(value=False, validate=Validate.NONE),
            UIParam.SEQUENCE_TYPE: make_enum_param(UIParam.SEQUENCE_TYPE, SequenceType.SPIN_ECHO),
            UIParam.user_value(USER_SLOT_RAMP_SAMPLE): DropdownFloatParam(
                value=0.0, min=0.0, max=1.0, incr=1.0, unit="", options=[0.0, 1.0], validate=Validate.NONE,
            ),
            UIParam.user_name(USER_SLOT_FAT_SAT): Description(text="Fat saturation before each EPI shot"),
            UIParam.user_value(USER_SLOT_FAT_SAT): DropdownFloatParam(
                value=0.0, min=0.0, max=1.0, incr=1.0, unit="", options=[0.0, 1.0], validate=Validate.NONE,
            ),
            UIParam.user_name(USER_SLOT_B0): Description(text="Main field for fat frequency"),
            UIParam.user_value(USER_SLOT_B0): TypeinFloatParam(
                value=3.0, min=0.1, max=14.0, incr=0.1, unit="T", validate=Validate.NONE,
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
            UIParam.user_name(USER_SLOT_SMS_REFERENCE): Description(
                text="Acquire single-band reference volume for SMS Slice-GRAPPA"
            ),
            UIParam.user_value(USER_SLOT_SMS_REFERENCE): DropdownFloatParam(
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
        if not (0.0 < cfg.flip_deg <= 90.0):
            return {"valid": False, "duration": None, "info": "Flip angle must be in (0, 90] deg"}
        if cfg.nx_ro < 1 or cfg.ny_pe < 1 or cfg.nslices < 1:
            return {"valid": False, "duration": None, "info": "NX, NY, and NSLICES must be >= 1"}
        if cfg.bandwidth_hz_px <= 0.0:
            return {"valid": False, "duration": None, "info": "Bandwidth must be > 0"}
        if cfg.etl < 1:
            return {"valid": False, "duration": None, "info": "ETL must be >= 1"}
        if cfg.b_value_s_mm2 < 0.0:
            return {"valid": False, "duration": None, "info": "Diffusion b-value must be >= 0"}
        if cfg.num_frames < 1:
            return {"valid": False, "duration": None, "info": "NUM_FRAMES must be >= 1"}
        if not np.isclose(cfg.multiband_value, cfg.multiband) or cfg.multiband < 1 or cfg.nslices % cfg.multiband:
            return {"valid": False, "duration": None, "info": "Multiband must be an integer factor of NSLICES"}
        if cfg.sms_reference and cfg.multiband == 1:
            return {"valid": False, "duration": None, "info": "SMS reference requires MULTIBAND > 1"}

        timing = _compute_timing(opts=opts, cfg=cfg, strict=True)
        if timing is None:
            return {
                "valid": False,
                "duration": None,
                "info": "TE too short for the 90/180/crushers/diffusion/prephasers, or TR too short",
            }

        n_shots = _n_shots(cfg)
        n_dirs = cfg.n_directions if cfg.b_value_s_mm2 > 0.0 else 1
        duration_s = cfg.tr_s * float(n_shots) * float(cfg.n_slice_groups) * float(n_dirs) * cfg.num_frames
        if cfg.phase_correction:
            duration_s += cfg.tr_s * float(cfg.n_slice_groups)
        if cfg.reverse_pe:
            duration_s += cfg.tr_s * float(n_shots) * float(cfg.n_slice_groups)
        if cfg.sms_reference:
            duration_s += cfg.tr_s * float(n_shots) * float(cfg.nslices)
        if cfg.ttl_output:
            duration_s += 1e-3 * cfg.num_frames
        return {"valid": True, "duration": duration_s, "info": f"TA = {duration_s:.2f} s"}

    def make_sequence(self, opts: pp.Opts, protocol: dict[str, dict], output_path: str) -> None:
        prot = dict_to_protocol(protocol)
        cfg = _read_protocol(prot)

        timing = _compute_timing(opts=opts, cfg=cfg, strict=False)

        pulse = timing["pulse"]
        pulses = timing["pulses"]
        epi = timing["epi"]
        navigator = timing["navigator"]
        reverse_epi = timing["reverse_epi"]
        reference_pulse = timing["reference_pulse"]
        reference_te_delay_s = timing["reference_te_delay_s"]
        reference_tr_delay_s = timing["reference_tr_delay_s"]
        diffusion = timing["diffusion"]
        fat_sat = timing["fat_sat"]
        te_delay_s = timing["te_delay_s"]
        tr_delay_s = timing["tr_delay_s"]

        te_delay = pp.make_delay(te_delay_s) if te_delay_s > 0.0 else None
        tr_delay = pp.make_delay(tr_delay_s) if tr_delay_s > 0.0 else None
        reference_te_delay = pp.make_delay(reference_te_delay_s) if reference_te_delay_s > 0.0 else None
        reference_tr_delay = pp.make_delay(reference_tr_delay_s) if reference_tr_delay_s > 0.0 else None

        seq = pp.Sequence(opts)

        slice_step_m = cfg.slice_spacing_m if cfg.nslices > 1 else 0.0
        shot_starts = list(range(0, cfg.ny_pe, cfg.etl))
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
        sms_label = (pp.make_label(type="SET", label="SMS", value=1),) if cfg.multiband > 1 else ()

        if cfg.phase_correction:
            navigator_labels = (
                pp.make_label(type="SET", label="NAV", value=1),
                pp.make_label(type="SET", label="REF", value=1),
                *sms_label,
            )
            for group in range(cfg.n_slice_groups):
                pulse = pulses[group]
                if cfg.multiband == 1:
                    slice_offset_m = (group - 0.5 * (cfg.n_slice_groups - 1)) * slice_step_m
                    pulse.set_state(freq_offset_hz=pulse.gradients[0].amplitude * slice_offset_m)
                for block_idx, block in enumerate(pulse):
                    labels = (pp.make_label(type="SET", label="SLC", value=group),) if block_idx == 0 else ()
                    seq.add_block(*block, *labels)
                if te_delay is not None:
                    seq.add_block(te_delay)
                navigator.set_state(lin_idx=0, adc_labels=navigator_labels)
                for block in navigator:
                    seq.add_block(*block)
                if tr_delay is not None:
                    seq.add_block(tr_delay)

        if cfg.sms_reference:
            for slice_index in range(cfg.nslices):
                slice_offset_m = (slice_index - 0.5 * (cfg.nslices - 1)) * slice_step_m
                reference_pulse.set_state(
                    freq_offset_hz=reference_pulse.gradients[0].amplitude * slice_offset_m
                )
                for ky_start in shot_starts:
                    for block_idx, block in enumerate(reference_pulse):
                        labels = (pp.make_label(type="SET", label="SLC", value=slice_index),) if block_idx == 0 else ()
                        seq.add_block(*block, *labels)
                    if reference_te_delay is not None:
                        seq.add_block(reference_te_delay)
                    epi.set_state(
                        lin_idx=ky_start,
                        adc_labels=(pp.make_label(type="SET", label="REF", value=1),),
                    )
                    for block in epi:
                        seq.add_block(*block)
                    if reference_tr_delay is not None:
                        seq.add_block(reference_tr_delay)

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
            for rotation in rotations[:n_directions]:
                for group in range(cfg.n_slice_groups):
                    slice_offset_m = (group - 0.5 * (cfg.n_slice_groups - 1)) * slice_step_m
                    pulse = pulses[group]
                    tr_labels = (pp.make_label(type="SET", label="SLC", value=group), *frame_labels)

                    for ky_start in shot_starts:
                        if fat_sat is not None:
                            for block in fat_sat:
                                seq.add_block(*block)
                        if diffusion is not None:
                            diffusion.set_state(b_value=cfg.b_value_s_mm2, rotation=rotation)
                            for block in diffusion:
                                seq.add_block(*block)
                        if cfg.multiband == 1:
                            pulse.set_state(
                                freq_offset_hz=pulse.gradients[0].amplitude * slice_offset_m
                            )
                        for block_idx, block in enumerate(pulse):
                            seq.add_block(*block, *(tr_labels if block_idx == 0 else ()))
                        if te_delay is not None:
                            seq.add_block(te_delay)
                        epi.set_state(lin_idx=ky_start, adc_labels=(*image_labels, *sms_label))
                        for block in epi:
                            seq.add_block(*block)

                        if tr_delay is not None:
                            seq.add_block(tr_delay)

        if cfg.reverse_pe:
            reverse_labels = (pp.make_label(type="SET", label="SET", value=1), *sms_label)
            for group in range(cfg.n_slice_groups):
                slice_offset_m = (group - 0.5 * (cfg.n_slice_groups - 1)) * slice_step_m
                pulse = pulses[group]
                if cfg.multiband == 1:
                    pulse.set_state(freq_offset_hz=pulse.gradients[0].amplitude * slice_offset_m)
                for ky_start in shot_starts:
                    for block_idx, block in enumerate(pulse):
                        labels = (pp.make_label(type="SET", label="SLC", value=group),) if block_idx == 0 else ()
                        seq.add_block(*block, *labels)
                    if te_delay is not None:
                        seq.add_block(te_delay)
                    reverse_epi.set_state(lin_idx=ky_start, adc_labels=reverse_labels)
                    for block in reverse_epi:
                        seq.add_block(*block)
                    if tr_delay is not None:
                        seq.add_block(tr_delay)

        seq.set_definition("Name", "epi_2d")
        seq.set_definition(
            "FOV",
            [cfg.fov_ro_m, cfg.fov_pe_m, slice_step_m * cfg.nslices if cfg.nslices > 1 else cfg.slice_thickness_m],
        )
        seq.set_definition("TE", cfg.te_s)
        seq.set_definition("TR", cfg.tr_s)
        seq.set_definition("Flip", cfg.flip_deg)
        seq.set_definition("ImagingMode", "2d")
        seq.set_definition("ETL", cfg.etl)
        seq.set_definition("NumShots", len(shot_starts))
        seq.set_definition("RampSample", cfg.ramp_sample)
        seq.set_definition("BValue", cfg.b_value_s_mm2)
        seq.set_definition("DiffusionDirections", n_directions)
        seq.set_definition("BandwidthHzPerPx", cfg.bandwidth_hz_px)
        seq.set_definition("Nx", cfg.nx_ro)
        seq.set_definition("Ny", cfg.ny_pe)
        seq.set_definition("NumSlices", cfg.nslices)
        seq.set_definition("MultibandFactor", cfg.multiband)
        seq.set_definition("CaipiBlipArea", timing["caipi_blip_area"])
        seq.set_definition("NumSliceGroups", cfg.n_slice_groups)
        seq.set_definition("NumFrames", cfg.num_frames)
        seq.set_definition("FatSaturation", cfg.fat_sat)
        seq.set_definition("TTLExternalOutput", cfg.ttl_output)
        seq.set_definition("EpiPhaseCorrectionNavigator", cfg.phase_correction)
        seq.set_definition("ReversePhaseEncodeReference", cfg.reverse_pe)
        seq.set_definition("SmsSingleBandReference", cfg.sms_reference)
        pio.write(seq, output=output_path, check_timing=False)


def _n_shots(cfg: _Config) -> int:
    return len(range(0, cfg.ny_pe, cfg.etl))


class _Config:
    __slots__ = (
        "b0_t",
        "b_value_s_mm2",
        "bandwidth_hz_px",
        "etl",
        "fat_sat",
        "flip_deg",
        "fov_pe_m",
        "fov_ro_m",
        "multiband",
        "multiband_value",
        "n_directions",
        "n_slice_groups",
        "nslices",
        "num_frames",
        "nx_ro",
        "ny_pe",
        "phase_correction",
        "ramp_sample",
        "reverse_pe",
        "slice_spacing_m",
        "slice_thickness_m",
        "sms_reference",
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
    cfg.slice_thickness_m = params.param_float(prot, UIParam.SLICE_THICKNESS) * 1e-3
    cfg.slice_spacing_m = params.param_float(prot, UIParam.SLICE_SPACING) * 1e-3
    cfg.nx_ro = params.param_int(prot, UIParam.NX)
    cfg.ny_pe = params.param_int(prot, UIParam.NY)
    cfg.nslices = params.param_int(prot, UIParam.NSLICES)
    cfg.etl = params.param_int_optional(prot, UIParam.ETL, cfg.ny_pe)
    cfg.bandwidth_hz_px = params.param_float_optional(prot, UIParam.BANDWIDTH, 250_000.0)
    cfg.ramp_sample = params.user_float(prot, USER_SLOT_RAMP_SAMPLE, 0.0) >= 0.5
    cfg.b_value_s_mm2 = params.param_float_optional(prot, UIParam.DIFFUSION_BVALUES, 0.0)
    cfg.n_directions = params.param_int_optional(prot, UIParam.DIFFUSION_DIRECTIONS, 3)
    cfg.multiband_value = params.param_float_optional(prot, UIParam.MULTIBAND, 1.0)
    cfg.multiband = round(cfg.multiband_value)
    cfg.n_slice_groups = cfg.nslices // max(1, cfg.multiband)
    cfg.num_frames = params.param_int_optional(prot, UIParam.NUM_FRAMES, 1)
    cfg.fat_sat = params.user_float(prot, USER_SLOT_FAT_SAT, 0.0) >= 0.5
    cfg.b0_t = params.user_float(prot, USER_SLOT_B0, 3.0)
    cfg.ttl_output = params.user_float(prot, USER_SLOT_TTL, 0.0) >= 0.5
    cfg.phase_correction = params.user_float(prot, USER_SLOT_PHASE_CORRECTION, 0.0) >= 0.5
    cfg.reverse_pe = params.user_float(prot, USER_SLOT_REVERSE_PE, 0.0) >= 0.5
    cfg.sms_reference = params.user_float(prot, USER_SLOT_SMS_REFERENCE, 0.0) >= 0.5
    return cfg


def _compute_timing(opts: pp.Opts, cfg: _Config, strict: bool):
    caipi_blip_area = 0.0
    if cfg.multiband > 1:
        sms_spacing = cfg.n_slice_groups * cfg.slice_spacing_m
        caipi_blip_area = 1.0 / (cfg.multiband * sms_spacing)
        pulses = tuple(
            design.make_pins_slice_selective_pulse(
                np.deg2rad(cfg.flip_deg), cfg.slice_thickness_m, cfg.multiband,
                sms_spacing,
                include_center=False,
                center_offset=(group - 0.5 * (cfg.n_slice_groups - 1)) * cfg.slice_spacing_m,
                system=opts,
            )
            for group in range(cfg.n_slice_groups)
        )
        pulse = pulses[0]
    else:
        pulse = design.make_slice_selective_pulse(
            np.deg2rad(cfg.flip_deg), cfg.slice_thickness_m, system=opts
        )
        # One object, indexed per group and re-configured via set_state() before each use
        # (see the cfg.multiband == 1 branch above) -- not one pulse per slice group.
        pulses = (pulse,) * cfg.n_slice_groups
    mask = np.arange(min(cfg.etl, cfg.ny_pe), dtype=int)
    epi = design.make_epi_readout(
        opts,
        (cfg.fov_ro_m, cfg.fov_pe_m),
        (cfg.nx_ro, cfg.ny_pe),
        _n_shots(cfg),
        mask,
        bandwidth_hz_px=cfg.bandwidth_hz_px,
        ramp_sample=cfg.ramp_sample,
        caipi_axis="z" if cfg.multiband > 1 else None,
        caipi_blip_area=caipi_blip_area,
    )
    navigator = design.make_epi_readout(
        opts,
        (cfg.fov_ro_m, cfg.fov_pe_m),
        (cfg.nx_ro, cfg.ny_pe),
        1,
        np.zeros(epi.etl, dtype=int),
        bandwidth_hz_px=cfg.bandwidth_hz_px,
        ramp_sample=cfg.ramp_sample,
        blip_duration_s=epi.blip_duration_s,
    )
    reverse_epi = design.make_epi_readout(
        opts,
        (cfg.fov_ro_m, cfg.fov_pe_m),
        (cfg.nx_ro, cfg.ny_pe),
        _n_shots(cfg),
        np.arange(epi.etl - 1, -1, -1, dtype=int),
        bandwidth_hz_px=cfg.bandwidth_hz_px,
        ramp_sample=cfg.ramp_sample,
        caipi_axis="z" if cfg.multiband > 1 else None,
        caipi_blip_area=caipi_blip_area,
    )
    reference_pulse = design.make_slice_selective_pulse(
        np.deg2rad(cfg.flip_deg), cfg.slice_thickness_m, system=opts
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
    fat_sat = (
        design.make_fat_saturation_pulse(
            b0=cfg.b0_t, voxel_size=cfg.fov_ro_m / cfg.nx_ro, system=opts
        )
        if cfg.fat_sat
        else None
    )

    d_pulse = sum(pp.calc_duration(*block) for block in pulse)
    rf_center_s = pp.calc_rf_center(pulse.rf)[0] + pulse.rf.delay
    first_echo_s = epi.duration - (epi.etl - 0.5) * epi.esp
    raster = opts.block_duration_raster
    te_delay_s = round((cfg.te_s - (d_pulse - rf_center_s) - first_echo_s) / raster) * raster
    if te_delay_s < -1e-9 and strict:
        return None
    te_delay_s = max(0.0, te_delay_s)
    prep_duration = (0.0 if diffusion is None else diffusion.duration) + (0.0 if fat_sat is None else fat_sat.duration)
    min_block_s = prep_duration + d_pulse + te_delay_s + epi.duration
    tr_delay_s = round((cfg.tr_s - min_block_s) / raster) * raster
    if tr_delay_s < -1e-9 and strict:
        return None
    if tr_delay_s < 0.0:
        tr_delay_s = 0.0

    d_reference = sum(pp.calc_duration(*block) for block in reference_pulse)
    reference_center_s = pp.calc_rf_center(reference_pulse.rf)[0] + reference_pulse.rf.delay
    reference_te_delay_s = round((cfg.te_s - (d_reference - reference_center_s) - first_echo_s) / raster) * raster
    if reference_te_delay_s < -1e-9 and strict:
        return None
    reference_te_delay_s = max(0.0, reference_te_delay_s)
    reference_min_block_s = d_reference + reference_te_delay_s + epi.duration
    reference_tr_delay_s = round((cfg.tr_s - reference_min_block_s) / raster) * raster
    if reference_tr_delay_s < -1e-9 and strict:
        return None
    reference_tr_delay_s = max(0.0, reference_tr_delay_s)

    return {
        "pulse": pulse,
        "pulses": pulses,
        "epi": epi,
        "navigator": navigator,
        "reverse_epi": reverse_epi,
        "reference_pulse": reference_pulse,
        "reference_te_delay_s": reference_te_delay_s,
        "reference_tr_delay_s": reference_tr_delay_s,
        "caipi_blip_area": caipi_blip_area,
        "diffusion": diffusion,
        "fat_sat": fat_sat,
        "te_delay_s": te_delay_s,
        "tr_delay_s": tr_delay_s,
        "min_block_s": min_block_s,
    }


PLUGIN = Epi2DPulseqSequence()


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
    ('--slice-thickness-mm', UIParam.SLICE_THICKNESS, float, ""),
    ('--slice-spacing-mm', UIParam.SLICE_SPACING, float, ""),
    ('--nx', UIParam.NX, int, ""),
    ('--ny', UIParam.NY, int, ""),
    ('--nslices', UIParam.NSLICES, int, ""),
    ('--multiband', UIParam.MULTIBAND, float, ""),
    ('--num-frames', UIParam.NUM_FRAMES, int, ""),
    ('--etl', UIParam.ETL, int, ""),
    ('--bandwidth-hz-px', UIParam.BANDWIDTH, float, ""),
    ('--ramp-sample', UIParam.user_value(USER_SLOT_RAMP_SAMPLE), ("const", 1.0), ""),
    ('--bvalue', UIParam.DIFFUSION_BVALUES, float, ""),
    ('--directions', UIParam.DIFFUSION_DIRECTIONS, int, ""),
    ('--fat-sat', UIParam.user_value(USER_SLOT_FAT_SAT), ("const", 1.0), ""),
    ('--b0-t', UIParam.user_value(USER_SLOT_B0), float, ""),
    ('--ttl-output', UIParam.user_value(USER_SLOT_TTL), ("const", 1.0), ""),
    ('--phase-correction', UIParam.user_value(USER_SLOT_PHASE_CORRECTION), ("const", 1.0), ""),
    ('--reverse-pe', UIParam.user_value(USER_SLOT_REVERSE_PE), ("const", 1.0), ""),
    ('--sms-reference', UIParam.user_value(USER_SLOT_SMS_REFERENCE), ("const", 1.0), ""),
    ('--swap-phase-freq', UIParam.SWAP_PHASE_FREQ, ("const", True), ""),
]

if __name__ == "__main__":
    raise SystemExit(
        run_cli(
            PLUGIN,
            sys.argv[1:],
            arg_map=_ARG_MAP,
            description='Generate a 2D spin-echo EPI .seq offline.',
            default_output='epi_2d.seq',
        )
    )
