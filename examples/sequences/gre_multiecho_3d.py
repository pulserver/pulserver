"""Standalone 3D multi-echo GRE sequence plugin for pulserver.

This module implements the three mandatory module-level entry points
required by the bridge dispatcher (see ``gre_multiecho_2d.py`` for the 2D
sibling and ``gre_3d.py`` for the single-echo 3D sibling — all three of the
newer plugins share low-level building blocks via ``_gre_common.py``):

- ``get_default_protocol(opts)``  — return the initial CV/protocol dictionary.
- ``validate_protocol(opts, protocol)``  — validate the current protocol and
  return a ``{"valid": bool, "duration": float | None, "info": str}`` dict.
- ``make_sequence(opts, protocol, output_path)``  — synthesise the sequence
  and write it to *output_path* using :func:`pulserver.io.write`.

The sequence is a single-slab-excitation 3D multi-echo Cartesian
gradient-echo with a Z phase-encode/partition loop:

- Sinc-pulse slab-selective excitation (2 ms, TBW = 4, apodization = 0.5).
- A shared Y phase encode and Z partition encode per TR (folded together
  with the RF slab-rephase/spoil gradients onto the Z channel — a pypulseq
  block carries only one gradient per channel); each TR plays an entire
  equally-spaced echo train (``NUM_ECHOES`` readouts at an ``ECHO_SPACING``
  ms apart) rather than a single readout.
- Either flyback (unipolar, rewound between echoes) or bipolar (alternating
  polarity, no rewinder, odd echoes reversed) readout trains.
- RF spoiling with a 117 deg quadratic phase increment (once per TR).
- Phase/frequency axis swap, same as ``gre.py``.

Parameters without a native GE UI counterpart (echo spacing, flyback
toggle) are carried as ``opuser`` custom variables.

Usage
-----
Register the plugin in the interpreter tree by symlinking (or copying) this
file to the ``sequences/src/`` directory and creating a numbered alias::

    cp gre_multiecho_3d.py pulserver-interpreter/package/pulserver/sequences/src/gre_multiecho_3d.py
    ln -sf src/gre_multiecho_3d.py pulserver-interpreter/package/pulserver/sequences/sequence3.py
"""

from __future__ import annotations

import sys

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
from pulserver.design import cli, encoding, excitation, params, preparations, readout, sampling, system



USER_SLOT_ECHO_SPACING = 0
USER_SLOT_FLYBACK = 1


class GreMultiEcho3DPulseqSequence(PulseqSequence):
    """Generate a 3D multi-echo Cartesian GRE sequence using pypulseq."""

    def get_default_protocol(self, opts: pp.Opts) -> dict[str, dict]:
        del opts
        protocol = {
            UIParam.TE: DropdownFloatParam(
                value=4.0, min=2.0, max=80.0, incr=0.1, unit="ms",
                options=[3.0, 4.0, 8.0, 15.0, 30.0], validate=Validate.NONE,
            ),
            UIParam.NUM_ECHOES: DropdownIntParam(
                value=4, min=2, max=8, incr=1, options=[2, 4, 6, 8], validate=Validate.NONE,
            ),
            UIParam.TR: DropdownFloatParam(
                value=200.0, min=10.0, max=1000.0, incr=0.1, unit="ms",
                options=[100.0, 150.0, 200.0, 300.0, 500.0], validate=Validate.NONE,
            ),
            UIParam.FLIP: DropdownFloatParam(
                value=12.0, min=1.0, max=90.0, incr=1.0, unit="deg",
                options=[5.0, 12.0, 30.0, 60.0, 90.0], validate=Validate.NONE,
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
                value=system.DEFAULT_BANDWIDTH_HZ_PX, min=5_000.0, max=500_000.0, incr=100.0,
                unit="Hz/px", validate=Validate.NONE,
            ),
            UIParam.SWAP_PHASE_FREQ: BoolParam(value=False, validate=Validate.NONE),
            UIParam.SEQUENCE_TYPE: make_enum_param(UIParam.SEQUENCE_TYPE, SequenceType.GRADIENT_ECHO),
            UIParam.user_name(USER_SLOT_ECHO_SPACING): Description(text="Echo spacing"),
            UIParam.user_value(USER_SLOT_ECHO_SPACING): TypeinFloatParam(
                value=5.0, min=0.5, max=50.0, incr=0.1, unit="ms", validate=Validate.NONE,
            ),
            UIParam.user_name(USER_SLOT_FLYBACK): Description(text="Flyback readout (0=bipolar, 1=flyback)"),
            UIParam.user_value(USER_SLOT_FLYBACK): DropdownFloatParam(
                value=1.0, min=0.0, max=1.0, incr=1.0, unit="", options=[0.0, 1.0], validate=Validate.NONE,
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
        if cfg.num_echoes < 1:
            return {"valid": False, "duration": None, "info": "NUM_ECHOES must be >= 1"}
        if cfg.echo_spacing_s <= 0.0:
            return {"valid": False, "duration": None, "info": "Echo spacing must be > 0"}

        timing = _compute_timing(opts=opts, cfg=cfg, strict=True, n_inner=1)
        if timing is None:
            return {
                "valid": False,
                "duration": None,
                "info": "TE, echo spacing, or TR too short for gradients and readout timing",
            }

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

        sampled_pe = sampling.sampled_lines(cfg.ny_pe, cfg.ry, cfg.acs_lines)
        duration_s = cfg.tr_s * float(len(sampled_pe))
        return {"valid": True, "duration": duration_s, "info": f"TA = {duration_s:.2f} s"}

    def make_sequence(self, opts: pp.Opts, protocol: dict[str, dict], output_path: str) -> None:
        prot = dict_to_protocol(protocol)
        cfg = _read_protocol(prot)

        timing = _compute_timing(opts=opts, cfg=cfg, strict=False)

        gz_reph = timing["gz_reph"]
        gz_spoil = timing["gz_spoil"]
        gz_pe_template = timing["gz_pe_template"]
        echo = timing["echo"]
        gy_template = timing["gy_template"]
        te_delay_s = timing["te_delay_s"]
        tr_delay_s = timing["tr_delay_s"]

        te_delay = pp.make_delay(te_delay_s) if te_delay_s > 0.0 else None
        tr_delay = pp.make_delay(tr_delay_s) if tr_delay_s > 0.0 else None

        seq = ps.Sequence(opts)

        delta_k_pe = 1.0 / cfg.fov_pe_m
        phase_areas = (np.arange(cfg.ny_pe) - 0.5 * cfg.ny_pe) * delta_k_pe
        max_pe_area = float(np.max(np.abs(phase_areas)))
        sampled_pe = sampling.sampled_lines(cfg.ny_pe, cfg.ry, cfg.acs_lines)

        par_areas, max_par_area = encoding.partition_geometry(cfg.npar, cfg.slice_spacing_m)
        sampled_par = sampling.sampled_lines(cfg.npar, cfg.rz, 0)

        rf_phase_deg = 0.0
        rf_phase_inc_deg = 0.0

        for ky in sampled_pe:
            y_scale = phase_areas[ky] / max_pe_area if max_pe_area > 0.0 else 0.0
            gy_pre = pp.scale_grad(gy_template, y_scale)
            gy_reph = pp.scale_grad(gy_template, -y_scale)
            label_lin = pp.make_label(type="SET", label="LIN", value=ky)

            for par in sampled_par:
                z_scale = par_areas[par] / max_par_area if max_par_area > 0.0 else 0.0
                gz_pre_combined, gz_post_combined = encoding.combined_z_gradients(z_scale, gz_pe_template, gz_reph, gz_spoil, opts)

                rf_curr = system.copy_event(timing["rf"])
                rf_curr.phase_offset = np.deg2rad(rf_phase_deg)

                label_par = pp.make_label(type="SET", label="PAR", value=par)
                label_slc = pp.make_label(type="SET", label="SLC", value=0)

                seq.add_block(rf_curr, timing["gz"], label_slc, label_par, label_lin)
                seq.add_block(echo["gx_pre"], gy_pre, gz_pre_combined)
                if te_delay is not None:
                    seq.add_block(te_delay)

                readout.add_echo_train_blocks(seq, echo, cfg.num_echoes, cfg.flyback, rf_curr.phase_offset)

                seq.add_block(echo["gx_spoil"], gy_reph, gz_post_combined)

                rf_phase_deg = (rf_phase_deg + rf_phase_inc_deg) % 360.0
                rf_phase_inc_deg = (rf_phase_inc_deg + excitation.RF_SPOILING_INC_DEG) % 360.0

            if tr_delay is not None:
                seq.add_block(tr_delay)

        echo_times_s = [cfg.te_s + n * cfg.echo_spacing_s for n in range(cfg.num_echoes)]

        seq.set_definition("Name", "gre_multiecho_3d")
        seq.set_definition("FOV", [cfg.fov_ro_m, cfg.fov_pe_m, cfg.slice_spacing_m * cfg.npar])
        seq.set_definition("TE1", cfg.te_s)
        seq.set_definition("EchoTimes", echo_times_s)
        seq.set_definition("EchoSpacing", cfg.echo_spacing_s)
        seq.set_definition("NumEchoes", cfg.num_echoes)
        seq.set_definition("FlybackReadout", cfg.flyback)
        seq.set_definition("BipolarReadout", not cfg.flyback)
        seq.set_definition("TR", cfg.tr_s)
        seq.set_definition("Flip", cfg.flip_deg)
        seq.set_definition("ImagingMode", "3d")
        seq.set_definition("ReadoutAxis", cfg.ro_axis)
        seq.set_definition("PhaseAxis", cfg.pe_axis)
        seq.set_definition("BandwidthHzPerPx", cfg.bandwidth_hz_px)
        seq.set_definition("Ry", cfg.ry)
        seq.set_definition("Rz", cfg.rz)
        seq.set_definition("AcsLines", cfg.acs_lines)
        seq.set_definition("RfSpoilingIncDeg", excitation.RF_SPOILING_INC_DEG)
        seq.set_definition("Nx", cfg.nx_ro)
        seq.set_definition("Ny", cfg.ny_pe)
        seq.set_definition("NySampled", len(sampled_pe))
        seq.set_definition("NumPartitions", cfg.npar)
        pio.write(seq, output=output_path, remove_duplicates=False, check_timing=False)


class _Config:
    __slots__ = (
        "te_s", "tr_s", "flip_deg", "fov_ro_m", "fov_pe_m", "slab_thickness_m", "slice_spacing_m",
        "nx_ro", "ny_pe", "npar", "bandwidth_hz_px", "ry", "rz", "acs_lines", "ro_axis", "pe_axis",
        "num_echoes", "echo_spacing_s", "flyback",
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
    cfg.bandwidth_hz_px = params.param_float_optional(prot, UIParam.BANDWIDTH, system.DEFAULT_BANDWIDTH_HZ_PX)
    cfg.ry = max(1, int(round(params.param_float_optional(prot, UIParam.RY, 1.0))))
    cfg.rz = max(1, int(round(params.param_float_optional(prot, UIParam.RZ, 1.0))))
    cfg.acs_lines = 0
    cfg.ro_axis, cfg.pe_axis = params.resolve_readout_phase_axes(prot)
    cfg.num_echoes = params.param_int_optional(prot, UIParam.NUM_ECHOES, 1)
    cfg.echo_spacing_s = params.user_float(prot, USER_SLOT_ECHO_SPACING, 5.0) * 1e-3
    cfg.flyback = params.user_float(prot, USER_SLOT_FLYBACK, 1.0) >= 0.5
    return cfg


def _compute_timing(opts: pp.Opts, cfg: _Config, strict: bool, n_inner: int | None = None):
    # n_inner decouples the TE-feasibility check (validate_protocol always
    # probes with n_inner=1) from the TR-fits-N-partitions budget
    # (make_sequence uses the real partition count) — see gre_multiecho_2d.py.
    if n_inner is None:
        n_inner = cfg.npar
    system.apply_system_derates(opts)

    rf, gz, gz_reph = excitation.slice_selective(opts, cfg.flip_deg, cfg.slab_thickness_m)

    echo = readout.compute_readout_and_echo_train(
        opts=opts,
        ro_axis=cfg.ro_axis,
        nx_ro=cfg.nx_ro,
        fov_ro_m=cfg.fov_ro_m,
        bandwidth_hz_px=cfg.bandwidth_hz_px,
        slice_thickness_m=cfg.slab_thickness_m,
        num_echoes=cfg.num_echoes,
        echo_spacing_s=cfg.echo_spacing_s,
        flyback=cfg.flyback,
        strict=strict,
    )
    if echo is None:
        return None

    gz_spoil = pp.make_trapezoid(channel="z", area=encoding.SPOIL_FACTOR_Z / cfg.slab_thickness_m, system=opts)
    max_pe_area = 0.5 * cfg.ny_pe * (1.0 / cfg.fov_pe_m)
    gy_template = pp.make_trapezoid(channel=cfg.pe_axis, area=max_pe_area, system=opts)

    _, max_par_area = encoding.partition_geometry(cfg.npar, cfg.slice_spacing_m)
    gz_pe_template = pp.make_trapezoid(channel="z", area=max_par_area, system=opts) if max_par_area > 0.0 else None
    gz_pre_worst, gz_post_worst = encoding.z_worst_case_trapezoids(gz_reph, gz_spoil, max_par_area, opts)

    d_rf = pp.calc_duration(rf, gz)
    d_pre = pp.calc_duration(echo["gx_pre"], gy_template, gz_pre_worst)
    d_post = pp.calc_duration(echo["gx_spoil"], gy_template, gz_post_worst)

    rf_center_s = pp.calc_rf_center(rf)[0]
    min_te_s = (d_rf - rf_center_s) + d_pre + echo["adc_center_s"]
    te_delay_s = cfg.te_s - min_te_s
    if te_delay_s < -1e-9 and strict:
        return None
    if te_delay_s < 0.0:
        te_delay_s = 0.0

    min_block_s = d_rf + d_pre + te_delay_s + echo["echo_train_span_s"] + d_post
    tr_delay_s = cfg.tr_s - n_inner * min_block_s
    if tr_delay_s < -1e-9 and strict:
        return None
    if tr_delay_s < 0.0:
        tr_delay_s = 0.0

    return {
        "rf": rf,
        "gz": gz,
        "gz_reph": gz_reph,
        "gz_spoil": gz_spoil,
        "gz_pe_template": gz_pe_template,
        "gy_template": gy_template,
        "echo": echo,
        "te_delay_s": te_delay_s,
        "tr_delay_s": tr_delay_s,
        "min_block_s": min_block_s,
    }


PLUGIN = GreMultiEcho3DPulseqSequence()


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
    ('--num-echoes', UIParam.NUM_ECHOES, int, ""),
    ('--echo-spacing-ms', UIParam.user_value(USER_SLOT_ECHO_SPACING), float, ""),
    ('--flyback', UIParam.user_value(USER_SLOT_FLYBACK), ("const", 1.0), ""),
    ('--bipolar', UIParam.user_value(USER_SLOT_FLYBACK), ("const", 0.0), ""),
    ('--tr-ms', UIParam.TR, float, ""),
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
]

if __name__ == "__main__":
    raise SystemExit(
        cli.run_cli(
            PLUGIN,
            sys.argv[1:],
            arg_map=_ARG_MAP,
            description='Generate a 3D multi-echo Cartesian GRE .seq offline.',
            default_output='gre_multiecho_3d.seq',
        )
    )
