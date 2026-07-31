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
- Phase/frequency axis swap, same as ``gre_2d.py``.

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

DEFAULT_BANDWIDTH_HZ_PX = 125_000.0

USER_SLOT_ECHO_SPACING = 0
USER_SLOT_FLYBACK = 1
USER_SLOT_SPSP = 2
USER_SLOT_SPSP_BW = 3


class GreMultiEcho3DPulseqSequence(Sequence):
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
                value=DEFAULT_BANDWIDTH_HZ_PX, min=5_000.0, max=500_000.0, incr=100.0,
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

        duration_s = cfg.tr_s * float(len(_phase_encode_loop(cfg)))
        return {"valid": True, "duration": duration_s, "info": f"TA = {duration_s:.2f} s"}

    def make_sequence(self, opts: pp.Opts, protocol: dict[str, dict], output_path: str) -> None:
        prot = dict_to_protocol(protocol)
        cfg = _read_protocol(prot)

        timing = _compute_timing(opts=opts, cfg=cfg, strict=False)

        pulse = timing["pulse"]
        line = timing["line"]
        te_delay_s = timing["te_delay_s"]
        tr_delay_s = timing["tr_delay_s"]

        te_delay = pp.make_delay(te_delay_s) if te_delay_s > 0.0 else None
        tr_delay = pp.make_delay(tr_delay_s) if tr_delay_s > 0.0 else None

        pe_loop = _phase_encode_loop(cfg)
        par_loop = _partition_loop(cfg)
        rf_phases = design.make_rf_spoiling_schedule(len(pe_loop) * len(par_loop))
        shot = 0

        # One shot's chronology, handed over once; the loop supplies only the
        # numbers that move. The TR delay stays outside it -- one is played per
        # phase-encode step, after every partition -- so it lands between
        # complete passes. Labels are set here because a label is an event the
        # template has to record.
        pulse.set_state(phase_offset_rad=0.0, SLC=0)
        line.set_state(lin_idx=0, par_idx=0)
        tr_struct = [pulse, *([te_delay] if te_delay is not None else []), line]
        seq = pp.Sequence(opts, len(pe_loop) * len(par_loop), *tr_struct)

        for pe_shot in pe_loop:
            for par_shot in par_loop:
                phase = float(rf_phases[shot])
                pulse.set_state(phase_offset_rad=phase)
                for block in pulse:
                    seq.add_block(*block)
                if te_delay is not None:
                    seq.add_block(te_delay)
                line.set_state(
                    lin_idx=int(pe_shot[0, 0]),
                    par_idx=int(par_shot[0, 0]),
                    phase_offset_rad=phase,
                )
                for block in line:
                    seq.add_block(*block)
                shot += 1

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
        seq.set_definition("ReadoutAxis", "x")
        seq.set_definition("PhaseAxis", "y")
        seq.set_definition("BandwidthHzPerPx", cfg.bandwidth_hz_px)
        seq.set_definition("Ry", cfg.ry)
        seq.set_definition("Rz", cfg.rz)
        seq.set_definition("AcsLines", cfg.acs_lines)
        seq.set_definition("RfSpoilingIncDeg", 117.0)
        seq.set_definition("Nx", cfg.nx_ro)
        seq.set_definition("Ny", cfg.ny_pe)
        seq.set_definition("NySampled", len(pe_loop))
        seq.set_definition("NumPartitions", cfg.npar)
        seq.set_definition("SPSPExcitation", cfg.spsp)
        pio.write(seq, output=output_path, check_timing=False)


class _Config:
    __slots__ = (
        "acs_lines",
        "bandwidth_hz_px",
        "echo_spacing_s",
        "flip_deg",
        "flyback",
        "fov_pe_m",
        "fov_ro_m",
        "npar",
        "num_echoes",
        "nx_ro",
        "ny_pe",
        "pe_axis",
        "ro_axis",
        "ry",
        "rz",
        "slab_thickness_m",
        "slice_spacing_m",
        "spsp",
        "spsp_bandwidth_hz",
        "te_s",
        "tr_s",
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
    cfg.bandwidth_hz_px = params.param_float_optional(prot, UIParam.BANDWIDTH, DEFAULT_BANDWIDTH_HZ_PX)
    cfg.ry = max(1, round(params.param_float_optional(prot, UIParam.RY, 1.0)))
    cfg.rz = max(1, round(params.param_float_optional(prot, UIParam.RZ, 1.0)))
    cfg.acs_lines = 0
    cfg.ro_axis, cfg.pe_axis = params.resolve_readout_phase_axes(prot)
    cfg.num_echoes = params.param_int_optional(prot, UIParam.NUM_ECHOES, 1)
    cfg.echo_spacing_s = params.user_float(prot, USER_SLOT_ECHO_SPACING, 5.0) * 1e-3
    cfg.flyback = params.user_float(prot, USER_SLOT_FLYBACK, 1.0) >= 0.5
    cfg.spsp = params.user_float(prot, USER_SLOT_SPSP, 0.0) >= 0.5
    cfg.spsp_bandwidth_hz = params.user_float(prot, USER_SLOT_SPSP_BW, 250.0)
    return cfg


def _phase_encode_loop(cfg: _Config):
    """In-plane phase-encode loop; one shot per TR."""
    return design.make_cartesian_sampling(
        (cfg.nx_ro, cfg.ny_pe), acceleration=cfg.ry, calibration=cfg.acs_lines
    )


def _partition_loop(cfg: _Config):
    """Partition (kz) loop, played inside one TR — its own encoded axis."""
    return design.make_cartesian_sampling((cfg.nx_ro, cfg.npar), acceleration=cfg.rz)


def _compute_timing(opts: pp.Opts, cfg: _Config, strict: bool, n_inner: int | None = None):
    # n_inner decouples the TE-feasibility check (validate_protocol always
    # probes with n_inner=1) from the TR-fits-N-partitions budget
    # (make_sequence uses the real partition count) — see gre_multiecho_2d.py.
    if n_inner is None:
        n_inner = cfg.npar
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
        num_echoes=cfg.num_echoes,
        bandwidth_hz_px=cfg.bandwidth_hz_px,
        esp_s=cfg.echo_spacing_s,
        flyback=cfg.flyback,
        spoil_position="post",
        spoil_cycles=1.0,
    )
    d_pulse = sum(pp.calc_duration(*block) for block in pulse.blocks)
    rf_center_s = pp.calc_rf_center(pulse.rf)[0] + pulse.rf.delay
    min_te_s = (d_pulse - rf_center_s) + line.t_first_echo_s
    raster = opts.block_duration_raster
    te_delay_s = round((cfg.te_s - min_te_s) / raster) * raster
    if te_delay_s < -1e-9 and strict:
        return None
    if te_delay_s < 0.0:
        te_delay_s = 0.0

    min_block_s = d_pulse + te_delay_s + line.duration
    tr_delay_s = round((cfg.tr_s - n_inner * min_block_s) / raster) * raster
    if tr_delay_s < -1e-9 and strict:
        return None
    if tr_delay_s < 0.0:
        tr_delay_s = 0.0

    return {
        "pulse": pulse,
        "line": line,
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
    ('--spsp', UIParam.user_value(USER_SLOT_SPSP), ("const", 1.0), ""),
    ('--spsp-bandwidth-hz', UIParam.user_value(USER_SLOT_SPSP_BW), float, ""),
]

if __name__ == "__main__":
    raise SystemExit(
        run_cli(
            PLUGIN,
            sys.argv[1:],
            arg_map=_ARG_MAP,
            description='Generate a 3D multi-echo Cartesian GRE .seq offline.',
            default_output='gre_multiecho_3d.seq',
        )
    )
