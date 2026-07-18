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
axis (``ROTATIONS`` extension, ``pulserver.pulseq.make_rotation`` — no
``pulserver.arbgrad`` waveform solver needed for straight spokes) while the
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
import pypulseq as pp
from scipy.spatial.transform import Rotation

import pulserver.io as pio
import pulserver.pulseq as ps

from pulserver import (
    PulseqSequence,
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
from pulserver import arbgrad
from pulserver.core import SequenceType
from pulserver.design import cli, encoding, excitation, params, preparations, readout, sampling, system



NUM_ECHOES = 1
FLYBACK = True
RO_AXIS = "x"

USER_SLOT_ORDER_MODE = 0


def _order_mode_name(code: float) -> str:
    return "golden" if code >= 0.5 else "uniform"


class GreRadial3DPulseqSequence(PulseqSequence):
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
            UIParam.RZ: TypeinFloatParam(value=1.0, min=1.0, max=8.0, incr=1.0, unit="", validate=Validate.NONE),
            UIParam.BANDWIDTH: TypeinFloatParam(
                value=system.DEFAULT_BANDWIDTH_HZ_PX, min=5_000.0, max=500_000.0, incr=100.0,
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

        sampled_par = sampling.sampled_lines(cfg.npar, cfg.rz, 0)
        duration_s = cfg.tr_s * float(cfg.num_shots)
        return {"valid": True, "duration": duration_s, "info": f"TA = {duration_s:.2f} s"}

    def make_sequence(self, opts: pp.Opts, protocol: dict[str, dict], output_path: str) -> None:
        prot = dict_to_protocol(protocol)
        cfg = _read_protocol(prot)

        timing = _compute_timing(opts=opts, cfg=cfg, strict=False)

        gz_reph = timing["gz_reph"]
        gz_spoil = timing["gz_spoil"]
        gz_pe_template = timing["gz_pe_template"]
        echo = timing["echo"]
        te_delay_s = timing["te_delay_s"]
        tr_delay_s = timing["tr_delay_s"]

        te_delay = pp.make_delay(te_delay_s) if te_delay_s > 0.0 else None
        tr_delay = pp.make_delay(tr_delay_s) if tr_delay_s > 0.0 else None

        seq = ps.Sequence(opts)

        angles = arbgrad.shot_angles(cfg.num_shots, mode=cfg.order_mode)
        par_areas, max_par_area = encoding.partition_geometry(cfg.npar, cfg.slice_spacing_m)
        sampled_par = sampling.sampled_lines(cfg.npar, cfg.rz, 0)

        rf_phase_deg = 0.0
        rf_phase_inc_deg = 0.0

        for spoke, angle in enumerate(angles):
            rotation = ps.make_rotation(Rotation.from_euler("z", float(angle)))
            label_lin = pp.make_label(type="SET", label="LIN", value=spoke)

            for par in sampled_par:
                z_scale = par_areas[par] / max_par_area if max_par_area > 0.0 else 0.0
                gz_pre_combined, gz_post_combined = encoding.combined_z_gradients(
                    z_scale, gz_pe_template, gz_reph, gz_spoil, opts
                )

                rf_curr = system.copy_event(timing["rf"])
                rf_curr.phase_offset = np.deg2rad(rf_phase_deg)
                adc_curr = system.copy_event(echo["adc"])
                adc_curr.phase_offset = rf_curr.phase_offset

                label_par = pp.make_label(type="SET", label="PAR", value=par)
                label_slc = pp.make_label(type="SET", label="SLC", value=0)

                seq.add_block(rf_curr, timing["gz"], label_slc, label_par, label_lin)
                seq.add_block(echo["gx_pre"], gz_pre_combined, rotation)
                if te_delay is not None:
                    seq.add_block(te_delay)
                seq.add_block(echo["gx_echo"], adc_curr, rotation)
                seq.add_block(echo["gx_spoil"], gz_post_combined, rotation)

                rf_phase_deg = (rf_phase_deg + rf_phase_inc_deg) % 360.0
                rf_phase_inc_deg = (rf_phase_inc_deg + excitation.RF_SPOILING_INC_DEG) % 360.0

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
        seq.set_definition("RfSpoilingIncDeg", excitation.RF_SPOILING_INC_DEG)
        seq.set_definition("Nx", cfg.nx_ro)
        seq.set_definition("NumShots", cfg.num_shots)
        seq.set_definition("NumPartitions", cfg.npar)
        pio.write(seq, output=output_path, remove_duplicates=False, check_timing=False)


class _Config:
    __slots__ = (
        "te_s", "tr_s", "flip_deg", "fov_m", "slab_thickness_m", "slice_spacing_m",
        "nx_ro", "npar", "bandwidth_hz_px", "rz", "num_shots", "order_mode",
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
    cfg.bandwidth_hz_px = params.param_float_optional(prot, UIParam.BANDWIDTH, system.DEFAULT_BANDWIDTH_HZ_PX)
    cfg.rz = max(1, int(round(params.param_float_optional(prot, UIParam.RZ, 1.0))))
    cfg.num_shots = params.param_int_optional(prot, UIParam.NUM_SHOTS, 100)
    cfg.order_mode = _order_mode_name(params.user_float(prot, USER_SLOT_ORDER_MODE, 1.0))
    return cfg


def _compute_timing(opts: pp.Opts, cfg: _Config, strict: bool, n_inner: int | None = None):
    if n_inner is None:
        n_inner = len(sampling.sampled_lines(cfg.npar, cfg.rz, 0))
    system.apply_system_derates(opts)

    rf, gz, gz_reph = excitation.slice_selective(opts, cfg.flip_deg, cfg.slab_thickness_m)

    echo = readout.compute_readout_and_echo_train(
        opts=opts,
        ro_axis=RO_AXIS,
        nx_ro=cfg.nx_ro,
        fov_ro_m=cfg.fov_m,
        bandwidth_hz_px=cfg.bandwidth_hz_px,
        slice_thickness_m=cfg.slab_thickness_m,
        num_echoes=NUM_ECHOES,
        echo_spacing_s=0.0,
        flyback=FLYBACK,
        strict=strict,
    )
    if echo is None:
        return None

    gz_spoil = pp.make_trapezoid(channel="z", area=encoding.SPOIL_FACTOR_Z / cfg.slab_thickness_m, system=opts)

    _, max_par_area = encoding.partition_geometry(cfg.npar, cfg.slice_spacing_m)
    gz_pe_template = pp.make_trapezoid(channel="z", area=max_par_area, system=opts) if max_par_area > 0.0 else None
    gz_pre_worst, gz_post_worst = encoding.z_worst_case_trapezoids(gz_reph, gz_spoil, max_par_area, opts)

    d_rf = pp.calc_duration(rf, gz)
    d_pre = pp.calc_duration(echo["gx_pre"], gz_pre_worst)
    d_post = pp.calc_duration(echo["gx_spoil"], gz_post_worst)

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
        "echo": echo,
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
]

if __name__ == "__main__":
    raise SystemExit(
        cli.run_cli(
            PLUGIN,
            sys.argv[1:],
            arg_map=_ARG_MAP,
            description='Generate a 3D stack-of-stars radial GRE .seq offline.',
            default_output='gre_radial_3d.seq',
        )
    )
