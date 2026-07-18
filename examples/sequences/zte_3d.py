"""Standalone 3D zero-echo-time (ZTE) sequence plugin.

This module implements the three mandatory module-level entry points
required by the bridge dispatcher (see ``zte_2d.py`` for the in-plane-only
sibling this extends, and ``_zte_common.py`` for the shared building
blocks):

- ``get_default_protocol(opts)``  — return the initial CV/protocol dictionary.
- ``validate_protocol(opts, protocol)``  — validate the current protocol and
  return a ``{"valid": bool, "duration": float | None, "info": str}`` dict.
- ``make_sequence(opts, protocol, output_path)``  — synthesise the sequence
  and write it to *output_path* using :func:`pulserver.io.write`.

Same gradient-before-pulse half-echo readout as ``zte_2d.py`` (see that
file's and ``_zte_common.py``'s docstrings for the full rationale and the
"gap" caveat), but with GENUINELY 3D spoke directions: unlike every other
non-Cartesian file in the zoo (which rotates a 2D in-plane trajectory about
the physical Z axis and, for 3D, stacks that with a separate Cartesian
partition encode — ``gre_radial_3d.py``, ``gre_noncart_3d.py``), true ZTE
has no separate slice/partition dimension to stack: the whole non-selective
FOV is covered by spokes pointing anywhere on the sphere. Directions come
from ``pulserver.arbgrad.generate_fibonacci_sphere`` (near-uniform 3D
coverage); each spoke's rotation is computed via
``scipy.spatial.transform.Rotation.align_vectors``, mapping the logical
+x readout direction onto that spoke's target unit vector (a general 3D
rotation, not just a Z-axis one).

``NumShots`` (native GE) is the number of 3D spokes.

Usage
-----
Register the plugin in the interpreter tree by symlinking (or copying) this
file to the ``sequences/src/`` directory and creating a numbered alias::

    cp zte_3d.py pulserver-interpreter/package/pulserver/sequences/src/zte_3d.py
    ln -sf src/zte_3d.py pulserver-interpreter/package/pulserver/sequences/sequence18.py
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
    DropdownFloatParam,
    DropdownIntParam,
    UIParam,
    Validate,
    dict_to_protocol,
    make_enum_param,
    protocol_to_dict,
)
from pulserver import arbgrad
from pulserver.core import SequenceType
from pulserver.design import cli, encoding, excitation, params, preparations, readout, sampling, system



REFERENCE_DIRECTION = np.array([1.0, 0.0, 0.0])


class Zte3DPulseqSequence(PulseqSequence):
    """Generate a 3D (genuinely spherical-spoke) ZTE sequence."""

    def get_default_protocol(self, opts: pp.Opts) -> dict[str, dict]:
        del opts
        protocol = {
            UIParam.TR: DropdownFloatParam(
                value=3.0, min=1.0, max=100.0, incr=0.1, unit="ms",
                options=[2.0, 3.0, 5.0, 10.0, 20.0], validate=Validate.NONE,
            ),
            UIParam.FLIP: DropdownFloatParam(
                value=4.0, min=1.0, max=20.0, incr=0.5, unit="deg",
                options=[2.0, 4.0, 6.0, 8.0, 10.0], validate=Validate.NONE,
            ),
            UIParam.FOV: DropdownFloatParam(
                value=220.0, min=80.0, max=500.0, incr=1.0, unit="mm",
                options=[180.0, 220.0, 280.0, 340.0, 500.0], validate=Validate.NONE,
            ),
            UIParam.NX: DropdownIntParam(
                value=64, min=16, max=512, incr=1, options=[64, 128, 192, 256, 384], validate=Validate.NONE,
            ),
            UIParam.NUM_SHOTS: DropdownIntParam(
                value=2000, min=8, max=100_000, incr=1,
                options=[1000, 2000, 4000, 8000, 16000], validate=Validate.NONE,
            ),
            UIParam.SEQUENCE_TYPE: make_enum_param(UIParam.SEQUENCE_TYPE, SequenceType.GRADIENT_ECHO),
        }
        return protocol_to_dict(protocol)

    def validate_protocol(self, opts: pp.Opts, protocol: dict[str, dict]) -> dict:
        prot = dict_to_protocol(protocol)
        cfg = _read_protocol(prot)

        if cfg.tr_s <= 0.0:
            return {"valid": False, "duration": None, "info": "TR must be > 0"}
        if cfg.fov_m <= 0.0:
            return {"valid": False, "duration": None, "info": "FOV must be > 0"}
        if not (0.0 < cfg.flip_deg <= 90.0):
            return {"valid": False, "duration": None, "info": "Flip angle must be in (0, 90] deg"}
        if cfg.nx_ro < 1:
            return {"valid": False, "duration": None, "info": "NX must be >= 1"}
        if cfg.num_shots < 1:
            return {"valid": False, "duration": None, "info": "NumShots must be >= 1"}

        timing = _compute_timing(opts=opts, cfg=cfg, strict=True)
        if timing is None:
            return {"valid": False, "duration": None, "info": "TR too short for the ZTE readout"}

        duration_s = cfg.tr_s * float(cfg.num_shots)
        return {"valid": True, "duration": duration_s, "info": f"TA = {duration_s:.2f} s"}

    def make_sequence(self, opts: pp.Opts, protocol: dict[str, dict], output_path: str) -> None:
        prot = dict_to_protocol(protocol)
        cfg = _read_protocol(prot)

        timing = _compute_timing(opts=opts, cfg=cfg, strict=False)

        gx = timing["gx"]
        adc = timing["adc"]
        tr_delay_s = timing["tr_delay_s"]
        tr_delay = pp.make_delay(tr_delay_s) if tr_delay_s > 0.0 else None

        seq = ps.Sequence(opts)

        directions = arbgrad.generate_fibonacci_sphere(cfg.num_shots)

        for spoke in range(cfg.num_shots):
            target = directions[spoke]
            rot, _ = Rotation.align_vectors([target], [REFERENCE_DIRECTION])
            rotation = ps.make_rotation(rot)
            rf = system.copy_event(timing["rf"])
            label_lin = pp.make_label(type="SET", label="LIN", value=spoke)

            seq.add_block(gx, rf, adc, label_lin, rotation)
            if tr_delay is not None:
                seq.add_block(tr_delay)

        seq.set_definition("Name", "zte_3d")
        seq.set_definition("FOV", [cfg.fov_m, cfg.fov_m, cfg.fov_m])
        seq.set_definition("Flip", cfg.flip_deg)
        seq.set_definition("TR", cfg.tr_s)
        seq.set_definition("ImagingMode", "3d")
        seq.set_definition("Trajectory", "zte")
        seq.set_definition("Gap", timing["gap_s"])
        seq.set_definition("Nx", cfg.nx_ro)
        seq.set_definition("NumShots", cfg.num_shots)
        pio.write(seq, output=output_path, remove_duplicates=False, check_timing=False)


class _Config:
    __slots__ = ("tr_s", "flip_deg", "fov_m", "nx_ro", "num_shots")


def _read_protocol(prot: dict) -> _Config:
    cfg = _Config()
    cfg.tr_s = params.param_float(prot, UIParam.TR) * 1e-3
    cfg.flip_deg = params.param_float(prot, UIParam.FLIP)
    cfg.fov_m = params.param_float(prot, UIParam.FOV) * 1e-3
    cfg.nx_ro = params.param_int(prot, UIParam.NX)
    cfg.num_shots = params.param_int_optional(prot, UIParam.NUM_SHOTS, 2000)
    return cfg


def _compute_timing(opts: pp.Opts, cfg: _Config, strict: bool):
    system.apply_system_derates(opts)

    rf = readout.build_hard_pulse(opts, cfg.flip_deg)
    ro_events = readout.build_half_echo_readout(opts, "x", cfg.nx_ro, cfg.fov_m)
    if ro_events is None:
        return None
    gx, adc, gap_s = ro_events
    rf.delay = gx.rise_time

    min_block_s = pp.calc_duration(gx, rf, adc)
    tr_delay_s = cfg.tr_s - min_block_s
    if tr_delay_s < -1e-9 and strict:
        return None
    if tr_delay_s < 0.0:
        tr_delay_s = 0.0

    return {
        "rf": rf,
        "gx": gx,
        "adc": adc,
        "gap_s": gap_s,
        "tr_delay_s": tr_delay_s,
        "min_block_s": min_block_s,
    }


PLUGIN = Zte3DPulseqSequence()


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
    ('--tr-ms', UIParam.TR, float, ""),
    ('--flip-deg', UIParam.FLIP, float, ""),
    ('--fov-mm', UIParam.FOV, float, ""),
    ('--nx', UIParam.NX, int, ""),
    ('--num-shots', UIParam.NUM_SHOTS, int, ""),
]

if __name__ == "__main__":
    raise SystemExit(
        cli.run_cli(
            PLUGIN,
            sys.argv[1:],
            arg_map=_ARG_MAP,
            description='Generate a 3D (spherical-spoke) ZTE .seq offline.',
            default_output='zte_3d.seq',
        )
    )
