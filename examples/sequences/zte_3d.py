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
file's docstring for the full rationale and the "gap" caveat), but with
GENUINELY 3D spoke directions: unlike every other
non-Cartesian file in the zoo (which rotates a 2D in-plane trajectory about
the physical Z axis and, for 3D, stacks that with a separate Cartesian
partition encode — ``gre_radial_3d.py``, ``gre_noncart_3d.py``), true ZTE
has no separate slice/partition dimension to stack: the whole non-selective
FOV is covered by spokes pointing anywhere on the sphere. Directions come
from ``pulserver.design.make_noncartesian_projection_sampling`` (near-uniform 3D
coverage). The full ordered direction list is passed to
:func:`pulserver.make_zte_readout`, which keeps the gradient live and
slews directly between consecutive sphere points.

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
import pulserver.io as pio
import pulserver.design as design
import pulserver.pypulseq as pp
from pulserver import (
    DropdownFloatParam,
    DropdownIntParam,
    Sequence,
    SequenceType,
    UIParam,
    Validate,
    dict_to_protocol,
    make_enum_param,
    params,
    protocol_to_dict,
    run_cli,
)

ZTE_RF_DURATION_S = 20e-6

class Zte3DPulseqSequence(Sequence):
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

        duration_s = timing["readout"].duration
        return {"valid": True, "duration": duration_s, "info": f"TA = {duration_s:.2f} s"}

    def make_sequence(self, opts: pp.Opts, protocol: dict[str, dict], output_path: str) -> None:
        prot = dict_to_protocol(protocol)
        cfg = _read_protocol(prot)

        timing = _compute_timing(opts=opts, cfg=cfg, strict=False)

        zte = timing["readout"]
        seq = pp.Sequence(opts)
        zte.set_state(lin_idx=np.arange(cfg.num_shots)).add_to(seq)

        seq.set_definition("Name", "zte_3d")
        seq.set_definition("FOV", [cfg.fov_m, cfg.fov_m, cfg.fov_m])
        seq.set_definition("Flip", cfg.flip_deg)
        seq.set_definition("TR", cfg.tr_s)
        seq.set_definition("ImagingMode", "3d")
        seq.set_definition("Trajectory", "zte")
        seq.set_definition("Gap", timing["gap_s"])
        seq.set_definition("MissingSamples", zte.num_missing_samples)
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
    directions = design.make_noncartesian_projection_sampling(
        (cfg.nx_ro, cfg.nx_ro, cfg.nx_ro), views=cfg.num_shots
    ).flatten()
    excitation = design.make_hard_pulse(
        np.deg2rad(cfg.flip_deg),
        duration=ZTE_RF_DURATION_S,
        system=opts,
        use="excitation",
    )
    try:
        module = design.make_zte_readout(
            opts,
            cfg.fov_m,
            cfg.nx_ro,
            directions,
            excitation,
            tr_s=cfg.tr_s,
        )
    except ValueError as error:
        if strict and "tr_s=" in str(error):
            return None
        raise

    return {
        "readout": module,
        "gap_s": module.gap_s,
        "min_block_s": module.duration,
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
        run_cli(
            PLUGIN,
            sys.argv[1:],
            arg_map=_ARG_MAP,
            description='Generate a 3D (spherical-spoke) ZTE .seq offline.',
            default_output='zte_3d.seq',
        )
    )
