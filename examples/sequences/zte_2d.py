"""Standalone 2D (in-plane) zero-echo-time (ZTE) sequence plugin.

This module implements the three mandatory module-level entry points
required by the bridge dispatcher (see ``gre_radial_2d.py`` for the
full-echo radial sibling this parallels, and ``_zte_common.py`` for the
shared building blocks):

- ``get_default_protocol(opts)``  — return the initial CV/protocol dictionary.
- ``validate_protocol(opts, protocol)``  — validate the current protocol and
  return a ``{"valid": bool, "duration": float | None, "info": str}`` dict.
- ``make_sequence(opts, protocol, output_path)``  — synthesise the sequence
  and write it to *output_path* using :func:`pulserver.io.write`.

ZTE structure: for every spoke, the readout gradient ramps up to its constant
plateau FIRST; only once it's stable does the short non-selective hard pulse
fire, followed by the minimum RF/ADC dead-time gap, then a half-echo
(0 -> kmax) readout. A few samples nearest k-space center are unavoidably
missing (the ZTE "gap", reported but not filled). The complete ordered
in-plane direction list is handed to :func:`pulserver.make_zte_readout`,
which slews directly between consecutive directions and returns to zero only
after the segment. This remains a deliberate 2D direction-set simplification
of inherently 3D ZTE (see ``zte_3d.py``). There is no slice-select gradient:
the excitation is non-selective, so there is no separate slice loop.

``NumShots`` (``IntKey.NUM_SHOTS``, native GE) is the number of spokes. The
spoke-ordering mode has no native GE counterpart, so it's an ``opuser``
custom variable, same as ``gre_radial_2d.py``. There is no ``TE`` or
``bandwidth`` control: TE is fixed by hardware dead times (not user-set),
and the excitation/readout bandwidth is fixed by this example.

Usage
-----
Register the plugin in the interpreter tree by symlinking (or copying) this
file to the ``sequences/src/`` directory and creating a numbered alias::

    cp zte_2d.py pulserver-interpreter/package/pulserver/sequences/src/zte_2d.py
    ln -sf src/zte_2d.py pulserver-interpreter/package/pulserver/sequences/sequence17.py
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
    UIParam,
    Validate,
    dict_to_protocol,
    make_enum_param,
    params,
    protocol_to_dict,
    run_cli,
)

ZTE_RF_DURATION_S = 20e-6

USER_SLOT_ORDER_MODE = 0


def _order_mode_name(code: float) -> str:
    return "golden" if code >= 0.5 else "uniform"


class Zte2DPulseqSequence(Sequence):
    """Generate a 2D (in-plane-spoke) ZTE sequence."""

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
                value=200, min=8, max=4096, incr=1, options=[100, 200, 400, 800, 1600], validate=Validate.NONE,
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

        seq.set_definition("Name", "zte_2d")
        seq.set_definition("FOV", [cfg.fov_m, cfg.fov_m, cfg.fov_m])
        seq.set_definition("Flip", cfg.flip_deg)
        seq.set_definition("TR", cfg.tr_s)
        seq.set_definition("ImagingMode", "2d")
        seq.set_definition("Trajectory", "zte")
        seq.set_definition("SpokeOrder", cfg.order_mode)
        seq.set_definition("Gap", timing["gap_s"])
        seq.set_definition("MissingSamples", zte.num_missing_samples)
        seq.set_definition("Nx", cfg.nx_ro)
        seq.set_definition("NumShots", cfg.num_shots)
        pio.write(seq, output=output_path, check_timing=False)


class _Config:
    __slots__ = ("flip_deg", "fov_m", "num_shots", "nx_ro", "order_mode", "tr_s")


def _read_protocol(prot: dict) -> _Config:
    cfg = _Config()
    cfg.tr_s = params.param_float(prot, UIParam.TR) * 1e-3
    cfg.flip_deg = params.param_float(prot, UIParam.FLIP)
    cfg.fov_m = params.param_float(prot, UIParam.FOV) * 1e-3
    cfg.nx_ro = params.param_int(prot, UIParam.NX)
    cfg.num_shots = params.param_int_optional(prot, UIParam.NUM_SHOTS, 200)
    cfg.order_mode = _order_mode_name(params.user_float(prot, USER_SLOT_ORDER_MODE, 1.0))
    return cfg


def _compute_timing(opts: pp.Opts, cfg: _Config, strict: bool):
    # Half (centre-out) projections, so the angular period is the full circle
    # rather than the pi of a through-centre spoke.
    angles = design.make_noncartesian_2d_sampling(
        (cfg.nx_ro, cfg.nx_ro),
        views=cfg.num_shots,
        scheme="golden" if cfg.order_mode == "golden" else "linear",
        period=2.0 * np.pi,
    ).flatten()[:, 0]
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
            angles,
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


PLUGIN = Zte2DPulseqSequence()


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
    ('--order-mode', UIParam.user_value(USER_SLOT_ORDER_MODE), {'uniform': 0.0, 'golden': 1.0}, ""),
]

if __name__ == "__main__":
    raise SystemExit(
        run_cli(
            PLUGIN,
            sys.argv[1:],
            arg_map=_ARG_MAP,
            description='Generate a 2D in-plane ZTE .seq offline.',
            default_output='zte_2d.seq',
        )
    )
