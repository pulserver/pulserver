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
FOV is covered by spokes pointing anywhere on the sphere.

Segmented, not flat. Only one segment of spokes is ever *designed*: the
scan is that segment replayed once per shot under a rigid rotation, carried
as a block ROTATIONS extension. This is not only a size optimisation — a
flat list of a few thousand freely placed directions gives every spoke its
own gradient waveform, and the interpreter stores a bounded number of shapes
per gradient definition, so it rejects the sequence outright
(``PULSEG_ERR_TOO_MANY_GRAD_SHOTS``). With one designed segment the shape
count is set by the views *per shot* and does not grow with the shot count.

Rotation encoding goes one level deeper than the shot: the view-to-view step
*inside* a shell is a per-block rotation too, so the whole shell costs two
gradient shapes however long it is. That needs consecutive views a constant
angle apart -- which is not only a circle. ``pulserver.design.make_rotated_projection_sampling``
offers ``spiral`` (the default: a pole-to-pole shell walked at a constant
angular step, so it spans the full polar range like a phyllotaxis interleaf),
``disk`` (a great circle, simplest, poles oversampled), and ``phyllotaxis``
(best uniformity but a wandering step, so it cannot be rotation-encoded and is
limited to a short shell).

``NumShots`` (native GE) is the number of rotated segments and ``ETL`` the
views per segment, so the scan holds ``NumShots * ETL`` spokes; ``LIN`` still
counts spokes globally.

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

USER_SLOT_SCHEME = 0

#: Shell orderings, in protocol-value order. The first two step by a constant
#: angle and so can be rotation-encoded; phyllotaxis cannot.
_SCHEMES = ("spiral", "disk", "phyllotaxis")
_ROTATION_ENCODED = frozenset({"spiral", "disk"})

# Without rotation encoding every view carries its own transition waveform, and
# they all share one gradient definition (same sample count, same delay), so the
# interpreter's 16-shapes-per-definition budget is spent about three at a time --
# one per driven axis. Five views is what reliably fits.
MAX_UNROTATED_VIEWS = 5

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
                value=13, min=1, max=10_000, incr=1,
                options=[8, 13, 21, 34, 55], validate=Validate.NONE,
            ),
            UIParam.ETL: DropdownIntParam(
                value=160, min=1, max=8192, incr=1,
                options=[64, 128, 160, 256, 512], validate=Validate.NONE,
            ),
            UIParam.SEQUENCE_TYPE: make_enum_param(UIParam.SEQUENCE_TYPE, SequenceType.GRADIENT_ECHO),
            UIParam.user_name(USER_SLOT_SCHEME): Description(text="Shell ordering (0=spiral, 1=disk, 2=phyllotaxis)"),
            UIParam.user_value(USER_SLOT_SCHEME): DropdownFloatParam(
                value=0.0, min=0.0, max=2.0, incr=1.0, unit="", options=[0.0, 1.0, 2.0], validate=Validate.NONE,
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
        if cfg.views_per_shot < 1:
            return {"valid": False, "duration": None, "info": "ETL (views per shot) must be >= 1"}
        if cfg.scheme not in _ROTATION_ENCODED and cfg.views_per_shot > MAX_UNROTATED_VIEWS:
            return {
                "valid": False,
                "duration": None,
                "info": (
                    f"the {cfg.scheme} ordering steps by a varying angle, so it needs one gradient "
                    f"waveform per view and the interpreter holds about {MAX_UNROTATED_VIEWS} per "
                    f"segment; use the spiral or disk ordering for longer shells, or ETL <= "
                    f"{MAX_UNROTATED_VIEWS}"
                ),
            }

        try:
            timing = _compute_timing(opts=opts, cfg=cfg, strict=True)
        except ValueError as error:
            return {"valid": False, "duration": None, "info": str(error)}
        if timing is None:
            return {"valid": False, "duration": None, "info": "TR too short for the ZTE readout"}

        duration_s = cfg.num_shots * timing["readout"].duration
        spokes = cfg.num_shots * cfg.views_per_shot
        return {"valid": True, "duration": duration_s, "info": f"TA = {duration_s:.2f} s, {spokes} spokes"}

    def make_sequence(self, opts: pp.Opts, protocol: dict[str, dict], output_path: str) -> None:
        prot = dict_to_protocol(protocol)
        cfg = _read_protocol(prot)

        timing = _compute_timing(opts=opts, cfg=cfg, strict=False)

        zte = timing["readout"]
        # One designed segment, replayed under one rotation per shot. The
        # spoke label stays global, so the reconstruction still sees a flat
        # 0..num_spokes-1 LIN counter.
        #
        # The segment *is* the TR, so it is handed over once and each shot
        # supplies only its spoke indices and its rotation quaternion -- a
        # rotation is one event whose numbers move, not a change of structure.
        views = np.arange(zte.num_views)
        rotations = timing["rotations"]
        zte.set_state(lin_idx=views, rotation=rotations[0])
        seq = pp.Sequence(opts, len(rotations), zte)
        for shot, rotation in enumerate(rotations):
            zte.set_state(lin_idx=shot * zte.num_views + views, rotation=rotation)
            for _block in zte:
                seq.add_block(*_block)
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
        seq.set_definition("ViewsPerShot", zte.num_views)
        seq.set_definition("NumSpokes", cfg.num_shots * zte.num_views)
        seq.set_definition("SegmentOrdering", cfg.scheme)
        pio.write(seq, output=output_path, check_timing=False)


class _Config:
    __slots__ = ("flip_deg", "fov_m", "num_shots", "nx_ro", "scheme", "tr_s", "views_per_shot")


def _read_protocol(prot: dict) -> _Config:
    cfg = _Config()
    cfg.tr_s = params.param_float(prot, UIParam.TR) * 1e-3
    cfg.flip_deg = params.param_float(prot, UIParam.FLIP)
    cfg.fov_m = params.param_float(prot, UIParam.FOV) * 1e-3
    cfg.nx_ro = params.param_int(prot, UIParam.NX)
    cfg.num_shots = params.param_int_optional(prot, UIParam.NUM_SHOTS, 13)
    cfg.views_per_shot = params.param_int_optional(prot, UIParam.ETL, 160)
    selected = round(params.user_float(prot, USER_SLOT_SCHEME, 0.0))
    cfg.scheme = _SCHEMES[min(max(selected, 0), len(_SCHEMES) - 1)]
    return cfg


def _compute_timing(opts: pp.Opts, cfg: _Config, strict: bool):
    # One base segment plus one rotation per shot, and -- for the disk ordering
    # -- one more rotation per view inside the segment. What the interpreter
    # bounds is distinct gradient *shapes* per definition, and every ZTE
    # transition lands in the same definition (same sample count, same delay),
    # so a flat list of num_shots * views_per_shot free directions asks for that
    # many shapes and is rejected (PULSEG_ERR_TOO_MANY_GRAD_SHOTS). Encoding the
    # view-to-view step as a rotation instead leaves two shapes for the whole
    # scan, which is what lets the segment be as long as the protocol wants.
    base, rotations = design.make_rotated_projection_sampling(
        (cfg.nx_ro, cfg.nx_ro, cfg.nx_ro),
        shots=cfg.num_shots,
        views=cfg.views_per_shot,
        scheme=cfg.scheme,
        # The Fibonacci rule keeps the base segment's azimuth steps small, but
        # it is a smoothness guide rather than a hard requirement -- the real
        # constraint is whether the readout can slew between consecutive views
        # within TR, and Zte checks that itself below.
        require_fibonacci=False,
    )
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
            base.flatten(),
            excitation,
            tr_s=cfg.tr_s,
            # Replaying one waveform per view needs a constant angular step;
            # the phyllotaxis ordering does not have one, so it keeps a
            # waveform per view and is limited by the shape budget.
            rotation_encoded=cfg.scheme in _ROTATION_ENCODED,
        )
    except ValueError as error:
        if strict and "tr_s=" in str(error):
            return None
        raise

    return {
        "readout": module,
        "rotations": rotations,
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
    ('--views-per-shot', UIParam.ETL, int, ""),
    ('--ordering', UIParam.user_value(USER_SLOT_SCHEME),
     {name: float(index) for index, name in enumerate(_SCHEMES)}, ""),
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
