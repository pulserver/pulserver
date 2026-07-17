"""Shared T2-preparation building blocks for the zoo plugin family.

Composite non-selective T2-prep module (Brittain et al.): 90x - tau/2 -
180y - tau/2 - (-90x), followed by a spoiler. ``TE_prep`` (``tau``) sets the
T2 weighting directly — unlike the inversion module's TI, there is no
separate post-module recovery wait baked into the module itself; the
sequence proceeds straight into imaging after the tip-up pulse and
spoiler.

Loaded via file-path (``importlib.util.spec_from_file_location``), same
convention as ``_gre_common.py``/``_prep_common.py``. Reuses
``_prep_common.py``'s non-selective 3-axis spoiler and raster-rounding
helper (loaded as a sibling here) rather than duplicating them.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pypulseq as pp


def _load_sibling_module(name: str):
    module_path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pc = _load_sibling_module("_prep_common")

T2PREP_90_DURATION_S = 0.5e-3
T2PREP_180_HARD_DURATION_S = 0.5e-3
T2PREP_180_ADIABATIC_DURATION_S = 10.24e-3

build_t2prep_spoiler = pc.build_inversion_spoiler
round_to_raster = pc.round_to_raster


def build_t2prep_pulses(opts: pp.Opts, refocus_mode: str = "hard"):
    """Build the (rf90_down, rf180, rf90_up) non-selective T2-prep pulses.

    ``refocus_mode`` selects a hard block 180 or an adiabatic hypsec 180 —
    same choice as the inversion module's hard-vs-adiabatic toggle.
    """
    rf90_down = pp.make_block_pulse(
        flip_angle=np.pi / 2.0, duration=T2PREP_90_DURATION_S, system=opts, use="preparation"
    )
    if refocus_mode == "adiabatic":
        rf180 = pp.make_adiabatic_pulse(
            pulse_type="hypsec",
            duration=T2PREP_180_ADIABATIC_DURATION_S,
            dwell=1e-5,
            system=opts,
            use="refocusing",
        )
    elif refocus_mode == "hard":
        rf180 = pp.make_block_pulse(
            flip_angle=np.pi, duration=T2PREP_180_HARD_DURATION_S, system=opts, use="refocusing"
        )
    else:
        raise ValueError(f"Unknown refocus_mode '{refocus_mode}' (expected 'hard' or 'adiabatic')")
    rf90_up = pp.make_block_pulse(
        flip_angle=np.pi / 2.0,
        duration=T2PREP_90_DURATION_S,
        phase_offset=np.pi,
        system=opts,
        use="preparation",
    )
    return rf90_down, rf180, rf90_up


def t2prep_delays_seconds(
    te_prep_s: float,
    rf90_down,
    rf180,
    rf90_up,
    strict: bool,
) -> tuple[float, float] | None:
    """Compute the two tau/2 delays either side of the 180 so the total
    transverse dwell time (tip-down center to tip-up center) equals
    ``te_prep_s``. Returns ``None`` if infeasible under ``strict``."""
    half_te_s = te_prep_s / 2.0

    d90_down_s = pp.calc_duration(rf90_down)
    c90_down_s = pp.calc_rf_center(rf90_down)[0]
    d180_s = pp.calc_duration(rf180)
    c180_s = pp.calc_rf_center(rf180)[0]
    c90_up_s = pp.calc_rf_center(rf90_up)[0]

    delay1_s = round_to_raster(half_te_s - (d90_down_s - c90_down_s) - c180_s)
    delay2_s = round_to_raster(half_te_s - (d180_s - c180_s) - c90_up_s)

    if (delay1_s < -1e-9 or delay2_s < -1e-9) and strict:
        return None
    return max(delay1_s, 0.0), max(delay2_s, 0.0)
