"""Shared magnetization-preparation building blocks for the zoo plugin family.

Used by ``mprage_2d.py`` and ``mprage_3d.py`` (and intended for reuse by the
non-Cartesian MPRAGE variants later in the backlog). Houses the
inversion-recovery module: a non-selective hard or adiabatic (hyperbolic
secant) inversion pulse, a post-inversion 3-axis crusher, and the TI/segment
scheduling arithmetic.

Loaded via file-path (``importlib.util.spec_from_file_location``), same
convention as ``_gre_common.py`` — plugins are loaded standalone (no package
context) by both the bridge dispatcher and tests.

TI convention (matches ``refcode/pulseq/matlab/demoSeq/writeMPRAGE_4ge.m``,
the project's approved non-EPIC pulseq reference): TI is the time from the
center of the inversion RF pulse to the temporal center of the segment of
excitations that follows (i.e. ``segment_duration_s / 2``), not to any
specific k-space line. This matches standard linear-order MPRAGE TI
definitions and keeps the scheduling arithmetic independent of view order.
"""

from __future__ import annotations

import numpy as np
import pypulseq as pp

INVERSION_HARD_DURATION_S = 1.0e-3
INVERSION_ADIABATIC_DURATION_S = 10.24e-3
INVERSION_SPOIL_CYCLES = 4.0


def build_inversion_pulse(opts: pp.Opts, mode: str):
    """Build a non-selective inversion RF pulse.

    Parameters
    ----------
    mode : str
        ``"hard"`` for a rectangular (block) 180 deg pulse, or ``"adiabatic"``
        for a hyperbolic-secant adiabatic inversion pulse.
    """
    if mode == "adiabatic":
        rf = pp.make_adiabatic_pulse(
            pulse_type="hypsec",
            duration=INVERSION_ADIABATIC_DURATION_S,
            dwell=1e-5,
            system=opts,
            use="inversion",
        )
        return rf
    if mode == "hard":
        rf = pp.make_block_pulse(
            flip_angle=np.pi,
            duration=INVERSION_HARD_DURATION_S,
            system=opts,
            use="inversion",
        )
        return rf
    raise ValueError(f"Unknown inversion mode '{mode}' (expected 'hard' or 'adiabatic')")


def build_inversion_spoiler(opts: pp.Opts, voxel_size_m: float, cycles: float = INVERSION_SPOIL_CYCLES):
    """Build a non-selective 3-axis crusher played after the inversion pulse.

    ``voxel_size_m`` is the in-plane resolution (FOV/N) used as the spoiling
    reference length — there is no slice thickness to reference since the
    inversion pulse is non-selective.
    """
    area = cycles / voxel_size_m
    gx = pp.make_trapezoid(channel="x", area=area, system=opts)
    gy = pp.make_trapezoid(channel="y", area=area, system=opts)
    gz = pp.make_trapezoid(channel="z", area=area, system=opts)
    return gx, gy, gz


def chunk_indices(indices: list[int], size: int) -> list[list[int]]:
    """Split ``indices`` into consecutive chunks of at most ``size`` entries."""
    size = max(1, int(size))
    return [indices[i : i + size] for i in range(0, len(indices), size)]


def round_to_raster(value_s: float, raster_s: float = 1e-5) -> float:
    """Round ``value_s`` to the nearest multiple of ``raster_s``.

    Needed because RF "center of mass" timings (e.g. an adiabatic hypsec
    pulse's weighted center) generally do not fall on a raster boundary, so
    a TI delay computed from them is not automatically block-duration-raster
    aligned the way gradient/ADC-derived durations are.
    """
    return round(value_s / raster_s) * raster_s


def ti_delay_seconds(
    ti_s: float,
    rf_inv,
    spoiler_duration_s: float,
    segment_duration_s: float,
    strict: bool,
) -> float | None:
    """Compute the delay after the inversion spoiler needed to reach ``ti_s``.

    ``ti_s`` is measured from the center of the inversion RF pulse to the
    temporal center of the following segment (see module docstring). Returns
    ``None`` if infeasible under ``strict`` (negative required delay).
    """
    inv_duration_s = pp.calc_duration(rf_inv)
    inv_center_s = pp.calc_rf_center(rf_inv)[0]
    elapsed_before_segment_s = (inv_duration_s - inv_center_s) + spoiler_duration_s
    ti_delay_s = ti_s - elapsed_before_segment_s - 0.5 * segment_duration_s
    ti_delay_s = round_to_raster(ti_delay_s)
    if ti_delay_s < -1e-9 and strict:
        return None
    if ti_delay_s < 0.0:
        ti_delay_s = 0.0
    return ti_delay_s
