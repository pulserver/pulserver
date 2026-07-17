"""Shared CPMG refocusing-train building blocks for the zoo plugin family.

Used by ``fse_2d.py`` and ``fse_3d.py``. Builds the fast/turbo spin-echo
readout train: one 90 excitation followed by ``ETL`` 180 refocusing pulses
(constant 90 deg CPMG phase offset relative to the 90, not incremented —
unlike GRE RF spoiling), each flanked by a crusher and followed by its own
independently phase-encoded readout (a different ``ky`` per echo, not a
blip train like ``_epi_common.py``'s EPI train).

Loaded via file-path (``importlib.util.spec_from_file_location``), same
convention as ``_gre_common.py`` (loaded as a sibling here).

CPMG timing: with the crusher/prephase/rephase pattern kept symmetric
before and after every 180 (same crusher both sides, same-duration
prephase/rephase either side of the readout), the steady-state inter-180
period collapses to a single repeated delay (``tau_steady``) — see
``fse_timing`` — which is what keeps every echo exactly ``ESP`` apart, the
CPMG condition required for clean refocusing.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pypulseq as pp

RF_REFOCUS_TIME_S = 3.0e-3
RF_REFOCUS_APODIZATION = 0.5
RF_REFOCUS_TIME_BW_PRODUCT = 4.0
CPMG_PHASE_OFFSET_RAD = 1.5707963267948966  # pi/2, constant for all refocusing pulses


def _load_sibling_module(name: str):
    module_path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gc = _load_sibling_module("_gre_common")
pc = _load_sibling_module("_prep_common")


def build_refocusing_pulse(opts: pp.Opts, slice_thickness_m: float):
    """Build the (rf180, gz180) CPMG refocusing pulse (no rephase lobe —
    the crusher on either side handles spoiling, not a small gradient
    rephase the way excitation pulses need)."""
    rf180, gz180, _ = pp.make_sinc_pulse(
        flip_angle=3.14159265358979,
        duration=RF_REFOCUS_TIME_S,
        slice_thickness=slice_thickness_m,
        apodization=RF_REFOCUS_APODIZATION,
        time_bw_product=RF_REFOCUS_TIME_BW_PRODUCT,
        system=opts,
        use="refocusing",
        return_gz=True,
    )
    rf180.phase_offset = CPMG_PHASE_OFFSET_RAD
    return rf180, gz180


def fse_timing(
    esp_s: float,
    d90_s: float,
    c90_s: float,
    d180_s: float,
    c180_s: float,
    crusher_dur_s: float,
    half_loop_fixed_s: float,
    strict: bool,
):
    """Return ``(tau_first_s, tau_steady_s)``.

    ``tau_first`` is the 90-to-first-180 gap; ``tau_steady`` is the
    (symmetric, repeated) gap either side of every 180 — see module
    docstring. ``half_loop_fixed_s`` is the fixed duration that must fit in
    each ``ESP/2`` half-period on the readout side (prephase/rephase +
    ADC-center offset); the crusher duration is passed separately since it
    appears on both the 90-side and readout-side halves identically.
    Returns ``None`` if infeasible under ``strict``."""
    half_esp_s = esp_s / 2.0
    tau_first_s = pc.round_to_raster(half_esp_s - (d90_s - c90_s) - crusher_dur_s - c180_s)
    tau_steady_s = pc.round_to_raster(half_esp_s - (d180_s - c180_s) - crusher_dur_s - half_loop_fixed_s)
    if (tau_first_s < -1e-9 or tau_steady_s < -1e-9) and strict:
        return None
    return max(tau_first_s, 0.0), max(tau_steady_s, 0.0)
