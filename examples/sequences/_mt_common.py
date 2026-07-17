"""Shared magnetization-transfer (MT) saturation building block.

An off-resonance, non-selective Gaussian saturation pulse played once per
TR immediately before excitation — unlike the inversion/T2-prep modules
(played once per segment/shot), MT saturation is stacked additively on
every view since MT contrast builds up over the steady state, not within a
single preparation-to-readout window. Reuses ``_prep_common.py``'s
non-selective 3-axis spoiler (loaded as a sibling here) for the small
post-pulse crusher.

Loaded via file-path (``importlib.util.spec_from_file_location``), same
convention as ``_gre_common.py``/``_prep_common.py``.
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

MT_DEFAULT_DURATION_S = 8.0e-3
MT_DEFAULT_FLIP_DEG = 500.0
MT_DEFAULT_FREQ_OFFSET_HZ = -1500.0

build_mt_spoiler = pc.build_inversion_spoiler


def build_mt_pulse(
    opts: pp.Opts,
    flip_deg: float = MT_DEFAULT_FLIP_DEG,
    freq_offset_hz: float = MT_DEFAULT_FREQ_OFFSET_HZ,
    duration_s: float = MT_DEFAULT_DURATION_S,
):
    """Build a non-selective, off-resonance Gaussian MT saturation pulse."""
    return pp.make_gauss_pulse(
        flip_angle=np.deg2rad(flip_deg),
        duration=duration_s,
        freq_offset=freq_offset_hz,
        time_bw_product=4.0,
        system=opts,
        use="saturation",
    )
