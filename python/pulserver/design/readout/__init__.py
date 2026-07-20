"""FSE / spin-echo and EPI readout trains."""

from __future__ import annotations

from .epi import Epi2D, Epi2DFlyback, Epi3D, Epi3DFlyback
from .fse import (
    CPMG_PHASE_OFFSET_RAD,
    DEFAULT_REFOCUS_FLIP_DEG,
    Fse2D,
    Fse3D,
    MultiEchoSE,
    build_refocus_flip_schedule,
    build_refocusing_pulse,
    build_z_crusher,
)

__all__ = [
    "MultiEchoSE",
    "Fse2D",
    "Fse3D",
    "build_refocusing_pulse",
    "build_refocus_flip_schedule",
    "build_z_crusher",
    "CPMG_PHASE_OFFSET_RAD",
    "DEFAULT_REFOCUS_FLIP_DEG",
    "Epi2D",
    "Epi3D",
    "Epi2DFlyback",
    "Epi3DFlyback",
]
