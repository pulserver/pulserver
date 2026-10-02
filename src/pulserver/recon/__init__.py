"""Reconstruction plugin contract and the runtime that drives it over MRD streams."""

from ._calibration import (
    CoilCompression,
    CoilSensitivities,
    MissingCalibration,
    coil_maps,
)
from ._loader import load_plugin
from .gadgets import AsymmetricEcho, Prewhiten, RemoveReadoutOversampling
from .plugin import (
    B0_MAP,
    B1_MAP,
    COIL_SENSITIVITIES,
    EXAM_ARTIFACTS,
    NOISE_COVARIANCE,
    ExamCache,
    Gadget,
    ReconBuffer,
    ReconContext,
    ReconData,
    ReconPlugin,
    ReconResult,
)

__all__ = [
    "B0_MAP",
    "B1_MAP",
    "COIL_SENSITIVITIES",
    "EXAM_ARTIFACTS",
    "NOISE_COVARIANCE",
    "AsymmetricEcho",
    "CoilCompression",
    "CoilSensitivities",
    "ExamCache",
    "Gadget",
    "MissingCalibration",
    "Prewhiten",
    "ReconBuffer",
    "ReconContext",
    "ReconData",
    "ReconPlugin",
    "ReconResult",
    "RemoveReadoutOversampling",
    "coil_maps",
    "load_plugin",
]
