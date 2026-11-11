"""Reconstruction plugin contract and the runtime that drives it over MRD streams."""

from ._calibration import (
    CoilCompression,
    CoilSensitivities,
    MissingCalibration,
    coil_maps,
)
from ._correct import (
    Corrections,
    IntensityCorrection,
    fermi_filtered,
    gradient_unwarped,
)
from ._exam_image import ExamImage
from ._figure import figure_rgb
from ._loader import load_plugin
from .gadgets import (
    AsymmetricEcho,
    CompressChannels,
    Prewhiten,
    RemoveReadoutOversampling,
)
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
    "CompressChannels",
    "Corrections",
    "ExamCache",
    "ExamImage",
    "Gadget",
    "IntensityCorrection",
    "MissingCalibration",
    "Prewhiten",
    "ReconBuffer",
    "ReconContext",
    "ReconData",
    "ReconPlugin",
    "ReconResult",
    "RemoveReadoutOversampling",
    "coil_maps",
    "fermi_filtered",
    "figure_rgb",
    "gradient_unwarped",
    "load_plugin",
]
