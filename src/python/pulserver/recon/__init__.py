"""Torch-first MRI reconstruction: physics, solvers, and the plugin contract.

One flat namespace. Everything a reconstruction needs is reached as
``pulserver.recon.<name>`` -- there is no submodule to remember, and no name
that lives in only one of two places::

    import pulserver.recon as recon

    physics = recon.NonCartesian2D(trajectory, (256, 256))
    image = recon.pics(kspace, physics)

The names are grouped by what they are for -- buffers, calibration, physics,
solvers, pre- and post-processing -- in the API documentation, which is where
a grouping belongs: it is a way of reading the library, not a constraint on
how to import from it.

Names resolve on first use, so importing this module needs neither an MRD
server environment nor any optional numerical backend; only the names actually
touched pull their dependencies in. The Gadgetron/MRD transport machinery
stays private under the streaming transport.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

#: Public name to the module defining it. Every name is reachable flat; this
#: is the only place the file layout is written down.
_MEMBERS = {
    "ADMM": "optim.admm",
    "ARCHITECTURE_ROOTS": "weights",
    "AverageDenoiser": "denoisers",
    "CGInfo": "optim.cg",
    "Cartesian2D": "physics",
    "Cartesian3D": "physics",
    "Checkpointed": "adapters",
    "CoefficientAccessor": "postprocessing",
    "CoilCompression": "gadgets",
    "ComplexDenoiser": "adapters",
    "ConjugateGradient": "optim.cg",
    "ContextAgnosticDenoiser": "adapters",
    "CudaStreaming": "execution",
    "EpiPhaseCorrection": "gadgets",
    "ExamCache": "plugin",
    "FISTA": "optim.fista",
    "Gadget": "gadgets",
    "GradientCoefficients": "postprocessing",
    "Gradunwarp": "postprocessing",
    "IRGNM": "optim.irgnm",
    "IXI": "datasets",
    "IXITiny": "datasets",
    "ImageGeometry": "postprocessing",
    "LLR": "denoisers",
    "MODEL_PATH_ENV": "weights",
    "MRIPhysics": "physics",
    "ModelBundle": "weights",
    "ModelStore": "weights",
    "NLINV": "calibration",
    "NLINVPhysics": "calibration",
    "NLINVResult": "calibration",
    "NavigatorMotionTracker": "motion",
    "NoiseAdjust": "gadgets",
    "NoiseConditioned": "adapters",
    "NonCartesian2D": "physics",
    "NonCartesian3D": "physics",
    "NormalEquationL2": "adapters",
    "OffResonance": "physics",
    "OptimResult": "optim.state",
    "OptimState": "optim.state",
    "PDHG": "optim.pdhg",
    "PmcPayload": "_server.rtp_connection",
    "PolynomialPreconditioner": "optim._algorithms",
    "Positive": "denoisers",
    "RampSampling": "gadgets",
    "ReconBuffer": "plugin",
    "ReconContext": "plugin",
    "ReconData": "plugin",
    "ReconPlugin": "plugin",
    "ReconResult": "plugin",
    "RigidMotionEKF": "motion",
    "RigidMotionEstimate": "motion",
    "RigidRegistration": "motion",
    "SMS": "physics",
    "ScaledAdjoint": "adapters",
    "StackedPrior": "optim.prior",
    "StepwiseUnroll": "adapters",
    "Subspace": "physics",
    "TGV": "denoisers",
    "TV": "denoisers",
    "Toeplitz": "physics",
    "TorchIODataset": "datasets",
    "WaveEncoding": "physics",
    "WavePSF": "calibration",
    "WavePSFCalibration": "calibration",
    "WavePSFResult": "calibration",
    "WaveShuffling": "physics",
    "Wavelet": "denoisers",
    "available_nufft_backends": "physics",
    "calibration_extent": "calibration",
    "cartesian_recon": "cartesian",
    "coil_maps_from_reference": "calibration",
    "default_model_paths": "weights",
    "field_map": "calibration",
    "image_result": "postprocessing",
    "load_model": "weights",
    "noncartesian_recon": "noncartesian",
    "pics": "optim._algorithms",
    "run_pyhysco": "postprocessing",
    "save_bundle": "weights",
}

__all__ = sorted(_MEMBERS)

#: Reachable as attributes for anyone who wants them, but not part of the
#: public namespace: the flat names above are.
_SUBMODULES = frozenset(value.split(".", 1)[0] for value in _MEMBERS.values()) - {
    "_mrd"
}


def __getattr__(name: str) -> Any:
    """Resolve one public name, or a submodule, on first use."""
    module = _MEMBERS.get(name)
    if module is not None:
        return getattr(importlib.import_module(f"{__name__}.{module}"), name)
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return the flat public reconstruction namespace."""
    return sorted(__all__)


if TYPE_CHECKING:
    from .cartesian import cartesian_recon as cartesian_recon
    from .calibration import NLINV as NLINV
    from .calibration import NLINVPhysics as NLINVPhysics
    from .calibration import NLINVResult as NLINVResult
    from .calibration import WavePSF as WavePSF
    from .calibration import WavePSFCalibration as WavePSFCalibration
    from .calibration import WavePSFResult as WavePSFResult
    from .calibration import calibration_extent as calibration_extent
    from .calibration import coil_maps_from_reference as coil_maps_from_reference
    from .calibration import field_map as field_map
    from .gadgets import CoilCompression as CoilCompression
    from .gadgets import EpiPhaseCorrection as EpiPhaseCorrection
    from .gadgets import Gadget as Gadget
    from .gadgets import NoiseAdjust as NoiseAdjust
    from .gadgets import RampSampling as RampSampling
    from .datasets import IXI as IXI
    from .datasets import IXITiny as IXITiny
    from .datasets import TorchIODataset as TorchIODataset
    from .denoisers import AverageDenoiser as AverageDenoiser
    from .denoisers import LLR as LLR
    from .denoisers import Positive as Positive
    from .denoisers import TGV as TGV
    from .denoisers import TV as TV
    from .denoisers import Wavelet as Wavelet
    from .execution import CudaStreaming as CudaStreaming
    from .adapters import NormalEquationL2 as NormalEquationL2
    from .adapters import ScaledAdjoint as ScaledAdjoint
    from .adapters import StepwiseUnroll as StepwiseUnroll
    from .adapters import Checkpointed as Checkpointed
    from .adapters import ComplexDenoiser as ComplexDenoiser
    from .adapters import ContextAgnosticDenoiser as ContextAgnosticDenoiser
    from .adapters import NoiseConditioned as NoiseConditioned
    from .motion import NavigatorMotionTracker as NavigatorMotionTracker
    from ._server.rtp_connection import PmcPayload as PmcPayload
    from .motion import RigidMotionEKF as RigidMotionEKF
    from .motion import RigidMotionEstimate as RigidMotionEstimate
    from .motion import RigidRegistration as RigidRegistration
    from .optim import ADMM as ADMM
    from .optim import CGInfo as CGInfo
    from .optim import ConjugateGradient as ConjugateGradient
    from .optim import FISTA as FISTA
    from .optim import IRGNM as IRGNM
    from .optim import OptimResult as OptimResult
    from .optim import OptimState as OptimState
    from .optim import PDHG as PDHG
    from .optim import PolynomialPreconditioner as PolynomialPreconditioner
    from .optim import StackedPrior as StackedPrior
    from .optim import pics as pics
    from .physics import Cartesian2D as Cartesian2D
    from .physics import Cartesian3D as Cartesian3D
    from .physics import MRIPhysics as MRIPhysics
    from .physics import NonCartesian2D as NonCartesian2D
    from .physics import NonCartesian3D as NonCartesian3D
    from .physics import OffResonance as OffResonance
    from .physics import SMS as SMS
    from .physics import Subspace as Subspace
    from .physics import Toeplitz as Toeplitz
    from .physics import WaveEncoding as WaveEncoding
    from .physics import WaveShuffling as WaveShuffling
    from .physics import available_nufft_backends as available_nufft_backends
    from .plugin import ExamCache as ExamCache
    from .plugin import ReconBuffer as ReconBuffer
    from .plugin import ReconContext as ReconContext
    from .plugin import ReconData as ReconData
    from .plugin import ReconPlugin as ReconPlugin
    from .plugin import ReconResult as ReconResult
    from .postprocessing import CoefficientAccessor as CoefficientAccessor
    from .postprocessing import GradientCoefficients as GradientCoefficients
    from .postprocessing import Gradunwarp as Gradunwarp
    from .postprocessing import ImageGeometry as ImageGeometry
    from .postprocessing import image_result as image_result
    from .noncartesian import noncartesian_recon as noncartesian_recon
    from .postprocessing import run_pyhysco as run_pyhysco
    from .weights import ARCHITECTURE_ROOTS as ARCHITECTURE_ROOTS
    from .weights import MODEL_PATH_ENV as MODEL_PATH_ENV
    from .weights import ModelBundle as ModelBundle
    from .weights import ModelStore as ModelStore
    from .weights import default_model_paths as default_model_paths
    from .weights import load_model as load_model
    from .weights import save_bundle as save_bundle
