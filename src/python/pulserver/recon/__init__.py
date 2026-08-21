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
stays private under :mod:`pulserver.recon._mrd`.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

#: Public name to the module defining it. Every name is reachable flat; this
#: is the only place the file layout is written down.
_MEMBERS = {
    "ADMM": "optim.admm",
    "AcquisitionBucket": "plugin",
    "AcquisitionBucketStats": "plugin",
    "AcquisitionFlag": "plugin",
    "AdcRole": "simulation",
    "AverageDenoiser": "denoisers",
    "CGInfo": "optim.cg",
    "Cartesian2D": "physics",
    "Cartesian3D": "physics",
    "CoefficientAccessor": "postprocessing",
    "CoilCompression": "gadgets",
    "ConjugateGradient": "optim.cg",
    "ContextAgnosticDenoiser": "models",
    "CudaStreaming": "execution",
    "EncodingSpace": "plugin",
    "EpiPhaseCorrection": "gadgets",
    "EventType": "simulation",
    "ExamCache": "plugin",
    "FISTA": "optim.fista",
    "Gadget": "gadgets",
    "GradientCoefficients": "postprocessing",
    "GradientDataConsistency": "learned",
    "Gradunwarp": "postprocessing",
    "Homodyne": "preprocessing",
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
    "NoiseAdjust": "gadgets",
    "NonCartesian2D": "physics",
    "NonCartesian3D": "physics",
    "OffResonance": "physics",
    "OptimResult": "optim.state",
    "OptimState": "optim.state",
    "PDHG": "optim.pdhg",
    "POCS": "preprocessing",
    "PhasePoleCorrection": "calibration",
    "PolynomialPreconditioner": "optim._algorithms",
    "Positive": "denoisers",
    "RampSampling": "gadgets",
    "ReconBuffer": "plugin",
    "ReconContext": "plugin",
    "ReconData": "plugin",
    "ReconPlugin": "plugin",
    "ReconResult": "plugin",
    "RfDefinition": "simulation",
    "RfShape": "simulation",
    "RfUse": "simulation",
    "RigidMotionEKF": "motion",
    "RigidMotionEstimate": "motion",
    "RigidRegistration": "motion",
    "SMS": "physics",
    "ScaledAdjoint": "learned",
    "SequenceDescription": "simulation",
    "SequenceDescriptionCollection": "simulation",
    "SequenceEvent": "simulation",
    "SequenceParameters": "simulation",
    "ShimDefinition": "simulation",
    "StackedPrior": "optim.prior",
    "StatefulReconstructor": "learned",
    "Subspace": "physics",
    "TGV": "denoisers",
    "TV": "denoisers",
    "Toeplitz": "physics",
    "TorchIODataset": "datasets",
    "UnrollResult": "learned",
    "UnrollState": "learned",
    "UnrolledReconstructor": "learned",
    "WaveEncoding": "physics",
    "WavePSF": "calibration",
    "WavePSFCalibration": "calibration",
    "WavePSFResult": "calibration",
    "WaveShuffling": "physics",
    "Wavelet": "denoisers",
    "as_numpy": "postprocessing",
    "available_nufft_backends": "physics",
    "calibration_extent": "calibration",
    "cartesian_recon": "cartesian",
    "center_crop": "postprocessing",
    "coil_combine": "postprocessing",
    "coil_maps_from_reference": "calibration",
    "coil_compress": "preprocessing",
    "correct_lines": "preprocessing",
    "decode_sequence_description": "simulation",
    "decompress_shape": "simulation",
    "default_model_paths": "weights",
    "diffusion_table": "_mrd.metadata",
    "epi_ramp_operator": "preprocessing",
    "estimate_epi_phase": "preprocessing",
    "fftc": "preprocessing",
    "fill_partial_echo": "preprocessing",
    "has_acquisition_flag": "plugin",
    "ifftc": "preprocessing",
    "image_result": "postprocessing",
    "load_model": "weights",
    "noise_prewhiten": "preprocessing",
    "noncartesian_recon": "noncartesian",
    "pics": "optim._algorithms",
    "pipe_menon_dcf": "preprocessing",
    "remove_readout_oversampling": "preprocessing",
    "run_pyhysco": "postprocessing",
    "save_bundle": "weights",
    "user_parameter": "_mrd.metadata",
}

__all__ = sorted(_MEMBERS)

#: Reachable as attributes for anyone who wants them, but not part of the
#: public namespace: the flat names above are.
_SUBMODULES = frozenset(
    value.split(".", 1)[0] for value in _MEMBERS.values()
) - {"_mrd"}


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
    from ._mrd.metadata import diffusion_table as diffusion_table
    from ._mrd.metadata import user_parameter as user_parameter
    from .cartesian import cartesian_recon as cartesian_recon
    from .calibration import NLINV as NLINV
    from .calibration import NLINVPhysics as NLINVPhysics
    from .calibration import NLINVResult as NLINVResult
    from .calibration import PhasePoleCorrection as PhasePoleCorrection
    from .calibration import WavePSF as WavePSF
    from .calibration import WavePSFCalibration as WavePSFCalibration
    from .calibration import WavePSFResult as WavePSFResult
    from .calibration import calibration_extent as calibration_extent
    from .calibration import coil_maps_from_reference as coil_maps_from_reference
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
    from .learned import GradientDataConsistency as GradientDataConsistency
    from .learned import ScaledAdjoint as ScaledAdjoint
    from .learned import StatefulReconstructor as StatefulReconstructor
    from .learned import UnrollResult as UnrollResult
    from .learned import UnrollState as UnrollState
    from .learned import UnrolledReconstructor as UnrolledReconstructor
    from .models import ContextAgnosticDenoiser as ContextAgnosticDenoiser
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
    from .plugin import AcquisitionBucket as AcquisitionBucket
    from .plugin import AcquisitionBucketStats as AcquisitionBucketStats
    from .plugin import AcquisitionFlag as AcquisitionFlag
    from .plugin import EncodingSpace as EncodingSpace
    from .plugin import ExamCache as ExamCache
    from .plugin import ReconBuffer as ReconBuffer
    from .plugin import ReconContext as ReconContext
    from .plugin import ReconData as ReconData
    from .plugin import ReconPlugin as ReconPlugin
    from .plugin import ReconResult as ReconResult
    from .plugin import has_acquisition_flag as has_acquisition_flag
    from .postprocessing import CoefficientAccessor as CoefficientAccessor
    from .postprocessing import GradientCoefficients as GradientCoefficients
    from .postprocessing import Gradunwarp as Gradunwarp
    from .postprocessing import ImageGeometry as ImageGeometry
    from .postprocessing import as_numpy as as_numpy
    from .postprocessing import image_result as image_result
    from .noncartesian import noncartesian_recon as noncartesian_recon
    from .postprocessing import center_crop as center_crop
    from .postprocessing import coil_combine as coil_combine
    from .postprocessing import run_pyhysco as run_pyhysco
    from .preprocessing import Homodyne as Homodyne
    from .preprocessing import POCS as POCS
    from .preprocessing import coil_compress as coil_compress
    from .preprocessing import correct_lines as correct_lines
    from .preprocessing import epi_ramp_operator as epi_ramp_operator
    from .preprocessing import estimate_epi_phase as estimate_epi_phase
    from .preprocessing import fftc as fftc
    from .preprocessing import fill_partial_echo as fill_partial_echo
    from .preprocessing import ifftc as ifftc
    from .preprocessing import noise_prewhiten as noise_prewhiten
    from .preprocessing import pipe_menon_dcf as pipe_menon_dcf
    from .preprocessing import remove_readout_oversampling as remove_readout_oversampling
    from .simulation import AdcRole as AdcRole
    from .simulation import EventType as EventType
    from .simulation import RfDefinition as RfDefinition
    from .simulation import RfShape as RfShape
    from .simulation import RfUse as RfUse
    from .simulation import SequenceDescription as SequenceDescription
    from .simulation import SequenceDescriptionCollection as SequenceDescriptionCollection
    from .simulation import SequenceEvent as SequenceEvent
    from .simulation import SequenceParameters as SequenceParameters
    from .simulation import ShimDefinition as ShimDefinition
    from .simulation import decode_sequence_description as decode_sequence_description
    from .simulation import decompress_shape as decompress_shape
    from .weights import MODEL_PATH_ENV as MODEL_PATH_ENV
    from .weights import ModelBundle as ModelBundle
    from .weights import ModelStore as ModelStore
    from .weights import default_model_paths as default_model_paths
    from .weights import load_model as load_model
    from .weights import save_bundle as save_bundle
