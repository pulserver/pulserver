"""MRD streaming server and high-level MRI reconstruction API.

The server components work with ``pulserver[recon]``.  Install
``pulserver[recon-cpu]`` for Cartesian/non-Cartesian reconstruction with
FINUFFT, MRPro, and DeepInverse; add ``pulserver[recon-nlinv]`` for PyGROG
NLINV calibration or use ``pulserver[recon-cuda]`` for the CUFINUFFT backend.
"""

from .algorithms import pics
from .calibration import estimate_sensitivities, nlinv_sensitivities
from ._cuda_streaming import CudaStreamedDenoiser, CudaStreaming
from .bloch import (
    BSSFP,
    FSE,
    SPGR,
    SSFPEcho,
    SSFPFID,
    EpgInterpreter,
    SimulationResult,
    SubspaceBasis,
    TissueProperties,
    make_interpreter,
    simulate_subspace,
)
from .connection import Connection
from .denoisers import (
    AverageDenoiser,
    LLR,
    TGV,
    TV,
    Wavelet,
    average,
    denoiser,
    llr,
    tgv,
    tv,
    wavelet,
)
from .density import PipeMenonDCF, pipe, pipe_menon_dcf
from .distortion import run_pyhysco
from .epi import EpiAcquisitionGroups, partition_epi_acquisitions
from .gradunwarp import (
    CoefficientAccessor,
    GradientCoefficients,
    Gradunwarp,
    ImageGeometry,
    MrdCoefficientAccessor,
)
from .grouping import filter_acquisitions, group_by_labels, split_on_flag
from .linops import available_nufft_backends
from .metadata import MrdMetadata, acquisition_labels, has_acquisition_flag
from .optimizers import PolynomialPreconditioner
from .physics import (
    Cartesian2D,
    MRIPhysics,
    NonCartesian2D,
    NonCartesian3D,
    OffResonance,
    Subspace,
    Toeplitz,
    cartesian_2d,
    noncartesian_2d,
    noncartesian_3d,
    off_resonance,
    subspace,
    toeplitz,
)
from .preprocessing import (
    cartesian_3d_to_2d,
    coil_compress,
    correct_epi_eddy_currents,
    epi_ramp_interpolate,
    estimate_epi_eddy_phase,
    noise_prewhiten,
    remove_readout_oversampling,
)
from .rtp_connection import PmcPayload, RtpServer, write_pmc_payload
from .serialization import images_to_dicom, write_dicom_series
from .seqdesc import (
    AdcRole,
    EventType,
    RfDefinition,
    RfShape,
    RfUse,
    SequenceDescription,
    SequenceDescriptionCollection,
    SequenceEvent,
    SequenceParameters,
    ShimDefinition,
    decode_sequence_description,
    decompress_shape,
)
from .server import Server
from .sms import SmsEpiInputs

__all__ = [
    "BSSFP",
    "FSE",
    "LLR",
    "SPGR",
    "SSFPFID",
    "TGV",
    "TV",
    "AdcRole",
    "AverageDenoiser",
    "Cartesian2D",
    "CoefficientAccessor",
    "Connection",
    "CudaStreamedDenoiser",
    "CudaStreaming",
    "EpgInterpreter",
    "EpiAcquisitionGroups",
    "EventType",
    "GradientCoefficients",
    "Gradunwarp",
    "ImageGeometry",
    "MRIPhysics",
    "MrdCoefficientAccessor",
    "MrdMetadata",
    "NonCartesian2D",
    "NonCartesian3D",
    "OffResonance",
    "PipeMenonDCF",
    "PmcPayload",
    "PolynomialPreconditioner",
    "RfDefinition",
    "RfShape",
    "RfUse",
    "RtpServer",
    "SSFPEcho",
    "SequenceDescription",
    "SequenceDescriptionCollection",
    "SequenceEvent",
    "SequenceParameters",
    "Server",
    "ShimDefinition",
    "SimulationResult",
    "SmsEpiInputs",
    "Subspace",
    "SubspaceBasis",
    "TissueProperties",
    "Toeplitz",
    "Wavelet",
    "acquisition_labels",
    "available_nufft_backends",
    "average",
    "cartesian_2d",
    "cartesian_3d_to_2d",
    "coil_compress",
    "correct_epi_eddy_currents",
    "decode_sequence_description",
    "decompress_shape",
    "denoiser",
    "epi_ramp_interpolate",
    "estimate_epi_eddy_phase",
    "estimate_sensitivities",
    "filter_acquisitions",
    "group_by_labels",
    "has_acquisition_flag",
    "images_to_dicom",
    "llr",
    "make_interpreter",
    "nlinv_sensitivities",
    "noise_prewhiten",
    "noncartesian_2d",
    "noncartesian_3d",
    "off_resonance",
    "partition_epi_acquisitions",
    "pics",
    "pipe",
    "pipe_menon_dcf",
    "remove_readout_oversampling",
    "run_pyhysco",
    "simulate_subspace",
    "split_on_flag",
    "subspace",
    "tgv",
    "toeplitz",
    "tv",
    "wavelet",
    "write_dicom_series",
    "write_pmc_payload",
]
