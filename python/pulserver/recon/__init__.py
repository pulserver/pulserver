"""MRD streaming server and composable MRI reconstruction primitives.

The server components work with ``pulserver[recon]``.  Install
``pulserver[recon-cpu]`` for Cartesian/non-Cartesian reconstruction with
FINUFFT, MRPro, and DeepInverse; add ``pulserver[recon-nlinv]`` for PyGROG
NLINV calibration or use ``pulserver[recon-cuda]`` for the CUFINUFFT backend.
"""

from .calibration import estimate_sensitivities, nlinv_sensitivities
from .connection import Connection
from .distortion import run_pyhysco
from .epi import EpiAcquisitionGroups, partition_epi_acquisitions
from .grouping import filter_acquisitions, group_by_labels, split_on_flag
from .linops import deepinverse_multicoil_mri, mrinufft_operator, mrpro_operator
from .metadata import MrdMetadata, acquisition_labels, has_acquisition_flag
from .optimizers import deepinverse_optimizer, mrpro_optimizer
from .prox import deepinverse_prior
from .rtp_connection import PmcPayload, RtpServer, write_pmc_payload
from .serialization import images_to_dicom, write_dicom_series
from .server import Server
from .sms import SmsEpiInputs

__all__ = [
    "Connection",
    "EpiAcquisitionGroups",
    "MrdMetadata",
    "PmcPayload",
    "RtpServer",
    "Server",
    "SmsEpiInputs",
    "acquisition_labels",
    "deepinverse_multicoil_mri",
    "deepinverse_optimizer",
    "deepinverse_prior",
    "estimate_sensitivities",
    "filter_acquisitions",
    "group_by_labels",
    "has_acquisition_flag",
    "images_to_dicom",
    "nlinv_sensitivities",
    "partition_epi_acquisitions",
    "mrpro_operator",
    "mrpro_optimizer",
    "mrinufft_operator",
    "split_on_flag",
    "run_pyhysco",
    "write_dicom_series",
    "write_pmc_payload",
]
