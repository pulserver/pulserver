"""RF pulse and magnetization-preparation design.

Public factories use pypulseq conventions: flip angles are radians, time is
seconds, frequency is Hz, distance is metres, and ``system`` carries scanner
limits. SLR design is local and does not import SigPy.
"""

from ._base import RfModule, RfPulse, RfState
from ._excitation import (
    DEFAULT_DURATION,
    DEFAULT_SLICE_DURATION,
    DEFAULT_TIME_BANDWIDTH_PRODUCT,
    RF_SPOILING_INCREMENT_DEG,
    make_adiabatic_pulse,
    make_frequency_selective_pulse,
    make_hard_pulse,
    make_inversion_pulse,
    make_refocusing_pulse,
    make_sigpy_pulse,
    make_slice_selective_pulse,
    make_slr_pulse,
)
from ._preparation import (
    BlochSiegertPulse,
    DiffusionPrepPulse,
    FatSaturationPulse,
    IhMTPulse,
    MTPulse,
    T2PrepPulse,
    fat_shift_hz,
    make_bloch_siegert_pulse,
    make_diffusion_prep,
    make_fat_saturation_pulse,
    make_ihmt_pulse,
    make_mt_pulse,
    make_t1t2_prep_pulse,
    make_t2prep_pulse,
)
from ._spatial import (
    SpatialPulse,
    make_2d_selective_pulse,
    make_3d_selective_pulse,
    make_spatially_selective_pulse,
    make_spiral_selective_pulse,
    make_spsp_pulse,
)

__all__ = [
    "DEFAULT_DURATION",
    "DEFAULT_SLICE_DURATION",
    "DEFAULT_TIME_BANDWIDTH_PRODUCT",
    "RF_SPOILING_INCREMENT_DEG",
    "RfState",
    "RfModule",
    "RfPulse",
    "SpatialPulse",
    "FatSaturationPulse",
    "MTPulse",
    "IhMTPulse",
    "BlochSiegertPulse",
    "T2PrepPulse",
    "DiffusionPrepPulse",
    "make_hard_pulse",
    "make_adiabatic_pulse",
    "make_sigpy_pulse",
    "make_slr_pulse",
    "make_frequency_selective_pulse",
    "make_slice_selective_pulse",
    "make_spsp_pulse",
    "make_spatially_selective_pulse",
    "make_spiral_selective_pulse",
    "make_2d_selective_pulse",
    "make_3d_selective_pulse",
    "make_inversion_pulse",
    "make_refocusing_pulse",
    "make_mt_pulse",
    "make_ihmt_pulse",
    "make_bloch_siegert_pulse",
    "make_t2prep_pulse",
    "make_t1t2_prep_pulse",
    "make_diffusion_prep",
    "make_fat_saturation_pulse",
    "fat_shift_hz",
]
