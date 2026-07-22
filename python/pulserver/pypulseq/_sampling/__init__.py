"""High-level acquisition plans and composable low-level sampling tools."""

from . import cartesian, epi, noncartesian, ordering, slices
from ._pattern import SamplingPattern
from ._plan import (
    Acquisition,
    AcquisitionPlan,
    make_cartesian_sampling,
    make_epi_sampling,
    make_noncartesian_2d_sampling,
    make_noncartesian_projection_sampling,
    make_noncartesian_stack_sampling,
)
from .cartesian import (
    calc_sampled_lines,
    make_caipirinha_mask,
    make_centric_order,
    make_linear_order,
    make_poisson_disc_mask,
    make_radial_adaptive_order,
    make_radial_order,
    make_random_mask,
    make_shuffling_order,
)
from .epi import make_skipped_caipi_order
from .noncartesian import (
    calc_golden_angles,
    calc_raga_angles,
    calc_tiny_golden_angles,
    calc_uniform_angles,
    make_golden_means_3d_tilt,
    make_radial_tilt,
    make_spiral_phyllotaxis_tilt,
)
from .ordering import calc_chunk_indices
from .slices import SliceGroup, SliceSampling, make_slice_sampling

__all__ = [
    "SamplingPattern",
    "Acquisition",
    "AcquisitionPlan",
    "SliceGroup",
    "SliceSampling",
    "cartesian",
    "epi",
    "noncartesian",
    "ordering",
    "slices",
    "calc_chunk_indices",
    "calc_golden_angles",
    "calc_raga_angles",
    "calc_sampled_lines",
    "calc_tiny_golden_angles",
    "calc_uniform_angles",
    "make_caipirinha_mask",
    "make_cartesian_sampling",
    "make_centric_order",
    "make_epi_sampling",
    "make_golden_means_3d_tilt",
    "make_linear_order",
    "make_noncartesian_2d_sampling",
    "make_noncartesian_projection_sampling",
    "make_noncartesian_stack_sampling",
    "make_poisson_disc_mask",
    "make_radial_adaptive_order",
    "make_radial_order",
    "make_radial_tilt",
    "make_random_mask",
    "make_shuffling_order",
    "make_skipped_caipi_order",
    "make_slice_sampling",
    "make_spiral_phyllotaxis_tilt",
]
