"""K-space support, acquisition-order, and slice-group sampling plans."""

from . import cartesian, epi, noncartesian, ordering, slices
from ._pattern import SamplingPattern
from .cartesian import (
    calc_sampled_lines,
    make_caipirinha_sampling,
    make_fse_linear_order,
    make_fse_radial_adaptive_order,
    make_fse_radial_order,
    make_fse_shuffling_order,
    make_poisson_disc_sampling,
    make_random_sampling,
)
from .epi import make_skipped_caipi_sampling
from .noncartesian import (
    calc_golden_angles,
    calc_raga_angles,
    calc_tiny_golden_angles,
    calc_uniform_angles,
    make_golden_means_3d_sampling,
    make_radial_sampling,
    make_spiral_phyllotaxis_sampling,
)
from .ordering import calc_chunk_indices, make_linear_order, make_outer_inner_order, make_outer_product
from .slices import SliceGroup, make_slice_groups

__all__ = [
    "SamplingPattern",
    "SliceGroup",
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
    "make_caipirinha_sampling",
    "make_fse_linear_order",
    "make_fse_radial_adaptive_order",
    "make_fse_radial_order",
    "make_fse_shuffling_order",
    "make_golden_means_3d_sampling",
    "make_linear_order",
    "make_outer_inner_order",
    "make_outer_product",
    "make_poisson_disc_sampling",
    "make_radial_sampling",
    "make_random_sampling",
    "make_skipped_caipi_sampling",
    "make_slice_groups",
    "make_spiral_phyllotaxis_sampling",
]
