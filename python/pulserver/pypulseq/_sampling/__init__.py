"""K-space support, acquisition-order, and slice-group sampling plans."""

from . import cartesian, epi, noncartesian, ordering, slices
from ._pattern import SamplingPattern
from .cartesian import (
    caipirinha_mask,
    fse_linear_order,
    fse_radial_adaptive_order,
    fse_radial_order,
    fse_shuffling_order,
    poisson_disc_mask,
    random_mask,
    sampled_lines,
)
from .epi import from_relative_shifts, skipped_caipi
from .noncartesian import (
    directions_to_rotations,
    golden_angles,
    golden_means_3d,
    radial_2d,
    spiral_phyllotaxis,
    uniform_angles,
)
from .ordering import chunk_indices, linear_order, outer_inner_order, outer_product
from .slices import SliceGroup, slice_groups

from_mask = cartesian.from_mask

__all__ = [
    "SamplingPattern",
    "SliceGroup",
    "cartesian",
    "epi",
    "noncartesian",
    "ordering",
    "slices",
    "chunk_indices",
    "linear_order",
    "outer_inner_order",
    "outer_product",
    "sampled_lines",
    "fse_linear_order",
    "fse_radial_order",
    "fse_radial_adaptive_order",
    "fse_shuffling_order",
    "random_mask",
    "caipirinha_mask",
    "poisson_disc_mask",
    "from_mask",
    "from_relative_shifts",
    "skipped_caipi",
    "radial_2d",
    "golden_angles",
    "uniform_angles",
    "golden_means_3d",
    "spiral_phyllotaxis",
    "directions_to_rotations",
    "slice_groups",
]
