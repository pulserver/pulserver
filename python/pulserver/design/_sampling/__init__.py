"""Sampling patterns, slice schedules, and the composable tools behind them."""

from . import cartesian, counters, epi, noncartesian, ordering, slices
from ._axes import EncodingAxis
from ._scanloop import ScanLoop
from .cartesian import (
    calc_encoding_scales,
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
from .counters import make_counter_loop
from .epi import make_skipped_caipi_order
from .noncartesian import (
    calc_golden_angles,
    calc_raga_angles,
    calc_tiny_golden_angles,
    calc_uniform_angles,
    make_golden_means_3d_tilt,
    make_radial_tilt,
    make_rotated_segment_tilt,
    make_spiral_phyllotaxis_tilt,
)
from .ordering import calc_chunk_indices, calc_traversal_order
from .patterns import (
    make_cartesian_sampling,
    make_epi_sampling,
    make_noncartesian_2d_sampling,
    make_noncartesian_projection_sampling,
    make_rotated_projection_sampling,
)
from .slices import make_slice_loop

__all__ = [
    "EncodingAxis",
    "ScanLoop",
    "cartesian",
    "counters",
    "epi",
    "noncartesian",
    "ordering",
    "slices",
    "calc_chunk_indices",
    "calc_encoding_scales",
    "calc_golden_angles",
    "calc_raga_angles",
    "calc_sampled_lines",
    "calc_tiny_golden_angles",
    "calc_traversal_order",
    "calc_uniform_angles",
    "make_caipirinha_mask",
    "make_cartesian_sampling",
    "make_centric_order",
    "make_counter_loop",
    "make_epi_sampling",
    "make_golden_means_3d_tilt",
    "make_linear_order",
    "make_noncartesian_2d_sampling",
    "make_noncartesian_projection_sampling",
    "make_poisson_disc_mask",
    "make_radial_adaptive_order",
    "make_radial_order",
    "make_radial_tilt",
    "make_random_mask",
    "make_rotated_projection_sampling",
    "make_rotated_segment_tilt",
    "make_shuffling_order",
    "make_skipped_caipi_order",
    "make_slice_loop",
    "make_spiral_phyllotaxis_tilt",
]
