"""Private building blocks behind the :mod:`pulserver.design` scan-loop factories.

**This module is not public API.** It is the layer the shipped factories are
assembled from — traversal orders, undersampling masks, projection tilts and
the angle generators that feed them — and its names, signatures and semantics
may change without notice. Everything a plugin needs is reachable from
:mod:`pulserver.design` itself; if something here looks necessary, that is a
gap in the public factory layer and should be closed there instead.

Nothing here returns a ``ScanLoop``: these produce the plain index arrays,
boolean masks and rotation matrices the public factories hand to ``ScanLoop``
or to one of its ``from_mask`` / ``from_relative_shifts`` constructors.

``arbgrad`` — the raw slew-limited waveform core behind the non-Cartesian
readouts — is re-exported here for the same reason; the public entry point is
:func:`pulserver.design.traj2grad`.
"""

from __future__ import annotations

from . import _arbgrad as arbgrad
from ._sampling import (
    calc_chunk_indices,
    calc_golden_angles,
    calc_raga_angles,
    calc_sampled_lines,
    calc_tiny_golden_angles,
    calc_traversal_order,
    calc_uniform_angles,
    make_caipirinha_mask,
    make_centric_order,
    make_golden_means_3d_tilt,
    make_linear_order,
    make_poisson_disc_mask,
    make_radial_adaptive_order,
    make_radial_order,
    make_radial_tilt,
    make_random_mask,
    make_rotated_segment_tilt,
    make_shuffling_order,
    make_skipped_caipi_order,
    make_spiral_phyllotaxis_tilt,
)

__all__ = [
    "arbgrad",
    "calc_chunk_indices",
    "calc_golden_angles",
    "calc_raga_angles",
    "calc_sampled_lines",
    "calc_tiny_golden_angles",
    "calc_traversal_order",
    "calc_uniform_angles",
    "make_caipirinha_mask",
    "make_centric_order",
    "make_golden_means_3d_tilt",
    "make_linear_order",
    "make_poisson_disc_mask",
    "make_radial_adaptive_order",
    "make_radial_order",
    "make_radial_tilt",
    "make_random_mask",
    "make_rotated_segment_tilt",
    "make_shuffling_order",
    "make_skipped_caipi_order",
    "make_spiral_phyllotaxis_tilt",
]
