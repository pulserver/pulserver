"""Utility functions for composing and manipulating orderings."""

__all__ = [
    "apply_order",
    "reorder_within_segments",
    "compose_orderings",
    "flatten_order",
]

import numpy as np
from numpy.typing import NDArray

from ._base import OrderingStrategy


def apply_order(
    order: NDArray[int],
    coordinates: NDArray,
    scaling: NDArray | None = None,
) -> tuple[NDArray, NDArray | None]:
    """
    Apply ordering to coordinates and optional scaling arrays.

    Parameters
    ----------
    order : NDArray[int]
        Ordering indices from OrderingStrategy.compute_order(). 
        Shape: (n_segments, n_points_per_segment). 
    coordinates : NDArray
        Coordinate array.  Shape: (n_dims, n_points) or (n_points,).
    scaling : NDArray | None
        Optional scaling array.  Same shape as coordinates. 

    Returns
    -------
    sorted_coordinates : NDArray
        Coordinates in acquisition order. 
        Shape: (n_dims, n_segments, n_points_per_segment) or
        (n_segments, n_points_per_segment) for 1D. 
    sorted_scaling : NDArray | None
        Scaling in acquisition order, or None if scaling was None. 

    Examples
    --------
    >>> order = strategy.compute_order(coordinates, mask, n_segments=4)
    >>> sorted_coords, sorted_scaling = apply_order(order, coordinates[:, mask], scaling[:, mask])
    """
    coordinates = np.atleast_2d(coordinates)
    n_dims = coordinates.shape[0]

    # Apply order
    sorted_coords = coordinates[:, order]

    if scaling is not None:
        scaling = np.atleast_2d(scaling)
        sorted_scaling = scaling[:, order]
    else:
        sorted_scaling = None

    # Squeeze if 1D
    if n_dims == 1:
        sorted_coords = sorted_coords[0]
        if sorted_scaling is not None:
            sorted_scaling = sorted_scaling[0]

    return sorted_coords, sorted_scaling


def reorder_within_segments(
    order: NDArray[int],
    coordinates: NDArray,
    strategy: OrderingStrategy,
) -> NDArray[int]:
    """
    Apply a secondary ordering within each segment.

    Useful for contrast weighting, e.g., center-out ordering within
    each shot of a segmented acquisition.

    Parameters
    ----------
    order : NDArray[int]
        Original ordering.  Shape: (n_segments, n_points_per_segment). 
    coordinates : NDArray
        Original coordinates. Shape: (n_dims, n_points) or (n_points,).
    strategy : OrderingStrategy
        Strategy to apply within each segment. 

    Returns
    -------
    reordered : NDArray[int]
        New ordering with within-segment reordering applied. 
        Same shape as input order.

    Examples
    --------
    >>> # Spiral ordering, then center-out within each shot
    >>> order = SpiralOrdering(). compute_order(coords, n_segments=8)
    >>> order = reorder_within_segments(order, coords, CenterOutOrdering())
    """
    coordinates = np.atleast_2d(coordinates)
    n_segments, n_per_segment = order.shape

    result = np.empty_like(order)

    for seg in range(n_segments):
        seg_indices = order[seg]
        seg_coords = coordinates[:, seg_indices]

        # Compute within-segment order
        within_order = strategy.compute_order(seg_coords, mask=None, n_segments=1)
        within_order = within_order.ravel()

        # Apply within-segment reordering
        result[seg] = seg_indices[within_order]

    return result


def compose_orderings(
    outer_order: NDArray[int],
    inner_order: NDArray[int],
) -> NDArray[int]:
    """
    Compose two levels of ordering for nested segmentation.

    Useful for dynamic/multi-contrast acquisitions where each frame
    is itself segmented (e.g., multi-shot EPI with multiple time frames).

    Parameters
    ----------
    outer_order : NDArray[int]
        Outer-level ordering. Shape: (n_outer, n_middle).
        Indices refer to "blocks" or "blades". 
    inner_order : NDArray[int]
        Inner-level ordering. Shape: (1, n_inner) or (n_inner,).
        Indices within each block, assumed same for all blocks.

    Returns
    -------
    composed : NDArray[int]
        Combined ordering. Shape: (n_outer, n_middle, n_inner).
        Full indices into the original coordinate array.

    Examples
    --------
    >>> # 10 time frames, 4 EPI shots per frame, 64 lines per shot
    >>> # Outer: which shots go in which frame
    >>> outer_order = SpiralOrdering(). compute_order(blade_centers, n_segments=10)
    >>> # Inner: line ordering within each EPI shot (fixed pattern)
    >>> inner_order = LinearOrdering(). compute_order(np.arange(64))
    >>> # Compose
    >>> full_order = compose_orderings(outer_order, inner_order)
    >>> # full_order. shape = (10, 4, 64)

    Notes
    -----
    This function assumes that:
    1. outer_order contains indices to "blocks" of size n_inner
    2. inner_order is the same pattern applied within each block
    3. The original data is organized as blocks, each containing n_inner points

    The composed index is: block_index * n_inner + within_block_index
    """
    inner_order = np.atleast_2d(inner_order)
    if inner_order.shape[0] == 1:
        inner_order = inner_order.ravel()
    else:
        inner_order = inner_order.ravel()

    n_outer, n_middle = outer_order.shape
    n_inner = len(inner_order)

    result = np.empty((n_outer, n_middle, n_inner), dtype=np.intp)

    for o in range(n_outer):
        for m in range(n_middle):
            block_idx = outer_order[o, m]
            base_idx = block_idx * n_inner
            result[o, m, :] = base_idx + inner_order

    return result


def flatten_order(order: NDArray[int]) -> NDArray[int]:
    """
    Flatten a multi-dimensional order to 1D acquisition sequence.

    Parameters
    ----------
    order : NDArray[int]
        Ordering array. Shape: (n_segments, n_per_segment) or
        (n_outer, n_middle, n_inner) or any shape. 

    Returns
    -------
    flat_order : NDArray[int]
        Flattened ordering. Shape: (n_total,). 

    Examples
    --------
    >>> order = strategy.compute_order(coords, n_segments=4)
    >>> flat = flatten_order(order)
    >>> # Use flat order for sequential acquisition
    >>> for idx in flat:
    ...     acquire(coordinates[:, idx])
    """
    return order.ravel()