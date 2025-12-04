"""Utility functions for trajectory ordering."""

__all__ = ["compose_orderings", "tile_repeat_order"]

import numpy as np
from numpy.typing import NDArray


def tile_repeat_order(
    *orderings: NDArray[np.intp],
) -> NDArray[np.intp]:
    """
    Combine multiple 1D orderings into a single flattened ordering. 

    Uses repeat/tile logic: first ordering is outermost (slowest varying),
    last is innermost (fastest varying).

    Parameters
    ----------
    *orderings : NDArray[np.intp]
        Variable number of 1D orderings.  Each is an array of indices
        representing the acquisition order for that dimension.

    Returns
    -------
    combined : NDArray[np.intp]
        Combined ordering as a 2D array of shape (n_dims, total_points).
        Row i contains the ordered indices for dimension i. 

    Examples
    --------
    >>> # Slice ordering: [0, 2, 1, 3] (interleaved)
    >>> # Phase encoding: [3, 4, 2, 5, 1, 6, 0, 7] (center-out)
    >>> slc_order = np.array([0, 2, 1, 3])
    >>> lin_order = np.array([3, 4, 2, 5, 1, 6, 0, 7])
    >>> combined = tile_repeat_order(slc_order, lin_order)
    >>> # combined[0] = slc indices (each repeated 8 times)
    >>> # combined[1] = lin indices (tiled 4 times)
    """
    if len(orderings) == 0:
        raise ValueError("At least one ordering is required")

    if len(orderings) == 1:
        return orderings[0].reshape(1, -1)

    # Compute total points and repetition factors
    sizes = [len(o) for o in orderings]
    total_points = int(np.prod(sizes))

    # Build combined ordering
    combined = np.zeros((len(orderings), total_points), dtype=np. intp)

    for i, ordering in enumerate(orderings):
        # Repeat factor: product of all sizes after this dimension
        repeat = int(np.prod(sizes[i + 1 :]))
        # Tile factor: product of all sizes before this dimension
        tile = int(np.prod(sizes[:i]))

        # np.repeat then np.tile
        expanded = np.tile(np.repeat(ordering, repeat), tile)
        combined[i] = expanded

    return combined


def compose_orderings(
    orderings: dict[str, NDArray[np.intp]],
    dim_order: tuple[str, ...],
) -> dict[str, NDArray[np.intp]]:
    """
    Combine multiple named 1D orderings into composed orderings.

    Convenience wrapper around tile_repeat_order that works with
    labeled dimensions.

    Parameters
    ----------
    orderings : dict[str, NDArray[np.intp]]
        Dictionary mapping dimension labels to their 1D orderings.
    dim_order : tuple[str, ...]
        Order of dimensions from outer (slowest) to inner (fastest). 

    Returns
    -------
    composed : dict[str, NDArray[np.intp]]
        Dictionary mapping dimension labels to their composed orderings. 

    Examples
    --------
    >>> slc_order = np.array([0, 2, 1, 3])  # interleaved
    >>> lin_order = np.array([3, 4, 2, 5, 1, 6, 0, 7])  # center-out
    >>> composed = compose_orderings(
    ...     orderings={'slc': slc_order, 'lin': lin_order},
    ...     dim_order=('slc', 'lin'),  # slc outer, lin inner
    ... )
    >>> # composed['slc'] = [0,0,0,0,0,0,0,0, 2,2,2,2,2,2,2,2, ...]
    >>> # composed['lin'] = [3,4,2,5,1,6,0,7, 3,4,2,5,1,6,0,7, ...]
    """
    for dim in dim_order:
        if dim not in orderings:
            raise ValueError(f"Dimension '{dim}' not found in orderings")

    # Extract orderings in specified order
    ordered_arrays = [orderings[dim] for dim in dim_order]

    # Combine
    combined = tile_repeat_order(*ordered_arrays)

    # Package back into dict
    return {dim: combined[i] for i, dim in enumerate(dim_order)}