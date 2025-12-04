"""Linear (sequential) ordering strategy."""

__all__ = ["LinearOrdering"]

from typing import Sequence

import numpy as np
from numpy. typing import NDArray

from ._base import OrderingStrategy


class LinearOrdering(OrderingStrategy):
    """
    Linear (sequential) ordering along specified dimension priority. 

    Acquires points in nested loop order, with the first dimension in
    `dim_priority` being the outermost (slowest varying) loop and the
    last being the innermost (fastest varying) loop. 

    Parameters
    ----------
    dim_priority : Sequence[str] | None
        Order of dimensions from outer to inner loop. If None, uses the
        default dim_labels order from the trajectory data.
    reverse : bool | dict[str, bool]
        Whether to reverse the ordering for each dimension.
        If a single bool, applies to all dimensions. 
        If a dict, maps dimension labels to their reverse flag. 

    Examples
    --------
    >>> # 2D Cartesian: k1 outer loop, k2 inner loop (row-by-row)
    >>> strategy = LinearOrdering(dim_priority=['k1', 'k2'])

    >>> # Same but k2 varies first (column-by-column)
    >>> strategy = LinearOrdering(dim_priority=['k2', 'k1'])

    >>> # Reverse k1 direction (top-to-bottom instead of bottom-to-top)
    >>> strategy = LinearOrdering(dim_priority=['k1', 'k2'], reverse={'k1': True})

    >>> # 3D Cartesian with k2 outer, k1 middle, averaging innermost
    >>> strategy = LinearOrdering(dim_priority=['k2', 'k1', 'avg'])
    """

    def __init__(
        self,
        dim_priority: Sequence[str] | None = None,
        reverse: bool | dict[str, bool] = False,
    ):
        self._dim_priority = tuple(dim_priority) if dim_priority else None
        self._reverse = reverse

    @property
    def name(self) -> str:
        return "linear"

    @property
    def dim_priority(self) -> tuple[str, ...] | None:
        """Return the dimension priority (outer to inner)."""
        return self._dim_priority

    def compute_order(
        self,
        scaling: dict[str, NDArray],
        indices: dict[str, NDArray],
        dim_labels: tuple[str, ... ],
    ) -> NDArray[np. intp]:
        """
        Compute linear acquisition order.

        Parameters
        ----------
        scaling : dict[str, NDArray]
            Scaling factors for each dimension (already masked, 1D arrays).
        indices : dict[str, NDArray]
            Grid indices for each dimension (already masked, 1D arrays).
        dim_labels : tuple[str, ...]
            Ordered dimension labels.

        Returns
        -------
        order : NDArray[np.intp]
            Indices that sort the points into linear acquisition order.
        """
        priority = self._dim_priority if self._dim_priority else dim_labels

        # Validate that all priority dimensions exist
        for dim in priority:
            if dim not in indices:
                raise ValueError(
                    f"Dimension '{dim}' in dim_priority not found in data.  "
                    f"Available: {list(indices. keys())}"
                )

        # Build sort keys for lexsort
        # lexsort sorts by last key first, so we need to reverse the priority
        # to get outer-to-inner ordering
        sort_keys = []
        for dim in reversed(priority):
            key = indices[dim]. copy()

            # Handle reverse flag
            if self._should_reverse(dim):
                key = -key

            sort_keys.append(key)

        return np.lexsort(sort_keys)

    def _should_reverse(self, dim: str) -> bool:
        """Check if dimension should be reversed."""
        if isinstance(self._reverse, bool):
            return self._reverse
        return self._reverse. get(dim, False)

    def __repr__(self) -> str:
        return f"LinearOrdering(dim_priority={self._dim_priority}, reverse={self._reverse})"