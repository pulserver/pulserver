"""Linear Ordering Strategies."""

__all__ = ["LinearOrdering"]

from typing import Optional, Sequence

import numpy as np

from numpy.typing import NDArray

from .base import OrderingStrategy


class LinearOrdering(OrderingStrategy):
    """
    Linear (sequential) ordering along specified dimension priority.

    Acquires points row-by-row (or column-by-column, etc.) based on
    the dimension priority order.

    Parameters
    ----------
    dim_priority : Sequence[str] | None
        Order of dimensions for sorting. If None, uses the default
        dim_labels order.  First dimension varies slowest.
    reverse : bool | Sequence[bool]
        Whether to reverse the ordering for each dimension.
        If a single bool, applies to all dimensions.
    """

    def __init__(
        self,
        dim_priority: Optional[Sequence[str]] = None,
        reverse: bool | Sequence[bool] = False,
    ):
        self.dim_priority = dim_priority
        self.reverse = reverse

    @property
    def name(self) -> str:
        return "linear"

    def compute_order(
        self,
        scaling: dict[str, NDArray],
        indices: dict[str, NDArray],
        dim_labels: tuple[str, ...],
    ) -> NDArray[int]:
        priority = self.dim_priority or dim_labels

        # Build sort keys (lexicographic sort)
        # Reverse priority so first dimension is primary sort key
        sort_keys = []
        for i, dim in enumerate(reversed(priority)):
            key = indices[dim].copy()

            # Handle reverse flag
            if isinstance(self.reverse, bool):
                if self.reverse:
                    key = -key
            elif self.reverse[len(priority) - 1 - i]:
                key = -key

            sort_keys.append(key)

        # Use lexsort (sorts by last key first, so we reversed above)
        return np.lexsort(sort_keys)

    def __repr__(self) -> str:
        return (
            f"LinearOrdering(dim_priority={self.dim_priority}, reverse={self. reverse})"
        )
