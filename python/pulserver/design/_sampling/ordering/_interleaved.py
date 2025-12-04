"""Interleaved ordering strategy for 1D acquisitions."""

__all__ = ["InterleavedOrdering"]

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ._base import OrderingStrategy


class InterleavedOrdering(OrderingStrategy):
    """
    Interleaved ordering for 1D acquisitions (e.g., slice ordering).

    Divides points into N interleaves (groups) and acquires each group
    sequentially.  Commonly used for multi-slice imaging to reduce
    crosstalk between adjacent slices.

    Parameters
    ----------
    n_interleaves : int
        Number of interleaves (groups).  Default is 2 (even/odd).
    order_within : str
        Ordering within each interleave:
        - 'ascending': Low to high index (default)
        - 'descending': High to low index
        - 'center_out': Center to edges
    interleave_order : Sequence[int] | str
        Order in which to acquire interleaves:
        - 'sequential': 0, 1, 2, ...  (default)
        - 'reversed': N-1, N-2, ..., 0
        - 'center_out': Start from middle interleave
        - Sequence[int]: Custom order, e.g., [0, 2, 1, 3]
    dim : str | None
        Dimension to interleave.  If None, uses first dimension.
        Only used for validation/clarity since this is a 1D strategy.

    Examples
    --------
    >>> # Standard even/odd interleaving
    >>> strategy = InterleavedOrdering(n_interleaves=2)

    >>> # 3-shot interleaving
    >>> strategy = InterleavedOrdering(n_interleaves=3)

    >>> # Even/odd with descending order within each group
    >>> strategy = InterleavedOrdering(n_interleaves=2, order_within='descending')

    >>> # Custom interleave order: odd slices first, then even
    >>> strategy = InterleavedOrdering(n_interleaves=2, interleave_order=[1, 0])

    >>> # 4-shot with center-out within each group
    >>> strategy = InterleavedOrdering(
    ...     n_interleaves=4,
    ...     order_within='center_out'
    ... )
    """

    def __init__(
        self,
        n_interleaves: int = 2,
        order_within: str = "ascending",
        interleave_order: Sequence[int] | str = "sequential",
        dim: str | None = None,
    ):
        if n_interleaves < 1:
            raise ValueError(f"n_interleaves must be >= 1, got {n_interleaves}")

        valid_within = ("ascending", "descending", "center_out")
        if order_within not in valid_within:
            raise ValueError(
                f"order_within must be one of {valid_within}, got '{order_within}'"
            )

        self._n_interleaves = n_interleaves
        self._order_within = order_within
        self._interleave_order = interleave_order
        self._dim = dim

    @property
    def name(self) -> str:
        return f"interleaved_{self._n_interleaves}"

    @property
    def n_interleaves(self) -> int:
        """Return the number of interleaves."""
        return self._n_interleaves

    @property
    def order_within(self) -> str:
        """Return the ordering within each interleave."""
        return self._order_within

    def compute_order(
        self,
        scaling: dict[str, NDArray],
        indices: dict[str, NDArray],
        dim_labels: tuple[str, ...],
    ) -> NDArray[np.intp]:
        """
        Compute interleaved acquisition order.

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
            Indices that sort the points into interleaved acquisition order.
        """
        # Determine which dimension to use
        dim = self._dim if self._dim else dim_labels[0]

        if dim not in indices:
            raise ValueError(
                f"Dimension '{dim}' not found in data.  "
                f"Available: {list(indices. keys())}"
            )

        idx = indices[dim]

        # Assign each point to an interleave based on its index
        interleave_ids = idx % self._n_interleaves

        # Get the order in which to process interleaves
        interleave_sequence = self._get_interleave_sequence()

        # Build the final ordering
        order_list = []
        for interleave_id in interleave_sequence:
            # Get points belonging to this interleave
            mask = interleave_ids == interleave_id
            point_indices = np.where(mask)[0]

            if len(point_indices) == 0:
                continue

            # Get the original indices for sorting within interleave
            original_indices = idx[point_indices]

            # Sort within interleave according to order_within
            within_order = self._compute_within_order(original_indices)
            sorted_points = point_indices[within_order]

            order_list.append(sorted_points)

        if order_list:
            return np.concatenate(order_list).astype(np.intp)
        else:
            return np.array([], dtype=np.intp)

    def _get_interleave_sequence(self) -> list[int]:
        """Get the sequence in which to acquire interleaves."""
        n = self._n_interleaves

        if isinstance(self._interleave_order, str):
            if self._interleave_order == "sequential":
                return list(range(n))
            elif self._interleave_order == "reversed":
                return list(range(n - 1, -1, -1))
            elif self._interleave_order == "center_out":
                # Start from middle, alternate outward
                mid = n // 2
                sequence = [mid]
                for offset in range(1, n):
                    if mid + offset < n:
                        sequence.append(mid + offset)
                    if mid - offset >= 0:
                        sequence.append(mid - offset)
                return sequence
            else:
                raise ValueError(
                    f"Unknown interleave_order: '{self._interleave_order}'"
                )
        else:
            # Custom sequence
            sequence = list(self._interleave_order)
            if set(sequence) != set(range(n)):
                raise ValueError(
                    f"Custom interleave_order must be a permutation of "
                    f"0.. {n-1}, got {sequence}"
                )
            return sequence

    def _compute_within_order(self, indices: NDArray) -> NDArray[np.intp]:
        """Compute ordering within an interleave."""
        if self._order_within == "ascending":
            return np.argsort(indices)
        elif self._order_within == "descending":
            return np.argsort(indices)[::-1]
        elif self._order_within == "center_out":
            center = (np.max(indices) + np.min(indices)) / 2
            distances = np.abs(indices - center)
            return np.argsort(distances)
        else:
            # Should not reach here due to __init__ validation
            raise ValueError(f"Unknown order_within: '{self._order_within}'")

    def __repr__(self) -> str:
        return (
            f"InterleavedOrdering(n_interleaves={self._n_interleaves}, "
            f"order_within='{self._order_within}', "
            f"interleave_order={self._interleave_order})"
        )
