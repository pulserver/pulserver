"""Random ordering strategy."""

__all__ = ["RandomOrdering"]

import numpy as np
from numpy.typing import NDArray

from ._base import OrderingStrategy


class RandomOrdering(OrderingStrategy):
    """
    Random (shuffled) ordering.

    Randomly permutes the acquisition order.  Useful for:
    - Incoherent aliasing artifacts in compressed sensing
    - Breaking up systematic errors
    - Randomized benchmarking

    Parameters
    ----------
    seed : int | None
        Random seed for reproducibility.  If None, uses non-deterministic
        random state.

    Examples
    --------
    >>> # Random ordering (non-reproducible)
    >>> strategy = RandomOrdering()

    >>> # Reproducible random ordering
    >>> strategy = RandomOrdering(seed=42)
    """

    def __init__(self, seed: int | None = None):
        self._seed = seed

    @property
    def name(self) -> str:
        return "random"

    @property
    def seed(self) -> int | None:
        """Return the random seed."""
        return self._seed

    def compute_order(
        self,
        scaling: dict[str, NDArray],
        indices: dict[str, NDArray],
        dim_labels: tuple[str, ...],
    ) -> NDArray[np.intp]:
        """
        Compute random acquisition order.

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
            Randomly permuted indices.
        """
        n_points = len(indices[dim_labels[0]])
        rng = np.random.default_rng(self._seed)
        order = rng.permutation(n_points)
        return order.astype(np.intp)

    def __repr__(self) -> str:
        return f"RandomOrdering(seed={self._seed})"
