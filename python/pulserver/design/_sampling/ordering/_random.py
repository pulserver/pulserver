"""Random ordering strategy."""

__all__ = ["RandomOrdering"]

import numpy as np
from numpy.typing import NDArray

from ._base import OrderingStrategy


class RandomOrdering(OrderingStrategy):
    """
    Random (shuffled) ordering of coordinates.
    
    Randomly permutes the acquisition order.  Useful for compressed sensing
    and incoherent sampling patterns.
    
    Parameters
    ----------
    seed : int | None
        Random seed for reproducibility.  If None, uses non-deterministic
        random state. 
        
    Examples
    --------
    >>> # Random ordering with fixed seed for reproducibility
    >>> strategy = RandomOrdering(seed=42)
    >>> order = strategy.compute_order(np.arange(256))
    
    >>> # Non-deterministic random ordering
    >>> strategy = RandomOrdering()
    >>> order = strategy.compute_order(np.arange(256))
    """
    
    def __init__(self, seed: int | None = None):
        self._seed = seed
    
    @property
    def name(self) -> str:
        return "random"
    
    @property
    def seed(self) -> int | None:
        """Return random seed."""
        return self._seed
    
    def compute_order(
        self,
        coordinates: NDArray,
        mask: NDArray[bool] | None = None,
        n_segments: int = 1,
    ) -> NDArray[int]:
        """
        Compute random acquisition order.
        
        Parameters
        ----------
        coordinates : NDArray
            Point coordinates.  Shape: (n_points,) or (n_dims, n_points). 
        mask : NDArray[bool] | None
            Sampling mask. If None, all points are sampled. 
        n_segments : int
            Number of segments to divide acquisition into.
            
        Returns
        -------
        order : NDArray[int]
            Shape: (n_segments, n_points_per_segment). 
        """
        coordinates, mask, n_sampled = self._validate_inputs(
            coordinates, mask, n_segments
        )
        
        # Create random generator
        rng = np.random.default_rng(self._seed)
        
        # Generate random permutation
        order = rng.permutation(n_sampled).astype(int)
        
        return self._apply_mask_and_reshape(order, mask, n_segments)
    
    def __repr__(self) -> str:
        if self._seed is not None:
            return f"RandomOrdering(seed={self._seed})"
        return "RandomOrdering()"