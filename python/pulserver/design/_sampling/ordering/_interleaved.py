"""Interleaved ordering strategy."""

__all__ = ["InterleavedOrdering"]

import numpy as np
from numpy.typing import NDArray

from ._base import OrderingStrategy


class InterleavedOrdering(OrderingStrategy):
    """
    Interleaved ordering for sequential coordinates.
    
    Divides coordinates into interleaved groups based on modular arithmetic. 
    Commonly used for slice ordering to minimize crosstalk between
    adjacent slices.
    
    Parameters
    ----------
    n_interleaves : int
        Number of interleave groups.  Default is 2 (even/odd).
    order_within : str
        Ordering within each interleave group. 
        - "ascending": Low to high values (default)
        - "descending": High to low values
        
    Examples
    --------
    >>> # Even/odd interleaving (slices 0,2,4,6 then 1,3,5,7)
    >>> strategy = InterleavedOrdering(n_interleaves=2)
    >>> order = strategy.compute_order(np.arange(8))
    
    >>> # Three-way interleaving
    >>> strategy = InterleavedOrdering(n_interleaves=3)
    
    >>> # Descending within groups
    >>> strategy = InterleavedOrdering(n_interleaves=2, order_within="descending")
    
    Notes
    -----
    This strategy expects 1D integer coordinates (e.g., slice indices).
    For 2D interleaved patterns like CAIPIRINHA, use a 2D strategy or
    provide a pre-computed mask.
    """
    
    def __init__(
        self,
        n_interleaves: int = 2,
        order_within: str = "ascending",
    ):
        if n_interleaves < 1:
            raise ValueError(f"n_interleaves must be >= 1, got {n_interleaves}")
        if order_within not in ("ascending", "descending"):
            raise ValueError(
                f"order_within must be 'ascending' or 'descending', "
                f"got '{order_within}'"
            )
        
        self._n_interleaves = n_interleaves
        self._order_within = order_within
    
    @property
    def name(self) -> str:
        base = f"interleaved_{self._n_interleaves}"
        if self._order_within == "descending":
            base += "_desc"
        return base
    
    @property
    def n_interleaves(self) -> int:
        """Return number of interleave groups."""
        return self._n_interleaves
    
    @property
    def order_within(self) -> str:
        """Return ordering within each group."""
        return self._order_within
    
    def compute_order(
        self,
        coordinates: NDArray,
        mask: NDArray[bool] | None = None,
        n_segments: int = 1,
    ) -> NDArray[int]:
        """
        Compute interleaved acquisition order. 
        
        Parameters
        ----------
        coordinates : NDArray
            Integer coordinates (1D).  Typically slice indices.
        mask : NDArray[bool] | None
            Sampling mask.  If None, all points are sampled. 
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
        
        # Check 1D
        if coordinates.shape[0] != 1:
            raise ValueError(
                f"InterleavedOrdering requires 1D coordinates, "
                f"got {coordinates.shape[0]} dimensions"
            )
        
        # Extract masked coordinates as integers
        masked_coords = coordinates[0, mask]. astype(int)
        
        # Compute interleave group for each point
        groups = masked_coords % self._n_interleaves
        
        # Sort within each group
        if self._order_within == "ascending":
            # Primary: group, Secondary: coordinate value
            sort_keys = (masked_coords, groups)
        else:
            # Primary: group, Secondary: -coordinate value (descending)
            sort_keys = (-masked_coords, groups)
        
        # lexsort uses last key as primary
        order = np.lexsort(sort_keys)
        
        return self._apply_mask_and_reshape(order, mask, n_segments)
    
    def __repr__(self) -> str:
        parts = [f"n_interleaves={self._n_interleaves}"]
        if self._order_within != "ascending":
            parts. append(f"order_within='{self._order_within}'")
        return f"InterleavedOrdering({', '.join(parts)})"