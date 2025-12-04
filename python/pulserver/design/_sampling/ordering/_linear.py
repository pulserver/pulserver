"""Linear ordering strategy."""

__all__ = ["LinearOrdering"]

import numpy as np
from numpy.typing import NDArray

from ._base import OrderingStrategy


class LinearOrdering(OrderingStrategy):
    """
    Linear (sequential) ordering of coordinates.
    
    Orders points sequentially along one or more dimensions. 
    For multidimensional coordinates, uses lexicographic ordering
    with configurable axis priority.
    
    Parameters
    ----------
    reverse : bool
        If True, order from high to low values.  Default is False.
    axis_priority : tuple[int, ... ] | None
        For ND coordinates, specifies axis ordering priority.
        First axis in tuple is primary sort key.
        If None, uses natural order (0, 1, 2, ...). 
        
    Examples
    --------
    >>> # Simple 1D linear ordering
    >>> strategy = LinearOrdering()
    >>> order = strategy.compute_order(np.array([3, 1, 4, 1, 5, 9, 2, 6]))
    
    >>> # Reverse ordering (high to low)
    >>> strategy = LinearOrdering(reverse=True)
    
    >>> # 2D with ky as primary, kz as secondary
    >>> strategy = LinearOrdering(axis_priority=(0, 1))
    
    >>> # 2D with kz as primary, ky as secondary
    >>> strategy = LinearOrdering(axis_priority=(1, 0))
    """
    
    def __init__(
        self,
        reverse: bool = False,
        axis_priority: tuple[int, ...] | None = None,
    ):
        self._reverse = reverse
        self._axis_priority = axis_priority
    
    @property
    def name(self) -> str:
        base = "linear"
        if self._reverse:
            base += "_reverse"
        return base
    
    @property
    def reverse(self) -> bool:
        """Return whether ordering is reversed."""
        return self._reverse
    
    @property
    def axis_priority(self) -> tuple[int, ... ] | None:
        """Return axis priority for multidimensional ordering."""
        return self._axis_priority
    
    def compute_order(
        self,
        coordinates: NDArray,
        mask: NDArray[bool] | None = None,
        n_segments: int = 1,
    ) -> NDArray[int]:
        """
        Compute linear acquisition order. 
        
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
        
        # Extract masked coordinates
        masked_coords = coordinates[:, mask]
        n_dims = masked_coords.shape[0]
        
        # Compute ordering
        if n_dims == 1:
            order = np.argsort(masked_coords[0])
        else:
            # Determine axis priority
            if self._axis_priority is not None:
                priority = self._axis_priority
            else:
                priority = tuple(range(n_dims))
            
            # Validate priority
            if len(priority) != n_dims or set(priority) != set(range(n_dims)):
                raise ValueError(
                    f"axis_priority {priority} invalid for {n_dims} dimensions"
                )
            
            # lexsort uses last key as primary, so reverse the priority
            keys = [masked_coords[i] for i in reversed(priority)]
            order = np.lexsort(keys)
        
        if self._reverse:
            order = order[::-1]
        
        return self._apply_mask_and_reshape(order, mask, n_segments)
    
    def __repr__(self) -> str:
        parts = []
        if self._reverse:
            parts.append("reverse=True")
        if self._axis_priority is not None:
            parts.append(f"axis_priority={self._axis_priority}")
        
        if parts:
            return f"LinearOrdering({', '.join(parts)})"
        return "LinearOrdering()"