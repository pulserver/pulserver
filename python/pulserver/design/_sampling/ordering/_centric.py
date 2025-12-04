"""Center-out ordering strategy."""

__all__ = ["CenterOutOrdering"]

import numpy as np
from numpy.typing import NDArray

from ._base import OrderingStrategy


class CenterOutOrdering(OrderingStrategy):
    """
    Center-out ordering starting from the center of k-space.
    
    Acquires points starting from the k-space center and moving outward. 
    Points are ordered by radial distance from center.  For 2D+ data,
    angular ordering can be applied within each radial shell.
    
    Parameters
    ----------
    center : NDArray | None
        Custom center coordinates. Shape: (n_dims,). 
        If None, uses the mean of coordinates as center.
    angular_offset : float
        Global angular offset in radians. Applied to all angular positions.
        Useful for creating incoherent orderings across slices or contrasts.
        Default is 0.0.
        
    Examples
    --------
    >>> # Simple 1D center-out
    >>> strategy = CenterOutOrdering()
    >>> order = strategy.compute_order(np.arange(256) - 128)
    
    >>> # 2D center-out
    >>> ky, kz = np.meshgrid(np. arange(64) - 32, np.arange(64) - 32, indexing='ij')
    >>> coords = np.stack([ky. ravel(), kz.ravel()])
    >>> order = strategy.compute_order(coords)
    
    >>> # Per-slice ordering with angular offset for incoherence
    >>> for slc in range(n_slices):
    ...     offset = slc * 0.618 * 2 * np. pi  # Golden angle
    ...     strategy = CenterOutOrdering(angular_offset=offset)
    ...     orders[slc] = strategy.compute_order(coords)
    """
    
    def __init__(
        self,
        center: NDArray | None = None,
        angular_offset: float = 0.0,
    ):
        self._center = np.asarray(center) if center is not None else None
        self._angular_offset = float(angular_offset)
    
    @property
    def name(self) -> str:
        return "center_out"
    
    @property
    def center(self) -> NDArray | None:
        """Return custom center coordinates."""
        return self._center
    
    @property
    def angular_offset(self) -> float:
        """Return angular offset in radians."""
        return self._angular_offset
    
    def compute_order(
        self,
        coordinates: NDArray,
        mask: NDArray[bool] | None = None,
        n_segments: int = 1,
    ) -> NDArray[int]:
        """
        Compute center-out acquisition order. 
        
        Parameters
        ----------
        coordinates : NDArray
            Point coordinates. Shape: (n_points,) or (n_dims, n_points). 
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
        
        # Compute center
        if self._center is not None:
            if self._center.shape != (n_dims,):
                raise ValueError(
                    f"center shape {self._center. shape} does not match "
                    f"coordinates with {n_dims} dimensions"
                )
            center = self._center.reshape(-1, 1)
        else:
            center = masked_coords.mean(axis=1, keepdims=True)
        
        # Compute radial distance from center
        delta = masked_coords - center
        radius = np.sqrt(np.sum(delta ** 2, axis=0))
        
        # For 1D, just sort by radius
        if n_dims == 1:
            order = np.argsort(radius)
        else:
            # For 2D+, compute angle and use it as secondary sort key
            angle = np.arctan2(delta[1], delta[0])
            angle = angle + self._angular_offset
            
            # Normalize angle to [0, 2*pi)
            angle = angle % (2 * np.pi)
            
            # Quantize radius to create discrete shells
            # Use relative tolerance based on coordinate range
            radius_range = radius.max() - radius.min() if radius.max() > radius.min() else 1.0
            radius_quantized = np.round(radius / radius_range * 1e6). astype(np.int64)
            angle_quantized = np.round(angle * 1e6). astype(np. int64)
            
            # Sort by radius (primary), then angle (secondary)
            order = np.lexsort((angle_quantized, radius_quantized))
        
        return self._apply_mask_and_reshape(order, mask, n_segments)
    
    def __repr__(self) -> str:
        parts = []
        if self._center is not None:
            parts.append(f"center={self._center. tolist()}")
        if self._angular_offset != 0.0:
            parts.append(f"angular_offset={self._angular_offset}")
        
        if parts:
            return f"CenterOutOrdering({', '. join(parts)})"
        return "CenterOutOrdering()"