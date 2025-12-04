"""Spiral ordering strategy."""

__all__ = ["SpiralOrdering"]

import numpy as np
from numpy.typing import NDArray

from ._base import OrderingStrategy


class SpiralOrdering(OrderingStrategy):
    """
    Spiral ordering for 2D k-space coordinates.
    
    Orders points in a spiral pattern starting from the center. 
    Points are sorted by angle at each radial distance, creating
    a continuous spiral trajectory through k-space.
    
    Parameters
    ----------
    center : NDArray | None
        Custom center coordinates. Shape: (2,). 
        If None, uses the mean of coordinates as center.
    clockwise : bool
        If True, spiral goes clockwise.  If False, counter-clockwise. 
        Default is False (counter-clockwise). 
    start_angle : float
        Starting angle in radians. Default is 0.0 (positive x-axis). 
        
    Examples
    --------
    >>> # Default spiral ordering
    >>> strategy = SpiralOrdering()
    >>> ky, kz = np.meshgrid(np.arange(64) - 32, np.arange(64) - 32, indexing='ij')
    >>> coords = np. stack([ky. ravel(), kz.ravel()])
    >>> order = strategy.compute_order(coords)
    
    >>> # Clockwise spiral starting from top
    >>> strategy = SpiralOrdering(clockwise=True, start_angle=np.pi/2)
    
    Notes
    -----
    This strategy is designed for 2D coordinates.  For 1D, it falls back
    to center-out ordering.  For 3D+, only the first two dimensions are
    used for angle computation.
    """
    
    def __init__(
        self,
        center: NDArray | None = None,
        clockwise: bool = False,
        start_angle: float = 0.0,
    ):
        self._center = np.asarray(center) if center is not None else None
        self._clockwise = clockwise
        self._start_angle = float(start_angle)
    
    @property
    def name(self) -> str:
        direction = "cw" if self._clockwise else "ccw"
        return f"spiral_{direction}"
    
    @property
    def center(self) -> NDArray | None:
        """Return custom center coordinates."""
        return self._center
    
    @property
    def clockwise(self) -> bool:
        """Return whether spiral is clockwise."""
        return self._clockwise
    
    @property
    def start_angle(self) -> float:
        """Return starting angle in radians."""
        return self._start_angle
    
    def compute_order(
        self,
        coordinates: NDArray,
        mask: NDArray[bool] | None = None,
        n_segments: int = 1,
    ) -> NDArray[int]:
        """
        Compute spiral acquisition order.
        
        Parameters
        ----------
        coordinates : NDArray
            Point coordinates. Shape: (n_points,) or (n_dims, n_points). 
            For 2D+, uses first two dimensions for spiral ordering.
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
        
        # For 1D, fall back to center-out
        if n_dims == 1:
            return self._compute_1d_order(masked_coords, mask, n_segments)
        
        # Compute center
        if self._center is not None:
            if self._center.shape[0] < 2:
                raise ValueError(
                    f"center must have at least 2 elements, got {self._center. shape}"
                )
            center = self._center[:2]. reshape(-1, 1)
        else:
            center = masked_coords[:2].mean(axis=1, keepdims=True)
        
        # Compute radius and angle from center (using first 2 dims)
        delta = masked_coords[:2] - center
        radius = np.sqrt(delta[0] ** 2 + delta[1] ** 2)
        angle = np.arctan2(delta[1], delta[0])
        
        # Adjust angle for start_angle and direction
        angle = angle - self._start_angle
        if self._clockwise:
            angle = -angle
        
        # Normalize angle to [0, 2*pi)
        angle = angle % (2 * np.pi)
        
        # Create spiral by combining radius and angle
        # Key insight: for a spiral, we want to traverse angles while
        # gradually increasing radius.  We achieve this by:
        # 1.  Quantizing radius into shells
        # 2. Adding a full rotation (2*pi) for each shell
        
        # Determine number of shells based on unique radii
        radius_tol = 1e-10
        if radius.max() > radius_tol:
            # Normalize radius to [0, 1]
            radius_norm = radius / radius. max()
            # Estimate number of shells from data
            n_shells = max(1, int(np.sqrt(n_sampled)))
            shell_idx = np.floor(radius_norm * n_shells).astype(int)
            shell_idx = np.clip(shell_idx, 0, n_shells - 1)
        else:
            # All points at center
            shell_idx = np.zeros(n_sampled, dtype=int)
            n_shells = 1
        
        # Spiral parameter: shell index + normalized angle
        spiral_param = shell_idx + angle / (2 * np.pi)
        
        # Sort by spiral parameter
        order = np.argsort(spiral_param)
        
        return self._apply_mask_and_reshape(order, mask, n_segments)
    
    def _compute_1d_order(
        self,
        masked_coords: NDArray,
        mask: NDArray[bool],
        n_segments: int,
    ) -> NDArray[int]:
        """Compute center-out order for 1D case."""
        center = masked_coords. mean()
        radius = np.abs(masked_coords[0] - center)
        order = np.argsort(radius)
        return self._apply_mask_and_reshape(order, mask, n_segments)
    
    def __repr__(self) -> str:
        parts = []
        if self._center is not None:
            parts.append(f"center={self._center. tolist()}")
        if self._clockwise:
            parts. append("clockwise=True")
        if self._start_angle != 0.0:
            parts.append(f"start_angle={self._start_angle}")
        
        if parts:
            return f"SpiralOrdering({', '. join(parts)})"
        return "SpiralOrdering()"