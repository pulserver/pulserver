"""Radial ordering strategy."""

__all__ = ["CenterOutOrdering", "FullSpokeOrdering"]

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
            radius_quantized = np.round(radius / radius_range * 1e6).astype(np.int64)
            angle_quantized = np.round(angle * 1e6). astype(np.int64)
            
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
    

class FullSpokeOrdering(OrderingStrategy):
    """
    Full spoke (radial) ordering for k-space coordinates. 

    Groups points into radial spokes emanating from the center. 
    Each spoke contains points at the same angle but different radii. 
    Spokes are ordered by angle, with configurable angular increment. 

    This is the natural ordering for radial trajectories where each
    readout acquires a full spoke through the k-space center. 

    Parameters
    ----------
    center : NDArray | None
        Custom center coordinates. Shape: (2,). 
        If None, uses the mean of coordinates as center.
    n_spokes : int | None
        Number of angular bins (spokes).  If None, automatically
        determined from the data.
    angular_offset : float
        Global angular offset in radians. Applied to all spoke angles.
        Useful for golden angle or rotated acquisitions.
        Default is 0.0. 
    bidirectional : bool
        If True, alternates spoke direction (in/out vs out/in). 
        Default is False (all spokes go outward from center). 

    Examples
    --------
    >>> # Default full spoke ordering
    >>> strategy = FullSpokeOrdering()
    >>> order = strategy.compute_order(coords)

    >>> # Golden angle rotation between segments
    >>> golden_angle = np.pi * (3 - np.sqrt(5))
    >>> strategy = FullSpokeOrdering(angular_offset=golden_angle)

    >>> # Bidirectional spokes for smoother gradient transitions
    >>> strategy = FullSpokeOrdering(bidirectional=True)
    """

    def __init__(
        self,
        center: NDArray | None = None,
        n_spokes: int | None = None,
        angular_offset: float = 0.0,
        bidirectional: bool = False,
    ):
        if n_spokes is not None and n_spokes < 1:
            raise ValueError(f"n_spokes must be >= 1, got {n_spokes}")

        self._center = np.asarray(center) if center is not None else None
        self._n_spokes = n_spokes
        self._angular_offset = float(angular_offset)
        self._bidirectional = bidirectional

    @property
    def name(self) -> str:
        base = "full_spoke"
        if self._bidirectional:
            base += "_bidir"
        return base

    @property
    def center(self) -> NDArray | None:
        """Return custom center coordinates."""
        return self._center

    @property
    def n_spokes(self) -> int | None:
        """Return number of spokes."""
        return self._n_spokes

    @property
    def angular_offset(self) -> float:
        """Return angular offset in radians."""
        return self._angular_offset

    @property
    def bidirectional(self) -> bool:
        """Return whether spokes alternate direction."""
        return self._bidirectional

    def compute_order(
        self,
        coordinates: NDArray,
        mask: NDArray[bool] | None = None,
        n_segments: int = 1,
    ) -> NDArray[int]:
        """
        Compute full spoke acquisition order.

        Parameters
        ----------
        coordinates : NDArray
            Point coordinates. Shape: (n_points,) or (n_dims, n_points). 
            For 2D+, uses first two dimensions for spoke ordering.
        mask : NDArray[bool] | None
            Sampling mask. If None, all points are sampled.
        n_segments : int
            Number of segments to divide acquisition into. 
            Typically corresponds to number of spoke groups.

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

        # For 1D, fall back to center-out with segments
        if n_dims == 1:
            return self._compute_1d_order(masked_coords, mask, n_segments)

        # Compute center
        if self._center is not None:
            if self._center.shape[0] < 2:
                raise ValueError(
                    f"center must have at least 2 elements, got {self._center.shape}"
                )
            center = self._center[:2].reshape(-1, 1)
        else:
            center = masked_coords[:2].mean(axis=1, keepdims=True)

        # Compute radius and angle from center (using first 2 dims)
        delta = masked_coords[:2] - center
        radius = np.sqrt(delta[0] ** 2 + delta[1] ** 2)
        angle = np.arctan2(delta[1], delta[0])

        # Apply angular offset
        angle = angle + self._angular_offset

        # Normalize angle to [0, 2*pi)
        angle = angle % (2 * np.pi)

        # Determine number of spokes
        if self._n_spokes is not None:
            n_spokes = self._n_spokes
        else:
            # Estimate from data: use n_segments as hint
            n_spokes = max(1, n_segments)

        # Assign points to spokes based on angle
        spoke_width = 2 * np.pi / n_spokes
        spoke_idx = np.floor(angle / spoke_width).astype(int)
        spoke_idx = np.clip(spoke_idx, 0, n_spokes - 1)

        # Sort spokes by their minimum angle to ensure consistent ordering
        spoke_order = np.argsort([
            angle[spoke_idx == s]. min() if np.any(spoke_idx == s) else np.inf
            for s in range(n_spokes)
        ])

        # Build order by iterating through spokes in sorted order
        order_list = []
        spoke_counter = 0
        for spoke in spoke_order:
            spoke_mask = spoke_idx == spoke
            if not np.any(spoke_mask):
                continue

            spoke_point_indices = np.where(spoke_mask)[0]
            spoke_radii = radius[spoke_mask]

            # Sort by radius within spoke
            radial_order = np.argsort(spoke_radii)

            # Apply bidirectional if needed
            if self._bidirectional and spoke_counter % 2 == 1:
                radial_order = radial_order[::-1]

            order_list.append(spoke_point_indices[radial_order])
            spoke_counter += 1

        order = np.concatenate(order_list)

        return self._apply_mask_and_reshape(order, mask, n_segments)

    def _compute_1d_order(
        self,
        masked_coords: NDArray,
        mask: NDArray[bool],
        n_segments: int,
    ) -> NDArray[int]:
        """Compute center-out order for 1D case."""
        center = masked_coords.mean()
        radius = np.abs(masked_coords[0] - center)
        order = np.argsort(radius)
        return self._apply_mask_and_reshape(order, mask, n_segments)

    def __repr__(self) -> str:
        parts = []
        if self._center is not None:
            parts.append(f"center={self._center. tolist()}")
        if self._n_spokes is not None:
            parts.append(f"n_spokes={self._n_spokes}")
        if self._angular_offset != 0.0:
            parts.append(f"angular_offset={self._angular_offset}")
        if self._bidirectional:
            parts.append("bidirectional=True")

        if parts:
            return f"FullSpokeOrdering({', '.join(parts)})"
        return "FullSpokeOrdering()"