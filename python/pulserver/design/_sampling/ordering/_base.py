"""Base classes for trajectory ordering strategies."""

__all__ = ["OrderingStrategy"]

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class OrderingStrategy(ABC):
    """
    Abstract base class for k-space ordering strategies. 
    
    Computes acquisition order for a set of coordinates, optionally
    divided into segments (shots, contrasts, frames, etc.).
    
    Strategies are agnostic to:
    - Coordinate semantics (ky, kz, slice, echo, etc.)
    - Scaling semantics (gradient amplitude, RF frequency, rotation angle)
    - Mask generation method (CAIPI, Poisson, regular undersampling)
    
    The user decides:
    - What coordinates represent
    - How to interpret segments (shots vs contrasts vs frames)
    - How to compose multiple orderings for nested acquisitions
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return a descriptive name for this strategy."""
        pass
    
    @abstractmethod
    def compute_order(
        self,
        coordinates: NDArray,
        mask: NDArray[bool] | None = None,
        n_segments: int = 1,
    ) -> NDArray[int]:
        """
        Compute acquisition order. 
        
        Parameters
        ----------
        coordinates : NDArray
            Point coordinates.  Shape: (n_points,) for 1D or
            (n_dims, n_points) for ND.
        mask : NDArray[np.bool_] | None
            Sampling mask. True = point is acquired. 
            If None, all points are acquired.
        n_segments : int
            Number of segments to divide the acquisition into.
            Interpretation is user-defined (shots, contrasts, frames, etc.).
            
        Returns
        -------
        order : NDArray[int]
            Indices into the masked coordinates array.
            Shape: (n_segments, n_points_per_segment). 
            
            Apply as: sorted_coords = coordinates[.. ., mask][:, order]
            
        Raises
        ------
        ValueError
            If no points are sampled or n_segments does not evenly divide
            the number of sampled points.
            
        Notes
        -----
        The output shape (n_segments, n_points_per_segment) allows the user
        to interpret segments as:
        
        - Shots (outer loop): Iterate over segments, acquire all points in each
        - Contrasts (inner loop): Each segment is one contrast's k-space
        
        For nested segmentation (e.g., segmented + dynamic), compose multiple
        orderings at different levels.
        """
        pass
    
    def _validate_inputs(
        self,
        coordinates: NDArray,
        mask: NDArray[bool] | None,
        n_segments: int,
    ) -> tuple[NDArray, NDArray[bool], int]:
        """
        Validate and normalize inputs.
        
        Returns
        -------
        coordinates : NDArray
            Shape: (n_dims, n_points)
        mask : NDArray[bool]
            Shape: (n_points,)
        n_sampled : int
            Number of sampled points
        """
        # Normalize coordinates to 2D
        coordinates = np.atleast_2d(coordinates)
        if coordinates.ndim != 2:
            raise ValueError(
                f"coordinates must be 1D or 2D, got shape {coordinates.shape}"
            )
        
        n_points = coordinates.shape[1]
        
        # Handle mask
        if mask is None:
            mask = np.ones(n_points, dtype=bool)
        else:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != (n_points,):
                raise ValueError(
                    f"mask shape {mask.shape} does not match "
                    f"coordinates shape {coordinates. shape}"
                )
        
        n_sampled = int(mask.sum())
        
        # Check for empty result
        if n_sampled == 0:
            raise ValueError("No points to order (mask is all False)")
        
        # Validate n_segments
        if n_segments < 1:
            raise ValueError(f"n_segments must be >= 1, got {n_segments}")
        
        if n_sampled % n_segments != 0:
            raise ValueError(
                f"n_segments ({n_segments}) must evenly divide "
                f"n_sampled ({n_sampled})"
            )
        
        return coordinates, mask, n_sampled
    
    def _apply_mask_and_reshape(
        self,
        order: NDArray[int],
        mask: NDArray[bool],
        n_segments: int,
    ) -> NDArray[int]:
        """
        Convert internal ordering to output format.
        
        Parameters
        ----------
        order : NDArray[int]
            Ordering indices into masked points.  Shape: (n_sampled,)
        mask : NDArray[bool]
            Original mask. 
        n_segments : int
            Number of segments.
            
        Returns
        -------
        result : NDArray[int]
            Shape: (n_segments, n_points_per_segment). 
            Indices into the masked coordinates. 
        """
        n_sampled = len(order)
        n_per_segment = n_sampled // n_segments
        return order.reshape(n_segments, n_per_segment)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"