"""Spiral Ordering strategies."""

__all__ = ["SpiralOrdering"]

from typing import Optional

import numpy as np

from numpy.typing import NDArray

from .base import OrderingStrategy


class SpiralOrdering(OrderingStrategy):
    """
    Cartesian spiral ordering from center outward.

    Creates a spiral pattern on a Cartesian grid, starting from the center
    and winding outward.  Works for 2D grids.

    Parameters
    ----------
    spiral_dims : tuple[str, str] | None
        The two dimensions defining the spiral plane.
        If None, uses first two dimensions.
    clockwise : bool
        If True, spiral winds clockwise; otherwise counter-clockwise.
    """

    def __init__(
        self,
        spiral_dims: Optional[tuple[str, str]] = None,
        clockwise: bool = True,
    ):
        self.spiral_dims = spiral_dims
        self.clockwise = clockwise

    @property
    def name(self) -> str:
        return "spiral"

    def compute_order(
        self,
        scaling: dict[str, NDArray],
        indices: dict[str, NDArray],
        dim_labels: tuple[str, ...],
    ) -> NDArray[np.intp]:
        # Determine spiral dimensions
        if self.spiral_dims is None:
            if len(dim_labels) < 2:
                raise ValueError("Spiral ordering requires at least 2 dimensions")
            spiral_dims = (dim_labels[0], dim_labels[1])
        else:
            spiral_dims = self.spiral_dims

        d0, d1 = spiral_dims

        # Compute center
        center0 = (np.max(indices[d0]) + np.min(indices[d0])) / 2
        center1 = (np.max(indices[d1]) + np.min(indices[d1])) / 2

        # Compute radius and angle from center
        dx = indices[d0] - center0
        dy = indices[d1] - center1

        radius = np.maximum(np.abs(dx), np.abs(dy))  # Chebyshev distance
        angle = np.arctan2(dy, dx)

        if not self.clockwise:
            angle = -angle

        # Normalize angle to [0, 2π)
        angle = (angle + 2 * np.pi) % (2 * np.pi)

        # For Cartesian spiral: sort by "shell" (Chebyshev distance),
        # then by angle within each shell
        radius_discrete = np.round(radius).astype(int)
        angle_discrete = np.round(angle * 10000).astype(int)

        return np.lexsort((angle_discrete, radius_discrete))

    def __repr__(self) -> str:
        return f"SpiralOrdering(spiral_dims={self.spiral_dims}, clockwise={self. clockwise})"
