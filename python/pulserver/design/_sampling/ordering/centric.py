"""Radial Ordering strategies."""

__all__ = ["CenterOutOrdering", "FullSpokeOrdering"]

from typing import Optional, Sequence

import numpy as np

from numpy.typing import NDArray

from .base import OrderingStrategy


class CenterOutOrdering(OrderingStrategy):
    """
    Center-out ordering starting from the center of k-space.

    For 1D: Alternates between positive and negative offsets from center.
    For 2D+: Orders by distance from center, with optional angular ordering.

    Parameters
    ----------
    radial_dim : str | Sequence[str] | None
        Dimension(s) to use for radial distance calculation.
        If None, uses all dimensions.
    angular_mode : str
        How to order points at similar radii:
        - 'linear': Sequential angular ordering
        - 'golden': Golden angle increment (~111.246°)
        - 'none': No angular ordering (arbitrary within radius)
    center : dict[str, float] | None
        Custom center coordinates. If None, uses index midpoint.
    """

    GOLDEN_ANGLE = np.pi * (3 - np.sqrt(5))  # ~111.246 degrees

    def __init__(
        self,
        radial_dim: Optional[str | Sequence[str]] = None,
        angular_mode: str = "linear",
        center: Optional[dict[str, float]] = None,
    ):
        self.radial_dim = radial_dim
        self.angular_mode = angular_mode
        self.center = center

        if angular_mode not in ("linear", "golden", "none"):
            raise ValueError(f"Unknown angular_mode: {angular_mode}")

    @property
    def name(self) -> str:
        return f"center_out_{self.angular_mode}"

    def compute_order(
        self,
        scaling: dict[str, NDArray],
        indices: dict[str, NDArray],
        dim_labels: tuple[str, ...],
    ) -> NDArray[int]:
        n_points = len(indices[dim_labels[0]])

        # Determine which dimensions to use for radial calculation
        if self.radial_dim is None:
            radial_dims = dim_labels
        elif isinstance(self.radial_dim, str):
            radial_dims = (self.radial_dim,)
        else:
            radial_dims = tuple(self.radial_dim)

        # Compute center for each dimension
        centers = {}
        for dim in radial_dims:
            if self.center and dim in self.center:
                centers[dim] = self.center[dim]
            else:
                # Use midpoint of index range
                centers[dim] = (np.max(indices[dim]) + np.min(indices[dim])) / 2

        # Compute radial distance from center
        radius_sq = np.zeros(n_points)
        for dim in radial_dims:
            radius_sq += (indices[dim] - centers[dim]) ** 2
        radius = np.sqrt(radius_sq)

        # Compute angular coordinate (for 2D+ with angular ordering)
        if len(radial_dims) >= 2 and self.angular_mode != "none":
            # Use first two radial dimensions for angle
            d0, d1 = radial_dims[0], radial_dims[1]
            angle = np.arctan2(indices[d1] - centers[d1], indices[d0] - centers[d0])

            if self.angular_mode == "golden":
                # Reorder angles using golden angle increments
                angle = self._apply_golden_angle_reorder(angle, radius)
        else:
            angle = np.zeros(n_points)

        # Sort by radius first, then by angle
        # Discretize radius to create "shells"
        radius_discrete = np.round(radius * 1000).astype(int)
        angle_discrete = np.round(angle * 1000).astype(int)

        return np.lexsort((angle_discrete, radius_discrete))

    def _apply_golden_angle_reorder(
        self,
        angles: NDArray,
        radii: NDArray,
    ) -> NDArray:
        """Reorder angles within each radius shell using golden angle."""
        result = angles.copy()
        unique_radii = np.unique(np.round(radii * 100))

        for r in unique_radii:
            mask = np.round(radii * 100) == r
            n_in_shell = np.sum(mask)
            if n_in_shell > 1:
                # Assign golden angle indices
                shell_angles = angles[mask]
                sort_idx = np.argsort(shell_angles)
                golden_order = np.zeros(n_in_shell)
                for i in range(n_in_shell):
                    golden_order[sort_idx[i]] = (i * self.GOLDEN_ANGLE) % (2 * np.pi)
                result[mask] = golden_order

        return result

    def __repr__(self) -> str:
        return (
            f"CenterOutOrdering(radial_dim={self.radial_dim}, "
            f"angular_mode='{self.angular_mode}')"
        )


class FullSpokeOrdering(OrderingStrategy):
    """
    Full spoke (edge-to-edge) radial ordering.

    Each "spoke" goes from one edge through center to the opposite edge.
    Useful for radial acquisitions where you want symmetric readouts.

    Parameters
    ----------
    spoke_dim : str
        Dimension along which spokes are acquired (readout direction).
    angular_dims : Sequence[str] | None
        Dimensions that define the angular position of each spoke.
        If None, uses all other dimensions.
    angular_mode : str
        How to order spokes: 'linear', 'golden', or 'uniform'.
    """

    GOLDEN_ANGLE = np.pi * (3 - np.sqrt(5))

    def __init__(
        self,
        spoke_dim: str,
        angular_dims: Optional[Sequence[str]] = None,
        angular_mode: str = "linear",
    ):
        self.spoke_dim = spoke_dim
        self.angular_dims = angular_dims
        self.angular_mode = angular_mode

    @property
    def name(self) -> str:
        return f"full_spoke_{self.angular_mode}"

    def compute_order(
        self,
        scaling: dict[str, NDArray],
        indices: dict[str, NDArray],
        dim_labels: tuple[str, ...],
    ) -> NDArray[int]:

        # Determine angular dimensions
        if self.angular_dims is None:
            angular_dims = tuple(d for d in dim_labels if d != self.spoke_dim)
        else:
            angular_dims = tuple(self.angular_dims)

        if not angular_dims:
            # 1D case: just sort by spoke dimension (edge to edge)
            return np.argsort(indices[self.spoke_dim])

        # Compute spoke index (which spoke each point belongs to)
        # based on angular dimensions
        spoke_keys = [indices[d] for d in angular_dims]

        # Create unique spoke identifier
        spoke_id = self._compute_spoke_ids(spoke_keys)

        # Compute angle for each spoke (for ordering spokes)
        if len(angular_dims) >= 2:
            d0, d1 = angular_dims[0], angular_dims[1]
            # Use center of each spoke's angular position
            spoke_angle = np.arctan2(
                indices[d1] - np.mean(indices[d1]), indices[d0] - np.mean(indices[d0])
            )
        else:
            spoke_angle = indices[angular_dims[0]].astype(float)

        # Apply angular ordering mode
        if self.angular_mode == "golden":
            unique_spokes = np.unique(spoke_id)
            spoke_order = {
                s: (i * self.GOLDEN_ANGLE) % (2 * np.pi)
                for i, s in enumerate(unique_spokes)
            }
            spoke_angle = np.array([spoke_order[s] for s in spoke_id])

        # Sort: first by spoke angle, then by position along spoke
        angle_discrete = np.round(spoke_angle * 10000).astype(int)
        return np.lexsort((indices[self.spoke_dim], angle_discrete))

    def _compute_spoke_ids(self, keys: list[NDArray]) -> NDArray:
        """Assign unique ID to each spoke based on angular coordinates."""
        # Stack and create unique combinations
        stacked = np.column_stack(keys)
        _, inverse = np.unique(stacked, axis=0, return_inverse=True)
        return inverse

    def __repr__(self) -> str:
        return (
            f"FullSpokeOrdering(spoke_dim='{self.spoke_dim}', "
            f"angular_dims={self.angular_dims}, angular_mode='{self.angular_mode}')"
        )
