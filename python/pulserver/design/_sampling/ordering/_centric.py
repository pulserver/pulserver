"""Centric (center-out and full-spoke) ordering strategies."""

__all__ = ["CenterOutOrdering", "FullSpokeOrdering"]

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ._base import OrderingStrategy
from ._angular_increments import AngularIncrement, LinearIncrement


class CenterOutOrdering(OrderingStrategy):
    """
    Center-out ordering starting from the center of k-space.

    Acquires points starting from the k-space center and moving outward.
    For 2D+ data, points are ordered by radial distance from center,
    with angular ordering applied within each radial shell.

    Parameters
    ----------
    radial_dims : Sequence[str] | None
        Dimension(s) to use for radial distance calculation.
        If None, uses all dimensions.
    angular_increment : AngularIncrement | None
        Angular increment scheme for ordering within radial shells.
        If None, uses LinearIncrement.  Only used for 2D+ data.
    center : dict[str, float] | None
        Custom center coordinates (in index units). If None, uses
        the midpoint of each dimension's index range.

    Examples
    --------
    >>> # Simple 1D center-out
    >>> strategy = CenterOutOrdering()

    >>> # 2D with golden angle ordering
    >>> from trajectory_ordering. angular_increments import GoldenAngle
    >>> strategy = CenterOutOrdering(angular_increment=GoldenAngle())

    >>> # 3D stack-of-stars with tiny golden angle
    >>> from trajectory_ordering.angular_increments import TinyGoldenAngle
    >>> strategy = CenterOutOrdering(
    ...     radial_dims=['k1', 'k2'],  # radial in k1-k2 plane
    ...     angular_increment=TinyGoldenAngle(order=7)
    ... )
    """

    def __init__(
        self,
        radial_dims: Sequence[str] | None = None,
        angular_increment: AngularIncrement | None = None,
        center: dict[str, float] | None = None,
    ):
        self._radial_dims = tuple(radial_dims) if radial_dims else None
        self._angular_increment = angular_increment or LinearIncrement()
        self._center = center

    @property
    def name(self) -> str:
        return f"center_out_{self._angular_increment.name}"

    @property
    def radial_dims(self) -> tuple[str, ...] | None:
        """Return the dimensions used for radial ordering."""
        return self._radial_dims

    @property
    def angular_increment(self) -> AngularIncrement:
        """Return the angular increment scheme."""
        return self._angular_increment

    def compute_order(
        self,
        scaling: dict[str, NDArray],
        indices: dict[str, NDArray],
        dim_labels: tuple[str, ...],
    ) -> NDArray[np.intp]:
        """
        Compute center-out acquisition order.

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
            Indices that sort the points into center-out acquisition order.
        """
        # Determine radial dimensions
        radial_dims = self._radial_dims if self._radial_dims else dim_labels

        # Validate dimensions exist
        for dim in radial_dims:
            if dim not in indices:
                raise ValueError(
                    f"Dimension '{dim}' in radial_dims not found in data.  "
                    f"Available: {list(indices.keys())}"
                )

        # Compute center for each dimension
        centers = self._compute_centers(indices, radial_dims)

        # Compute radial distance from center
        radius = self._compute_radius(indices, radial_dims, centers)

        # For 1D, just sort by radius
        if len(radial_dims) == 1:
            return np.argsort(radius)

        # For 2D+, compute angular coordinate and apply ordering
        angle = self._compute_angle(indices, radial_dims, centers)
        reordered_angle = self._apply_angular_ordering(angle, radius)

        # Sort by radius first, then by reordered angle
        # Use quantized values to create discrete shells
        radius_quantized = np.round(radius * 1e6).astype(np.int64)
        angle_quantized = np.round(reordered_angle * 1e6).astype(np.int64)

        return np.lexsort((angle_quantized, radius_quantized))

    def _compute_centers(
        self,
        indices: dict[str, NDArray],
        radial_dims: tuple[str, ...] | Sequence[str],
    ) -> dict[str, float]:
        """Compute center coordinates for each dimension."""
        centers = {}
        for dim in radial_dims:
            if self._center and dim in self._center:
                centers[dim] = self._center[dim]
            else:
                # Midpoint of index range
                centers[dim] = (np.max(indices[dim]) + np.min(indices[dim])) / 2
        return centers

    def _compute_radius(
        self,
        indices: dict[str, NDArray],
        radial_dims: tuple[str, ...] | Sequence[str],
        centers: dict[str, float],
    ) -> NDArray[np.floating]:
        """Compute radial distance from center."""
        radius_sq = np.zeros(len(indices[radial_dims[0]]), dtype=np.float64)
        for dim in radial_dims:
            radius_sq += (indices[dim] - centers[dim]) ** 2
        return np.sqrt(radius_sq)

    def _compute_angle(
        self,
        indices: dict[str, NDArray],
        radial_dims: tuple[str, ...] | Sequence[str],
        centers: dict[str, float],
    ) -> NDArray[np.floating]:
        """Compute angular coordinate from first two radial dimensions."""
        d0, d1 = radial_dims[0], radial_dims[1]
        return np.arctan2(
            indices[d1] - centers[d1],
            indices[d0] - centers[d0],
        )

    def _apply_angular_ordering(
        self,
        angles: NDArray[np.floating],
        radii: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Reorder angles within each radial shell using angular increment."""
        result = np.zeros_like(angles)

        # Group by quantized radius
        radius_quantized = np.round(radii * 1000).astype(np.int64)
        unique_radii = np.unique(radius_quantized)

        for r in unique_radii:
            shell_mask = radius_quantized == r
            n_in_shell = np.sum(shell_mask)

            if n_in_shell <= 1:
                result[shell_mask] = 0.0
                continue

            # Get target angles from increment scheme
            target_angles = self._angular_increment.get_angles(n_in_shell)

            # Sort points in shell by their original angle
            shell_indices = np.where(shell_mask)[0]
            shell_angles = angles[shell_mask]
            angle_order = np.argsort(shell_angles)

            # Assign target angles based on sorted position
            for i, target in enumerate(target_angles):
                result[shell_indices[angle_order[i]]] = target

        return result

    def __repr__(self) -> str:
        return (
            f"CenterOutOrdering(radial_dims={self._radial_dims}, "
            f"angular_increment={self._angular_increment})"
        )


class FullSpokeOrdering(OrderingStrategy):
    """
    Full spoke (edge-to-edge) radial ordering.

    Each "spoke" traverses from one edge of k-space through the center
    to the opposite edge.  Spokes are ordered according to the specified
    angular increment scheme.

    Parameters
    ----------
    spoke_dim : str
        Dimension along which spokes are acquired (readout direction).
    angular_dims : Sequence[str] | None
        Dimensions that define the angular position of each spoke.
        If None, uses all dimensions except spoke_dim.
    angular_increment : AngularIncrement | None
        Angular increment scheme for spoke ordering.
        If None, uses LinearIncrement.

    Examples
    --------
    >>> # Basic radial with linear spoke ordering
    >>> strategy = FullSpokeOrdering(spoke_dim='k0')

    >>> # Radial with golden angle
    >>> from trajectory_ordering.angular_increments import GoldenAngle
    >>> strategy = FullSpokeOrdering(
    ...     spoke_dim='k0',
    ...     angular_increment=GoldenAngle(full_circle=True)
    ... )
    """

    def __init__(
        self,
        spoke_dim: str,
        angular_dims: Sequence[str] | None = None,
        angular_increment: AngularIncrement | None = None,
    ):
        self._spoke_dim = spoke_dim
        self._angular_dims = tuple(angular_dims) if angular_dims else None
        self._angular_increment = angular_increment or LinearIncrement()

    @property
    def name(self) -> str:
        return f"full_spoke_{self._angular_increment.name}"

    @property
    def spoke_dim(self) -> str:
        """Return the spoke (readout) dimension."""
        return self._spoke_dim

    @property
    def angular_dims(self) -> tuple[str, ...] | None:
        """Return the angular dimensions."""
        return self._angular_dims

    @property
    def angular_increment(self) -> AngularIncrement:
        """Return the angular increment scheme."""
        return self._angular_increment

    def compute_order(
        self,
        scaling: dict[str, NDArray],
        indices: dict[str, NDArray],
        dim_labels: tuple[str, ...],
    ) -> NDArray[np.intp]:
        """
        Compute full-spoke acquisition order.

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
        order : NDArray[np. intp]
            Indices that sort the points into full-spoke acquisition order.
        """
        # Validate spoke dimension
        if self._spoke_dim not in indices:
            raise ValueError(
                f"Spoke dimension '{self._spoke_dim}' not found in data. "
                f"Available: {list(indices.keys())}"
            )

        # Determine angular dimensions
        if self._angular_dims is None:
            angular_dims = tuple(d for d in dim_labels if d != self._spoke_dim)
        else:
            angular_dims = self._angular_dims

        # Validate angular dimensions
        for dim in angular_dims:
            if dim not in indices:
                raise ValueError(
                    f"Angular dimension '{dim}' not found in data.  "
                    f"Available: {list(indices.keys())}"
                )

        # If no angular dimensions, just sort by spoke position
        if not angular_dims:
            return np.argsort(indices[self._spoke_dim])

        # Assign each point to a spoke based on angular dimensions
        spoke_ids = self._compute_spoke_ids(indices, angular_dims)

        # Compute spoke angles for ordering
        spoke_angles = self._compute_spoke_angles(indices, angular_dims, spoke_ids)

        # Apply angular increment ordering to spokes
        reordered_angles = self._apply_spoke_ordering(spoke_angles, spoke_ids)

        # Sort by spoke angle first, then by position along spoke
        angle_quantized = np.round(reordered_angles * 1e6).astype(np.int64)

        return np.lexsort((indices[self._spoke_dim], angle_quantized))

    def _compute_spoke_ids(
        self,
        indices: dict[str, NDArray],
        angular_dims: tuple[str, ...],
    ) -> NDArray[np.intp]:
        """Assign unique spoke ID to each point based on angular coordinates."""
        # Stack angular dimension indices
        stacked = np.column_stack([indices[d] for d in angular_dims])
        _, inverse = np.unique(stacked, axis=0, return_inverse=True)
        return inverse.astype(np.intp)

    def _compute_spoke_angles(
        self,
        indices: dict[str, NDArray],
        angular_dims: tuple[str, ...],
        spoke_ids: NDArray[np.intp],
    ) -> NDArray[np.floating]:
        """Compute angle for each spoke."""
        n_points = len(spoke_ids)
        angles = np.zeros(n_points, dtype=np.float64)

        if len(angular_dims) >= 2:
            # Use first two angular dimensions for angle calculation
            d0, d1 = angular_dims[0], angular_dims[1]
            center0 = (np.max(indices[d0]) + np.min(indices[d0])) / 2
            center1 = (np.max(indices[d1]) + np.min(indices[d1])) / 2
            angles = np.arctan2(
                indices[d1] - center1,
                indices[d0] - center0,
            )
        elif len(angular_dims) == 1:
            # Use single dimension as angle proxy
            d0 = angular_dims[0]
            center = (np.max(indices[d0]) + np.min(indices[d0])) / 2
            angles = (indices[d0] - center).astype(np.float64)

        return angles

    def _apply_spoke_ordering(
        self,
        angles: NDArray[np.floating],
        spoke_ids: NDArray[np.intp],
    ) -> NDArray[np.floating]:
        """Reorder spokes according to angular increment scheme."""
        unique_spokes = np.unique(spoke_ids)
        n_spokes = len(unique_spokes)

        # Get one representative angle per spoke
        spoke_angles = np.zeros(n_spokes, dtype=np.float64)
        for i, sid in enumerate(unique_spokes):
            mask = spoke_ids == sid
            spoke_angles[i] = angles[mask][0]

        # Sort spokes by their original angle
        spoke_order = np.argsort(spoke_angles)

        # Get target angles from increment scheme
        target_angles = self._angular_increment.get_angles(n_spokes)

        # Create mapping from spoke_id to target angle
        spoke_to_target = {}
        for i, target in enumerate(target_angles):
            spoke_to_target[unique_spokes[spoke_order[i]]] = target

        # Apply to all points
        result = np.zeros_like(angles)
        for i, sid in enumerate(spoke_ids):
            result[i] = spoke_to_target[sid]

        return result

    def __repr__(self) -> str:
        return (
            f"FullSpokeOrdering(spoke_dim='{self._spoke_dim}', "
            f"angular_dims={self._angular_dims}, "
            f"angular_increment={self._angular_increment})"
        )
