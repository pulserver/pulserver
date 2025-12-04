"""Spiral ordering strategy for Cartesian grids."""

__all__ = [
    "SpiralOrdering",
    "DensityFunction",
    "UniformDensity",
    "VariableDensity",
    "CustomDensity",
]

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from ._base import OrderingStrategy
from ._angular_increments import AngularIncrement


class DensityFunction(ABC):
    """
    Abstract base class for spiral density functions.

    Density functions control how tightly wound the spiral is at different
    radial positions. Higher density = more points per unit radius.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a descriptive name for this density function."""
        pass

    @abstractmethod
    def __call__(self, r: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Compute density weights at given normalized radii.

        Parameters
        ----------
        r : NDArray[np.floating]
            Normalized radial positions in [0, 1], where 0 is center
            and 1 is the edge of k-space.

        Returns
        -------
        weights : NDArray[np.floating]
            Density weights.  Higher values = denser sampling.
            Will be used to modulate the spiral progression.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class UniformDensity(DensityFunction):
    """
    Uniform density (constant) spiral.

    All radial positions have equal density weight.
    """

    @property
    def name(self) -> str:
        return "uniform"

    def __call__(self, r: NDArray[np.floating]) -> NDArray[np.floating]:
        return np.ones_like(r)


class VariableDensity(DensityFunction):
    """
    Variable density spiral with configurable center and edge densities.

    Interpolates between center and edge density using a polynomial
    transition.  Useful for compressed sensing where center of k-space
    should be more densely sampled.

    Parameters
    ----------
    center_density : float
        Relative density at k-space center (r=0).  Default is 1.0.
    edge_density : float
        Relative density at k-space edge (r=1).  Default is 0. 2.
    transition_power : float
        Power of the polynomial transition.  Higher values create
        sharper transitions. Default is 2.0 (quadratic).
    transition_point : float
        Normalized radius [0, 1] where transition is centered.
        Default is 0. 2 (20% from center).

    Examples
    --------
    >>> # Standard variable density: dense center, sparse edge
    >>> density = VariableDensity(center_density=1.0, edge_density=0.2)

    >>> # Aggressive undersampling at edge
    >>> density = VariableDensity(center_density=1.0, edge_density=0.1, transition_power=3)
    """

    def __init__(
        self,
        center_density: float = 1.0,
        edge_density: float = 0.2,
        transition_power: float = 2.0,
        transition_point: float = 0.2,
    ):
        if not 0 <= transition_point <= 1:
            raise ValueError(
                f"transition_point must be in [0, 1], got {transition_point}"
            )
        if center_density <= 0 or edge_density <= 0:
            raise ValueError("Densities must be positive")

        self._center_density = center_density
        self._edge_density = edge_density
        self._transition_power = transition_power
        self._transition_point = transition_point

    @property
    def name(self) -> str:
        return "variable"

    @property
    def center_density(self) -> float:
        """Return center density."""
        return self._center_density

    @property
    def edge_density(self) -> float:
        """Return edge density."""
        return self._edge_density

    def __call__(self, r: NDArray[np.floating]) -> NDArray[np.floating]:
        # Normalize r relative to transition point for smooth interpolation
        # Use polynomial interpolation from center to edge density
        t = np.clip(r, 0, 1)

        # Polynomial blend from center_density to edge_density
        blend = t**self._transition_power
        density = self._center_density * (1 - blend) + self._edge_density * blend

        return density

    def __repr__(self) -> str:
        return (
            f"VariableDensity(center_density={self._center_density}, "
            f"edge_density={self._edge_density}, "
            f"transition_power={self._transition_power})"
        )


class CustomDensity(DensityFunction):
    """
    Custom density using a user-provided function.

    Parameters
    ----------
    density_func : Callable[[NDArray], NDArray]
        Function mapping normalized radius [0, 1] to density weights.
    name : str
        Descriptive name for this density function.

    Examples
    --------
    >>> # Gaussian density profile
    >>> density = CustomDensity(
    ...     density_func=lambda r: np.exp(-r**2 / 0.5),
    ...     name="gaussian"
    ... )
    """

    def __init__(
        self,
        density_func: Callable[[NDArray[np.floating]], NDArray[np.floating]],
        name: str = "custom",
    ):
        self._density_func = density_func
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def __call__(self, r: NDArray[np.floating]) -> NDArray[np.floating]:
        return self._density_func(r)

    def __repr__(self) -> str:
        return f"CustomDensity(name='{self._name}')"


class SpiralOrdering(OrderingStrategy):
    """
    Cartesian spiral ordering from center outward.

    Creates a spiral acquisition pattern on a Cartesian grid.  The spiral
    can have multiple interleaves with configurable rotation between them,
    and variable density for compressed sensing applications.

    Parameters
    ----------
    spiral_dims : tuple[str, str] | None
        The two dimensions defining the spiral plane.
        If None, uses first two dimensions from dim_labels.
    n_interleaves : int
        Number of spiral interleaves.  Default is 1 (single spiral).
    interleave_rotation : AngularIncrement | float | None
        Rotation scheme between consecutive interleaves.
        - If AngularIncrement: uses the increment scheme
        - If float: constant rotation angle in radians
        - If None: uniform rotation (2π / n_interleaves)
    density : DensityFunction | None
        Density function for variable density spiral.
        If None, uses UniformDensity.
    clockwise : bool
        If True, spiral winds clockwise; otherwise counter-clockwise.
        Default is True.
    center : dict[str, float] | None
        Custom center coordinates (in index units).
        If None, uses midpoint of each dimension.

    Examples
    --------
    >>> # Simple single spiral
    >>> strategy = SpiralOrdering()

    >>> # 4-interleave spiral with uniform rotation
    >>> strategy = SpiralOrdering(n_interleaves=4)

    >>> # Golden angle rotated interleaves
    >>> from trajectory_ordering. angular_increments import GoldenAngle
    >>> strategy = SpiralOrdering(
    ...     n_interleaves=8,
    ...     interleave_rotation=GoldenAngle()
    ...  )

    >>> # Variable density spiral for compressed sensing
    >>> strategy = SpiralOrdering(
    ...     density=VariableDensity(center_density=1.0, edge_density=0.3)
    ... )

    >>> # Combining interleaves and variable density
    >>> strategy = SpiralOrdering(
    ...      n_interleaves=4,
    ...     interleave_rotation=GoldenAngle(),
    ...     density=VariableDensity(center_density=1. 0, edge_density=0.2)
    ... )
    """

    def __init__(
        self,
        spiral_dims: tuple[str, str] | None = None,
        n_interleaves: int = 1,
        interleave_rotation: AngularIncrement | float | None = None,
        density: DensityFunction | None = None,
        clockwise: bool = True,
        center: dict[str, float] | None = None,
    ):
        if n_interleaves < 1:
            raise ValueError(f"n_interleaves must be >= 1, got {n_interleaves}")

        self._spiral_dims = spiral_dims
        self._n_interleaves = n_interleaves
        self._interleave_rotation = interleave_rotation
        self._density = density or UniformDensity()
        self._clockwise = clockwise
        self._center = center

    @property
    def name(self) -> str:
        base = f"spiral_{self._density.name}"
        if self._n_interleaves > 1:
            base += f"_{self._n_interleaves}int"
        return base

    @property
    def spiral_dims(self) -> tuple[str, str] | None:
        """Return the spiral plane dimensions."""
        return self._spiral_dims

    @property
    def n_interleaves(self) -> int:
        """Return the number of interleaves."""
        return self._n_interleaves

    @property
    def density(self) -> DensityFunction:
        """Return the density function."""
        return self._density

    def compute_order(
        self,
        scaling: dict[str, NDArray],
        indices: dict[str, NDArray],
        dim_labels: tuple[str, ...],
    ) -> NDArray[np.intp]:
        """
        Compute spiral acquisition order.

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
            Indices that sort the points into spiral acquisition order.
        """
        # Determine spiral dimensions
        if self._spiral_dims is None:
            if len(dim_labels) < 2:
                raise ValueError("Spiral ordering requires at least 2 dimensions")
            spiral_dims = (dim_labels[0], dim_labels[1])
        else:
            spiral_dims = self._spiral_dims

        # Validate dimensions
        for dim in spiral_dims:
            if dim not in indices:
                raise ValueError(
                    f"Spiral dimension '{dim}' not found in data.  "
                    f"Available: {list(indices. keys())}"
                )

        d0, d1 = spiral_dims
        n_points = len(indices[d0])

        # Compute center
        centers = self._compute_centers(indices, spiral_dims)

        # Compute polar coordinates from center
        dx = indices[d0] - centers[d0]
        dy = indices[d1] - centers[d1]

        radius = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx)

        if not self._clockwise:
            angle = -angle

        # Normalize radius to [0, 1]
        max_radius = np.max(radius) if np.max(radius) > 0 else 1.0
        radius_norm = radius / max_radius

        # Apply density function to get effective radius for ordering
        # Higher density = slower radius progression = smaller effective radius
        density_weights = self._density(radius_norm)

        # Integrate density to get cumulative "distance" from center
        # This creates the variable density effect
        effective_radius = self._compute_effective_radius(radius_norm, density_weights)

        # Compute spiral parameter (combined radius and angle)
        # For Archimedean spiral: r = a * theta
        spiral_param = self._compute_spiral_parameter(effective_radius, angle)

        # Assign points to interleaves
        interleave_ids = self._assign_interleaves(angle, n_points)

        # Get rotation for each interleave
        interleave_rotations = self._get_interleave_rotations()

        # Apply interleave rotation to spiral parameter
        rotated_param = spiral_param.copy()
        for i in range(self._n_interleaves):
            mask = interleave_ids == i
            rotated_param[mask] = spiral_param[mask] - interleave_rotations[i]

        # Sort by interleave first, then by spiral parameter within interleave
        interleave_quantized = interleave_ids * int(1e9)
        param_quantized = np.round(rotated_param * 1e6).astype(np.int64)

        return np.lexsort((param_quantized, interleave_quantized))

    def _compute_centers(
        self,
        indices: dict[str, NDArray],
        spiral_dims: tuple[str, str],
    ) -> dict[str, float]:
        """Compute center coordinates for spiral dimensions."""
        centers = {}
        for dim in spiral_dims:
            if self._center and dim in self._center:
                centers[dim] = self._center[dim]
            else:
                centers[dim] = (np.max(indices[dim]) + np.min(indices[dim])) / 2
        return centers

    def _compute_effective_radius(
        self,
        radius_norm: NDArray[np.floating],
        density_weights: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Compute effective radius accounting for variable density.

        For variable density, we want denser regions to progress more slowly
        through the spiral. This is achieved by integrating the inverse of
        density to get an effective cumulative distance.
        """
        if isinstance(self._density, UniformDensity):
            return radius_norm

        # Sort by radius to compute cumulative effect
        sort_idx = np.argsort(radius_norm)
        sorted_radius = radius_norm[sort_idx]
        sorted_density = density_weights[sort_idx]

        # Compute cumulative "distance" weighted by inverse density
        # Higher density = smaller steps = slower progression
        inv_density = 1.0 / np.maximum(sorted_density, 1e-6)

        # Approximate integration using cumulative sum
        dr = np.diff(sorted_radius, prepend=0)
        cumulative = np.cumsum(dr * inv_density)

        # Normalize to [0, 1]
        if cumulative[-1] > 0:
            cumulative = cumulative / cumulative[-1]

        # Map back to original order
        effective = np.zeros_like(radius_norm)
        effective[sort_idx] = cumulative

        return effective

    def _compute_spiral_parameter(
        self,
        effective_radius: NDArray[np.floating],
        angle: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Compute spiral ordering parameter.

        Uses Archimedean spiral: each full rotation increases radius linearly.
        The parameter combines radius and angle for proper spiral ordering.
        """
        # Normalize angle to [0, 2π)
        angle_norm = (angle + 2 * np.pi) % (2 * np.pi)

        # For Archimedean spiral ordering:
        # Points with same "spiral distance" from center should have same param
        # spiral_param = radius + angle / (2π) gives spiral ordering
        # But we want to order by "how far along the spiral" each point is

        # Approximate number of turns based on effective radius
        # Then add fractional turn from angle
        n_turns = effective_radius * 10  # Scale factor for turn count
        spiral_param = n_turns + angle_norm / (2 * np.pi)

        return spiral_param

    def _assign_interleaves(
        self,
        angle: NDArray[np.floating],
        n_points: int,
    ) -> NDArray[np.intp]:
        """Assign each point to an interleave based on angle."""
        if self._n_interleaves == 1:
            return np.zeros(n_points, dtype=np.intp)

        # Normalize angle to [0, 2π)
        angle_norm = (angle + 2 * np.pi) % (2 * np.pi)

        # Divide angle space into interleaves
        interleave_width = 2 * np.pi / self._n_interleaves
        interleave_ids = (angle_norm / interleave_width).astype(np.intp)
        interleave_ids = np.clip(interleave_ids, 0, self._n_interleaves - 1)

        return interleave_ids

    def _get_interleave_rotations(self) -> NDArray[np.floating]:
        """Get rotation angle for each interleave."""
        if self._n_interleaves == 1:
            return np.array([0.0])

        if self._interleave_rotation is None:
            # Uniform rotation
            return np.linspace(0, 2 * np.pi, self._n_interleaves, endpoint=False)

        if isinstance(self._interleave_rotation, (int, float)):
            # Constant rotation
            return np.arange(self._n_interleaves) * float(self._interleave_rotation)

        # Use AngularIncrement
        return self._interleave_rotation.get_angles(self._n_interleaves)

    def __repr__(self) -> str:
        return (
            f"SpiralOrdering(spiral_dims={self._spiral_dims}, "
            f"n_interleaves={self._n_interleaves}, "
            f"density={self._density}, "
            f"clockwise={self._clockwise})"
        )
