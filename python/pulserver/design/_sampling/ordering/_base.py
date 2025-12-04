"""Trajectory Ordering Module for MR Imaging."""

__all__ = [
    "TrajectoryData",
    "OrderedTrajectory",
    "OrderingStrategy",
    "CustomOrdering",
    "TrajectoryOrderer",
]

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import NDArray


@dataclass
class TrajectoryData:
    """
    Container for trajectory data with labeled dimensions.

    Attributes
    ----------
    scaling : dict[str, NDArray]
        Dictionary mapping dimension labels to scaling factor arrays.
        Each array has shape matching the full grid.
        Example: {'k1': array(... ), 'k2': array(... )}

    indices : dict[str, NDArray]
        Dictionary mapping dimension labels to grid index arrays.
        Each array has shape matching the full grid.
        Example: {'k1': array(...), 'k2': array(...)}

    mask : NDArray[np.bool_]
        Binary mask indicating sampled locations (True = sampled).
        Shape must match the scaling/indices arrays.

    dim_labels : tuple[str, ...]
        Ordered tuple of dimension labels, defining the axis order.
    """

    scaling: dict[str, NDArray]
    indices: dict[str, NDArray]
    mask: NDArray[np.bool_]
    dim_labels: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self):
        # Auto-extract dim_labels if not provided
        if not self.dim_labels:
            self.dim_labels = tuple(self.scaling.keys())

        self._validate()

    def _validate(self):
        """Validate that all arrays have consistent shapes."""
        shapes = set()

        for label in self.dim_labels:
            if label not in self.scaling:
                raise ValueError(f"Missing scaling array for dimension '{label}'")
            if label not in self.indices:
                raise ValueError(f"Missing indices array for dimension '{label}'")
            shapes.add(self.scaling[label].shape)
            shapes.add(self.indices[label].shape)

        shapes.add(self.mask.shape)

        if len(shapes) > 1:
            raise ValueError(f"Inconsistent shapes found: {shapes}")

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the trajectory grid."""
        return self.mask.shape

    @property
    def ndim(self) -> int:
        """Return the number of dimensions."""
        return len(self.dim_labels)

    @property
    def n_sampled(self) -> int:
        """Return the number of sampled points."""
        return int(np.sum(self.mask))


@dataclass
class OrderedTrajectory:
    """
    Result container for ordered trajectory data.

    Attributes
    ----------
    scaling : dict[str, NDArray]
        Flattened, ordered scaling factors for each dimension.
        Each array has shape (n_sampled,).

    indices : dict[str, NDArray]
        Flattened, ordered grid indices for each dimension.
        Each array has shape (n_sampled,).

    dim_labels : tuple[str, ...]
        Dimension labels preserved from input.
    """

    scaling: dict[str, NDArray]
    indices: dict[str, NDArray]
    dim_labels: tuple[str, ...]

    @property
    def n_points(self) -> int:
        """Return the number of ordered points."""
        return len(next(iter(self.scaling.values())))

    def to_arrays(self) -> tuple[NDArray, NDArray]:
        """
        Return scaling and indices as 2D arrays.

        Returns
        -------
        scaling_array : NDArray
            Shape (n_dims, n_points) array of scaling factors.
        indices_array : NDArray
            Shape (n_dims, n_points) array of indices.
        """
        scaling_arr = np.stack([self.scaling[d] for d in self.dim_labels])
        indices_arr = np.stack([self.indices[d] for d in self.dim_labels])
        return scaling_arr, indices_arr


class OrderingStrategy(ABC):
    """
    Abstract base class for trajectory ordering strategies.

    Subclasses must implement the `compute_order` method which returns
    indices that define the acquisition order for the masked points.

    The strategy receives only the masked (sampled) points and must return
    an ordering of those points.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a descriptive name for this ordering strategy."""
        pass

    @abstractmethod
    def compute_order(
        self,
        scaling: dict[str, NDArray],
        indices: dict[str, NDArray],
        dim_labels: tuple[str, ...],
    ) -> NDArray[np.intp]:
        """
        Compute the acquisition order for the given points.

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
            Indices that sort the points into acquisition order.
            Should be a permutation of np.arange(len(scaling[dim_labels[0]])).
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class CustomOrdering(OrderingStrategy):
    """
    Custom ordering using a user-provided function.

    Parameters
    ----------
    order_func : Callable
        Function with signature:
        (scaling: dict, indices: dict, dim_labels: tuple) -> NDArray[np.intp]
    name : str
        Descriptive name for this ordering.
    """

    def __init__(
        self,
        order_func: Callable[
            [dict[str, NDArray], dict[str, NDArray], tuple[str, ...]],
            NDArray[np.intp],
        ],
        name: str = "custom",
    ):
        self._order_func = order_func
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def compute_order(
        self,
        scaling: dict[str, NDArray],
        indices: dict[str, NDArray],
        dim_labels: tuple[str, ...],
    ) -> NDArray[np.intp]:
        return self._order_func(scaling, indices, dim_labels)

    def __repr__(self) -> str:
        return f"CustomOrdering(name='{self._name}')"


class TrajectoryOrderer:
    """
    Main class for ordering MR trajectory points.

    Handles the common logic of masking, validation, and coordination,
    delegating the actual ordering to a strategy object.

    Parameters
    ----------
    strategy : OrderingStrategy
        The ordering strategy to use.

    Examples
    --------
    >>> # 1D center-out ordering
    >>> k1_scaling = np.linspace(-1, 1, 128)
    >>> k1_indices = np.arange(128)
    >>> mask = np. zeros(128, dtype=bool)
    >>> mask[::2] = True  # R=2 acceleration
    >>>
    >>> data = TrajectoryData(
    ...     scaling={'k1': k1_scaling},
    ...     indices={'k1': k1_indices},
    ...      mask=mask,
    ...     dim_labels=('k1',)
    ... )
    >>>
    >>> orderer = TrajectoryOrderer(CenterOutOrdering())
    >>> result = orderer.order(data)
    >>> print(result. scaling['ky'])  # Center-out ordered

    >>> # 2D with golden angle
    >>> orderer = TrajectoryOrderer(
    ...     CenterOutOrdering(angular_mode='golden')
    ... )
    """

    def __init__(self, strategy: OrderingStrategy):
        self.strategy = strategy

    def order(self, data: TrajectoryData) -> OrderedTrajectory:
        """
        Apply the ordering strategy to the trajectory data.

        Parameters
        ----------
        data : TrajectoryData
            Input trajectory data with scaling, indices, and mask.

        Returns
        -------
        OrderedTrajectory
            Flattened, ordered trajectory data for sampled points only.
        """
        # Extract masked points
        masked_scaling = {
            dim: data.scaling[dim][data.mask].flatten() for dim in data.dim_labels
        }
        masked_indices = {
            dim: data.indices[dim][data.mask].flatten() for dim in data.dim_labels
        }

        # Compute ordering
        order = self.strategy.compute_order(
            masked_scaling,
            masked_indices,
            data.dim_labels,
        )

        # Apply ordering
        ordered_scaling = {dim: masked_scaling[dim][order] for dim in data.dim_labels}
        ordered_indices = {dim: masked_indices[dim][order] for dim in data.dim_labels}

        return OrderedTrajectory(
            scaling=ordered_scaling,
            indices=ordered_indices,
            dim_labels=data.dim_labels,
        )

    def __repr__(self) -> str:
        return f"TrajectoryOrderer(strategy={self.strategy})"
