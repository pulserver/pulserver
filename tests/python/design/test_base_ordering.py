"""Tests for trajectory ordering module."""

import numpy as np
import pytest
from numpy.typing import NDArray

from pulserver.design import (
    CustomOrdering,
    OrderedTrajectory,
    OrderingStrategy,
    TrajectoryData,
    TrajectoryOrderer,
)


@pytest.fixture
def simple_1d_data() -> TrajectoryData:
    """Create simple 1D trajectory data (single phase-encode dimension)."""
    n = 16
    return TrajectoryData(
        scaling={"k1": np.linspace(-1, 1, n)},
        indices={"k1": np.arange(n)},
        mask=np.ones(n, dtype=bool),
        dim_labels=("k1",),
    )


@pytest.fixture
def accelerated_1d_data() -> TrajectoryData:
    """Create 1D trajectory data with R=2 acceleration."""
    n = 16
    mask = np.zeros(n, dtype=bool)
    mask[::2] = True  # R=2
    return TrajectoryData(
        scaling={"k1": np.linspace(-1, 1, n)},
        indices={"k1": np.arange(n)},
        mask=mask,
        dim_labels=("k1",),
    )


@pytest.fixture
def simple_2d_data() -> TrajectoryData:
    """Create simple 2D trajectory data (k1 x k2 for 3D Cartesian)."""
    n_k1, n_k2 = 8, 6
    k1, k2 = np.meshgrid(
        np.linspace(-1, 1, n_k1),
        np.linspace(-1, 1, n_k2),
        indexing="ij",
    )
    i_k1, i_k2 = np.meshgrid(np.arange(n_k1), np.arange(n_k2), indexing="ij")
    return TrajectoryData(
        scaling={"k1": k1, "k2": k2},
        indices={"k1": i_k1, "k2": i_k2},
        mask=np.ones((n_k1, n_k2), dtype=bool),
        dim_labels=("k1", "k2"),
    )


@pytest.fixture
def identity_strategy() -> OrderingStrategy:
    """Strategy that returns points in original order."""
    return CustomOrdering(
        order_func=lambda s, i, d: np.arange(len(s[d[0]]), dtype=np.intp),
        name="identity",
    )


@pytest.fixture
def reverse_strategy() -> OrderingStrategy:
    """Strategy that returns points in reverse order."""
    return CustomOrdering(
        order_func=lambda s, i, d: np.arange(len(s[d[0]]) - 1, -1, -1, dtype=np.intp),
        name="reverse",
    )


# =============================================================================
# TrajectoryData Tests
# =============================================================================


class TestTrajectoryData:
    """Tests for TrajectoryData container."""

    def test_basic_creation(self, simple_1d_data: TrajectoryData):
        """Test basic 1D data creation."""
        assert simple_1d_data.ndim == 1
        assert simple_1d_data.shape == (16,)
        assert simple_1d_data.n_sampled == 16

    def test_2d_creation(self, simple_2d_data: TrajectoryData):
        """Test 2D data creation."""
        assert simple_2d_data.ndim == 2
        assert simple_2d_data.shape == (8, 6)
        assert simple_2d_data.n_sampled == 48

    def test_auto_dim_labels(self):
        """Test automatic dimension label extraction from scaling keys."""
        data = TrajectoryData(
            scaling={"k1": np.zeros(8), "k2": np.zeros(8)},
            indices={"k1": np.arange(8), "k2": np.arange(8)},
            mask=np.ones(8, dtype=bool),
        )
        assert set(data.dim_labels) == {"k1", "k2"}

    def test_n_sampled_with_mask(self, accelerated_1d_data: TrajectoryData):
        """Test n_sampled correctly counts masked points."""
        assert accelerated_1d_data.n_sampled == 8

    def test_missing_scaling_raises(self):
        """Test that missing scaling array raises ValueError."""
        with pytest.raises(ValueError, match="Missing scaling array"):
            TrajectoryData(
                scaling={"k1": np.zeros(8)},
                indices={"k1": np.arange(8), "k2": np.arange(8)},
                mask=np.ones(8, dtype=bool),
                dim_labels=("k1", "k2"),
            )

    def test_missing_indices_raises(self):
        """Test that missing indices array raises ValueError."""
        with pytest.raises(ValueError, match="Missing indices array"):
            TrajectoryData(
                scaling={"k1": np.zeros(8), "k2": np.zeros(8)},
                indices={"k1": np.arange(8)},
                mask=np.ones(8, dtype=bool),
                dim_labels=("k1", "k2"),
            )

    def test_shape_mismatch_raises(self):
        """Test that inconsistent shapes raise ValueError."""
        with pytest.raises(ValueError, match="Inconsistent shapes"):
            TrajectoryData(
                scaling={"k1": np.zeros(8)},
                indices={"k1": np.arange(16)},  # Wrong shape
                mask=np.ones(8, dtype=bool),
            )

    def test_mask_shape_mismatch_raises(self):
        """Test that mask shape mismatch raises ValueError."""
        with pytest.raises(ValueError, match="Inconsistent shapes"):
            TrajectoryData(
                scaling={"k1": np.zeros(8)},
                indices={"k1": np.arange(8)},
                mask=np.ones(16, dtype=bool),  # Wrong shape
            )


# =============================================================================
# OrderedTrajectory Tests
# =============================================================================


class TestOrderedTrajectory:
    """Tests for OrderedTrajectory container."""

    def test_n_points(self):
        """Test n_points property."""
        result = OrderedTrajectory(
            scaling={"k1": np.array([0.0, 0.5, 1.0])},
            indices={"k1": np.array([0, 1, 2])},
            dim_labels=("k1",),
        )
        assert result.n_points == 3

    def test_to_arrays_1d(self):
        """Test to_arrays with 1D data."""
        result = OrderedTrajectory(
            scaling={"k1": np.array([0.0, 0.5, 1.0])},
            indices={"k1": np.array([0, 1, 2])},
            dim_labels=("k1",),
        )
        scaling_arr, indices_arr = result.to_arrays()
        assert scaling_arr.shape == (1, 3)
        assert indices_arr.shape == (1, 3)

    def test_to_arrays_2d(self):
        """Test to_arrays with 2D data."""
        result = OrderedTrajectory(
            scaling={"k1": np.array([0.0, 0.5]), "k2": np.array([0.1, 0.2])},
            indices={"k1": np.array([0, 1]), "k2": np.array([2, 3])},
            dim_labels=("k1", "k2"),
        )
        scaling_arr, indices_arr = result.to_arrays()
        assert scaling_arr.shape == (2, 2)
        assert indices_arr.shape == (2, 2)

    def test_to_arrays_preserves_dim_order(self):
        """Test that to_arrays respects dim_labels ordering."""
        result = OrderedTrajectory(
            scaling={"k2": np.array([0.1, 0.2]), "k1": np.array([0.0, 0.5])},
            indices={"k2": np.array([2, 3]), "k1": np.array([0, 1])},
            dim_labels=("k1", "k2"),  # k1 first
        )
        scaling_arr, _ = result.to_arrays()
        np.testing.assert_array_equal(scaling_arr[0], [0.0, 0.5])  # k1
        np.testing.assert_array_equal(scaling_arr[1], [0.1, 0.2])  # k2


# =============================================================================
# TrajectoryOrderer Tests
# =============================================================================


class TestTrajectoryOrderer:
    """Tests for TrajectoryOrderer."""

    def test_order_returns_correct_type(
        self,
        simple_1d_data: TrajectoryData,
        identity_strategy: OrderingStrategy,
    ):
        """Test that order() returns OrderedTrajectory."""
        orderer = TrajectoryOrderer(identity_strategy)
        result = orderer.order(simple_1d_data)
        assert isinstance(result, OrderedTrajectory)

    def test_order_preserves_dim_labels(
        self,
        simple_2d_data: TrajectoryData,
        identity_strategy: OrderingStrategy,
    ):
        """Test that dim_labels are preserved in output."""
        orderer = TrajectoryOrderer(identity_strategy)
        result = orderer.order(simple_2d_data)
        assert result.dim_labels == simple_2d_data.dim_labels

    def test_order_applies_mask(
        self,
        accelerated_1d_data: TrajectoryData,
        identity_strategy: OrderingStrategy,
    ):
        """Test that mask is applied correctly."""
        orderer = TrajectoryOrderer(identity_strategy)
        result = orderer.order(accelerated_1d_data)
        assert result.n_points == accelerated_1d_data.n_sampled

    def test_order_flattens_output(
        self,
        simple_2d_data: TrajectoryData,
        identity_strategy: OrderingStrategy,
    ):
        """Test that output arrays are flattened."""
        orderer = TrajectoryOrderer(identity_strategy)
        result = orderer.order(simple_2d_data)
        for dim in result.dim_labels:
            assert result.scaling[dim].ndim == 1
            assert result.indices[dim].ndim == 1

    def test_order_applies_strategy(
        self,
        simple_1d_data: TrajectoryData,
        reverse_strategy: OrderingStrategy,
    ):
        """Test that strategy ordering is applied."""
        orderer = TrajectoryOrderer(reverse_strategy)
        result = orderer.order(simple_1d_data)
        # First point should be last index (15), last should be first (0)
        assert result.indices["k1"][0] == 15
        assert result.indices["k1"][-1] == 0

    def test_scaling_and_indices_stay_aligned(
        self,
        simple_1d_data: TrajectoryData,
        reverse_strategy: OrderingStrategy,
    ):
        """Test that scaling and indices remain aligned after ordering."""
        orderer = TrajectoryOrderer(reverse_strategy)
        result = orderer.order(simple_1d_data)
        # In original data, index 15 has scaling = 1.0, index 0 has scaling = -1.0
        assert result.scaling["k1"][0] == pytest.approx(1.0)
        assert result.scaling["k1"][-1] == pytest.approx(-1.0)


# =============================================================================
# CustomOrdering Tests
# =============================================================================


class TestCustomOrdering:
    """Tests for CustomOrdering strategy."""

    def test_name_property(self):
        """Test that custom name is returned."""
        strategy = CustomOrdering(
            order_func=lambda s, i, d: np.arange(len(s[d[0]]), dtype=np.intp),
            name="my_ordering",
        )
        assert strategy.name == "my_ordering"

    def test_default_name(self):
        """Test default name."""
        strategy = CustomOrdering(
            order_func=lambda s, i, d: np.arange(len(s[d[0]]), dtype=np.intp),
        )
        assert strategy.name == "custom"

    def test_repr(self):
        """Test string representation."""
        strategy = CustomOrdering(
            order_func=lambda s, i, d: np.arange(len(s[d[0]]), dtype=np.intp),
            name="test",
        )
        assert "CustomOrdering" in repr(strategy)
        assert "test" in repr(strategy)

    def test_receives_masked_data(self):
        """Test that strategy receives only masked points."""
        received_lengths = []

        def capture_strategy(
            scaling: dict[str, NDArray],
            indices: dict[str, NDArray],
            dim_labels: tuple[str, ...],
        ) -> NDArray[np.intp]:
            received_lengths.append(len(scaling[dim_labels[0]]))
            return np.arange(len(scaling[dim_labels[0]]), dtype=np.intp)

        # Create data with R=2 acceleration (8 out of 16 points)
        n = 16
        mask = np.zeros(n, dtype=bool)
        mask[::2] = True
        data = TrajectoryData(
            scaling={"k1": np.linspace(-1, 1, n)},
            indices={"k1": np.arange(n)},
            mask=mask,
            dim_labels=("k1",),
        )

        orderer = TrajectoryOrderer(CustomOrdering(capture_strategy))
        orderer.order(data)

        assert received_lengths[0] == 8  # Only masked points
