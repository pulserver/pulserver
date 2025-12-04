"""Tests for LinearOrdering strategy."""

import numpy as np
import pytest

from pulserver.design import TrajectoryData, TrajectoryOrderer, LinearOrdering


@pytest.fixture
def simple_1d_data() -> TrajectoryData:
    """Create simple 1D trajectory data."""
    n = 8
    return TrajectoryData(
        scaling={"k1": np.linspace(-1, 1, n)},
        indices={"k1": np.arange(n)},
        mask=np.ones(n, dtype=bool),
        dim_labels=("k1",),
    )


@pytest.fixture
def simple_2d_data() -> TrajectoryData:
    """Create simple 2D trajectory data (k1 x k2)."""
    n_k1, n_k2 = 4, 3
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
def simple_3d_data() -> TrajectoryData:
    """Create simple 3D trajectory data (k1 x k2 x avg)."""
    n_k1, n_k2, n_avg = 3, 3, 2
    k1, k2, avg = np.meshgrid(
        np.linspace(-1, 1, n_k1),
        np.linspace(-1, 1, n_k2),
        np.zeros(n_avg),  # avg doesn't have scaling
        indexing="ij",
    )
    i_k1, i_k2, i_avg = np.meshgrid(
        np.arange(n_k1),
        np.arange(n_k2),
        np.arange(n_avg),
        indexing="ij",
    )
    return TrajectoryData(
        scaling={"k1": k1, "k2": k2, "avg": avg},
        indices={"k1": i_k1, "k2": i_k2, "avg": i_avg},
        mask=np.ones((n_k1, n_k2, n_avg), dtype=bool),
        dim_labels=("k1", "k2", "avg"),
    )


class TestLinearOrderingBasic:
    """Basic functionality tests for LinearOrdering."""

    def test_name(self):
        """Test strategy name."""
        strategy = LinearOrdering()
        assert strategy.name == "linear"

    def test_dim_priority_property(self):
        """Test dim_priority property."""
        strategy = LinearOrdering(dim_priority=["k1", "k2"])
        assert strategy.dim_priority == ("k1", "k2")

    def test_dim_priority_none(self):
        """Test dim_priority defaults to None."""
        strategy = LinearOrdering()
        assert strategy.dim_priority is None

    def test_repr(self):
        """Test string representation."""
        strategy = LinearOrdering(dim_priority=["k1", "k2"], reverse={"k1": True})
        repr_str = repr(strategy)
        assert "LinearOrdering" in repr_str
        assert "k1" in repr_str
        assert "k2" in repr_str


class TestLinearOrdering1D:
    """Tests for 1D linear ordering."""

    def test_1d_ascending(self, simple_1d_data: TrajectoryData):
        """Test 1D ordering in ascending order."""
        orderer = TrajectoryOrderer(LinearOrdering())
        result = orderer.order(simple_1d_data)

        # Should be in sequential ascending order
        expected_indices = np.arange(8)
        np.testing.assert_array_equal(result.indices["k1"], expected_indices)

    def test_1d_descending(self, simple_1d_data: TrajectoryData):
        """Test 1D ordering in descending order."""
        orderer = TrajectoryOrderer(LinearOrdering(reverse=True))
        result = orderer.order(simple_1d_data)

        # Should be in sequential descending order
        expected_indices = np.arange(7, -1, -1)
        np.testing.assert_array_equal(result.indices["k1"], expected_indices)


class TestLinearOrdering2D:
    """Tests for 2D linear ordering."""

    def test_2d_k1_outer_k2_inner(self, simple_2d_data: TrajectoryData):
        """Test 2D ordering with k1 outer, k2 inner (row-by-row)."""
        orderer = TrajectoryOrderer(LinearOrdering(dim_priority=["k1", "k2"]))
        result = orderer.order(simple_2d_data)

        # k1=0: k2=0,1,2; k1=1: k2=0,1,2; etc.
        expected_k1 = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
        expected_k2 = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
        np.testing.assert_array_equal(result.indices["k1"], expected_k1)
        np.testing.assert_array_equal(result.indices["k2"], expected_k2)

    def test_2d_k2_outer_k1_inner(self, simple_2d_data: TrajectoryData):
        """Test 2D ordering with k2 outer, k1 inner (column-by-column)."""
        orderer = TrajectoryOrderer(LinearOrdering(dim_priority=["k2", "k1"]))
        result = orderer.order(simple_2d_data)

        # k2=0: k1=0,1,2,3; k2=1: k1=0,1,2,3; etc.
        expected_k1 = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
        expected_k2 = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
        np.testing.assert_array_equal(result.indices["k1"], expected_k1)
        np.testing.assert_array_equal(result.indices["k2"], expected_k2)

    def test_2d_default_uses_dim_labels_order(self, simple_2d_data: TrajectoryData):
        """Test that default priority follows dim_labels order."""
        orderer = TrajectoryOrderer(LinearOrdering())  # No explicit priority
        result = orderer.order(simple_2d_data)

        # dim_labels is ('k1', 'k2'), so k1 outer, k2 inner
        expected_k1 = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
        expected_k2 = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
        np.testing.assert_array_equal(result.indices["k1"], expected_k1)
        np.testing.assert_array_equal(result.indices["k2"], expected_k2)

    def test_2d_reverse_outer_dimension(self, simple_2d_data: TrajectoryData):
        """Test reversing outer dimension only."""
        orderer = TrajectoryOrderer(
            LinearOrdering(dim_priority=["k1", "k2"], reverse={"k1": True})
        )
        result = orderer.order(simple_2d_data)

        # k1 reversed: k1=3,2,1,0; k2 still ascending within each k1
        expected_k1 = [3, 3, 3, 2, 2, 2, 1, 1, 1, 0, 0, 0]
        expected_k2 = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
        np.testing.assert_array_equal(result.indices["k1"], expected_k1)
        np.testing.assert_array_equal(result.indices["k2"], expected_k2)

    def test_2d_reverse_inner_dimension(self, simple_2d_data: TrajectoryData):
        """Test reversing inner dimension only."""
        orderer = TrajectoryOrderer(
            LinearOrdering(dim_priority=["k1", "k2"], reverse={"k2": True})
        )
        result = orderer.order(simple_2d_data)

        # k1 ascending, k2 reversed within each k1
        expected_k1 = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
        expected_k2 = [2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0]
        np.testing.assert_array_equal(result.indices["k1"], expected_k1)
        np.testing.assert_array_equal(result.indices["k2"], expected_k2)

    def test_2d_reverse_all_dimensions(self, simple_2d_data: TrajectoryData):
        """Test reversing all dimensions with bool flag."""
        orderer = TrajectoryOrderer(
            LinearOrdering(dim_priority=["k1", "k2"], reverse=True)
        )
        result = orderer.order(simple_2d_data)

        # Both dimensions reversed
        expected_k1 = [3, 3, 3, 2, 2, 2, 1, 1, 1, 0, 0, 0]
        expected_k2 = [2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0]
        np.testing.assert_array_equal(result.indices["k1"], expected_k1)
        np.testing.assert_array_equal(result.indices["k2"], expected_k2)


class TestLinearOrdering3D:
    """Tests for 3D linear ordering."""

    def test_3d_ordering(self, simple_3d_data: TrajectoryData):
        """Test 3D ordering with explicit priority."""
        orderer = TrajectoryOrderer(LinearOrdering(dim_priority=["avg", "k1", "k2"]))
        result = orderer.order(simple_3d_data)

        # avg=0: all k1,k2 combos; avg=1: all k1,k2 combos
        # Within each avg: k1 outer, k2 inner
        assert result.n_points == 18

        # First 9 points should have avg=0
        np.testing.assert_array_equal(result.indices["avg"][:9], np.zeros(9))
        # Last 9 points should have avg=1
        np.testing.assert_array_equal(result.indices["avg"][9:], np.ones(9))

    def test_3d_partial_priority(self, simple_3d_data: TrajectoryData):
        """Test that partial dim_priority raises error for missing dims."""
        orderer = TrajectoryOrderer(
            LinearOrdering(dim_priority=["k1", "k2"])  # Missing 'avg'
        )
        # This should still work - only specified dimensions are used for sorting
        # Unspecified dimensions will have arbitrary order among ties
        result = orderer.order(simple_3d_data)
        assert result.n_points == 18


class TestLinearOrderingEdgeCases:
    """Edge cases and error handling tests."""

    def test_invalid_dimension_raises(self, simple_2d_data: TrajectoryData):
        """Test that invalid dimension in priority raises ValueError."""
        orderer = TrajectoryOrderer(
            LinearOrdering(dim_priority=["k1", "k3"])  # k3 doesn't exist
        )
        with pytest.raises(ValueError, match="not found in data"):
            orderer.order(simple_2d_data)

    def test_with_acceleration(self):
        """Test linear ordering with undersampled data."""
        n_k1, n_k2 = 8, 6
        k1, k2 = np.meshgrid(
            np.linspace(-1, 1, n_k1),
            np.linspace(-1, 1, n_k2),
            indexing="ij",
        )
        i_k1, i_k2 = np.meshgrid(np.arange(n_k1), np.arange(n_k2), indexing="ij")

        # R=2 undersampling in k1
        mask = np.zeros((n_k1, n_k2), dtype=bool)
        mask[::2, :] = True

        data = TrajectoryData(
            scaling={"k1": k1, "k2": k2},
            indices={"k1": i_k1, "k2": i_k2},
            mask=mask,
            dim_labels=("k1", "k2"),
        )

        orderer = TrajectoryOrderer(LinearOrdering(dim_priority=["k1", "k2"]))
        result = orderer.order(data)

        # Should have half the points
        assert result.n_points == 24  # 4 * 6

        # Only even k1 indices should be present
        assert set(result.indices["k1"]) == {0, 2, 4, 6}

    def test_scaling_stays_aligned_with_indices(self, simple_2d_data: TrajectoryData):
        """Test that scaling values stay aligned with their indices."""
        orderer = TrajectoryOrderer(LinearOrdering(dim_priority=["k2", "k1"]))
        result = orderer.order(simple_2d_data)

        # For each point, verify scaling matches expected value for that index
        for i in range(result.n_points):
            k1_idx = result.indices["k1"][i]
            k2_idx = result.indices["k2"][i]

            # Scaling should be linearly spaced -1 to 1
            expected_k1_scaling = -1 + k1_idx * (2 / 3)  # 4 points: -1, -1/3, 1/3, 1
            expected_k2_scaling = -1 + k2_idx * (2 / 2)  # 3 points: -1, 0, 1

            assert result.scaling["k1"][i] == pytest.approx(expected_k1_scaling)
            assert result.scaling["k2"][i] == pytest.approx(expected_k2_scaling)
