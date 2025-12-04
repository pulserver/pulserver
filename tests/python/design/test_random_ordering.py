"""Tests for random ordering strategy."""

import numpy as np
import pytest

from pulserver.design import TrajectoryData, TrajectoryOrderer, RandomOrdering


@pytest.fixture
def simple_1d_data() -> TrajectoryData:
    """Create simple 1D trajectory data."""
    n = 64
    return TrajectoryData(
        scaling={"k1": np.linspace(-1, 1, n)},
        indices={"k1": np.arange(n)},
        mask=np.ones(n, dtype=bool),
        dim_labels=("k1",),
    )


@pytest.fixture
def simple_2d_data() -> TrajectoryData:
    """Create simple 2D trajectory data."""
    n_k1, n_k2 = 8, 8
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


class TestRandomOrderingBasic:
    """Basic functionality tests for RandomOrdering."""

    def test_name(self):
        """Test strategy name."""
        strategy = RandomOrdering()
        assert strategy.name == "random"

    def test_seed_property(self):
        """Test seed property."""
        strategy = RandomOrdering(seed=42)
        assert strategy.seed == 42

    def test_seed_none(self):
        """Test default seed is None."""
        strategy = RandomOrdering()
        assert strategy.seed is None

    def test_repr(self):
        """Test string representation."""
        strategy = RandomOrdering(seed=42)
        assert "RandomOrdering" in repr(strategy)
        assert "42" in repr(strategy)


class TestRandomOrderingBehavior:
    """Behavior tests for RandomOrdering."""

    def test_returns_permutation(self, simple_1d_data: TrajectoryData):
        """Test that output is a valid permutation."""
        orderer = TrajectoryOrderer(RandomOrdering(seed=42))
        result = orderer.order(simple_1d_data)

        # Should contain all indices exactly once
        assert result.n_points == 64
        assert set(result.indices["k1"]) == set(range(64))

    def test_is_shuffled(self, simple_1d_data: TrajectoryData):
        """Test that output is actually shuffled (not sequential)."""
        orderer = TrajectoryOrderer(RandomOrdering(seed=42))
        result = orderer.order(simple_1d_data)

        # Should not be in sequential order
        sequential = np.arange(64)
        assert not np.array_equal(result.indices["k1"], sequential)

    def test_reproducible_with_seed(self, simple_1d_data: TrajectoryData):
        """Test that same seed gives same result."""
        orderer1 = TrajectoryOrderer(RandomOrdering(seed=42))
        orderer2 = TrajectoryOrderer(RandomOrdering(seed=42))

        result1 = orderer1.order(simple_1d_data)
        result2 = orderer2.order(simple_1d_data)

        np.testing.assert_array_equal(result1.indices["k1"], result2.indices["k1"])

    def test_different_seeds_differ(self, simple_1d_data: TrajectoryData):
        """Test that different seeds give different results."""
        orderer1 = TrajectoryOrderer(RandomOrdering(seed=42))
        orderer2 = TrajectoryOrderer(RandomOrdering(seed=43))

        result1 = orderer1.order(simple_1d_data)
        result2 = orderer2.order(simple_1d_data)

        assert not np.array_equal(result1.indices["k1"], result2.indices["k1"])

    def test_none_seed_varies(self, simple_1d_data: TrajectoryData):
        """Test that None seed gives different results each time."""
        orderer1 = TrajectoryOrderer(RandomOrdering(seed=None))
        orderer2 = TrajectoryOrderer(RandomOrdering(seed=None))

        result1 = orderer1.order(simple_1d_data)
        result2 = orderer2.order(simple_1d_data)

        # Very unlikely to be equal by chance (1/64!  probability)
        assert not np.array_equal(result1.indices["k1"], result2.indices["k1"])

    def test_2d_data(self, simple_2d_data: TrajectoryData):
        """Test random ordering on 2D data."""
        orderer = TrajectoryOrderer(RandomOrdering(seed=42))
        result = orderer.order(simple_2d_data)

        # Should have all 64 points
        assert result.n_points == 64

        # Both dimensions should be shuffled together
        # (scaling and indices stay aligned)
        for i in range(result.n_points):
            k1_idx = result.indices["k1"][i]
            k2_idx = result.indices["k2"][i]

            # Verify scaling matches expected value for those indices
            expected_k1_scaling = -1 + k1_idx * (2 / 7)
            expected_k2_scaling = -1 + k2_idx * (2 / 7)

            assert result.scaling["k1"][i] == pytest.approx(expected_k1_scaling)
            assert result.scaling["k2"][i] == pytest.approx(expected_k2_scaling)

    def test_with_mask(self):
        """Test random ordering with undersampled data."""
        n = 32
        mask = np.zeros(n, dtype=bool)
        mask[::2] = True  # R=2

        data = TrajectoryData(
            scaling={"k1": np.linspace(-1, 1, n)},
            indices={"k1": np.arange(n)},
            mask=mask,
            dim_labels=("k1",),
        )

        orderer = TrajectoryOrderer(RandomOrdering(seed=42))
        result = orderer.order(data)

        # Should only have masked points
        assert result.n_points == 16

        # All indices should be even (from mask)
        assert all(idx % 2 == 0 for idx in result.indices["k1"])

        # Should still be a permutation
        assert set(result.indices["k1"]) == set(range(0, 32, 2))
