"""Tests for centric ordering strategies."""

import math

import numpy as np
import pytest

from pulserver.design import (
    TrajectoryData,
    TrajectoryOrderer,
    GoldenAngle,
    CenterOutOrdering,
    FullSpokeOrdering,
)


@pytest.fixture
def simple_1d_data() -> TrajectoryData:
    """Create simple 1D trajectory data."""
    n = 16
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


@pytest.fixture
def radial_2d_data() -> TrajectoryData:
    """Create 2D radial trajectory data (spokes x readout)."""
    n_readout, n_spokes = 16, 8
    # k0 is readout (spoke direction), k1 is spoke index
    k0 = np.linspace(-1, 1, n_readout)
    k1 = np.arange(n_spokes)
    k0_grid, k1_grid = np.meshgrid(k0, k1, indexing="ij")
    i_k0, i_k1 = np.meshgrid(np.arange(n_readout), np.arange(n_spokes), indexing="ij")
    return TrajectoryData(
        scaling={"k0": k0_grid, "k1": k1_grid.astype(float)},
        indices={"k0": i_k0, "k1": i_k1},
        mask=np.ones((n_readout, n_spokes), dtype=bool),
        dim_labels=("k0", "k1"),
    )


class TestCenterOutOrderingBasic:
    """Basic functionality tests for CenterOutOrdering."""

    def test_name_with_linear(self):
        """Test name with linear increment."""
        strategy = CenterOutOrdering()
        assert strategy.name == "center_out_linear"

    def test_name_with_golden_angle(self):
        """Test name with golden angle increment."""
        strategy = CenterOutOrdering(angular_increment=GoldenAngle())
        assert strategy.name == "center_out_golden_angle"

    def test_radial_dims_property(self):
        """Test radial_dims property."""
        strategy = CenterOutOrdering(radial_dims=["k1", "k2"])
        assert strategy.radial_dims == ("k1", "k2")

    def test_angular_increment_property(self):
        """Test angular_increment property."""
        ga = GoldenAngle()
        strategy = CenterOutOrdering(angular_increment=ga)
        assert strategy.angular_increment is ga

    def test_repr(self):
        """Test string representation."""
        strategy = CenterOutOrdering(radial_dims=["k1"])
        repr_str = repr(strategy)
        assert "CenterOutOrdering" in repr_str
        assert "k1" in repr_str


class TestCenterOutOrdering1D:
    """Tests for 1D center-out ordering."""

    def test_1d_starts_at_center(self, simple_1d_data: TrajectoryData):
        """Test that 1D center-out starts near center."""
        orderer = TrajectoryOrderer(CenterOutOrdering())
        result = orderer.order(simple_1d_data)

        # Center is at index 7. 5, so first point should be 7 or 8
        assert result.indices["k1"][0] in [7, 8]

    def test_1d_moves_outward(self, simple_1d_data: TrajectoryData):
        """Test that 1D center-out moves outward from center."""
        orderer = TrajectoryOrderer(CenterOutOrdering())
        result = orderer.order(simple_1d_data)

        center = 7.5
        distances = np.abs(result.indices["k1"] - center)

        # Distances should generally increase (with some tolerance for ties)
        # Check that average distance in first half < second half
        mid = len(distances) // 2
        assert np.mean(distances[:mid]) < np.mean(distances[mid:])


class TestCenterOutOrdering2D:
    """Tests for 2D center-out ordering."""

    def test_2d_starts_at_center(self, simple_2d_data: TrajectoryData):
        """Test that 2D center-out starts near center."""
        orderer = TrajectoryOrderer(CenterOutOrdering())
        result = orderer.order(simple_2d_data)

        center = 3.5  # (0+7)/2
        first_k1 = result.indices["k1"][0]
        first_k2 = result.indices["k2"][0]

        # First point should be within 1 of center
        dist = math.sqrt((first_k1 - center) ** 2 + (first_k2 - center) ** 2)
        assert dist < 1.5

    def test_2d_moves_outward(self, simple_2d_data: TrajectoryData):
        """Test that 2D center-out moves outward."""
        orderer = TrajectoryOrderer(CenterOutOrdering())
        result = orderer.order(simple_2d_data)

        center = 3.5
        distances = np.sqrt(
            (result.indices["k1"] - center) ** 2 + (result.indices["k2"] - center) ** 2
        )

        # Distances should generally increase
        mid = len(distances) // 2
        assert np.mean(distances[:mid]) < np.mean(distances[mid:])

    def test_2d_with_golden_angle(self, simple_2d_data: TrajectoryData):
        """Test 2D center-out with golden angle."""
        orderer = TrajectoryOrderer(CenterOutOrdering(angular_increment=GoldenAngle()))
        result = orderer.order(simple_2d_data)

        # Should still start at center
        center = 3.5
        first_dist = math.sqrt(
            (result.indices["k1"][0] - center) ** 2
            + (result.indices["k2"][0] - center) ** 2
        )
        assert first_dist < 1.5

    def test_custom_center(self, simple_2d_data: TrajectoryData):
        """Test custom center specification."""
        custom_center = {"k1": 2.0, "k2": 2.0}
        orderer = TrajectoryOrderer(CenterOutOrdering(center=custom_center))
        result = orderer.order(simple_2d_data)

        # First point should be near custom center
        first_dist = math.sqrt(
            (result.indices["k1"][0] - 2.0) ** 2 + (result.indices["k2"][0] - 2.0) ** 2
        )
        assert first_dist < 1.5

    def test_invalid_radial_dim_raises(self, simple_2d_data: TrajectoryData):
        """Test that invalid radial dimension raises."""
        orderer = TrajectoryOrderer(CenterOutOrdering(radial_dims=["k1", "k3"]))
        with pytest.raises(ValueError, match="not found in data"):
            orderer.order(simple_2d_data)


class TestFullSpokeOrderingBasic:
    """Basic functionality tests for FullSpokeOrdering."""

    def test_name_with_linear(self):
        """Test name with linear increment."""
        strategy = FullSpokeOrdering(spoke_dim="k0")
        assert strategy.name == "full_spoke_linear"

    def test_name_with_golden_angle(self):
        """Test name with golden angle increment."""
        strategy = FullSpokeOrdering(
            spoke_dim="k0",
            angular_increment=GoldenAngle(full_circle=True),
        )
        assert strategy.name == "full_spoke_golden_angle_full"

    def test_spoke_dim_property(self):
        """Test spoke_dim property."""
        strategy = FullSpokeOrdering(spoke_dim="k0")
        assert strategy.spoke_dim == "k0"

    def test_angular_dims_property(self):
        """Test angular_dims property."""
        strategy = FullSpokeOrdering(spoke_dim="k0", angular_dims=["k1", "k2"])
        assert strategy.angular_dims == ("k1", "k2")

    def test_repr(self):
        """Test string representation."""
        strategy = FullSpokeOrdering(spoke_dim="k0")
        repr_str = repr(strategy)
        assert "FullSpokeOrdering" in repr_str
        assert "k0" in repr_str


class TestFullSpokeOrdering:
    """Tests for FullSpokeOrdering behavior."""

    def test_spokes_are_contiguous(self, radial_2d_data: TrajectoryData):
        """Test that each spoke's readout points are contiguous."""
        orderer = TrajectoryOrderer(FullSpokeOrdering(spoke_dim="k0"))
        result = orderer.order(radial_2d_data)

        # Group by k1 (spoke index) and check each group is contiguous
        n_readout = 16
        n_spokes = 8

        for spoke_idx in range(n_spokes):
            # Find where this spoke appears in output
            spoke_positions = np.where(result.indices["k1"] == spoke_idx)[0]
            assert len(spoke_positions) == n_readout

            # Positions should be contiguous
            assert np.all(np.diff(spoke_positions) == 1)

    def test_spoke_readout_order(self, radial_2d_data: TrajectoryData):
        """Test that readout within spoke is sequential."""
        orderer = TrajectoryOrderer(FullSpokeOrdering(spoke_dim="k0"))
        result = orderer.order(radial_2d_data)

        n_readout = 16

        # Check first spoke's readout order
        first_spoke_k0 = result.indices["k0"][:n_readout]
        expected = np.arange(n_readout)
        np.testing.assert_array_equal(first_spoke_k0, expected)

    def test_with_golden_angle(self, radial_2d_data: TrajectoryData):
        """Test full spoke with golden angle ordering."""
        orderer = TrajectoryOrderer(
            FullSpokeOrdering(
                spoke_dim="k0",
                angular_increment=GoldenAngle(full_circle=True),
            )
        )
        result = orderer.order(radial_2d_data)

        # Spokes should still be contiguous
        n_readout = 16
        for i in range(0, result.n_points, n_readout):
            spoke_k1 = result.indices["k1"][i : i + n_readout]
            # All points in this segment should have same spoke index
            assert len(np.unique(spoke_k1)) == 1

    def test_invalid_spoke_dim_raises(self, radial_2d_data: TrajectoryData):
        """Test that invalid spoke dimension raises."""
        orderer = TrajectoryOrderer(FullSpokeOrdering(spoke_dim="k5"))
        with pytest.raises(ValueError, match="not found in data"):
            orderer.order(radial_2d_data)

    def test_invalid_angular_dim_raises(self, radial_2d_data: TrajectoryData):
        """Test that invalid angular dimension raises."""
        orderer = TrajectoryOrderer(
            FullSpokeOrdering(spoke_dim="k0", angular_dims=["k5"])
        )
        with pytest.raises(ValueError, match="not found in data"):
            orderer.order(radial_2d_data)
