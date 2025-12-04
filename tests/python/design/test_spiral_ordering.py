"""Tests for spiral ordering strategy."""

import math

import numpy as np
import pytest

from pulserver.design import (
    TrajectoryData,
    TrajectoryOrderer,
    GoldenAngle,
    SpiralOrdering,
    CustomDensity,
    UniformDensity,
    VariableDensity,
)


@pytest.fixture
def simple_2d_data() -> TrajectoryData:
    """Create simple 2D trajectory data."""
    n_k1, n_k2 = 16, 16
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
def small_2d_data() -> TrajectoryData:
    """Create small 2D trajectory data for detailed testing."""
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


class TestUniformDensity:
    """Tests for UniformDensity."""

    def test_name(self):
        """Test density name."""
        density = UniformDensity()
        assert density.name == "uniform"

    def test_returns_ones(self):
        """Test that uniform density returns all ones."""
        density = UniformDensity()
        r = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        result = density(r)
        np.testing.assert_array_equal(result, np.ones(5))


class TestVariableDensity:
    """Tests for VariableDensity."""

    def test_name(self):
        """Test density name."""
        density = VariableDensity()
        assert density.name == "variable"

    def test_center_density(self):
        """Test density at center."""
        density = VariableDensity(center_density=1.0, edge_density=0.2)
        r = np.array([0.0])
        result = density(r)
        assert result[0] == pytest.approx(1.0)

    def test_edge_density(self):
        """Test density at edge."""
        density = VariableDensity(center_density=1.0, edge_density=0.2)
        r = np.array([1.0])
        result = density(r)
        assert result[0] == pytest.approx(0.2)

    def test_monotonic_decrease(self):
        """Test that density decreases from center to edge."""
        density = VariableDensity(center_density=1.0, edge_density=0.2)
        r = np.linspace(0, 1, 10)
        result = density(r)
        assert np.all(np.diff(result) <= 0)

    def test_custom_densities(self):
        """Test custom center and edge densities."""
        density = VariableDensity(center_density=2.0, edge_density=0.5)
        r = np.array([0.0, 1.0])
        result = density(r)
        assert result[0] == pytest.approx(2.0)
        assert result[1] == pytest.approx(0.5)

    def test_transition_power(self):
        """Test that higher power creates sharper transition."""
        density_linear = VariableDensity(transition_power=1.0)
        density_cubic = VariableDensity(transition_power=3.0)

        r = np.array([0.5])
        linear_mid = density_linear(r)[0]
        cubic_mid = density_cubic(r)[0]

        # Cubic should be higher at midpoint (slower transition)
        assert cubic_mid > linear_mid

    def test_invalid_transition_point_raises(self):
        """Test that invalid transition point raises."""
        with pytest.raises(ValueError, match="transition_point"):
            VariableDensity(transition_point=1.5)

    def test_invalid_density_raises(self):
        """Test that non-positive density raises."""
        with pytest.raises(ValueError, match="Densities must be positive"):
            VariableDensity(center_density=0)


class TestCustomDensity:
    """Tests for CustomDensity."""

    def test_name(self):
        """Test custom name."""
        density = CustomDensity(lambda r: r, name="linear_ramp")
        assert density.name == "linear_ramp"

    def test_default_name(self):
        """Test default name."""
        density = CustomDensity(lambda r: r)
        assert density.name == "custom"

    def test_custom_function(self):
        """Test custom density function."""
        # Gaussian density
        density = CustomDensity(lambda r: np.exp(-(r**2) / 0.5))
        r = np.array([0.0, 0.5, 1.0])
        result = density(r)

        assert result[0] == pytest.approx(1.0)  # exp(0) = 1
        assert result[1] < result[0]  # Decreasing
        assert result[2] < result[1]  # Still decreasing


class TestSpiralOrderingBasic:
    """Basic functionality tests for SpiralOrdering."""

    def test_name_uniform(self):
        """Test name with uniform density."""
        strategy = SpiralOrdering()
        assert strategy.name == "spiral_uniform"

    def test_name_variable(self):
        """Test name with variable density."""
        strategy = SpiralOrdering(density=VariableDensity())
        assert strategy.name == "spiral_variable"

    def test_name_with_interleaves(self):
        """Test name includes interleave count."""
        strategy = SpiralOrdering(n_interleaves=4)
        assert strategy.name == "spiral_uniform_4int"

    def test_spiral_dims_property(self):
        """Test spiral_dims property."""
        strategy = SpiralOrdering(spiral_dims=("k1", "k2"))
        assert strategy.spiral_dims == ("k1", "k2")

    def test_n_interleaves_property(self):
        """Test n_interleaves property."""
        strategy = SpiralOrdering(n_interleaves=8)
        assert strategy.n_interleaves == 8

    def test_density_property(self):
        """Test density property."""
        density = VariableDensity()
        strategy = SpiralOrdering(density=density)
        assert strategy.density is density

    def test_invalid_interleaves_raises(self):
        """Test that n_interleaves < 1 raises."""
        with pytest.raises(ValueError, match="n_interleaves must be >= 1"):
            SpiralOrdering(n_interleaves=0)

    def test_repr(self):
        """Test string representation."""
        strategy = SpiralOrdering(n_interleaves=4)
        repr_str = repr(strategy)
        assert "SpiralOrdering" in repr_str
        assert "n_interleaves=4" in repr_str


class TestSpiralOrderingBehavior:
    """Behavior tests for SpiralOrdering."""

    def test_starts_at_center(self, simple_2d_data: TrajectoryData):
        """Test that spiral starts near center."""
        orderer = TrajectoryOrderer(SpiralOrdering())
        result = orderer.order(simple_2d_data)

        center = 7.5  # (0 + 15) / 2
        first_k1 = result.indices["k1"][0]
        first_k2 = result.indices["k2"][0]

        dist = math.sqrt((first_k1 - center) ** 2 + (first_k2 - center) ** 2)
        assert dist < 2.0

    def test_ends_at_edge(self, simple_2d_data: TrajectoryData):
        """Test that spiral ends near edge."""
        orderer = TrajectoryOrderer(SpiralOrdering())
        result = orderer.order(simple_2d_data)

        center = 7.5
        last_k1 = result.indices["k1"][-1]
        last_k2 = result.indices["k2"][-1]

        dist = math.sqrt((last_k1 - center) ** 2 + (last_k2 - center) ** 2)
        max_dist = math.sqrt(2) * 7.5  # Corner distance

        # Should be in outer region (> 50% of max distance)
        assert dist > 0.5 * max_dist

    def test_radius_generally_increases(self, simple_2d_data: TrajectoryData):
        """Test that radius generally increases along spiral."""
        orderer = TrajectoryOrderer(SpiralOrdering())
        result = orderer.order(simple_2d_data)

        center = 7.5
        radii = np.sqrt(
            (result.indices["k1"] - center) ** 2 + (result.indices["k2"] - center) ** 2
        )

        # Compare average radius in first vs last quarter
        n = len(radii)
        first_quarter_avg = np.mean(radii[: n // 4])
        last_quarter_avg = np.mean(radii[3 * n // 4 :])

        assert first_quarter_avg < last_quarter_avg

    def test_counterclockwise(self, small_2d_data: TrajectoryData):
        """Test counter-clockwise spiral."""
        orderer = TrajectoryOrderer(SpiralOrdering(clockwise=False))
        result = orderer.order(small_2d_data)

        # Just verify it runs without error and produces valid output
        assert result.n_points == 64

    def test_custom_center(self, simple_2d_data: TrajectoryData):
        """Test custom center specification."""
        custom_center = {"k1": 4.0, "k2": 4.0}
        orderer = TrajectoryOrderer(SpiralOrdering(center=custom_center))
        result = orderer.order(simple_2d_data)

        # First point should be near custom center
        first_dist = math.sqrt(
            (result.indices["k1"][0] - 4.0) ** 2 + (result.indices["k2"][0] - 4.0) ** 2
        )
        assert first_dist < 2.0

    def test_invalid_spiral_dim_raises(self, simple_2d_data: TrajectoryData):
        """Test that invalid spiral dimension raises."""
        orderer = TrajectoryOrderer(SpiralOrdering(spiral_dims=("k1", "k5")))
        with pytest.raises(ValueError, match="not found in data"):
            orderer.order(simple_2d_data)


class TestSpiralInterleaves:
    """Tests for spiral interleaves."""

    def test_single_interleave(self, small_2d_data: TrajectoryData):
        """Test single interleave (default)."""
        orderer = TrajectoryOrderer(SpiralOrdering(n_interleaves=1))
        result = orderer.order(small_2d_data)
        assert result.n_points == 64

    def test_multiple_interleaves(self, small_2d_data: TrajectoryData):
        """Test multiple interleaves."""
        orderer = TrajectoryOrderer(SpiralOrdering(n_interleaves=4))
        result = orderer.order(small_2d_data)
        assert result.n_points == 64

    def test_interleaves_with_golden_angle(self, small_2d_data: TrajectoryData):
        """Test interleaves with golden angle rotation."""
        orderer = TrajectoryOrderer(
            SpiralOrdering(
                n_interleaves=4,
                interleave_rotation=GoldenAngle(),
            )
        )
        result = orderer.order(small_2d_data)
        assert result.n_points == 64

    def test_interleaves_with_constant_rotation(self, small_2d_data: TrajectoryData):
        """Test interleaves with constant rotation angle."""
        orderer = TrajectoryOrderer(
            SpiralOrdering(
                n_interleaves=4,
                interleave_rotation=math.pi / 4,  # 45 degrees
            )
        )
        result = orderer.order(small_2d_data)
        assert result.n_points == 64


class TestSpiralVariableDensity:
    """Tests for variable density spiral."""

    def test_variable_density_ordering(self, simple_2d_data: TrajectoryData):
        """Test variable density spiral runs correctly."""
        orderer = TrajectoryOrderer(
            SpiralOrdering(
                density=VariableDensity(center_density=1.0, edge_density=0.2)
            )
        )
        result = orderer.order(simple_2d_data)
        assert result.n_points == 256

    def test_variable_density_starts_at_center(self, simple_2d_data: TrajectoryData):
        """Test that variable density spiral still starts at center."""
        orderer = TrajectoryOrderer(
            SpiralOrdering(
                density=VariableDensity(center_density=1.0, edge_density=0.2)
            )
        )
        result = orderer.order(simple_2d_data)

        center = 7.5
        first_dist = math.sqrt(
            (result.indices["k1"][0] - center) ** 2
            + (result.indices["k2"][0] - center) ** 2
        )
        assert first_dist < 2.0

    def test_variable_density_more_center_points_early(
        self, simple_2d_data: TrajectoryData
    ):
        """Test that variable density acquires more center points early."""
        orderer = TrajectoryOrderer(
            SpiralOrdering(
                density=VariableDensity(center_density=1.0, edge_density=0.1)
            )
        )
        result = orderer.order(simple_2d_data)

        center = 7.5
        radii = np.sqrt(
            (result.indices["k1"] - center) ** 2 + (result.indices["k2"] - center) ** 2
        )

        # In first 25% of acquisition, should be mostly near center
        n = len(radii)
        first_quarter_radii = radii[: n // 4]
        median_first_quarter = np.median(first_quarter_radii)

        # Median radius in first quarter should be less than half max
        assert median_first_quarter < 0.5 * np.max(radii)

    def test_custom_density_function(self, small_2d_data: TrajectoryData):
        """Test spiral with custom density function."""
        # Gaussian density
        gaussian_density = CustomDensity(
            lambda r: np.exp(-(r**2) / 0.3),
            name="gaussian",
        )
        orderer = TrajectoryOrderer(SpiralOrdering(density=gaussian_density))
        result = orderer.order(small_2d_data)
        assert result.n_points == 64


class TestSpiralIntegration:
    """Integration tests combining multiple features."""

    def test_variable_density_with_interleaves(self, simple_2d_data: TrajectoryData):
        """Test variable density combined with interleaves."""
        orderer = TrajectoryOrderer(
            SpiralOrdering(
                n_interleaves=4,
                interleave_rotation=GoldenAngle(),
                density=VariableDensity(center_density=1.0, edge_density=0.3),
            )
        )
        result = orderer.order(simple_2d_data)
        assert result.n_points == 256

    def test_all_features(self, simple_2d_data: TrajectoryData):
        """Test spiral with all features enabled."""
        orderer = TrajectoryOrderer(
            SpiralOrdering(
                spiral_dims=("k1", "k2"),
                n_interleaves=8,
                interleave_rotation=GoldenAngle(),
                density=VariableDensity(
                    center_density=1.0,
                    edge_density=0.2,
                    transition_power=2.5,
                ),
                clockwise=False,
                center={"k1": 7.5, "k2": 7.5},
            )
        )
        result = orderer.order(simple_2d_data)

        assert result.n_points == 256
        # Verify output has correct dimensions
        assert "k1" in result.indices
        assert "k2" in result.indices

    def test_with_undersampling(self):
        """Test spiral with undersampled data."""
        n_k1, n_k2 = 16, 16
        k1, k2 = np.meshgrid(
            np.linspace(-1, 1, n_k1),
            np.linspace(-1, 1, n_k2),
            indexing="ij",
        )
        i_k1, i_k2 = np.meshgrid(np.arange(n_k1), np.arange(n_k2), indexing="ij")

        # Random undersampling
        rng = np.random.default_rng(42)
        mask = rng.random((n_k1, n_k2)) < 0.5

        data = TrajectoryData(
            scaling={"k1": k1, "k2": k2},
            indices={"k1": i_k1, "k2": i_k2},
            mask=mask,
            dim_labels=("k1", "k2"),
        )

        orderer = TrajectoryOrderer(SpiralOrdering())
        result = orderer.order(data)

        # Should have approximately half the points
        assert result.n_points == np.sum(mask)
        assert result.n_points < 256
