# """Tests for angular increment generators."""

# import math

# import numpy as np
# import pytest

# from pulserver.design import (
#     GOLDEN_ANGLE,
#     GOLDEN_RATIO,
#     GoldenAngle,
#     GoldenMeans2D,
#     GoldenMeans3D,
#     LinearIncrement,
#     RationalGoldenAngle,
#     TinyGoldenAngle,
# )


# class TestConstants:
#     """Tests for module constants."""

#     def test_golden_ratio(self):
#         """Test golden ratio value."""
#         assert GOLDEN_RATIO == pytest.approx((1 + math.sqrt(5)) / 2)

#     def test_golden_angle(self):
#         """Test golden angle value."""
#         assert GOLDEN_ANGLE == pytest.approx(math.pi / GOLDEN_RATIO)
#         # ~111.246 degrees
#         assert math.degrees(GOLDEN_ANGLE) == pytest.approx(111.246, rel=1e-3)


# class TestLinearIncrement:
#     """Tests for LinearIncrement."""

#     def test_name(self):
#         """Test strategy name."""
#         inc = LinearIncrement()
#         assert inc.name == "linear"

#     def test_default_angular_range(self):
#         """Test default angular range is π."""
#         inc = LinearIncrement()
#         assert inc.angular_range == pytest.approx(math.pi)

#     def test_custom_angular_range(self):
#         """Test custom angular range."""
#         inc = LinearIncrement(angular_range=2 * math.pi)
#         assert inc.angular_range == pytest.approx(2 * math.pi)

#     def test_get_angles_single(self):
#         """Test single angle."""
#         inc = LinearIncrement()
#         angles = inc.get_angles(1)
#         assert len(angles) == 1
#         assert angles[0] == pytest.approx(0.0)

#     def test_get_angles_multiple(self):
#         """Test multiple angles with uniform spacing."""
#         inc = LinearIncrement(angular_range=math.pi)
#         angles = inc.get_angles(4)
#         expected = np.array([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
#         np.testing.assert_array_almost_equal(angles, expected)

#     def test_get_angles_with_start(self):
#         """Test angles with custom start."""
#         inc = LinearIncrement(angular_range=math.pi)
#         angles = inc.get_angles(2, start=math.pi / 4)
#         expected = np.array([math.pi / 4, 3 * math.pi / 4])
#         np.testing.assert_array_almost_equal(angles, expected)

#     def test_get_angles_empty(self):
#         """Test zero angles returns empty array."""
#         inc = LinearIncrement()
#         angles = inc.get_angles(0)
#         assert len(angles) == 0

#     def test_get_increment_raises(self):
#         """Test that get_increment raises for linear (depends on n)."""
#         inc = LinearIncrement()
#         with pytest.raises(ValueError, match="depends on n"):
#             inc.get_increment()


# class TestGoldenAngle:
#     """Tests for GoldenAngle."""

#     def test_name_half_circle(self):
#         """Test name for half circle mode."""
#         inc = GoldenAngle(full_circle=False)
#         assert inc.name == "golden_angle"

#     def test_name_full_circle(self):
#         """Test name for full circle mode."""
#         inc = GoldenAngle(full_circle=True)
#         assert inc.name == "golden_angle_full"

#     def test_increment_half_circle(self):
#         """Test increment for half circle mode."""
#         inc = GoldenAngle(full_circle=False)
#         assert inc.get_increment() == pytest.approx(GOLDEN_ANGLE)

#     def test_increment_full_circle(self):
#         """Test increment for full circle mode."""
#         inc = GoldenAngle(full_circle=True)
#         assert inc.get_increment() == pytest.approx(2 * math.pi / GOLDEN_RATIO)

#     def test_get_angles(self):
#         """Test angle generation."""
#         inc = GoldenAngle()
#         angles = inc.get_angles(3)
#         expected = np.array([0, GOLDEN_ANGLE, 2 * GOLDEN_ANGLE])
#         np.testing.assert_array_almost_equal(angles, expected)

#     def test_angles_cover_space_well(self):
#         """Test that golden angle provides good coverage."""
#         inc = GoldenAngle()
#         angles = inc.get_angles(100) % math.pi  # Wrap to [0, π)

#         # Check that angles are well distributed (low variance in gaps)
#         sorted_angles = np.sort(angles)
#         gaps = np.diff(sorted_angles)
#         # Gap variance should be relatively low for good coverage
#         assert np.std(gaps) < np.mean(gaps)


# class TestTinyGoldenAngle:
#     """Tests for TinyGoldenAngle."""

#     def test_name(self):
#         """Test name includes order."""
#         inc = TinyGoldenAngle(order=7)
#         assert inc.name == "tiny_golden_angle_7"

#     def test_order_property(self):
#         """Test order property."""
#         inc = TinyGoldenAngle(order=5)
#         assert inc.order == 5

#     def test_invalid_order_raises(self):
#         """Test that order < 1 raises."""
#         with pytest.raises(ValueError, match="must be >= 1"):
#             TinyGoldenAngle(order=0)

#     def test_order_1_equals_golden_angle(self):
#         """Test that order 1 gives standard golden angle."""
#         inc = TinyGoldenAngle(order=1)
#         assert inc.get_increment() == pytest.approx(GOLDEN_ANGLE, rel=1e-2)

#     def test_higher_order_smaller_increment(self):
#         """Test that higher orders give smaller increments."""
#         inc3 = TinyGoldenAngle(order=3)
#         inc7 = TinyGoldenAngle(order=7)
#         assert inc7.get_increment() < inc3.get_increment()

#     def test_order_7_value(self):
#         """Test known value for order 7 (~23.6°)."""
#         inc = TinyGoldenAngle(order=7)
#         # τ_7 ≈ 23.6° = 0.412 radians
#         assert math.degrees(inc.get_increment()) == pytest.approx(23.6, rel=0.1)


# class TestGoldenMeans2D:
#     """Tests for GoldenMeans2D."""

#     def test_name(self):
#         """Test strategy name."""
#         inc = GoldenMeans2D()
#         assert inc.name == "golden_means_2d"

#     def test_golden_mean_value(self):
#         """Test 2D golden mean is correct (root of x³ = x + 1)."""
#         gm = GoldenMeans2D.GOLDEN_MEAN_2D
#         assert gm**3 == pytest.approx(gm + 1, rel=1e-10)

#     def test_get_increment(self):
#         """Test increment is π / golden_mean_2d."""
#         inc = GoldenMeans2D()
#         expected = math.pi / GoldenMeans2D.GOLDEN_MEAN_2D
#         assert inc.get_increment() == pytest.approx(expected)


# class TestGoldenMeans3D:
#     """Tests for GoldenMeans3D."""

#     def test_name(self):
#         """Test strategy name."""
#         inc = GoldenMeans3D()
#         assert inc.name == "golden_means_3d"

#     def test_golden_mean_value(self):
#         """Test 3D golden mean is correct (root of x⁴ = x + 1)."""
#         gm = GoldenMeans3D.GOLDEN_MEAN_3D
#         assert gm**4 == pytest.approx(gm + 1, rel=1e-10)

#     def test_get_angles_3d(self):
#         """Test 3D angle generation."""
#         inc = GoldenMeans3D()
#         az, pol = inc.get_angles_3d(5)
#         assert len(az) == 5
#         assert len(pol) == 5

#     def test_polar_increment_smaller(self):
#         """Test that polar increment is smaller (squared golden mean)."""
#         inc = GoldenMeans3D()
#         assert inc.increment_polar < inc.get_increment()


# class TestRationalGoldenAngle:
#     """Tests for RationalGoldenAngle."""

#     def test_name(self):
#         """Test name includes index."""
#         inc = RationalGoldenAngle(fibonacci_index=7)
#         assert inc.name == "rational_golden_7"

#     def test_fibonacci_index_property(self):
#         """Test fibonacci_index property."""
#         inc = RationalGoldenAngle(fibonacci_index=5)
#         assert inc.fibonacci_index == 5

#     def test_uniform_at_property(self):
#         """Test uniform_at gives correct Fibonacci number."""
#         inc = RationalGoldenAngle(fibonacci_index=7)
#         # F_7 = 13, F_9 = 34 -> uniform at 34
#         assert inc.uniform_at == 34

#     def test_invalid_index_low_raises(self):
#         """Test that index < 2 raises."""
#         with pytest.raises(ValueError, match="must be >= 2"):
#             RationalGoldenAngle(fibonacci_index=1)

#     def test_invalid_index_high_raises(self):
#         """Test that too-high index raises."""
#         with pytest.raises(ValueError, match="must be <"):
#             RationalGoldenAngle(fibonacci_index=100)

#     def test_uniform_coverage(self):
#         """Test that n=uniform_at gives uniform coverage."""
#         inc = RationalGoldenAngle(fibonacci_index=5)  # uniform at 13
#         angles = inc.get_angles(13) % math.pi

#         # Sort and check gaps are nearly equal
#         sorted_angles = np.sort(angles)
#         gaps = np.diff(sorted_angles)
#         assert np.std(gaps) < 0.01 * np.mean(gaps)
