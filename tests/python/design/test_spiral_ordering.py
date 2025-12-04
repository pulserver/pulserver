"""Tests for spiral ordering strategy."""

import numpy as np
import pytest

from pulserver.design import SpiralOrdering


class TestSpiralOrderingBasic:
    """Basic functionality tests."""
    
    def test_name(self):
        """Test strategy name."""
        assert SpiralOrdering().name == "spiral_ccw"
        assert SpiralOrdering(clockwise=True).name == "spiral_cw"
    
    def test_properties_default(self):
        """Test default property values."""
        strategy = SpiralOrdering()
        
        assert strategy.center is None
        assert strategy.clockwise is False
        assert strategy.start_angle == 0.0
    
    def test_properties_custom(self):
        """Test custom property values."""
        center = np.array([1.0, 2.0])
        strategy = SpiralOrdering(center=center, clockwise=True, start_angle=0.5)
        
        np.testing.assert_array_equal(strategy.center, center)
        assert strategy.clockwise is True
        assert strategy.start_angle == 0.5
    
    def test_repr_default(self):
        """Test repr with defaults."""
        assert repr(SpiralOrdering()) == "SpiralOrdering()"
    
    def test_repr_with_params(self):
        """Test repr with parameters."""
        strategy = SpiralOrdering(clockwise=True, start_angle=0.5)
        repr_str = repr(strategy)
        
        assert "clockwise=True" in repr_str
        assert "start_angle=0.5" in repr_str


class TestSpiralOrdering2D:
    """Tests for 2D spiral ordering."""
    
    def test_output_shape(self):
        """Test output shape."""
        strategy = SpiralOrdering()
        coords = self._make_grid(8)
        
        result = strategy.compute_order(coords)
        
        assert result.shape == (1, 64)
    
    def test_starts_at_center(self):
        """Test that spiral starts at center."""
        strategy = SpiralOrdering()
        coords = self._make_grid(8)
        
        result = strategy.compute_order(coords)
        
        # First points should be near center
        center = coords.mean(axis=1)
        sorted_coords = coords[:, result[0]]
        first_point = sorted_coords[:, 0]
        
        # Distance from center should be small
        dist = np.sqrt(np.sum((first_point - center) ** 2))
        max_dist = np. sqrt(np.sum((coords[:, 0] - center) ** 2))
        assert dist < max_dist * 0.3
    
    def test_radius_generally_increases(self):
        """Test that radius generally increases along spiral."""
        strategy = SpiralOrdering()
        coords = self._make_grid(16)
        
        result = strategy.compute_order(coords)
        
        center = coords.mean(axis=1, keepdims=True)
        sorted_coords = coords[:, result[0]]
        radii = np.sqrt(np.sum((sorted_coords - center) ** 2, axis=0))
        
        # Divide into quarters and check mean radius increases
        n = len(radii)
        quarter_means = [radii[i*n//4:(i+1)*n//4]. mean() for i in range(4)]
        assert quarter_means[0] < quarter_means[-1]
    
    def test_clockwise_differs_from_ccw(self):
        """Test that clockwise and counter-clockwise differ."""
        coords = self._make_grid(8)
        
        order_ccw = SpiralOrdering(clockwise=False). compute_order(coords)
        order_cw = SpiralOrdering(clockwise=True).compute_order(coords)
        
        assert not np.array_equal(order_ccw, order_cw)
    
    def test_start_angle_rotates_spiral(self):
        """Test that start_angle rotates the spiral."""
        coords = self._make_grid(8)
        
        order1 = SpiralOrdering(start_angle=0.0).compute_order(coords)
        order2 = SpiralOrdering(start_angle=np.pi/2). compute_order(coords)
        
        assert not np.array_equal(order1, order2)
    
    def test_custom_center(self):
        """Test spiral with custom center."""
        strategy = SpiralOrdering(center=np.array([0.0, 0.0]))
        coords = self._make_grid(8)
        
        result = strategy.compute_order(coords)
        
        # First point should be closest to (0, 0)
        sorted_coords = coords[:, result[0]]
        first_dist = np.sqrt(sorted_coords[0, 0]**2 + sorted_coords[1, 0]**2)
        
        # Should be among the closest points to origin
        all_dists = np.sqrt(coords[0]**2 + coords[1]**2)
        assert first_dist <= np.sort(all_dists)[4]  # Within first few closest
    
    def _make_grid(self, size: int) -> NDArray:
        """Helper to create a centered 2D grid."""
        half = size // 2
        ky, kz = np.meshgrid(
            np.arange(size) - half,
            np.arange(size) - half,
            indexing='ij'
        )
        return np.stack([ky.ravel(), kz.ravel()])


class TestSpiralOrdering1D:
    """Tests for 1D fallback behavior."""
    
    def test_1d_fallback_to_center_out(self):
        """Test that 1D uses center-out ordering."""
        strategy = SpiralOrdering()
        coords = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])
        
        result = strategy. compute_order(coords)
        
        assert result.shape == (1, 9)
        # Should start from center (0)
        sorted_coords = coords[result[0]]
        assert sorted_coords[0] == 0
        # Radii should be non-decreasing
        radii = np.abs(sorted_coords)
        assert np.all(np.diff(radii) >= 0)


class TestSpiralOrdering3D:
    """Tests for 3D coordinates."""
    
    def test_3d_uses_first_two_dims(self):
        """Test that 3D uses first two dimensions for spiral."""
        strategy = SpiralOrdering()
        # 4x4x4 grid
        k0, k1, k2 = np.meshgrid(
            np.arange(4) - 1. 5,
            np.arange(4) - 1. 5,
            np.arange(4) - 1. 5,
            indexing='ij'
        )
        coords = np.stack([k0.ravel(), k1. ravel(), k2.ravel()])
        
        result = strategy.compute_order(coords)
        
        assert result.shape == (1, 64)
        # Should be valid permutation
        assert set(result[0]) == set(range(64))


class TestSpiralOrderingWithMask:
    """Tests with sampling mask."""
    
    def test_masked_spiral(self):
        """Test spiral ordering with mask."""
        strategy = SpiralOrdering()
        ky, kz = np.meshgrid(np.arange(8) - 4, np. arange(8) - 4, indexing='ij')
        coords = np.stack([ky.ravel(), kz.ravel()])
        
        # Random mask
        rng = np.random.default_rng(42)
        mask = rng.random(64) > 0.5
        n_sampled = mask.sum()
        
        result = strategy.compute_order(coords, mask=mask)
        
        assert result.shape == (1, n_sampled)
        assert set(result[0]) == set(range(n_sampled))


class TestSpiralOrderingWithSegments:
    """Tests with segmentation."""
    
    def test_segments(self):
        """Test spiral ordering with segments."""
        strategy = SpiralOrdering()
        ky, kz = np.meshgrid(np.arange(8) - 4, np.arange(8) - 4, indexing='ij')
        coords = np.stack([ky.ravel(), kz.ravel()])
        
        result = strategy.compute_order(coords, n_segments=4)
        
        assert result.shape == (4, 16)
        assert set(result. ravel()) == set(range(64))
    
    def test_segments_radial_progression(self):
        """Test that segments progress radially outward."""
        strategy = SpiralOrdering()
        ky, kz = np.meshgrid(np.arange(16) - 8, np.arange(16) - 8, indexing='ij')
        coords = np.stack([ky.ravel(), kz.ravel()])
        
        result = strategy.compute_order(coords, n_segments=4)
        
        center = coords.mean(axis=1, keepdims=True)
        
        # Compute mean radius for each segment
        mean_radii = []
        for seg in range(4):
            seg_coords = coords[:, result[seg]]
            radii = np.sqrt(np.sum((seg_coords - center) ** 2, axis=0))
            mean_radii.append(radii.mean())
        
        # Mean radius should generally increase
        assert mean_radii[0] < mean_radii[-1]


class TestSpiralOrderingEdgeCases:
    """Edge case tests."""
    
    def test_single_point(self):
        """Test with single point."""
        strategy = SpiralOrdering()
        coords = np. array([[0], [0]])
        
        result = strategy.compute_order(coords)
        
        assert result.shape == (1, 1)
        assert result[0, 0] == 0
    
    def test_two_points(self):
        """Test with two points."""
        strategy = SpiralOrdering()
        coords = np. array([[0, 1], [0, 1]])
        
        result = strategy. compute_order(coords)
        
        assert result.shape == (1, 2)
        assert set(result[0]) == {0, 1}
    
    def test_all_same_radius(self):
        """Test with points on a circle (same radius)."""
        strategy = SpiralOrdering()
        n_points = 8
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        coords = np.stack([np.cos(angles), np.sin(angles)])
        
        result = strategy.compute_order(coords)
        
        assert result.shape == (1, n_points)
        # Should order by angle since all same radius
        sorted_coords = coords[:, result[0]]
        sorted_angles = np.arctan2(sorted_coords[1], sorted_coords[0])
        sorted_angles = sorted_angles % (2 * np.pi)
        # Angles should be generally increasing
        assert np.sum(np.diff(sorted_angles) > 0) > n_points // 2
    
    def test_center_shape_mismatch_raises(self):
        """Test that 1D center raises ValueError for 2D coords."""
        strategy = SpiralOrdering(center=np.array([0. 0]))
        coords = np.stack([np.arange(4), np.arange(4)])
        
        with pytest.raises(ValueError, match="at least 2 elements"):
            strategy. compute_order(coords)