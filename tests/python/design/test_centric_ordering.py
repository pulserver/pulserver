"""Tests for center-out ordering strategy."""

import numpy as np
import pytest

from pulserver.design import CenterOutOrdering


class TestCenterOutOrderingBasic:
    """Basic functionality tests."""
    
    def test_name(self):
        """Test strategy name."""
        assert CenterOutOrdering().name == "center_out"
    
    def test_properties_default(self):
        """Test default property values."""
        strategy = CenterOutOrdering()
        
        assert strategy.center is None
        assert strategy.angular_offset == 0.0
    
    def test_properties_custom(self):
        """Test custom property values."""
        center = np.array([1.0, 2.0])
        strategy = CenterOutOrdering(center=center, angular_offset=0.5)
        
        np.testing.assert_array_equal(strategy.center, center)
        assert strategy.angular_offset == 0.5
    
    def test_repr_default(self):
        """Test repr with defaults."""
        assert repr(CenterOutOrdering()) == "CenterOutOrdering()"
    
    def test_repr_with_params(self):
        """Test repr with parameters."""
        strategy = CenterOutOrdering(center=np. array([1.0, 2.0]), angular_offset=0.5)
        repr_str = repr(strategy)
        
        assert "center=" in repr_str
        assert "angular_offset=0.5" in repr_str


class TestCenterOutOrdering1D:
    """Tests for 1D center-out ordering."""
    
    def test_symmetric_coordinates(self):
        """Test with symmetric coordinates around zero."""
        strategy = CenterOutOrdering()
        coords = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])
        
        result = strategy.compute_order(coords)
        
        assert result.shape == (1, 9)
        # Center (0) should come first
        sorted_coords = coords[result[0]]
        assert sorted_coords[0] == 0
        # Then ±1, ±2, etc.  (order within same radius is arbitrary)
        radii = np.abs(sorted_coords)
        assert np.all(np.diff(radii) >= 0)
    
    def test_asymmetric_coordinates(self):
        """Test with asymmetric coordinates."""
        strategy = CenterOutOrdering()
        coords = np. array([0, 1, 2, 3, 4, 5, 6, 7])
        
        result = strategy.compute_order(coords)
        
        # Center is at mean = 3. 5
        # Closest points are 3 and 4
        sorted_coords = coords[result[0]]
        assert sorted_coords[0] in [3, 4]
        assert sorted_coords[1] in [3, 4]
    
    def test_custom_center(self):
        """Test with custom center."""
        strategy = CenterOutOrdering(center=np.array([2.0]))
        coords = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        
        result = strategy.compute_order(coords)
        
        # Center is at 2
        sorted_coords = coords[result[0]]
        assert sorted_coords[0] == 2
        # Then 1 and 3 (distance 1)
        assert set(sorted_coords[1:3]) == {1, 3}
    
    def test_single_point(self):
        """Test with single point."""
        strategy = CenterOutOrdering()
        coords = np.array([5])
        
        result = strategy.compute_order(coords)
        
        assert result.shape == (1, 1)
        assert result[0, 0] == 0


class TestCenterOutOrdering2D:
    """Tests for 2D center-out ordering."""
    
    def test_2d_symmetric(self):
        """Test 2D center-out with symmetric coordinates."""
        strategy = CenterOutOrdering()
        # 5x5 grid centered at origin
        ky, kz = np.meshgrid(np.arange(5) - 2, np.arange(5) - 2, indexing='ij')
        coords = np.stack([ky.ravel(), kz.ravel()])
        
        result = strategy.compute_order(coords)
        
        assert result.shape == (1, 25)
        # Center (0, 0) should come first
        sorted_coords = coords[:, result[0]]
        assert sorted_coords[0, 0] == 0
        assert sorted_coords[1, 0] == 0
        
        # Radii should be non-decreasing
        radii = np.sqrt(sorted_coords[0] ** 2 + sorted_coords[1] ** 2)
        # Allow small tolerance for floating point
        assert np.all(np.diff(radii) >= -1e-10)
    
    def test_2d_asymmetric(self):
        """Test 2D center-out with asymmetric grid."""
        strategy = CenterOutOrdering()
        # 4x6 grid
        ky, kz = np.meshgrid(np. arange(4), np.arange(6), indexing='ij')
        coords = np.stack([ky.ravel(), kz. ravel()])
        
        result = strategy.compute_order(coords)
        
        # Center is at (1. 5, 2.5)
        sorted_coords = coords[:, result[0]]
        radii = np.sqrt((sorted_coords[0] - 1.5) ** 2 + (sorted_coords[1] - 2.5) ** 2)
        assert np.all(np.diff(radii) >= -1e-10)
    
    def test_2d_custom_center(self):
        """Test 2D with custom center."""
        strategy = CenterOutOrdering(center=np.array([0.0, 0.0]))
        ky, kz = np.meshgrid(np.arange(4), np. arange(4), indexing='ij')
        coords = np.stack([ky.ravel(), kz.ravel()])
        
        result = strategy.compute_order(coords)
        
        # Center at (0, 0), so point (0, 0) should be first
        sorted_coords = coords[:, result[0]]
        assert sorted_coords[0, 0] == 0
        assert sorted_coords[1, 0] == 0
    
    def test_center_shape_mismatch_raises(self):
        """Test that mismatched center shape raises ValueError."""
        strategy = CenterOutOrdering(center=np.array([0.0, 0.0, 0.0]))
        coords = np.stack([np.arange(4), np. arange(4)])
        
        with pytest.raises(ValueError, match="center shape"):
            strategy.compute_order(coords)


class TestCenterOutOrderingAngularOffset:
    """Tests for angular offset functionality."""
    
    def test_angular_offset_changes_order(self):
        """Test that angular offset changes the ordering."""
        coords = self._make_2d_grid()
        
        order1 = CenterOutOrdering(angular_offset=0.0). compute_order(coords)
        order2 = CenterOutOrdering(angular_offset=np.pi / 4).compute_order(coords)
        
        # Orders should be different
        assert not np.array_equal(order1, order2)
    
    def test_angular_offset_preserves_radial_structure(self):
        """Test that angular offset preserves radial ordering."""
        coords = self._make_2d_grid()
        
        strategy = CenterOutOrdering(angular_offset=np.pi / 3)
        result = strategy.compute_order(coords)
        
        # Radii should still be non-decreasing
        center = coords. mean(axis=1, keepdims=True)
        sorted_coords = coords[:, result[0]]
        radii = np.sqrt(np.sum((sorted_coords - center) ** 2, axis=0))
        assert np.all(np. diff(radii) >= -1e-10)
    
    def test_per_slice_offset_creates_variation(self):
        """Test creating per-slice orderings with different offsets."""
        coords = self._make_2d_grid()
        golden_angle = 0.618 * 2 * np. pi
        
        orders = []
        for slc in range(4):
            offset = slc * golden_angle
            strategy = CenterOutOrdering(angular_offset=offset)
            orders.append(strategy.compute_order(coords))
        
        # All orders should be different
        for i in range(4):
            for j in range(i + 1, 4):
                assert not np.array_equal(orders[i], orders[j])
    
    def _make_2d_grid(self):
        """Helper to create a 2D grid."""
        ky, kz = np.meshgrid(np.arange(8) - 4, np.arange(8) - 4, indexing='ij')
        return np.stack([ky.ravel(), kz.ravel()])


class TestCenterOutOrdering3D:
    """Tests for 3D center-out ordering."""
    
    def test_3d_ordering(self):
        """Test 3D center-out ordering."""
        strategy = CenterOutOrdering()
        # 4x4x4 grid
        k0, k1, k2 = np. meshgrid(
            np.arange(4) - 1.5,
            np.arange(4) - 1.5,
            np.arange(4) - 1.5,
            indexing='ij'
        )
        coords = np.stack([k0.ravel(), k1.ravel(), k2.ravel()])
        
        result = strategy.compute_order(coords)
        
        assert result.shape == (1, 64)
        # Radii should be non-decreasing
        sorted_coords = coords[:, result[0]]
        radii = np.sqrt(np.sum(sorted_coords ** 2, axis=0))
        assert np.all(np. diff(radii) >= -1e-10)


class TestCenterOutOrderingWithMask:
    """Tests with sampling mask."""
    
    def test_masked_2d(self):
        """Test center-out with mask."""
        strategy = CenterOutOrdering()
        ky, kz = np.meshgrid(np.arange(8) - 4, np.arange(8) - 4, indexing='ij')
        coords = np.stack([ky.ravel(), kz.ravel()])
        
        # Random mask
        rng = np.random.default_rng(42)
        mask = rng.random(64) > 0.5
        n_sampled = mask.sum()
        
        result = strategy.compute_order(coords, mask=mask)
        
        assert result.shape == (1, n_sampled)
        # Indices should be valid for masked array
        assert np. all(result >= 0)
        assert np.all(result < n_sampled)


class TestCenterOutOrderingWithSegments:
    """Tests with segmentation."""
    
    def test_segments(self):
        """Test center-out with segments."""
        strategy = CenterOutOrdering()
        ky, kz = np.meshgrid(np.arange(8) - 4, np.arange(8) - 4, indexing='ij')
        coords = np. stack([ky. ravel(), kz.ravel()])
        
        result = strategy.compute_order(coords, n_segments=4)
        
        assert result.shape == (4, 16)
        # First segment should contain innermost points
        # Last segment should contain outermost points
        center = coords.mean(axis=1, keepdims=True)
        
        seg0_coords = coords[:, result[0]]
        seg3_coords = coords[:, result[3]]
        
        seg0_radii = np.sqrt(np.sum((seg0_coords - center) ** 2, axis=0))
        seg3_radii = np.sqrt(np.sum((seg3_coords - center) ** 2, axis=0))
        
        assert seg0_radii. mean() < seg3_radii. mean()


class TestCenterOutOrderingEdgeCases:
    """Edge case tests."""
    
    def test_all_same_coordinates(self):
        """Test with all identical coordinates."""
        strategy = CenterOutOrdering()
        coords = np.array([5, 5, 5, 5])
        
        result = strategy.compute_order(coords)
        
        # All points at same location, order is arbitrary but valid
        assert result.shape == (1, 4)
        assert set(result[0]) == {0, 1, 2, 3}
    
    def test_negative_coordinates(self):
        """Test with negative coordinates."""
        strategy = CenterOutOrdering()
        coords = np.array([-10, -5, -1, 0, 1, 5, 10])
        
        result = strategy.compute_order(coords)
        
        # Center is at 0, so 0 should be first
        sorted_coords = coords[result[0]]
        assert sorted_coords[0] == 0
    
    def test_float_coordinates(self):
        """Test with floating point coordinates."""
        strategy = CenterOutOrdering()
        coords = np.array([-1.5, -0.5, 0.0, 0.5, 1.5])
        
        result = strategy.compute_order(coords)
        
        sorted_coords = coords[result[0]]
        # 0.0 is at center, should be first
        assert sorted_coords[0] == 0.0
        # Radii should be non-decreasing
        radii = np. abs(sorted_coords)
        assert np. all(np.diff(radii) >= -1e-10)