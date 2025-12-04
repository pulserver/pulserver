"""Tests for linear ordering strategy."""

import numpy as np
import pytest

from pulserver.design import LinearOrdering


class TestLinearOrderingBasic:
    """Basic functionality tests."""
    
    def test_name(self):
        """Test strategy name."""
        assert LinearOrdering().name == "linear"
        assert LinearOrdering(reverse=True).name == "linear_reverse"
    
    def test_properties(self):
        """Test property accessors."""
        strategy = LinearOrdering(reverse=True, axis_priority=(1, 0))
        
        assert strategy.reverse is True
        assert strategy.axis_priority == (1, 0)
    
    def test_repr_default(self):
        """Test repr with defaults."""
        assert repr(LinearOrdering()) == "LinearOrdering()"
    
    def test_repr_with_params(self):
        """Test repr with parameters."""
        strategy = LinearOrdering(reverse=True, axis_priority=(1, 0))
        repr_str = repr(strategy)
        
        assert "reverse=True" in repr_str
        assert "axis_priority=(1, 0)" in repr_str


class TestLinearOrdering1D:
    """Tests for 1D linear ordering."""
    
    def test_ascending_order(self):
        """Test ascending order (default)."""
        strategy = LinearOrdering()
        coords = np.array([3, 1, 4, 1, 5, 9, 2, 6])
        
        result = strategy.compute_order(coords)
        
        assert result. shape == (1, 8)
        # Check ordering is correct
        sorted_coords = coords[result[0]]
        np.testing.assert_array_equal(sorted_coords, np.sort(coords))
    
    def test_descending_order(self):
        """Test descending order."""
        strategy = LinearOrdering(reverse=True)
        coords = np.array([3, 1, 4, 1, 5, 9, 2, 6])
        
        result = strategy.compute_order(coords)
        
        sorted_coords = coords[result[0]]
        np.testing. assert_array_equal(sorted_coords, np.sort(coords)[::-1])
    
    def test_already_sorted(self):
        """Test already sorted input."""
        strategy = LinearOrdering()
        coords = np. arange(8)
        
        result = strategy.compute_order(coords)
        
        np.testing.assert_array_equal(result[0], np.arange(8))
    
    def test_with_segments(self):
        """Test segmented output."""
        strategy = LinearOrdering()
        coords = np.array([7, 5, 3, 1, 6, 4, 2, 0])
        
        result = strategy.compute_order(coords, n_segments=4)
        
        assert result.shape == (4, 2)
        # Full ordering should still be sorted
        full_order = result.ravel()
        sorted_coords = coords[full_order]
        np.testing.assert_array_equal(sorted_coords, np. arange(8))
    
    def test_with_mask(self):
        """Test with sampling mask."""
        strategy = LinearOrdering()
        coords = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        mask = np.array([True, False, True, False, True, False, True, False])
        
        result = strategy.compute_order(coords, mask=mask)
        
        assert result.shape == (1, 4)
        # Indices should be into masked array (0-3, not 0-7)
        assert np.all(result >= 0)
        assert np.all(result < 4)


class TestLinearOrdering2D:
    """Tests for 2D linear ordering."""
    
    def test_default_priority(self):
        """Test default axis priority (axis 0 primary)."""
        strategy = LinearOrdering()
        # 3x3 grid
        ky = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        kz = np. array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        coords = np.stack([ky, kz])
        
        result = strategy.compute_order(coords)
        
        # Should order by ky first, then kz
        sorted_ky = ky[result[0]]
        sorted_kz = kz[result[0]]
        
        # ky should be non-decreasing
        assert np.all(np.diff(sorted_ky) >= 0)
        # Within same ky, kz should be non-decreasing
        for k in range(3):
            kz_at_ky = sorted_kz[sorted_ky == k]
            assert np.all(np.diff(kz_at_ky) >= 0)
    
    def test_reversed_priority(self):
        """Test reversed axis priority (axis 1 primary)."""
        strategy = LinearOrdering(axis_priority=(1, 0))
        # 3x3 grid
        ky = np. array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        kz = np. array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        coords = np.stack([ky, kz])
        
        result = strategy.compute_order(coords)
        
        # Should order by kz first, then ky
        sorted_ky = ky[result[0]]
        sorted_kz = kz[result[0]]
        
        # kz should be non-decreasing
        assert np.all(np. diff(sorted_kz) >= 0)
        # Within same kz, ky should be non-decreasing
        for k in range(3):
            ky_at_kz = sorted_ky[sorted_kz == k]
            assert np.all(np.diff(ky_at_kz) >= 0)
    
    def test_2d_reverse(self):
        """Test reverse with 2D coordinates."""
        strategy = LinearOrdering(reverse=True)
        ky = np.array([0, 0, 1, 1])
        kz = np.array([0, 1, 0, 1])
        coords = np. stack([ky, kz])
        
        result = strategy.compute_order(coords)
        
        sorted_ky = ky[result[0]]
        sorted_kz = kz[result[0]]
        
        # Should be reverse lexicographic order
        np.testing.assert_array_equal(sorted_ky, [1, 1, 0, 0])
        np. testing.assert_array_equal(sorted_kz, [1, 0, 1, 0])
    
    def test_invalid_axis_priority_raises(self):
        """Test that invalid axis_priority raises ValueError."""
        strategy = LinearOrdering(axis_priority=(0, 1, 2))  # 3 axes for 2D data
        coords = np.stack([np.arange(4), np.arange(4)])
        
        with pytest. raises(ValueError, match="invalid for 2 dimensions"):
            strategy.compute_order(coords)


class TestLinearOrdering3D:
    """Tests for 3D linear ordering."""
    
    def test_3d_default_priority(self):
        """Test 3D with default priority."""
        strategy = LinearOrdering()
        # 2x2x2 grid
        k0, k1, k2 = np.meshgrid(
            np.arange(2), np.arange(2), np.arange(2), indexing='ij'
        )
        coords = np. stack([k0. ravel(), k1.ravel(), k2.ravel()])
        
        result = strategy.compute_order(coords)
        
        assert result.shape == (1, 8)
        # First point should be (0,0,0), last should be (1,1,1)
        assert np.all(coords[:, result[0, 0]] == [0, 0, 0])
        assert np.all(coords[:, result[0, -1]] == [1, 1, 1])
    
    def test_3d_custom_priority(self):
        """Test 3D with custom priority."""
        strategy = LinearOrdering(axis_priority=(2, 0, 1))
        k0, k1, k2 = np.meshgrid(
            np.arange(2), np.arange(2), np.arange(2), indexing='ij'
        )
        coords = np.stack([k0.ravel(), k1.ravel(), k2. ravel()])
        
        result = strategy.compute_order(coords)
        
        # Primary sort by axis 2, then axis 0, then axis 1
        sorted_coords = coords[:, result[0]]
        
        # Axis 2 should be non-decreasing overall
        assert np. all(np.diff(sorted_coords[2]) >= 0)


class TestLinearOrderingEdgeCases:
    """Edge case tests."""
    
    def test_single_point(self):
        """Test with single point."""
        strategy = LinearOrdering()
        coords = np.array([5])
        
        result = strategy.compute_order(coords)
        
        assert result.shape == (1, 1)
        assert result[0, 0] == 0
    
    def test_all_same_values(self):
        """Test with all identical values."""
        strategy = LinearOrdering()
        coords = np.array([3, 3, 3, 3])
        
        result = strategy.compute_order(coords)
        
        # Order is stable but arbitrary for ties
        assert result. shape == (1, 4)
        assert set(result[0]) == {0, 1, 2, 3}
    
    def test_negative_values(self):
        """Test with negative values."""
        strategy = LinearOrdering()
        coords = np. array([-3, -1, -4, -1, -5, -9, -2, -6])
        
        result = strategy.compute_order(coords)
        
        sorted_coords = coords[result[0]]
        np.testing. assert_array_equal(sorted_coords, np.sort(coords))
    
    def test_float_coordinates(self):
        """Test with floating point coordinates."""
        strategy = LinearOrdering()
        coords = np.array([0.1, 0.3, 0.2, 0.15])
        
        result = strategy.compute_order(coords)
        
        sorted_coords = coords[result[0]]
        np.testing.assert_array_almost_equal(sorted_coords, np.sort(coords))
    
    def test_empty_after_mask(self):
        """Test with mask that excludes all points."""
        strategy = LinearOrdering()
        coords = np.array([1, 2, 3, 4])
        mask = np.zeros(4, dtype=bool)
        
        # Should raise because 0 points can't be divided into 1 segment
        with pytest.raises(ValueError):
            strategy.compute_order(coords, mask=mask)