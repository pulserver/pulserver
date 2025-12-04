"""Tests for interleaved ordering strategy."""

import numpy as np
import pytest

from pulserver.design import InterleavedOrdering


class TestInterleavedOrderingBasic:
    """Basic functionality tests."""
    
    def test_name(self):
        """Test strategy name."""
        assert InterleavedOrdering().name == "interleaved_2"
        assert InterleavedOrdering(n_interleaves=3).name == "interleaved_3"
        assert InterleavedOrdering(order_within="descending").name == "interleaved_2_desc"
    
    def test_properties(self):
        """Test property accessors."""
        strategy = InterleavedOrdering(n_interleaves=4, order_within="descending")
        
        assert strategy.n_interleaves == 4
        assert strategy.order_within == "descending"
    
    def test_repr(self):
        """Test string representation."""
        assert repr(InterleavedOrdering()) == "InterleavedOrdering(n_interleaves=2)"
        
        strategy = InterleavedOrdering(n_interleaves=3, order_within="descending")
        repr_str = repr(strategy)
        assert "n_interleaves=3" in repr_str
        assert "order_within='descending'" in repr_str
    
    def test_invalid_n_interleaves_raises(self):
        """Test that n_interleaves < 1 raises ValueError."""
        with pytest.raises(ValueError, match="must be >= 1"):
            InterleavedOrdering(n_interleaves=0)
    
    def test_invalid_order_within_raises(self):
        """Test that invalid order_within raises ValueError."""
        with pytest.raises(ValueError, match="must be 'ascending' or 'descending'"):
            InterleavedOrdering(order_within="invalid")


class TestInterleavedOrderingTwoWay:
    """Tests for two-way (even/odd) interleaving."""
    
    def test_even_odd_ascending(self):
        """Test even/odd interleaving with ascending order."""
        strategy = InterleavedOrdering(n_interleaves=2)
        coords = np.arange(8)
        
        result = strategy.compute_order(coords)
        
        assert result.shape == (1, 8)
        # Should be: 0, 2, 4, 6, 1, 3, 5, 7
        sorted_coords = coords[result[0]]
        np.testing.assert_array_equal(sorted_coords, [0, 2, 4, 6, 1, 3, 5, 7])
    
    def test_even_odd_descending(self):
        """Test even/odd interleaving with descending order."""
        strategy = InterleavedOrdering(n_interleaves=2, order_within="descending")
        coords = np.arange(8)
        
        result = strategy.compute_order(coords)
        
        # Should be: 6, 4, 2, 0, 7, 5, 3, 1
        sorted_coords = coords[result[0]]
        np. testing.assert_array_equal(sorted_coords, [6, 4, 2, 0, 7, 5, 3, 1])
    
    def test_odd_count(self):
        """Test with odd number of slices."""
        strategy = InterleavedOrdering(n_interleaves=2)
        coords = np.arange(7)
        
        result = strategy.compute_order(coords)
        
        # Should be: 0, 2, 4, 6, 1, 3, 5
        sorted_coords = coords[result[0]]
        np.testing.assert_array_equal(sorted_coords, [0, 2, 4, 6, 1, 3, 5])


class TestInterleavedOrderingThreeWay:
    """Tests for three-way interleaving."""
    
    def test_three_way_ascending(self):
        """Test three-way interleaving."""
        strategy = InterleavedOrdering(n_interleaves=3)
        coords = np.arange(9)
        
        result = strategy.compute_order(coords)
        
        # Should be: 0, 3, 6, 1, 4, 7, 2, 5, 8
        sorted_coords = coords[result[0]]
        np.testing.assert_array_equal(sorted_coords, [0, 3, 6, 1, 4, 7, 2, 5, 8])
    
    def test_three_way_uneven(self):
        """Test three-way interleaving with uneven groups."""
        strategy = InterleavedOrdering(n_interleaves=3)
        coords = np.arange(7)
        
        result = strategy.compute_order(coords)
        
        # Group 0: 0, 3, 6
        # Group 1: 1, 4
        # Group 2: 2, 5
        # Should be: 0, 3, 6, 1, 4, 2, 5
        sorted_coords = coords[result[0]]
        np.testing.assert_array_equal(sorted_coords, [0, 3, 6, 1, 4, 2, 5])


class TestInterleavedOrderingWithMask:
    """Tests with sampling mask."""
    
    def test_masked_even_odd(self):
        """Test even/odd interleaving with mask."""
        strategy = InterleavedOrdering(n_interleaves=2)
        coords = np.arange(8)
        # Only acquire slices 1, 2, 4, 5
        mask = np.array([False, True, True, False, True, True, False, False])
        
        result = strategy.compute_order(coords, mask=mask)
        
        assert result.shape == (1, 4)
        # Masked coords are [1, 2, 4, 5]
        # Even (group 0): 2, 4 -> indices 1, 2 in masked array
        # Odd (group 1): 1, 5 -> indices 0, 3 in masked array
        # Order should give coords: 2, 4, 1, 5
        masked_coords = coords[mask]
        sorted_coords = masked_coords[result[0]]
        np.testing.assert_array_equal(sorted_coords, [2, 4, 1, 5])


class TestInterleavedOrderingWithSegments:
    """Tests with segmentation."""
    
    def test_segments(self):
        """Test interleaved ordering with segments."""
        strategy = InterleavedOrdering(n_interleaves=2)
        coords = np.arange(8)
        
        result = strategy.compute_order(coords, n_segments=2)
        
        assert result.shape == (2, 4)
        # Full order: 0, 2, 4, 6, 1, 3, 5, 7
        # Segment 0: 0, 2, 4, 6
        # Segment 1: 1, 3, 5, 7
        sorted_coords_0 = coords[result[0]]
        sorted_coords_1 = coords[result[1]]
        np.testing.assert_array_equal(sorted_coords_0, [0, 2, 4, 6])
        np.testing.assert_array_equal(sorted_coords_1, [1, 3, 5, 7])
    
    def test_segments_match_interleaves(self):
        """Test with n_segments matching n_interleaves."""
        strategy = InterleavedOrdering(n_interleaves=4)
        coords = np.arange(12)
        
        result = strategy.compute_order(coords, n_segments=4)
        
        assert result.shape == (4, 3)
        # Each segment should contain one interleave group
        for seg in range(4):
            seg_coords = coords[result[seg]]
            # All coords in segment should have same modulo
            assert len(np.unique(seg_coords % 4)) == 1


class TestInterleavedOrderingEdgeCases:
    """Edge case tests."""
    
    def test_single_interleave(self):
        """Test with n_interleaves=1 (no interleaving)."""
        strategy = InterleavedOrdering(n_interleaves=1)
        coords = np.array([3, 1, 4, 1, 5])
        
        result = strategy.compute_order(coords)
        
        # Should just be ascending order
        sorted_coords = coords[result[0]]
        np.testing.assert_array_equal(sorted_coords, np.sort(coords))
    
    def test_more_interleaves_than_points(self):
        """Test with more interleaves than points."""
        strategy = InterleavedOrdering(n_interleaves=10)
        coords = np.arange(4)
        
        result = strategy.compute_order(coords)
        
        # Each point in its own group, should be ascending
        sorted_coords = coords[result[0]]
        np.testing.assert_array_equal(sorted_coords, [0, 1, 2, 3])
    
    def test_single_point(self):
        """Test with single point."""
        strategy = InterleavedOrdering()
        coords = np.array([5])
        
        result = strategy.compute_order(coords)
        
        assert result.shape == (1, 1)
        assert result[0, 0] == 0
    
    def test_non_sequential_coords(self):
        """Test with non-sequential coordinates."""
        strategy = InterleavedOrdering(n_interleaves=2)
        # Use odd and even values: 11, 22, 33, 44, 55, 66
        coords = np.array([11, 22, 33, 44, 55, 66])
        
        result = strategy.compute_order(coords)
        
        # Even (group 0): 22, 44, 66 -> indices 1, 3, 5
        # Odd (group 1): 11, 33, 55 -> indices 0, 2, 4
        sorted_coords = coords[result[0]]
        np. testing.assert_array_equal(sorted_coords, [22, 44, 66, 11, 33, 55])
    
    def test_2d_coordinates_raises(self):
        """Test that 2D coordinates raise ValueError."""
        strategy = InterleavedOrdering()
        coords = np.stack([np.arange(4), np. arange(4)])
        
        with pytest.raises(ValueError, match="requires 1D coordinates"):
            strategy.compute_order(coords)
    
    def test_negative_coordinates(self):
        """Test with negative coordinates."""
        strategy = InterleavedOrdering(n_interleaves=2)
        coords = np.array([-4, -3, -2, -1, 0, 1, 2, 3])
        
        result = strategy.compute_order(coords)
        
        # Groups based on modulo (Python handles negative modulo correctly)
        sorted_coords = coords[result[0]]
        # Group 0 (even): -4, -2, 0, 2
        # Group 1 (odd): -3, -1, 1, 3
        np.testing.assert_array_equal(sorted_coords, [-4, -2, 0, 2, -3, -1, 1, 3])