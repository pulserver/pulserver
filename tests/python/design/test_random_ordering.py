"""Tests for random ordering strategy."""

import numpy as np

from  pulserver.design import RandomOrdering


class TestRandomOrderingBasic:
    """Basic functionality tests."""
    
    def test_name(self):
        """Test strategy name."""
        assert RandomOrdering().name == "random"
    
    def test_seed_property(self):
        """Test seed property."""
        assert RandomOrdering().seed is None
        assert RandomOrdering(seed=42).seed == 42
    
    def test_repr_default(self):
        """Test repr without seed."""
        assert repr(RandomOrdering()) == "RandomOrdering()"
    
    def test_repr_with_seed(self):
        """Test repr with seed."""
        assert repr(RandomOrdering(seed=42)) == "RandomOrdering(seed=42)"


class TestRandomOrderingBehavior:
    """Tests for random ordering behavior."""
    
    def test_output_is_permutation(self):
        """Test that output is a valid permutation."""
        strategy = RandomOrdering(seed=42)
        coords = np.arange(100)
        
        result = strategy.compute_order(coords)
        
        assert result.shape == (1, 100)
        # Should contain all indices exactly once
        assert set(result[0]) == set(range(100))
    
    def test_seed_reproducibility(self):
        """Test that same seed gives same result."""
        coords = np.arange(100)
        
        result1 = RandomOrdering(seed=42).compute_order(coords)
        result2 = RandomOrdering(seed=42).compute_order(coords)
        
        np.testing.assert_array_equal(result1, result2)
    
    def test_different_seeds_differ(self):
        """Test that different seeds give different results."""
        coords = np.arange(100)
        
        result1 = RandomOrdering(seed=42).compute_order(coords)
        result2 = RandomOrdering(seed=123).compute_order(coords)
        
        assert not np.array_equal(result1, result2)
    
    def test_no_seed_varies(self):
        """Test that no seed gives varying results."""
        coords = np.arange(1000)
        
        # Run multiple times, should get different results
        # (extremely unlikely to get same result twice)
        results = []
        for _ in range(3):
            result = RandomOrdering().compute_order(coords)
            results. append(result[0].copy())
        
        # At least two should differ
        all_same = all(np.array_equal(results[0], r) for r in results[1:])
        assert not all_same
    
    def test_order_differs_from_input(self):
        """Test that order is actually shuffled."""
        strategy = RandomOrdering(seed=42)
        coords = np. arange(100)
        
        result = strategy.compute_order(coords)
        
        # Should not be in original order (extremely unlikely with seed=42)
        assert not np.array_equal(result[0], np.arange(100))


class TestRandomOrderingWithMask:
    """Tests with sampling mask."""
    
    def test_masked_permutation(self):
        """Test random ordering with mask."""
        strategy = RandomOrdering(seed=42)
        coords = np.arange(100)
        mask = np.array([True, False] * 50)  # 50 points
        
        result = strategy.compute_order(coords, mask=mask)
        
        assert result.shape == (1, 50)
        # Should be permutation of 0-49
        assert set(result[0]) == set(range(50))
    
    def test_masked_reproducibility(self):
        """Test reproducibility with mask."""
        coords = np.arange(100)
        mask = np.array([True, False] * 50)
        
        result1 = RandomOrdering(seed=42).compute_order(coords, mask=mask)
        result2 = RandomOrdering(seed=42).compute_order(coords, mask=mask)
        
        np.testing.assert_array_equal(result1, result2)


class TestRandomOrderingWithSegments:
    """Tests with segmentation."""
    
    def test_segments(self):
        """Test random ordering with segments."""
        strategy = RandomOrdering(seed=42)
        coords = np.arange(100)
        
        result = strategy.compute_order(coords, n_segments=10)
        
        assert result.shape == (10, 10)
        # Flattened should be permutation
        assert set(result.ravel()) == set(range(100))
    
    def test_segments_reproducibility(self):
        """Test segment reproducibility."""
        coords = np.arange(100)
        
        result1 = RandomOrdering(seed=42).compute_order(coords, n_segments=5)
        result2 = RandomOrdering(seed=42).compute_order(coords, n_segments=5)
        
        np.testing.assert_array_equal(result1, result2)


class TestRandomOrdering2D:
    """Tests with 2D coordinates."""
    
    def test_2d_permutation(self):
        """Test random ordering with 2D coordinates."""
        strategy = RandomOrdering(seed=42)
        ky, kz = np.meshgrid(np.arange(8), np.arange(8), indexing='ij')
        coords = np.stack([ky.ravel(), kz. ravel()])
        
        result = strategy.compute_order(coords)
        
        assert result. shape == (1, 64)
        assert set(result[0]) == set(range(64))
    
    def test_2d_with_segments(self):
        """Test 2D random ordering with segments."""
        strategy = RandomOrdering(seed=42)
        ky, kz = np.meshgrid(np. arange(8), np.arange(8), indexing='ij')
        coords = np.stack([ky.ravel(), kz. ravel()])
        
        result = strategy.compute_order(coords, n_segments=4)
        
        assert result.shape == (4, 16)
        assert set(result. ravel()) == set(range(64))


class TestRandomOrderingEdgeCases:
    """Edge case tests."""
    
    def test_single_point(self):
        """Test with single point."""
        strategy = RandomOrdering(seed=42)
        coords = np.array([5])
        
        result = strategy.compute_order(coords)
        
        assert result.shape == (1, 1)
        assert result[0, 0] == 0
    
    def test_two_points(self):
        """Test with two points."""
        strategy = RandomOrdering(seed=42)
        coords = np.array([0, 1])
        
        result = strategy.compute_order(coords)
        
        assert result.shape == (1, 2)
        assert set(result[0]) == {0, 1}
    
    def test_large_array(self):
        """Test with large array."""
        strategy = RandomOrdering(seed=42)
        coords = np.arange(10000)
        
        result = strategy.compute_order(coords)
        
        assert result.shape == (1, 10000)
        assert set(result[0]) == set(range(10000))