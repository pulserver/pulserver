"""Tests for base ordering strategy."""

import numpy as np
import pytest

from pulserver.design import OrderingStrategy

from numpy.typing import NDArray

class DummyStrategy(OrderingStrategy):
    """Concrete implementation for testing."""
    
    @property
    def name(self) -> str:
        return "dummy"
    
    def compute_order(
        self,
        coordinates: NDArray,
        mask: NDArray[bool] | None = None,
        n_segments: int = 1,
    ) -> NDArray[int]:
        coordinates, mask, n_sampled = self._validate_inputs(
            coordinates, mask, n_segments
        )
        # Simple sequential order
        order = np.arange(n_sampled, dtype=np.intp)
        return self._apply_mask_and_reshape(order, mask, n_segments)


class TestValidateInputs:
    """Tests for input validation."""
    
    def test_1d_coordinates(self):
        """Test 1D coordinates are normalized to 2D."""
        strategy = DummyStrategy()
        coords = np.array([0, 1, 2, 3])
        
        result = strategy.compute_order(coords)
        
        assert result.shape == (1, 4)
    
    def test_2d_coordinates(self):
        """Test 2D coordinates are accepted."""
        strategy = DummyStrategy()
        coords = np.array([[0, 1, 2, 3], [0, 0, 1, 1]])
        
        result = strategy.compute_order(coords)
        
        assert result.shape == (1, 4)
    
    def test_mask_filters_points(self):
        """Test that mask filters points correctly."""
        strategy = DummyStrategy()
        coords = np. array([0, 1, 2, 3, 4, 5, 6, 7])
        mask = np.array([True, False, True, False, True, False, True, False])
        
        result = strategy.compute_order(coords, mask=mask)
        
        assert result.shape == (1, 4)
    
    def test_n_segments_divides_output(self):
        """Test that n_segments divides output correctly."""
        strategy = DummyStrategy()
        coords = np.arange(12)
        
        result = strategy.compute_order(coords, n_segments=3)
        
        assert result.shape == (3, 4)
    
    def test_invalid_n_segments_raises(self):
        """Test that invalid n_segments raises ValueError."""
        strategy = DummyStrategy()
        coords = np.arange(10)
        
        with pytest.raises(ValueError, match="must evenly divide"):
            strategy.compute_order(coords, n_segments=3)
    
    def test_n_segments_zero_raises(self):
        """Test that n_segments=0 raises ValueError."""
        strategy = DummyStrategy()
        coords = np. arange(10)
        
        with pytest.raises(ValueError, match="must be >= 1"):
            strategy.compute_order(coords, n_segments=0)
    
    def test_mask_shape_mismatch_raises(self):
        """Test that mask shape mismatch raises ValueError."""
        strategy = DummyStrategy()
        coords = np.arange(10)
        mask = np.ones(5, dtype=bool)
        
        with pytest.raises(ValueError, match="does not match"):
            strategy.compute_order(coords, mask=mask)
    
    def test_3d_coordinates_raises(self):
        """Test that 3D coordinates raise ValueError."""
        strategy = DummyStrategy()
        coords = np. ones((2, 3, 4))
        
        with pytest.raises(ValueError, match="must be 1D or 2D"):
            strategy.compute_order(coords)
    
    def test_empty_mask_raises(self):
        """Test that empty mask raises ValueError."""
        strategy = DummyStrategy()
        coords = np.array([1, 2, 3, 4])
        mask = np.zeros(4, dtype=bool)
        
        with pytest.raises(ValueError, match="No points to order"):
            strategy.compute_order(coords, mask=mask)


class TestOutputFormat:
    """Tests for output format."""
    
    def test_output_dtype(self):
        """Test output has correct dtype."""
        strategy = DummyStrategy()
        coords = np.arange(8)
        
        result = strategy.compute_order(coords, n_segments=2)
        
        assert result.dtype == np.intp
    
    def test_output_shape_single_segment(self):
        """Test output shape with single segment."""
        strategy = DummyStrategy()
        coords = np.arange(8)
        
        result = strategy.compute_order(coords, n_segments=1)
        
        assert result. shape == (1, 8)
    
    def test_output_shape_multiple_segments(self):
        """Test output shape with multiple segments."""
        strategy = DummyStrategy()
        coords = np. arange(24)
        
        result = strategy.compute_order(coords, n_segments=6)
        
        assert result.shape == (6, 4)
    
    def test_output_indices_valid(self):
        """Test output contains valid indices."""
        strategy = DummyStrategy()
        coords = np.arange(12)
        mask = np.array([True, False] * 6)  # 6 sampled points
        
        result = strategy.compute_order(coords, mask=mask, n_segments=2)
        
        # All indices should be in valid range for masked array
        assert np. all(result >= 0)
        assert np.all(result < 6)
    
    def test_output_indices_unique(self):
        """Test output contains unique indices (permutation)."""
        strategy = DummyStrategy()
        coords = np.arange(12)
        
        result = strategy.compute_order(coords, n_segments=3)
        
        # Flattened result should be a permutation
        flat = result.ravel()
        assert len(np.unique(flat)) == 12


class TestRepr:
    """Tests for string representation."""
    
    def test_repr(self):
        """Test __repr__ returns class name."""
        strategy = DummyStrategy()
        
        assert repr(strategy) == "DummyStrategy()"