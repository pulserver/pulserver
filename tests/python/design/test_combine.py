# """Tests for ordering utility functions."""

# import numpy as np
# import pytest

# from pulserver.design import compose_orderings, tile_repeat_order


# class TestTileRepeatOrder:
#     """Tests for tile_repeat_order function."""

#     def test_single_ordering(self):
#         """Test with single ordering."""
#         order = np.array([2, 0, 1])
#         result = tile_repeat_order(order)

#         assert result.shape == (1, 3)
#         np.testing.assert_array_equal(result[0], [2, 0, 1])

#     def test_two_orderings(self):
#         """Test with two orderings."""
#         outer = np.array([1, 0])  # 2 elements
#         inner = np.array([2, 0, 1])  # 3 elements

#         result = tile_repeat_order(outer, inner)

#         assert result.shape == (2, 6)
#         # Outer: each element repeated 3 times
#         np.testing.assert_array_equal(result[0], [1, 1, 1, 0, 0, 0])
#         # Inner: tiled 2 times
#         np.testing.assert_array_equal(result[1], [2, 0, 1, 2, 0, 1])

#     def test_three_orderings(self):
#         """Test with three orderings."""
#         outer = np.array([1, 0])  # 2 elements
#         middle = np.array([2, 0, 1])  # 3 elements
#         inner = np.array([1, 0])  # 2 elements

#         result = tile_repeat_order(outer, middle, inner)

#         assert result.shape == (3, 12)
#         # Outer: repeated 6 times each (3*2)
#         np.testing.assert_array_equal(result[0], [1] * 6 + [0] * 6)
#         # Middle: repeated 2 times each, tiled 2 times
#         np.testing.assert_array_equal(result[1], [2, 2, 0, 0, 1, 1, 2, 2, 0, 0, 1, 1])
#         # Inner: tiled 6 times (2*3)
#         np.testing.assert_array_equal(result[2], [1, 0] * 6)

#     def test_empty_raises(self):
#         """Test that empty input raises."""
#         with pytest.raises(ValueError, match="At least one"):
#             tile_repeat_order()

#     def test_realistic_slc_lin(self):
#         """Test realistic slice/phase encoding case."""
#         # Interleaved 4 slices
#         slc_order = np.array([0, 2, 1, 3])
#         # Center-out 8 phase encodes
#         lin_order = np.array([3, 4, 2, 5, 1, 6, 0, 7])

#         result = tile_repeat_order(slc_order, lin_order)

#         assert result.shape == (2, 32)

#         # First 8 points: slice 0, all lin orderings
#         np.testing.assert_array_equal(result[0][:8], [0] * 8)
#         np.testing.assert_array_equal(result[1][:8], lin_order)

#         # Next 8 points: slice 2, all lin orderings
#         np.testing.assert_array_equal(result[0][8:16], [2] * 8)
#         np.testing.assert_array_equal(result[1][8:16], lin_order)


# class TestComposeOrderings:
#     """Tests for compose_orderings function."""

#     def test_basic(self):
#         """Test basic composition."""
#         orderings = {
#             "slc": np.array([1, 0]),
#             "lin": np.array([2, 0, 1]),
#         }
#         result = compose_orderings(orderings, dim_order=("slc", "lin"))

#         assert "slc" in result
#         assert "lin" in result
#         np.testing.assert_array_equal(result["slc"], [1, 1, 1, 0, 0, 0])
#         np.testing.assert_array_equal(result["lin"], [2, 0, 1, 2, 0, 1])

#     def test_reversed_dim_order(self):
#         """Test that dim_order affects result."""
#         orderings = {
#             "slc": np.array([1, 0]),
#             "lin": np.array([2, 0, 1]),
#         }

#         result1 = compose_orderings(orderings, dim_order=("slc", "lin"))
#         result2 = compose_orderings(orderings, dim_order=("lin", "slc"))

#         # Results should be different
#         assert not np.array_equal(result1["slc"], result2["slc"])

#         # In result2, lin is outer
#         np.testing.assert_array_equal(result2["lin"], [2, 2, 0, 0, 1, 1])
#         np.testing.assert_array_equal(result2["slc"], [1, 0, 1, 0, 1, 0])

#     def test_missing_dimension_raises(self):
#         """Test that missing dimension raises."""
#         orderings = {"slc": np.array([0, 1])}
#         with pytest.raises(ValueError, match="not found"):
#             compose_orderings(orderings, dim_order=("slc", "lin"))

#     def test_three_dimensions(self):
#         """Test with three dimensions."""
#         orderings = {
#             "avg": np.array([1, 0]),
#             "slc": np.array([0, 2, 1, 3]),
#             "lin": np.array([3, 4, 2, 5, 1, 6, 0, 7]),
#         }
#         result = compose_orderings(
#             orderings,
#             dim_order=("avg", "slc", "lin"),
#         )

#         total = 2 * 4 * 8
#         assert len(result["avg"]) == total
#         assert len(result["slc"]) == total
#         assert len(result["lin"]) == total

#         # First half should be avg=1, second half avg=0
#         np.testing.assert_array_equal(result["avg"][: total // 2], [1] * (total // 2))
#         np.testing.assert_array_equal(result["avg"][total // 2 :], [0] * (total // 2))
