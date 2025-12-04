"""Tests for interleaved ordering strategy."""

import numpy as np
import pytest

from pulserver.design import TrajectoryData, TrajectoryOrderer, InterleavedOrdering


@pytest.fixture
def slices_8() -> TrajectoryData:
    """Create 8-slice data."""
    n = 8
    return TrajectoryData(
        scaling={"slc": np.linspace(0, 1, n)},
        indices={"slc": np.arange(n)},
        mask=np.ones(n, dtype=bool),
        dim_labels=("slc",),
    )


@pytest.fixture
def slices_12() -> TrajectoryData:
    """Create 12-slice data."""
    n = 12
    return TrajectoryData(
        scaling={"slc": np.linspace(0, 1, n)},
        indices={"slc": np.arange(n)},
        mask=np.ones(n, dtype=bool),
        dim_labels=("slc",),
    )


@pytest.fixture
def slices_10() -> TrajectoryData:
    """Create 10-slice data (for odd interleave counts)."""
    n = 10
    return TrajectoryData(
        scaling={"slc": np.linspace(0, 1, n)},
        indices={"slc": np.arange(n)},
        mask=np.ones(n, dtype=bool),
        dim_labels=("slc",),
    )


class TestInterleavedOrderingBasic:
    """Basic functionality tests for InterleavedOrdering."""

    def test_name(self):
        """Test strategy name."""
        strategy = InterleavedOrdering(n_interleaves=2)
        assert strategy.name == "interleaved_2"

    def test_name_with_n(self):
        """Test name includes interleave count."""
        strategy = InterleavedOrdering(n_interleaves=4)
        assert strategy.name == "interleaved_4"

    def test_n_interleaves_property(self):
        """Test n_interleaves property."""
        strategy = InterleavedOrdering(n_interleaves=3)
        assert strategy.n_interleaves == 3

    def test_order_within_property(self):
        """Test order_within property."""
        strategy = InterleavedOrdering(order_within="descending")
        assert strategy.order_within == "descending"

    def test_invalid_n_interleaves_raises(self):
        """Test that n_interleaves < 1 raises."""
        with pytest.raises(ValueError, match="n_interleaves must be >= 1"):
            InterleavedOrdering(n_interleaves=0)

    def test_invalid_order_within_raises(self):
        """Test that invalid order_within raises."""
        with pytest.raises(ValueError, match="order_within must be one of"):
            InterleavedOrdering(order_within="invalid")

    def test_repr(self):
        """Test string representation."""
        strategy = InterleavedOrdering(n_interleaves=2, order_within="descending")
        repr_str = repr(strategy)
        assert "InterleavedOrdering" in repr_str
        assert "n_interleaves=2" in repr_str
        assert "descending" in repr_str


class TestEvenOddInterleaving:
    """Tests for standard even/odd interleaving."""

    def test_even_odd_ascending(self, slices_8: TrajectoryData):
        """Test even/odd with ascending order within."""
        orderer = TrajectoryOrderer(InterleavedOrdering(n_interleaves=2))
        result = orderer.order(slices_8)

        # Should be: 0, 2, 4, 6 (even), then 1, 3, 5, 7 (odd)
        expected = [0, 2, 4, 6, 1, 3, 5, 7]
        np.testing.assert_array_equal(result.indices["slc"], expected)

    def test_even_odd_descending(self, slices_8: TrajectoryData):
        """Test even/odd with descending order within."""
        orderer = TrajectoryOrderer(
            InterleavedOrdering(n_interleaves=2, order_within="descending")
        )
        result = orderer.order(slices_8)

        # Should be: 6, 4, 2, 0 (even descending), then 7, 5, 3, 1 (odd descending)
        expected = [6, 4, 2, 0, 7, 5, 3, 1]
        np.testing.assert_array_equal(result.indices["slc"], expected)

    def test_odd_first(self, slices_8: TrajectoryData):
        """Test odd slices first using custom interleave order."""
        orderer = TrajectoryOrderer(
            InterleavedOrdering(n_interleaves=2, interleave_order=[1, 0])
        )
        result = orderer.order(slices_8)

        # Should be: 1, 3, 5, 7 (odd), then 0, 2, 4, 6 (even)
        expected = [1, 3, 5, 7, 0, 2, 4, 6]
        np.testing.assert_array_equal(result.indices["slc"], expected)

    def test_even_odd_center_out(self, slices_8: TrajectoryData):
        """Test even/odd with center-out order within."""
        orderer = TrajectoryOrderer(
            InterleavedOrdering(n_interleaves=2, order_within="center_out")
        )
        result = orderer.order(slices_8)

        # Even: 0, 2, 4, 6 -> center is 3, so order by distance: 2 or 4 first
        # Odd: 1, 3, 5, 7 -> center is 4, so order by distance: 3 or 5 first
        # Just verify structure is correct
        even_part = result.indices["slc"][:4]
        odd_part = result.indices["slc"][4:]

        assert set(even_part) == {0, 2, 4, 6}
        assert set(odd_part) == {1, 3, 5, 7}


class TestNShotInterleaving:
    """Tests for N-shot interleaving."""

    def test_3shot_interleaving(self, slices_12: TrajectoryData):
        """Test 3-shot interleaving."""
        orderer = TrajectoryOrderer(InterleavedOrdering(n_interleaves=3))
        result = orderer.order(slices_12)

        # Group 0: 0, 3, 6, 9
        # Group 1: 1, 4, 7, 10
        # Group 2: 2, 5, 8, 11
        expected = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
        np.testing.assert_array_equal(result.indices["slc"], expected)

    def test_4shot_interleaving(self, slices_8: TrajectoryData):
        """Test 4-shot interleaving."""
        orderer = TrajectoryOrderer(InterleavedOrdering(n_interleaves=4))
        result = orderer.order(slices_8)

        # Group 0: 0, 4
        # Group 1: 1, 5
        # Group 2: 2, 6
        # Group 3: 3, 7
        expected = [0, 4, 1, 5, 2, 6, 3, 7]
        np.testing.assert_array_equal(result.indices["slc"], expected)

    def test_uneven_groups(self, slices_10: TrajectoryData):
        """Test interleaving when groups have different sizes."""
        orderer = TrajectoryOrderer(InterleavedOrdering(n_interleaves=3))
        result = orderer.order(slices_10)

        # Group 0: 0, 3, 6, 9 (4 items)
        # Group 1: 1, 4, 7 (3 items)
        # Group 2: 2, 5, 8 (3 items)
        expected = [0, 3, 6, 9, 1, 4, 7, 2, 5, 8]
        np.testing.assert_array_equal(result.indices["slc"], expected)


class TestInterleaveOrder:
    """Tests for interleave ordering options."""

    def test_sequential(self, slices_12: TrajectoryData):
        """Test sequential interleave order (default)."""
        orderer = TrajectoryOrderer(
            InterleavedOrdering(n_interleaves=3, interleave_order="sequential")
        )
        result = orderer.order(slices_12)

        # Groups acquired in order 0, 1, 2
        group0 = result.indices["slc"][:4]
        group1 = result.indices["slc"][4:8]
        group2 = result.indices["slc"][8:]

        assert set(group0) == {0, 3, 6, 9}
        assert set(group1) == {1, 4, 7, 10}
        assert set(group2) == {2, 5, 8, 11}

    def test_reversed(self, slices_12: TrajectoryData):
        """Test reversed interleave order."""
        orderer = TrajectoryOrderer(
            InterleavedOrdering(n_interleaves=3, interleave_order="reversed")
        )
        result = orderer.order(slices_12)

        # Groups acquired in order 2, 1, 0
        group2 = result.indices["slc"][:4]
        group1 = result.indices["slc"][4:8]
        group0 = result.indices["slc"][8:]

        assert set(group2) == {2, 5, 8, 11}
        assert set(group1) == {1, 4, 7, 10}
        assert set(group0) == {0, 3, 6, 9}

    def test_center_out_interleave_order(self, slices_8: TrajectoryData):
        """Test center-out interleave order."""
        orderer = TrajectoryOrderer(
            InterleavedOrdering(n_interleaves=4, interleave_order="center_out")
        )
        result = orderer.order(slices_8)

        # 4 interleaves: center_out order is [2, 3, 1, 0] or similar
        # (starting from middle, alternating outward)
        # Just verify all points are present
        assert result.n_points == 8
        assert set(result.indices["slc"]) == set(range(8))

    def test_custom_interleave_order(self, slices_12: TrajectoryData):
        """Test custom interleave order."""
        orderer = TrajectoryOrderer(
            InterleavedOrdering(n_interleaves=3, interleave_order=[2, 0, 1])
        )
        result = orderer.order(slices_12)

        # Groups acquired in order 2, 0, 1
        group2 = result.indices["slc"][:4]
        group0 = result.indices["slc"][4:8]
        group1 = result.indices["slc"][8:]

        assert set(group2) == {2, 5, 8, 11}
        assert set(group0) == {0, 3, 6, 9}
        assert set(group1) == {1, 4, 7, 10}

    def test_invalid_custom_order_raises(self):
        """Test that invalid custom order raises."""
        with pytest.raises(ValueError, match="must be a permutation"):
            orderer = TrajectoryOrderer(
                InterleavedOrdering(
                    n_interleaves=3, interleave_order=[0, 1]
                )  # Missing 2
            )
            # Need to actually call compute_order to trigger validation
            data = TrajectoryData(
                scaling={"slc": np.arange(6, dtype=float)},
                indices={"slc": np.arange(6)},
                mask=np.ones(6, dtype=bool),
                dim_labels=("slc",),
            )
            orderer.order(data)


class TestInterleavedEdgeCases:
    """Edge cases and special scenarios."""

    def test_single_interleave(self, slices_8: TrajectoryData):
        """Test single interleave (effectively sequential)."""
        orderer = TrajectoryOrderer(InterleavedOrdering(n_interleaves=1))
        result = orderer.order(slices_8)

        # Should just be ascending order
        expected = list(range(8))
        np.testing.assert_array_equal(result.indices["slc"], expected)

    def test_n_interleaves_equals_n_points(self, slices_8: TrajectoryData):
        """Test when n_interleaves equals number of points."""
        orderer = TrajectoryOrderer(InterleavedOrdering(n_interleaves=8))
        result = orderer.order(slices_8)

        # Each point is its own interleave, should be sequential
        expected = list(range(8))
        np.testing.assert_array_equal(result.indices["slc"], expected)

    def test_n_interleaves_exceeds_n_points(self, slices_8: TrajectoryData):
        """Test when n_interleaves exceeds number of points."""
        orderer = TrajectoryOrderer(InterleavedOrdering(n_interleaves=16))
        result = orderer.order(slices_8)

        # Should still work, just some interleaves are empty
        assert result.n_points == 8
        assert set(result.indices["slc"]) == set(range(8))

    def test_with_mask(self):
        """Test interleaved ordering with masked data."""
        n = 8
        mask = np.ones(n, dtype=bool)
        mask[3] = False  # Skip slice 3
        mask[6] = False  # Skip slice 6

        data = TrajectoryData(
            scaling={"slc": np.linspace(0, 1, n)},
            indices={"slc": np.arange(n)},
            mask=mask,
            dim_labels=("slc",),
        )

        orderer = TrajectoryOrderer(InterleavedOrdering(n_interleaves=2))
        result = orderer.order(data)

        # Should only have 6 slices
        assert result.n_points == 6

        # Even: 0, 2, 4 (6 is masked)
        # Odd: 1, 5, 7 (3 is masked)
        expected = [0, 2, 4, 1, 5, 7]
        np.testing.assert_array_equal(result.indices["slc"], expected)

    def test_invalid_dimension_raises(self, slices_8: TrajectoryData):
        """Test that invalid dimension raises."""
        orderer = TrajectoryOrderer(InterleavedOrdering(dim="invalid"))
        with pytest.raises(ValueError, match="not found in data"):
            orderer.order(slices_8)

    def test_explicit_dimension(self, slices_8: TrajectoryData):
        """Test explicit dimension specification."""
        orderer = TrajectoryOrderer(InterleavedOrdering(n_interleaves=2, dim="slc"))
        result = orderer.order(slices_8)

        # Should work the same as default
        expected = [0, 2, 4, 6, 1, 3, 5, 7]
        np.testing.assert_array_equal(result.indices["slc"], expected)

    def test_scaling_stays_aligned(self, slices_8: TrajectoryData):
        """Test that scaling stays aligned with indices."""
        orderer = TrajectoryOrderer(InterleavedOrdering(n_interleaves=2))
        result = orderer.order(slices_8)

        # Verify each point's scaling matches its index
        for i in range(result.n_points):
            idx = result.indices["slc"][i]
            expected_scaling = idx / 7  # linspace(0, 1, 8)
            assert result.scaling["slc"][i] == pytest.approx(expected_scaling)
