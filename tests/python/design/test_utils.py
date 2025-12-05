"""Tests for ordering utility functions."""

import numpy as np
import pytest

from numpy.typing import NDArray

from pulserver.design import (
    apply_order,
    reorder_within_segments,
    compose_orderings,
    flatten_order,
    LinearOrdering,
    CenterOutOrdering,
    RandomOrdering
)



class TestApplyOrder:
    """Tests for apply_order function."""

    def test_basic_application(self):
        """Test basic order application."""
        coordinates = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
        order = np.array([[7, 5, 3, 1], [6, 4, 2, 0]])

        sorted_coords, sorted_scaling = apply_order(order, coordinates)

        assert sorted_coords.shape == (2, 4)
        np.testing.assert_array_equal(sorted_coords[0], [7, 5, 3, 1])
        np.testing.assert_array_equal(sorted_coords[1], [6, 4, 2, 0])
        assert sorted_scaling is None

    def test_with_scaling(self):
        """Test order application with scaling."""
        coordinates = np.array([[0, 1, 2, 3]])
        scaling = np.array([[0.0, 0.5, 1.0, 1.5]])
        order = np.array([[3, 1, 2, 0]])

        sorted_coords, sorted_scaling = apply_order(order, coordinates, scaling)

        np.testing.assert_array_equal(sorted_coords, [[3, 1, 2, 0]])
        np. testing.assert_array_equal(sorted_scaling, [[1.5, 0.5, 1.0, 0.0]])

    def test_2d_coordinates(self):
        """Test with 2D coordinates."""
        ky = np.array([0, 0, 1, 1])
        kz = np.array([0, 1, 0, 1])
        coordinates = np.stack([ky, kz])
        order = np.array([[3, 2, 1, 0]])

        sorted_coords, _ = apply_order(order, coordinates)

        assert sorted_coords.shape == (2, 1, 4)
        np.testing.assert_array_equal(sorted_coords[0, 0], [1, 1, 0, 0])
        np.testing.assert_array_equal(sorted_coords[1, 0], [1, 0, 1, 0])

    def test_1d_coordinates_squeezed(self):
        """Test that 1D coordinates are squeezed."""
        coordinates = np.array([0, 1, 2, 3])
        order = np.array([[3, 2, 1, 0]])

        sorted_coords, _ = apply_order(order, coordinates)

        # Should be squeezed to 2D (segments, points)
        assert sorted_coords.shape == (1, 4)

    def test_multisegment(self):
        """Test with multiple segments."""
        coordinates = np.array([[0, 1, 2, 3, 4, 5]])
        scaling = np.array([[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]])
        order = np.array([[0, 2, 4], [1, 3, 5]])

        sorted_coords, sorted_scaling = apply_order(order, coordinates, scaling)

        assert sorted_coords.shape == (2, 3)
        np.testing.assert_array_equal(sorted_coords[0], [0, 2, 4])
        np.testing.assert_array_equal(sorted_coords[1], [1, 3, 5])
        np.testing.assert_array_almost_equal(sorted_scaling[0], [0.0, 0.4, 0.8])
        np.testing.assert_array_almost_equal(sorted_scaling[1], [0.2, 0.6, 1.0])


class TestReorderWithinSegments:
    """Tests for reorder_within_segments function."""

    def test_center_out_within_linear(self):
        """Test center-out reordering within linear segments."""
        # Create linear coordinates
        coords = np.arange(16).reshape(1, -1) - 8  # Centered at -0.5

        # Linear ordering with 4 segments
        linear_order = LinearOrdering().compute_order(coords, n_segments=4)
        # Each segment has 4 points in linear order

        # Reorder within segments to be center-out
        reordered = reorder_within_segments(
            linear_order, coords, CenterOutOrdering()
        )

        assert reordered.shape == linear_order.shape

        # Each segment should now be center-out
        for seg in range(4):
            seg_coords = coords[0, reordered[seg]]
            center = seg_coords.mean()
            radii = np.abs(seg_coords - center)
            # Radii should be non-decreasing
            assert np.all(np.diff(radii) >= -1e-10)

    def test_preserves_segment_points(self):
        """Test that reordering preserves segment membership."""
        coords = np.arange(12).reshape(1, -1)

        original_order = LinearOrdering().compute_order(coords, n_segments=3)
        reordered = reorder_within_segments(
            original_order, coords, RandomOrdering(seed=42)
        )

        # Each segment should contain the same points
        for seg in range(3):
            assert set(original_order[seg]) == set(reordered[seg])

    def test_2d_coordinates(self):
        """Test reordering with 2D coordinates."""
        ky, kz = np. meshgrid(np.arange(4) - 2, np.arange(4) - 2, indexing='ij')
        coords = np.stack([ky.ravel(), kz.ravel()])

        order = LinearOrdering().compute_order(coords, n_segments=4)
        reordered = reorder_within_segments(order, coords, CenterOutOrdering())

        assert reordered.shape == order.shape
        # All indices should still be valid
        assert set(reordered.ravel()) == set(range(16))


class TestComposeOrderings:
    """Tests for compose_orderings function."""

    def test_basic_composition(self):
        """Test basic ordering composition."""
        # 3 frames, 2 blocks per frame
        outer_order = np.array([[0, 1], [2, 3], [4, 5]])
        # 4 points per block
        inner_order = np.array([3, 1, 2, 0])

        composed = compose_orderings(outer_order, inner_order)

        assert composed.shape == (3, 2, 4)

        # Check first frame, first block
        # Block 0: indices 0*4 + [3,1,2,0] = [3,1,2,0]
        np.testing.assert_array_equal(composed[0, 0], [3, 1, 2, 0])

        # Check first frame, second block
        # Block 1: indices 1*4 + [3,1,2,0] = [7,5,6,4]
        np. testing.assert_array_equal(composed[0, 1], [7, 5, 6, 4])

        # Check second frame, first block
        # Block 2: indices 2*4 + [3,1,2,0] = [11,9,10,8]
        np.testing.assert_array_equal(composed[1, 0], [11, 9, 10, 8])

    def test_with_2d_inner_order(self):
        """Test composition with 2D inner order."""
        outer_order = np.array([[0, 1], [2, 3]])
        inner_order = np.array([[2, 0, 1]])  # 2D with shape (1, 3)

        composed = compose_orderings(outer_order, inner_order)

        assert composed.shape == (2, 2, 3)
        np.testing.assert_array_equal(composed[0, 0], [2, 0, 1])
        np.testing.assert_array_equal(composed[0, 1], [5, 3, 4])

    def test_all_indices_unique(self):
        """Test that composed indices are all unique."""
        outer_order = np.array([[2, 0], [3, 1]])  # Shuffled blocks
        inner_order = np.array([1, 0, 2])  # 3 points per block

        composed = compose_orderings(outer_order, inner_order)
        flat = composed.ravel()

        assert len(flat) == len(np.unique(flat))

    def test_epi_multiframe_scenario(self):
        """Test realistic EPI + multi-frame scenario."""
        # 10 time frames, 4 EPI shots per frame = 40 total shots
        n_frames = 10
        n_shots_per_frame = 4
        n_lines_per_shot = 64

        # Outer: spiral ordering of shots across frames
        shot_indices = np.arange(n_frames * n_shots_per_frame)
        outer_order = LinearOrdering().compute_order(
            shot_indices.reshape(1, -1), n_segments=n_frames
        )
        assert outer_order.shape == (n_frames, n_shots_per_frame)

        # Inner: linear ordering within each EPI shot
        line_indices = np.arange(n_lines_per_shot)
        inner_order = LinearOrdering(). compute_order(
            line_indices.reshape(1, -1), n_segments=1
        ). ravel()

        # Compose
        full_order = compose_orderings(outer_order, inner_order)

        assert full_order.shape == (n_frames, n_shots_per_frame, n_lines_per_shot)
        assert full_order.size == n_frames * n_shots_per_frame * n_lines_per_shot


class TestFlattenOrder:
    """Tests for flatten_order function."""

    def test_2d_flatten(self):
        """Test flattening 2D order."""
        order = np.array([[0, 2, 4], [1, 3, 5]])

        flat = flatten_order(order)

        np.testing.assert_array_equal(flat, [0, 2, 4, 1, 3, 5])

    def test_3d_flatten(self):
        """Test flattening 3D order."""
        order = np.array([
            [[0, 1], [2, 3]],
            [[4, 5], [6, 7]],
        ])

        flat = flatten_order(order)

        np. testing.assert_array_equal(flat, [0, 1, 2, 3, 4, 5, 6, 7])

    def test_already_1d(self):
        """Test flattening already 1D order."""
        order = np.array([3, 1, 4, 1, 5])

        flat = flatten_order(order)

        np.testing. assert_array_equal(flat, order)


class TestIntegration:
    """Integration tests combining multiple utilities."""

    def test_full_workflow(self):
        """Test complete ordering workflow."""
        # Create 2D k-space coordinates
        ky, kz = np.meshgrid(np.arange(16) - 8, np.arange(16) - 8, indexing='ij')
        coordinates = np.stack([ky.ravel(), kz.ravel()])
        scaling = coordinates / 8.0  # Normalized scaling

        # Create mask (simulate undersampling)
        rng = np.random.default_rng(42)
        mask = rng.random(256) > 0.25  # ~75% sampled
        n_sampled = mask.sum()

        # Find n_segments that divides evenly
        n_segments = 4
        while n_sampled % n_segments != 0:
            n_segments -= 1

        # Step 1: Compute linear order with segments
        order = LinearOrdering(). compute_order(
            coordinates, mask=mask, n_segments=n_segments
        )

        # Step 2: Reorder within segments (center-out)
        order = reorder_within_segments(
            order, coordinates[:, mask], CenterOutOrdering()
        )

        # Step 3: Apply order to get sorted coordinates and scaling
        sorted_coords, sorted_scaling = apply_order(
            order, coordinates[:, mask], scaling[:, mask]
        )

        # Verify shapes
        n_per_seg = n_sampled // n_segments
        assert sorted_coords.shape == (2, n_segments, n_per_seg)
        assert sorted_scaling.shape == (2, n_segments, n_per_seg)

        # Step 4: Flatten for acquisition
        flat_order = flatten_order(order)
        assert len(flat_order) == n_sampled

        # Verify all indices are unique
        assert len(np.unique(flat_order)) == n_sampled

    def test_nested_segmentation_workflow(self):
        """Test workflow with nested segmentation."""
        # Scenario: 4 contrasts, 8 shots per contrast, 32 points per shot
        n_contrasts = 4
        n_shots = 8
        n_per_shot = 32
        n_total = n_contrasts * n_shots * n_per_shot  # 1024

        # Create coordinates for "shots" (blade centers)
        shot_coords = np.arange(n_contrasts * n_shots). reshape(1, -1)

        # Create coordinates for points within shot
        point_coords = np.arange(n_per_shot).reshape(1, -1) - n_per_shot // 2

        # Outer ordering: divide shots into contrasts
        outer_order = LinearOrdering().compute_order(
            shot_coords, n_segments=n_contrasts
        )
        assert outer_order.shape == (n_contrasts, n_shots)

        # Inner ordering: center-out within each shot
        inner_order = CenterOutOrdering(). compute_order(
            point_coords, n_segments=1
        ).ravel()
        assert len(inner_order) == n_per_shot

        # Compose
        full_order = compose_orderings(outer_order, inner_order)
        assert full_order.shape == (n_contrasts, n_shots, n_per_shot)

        # Flatten
        flat = flatten_order(full_order)
        assert len(flat) == n_total
        assert len(np. unique(flat)) == n_total