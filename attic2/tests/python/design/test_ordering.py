"""
Pytest test suite for ordering generation routines (function-based tests).

These tests exercise the public functions in pulserver.design (ordering utilities).
They are written to check invariants (shapes, ranges, uniqueness, and simple
expected outputs) rather than exact layouts for randomized or geometry-dependent
outputs.
"""

import numpy as np

import pulserver.tools as pt


def test_make_interleaved_ordering_1d_even_odd():
    ordering = pt.make_interleaved_ordering_1d(10, 2)
    assert ordering.shape == (10,)
    expected = np.array([0, 2, 4, 6, 8, 1, 3, 5, 7, 9])
    assert np.array_equal(ordering, expected)


def test_make_interleaved_ordering_1d_three_groups():
    ordering = pt.make_interleaved_ordering_1d(9, 3)
    assert ordering.shape == (9,)
    expected = np.array([0, 3, 6, 1, 4, 7, 2, 5, 8])
    assert np.array_equal(ordering, expected)


def test_make_centerout_ordering_1d_even_and_odd():
    # even length
    out_even = pt.make_centerout_ordering_1d(10)
    assert out_even.shape == (10,)
    # docstring example ordering for n1=10
    assert np.array_equal(out_even, np.array([5, 4, 6, 3, 7, 2, 8, 1, 9, 0]))

    # odd length should still include all indices exactly once
    out_odd = pt.make_centerout_ordering_1d(7)
    assert out_odd.shape == (7,)
    assert set(out_odd.tolist()) == set(range(7))


def test_make_random_ordering_1d_is_permutation():
    # random-based; check invariant: sorted result equals full range
    n = 20

    # seed global numpy RNG to make function output deterministic if it uses global state
    np.random.seed(12345)
    ordering = pt.make_random_ordering_1d(n)

    assert ordering.shape == (n,)
    assert set(ordering.tolist()) == set(range(n))

    # ordering should not be the trivial sorted order in general
    assert not np.array_equal(ordering, np.arange(n))


def test_make_random_ordering_2d_is_permutation_and_shape():
    ny, nz = 8, 8
    np.random.seed(0)

    ordering = pt.make_random_ordering_2d(ny, nz)
    assert ordering.shape == (ny, nz)

    flat = ordering.ravel()
    assert set(flat.tolist()) == set(range(ny * nz))

    # Each position picked exactly once
    assert len(flat) == len(np.unique(flat))


def test_make_radial_ordering_2d_basic_properties():
    n1, n2 = 8, 4
    inc = np.deg2rad(45.0)
    ordering = pt.make_radial_ordering_2d(n1, n2, inc, theta0=0.0, prune=True)
    # Should return 2D array with second axis == n2
    assert ordering.ndim == 2
    assert ordering.shape[1] == n2
    # All indices should be valid flattened indices for (n1,n1)
    assert ordering.min() >= 0
    assert ordering.max() < n1 * n1
    # Columns should have equal length (prune enforces this)
    col_lengths = [ordering[:, j].size for j in range(ordering.shape[1])]
    assert len(set(col_lengths)) == 1


def test_make_centerout_ordering_2d_basic_properties():
    n1, n2 = 8, 4
    inc = np.deg2rad(90.0)
    ordering = pt.make_centerout_ordering_2d(n1, n2, inc, theta0=0.0, prune=True)
    assert ordering.ndim == 2
    assert ordering.shape[1] == n2
    assert ordering.min() >= 0
    assert ordering.max() < n1 * n1
    # Check no duplicates inside each column
    for j in range(ordering.shape[1]):
        col = ordering[:, j]
        assert col.size == len(np.unique(col))


def test_make_spiral_ordering_2d_basic_properties():
    n1, n2 = 8, 2
    inc = np.deg2rad(180.0)
    ordering = pt.make_spiral_ordering_2d(n1, n2, inc, prune=True)
    assert ordering.ndim == 2
    assert ordering.shape[1] == n2
    # All values in valid range
    assert ordering.min() >= 0
    assert ordering.max() < n1 * n1
    # Prune ensures equal column length
    assert ordering.shape[0] > 0
    # Check columns have unique indices (within each interleaf)
    for j in range(ordering.shape[1]):
        assert len(np.unique(ordering[:, j])) == ordering.shape[0]
