"""
Pytest test suite for sampling.py (non-class-based tests).

These tests use the real CPD bindings provided by the module (no monkeypatching).
They check invariants (shapes, dtype, and calibration insertion) rather than exact
mask layouts because Poisson-disc sampling is randomized.
"""

import numpy as np
import pytest

import pulserver.tools as pt


def test_make_regular_sampling_basic():
    # Basic regular sampling without calibration
    m = pt.make_regular_sampling(12, 3)
    expected = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0])
    assert isinstance(m, np.ndarray)
    assert m.shape == (12,)
    assert np.array_equal(m, expected)


def test_make_regular_sampling_with_calib():
    # Regular sampling with calibration region
    m = pt.make_regular_sampling(12, 3, calib=4)
    # central 4 values (indices 4..7) should be 1
    center = m[12 // 2 - 4 // 2 : 12 // 2 + 4 // 2]
    assert center.shape == (4,)
    assert np.all(center == 1)


def test_make_caipirinha_sampling_basic_and_shift():
    # Basic CAIPIRINHA sampling on an 8x8 grid with R=(2,2)
    m = pt.make_caipirinha_sampling((8, 8), accel=(2, 2), shift=0, elliptical=False)
    assert m.shape == (8, 8)
    # For 2x2 acceleration on 8x8 regular grid we expect 8*8/(2*2) = 16 samples
    assert int(np.sum(m)) == 16

    # With shift=1 and a 2x2 central calibration, the center block should be ones
    m_shift = pt.make_caipirinha_sampling(
        (8, 8), accel=(2, 2), shift=1, calib=2, elliptical=False
    )
    assert m_shift.shape == (8, 8)
    center_block = m_shift[3:5, 3:5]
    assert center_block.shape == (2, 2)
    assert np.all(center_block == 1)


def test_make_partial_fourier_sampling_snippet():
    # Matches the snippet provided in the repository
    mask = pt.make_partial_fourier_sampling(16, 0.75)
    assert mask.sum() == 12
    assert int(mask[-1]) == 0

    # undersampling == 1.0 returns all ones
    full = pt.make_partial_fourier_sampling(8, 1.0)
    assert np.array_equal(full, np.ones(8, dtype=int))


def test_make_poisson_sampling_vardens_inserts_calib():
    """
    Test make_poisson_sampling with vardens=True. Ensure that when `calib`
    is provided the returned mask contains a central calibration block of ones.
    Uses the real CPD bindings provided by the module.
    """
    cy = cz = 20
    mask = pt.make_poisson_sampling(
        (64, 64), accel=(2, 2), calib=cy, nt=1, vardens=True
    )

    # Returned shape should include the temporal dimension (nt=1)
    assert mask.ndim in (2, 3)
    # Normalize to 3D for assertions
    if mask.ndim == 2:
        mask = mask[:, :, None]

    ny, nz, nt = mask.shape
    assert ny == 64 and nz == 64
    assert nt == 1

    center_block = mask[32 - cy // 2 : 32 + cy // 2, 32 - cz // 2 : 32 + cz // 2, 0]
    assert center_block.shape == (cy, cz)
    # The calibration region should be fully set to ones
    assert int(np.sum(center_block)) == cy * cz


def test_make_poisson_sampling_ud_multiple_temporal():
    """
    Test make_poisson_sampling with vardens=False (UD case) and multiple temporal phases.
    Use real bindings and verify shape and that some sampling points exist across
    the temporal frames.
    """
    m_t = pt.make_poisson_sampling((32, 32), accel=(4, 1), nt=3, vardens=False)

    assert m_t.ndim == 3
    assert m_t.shape == (32, 32, 3)
    # Combined logical OR across temporal frames yields a 2D sampling coverage map
    combined = np.sum(m_t, axis=2) > 0
    assert combined.dtype == bool
    # There should be at least one sampled location across temporal frames
    assert combined.any()


@pytest.mark.parametrize(
    "shape, accel, expected_total",
    [
        ((8, 8), (1, 1), 64),  # no accel -> full
        ((8, 8), (2, 1), 32),  # 2x along first axis only
        ((6, 6), (2, 3), 6),  # general case: 6*6/(2*3) = 6
    ],
)
def test_make_caipirinha_sampling_counts(shape, accel, expected_total):
    # elliptical=False to avoid corner cropping affecting counts
    m = pt.make_caipirinha_sampling(shape, accel=accel, shift=0, elliptical=False)
    assert int(np.sum(m)) == expected_total
