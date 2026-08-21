"""Cartesian view sampling: the ACS block, its count, and the 3D rectangle.

These pin the invariants the Cartesian sequence plugins lean on now that they
build their traversals from the builtins rather than by hand: the
autocalibration lines are a subset of the sampled lines, the rectangle
:func:`calc_sampled_pairs` carves is exactly the pairs whose line and
partition are both calibration views, and the leading ``n_calibration`` pairs
are that rectangle.
"""

from __future__ import annotations

import pytest

from pulserver.pypulseq import (
    calc_calibration_lines,
    calc_sampled_lines,
    calc_sampled_pairs,
)


@pytest.mark.parametrize("n", [64, 120, 128, 127])
@pytest.mark.parametrize("acs", [0, 16, 24, 25])
@pytest.mark.parametrize("pf", [1.0, 0.75, 0.8])
def test_calibration_is_a_subset_of_the_sampled_lines(n, acs, pf):
    calibration = calc_calibration_lines(n, acs, partial_fourier=pf)
    for r in (1, 2, 3):
        sampled = set(calc_sampled_lines(n, r, acs, partial_fourier=pf))
        assert set(calibration) <= sampled


def test_calibration_count_matches_calibration_first_prefix():
    # calibration_first leads with exactly the calibration lines, so their
    # count is the length of the block the builtin reports.
    n, r, acs = 128, 2, 24
    sampled = calc_sampled_lines(n, r, acs, order="calibration_first")
    calibration = calc_calibration_lines(n, acs)
    assert sampled[: len(calibration)] == calibration


def test_calibration_lines_respect_partial_fourier():
    # A calibration view dropped by the truncation is not reported.
    n, acs = 16, 8
    full = calc_calibration_lines(n, acs)
    truncated = calc_calibration_lines(n, acs, partial_fourier=0.75)
    assert set(truncated) <= set(full)
    assert min(truncated) >= n - round(0.75 * n)


@pytest.mark.parametrize("acs_y", [0, 8, 24])
def test_no_calibration_means_empty_rectangle(acs_y):
    _pairs, n_cal = calc_sampled_pairs((64, 32), (2, 2), (acs_y, 0))
    assert n_cal == 0  # the rectangle is empty when either axis has no ACS


def test_pairs_are_the_full_grid_regardless_of_order():
    shape, accel, calib = (128, 64), (2, 2), (24, 16)
    ascending, n_a = calc_sampled_pairs(shape, accel, calib, order="ascending")
    first, n_f = calc_sampled_pairs(shape, accel, calib, order="calibration_first")
    assert n_a == n_f
    assert set(ascending) == set(first)  # same pairs, different order
    assert len(ascending) == len(set(ascending))  # no duplicates


def test_calibration_first_leads_with_the_rectangle():
    shape, accel, calib = (128, 64), (2, 2), (24, 16)
    lines = set(calc_calibration_lines(128, 24))
    partitions = set(calc_calibration_lines(64, 16))
    pairs, n_cal = calc_sampled_pairs(shape, accel, calib)
    rectangle = pairs[:n_cal]
    # Every leading pair calibrates on both axes...
    assert all(line in lines and part in partitions for line, part in rectangle)
    # ...and none of the pairs after it does.
    assert not any(line in lines and part in partitions for line, part in pairs[n_cal:])


def test_partial_fourier_on_z_drops_leading_partitions():
    shape, accel, calib = (64, 64), (1, 1), (0, 0)
    full, _ = calc_sampled_pairs(shape, accel, calib)
    truncated, _ = calc_sampled_pairs(shape, accel, calib, partial_fourier=(1.0, 0.75))
    partitions_full = {part for _, part in full}
    partitions_trunc = {part for _, part in truncated}
    assert partitions_trunc < partitions_full  # z extent genuinely shrank
    assert min(partitions_trunc) >= 64 - round(0.75 * 64)


def test_caipi_shift_staggers_kz_without_changing_the_count():
    # A non-zero shift keeps the same number of samples but moves them.
    plain, _ = calc_sampled_pairs(
        (16, 16), (2, 2), (0, 0), caipi_shift=0, elliptical=False
    )
    shifted, _ = calc_sampled_pairs(
        (16, 16), (2, 2), (0, 0), caipi_shift=1, elliptical=False
    )
    assert len(plain) == len(shifted)
    assert set(plain) != set(shifted)


def test_no_acceleration_samples_the_whole_grid():
    pairs, n_cal = calc_sampled_pairs(
        (8, 8), (1, 1), (0, 0), caipi_shift=1, elliptical=False
    )
    assert len(pairs) == 64  # a shift is a no-op when nothing is skipped
    assert n_cal == 0


def test_the_support_is_the_inscribed_ellipse_unless_told_otherwise():
    """The corners a round object never fills are not worth their scan time."""
    disk, _ = calc_sampled_pairs((16, 16), (1, 1), (0, 0))
    full, _ = calc_sampled_pairs((16, 16), (1, 1), (0, 0), elliptical=False)
    assert set(disk) < set(full)
    # Every dropped pair is outside the inscribed ellipse, and the corners are.
    dropped = set(full) - set(disk)
    assert {(0, 0), (0, 15), (15, 0), (15, 15)} <= dropped
    assert all(((y - 8) / 8) ** 2 + ((z - 8) / 8) ** 2 > 1.0 for y, z in dropped)


def test_caipi_lattice_carries_the_acs_rectangle():
    shape, accel, calib = (32, 32), (2, 2), (8, 8)
    pairs, n_cal = calc_sampled_pairs(shape, accel, calib, caipi_shift=1)
    lines = set(calc_calibration_lines(32, 8))
    partitions = set(calc_calibration_lines(32, 8))
    rectangle = pairs[:n_cal]
    assert n_cal == len(lines) * len(partitions)
    assert all(line in lines and part in partitions for line, part in rectangle)


def test_bad_order_and_fraction_are_rejected():
    with pytest.raises(ValueError):
        calc_sampled_pairs((32, 32), (1, 1), (0, 0), order="nonsense")
    with pytest.raises(ValueError):
        calc_sampled_pairs((32, 32), (1, 1), (0, 0), partial_fourier=(0.4, 1.0))
    with pytest.raises(ValueError):
        calc_calibration_lines(32, 8, partial_fourier=1.5)
