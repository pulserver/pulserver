"""Unit tests for pulserver private sampling view-ordering helpers."""

from __future__ import annotations

import numpy as np
import pytest
from pulserver.pypulseq import _sampling as sampling


def _coords_grid(n: int) -> np.ndarray:
    ky, kz = np.meshgrid(np.arange(n), np.arange(n))
    return np.column_stack([ky.ravel(), kz.ravel()]).astype(float)


def _all_views_once(shots: list[list[int]], n: int) -> bool:
    flat = [i for shot in shots for i in shot]
    return sorted(flat) == list(range(n))


def test_chunk_and_linear_order_agree() -> None:
    assert sampling.make_linear_order(5, 2) == [[0, 1], [2, 3], [4]]
    assert sampling.calc_chunk_indices([0, 1, 2, 3, 4], 2) == sampling.make_linear_order(5, 2)


def test_outer_inner_order() -> None:
    assert sampling.make_outer_inner_order([0, 1], 2) == [
        [(0, 0), (0, 1)],
        [(1, 0), (1, 1)],
    ]


@pytest.mark.parametrize(
    "fn",
    [sampling.make_linear_order, sampling.make_radial_order, sampling.make_radial_adaptive_order],
)
def test_orderings_cover_every_view_once(fn) -> None:
    coords = _coords_grid(6)
    etl = 6
    shots = fn(coords, etl)
    assert _all_views_once(shots, len(coords))
    assert all(len(s) <= etl for s in shots)


def test_radial_order_is_center_out() -> None:
    coords = _coords_grid(7)  # 49 points, center at (3, 3)
    etl = 7
    shots = sampling.make_radial_order(coords, etl)
    center = coords.mean(axis=0)
    for shot in shots:
        radii = [np.hypot(*(coords[i] - center)) for i in shot]
        assert radii == sorted(radii), "each echo train must run center-out"


def test_radial_adaptive_early_echoes_are_central() -> None:
    coords = _coords_grid(8)
    etl = 8
    shots = sampling.make_radial_adaptive_order(coords, etl, n_sections=4)
    center = coords.mean(axis=0)
    # First echo of every shot should be nearer the center than its last echo.
    for shot in shots:
        if len(shot) < 2:
            continue
        r_first = np.hypot(*(coords[shot[0]] - center))
        r_last = np.hypot(*(coords[shot[-1]] - center))
        assert r_first <= r_last


def test_shuffling_reproducible_and_incoherent() -> None:
    coords = _coords_grid(8)
    etl = 8
    a = sampling.make_shuffling_order(coords, etl, seed=42)
    b = sampling.make_shuffling_order(coords, etl, seed=42)
    c = sampling.make_shuffling_order(coords, etl, seed=7)
    assert a == b
    assert a != c
    assert _all_views_once(a, len(coords))
    # A shuffled train should not be in strict raster order (incoherence).
    linear = sampling.make_linear_order(coords, etl)
    assert a != linear


def test_empty_inputs() -> None:
    assert sampling.make_linear_order(np.empty((0, 2)), 4) == []
    assert sampling.make_radial_order(np.empty((0, 2)), 4) == []
    assert sampling.make_shuffling_order(np.empty((0, 2)), 4, seed=0) == []


def test_1d_coords_accepted() -> None:
    shots = sampling.make_linear_order(np.arange(5.0), 2)
    assert _all_views_once(shots, 5)


def test_random_mask_acceleration_and_calib() -> None:
    mask = sampling.make_random_mask((32, 32), 4.0, calib=(8, 8), seed=0)
    accel = mask.size / mask.sum()
    assert accel == pytest.approx(4.0, rel=0.05)
    assert mask[12:20, 12:20].all()
    again = sampling.make_random_mask((32, 32), 4.0, calib=(8, 8), seed=0)
    assert np.array_equal(mask, again)


def test_caipirinha_mask_acceleration_and_shift() -> None:
    mask = sampling.make_caipirinha_mask((12, 12), 2, 2, delta=1)
    assert mask.size / mask.sum() == pytest.approx(4.0)
    row0 = np.flatnonzero(mask[0])
    row2 = np.flatnonzero(mask[2])
    assert set((row0 + 1) % 12) == set(row2)
    plain = sampling.make_caipirinha_mask((12, 12), 2, 2, delta=0)
    assert np.array_equal(np.flatnonzero(plain[0]), np.flatnonzero(plain[2]))


def test_poisson_disc_mask_acceleration() -> None:
    mask = sampling.make_poisson_disc_mask((48, 48), 4.0, calib=(8, 8), seed=1)
    accel = mask.size / mask.sum()
    assert accel == pytest.approx(4.0, abs=0.15)
    assert mask[20:28, 20:28].all()


def test_mask_rejects_bad_accel() -> None:
    with pytest.raises(ValueError):
        sampling.make_random_mask((16, 16), 1.0)
    with pytest.raises(ValueError):
        sampling.make_poisson_disc_mask((16, 16), 0.5)


def test_golden_and_uniform_angles() -> None:
    golden = sampling.calc_golden_angles(8)
    inc_deg = np.rad2deg(np.diff(golden) % (2 * np.pi))
    assert np.allclose(inc_deg, 111.246, atol=1e-3)
    uniform = sampling.calc_uniform_angles(8)
    assert np.allclose(np.diff(uniform), 2 * np.pi / 8)
