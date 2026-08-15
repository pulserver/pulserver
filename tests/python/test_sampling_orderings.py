"""Echo-train orderings: the target-echo, padding, and coverage contract.

The Cartesian echo-train plugins deal their views into trains with the
builtin orderings rather than by hand, so these pin the properties they lean
on: every ordering covers its views once, ``center_echo`` lands the k-space
centre on the requested echo (folding the radius away from it for the
adaptive one), and ``pad`` gives every train the same ``None``-filled length.
"""

from __future__ import annotations

import numpy as np
import pytest

from pulserver.pypulseq import (
    make_centric_order,
    make_linear_order,
    make_radial_adaptive_order,
    make_radial_order,
    make_shuffling_order,
)

GRID = np.column_stack(
    [g.ravel() for g in np.meshgrid(np.arange(8), np.arange(4), indexing="ij")]
)
CENTER = (4, 2)
ETL = 8


def _radius(index):
    return float(np.hypot(GRID[index, 0] - 4, GRID[index, 1] - 2))


def _centre_echo(trains):
    for train in trains:
        for echo, index in enumerate(train):
            if index is not None and tuple(GRID[index]) == CENTER:
                return echo
    raise AssertionError("centre view was never encoded")


ALL_ORDERINGS = [make_linear_order, make_centric_order, make_radial_order,
                 make_radial_adaptive_order, make_shuffling_order]


@pytest.mark.parametrize("order", ALL_ORDERINGS)
def test_every_ordering_covers_its_views_once(order):
    trains = order(GRID, ETL)
    flat = [index for train in trains for index in train if index is not None]
    assert sorted(flat) == list(range(len(GRID))), order.__name__


@pytest.mark.parametrize("order", ALL_ORDERINGS)
def test_pad_gives_every_train_the_train_length(order):
    trains = order(GRID, ETL, pad=True)
    assert all(len(train) == ETL for train in trains)
    # The real views still cover the grid exactly once.
    flat = [index for train in trains for index in train if index is not None]
    assert sorted(flat) == list(range(len(GRID)))


@pytest.mark.parametrize(
    "order", [make_linear_order, make_centric_order, make_radial_adaptive_order]
)
def test_center_echo_places_the_centre_on_the_target_echo(order):
    target = 3
    trains = order(GRID, ETL, center=CENTER, center_echo=target, pad=True)
    assert _centre_echo(trains) == target


def test_radial_puts_the_centre_on_the_first_echo():
    trains = make_radial_order(GRID, ETL, center=CENTER, pad=True)
    assert _centre_echo(trains) == 0


def test_radial_adaptive_radius_grows_away_from_the_target_echo():
    target = 3
    trains = make_radial_adaptive_order(
        GRID, ETL, center=CENTER, center_echo=target, pad=True
    )
    by_echo: dict[int, list[float]] = {}
    for train in trains:
        for echo, index in enumerate(train):
            if index is not None:
                by_echo.setdefault(echo, []).append(_radius(index))
    mean_radius = {echo: np.mean(values) for echo, values in by_echo.items()}
    assert min(mean_radius, key=mean_radius.get) == target
    # Monotone growth on each side of the target echo.
    for echo in range(target, max(mean_radius)):
        assert mean_radius[echo + 1] >= mean_radius[echo] - 1e-9
    for echo in range(target, 0, -1):
        assert mean_radius[echo - 1] >= mean_radius[echo] - 1e-9


def test_shuffling_is_reproducible_by_seed():
    a = make_shuffling_order(GRID, ETL, seed=7, pad=True)
    b = make_shuffling_order(GRID, ETL, seed=7, pad=True)
    c = make_shuffling_order(GRID, ETL, seed=8, pad=True)
    assert a == b
    assert a != c
