"""The variable-density Poisson-disc mask, and the search that finds it.

Bridson dart-throwing is stochastic, so the acceleration a density slope
yields is a noisy function of that slope and the binary search over it is a
search over a noisy objective. These tests pin the three properties that makes
necessary: it must not fail on a request it can satisfy, it must not report a
worse mask than one it already saw, and it must say which of the two reasons
it is refusing for.
"""

from __future__ import annotations

import numpy as np
import pytest

from pulserver.design._sampling import make_poisson_disc_mask

SHAPES = [(48, 48), (48, 40), (64, 64), (32, 96), (100, 100), (20, 20)]
ACCELERATIONS = [2.0, 3.0, 4.0, 6.0, 8.0]
CALIBRATIONS = [(0, 0), (8, 6), (12, 12)]


def _ceiling(shape, calib):
    """The largest acceleration the forced calibration block leaves reachable."""
    forced = calib[0] * calib[1]
    return float("inf") if forced == 0 else shape[0] * shape[1] / forced


def _quantum(shape, acceleration):
    """How much one sample moves the acceleration near the target."""
    return acceleration**2 / (shape[0] * shape[1])


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("acceleration", ACCELERATIONS)
@pytest.mark.parametrize("calib", CALIBRATIONS)
@pytest.mark.parametrize("seed", range(6))
def test_a_request_is_either_met_or_refused_for_a_reason_it_states(
    shape, acceleration, calib, seed
):
    """No seed-dependent failures left over.

    Two refusals are legitimate and both are arithmetic, not luck: the
    calibration block alone is denser than the request, or the grid is too
    coarse for one sample to land inside the tolerance.
    """
    tolerance = 0.1
    try:
        mask = make_poisson_disc_mask(shape, acceleration, calib=calib, seed=seed, tol=tolerance)
    except ValueError as refusal:
        reason = str(refusal)
        if "unreachable" in reason:
            assert acceleration - tolerance > _ceiling(shape, calib), reason
        else:
            assert tolerance < _quantum(shape, acceleration), reason
            assert "one sample moves the acceleration" in reason
        return

    assert mask.shape == shape
    assert abs(mask.size / mask.sum() - acceleration) < tolerance


def test_an_impossible_calibration_block_is_named_before_any_searching():
    """The 12x12 block is 144 of 400 points, so 3x cannot happen at any seed."""
    with pytest.raises(ValueError, match=r"unreachable.*calibration block.*caps acceleration"):
        make_poisson_disc_mask((20, 20), 3.0, calib=(12, 12), seed=0)


def test_a_tolerance_finer_than_the_grid_can_express_says_so():
    """256 points at 9x is 28 samples, and 27 or 29 is 0.32 away.

    No mask on this grid takes a value within 0.005 of 9, so the refusal has to
    name the quantisation rather than read as an unlucky search.
    """
    with pytest.raises(ValueError, match=r"one sample moves the acceleration by about 0\.31"):
        make_poisson_disc_mask((16, 16), 9.0, calib=(0, 0), seed=0, tol=0.005)


def test_tightening_the_tolerance_never_returns_a_worse_mask():
    """A looser tolerance stops the search sooner; it must not search better."""
    errors = []
    for tolerance in (2.0, 1.0, 0.5, 0.3, 0.2, 0.1):
        mask = make_poisson_disc_mask((20, 20), 4.0, calib=(0, 0), seed=3, tol=tolerance)
        errors.append(abs(mask.size / mask.sum() - 4.0))

    assert errors == sorted(errors, reverse=True)


def test_one_seed_gives_one_mask():
    """Re-drawing on fresh substreams must not cost reproducibility."""
    first = make_poisson_disc_mask((48, 40), 4.0, calib=(8, 6), seed=9, tol=0.15)
    second = make_poisson_disc_mask((48, 40), 4.0, calib=(8, 6), seed=9, tol=0.15)
    np.testing.assert_array_equal(first, second)


def test_different_seeds_give_different_masks():
    first = make_poisson_disc_mask((48, 40), 4.0, calib=(8, 6), seed=1, tol=0.15)
    second = make_poisson_disc_mask((48, 40), 4.0, calib=(8, 6), seed=2, tol=0.15)
    assert not np.array_equal(first, second)


def test_the_calibration_block_is_always_fully_sampled():
    mask = make_poisson_disc_mask((64, 64), 6.0, calib=(12, 10), seed=4)
    centre_y, centre_x = 64 // 2, 64 // 2
    block = mask[centre_y - 6 : centre_y + 6, centre_x - 5 : centre_x + 5]
    assert block.all()


def test_an_acceleration_below_one_is_refused():
    with pytest.raises(ValueError, match="accel must be greater than 1"):
        make_poisson_disc_mask((16, 16), 0.5)
