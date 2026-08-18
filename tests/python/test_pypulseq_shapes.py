"""``restore_additional_shape_samples``: the properties that define it.

The function turns a waveform stored at raster-interval centres back into
the vertex polyline it came from. What that has to mean is checkable with
no reference data: recovered corners land on the exact knot times, every
original centre sample still lies on the returned polyline, and the shapes
that cannot be restored come back untouched, bracketed by ``first`` and
``last``, with a warning saying so.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pulserver.pypulseq._shapes import (
    COLLINEAR_TOLERANCE,
    RESTORE_TOLERANCE,
    restore_additional_shape_samples,
)

RASTER = 10e-6


def _centres(count: int) -> np.ndarray:
    """Sample times at the centres of the raster intervals, as Pulseq stores them."""
    return (np.arange(count) + 0.5) * RASTER


def _trapezoid(rise: float, flat: float, fall: float, amplitude: float):
    count = round((rise + flat + fall) / RASTER)
    times = _centres(count)
    waveform = np.interp(
        times, [0, rise, rise + flat, rise + flat + fall], [0, amplitude, amplitude, 0]
    )
    return times, waveform, 0.0, 0.0


def _cases() -> list[tuple[str, np.ndarray, np.ndarray, float, float]]:
    """One shape per branch of the restoration: corners, ramps, curves, edge sizes."""
    cases: list[tuple[str, np.ndarray, np.ndarray, float, float]] = []
    cases.append(("symmetric trapezoid", *_trapezoid(200e-6, 400e-6, 200e-6, 1000.0)))
    cases.append(("asymmetric trapezoid", *_trapezoid(120e-6, 50e-6, 300e-6, -750.0)))
    cases.append(("triangle", *_trapezoid(150e-6, 0.0, 150e-6, 900.0)))

    count = 40
    ramp = np.linspace(100.0, 900.0, count)
    half_step = (ramp[1] - ramp[0]) / 2.0
    cases.append(
        (
            "linear ramp",
            _centres(count),
            ramp,
            ramp[0] - half_step,
            ramp[-1] + half_step,
        )
    )

    knot_times = np.array([0, 100e-6, 260e-6, 300e-6, 520e-6])
    knot_amplitudes = np.array([0.0, 800.0, 800.0, -400.0, 0.0])
    times = _centres(round(knot_times[-1] / RASTER))
    cases.append(
        (
            "extended trapezoid",
            times,
            np.interp(times, knot_times, knot_amplitudes),
            0.0,
            0.0,
        )
    )

    count = 200
    angle = np.linspace(0, 8 * np.pi, count)
    cases.append(
        (
            "spiral arm",
            _centres(count),
            1000.0 * angle / (8 * np.pi) * np.cos(angle),
            0.0,
            0.0,
        )
    )

    cases.append(("one sample", _centres(1), np.array([500.0]), 0.0, 1000.0))
    return cases


CASES = _cases()
IDS = [name for name, *_ in CASES]

#: The cases the restoration succeeds on, by index into CASES. The spiral arm
#: (index 5) exceeds the restoration tolerance by construction: a curve
#: sampled at centres has no polyline through vertices that reproduces it.
RESTORED = {0, 1, 2, 3, 4, 6}


@pytest.mark.parametrize("index", range(len(CASES)), ids=IDS)
def test_which_branch_ran_is_part_of_the_contract(index):
    """The bail-out is a documented branch, so which one ran is asserted.

    A shape that bails out comes back as the samples it went in as, bracketed
    by ``first`` and ``last`` -- unchanged, so nothing downstream sees a
    half-restored waveform.
    """
    _, times, waveform, first, last = CASES[index]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got_times, got_waveform = restore_additional_shape_samples(
            times, waveform, first, last, RASTER
        )

    if index in RESTORED:
        assert not caught
        # Never more than the interleaved edges-and-centres grid it decimates.
        assert got_times.size <= 2 * waveform.size + 1
    else:
        assert caught, "a dropped restoration has to say so"
        np.testing.assert_allclose(
            got_waveform, np.concatenate(([first], waveform, [last]))
        )


def test_a_restored_trapezoid_recovers_its_corners_exactly():
    """The point of the whole function: corners on the grid, not rounded over it.

    A trapezoid stored on a centres raster has no sample at its corners, so
    reading it back without this gives ramps smeared over one raster interval
    and a peak slew lower than the sequence asks for.
    """
    rise, flat, fall, amplitude = 200e-6, 400e-6, 200e-6, 1000.0
    times, waveform, first, last = _trapezoid(rise, flat, fall, amplitude)

    got_times, got_waveform = restore_additional_shape_samples(
        times, waveform, first, last, RASTER
    )

    np.testing.assert_allclose(got_times, [0, rise, rise + flat, rise + flat + fall])
    np.testing.assert_allclose(got_waveform, [0, amplitude, amplitude, 0])

    # And the slew the corners imply is the real one, not the smeared one.
    assert np.max(np.abs(np.diff(got_waveform) / np.diff(got_times))) == pytest.approx(
        amplitude / rise
    )


def test_a_restored_extended_trapezoid_recovers_its_knots():
    """Interior knots off the raster grid come back at their exact times."""
    knot_times = np.array([0, 100e-6, 260e-6, 300e-6, 520e-6])
    knot_amplitudes = np.array([0.0, 800.0, 800.0, -400.0, 0.0])
    times = _centres(round(knot_times[-1] / RASTER))
    waveform = np.interp(times, knot_times, knot_amplitudes)

    got_times, got_waveform = restore_additional_shape_samples(
        times, waveform, 0.0, 0.0, RASTER
    )

    for t, a in zip(knot_times, knot_amplitudes, strict=False):
        assert float(np.interp(t, got_times, got_waveform)) == pytest.approx(
            a, abs=1e-6
        )


def test_the_restored_form_is_lossless_where_it_decimates():
    """Dropping interior samples of straight segments must not move the waveform."""
    for name, times, waveform, first, last in CASES:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            got_times, got_waveform = restore_additional_shape_samples(
                times, waveform, first, last, RASTER
            )
        # Every original centre sample still lies on the returned polyline.
        resampled = np.interp(times, got_times, got_waveform)
        np.testing.assert_allclose(
            resampled,
            waveform,
            rtol=0,
            atol=1e-6 * max(np.max(np.abs(waveform)), 1.0),
            err_msg=f"{name}: decimation moved the waveform",
        )


def test_a_mismatched_shape_is_refused():
    with pytest.raises(ValueError, match="sample times"):
        restore_additional_shape_samples(_centres(5), np.zeros(4), 0.0, 0.0, RASTER)
    with pytest.raises(ValueError, match="0 samples"):
        restore_additional_shape_samples(np.empty(0), np.empty(0), 0.0, 0.0, RASTER)


def test_the_thresholds_are_pinned():
    """The thresholds decide which branch runs and what decimates. They are
    named so they can be found, and pinned so tuning them is a decision."""
    assert RESTORE_TOLERANCE == 2e-5
    assert COLLINEAR_TOLERANCE == 1e-8
