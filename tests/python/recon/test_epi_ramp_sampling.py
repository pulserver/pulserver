"""Ramp sampling, and the regridding an EPI reconstruction owes it.

A train worth playing samples across its read ramps rather than idling through
them, so k advances slowly while the gradient is still rising and quickly on the
plateau. Placing those samples on the grid as if they were evenly spaced is a
geometric distortion along the readout; what the reconstruction owes is to put
them where the trajectory says they landed.

The positions here are built the way a gradient makes them -- the running
integral of a trapezoid -- rather than read from the code under test, so what
is pinned is the physics and not a shared derivation.
"""

from __future__ import annotations

import numpy as np
import pytest

from pulserver.recon import epi_ramp_operator

N = 128
#: The object occupies half the digitised readout, which is what the twofold
#: readout oversampling of a real scan buys -- and what leaves the samples
#: outnumbering the pixels they have to determine.
SUPPORT = N // 2
RAMP_UP = 120e-6
FLAT_TOP = 200e-6
RAMP_DOWN = 120e-6
DWELL = (RAMP_UP + FLAT_TOP + RAMP_DOWN) / N


def _trapezoid_positions():
    """Where a readout across a trapezoid's ramps takes its samples, in k.

    k is the running integral of the gradient, so the position of a sample is
    the area swept by the time it is taken -- quadratic on the ramps, linear on
    the plateau. Normalised so the whole lobe sweeps ``[-0.5, 0.5]``.
    """
    times = (np.arange(N) + 0.5) * DWELL
    gradient = np.piecewise(
        times,
        [
            times <= RAMP_UP,
            (times > RAMP_UP) & (times <= RAMP_UP + FLAT_TOP),
            times > RAMP_UP + FLAT_TOP,
        ],
        [
            lambda t: t / RAMP_UP,
            1.0,
            lambda t: 1.0 - (t - RAMP_UP - FLAT_TOP) / RAMP_DOWN,
        ],
    )
    swept = np.cumsum(gradient) * DWELL
    total = 0.5 * RAMP_UP + FLAT_TOP + 0.5 * RAMP_DOWN
    sampled = swept / total - 0.5
    return sampled, np.linspace(sampled[0], sampled[-1], N)


def _object():
    """Hard-edged bars, which a readout distortion visibly smears."""
    image = np.zeros(SUPPORT)
    image[12:20] = 1.0
    image[30:32] = 1.0
    image[44:56] = 0.6
    return image


def _sampled_at(positions, image):
    """The signal a readout taking its samples at ``positions`` in k measures."""
    coordinate = np.arange(SUPPORT) - SUPPORT // 2
    return np.exp(-2j * np.pi * np.outer(positions, coordinate)) @ image


def _reconstruct(line):
    return np.abs(np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(line))))


def test_k_advances_more_slowly_on_the_ramps_than_on_the_plateau():
    """The distortion the correction exists for, in the positions themselves."""
    sampled, uniform = _trapezoid_positions()
    steps = np.diff(sampled)
    assert np.all(steps > 0)
    assert steps[0] < 0.2 * steps[N // 2]
    assert np.allclose(np.diff(uniform), np.diff(uniform)[0])


def test_resampling_recovers_what_uniform_placement_distorts():
    """The measurement is the same either way; what differs is whether the
    reconstruction knows where the samples were taken.

    Exactly, not approximately: while the samples outnumber the pixels they have
    to determine, moving them onto the grid is a change of basis.
    """
    image = _object()
    sampled, uniform = _trapezoid_positions()
    measured = _sampled_at(sampled, image)[None]
    operator = epi_ramp_operator(sampled, uniform, SUPPORT)

    truth = _reconstruct(_sampled_at(uniform, image))

    def error(estimate):
        estimate = estimate / estimate.max()
        reference = truth / truth.max()
        return float(np.linalg.norm(estimate - reference) / np.linalg.norm(reference))

    assert error(_reconstruct(measured[0])) > 0.5  # placed as if uniform
    assert error(_reconstruct((measured @ operator.T)[0])) < 1e-3


def test_resampling_beats_the_linear_interpolation_it_replaces():
    """The stand-in every ramp regridder starts with, on the same readout."""
    image = _object()
    sampled, uniform = _trapezoid_positions()
    measured = _sampled_at(sampled, image)
    truth = _sampled_at(uniform, image)

    linear = np.interp(uniform, sampled, measured.real) + 1j * np.interp(
        uniform, sampled, measured.imag
    )
    exact = measured @ epi_ramp_operator(sampled, uniform, SUPPORT).T

    def error(estimate):
        return float(np.linalg.norm(estimate - truth) / np.linalg.norm(truth))

    assert error(exact) < 0.05 * error(linear)


def test_a_readout_already_on_the_grid_is_left_where_it_is():
    """A train that waits for its plateau samples uniformly, and a client
    attaches it no trajectory -- but handed one, the operator is the identity."""
    uniform = np.linspace(-0.5, 0.5, N)
    operator = epi_ramp_operator(uniform, uniform, SUPPORT)
    measured = _sampled_at(uniform, _object())
    np.testing.assert_allclose(measured @ operator.T, measured, rtol=1e-4, atol=1e-4)


def test_the_operator_needs_two_position_sets_and_a_support():
    sampled, uniform = _trapezoid_positions()
    with pytest.raises(ValueError, match="describe a readout"):
        epi_ramp_operator(sampled[:1], uniform, SUPPORT)
    with pytest.raises(ValueError, match="support must be positive"):
        epi_ramp_operator(sampled, uniform, 0)
