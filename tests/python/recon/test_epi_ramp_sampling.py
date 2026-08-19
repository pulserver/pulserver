"""Ramp sampling, and the regridding an EPI reconstruction owes it.

A train worth playing samples across its read ramps rather than idling through
them, so k advances slowly while the gradient is still rising and quickly on the
plateau. Placing those samples on the grid as if they were evenly spaced is a
geometric distortion along the readout; what the reconstruction owes is to say
where each one landed and resample.
"""

from __future__ import annotations

import numpy as np
import pytest

from pulserver.recon import epi_ramp_operator, epi_ramp_positions

N = 128
#: The object occupies half the digitised readout, which is what the twofold
#: readout oversampling of a real scan buys -- and what leaves the samples
#: outnumbering the pixels they have to determine.
SUPPORT = N // 2
RAMP_UP = 120e-6
FLAT_TOP = 200e-6
RAMP_DOWN = 120e-6
DWELL = (RAMP_UP + FLAT_TOP + RAMP_DOWN) / N


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


def test_k_advances_more_slowly_on_the_ramps_than_on_the_plateau():
    sampled, uniform = epi_ramp_positions(
        N, DWELL, ramp_up=RAMP_UP, flat_top=FLAT_TOP, ramp_down=RAMP_DOWN
    )
    steps = np.diff(sampled)
    assert np.all(steps > 0)
    assert steps[0] < 0.2 * steps[N // 2]
    # The whole lobe sweeps one full k width, centred.
    assert sampled[0] == pytest.approx(-uniform[-1], abs=1e-3)
    assert np.allclose(np.diff(uniform), np.diff(uniform)[0])


def test_a_plateau_only_readout_is_already_on_the_grid():
    """Nothing to regrid when the ADC waits for the flat top."""
    sampled, uniform = epi_ramp_positions(
        N, DWELL, ramp_up=0.0, flat_top=N * DWELL, ramp_down=0.0
    )
    np.testing.assert_allclose(sampled, uniform, atol=1e-12)


def test_resampling_recovers_what_uniform_placement_distorts():
    """The measurement is the same either way; what differs is whether the
    reconstruction knows where the samples were taken.

    Exactly, not approximately: while the samples outnumber the pixels they have
    to determine, moving them onto the grid is a change of basis.
    """
    image = _object()
    sampled, uniform = epi_ramp_positions(
        N, DWELL, ramp_up=RAMP_UP, flat_top=FLAT_TOP, ramp_down=RAMP_DOWN
    )
    measured = _sampled_at(sampled, image)[None]
    operator = epi_ramp_operator(sampled, uniform, SUPPORT)

    def reconstruct(line):
        return np.abs(np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(line))))

    truth = reconstruct(_sampled_at(uniform, image))

    def error(estimate):
        estimate = estimate / estimate.max()
        reference = truth / truth.max()
        return float(np.linalg.norm(estimate - reference) / np.linalg.norm(reference))

    assert error(reconstruct(measured[0])) > 0.5  # placed as if uniform
    assert error(reconstruct((measured @ operator.T)[0])) < 1e-3


def test_resampling_beats_the_linear_interpolation_it_replaces():
    """The stand-in every ramp regridder starts with, on the same readout."""
    image = _object()
    sampled, uniform = epi_ramp_positions(
        N, DWELL, ramp_up=RAMP_UP, flat_top=FLAT_TOP, ramp_down=RAMP_DOWN
    )
    measured = _sampled_at(sampled, image)
    truth = _sampled_at(uniform, image)

    linear = np.interp(uniform, sampled, measured.real) + 1j * np.interp(
        uniform, sampled, measured.imag
    )
    exact = measured @ epi_ramp_operator(sampled, uniform, SUPPORT).T

    def error(estimate):
        return float(np.linalg.norm(estimate - truth) / np.linalg.norm(truth))

    assert error(exact) < 0.05 * error(linear)


def test_the_readout_has_to_fit_the_lobe_it_was_played_on():
    with pytest.raises(ValueError, match="outside the"):
        epi_ramp_positions(N, DWELL, ramp_up=1e-6, flat_top=1e-6, ramp_down=1e-6)
    with pytest.raises(ValueError, match="positive duration"):
        epi_ramp_positions(N, DWELL, ramp_up=0.0, flat_top=0.0, ramp_down=0.0)
    with pytest.raises(ValueError, match="dwell must be positive"):
        epi_ramp_positions(
            N, 0.0, ramp_up=RAMP_UP, flat_top=FLAT_TOP, ramp_down=RAMP_DOWN
        )
