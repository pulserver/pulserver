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

from pulserver.recon import epi_ramp_interpolate, epi_ramp_positions

N = 128
RAMP_UP = 120e-6
FLAT_TOP = 200e-6
RAMP_DOWN = 120e-6
DWELL = (RAMP_UP + FLAT_TOP + RAMP_DOWN) / N


def _object():
    """Hard-edged bars, which a readout distortion visibly smears."""
    image = np.zeros(N)
    image[30:40] = 1.0
    image[60:64] = 1.0
    image[90:110] = 0.6
    return image


def _sampled_at(positions, image):
    """The signal a readout taking its samples at ``positions`` in k measures."""
    coordinate = np.arange(N) - N // 2
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


def test_regridding_recovers_what_uniform_placement_distorts():
    """The measurement is the same either way; what differs is whether the
    reconstruction knows where the samples were taken.

    The ramps here are more than half the readout, which is the hard case: the
    uniform grid is denser than the plateau's samples, so the linear
    interpolation leaves something behind. It still removes most of what naive
    placement costs.
    """
    image = _object()
    sampled, uniform = epi_ramp_positions(
        N, DWELL, ramp_up=RAMP_UP, flat_top=FLAT_TOP, ramp_down=RAMP_DOWN
    )
    measured = _sampled_at(sampled, image)[None]

    def reconstruct(line):
        return np.abs(np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(line))))

    naive = reconstruct(measured[0])
    regridded = reconstruct(epi_ramp_interpolate(measured, sampled, uniform)[0])
    truth = reconstruct(_sampled_at(uniform, image))

    def error(estimate):
        estimate = estimate / estimate.max()
        reference = truth / truth.max()
        return float(np.linalg.norm(estimate - reference) / np.linalg.norm(reference))

    assert error(regridded) < 0.4 * error(naive)


def test_the_readout_has_to_fit_the_lobe_it_was_played_on():
    with pytest.raises(ValueError, match="outside the"):
        epi_ramp_positions(N, DWELL, ramp_up=1e-6, flat_top=1e-6, ramp_down=1e-6)
    with pytest.raises(ValueError, match="positive duration"):
        epi_ramp_positions(N, DWELL, ramp_up=0.0, flat_top=0.0, ramp_down=0.0)
    with pytest.raises(ValueError, match="dwell must be positive"):
        epi_ramp_positions(
            N, 0.0, ramp_up=RAMP_UP, flat_top=FLAT_TOP, ramp_down=RAMP_DOWN
        )
