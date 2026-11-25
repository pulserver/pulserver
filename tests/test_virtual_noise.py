"""Receive noise of the virtual scanner: channel covariance, reproducibility and the noise scan."""

import numpy as np
import pytest

from pulserver import virtual


def head32_tissue():
    """A water disk received by the 32-channel head array."""
    disk = virtual.Phantom([virtual.Ellipse((0.0, 0.0, 0.0), (0.07, 0.09))])
    return disk.tissue(4e-3, coil=virtual.COILS["head8/head32"])


def test_noise_drawn_over_many_samples_has_the_channel_covariance_it_was_given():
    a = np.array([[1.0, 0.3 + 0.2j, 0.0], [0.3 - 0.2j, 2.0, 0.5], [0.0, 0.5, 0.5]])
    noise = virtual.ReceiveNoise(a, std=1.0, seed=4)

    drawn = noise(0, 200_000)
    measured = drawn @ drawn.conj().T / drawn.shape[1] / 2.0

    np.testing.assert_allclose(measured, a, atol=0.02)


def test_a_readout_receives_the_same_noise_each_time_it_is_drawn_and_another_readout_other_noise():
    noise = virtual.ReceiveNoise(np.eye(2), std=1.0, seed=1)

    np.testing.assert_array_equal(noise(7, 64), noise(7, 64))
    assert not np.allclose(noise(7, 64), noise(8, 64))
    assert not np.allclose(noise.readouts(1, 64)[0], noise(0, 64))


def test_a_covariance_that_is_not_positive_definite_is_refused():
    with pytest.raises(ValueError, match="positive definite"):
        virtual.ReceiveNoise(np.array([[1.0, 2.0], [2.0, 1.0]]), std=1.0)


def test_the_coil_noise_of_each_loop_narrows_the_spread_of_channel_noise_the_sample_alone_gives():
    tissue = head32_tissue()

    sample = virtual.ReceiveNoise.of_tissue(tissue, 1.0, coil_noise=0.0).covariance
    loaded = virtual.ReceiveNoise.of_tissue(tissue, 1.0).covariance

    spread = [np.ptp(np.log(np.real(np.diag(c)))) for c in (sample, loaded)]
    assert spread[1] < spread[0]
    for covariance in (sample, loaded):
        assert np.real(np.trace(covariance)) == pytest.approx(tissue.coils)
        np.testing.assert_allclose(covariance, covariance.conj().T, atol=1e-12)


def test_a_coil_with_a_preamplifier_gain_twice_another_has_four_times_its_noise_variance():
    tissue = head32_tissue()
    gains = np.ones(tissue.coils)
    gains[3] = 2.0

    plain = virtual.ReceiveNoise.of_tissue(tissue, 1.0).covariance
    gained = virtual.ReceiveNoise.of_tissue(tissue, 1.0, gains=gains).covariance

    ratio = np.real(np.diag(gained)) / np.real(np.diag(plain))
    assert ratio[3] / ratio[0] == pytest.approx(4.0)
