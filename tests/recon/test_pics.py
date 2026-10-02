"""The pics reconstruction's per-readout and per-image steps, against their definitions."""

from types import SimpleNamespace

import ismrmrd
import numpy as np
import pytest

from pulserver import recon
from pulserver.recon.handlers.pics import RemoveReadoutOversampling, partial_fourier

COLUMNS = 16


def _gadget():
    matrix = SimpleNamespace(matrixSize=SimpleNamespace(x=COLUMNS, y=8, z=1))
    header = SimpleNamespace(
        encoding=[SimpleNamespace(encodedSpace=matrix, reconSpace=matrix)],
        acquisitionSystemInformation=SimpleNamespace(receiverChannels=1),
    )
    gadget = RemoveReadoutOversampling()
    gadget.startup(recon.ReconContext.offline(header))
    return gadget


def _centred(transform, data):
    return np.fft.fftshift(transform(np.fft.ifftshift(data, axes=-1)), axes=-1)


def _echo(samples, centre):
    return SimpleNamespace(center_sample=centre, number_of_samples=samples)


def test_an_oversampled_readout_is_the_centre_of_its_profile():
    generator = np.random.default_rng(0)
    profile = np.zeros((2, 2 * COLUMNS), dtype=complex)
    profile[:, COLUMNS // 2 : 3 * COLUMNS // 2] = generator.standard_normal(
        (2, COLUMNS)
    )
    readout = _centred(np.fft.fft, profile)

    cropped = _gadget()(_echo(2 * COLUMNS, COLUMNS), readout)

    expected = _centred(np.fft.fft, profile[:, COLUMNS // 2 : 3 * COLUMNS // 2])
    np.testing.assert_allclose(cropped, expected, atol=1e-4)


def test_a_partial_echo_keeps_only_its_acquired_part():
    readout = np.ones((1, 24), dtype=complex)

    cropped = _gadget()(_echo(24, 8), readout)

    assert cropped.shape == (1, COLUMNS - round(8 * COLUMNS / 32))


def test_a_readout_at_the_matrix_passes_unchanged():
    readout = np.ones((1, COLUMNS), dtype=complex)

    assert _gadget()(_echo(COLUMNS, COLUMNS // 2), readout) is readout


@pytest.mark.parametrize(
    "flag", [ismrmrd.ACQ_IS_NAVIGATION_DATA, ismrmrd.ACQ_IS_RTFEEDBACK_DATA]
)
def test_a_navigator_readout_passes_the_oversampling_removal_unchanged(flag):
    navigator = _echo(2 * COLUMNS, COLUMNS)
    navigator.flags = 1 << (flag - 1)
    readout = np.ones((1, 2 * COLUMNS), dtype=complex)

    assert _gadget()(navigator, readout) is readout


@pytest.mark.parametrize("high", [False, True])
def test_partial_fourier_is_the_side_missing_more_than_the_undersampling_gap(high):
    sampled = np.zeros((1, 32, 8), dtype=bool)
    sampled[:, 8::2] = True
    sampled[:, 12:20] = True
    if not high:
        sampled = sampled[:, ::-1]

    assert partial_fourier(sampled, 1) == (24 / 32, high)
    assert partial_fourier(sampled, 2) == (1.0, False)


def test_undersampling_alone_is_not_partial_fourier():
    sampled = np.zeros((32, 8), dtype=bool)
    sampled[1::3] = True

    assert partial_fourier(sampled, 0) == (1.0, False)


def _root_sum_of_squares(kspace):
    axes = (1, 2)
    images = np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(kspace, axes=axes)), axes=axes
    )
    return np.sqrt((np.abs(images) ** 2).sum(axis=0))


def test_a_partial_fourier_image_is_closer_to_the_full_one_than_its_zero_filling():
    tools = pytest.importorskip("bartorch.tools")
    from pulserver.recon.handlers.pics import PLUGIN

    kspace = tools.phantom(64, kspace=True, coils=8).reshape(8, 64, 64).numpy()
    partial = kspace.copy()
    partial[:, :16] = 0
    full = _root_sum_of_squares(kspace)

    def error(image):
        image = image * (image * full).sum() / (image * image).sum()
        return np.linalg.norm(image - full) / np.linalg.norm(full)

    completed = PLUGIN.spawn().image(partial, (64, 64), None)

    assert error(completed) < error(_root_sum_of_squares(partial))
