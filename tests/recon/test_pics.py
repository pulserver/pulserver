"""The pics reconstruction's per-image steps, against their definitions."""

import numpy as np
import pytest

from pulserver.recon.handlers.pics import partial_fourier


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
