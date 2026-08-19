"""Tests for the shared centered Fourier transform."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pulserver.recon import fftc, ifftc
from pulserver.recon._fourier import centered_fftn, resize_centered_axis


def test_centered_fft_matches_the_numpy_reference_on_both_backends():
    """One implementation serves every module, checked against the literal
    ``ifftshift -> fftn(norm="ortho") -> fftshift`` composition."""
    rng = np.random.default_rng(11)
    value = (
        rng.standard_normal((3, 8, 8)) + 1j * rng.standard_normal((3, 8, 8))
    ).astype(np.complex64)
    axes = (-2, -1)
    reference = np.fft.fftshift(
        np.fft.fftn(np.fft.ifftshift(value, axes=axes), axes=axes, norm="ortho"),
        axes=axes,
    )
    np.testing.assert_allclose(
        centered_fftn(value, axes=axes, inverse=False), reference, atol=1e-6
    )
    torch_result = centered_fftn(torch.as_tensor(value), axes=axes, inverse=False)
    np.testing.assert_allclose(torch_result.numpy(), reference, atol=1e-6)


def test_torch_and_numpy_fftc_agree_and_invert():
    rng = np.random.default_rng(7)
    value = (rng.standard_normal((4, 6)) + 1j * rng.standard_normal((4, 6))).astype(
        np.complex64
    )
    spectrum = fftc(value, axes=(-2, -1))
    np.testing.assert_allclose(ifftc(spectrum, axes=(-2, -1)), value, atol=1e-6)
    np.testing.assert_allclose(
        fftc(torch.as_tensor(value), axes=(-2, -1)).numpy(), spectrum, atol=1e-6
    )


def test_resizing_one_axis_pads_and_crops_about_the_centre():
    value = torch.arange(6, dtype=torch.float32).reshape(1, 6)
    padded = resize_centered_axis(value, 8, axis=-1)
    assert padded.shape == (1, 8)
    torch.testing.assert_close(padded[0, 1:7], value[0])
    cropped = resize_centered_axis(value, 4, axis=-1)
    torch.testing.assert_close(cropped[0], value[0, 1:5])
