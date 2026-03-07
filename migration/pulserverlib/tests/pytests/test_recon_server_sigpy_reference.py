"""Unit tests for educational SigPy reference plugin helpers."""

from __future__ import annotations

import numpy as np
import pytest

from pulserver.recon_server.plugins.sigpy_reference import ifft2_sigpy_style


def test_ifft2_sigpy_style_shape_preserved():
    rng = np.random.default_rng(7)
    kspace = rng.standard_normal((8, 6, 1, 4)) + 1j * rng.standard_normal((8, 6, 1, 4))
    out = ifft2_sigpy_style(kspace)
    assert out.shape == kspace.shape


def test_ifft2_sigpy_style_matches_numpy_fallback_when_sigpy_missing(monkeypatch):
    rng = np.random.default_rng(11)
    kspace = rng.standard_normal((10, 5, 1, 2)) + 1j * rng.standard_normal((10, 5, 1, 2))

    # Force import failure path to test deterministic numpy fallback branch.
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == 'sigpy':
            raise ImportError('forced for test')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr('builtins.__import__', fake_import)

    out = ifft2_sigpy_style(kspace)

    ksp = np.transpose(kspace, (3, 2, 1, 0))
    expected = np.transpose(np.fft.ifft2(ksp, axes=(-1, -2), norm='ortho'), (3, 2, 1, 0))
    np.testing.assert_allclose(out, expected)


def test_ifft2_sigpy_style_rejects_non_4d():
    with pytest.raises(ValueError):
        ifft2_sigpy_style(np.zeros((8, 6, 4), dtype=np.complex64))
