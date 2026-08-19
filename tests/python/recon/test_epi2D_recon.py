"""The EPI reconstruction's phase-correction preprocessing.

The odd/even fit must recover a known gradient-delay ramp from a synthetic
blip-nulled navigator, and correcting a reversed line with it must undo both
the sample reversal and the ramp.
"""

from __future__ import annotations

import numpy as np
import pytest

from pulserver.recon import correct_lines, odd_even_fit

N = 64
COILS = 3
SLOPE = 0.05
INTERCEPT = 0.3


def _row():
    """A smooth complex profile with off-centre structure."""
    x = np.linspace(-1, 1, N)
    profile = np.exp(-((x - 0.2) ** 2) / 0.1) + 0.5 * np.exp(-((x + 0.4) ** 2) / 0.05)
    coils = np.stack([profile * (c + 1) for c in range(COILS)]).astype(np.complex128)
    return np.fft.fftshift(
        np.fft.fft(np.fft.ifftshift(coils, axes=-1), axis=-1, norm="ortho"), axes=-1
    )


def _with_ramp(row):
    """The row as a backwards readout sees it: hybrid-space ramp, then flip."""
    hybrid = np.fft.fftshift(
        np.fft.ifft(np.fft.ifftshift(row, axes=-1), axis=-1, norm="ortho"), axes=-1
    )
    ramp = SLOPE * np.arange(N) + INTERCEPT
    corrupted = hybrid * np.exp(-1j * ramp)
    return np.fft.fftshift(
        np.fft.fft(np.fft.ifftshift(corrupted, axes=-1), axis=-1, norm="ortho"),
        axes=-1,
    )[..., ::-1]


def test_the_fit_recovers_the_ramp():
    clean = _row()
    backwards = _with_ramp(clean)[..., ::-1]  # flipped back, ramp left in
    slope, intercept = odd_even_fit([clean, backwards, clean])
    assert slope == pytest.approx(SLOPE, abs=1e-3)
    assert intercept == pytest.approx(INTERCEPT, abs=5e-2)


def test_correcting_a_reversed_line_restores_it():
    clean = _row()
    reversed_line = _with_ramp(clean)
    slope, intercept = odd_even_fit([clean, reversed_line[..., ::-1], clean])
    (restored,) = correct_lines([(reversed_line, True)], slope, intercept)
    scale = np.vdot(restored, clean) / np.vdot(restored, restored)
    error = np.linalg.norm(scale * restored - clean) / np.linalg.norm(clean)
    assert error < 5e-2


def test_a_forward_line_passes_through_untouched():
    clean = _row()
    (passed,) = correct_lines([(clean, False)], 0.123, 0.456)
    assert np.allclose(passed, clean)
