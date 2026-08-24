"""The EPI reconstruction's phase-correction preprocessing.

The estimator must recover a known phase from a synthetic blip-nulled
navigator, and correcting a reversed line with it must undo both the sample
reversal and the phase. First order is the gradient-delay ramp every product
reconstruction corrects; the order is a parameter because an eddy current
leaves more than a ramp, and a first-order fit cannot see it.
"""

from __future__ import annotations

import numpy as np
import pytest

from pulserver.mrd import correct_lines, estimate_epi_phase

N = 64
COILS = 3
SLOPE = 0.05
INTERCEPT = 0.3
CURVATURE = 0.4


def _row():
    """A smooth complex profile with off-centre structure."""
    x = np.linspace(-1, 1, N)
    profile = np.exp(-((x - 0.2) ** 2) / 0.1) + 0.5 * np.exp(-((x + 0.4) ** 2) / 0.05)
    coils = np.stack([profile * (c + 1) for c in range(COILS)]).astype(np.complex128)
    return np.fft.fftshift(
        np.fft.fft(np.fft.ifftshift(coils, axes=-1), axis=-1, norm="ortho"), axes=-1
    )


def _ramp(curvature=0.0):
    """The phase a backwards readout carries, over the readout samples."""
    samples = np.arange(N)
    coordinate = np.linspace(-1.0, 1.0, N)
    return SLOPE * samples + INTERCEPT + curvature * coordinate**2


def _with_phase(row, curvature=0.0):
    """The row as a backwards readout sees it: hybrid-space phase, then flip."""
    hybrid = np.fft.fftshift(
        np.fft.ifft(np.fft.ifftshift(row, axes=-1), axis=-1, norm="ortho"), axes=-1
    )
    corrupted = hybrid * np.exp(-1j * _ramp(curvature))
    return np.fft.fftshift(
        np.fft.fft(np.fft.ifftshift(corrupted, axes=-1), axis=-1, norm="ortho"),
        axes=-1,
    )[..., ::-1]


def _fitted(phase):
    """The phase the coefficients describe, over the readout samples."""
    return np.polynomial.polynomial.polyval(np.linspace(-1.0, 1.0, N), phase)


def test_the_first_order_fit_recovers_the_gradient_delay_ramp():
    clean = _row()
    backwards = _with_phase(clean)[..., ::-1]  # flipped back, phase left in
    fitted = _fitted(estimate_epi_phase([clean, backwards, clean]))
    # Compared where the profile has signal: the fit is weighted, and the
    # readout's empty edges carry no phase to recover.
    core = slice(N // 4, 3 * N // 4)
    assert np.allclose(fitted[core], _ramp()[core], atol=5e-2)


def test_correcting_a_reversed_line_restores_it():
    clean = _row()
    reversed_line = _with_phase(clean)
    phase = estimate_epi_phase([clean, reversed_line[..., ::-1], clean])
    (restored,) = correct_lines([(reversed_line, True)], phase)
    scale = np.vdot(restored, clean) / np.vdot(restored, restored)
    error = np.linalg.norm(scale * restored - clean) / np.linalg.norm(clean)
    assert error < 5e-2


def test_a_forward_line_passes_through_untouched():
    clean = _row()
    (passed,) = correct_lines([(clean, False)], estimate_epi_phase([clean] * 3))
    assert np.allclose(passed, clean)


def test_a_reversed_line_is_flipped_even_before_a_navigator_has_arrived():
    """The flip is what the REV flag says; the phase is what the navigator adds."""
    clean = _row()
    (flipped,) = correct_lines([(clean, True)], None)
    assert np.allclose(flipped, clean[..., ::-1])


def test_raising_the_order_catches_what_a_ramp_cannot():
    """A curved phase is what an eddy current leaves beyond the delay: the
    first-order fit misses it by construction, and the second-order one does not."""
    clean = _row()
    curved = _with_phase(clean, curvature=CURVATURE)
    navigator = [clean, curved[..., ::-1], clean]

    errors = {}
    for order in (1, 2):
        phase = estimate_epi_phase(navigator, polynomial_order=order)
        (restored,) = correct_lines([(curved, True)], phase)
        scale = np.vdot(restored, clean) / np.vdot(restored, restored)
        errors[order] = float(
            np.linalg.norm(scale * restored - clean) / np.linalg.norm(clean)
        )

    assert errors[2] < 0.5 * errors[1]


def test_the_navigator_is_three_lines():
    with pytest.raises(ValueError, match="three lines"):
        estimate_epi_phase([_row(), _row()])
    with pytest.raises(ValueError, match="non-negative"):
        estimate_epi_phase([_row()] * 3, polynomial_order=-1)


def test_a_fit_applies_to_a_readout_of_another_length():
    """The coefficients are in a coordinate across the readout, not in samples,
    so a navigator and the lines it corrects need not be the same width."""
    phase = estimate_epi_phase([_row(), _with_phase(_row())[..., ::-1], _row()])
    (corrected,) = correct_lines([(np.ones((COILS, 2 * N), np.complex64), True)], phase)
    assert corrected.shape == (COILS, 2 * N)
