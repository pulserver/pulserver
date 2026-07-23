"""Unit tests for pulserver.excitation (incl. Phase-10 additions)."""

from __future__ import annotations

import numpy as np
import pytest

pp = pytest.importorskip("pypulseq")

from pulserver.design._rf import _excitation_helpers as excitation


def _flip_deg_from_integral(rf) -> float:
    return np.rad2deg(2 * np.pi * np.abs(np.trapz(rf.signal, rf.t)))


@pytest.mark.parametrize("flip", [30.0, 90.0, 180.0])
def test_nonselective_flip_integral(flip: float) -> None:
    rf = excitation.nonselective(pp.Opts(), flip)
    assert _flip_deg_from_integral(rf) == pytest.approx(flip, rel=0.01)


@pytest.mark.parametrize("flip", [12.0, 90.0])
def test_slice_selective_flip_integral(flip: float) -> None:
    rf, gz, gz_reph = excitation.slice_selective(pp.Opts(), flip, 5e-3)
    assert _flip_deg_from_integral(rf) == pytest.approx(flip, rel=0.01)
    assert gz.channel == "z"
    assert np.sign(gz_reph.area) == -np.sign(gz.area)


def test_frequency_selective_flip_and_offset() -> None:
    rf = excitation.frequency_selective(pp.Opts(), 90.0, 200.0, center_hz=-440.0)
    assert _flip_deg_from_integral(rf) == pytest.approx(90.0, rel=0.01)
    assert rf.freq_offset == pytest.approx(-440.0)
    # duration = tbw / bandwidth
    assert rf.t[-1] == pytest.approx(4.0 / 200.0, rel=0.01)


def test_frequency_selective_rejects_bad_args() -> None:
    with pytest.raises(ValueError):
        excitation.frequency_selective(pp.Opts(), 90.0, 0.0)
    with pytest.raises(ValueError):
        excitation.frequency_selective(pp.Opts(), 90.0, 200.0, n_bands=3)


def _count_bands(signal: np.ndarray) -> int:
    profile = np.abs(np.fft.fftshift(np.fft.fft(signal, 16 * signal.size)))
    above = profile > 0.5 * profile.max()
    return int(np.sum(np.diff(above.astype(int)) == 1) + (1 if above[0] else 0))


@pytest.mark.parametrize("n_bands", [2, 3, 5])
def test_multiband_profile_has_expected_bands(n_bands: int) -> None:
    rf, _, _ = excitation.slice_selective(pp.Opts(), 90.0, 5e-3)
    assert _count_bands(np.asarray(rf.signal)) == 1
    rf_mb = excitation.multiband(rf, n_bands, band_separation=16.0)
    assert _count_bands(np.asarray(rf_mb.signal)) == n_bands


def test_multiband_preserves_base_pulse() -> None:
    rf, _, _ = excitation.slice_selective(pp.Opts(), 90.0, 5e-3)
    before = np.array(rf.signal, copy=True)
    excitation.multiband(rf, 3, band_separation=12.0)
    assert np.array_equal(rf.signal, before)


def test_multiband_phase_tables() -> None:
    assert np.allclose(excitation.multiband_phases(3, "wong"), [0.0, 0.73, 4.602])
    assert np.allclose(excitation.multiband_phases(4, "malik"), [0.0, np.pi, np.pi, 0.0])
    quad = excitation.multiband_phases(4, "quadratic")
    assert quad[0] == quad[-1] and quad[1] == quad[2]
    assert np.all(excitation.multiband_phases(5, "none") == 0.0)
    with pytest.raises(ValueError):
        excitation.multiband_phases(2, "wong")
    with pytest.raises(ValueError):
        excitation.multiband_phases(3, "malik")
    with pytest.raises(ValueError):
        excitation.multiband_phases(3, "bogus")


def test_frequency_selective_multiband() -> None:
    rf = excitation.frequency_selective(
        pp.Opts(), 90.0, 200.0, n_bands=2, band_separation_hz=1000.0
    )
    assert _count_bands(np.asarray(rf.signal)) == 2
