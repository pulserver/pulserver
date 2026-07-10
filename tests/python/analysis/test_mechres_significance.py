"""Regression tests for the cross-axis spectral-significance candidate filter.

Guards the fix for the on-scanner mechanical-resonance FALSE POSITIVE: a 32x32
GRE (short bipolar phase-encode blip) was rejected because the peak-amplitude-
anchored analytical model over-promoted a candidate at ~1361 Hz (a
zero-tolerance forbidden band) even though the blip's true oscillatory power
there is ~1e-3 % of the gradient power.  The significance filter
(SA_FFT_SIGNIFICANCE_FRAC) keeps a candidate only if its dense-FFT amplitude at
its frequency reaches a fraction of the global cross-axis peak — dropping the
transient ghost while leaving genuine readout combs (MPRAGE / bSSFP) flagged.

See the ``mechres_plots/`` survey and the ``mechres-zero-threshold-false-positive``
project note.
"""

from pathlib import Path

import pytest

from pulserver.analysis import Opts, SequenceCollection

EXPECTED = Path(__file__).resolve().parents[2] / "utils" / "expected"

GAM_HZ_PER_G = 4257.6  # GE EPIC GAM; matches CVInit amp_gcm*GAM*100

# HRMbUHP ESP lockout table (epiespHRMbUHP.dat), all bands zero-tolerance.
# (esp_min_us, esp_max_us, amp_G_per_cm); freq band = (5e5/esp_max, 5e5/esp_min).
_ESP_TRIPLES = [
    (350, 445, 0.0),  # X/Y: 1124-1429 Hz
    (481, 488, 0.0),  # X/Y: 1025-1040 Hz
    (418, 426, 0.0),  # Z:   1174-1196 Hz
]
HRMB_BANDS = [
    (5.0e5 / esp_max, 5.0e5 / esp_min, amp * GAM_HZ_PER_G * 100.0)
    for esp_min, esp_max, amp in _ESP_TRIPLES
]


def _sc(name: str, raster: float) -> SequenceCollection:
    sys = Opts(
        max_grad=50.0, max_slew=350.0, B0=3.0,
        grad_raster_time=raster, block_duration_raster=raster,
    )
    return SequenceCollection(str(EXPECTED / name), system=sys)


def test_gre32_pe_blip_ghost_passes():
    """The 32x32 GRE PE-blip transient must NOT trip a zero-tolerance band."""
    sc = _sc("gre_32x32_pe_blip.seq", 4e-6)
    sc.check(forbidden_bands=HRMB_BANDS)  # must not raise


@pytest.mark.parametrize("name", ["gre_2d_1sl_1avg.seq", "epi_2d_1sl_1avg.seq",
                                  "fse_2d_1sl_1avg.seq"])
def test_representative_fixtures_pass(name):
    """These fixtures have no genuine gradient power inside the HRMbUHP bands."""
    _sc(name, 20e-6).check(forbidden_bands=HRMB_BANDS)  # must not raise


@pytest.mark.parametrize("name", ["mprage_2d_1sl_1avg.seq", "bssfp_2d_1sl_1avg.seq"])
def test_real_readout_combs_still_flagged(name):
    """Genuine sustained readout-comb harmonics in a zero-tolerance band MUST
    still be caught (true positives preserved by the significance filter)."""
    with pytest.raises(RuntimeError, match="mech"):
        _sc(name, 20e-6).check(forbidden_bands=HRMB_BANDS)
