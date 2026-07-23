"""Corpus verdict regression tests for the A_eq mechanical-resonance criterion.

The A_eq criterion (PLAN_mechres_aeq_FINAL.md) reports, at each TR-harmonic line
inside a guarded forbidden band, the equivalent-sustained gradient amplitude
A_eq = (2/T_TR)|S_ax(f_L)|, and flags iff it exceeds the readout-scale floor
eps = max(vendor_limit, k*G_max) (k=0.08). This inherently rejects the on-scanner
FALSE POSITIVE (a 32x32 GRE PE-blip: a broadband transient whose sustained
in-band drive is negligible) without a separate significance heuristic, while
keeping a genuine sustained readout comb (bSSFP) flagged.

Corpus verdicts (ratified 2026-07-13): gre / gre_32x32 / epi / fse / mprage PASS;
only bSSFP FAILs. mprage = GRE-family + a dead IR delay, so it drives the bands
LESS than plain GRE -- the earlier mprage-FAIL was a sharp-comb artifact.

See the ``mechres_plots/`` survey and the ``mechres-aeq-*`` project notes.
The gate is the C engine itself; there is no Python wrapper in between.
"""

import pytest
from pulserver._ext._pulseg_wrapper import _check_consistency, _check_safety
from pulserver.pypulseq import Opts

from .conftest import EXPECTED, build_collection

GAM_HZ_PER_G = 4257.6  # GE EPIC GAM; matches CVInit amp_gcm*GAM*100

# HRMbUHP ESP lockout table (epiespHRMbUHP.dat), all bands zero-tolerance.
# (esp_min_us, esp_max_us, amp_G_per_cm); freq band = (5e5/esp_max, 5e5/esp_min).
_ESP_TRIPLES = [
    (350, 445, 0.0),  # X/Y: 1124-1429 Hz
    (481, 488, 0.0),  # X/Y: 1025-1040 Hz
    (418, 426, 0.0),  # Z:   1174-1196 Hz
]
HRMB_BANDS = [(5.0e5 / esp_max, 5.0e5 / esp_min, amp * GAM_HZ_PER_G * 100.0) for esp_min, esp_max, amp in _ESP_TRIPLES]


def _check(name: str, raster: float, bands) -> None:
    """Run the C consistency + safety gate over a fixture, as predownload does."""
    system = Opts(
        max_grad=50.0,
        grad_unit="mT/m",
        max_slew=350.0,
        slew_unit="T/m/s",
        B0=3.0,
        grad_raster_time=raster,
        block_duration_raster=raster,
    )
    collection = build_collection(EXPECTED / name, system)
    _check_consistency(collection)
    _check_safety(collection, forbidden_bands=bands, skip_pns=True)


def test_gre32_pe_blip_ghost_passes():
    """The 32x32 GRE PE-blip transient must NOT trip a zero-tolerance band:
    its sustained in-band A_eq is far below the readout-scale floor."""
    _check("gre_32x32_pe_blip.seq", 4e-6, HRMB_BANDS)  # must not raise


@pytest.mark.parametrize(
    "name", ["gre_2d_1sl_1avg.seq", "epi_2d_1sl_1avg.seq", "fse_2d_1sl_1avg.seq", "mprage_2d_1sl_1avg.seq"]
)
def test_representative_fixtures_pass(name):
    """No sustained readout-scale gradient drive inside the HRMbUHP bands. mprage
    (GRE-family + IR dead delay) belongs here: it drives the bands less than gre
    -- the ratified corpus verdict is PASS."""
    _check(name, 20e-6, HRMB_BANDS)  # must not raise


@pytest.mark.parametrize("name", ["bssfp_2d_1sl_1avg.seq"])
def test_real_readout_combs_still_flagged(name):
    """A genuine sustained balanced-readout comb harmonic landing in a
    zero-tolerance band MUST still be caught (bSSFP is the corpus true positive:
    A_eq ~= 7.8 mT/m at ~1233 Hz)."""
    with pytest.raises(RuntimeError, match="mech"):
        _check(name, 20e-6, HRMB_BANDS)
