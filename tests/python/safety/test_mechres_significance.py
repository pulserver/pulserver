"""Corpus verdicts of the A_eq mechanical-resonance criterion.

At each TR-harmonic line inside a guarded forbidden band the engine reports
the equivalent-sustained gradient amplitude A_eq = (2/T_TR)|S_ax(f_L)| and
flags iff it exceeds the readout-scale floor eps = max(vendor_limit, k*G_max)
(k=0.08). Two properties are held here, on the corpus, against the C engine
itself (no Python wrapper in between):

* a broadband transient (the 32x32 GRE PE-blip) whose sustained in-band
  drive is negligible must NOT trip a zero-tolerance band, and neither must
  scans whose readout combs fall outside the bands;
* a genuine sustained readout comb landing inside a band (bSSFP, the EPI
  train) MUST be flagged.
"""

import pytest
from pulserver._ext.pulseg import _check_consistency, _check_safety
from pulserver.pypulseq import Opts

from .conftest import CORPUS, EXPECTED, build_collection

#: A plausible zero-tolerance mechanical-resonance lockout: a forbidden band
#: in the low-kHz range a gradient coil could resonate in. Synthetic -- no
#: vendor table ships with (or is quoted by) this repository. The band
#: brackets the corpus readout combs (bSSFP ~1176 Hz, EPI ~1158 Hz), so the
#: true-positive cases below land inside it.
SYNTHETIC_BANDS = [
    (1100.0, 1250.0, 0.0),
]


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
    root = EXPECTED if (EXPECTED / name).exists() else CORPUS
    collection = build_collection(root / name, system)
    _check_consistency(collection)
    _check_safety(collection, forbidden_bands=bands, skip_pns=True)


def test_gre32_pe_blip_ghost_passes():
    """The 32x32 GRE PE-blip transient must NOT trip a zero-tolerance band:
    its sustained in-band A_eq is far below the readout-scale floor."""
    _check("gre_32x32_pe_blip.seq", 4e-6, SYNTHETIC_BANDS)  # must not raise


@pytest.mark.parametrize(
    "name", ["gre_2d.seq", "gre_2d_3sl.seq", "fse_2d.seq", "mprage_3d.seq"]
)
def test_representative_fixtures_pass(name):
    """No sustained readout-scale gradient drive inside the forbidden bands.
    mprage (GRE-family + IR dead delay) belongs here: it drives the bands
    less than gre -- the corpus verdict is PASS."""
    _check(name, 20e-6, SYNTHETIC_BANDS)  # must not raise


STATED_BANDS = [(b[0], b[1], 0.008 * 42.576e6, *b[3:]) for b in SYNTHETIC_BANDS]


@pytest.mark.parametrize("name", ["bssfp_2d.seq", "epi_2d_main.seq"])
def test_real_readout_combs_are_refused_by_a_tolerance_they_exceed(name):
    """A genuine sustained readout comb inside the band: the EPI train's
    tooth and the bSSFP's harmonic. A band that states an 8 mT/m plateau
    refuses both. A zero band refuses exactly the one whose reading exceeds
    the floor, which the drawn verdict reports and the gate must match."""
    import pulserver.pypulseq as pp

    with pytest.raises(RuntimeError, match="mech"):
        _check(name, 20e-6, STATED_BANDS)
    system = pp.Opts(
        max_grad=50.0, grad_unit="mT/m", max_slew=350.0, slew_unit="T/m/s", B0=3.0
    )
    seq = pp.Sequence(system=system)
    seq.read((EXPECTED if (EXPECTED / name).exists() else CORPUS) / name)
    drawn = seq.calculate_gradient_spectrum(
        plot=False,
        max_frequency=3000.0,
        tr="worst_case",
        resonance_lines=True,
        bands=SYNTHETIC_BANDS,
    )[4]
    if drawn.ok:
        _check(name, 20e-6, SYNTHETIC_BANDS)
    else:
        with pytest.raises(RuntimeError, match="mech"):
            _check(name, 20e-6, SYNTHETIC_BANDS)
