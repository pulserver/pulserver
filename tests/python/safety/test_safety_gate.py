"""The C safety gate over the generated-sequence corpus.

These are the checks a scanner runs at predownload, asserted against the same
compiled engine. There is no Python re-implementation to drift from.
"""

import numpy as np
import pytest
from pulserver._ext._pulseg_wrapper import (
    _calc_mech_resonances,
    _check_consistency,
    _check_safety,
    _PulseqCollection,
)
from pulserver.pypulseq import Opts

from .conftest import EXPECTED, build_collection

GENERATED_SEQUENCES = [
    "gre_2d_1sl_1avg.seq",
    "gre_2d_3sl_1avg.seq",
    "epi_2d_1sl_1avg.seq",
    "epi_2d_3sl_1avg.seq",
    "fse_2d_1sl_1avg.seq",
    "fse_2d_3sl_1avg.seq",
    "bssfp_2d_1sl_1avg.seq",
    "bssfp_2d_3sl_1avg.seq",
    "mprage_2d_1sl_1avg.seq",
    "mprage_2d_3sl_1avg.seq",
    "mprage_nav_2d_1sl_1avg.seq",
    "mprage_nav_2d_3sl_1avg.seq",
    "mprage_noncart_3d_1sl_1avg_userotext0.seq",
    "mprage_noncart_3d_3sl_1avg_userotext0.seq",
]

REPRESENTATIVE_SEQUENCES = [
    "gre_2d_1sl_1avg.seq",
    "epi_2d_1sl_1avg.seq",
    "fse_2d_1sl_1avg.seq",
    "bssfp_2d_1sl_1avg.seq",
    "mprage_2d_1sl_1avg.seq",
    "mprage_nav_2d_1sl_1avg.seq",
    "mprage_noncart_3d_1sl_1avg_userotext0.seq",
]


def _id(name: str) -> str:
    return name.replace(".seq", "")


def _opts(*, max_grad: float = 50.0, max_slew: float = 350.0, raster: float = 20e-6) -> Opts:
    return Opts(
        max_grad=max_grad,
        grad_unit="mT/m",
        max_slew=max_slew,
        slew_unit="T/m/s",
        B0=3.0,
        grad_raster_time=raster,
        block_duration_raster=raster,
    )


@pytest.fixture(params=GENERATED_SEQUENCES, ids=_id)
def generated_seq_path(request):
    return EXPECTED / request.param


@pytest.fixture(params=REPRESENTATIVE_SEQUENCES, ids=_id)
def representative_seq_path(request):
    return EXPECTED / request.param


# ── the gate ─────────────────────────────────────────────────────────


def test_gate_passes_when_limits_are_above_the_sequence_peaks(generated_seq_path):
    collection = build_collection(generated_seq_path, _opts())
    _check_consistency(collection)
    _check_safety(collection, skip_pns=True)


def test_gate_catches_a_slew_rate_over_the_limit(representative_seq_path):
    # 10 T/m/s against fixtures running 150-200, and the engine applies a
    # 1/sqrt(3) vectorial factor on top.
    collection = build_collection(representative_seq_path, _opts(max_slew=10.0))
    with pytest.raises(RuntimeError, match="slew"):
        _check_safety(collection, skip_pns=True)


def test_gate_catches_a_gradient_over_the_limit(representative_seq_path):
    collection = build_collection(representative_seq_path, _opts(max_grad=1.0))
    with pytest.raises(RuntimeError, match="gmax exceeded"):
        _check_safety(collection, skip_pns=True)


def test_gate_passes_pns_when_the_threshold_cannot_be_reached(representative_seq_path):
    collection = build_collection(representative_seq_path, _opts())
    _check_safety(
        collection,
        stim_threshold=1e9,
        decay_constant_us=360.0,
        pns_threshold_percent=1000.0,
        skip_pns=False,
    )


def test_incompatible_raster_is_rejected_at_load(representative_seq_path):
    # 7 us does not evenly divide the fixtures' 20 us raster.
    with pytest.raises(RuntimeError, match="raster"):
        build_collection(representative_seq_path, _opts(raster=7e-6))


# ── mechanical resonance engine ──────────────────────────────────────


def test_mech_resonance_peak_defaults_match_the_documented_values(generated_seq_path):
    collection = build_collection(generated_seq_path, _opts())
    bands = [(500.0, 600.0, 2000.0)]

    implicit = _calc_mech_resonances(
        collection, subsequence_idx=0, target_resolution_hz=5.0, max_freq_hz=1200.0, forbidden_bands=bands
    )
    explicit = _calc_mech_resonances(
        collection,
        subsequence_idx=0,
        target_resolution_hz=5.0,
        max_freq_hz=1200.0,
        forbidden_bands=bands,
        peak_log10_threshold=2.25,
        peak_norm_scale=10.0,
        peak_eps=1e-30,
    )

    for key in ("spectrum_full_gx", "spectrum_full_gy", "spectrum_full_gz"):
        np.testing.assert_allclose(
            np.asarray(implicit[key], dtype=np.float32),
            np.asarray(explicit[key], dtype=np.float32),
        )


def test_mech_resonance_candidates_pair_a_frequency_with_an_amplitude(generated_seq_path):
    collection = build_collection(generated_seq_path, _opts())
    result = _calc_mech_resonances(
        collection,
        subsequence_idx=0,
        target_resolution_hz=5.0,
        max_freq_hz=1200.0,
        forbidden_bands=[(500.0, 600.0, 2000.0)],
    )

    amplitudes = np.asarray(result["candidate_grad_amps"], dtype=np.float32)
    frequencies = np.asarray(result.get("candidate_freqs", []), dtype=np.float32)
    assert len(amplitudes) == len(frequencies)
    assert np.all(amplitudes >= 0.0)


# ── band units ───────────────────────────────────────────────────────


def test_opts_converts_public_bands_to_backend_units():
    opts = Opts(
        max_grad=40.0, grad_unit="mT/m", max_slew=150.0, slew_unit="T/m/s", forbidden_bands=[(100.0, 200.0, 1.0)]
    )

    ((fmin, fmax, amp),) = opts.forbidden_bands_hz_per_m()
    assert np.isclose(fmin, 100.0)
    assert np.isclose(fmax, 200.0)
    assert np.isclose(amp, 1.0e-3 * float(opts.gamma))


def test_multi_sequence_collections_load_from_payloads():
    payloads = [(EXPECTED / name).read_bytes() for name in ("gre_2d_1sl_1avg.seq", "epi_2d_1sl_1avg.seq")]
    system = _opts()
    collection = _PulseqCollection(
        payloads,
        float(system.gamma),
        float(system.B0),
        float(system.max_grad),
        float(system.max_slew),
        float(system.rf_raster_time),
        float(system.grad_raster_time),
        float(system.adc_raster_time),
        float(system.block_duration_raster),
        True,
        1,
    )
    _check_consistency(collection)
    _check_safety(collection, skip_pns=True)
