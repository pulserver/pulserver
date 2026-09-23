"""The spectral statistics of each RF definition, as the IR cache carries them."""

import numpy as np
import pypulseqpp as pp
import pytest

from pulserver.ir import convert, summary

SYSTEM = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")


def _pulses():
    sinc = pp.make_sinc_pulse(
        np.pi / 6, duration=2e-3, time_bw_product=4, system=SYSTEM
    )
    sms, _, _ = pp.make_sms_pulse(sinc, 3, 5000.0)
    hard = pp.make_block_pulse(np.pi / 2, duration=1e-3, system=SYSTEM)
    return sinc, sms, hard


def _written(tmp_path, pulses, repetitions=2):
    seq = pp.Sequence(SYSTEM)
    for _ in range(repetitions):
        for pulse in pulses:
            seq.add_block(pulse)
            seq.add_block(pp.make_delay(5e-3))
    path = tmp_path / "rf.seq"
    seq.write(path)
    return path


def _expected(pulse):
    measured = pp.calc_rf_bandwidth(pulse, dt=SYSTEM.rf_raster_time, compat=False)
    return (
        measured.bandwidth,
        measured.num_bands,
        list(measured.band_offsets),
        float(measured.band_bandwidths.max()),
    )


def _reported(entry):
    return (
        entry["bandwidth_hz"],
        entry["num_bands"],
        entry["band_freq_offsets_hz"],
        entry["band_bandwidth_hz"],
    )


def _assert_same(reported, expected):
    assert len(reported) == len(expected)
    for (bw, bands, offsets, widest), (e_bw, e_bands, e_offsets, e_widest) in zip(
        sorted(reported), sorted(expected), strict=True
    ):
        assert bw == pytest.approx(e_bw, rel=1e-5)
        assert bands == e_bands
        np.testing.assert_allclose(offsets, e_offsets, atol=1e-2)
        assert widest == pytest.approx(e_widest, rel=1e-5)


@pytest.mark.parametrize("from_cache", [False, True], ids=["chain", "cache"])
def test_each_rf_definition_carries_the_spectrum_pypulseqpp_measures(
    tmp_path, from_cache
):
    pulses = _pulses()
    path = _written(tmp_path, pulses)
    if from_cache:
        convert(path, SYSTEM)
    report = summary(path, SYSTEM, cache_ext=".pseg" if from_cache else None)

    (unit,) = report["subsequences"]
    _assert_same([_reported(e) for e in unit["rf"]], [_expected(p) for p in pulses])


def test_a_multiband_pulse_is_cached_with_each_band_and_the_width_of_one(tmp_path):
    sinc, sms, _ = _pulses()
    report = summary(_written(tmp_path, [sms]), SYSTEM)

    (entry,) = report["subsequences"][0]["rf"]

    assert entry["num_bands"] == 3
    np.testing.assert_allclose(entry["band_freq_offsets_hz"], [-5000, 0, 5000], atol=10)
    assert entry["band_bandwidth_hz"] == pytest.approx(
        pp.calc_rf_bandwidth(sinc), rel=0.02
    )
    assert entry["bandwidth_hz"] > 2 * 5000


def test_amplitude_and_frequency_offset_leave_a_definitions_spectrum_alone(tmp_path):
    sinc, _, _ = _pulses()
    scaled = pp.make_sinc_pulse(
        np.pi / 3, duration=2e-3, time_bw_product=4, system=SYSTEM, freq_offset=800.0
    )
    report = summary(_written(tmp_path, [sinc, scaled]), SYSTEM)

    bandwidths = {round(e["bandwidth_hz"], 3) for e in report["subsequences"][0]["rf"]}

    assert bandwidths == {round(pp.calc_rf_bandwidth(sinc), 3)}
