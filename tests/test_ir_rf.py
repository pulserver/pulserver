"""The spectral statistics of each RF definition, as the IR cache carries them."""

import numpy as np
import pypulseqpp as pp
import pytest

from pulserver.ir import convert, play, summary

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


@pytest.mark.parametrize("channels", [1, 2, 3])
def test_a_dynamic_ptx_pulse_is_cached_with_as_many_channels_as_it_drives(
    tmp_path, channels
):
    samples = np.hanning(100) * 200.0
    signal = np.stack([samples * np.exp(0.3j * channel) for channel in range(channels)])
    pulse = pp.make_ptx_pulse(signal, dwell=1e-5, system=SYSTEM, use="excitation")
    path = _written(tmp_path, [pulse])
    convert(path, SYSTEM)
    played = play(path)
    assert set(played["rf_channels"][played["rf_amp_hz"] != 0]) == {channels}


@pytest.mark.parametrize("from_cache", [False, True], ids=["chain", "cache"])
def test_each_rf_definition_carries_the_flip_angle_pypulseqpp_gives_it(
    tmp_path, from_cache
):
    path = _written(tmp_path, _pulses())
    if from_cache:
        convert(path, SYSTEM)
    report = summary(path, SYSTEM, cache_ext=".pseg" if from_cache else None)
    reported = [entry["flip_angle_deg"] for entry in report["subsequences"][0]["rf"]]
    expected = sorted(set(pp.io.read(path).rf_flip_angles()))
    assert sorted(reported) == pytest.approx(expected, rel=1e-5)


def test_a_dynamic_ptx_pulse_flips_by_the_coherent_sum_of_its_channels(tmp_path):
    """Channels played in antiphase cancel: the flip is not the root sum of squares."""
    samples = np.hanning(100) * 200.0
    signal = np.stack([samples, -samples]).astype(complex)
    pulse = pp.make_ptx_pulse(signal, dwell=1e-5, system=SYSTEM, use="excitation")
    path = _written(tmp_path, [pulse])
    (entry,) = summary(path, SYSTEM)["subsequences"][0]["rf"]
    assert entry["flip_angle_deg"] == pytest.approx(0.0, abs=1e-6)


def test_an_unlabelled_pulse_is_cached_with_the_use_pypulseqpp_detects(tmp_path):
    """Past 90 degrees pypulseqpp calls it refocusing, where the cache alone would not."""
    pulse = pp.make_block_pulse(np.deg2rad(120.0), duration=1e-3, system=SYSTEM)
    path = _written(tmp_path, [pulse], repetitions=1)
    assert pp.io.read(path).libraries().rf_use == ("undefined",)
    convert(path, SYSTEM)
    played = play(path)
    assert set(played["rf_use"][played["rf_amp_hz"] != 0]) == {2}


def _b1sq_integral(pulse, channels=1):
    """pypulseqpp's energy of a pulse over the square of its root-sum-of-squares peak, in s."""
    energy, _, _ = pp.calc_rf_power(pulse, dt=SYSTEM.rf_raster_time)
    per_channel = np.abs(np.asarray(pulse.signal)).reshape(channels, -1)
    return energy / float(np.max(np.sum(per_channel**2, axis=0)))


@pytest.mark.parametrize("from_cache", [False, True], ids=["chain", "cache"])
def test_each_rf_definition_carries_the_energy_pypulseqpp_measures(
    tmp_path, from_cache
):
    pulses = _pulses()
    path = _written(tmp_path, pulses)
    if from_cache:
        convert(path, SYSTEM)
    report = summary(path, SYSTEM, cache_ext=".pseg" if from_cache else None)
    reported = sorted(e["b1sq_integral_s"] for e in report["subsequences"][0]["rf"])
    expected = sorted(_b1sq_integral(pulse) for pulse in pulses)
    assert reported == pytest.approx(expected, rel=1e-5)


def test_a_hard_pulse_at_unit_peak_integrates_to_its_duration(tmp_path):
    _, _, hard = _pulses()
    (entry,) = summary(_written(tmp_path, [hard]), SYSTEM)["subsequences"][0]["rf"]
    assert entry["b1sq_integral_s"] == pytest.approx(1e-3, rel=1e-6)


def test_a_dynamic_ptx_pulse_integrates_the_power_of_every_channel(tmp_path):
    samples = np.hanning(100) * 200.0
    signal = np.stack([samples, 0.5 * samples * np.exp(0.3j)])
    pulse = pp.make_ptx_pulse(signal, dwell=1e-5, system=SYSTEM, use="excitation")
    (entry,) = summary(_written(tmp_path, [pulse]), SYSTEM)["subsequences"][0]["rf"]
    assert entry["b1sq_integral_s"] == pytest.approx(
        _b1sq_integral(pulse, channels=2), rel=1e-5
    )
