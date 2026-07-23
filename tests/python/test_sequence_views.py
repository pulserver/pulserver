"""The inspection views of pulserver.pypulseq.Sequence, and io.read."""

from __future__ import annotations

import matplotlib
import numpy as np
import pulserver.io as pio
import pulserver.pypulseq as ps
import pypulseq as pp
import pytest
from scipy.spatial.transform import Rotation

matplotlib.use("Agg")

SHIM = np.array([1.0, 0.5j, -0.8, 0.3 + 0.3j])
ANGLES = (0.0, 0.4, 0.8, 1.2)


@pytest.fixture
def system():
    return ps.Opts(
        max_grad=30,
        grad_unit="mT/m",
        max_slew=120,
        slew_unit="T/m/s",
        adc_raster_time=1e-7,
    )


@pytest.fixture
def gradient(system):
    return pp.make_trapezoid(channel="x", area=1000, system=system)


def _build(system, gradient, *, rotate=True, shim=True):
    """Four TRs of excitation + readout + recovery delay."""
    seq = ps.Sequence(system)
    rf = pp.make_sinc_pulse(flip_angle=np.pi / 6, duration=1e-3, system=system, use="excitation", return_gz=False)
    adc = pp.make_adc(num_samples=64, duration=gradient.flat_time, delay=gradient.rise_time, system=system)
    for i, angle in enumerate(ANGLES):
        excitation = [rf, ps.make_label("LIN", "SET", i)]
        if shim:
            excitation.append(ps.make_rf_shim(SHIM))
        seq.add_block(*excitation)

        readout = [gradient, adc]
        if rotate:
            readout.append(ps.make_rotation(Rotation.from_euler("z", angle)))
        seq.add_block(*readout)

        seq.add_block(pp.make_delay(5e-3))
    return seq


# ── views ────────────────────────────────────────────────────────────


def test_plotting_view_rotates_gradients_onto_their_played_axes(system, gradient):
    seq = _build(system, gradient, shim=False)

    for tr, angle in enumerate(ANGLES):
        block = seq._seqplot.get_block(3 * tr + 2)
        assert block.gx.amplitude == pytest.approx(gradient.amplitude * np.cos(angle), rel=1e-5)
        if angle:
            assert block.gy.amplitude == pytest.approx(gradient.amplitude * np.sin(angle), rel=1e-5)


def test_plain_view_keeps_the_unrotated_waveform(system, gradient):
    seq = _build(system, gradient, shim=False)

    block = seq._seq.get_block(5)
    assert block.gy is None
    assert block.gx.amplitude == pytest.approx(gradient.amplitude, rel=1e-5)


def test_plotting_view_expands_rf_shims_across_channels(system, gradient):
    seq = _build(system, gradient, rotate=False)

    plain = seq._seq.get_block(1).rf
    shimmed = seq._seqplot.get_block(1).rf
    n = plain.signal.size

    assert shimmed.signal.shape == (SHIM.size * n,)
    assert shimmed.t.shape == (SHIM.size * n,)
    for channel, weight in enumerate(SHIM):
        assert np.allclose(shimmed.signal[channel * n : (channel + 1) * n], weight * plain.signal)
        # Each channel restarts at t=0 so every trace covers the same interval.
        assert np.allclose(shimmed.t[channel * n : (channel + 1) * n], plain.t)


def test_views_are_shared_when_no_extension_needs_resolving(system, gradient):
    seq = _build(system, gradient, rotate=False, shim=False)
    assert seq._seqplot is seq._seq


def test_views_are_rebuilt_after_more_blocks_are_appended(system, gradient):
    seq = _build(system, gradient, rotate=False, shim=False)
    assert len(seq._seq.block_events) == 12

    seq.add_block(pp.make_delay(1e-3))
    assert len(seq._seq.block_events) == 13


def test_views_share_the_sequence_system(system, gradient):
    seq = _build(system, gradient)
    assert seq._seqplot.system is system
    assert seq.segments[0].to_sequence().system is system


# ── TR and segment structure ─────────────────────────────────────────


def test_tr_structure_comes_from_the_c_backend(system, gradient):
    seq = _build(system, gradient)

    assert seq.num_trs == len(ANGLES)
    assert seq.tr_info["tr_size"] == 3
    assert seq.tr_block_range(0) == (1, 3)
    assert seq.tr_block_range(1) == (4, 6)
    assert seq.tr_block_range(-1) == (10, 12)


def test_tr_index_out_of_range_is_rejected(system, gradient):
    seq = _build(system, gradient)
    with pytest.raises(IndexError):
        seq.tr_block_range(len(ANGLES))


def test_tr_time_range_matches_the_block_durations(system, gradient):
    seq = _build(system, gradient)

    start, stop = seq._time_range(1, None)
    durations = seq._seqplot.block_durations
    assert start == pytest.approx(sum(durations[i] for i in (1, 2, 3)))
    assert stop - start == pytest.approx(sum(durations[i] for i in (4, 5, 6)))


def test_segments_cover_the_blocks_of_their_max_energy_instance(system, gradient):
    seq = _build(system, gradient)

    segments = seq.segments
    assert [s.num_blocks for s in segments] == [2, 1]
    assert all(s.from_max_energy_instance for s in segments)
    assert segments[1].pure_delay

    sliced = segments[0].to_sequence()
    assert len(sliced.block_events) == segments[0].num_blocks


def test_scope_arguments_are_mutually_exclusive(system, gradient):
    seq = _build(system, gradient)
    with pytest.raises(ValueError, match="not both"):
        seq._scope(tr_index=0, segment=0)


# ── analysis entry points ────────────────────────────────────────────


def test_plot_and_kspace_accept_a_tr_index(system, gradient):
    seq = _build(system, gradient)

    seq.plot(tr_index=0, plot_now=False)
    seq.plot_kspace(tr_index=0, plot_now=False)

    scoped = seq.calculate_kspace(tr_index=0)[0]
    full = seq.calculate_kspace()[0]
    assert scoped.shape[1] * len(ANGLES) == full.shape[1]


def test_grad_spectrum_defaults_to_a_single_window_per_tr(system, gradient):
    seq = _build(system, gradient)

    _, _, _, times = seq.grad_spectrum(tr_index=0, bands=[(500, 700, 0.0, "gx")], plot=False)
    assert times.size == 1


def test_pns_chronaxie_scales_inversely_with_rheobase(system, gradient):
    seq = _build(system, gradient)

    strict, _ = seq.pns(tr_index=0, chronaxie_us=360.0, rheobase=4.25e8, plot=False)
    lax, _ = seq.pns(tr_index=0, chronaxie_us=360.0, rheobase=8.5e8, plot=False)

    assert strict.shape[1] == 3
    assert np.allclose(strict, 2.0 * lax)


def test_pns_needs_model_parameters(system, gradient):
    seq = _build(system, gradient)
    with pytest.raises(ValueError, match="chronaxie_us and rheobase"):
        seq.pns(plot=False)


def test_pns_rejects_an_unknown_model(system, gradient):
    seq = _build(system, gradient)
    with pytest.raises(ValueError, match="Unknown PNS model"):
        seq.pns(model="safest", plot=False)


# ── io.read ──────────────────────────────────────────────────────────


def test_read_round_trips_events_and_extensions(system, gradient, tmp_path):
    seq = _build(system, gradient)
    path = tmp_path / "roundtrip.seq"
    pio.write(seq, output=path)

    restored = pio.read(path)

    assert len(restored._seq.block_events) == len(seq._seq.block_events)
    assert restored.extensions.keys() == seq.extensions.keys()
    assert np.allclose(restored.extensions[5].rotation, seq.extensions[5].rotation)
    assert np.allclose(restored.extensions[1].rf_shim, SHIM)
    assert restored.extensions[1].label_sets == [("LIN", 0.0)]


def test_read_recovers_raster_times_from_the_file(system, gradient, tmp_path):
    seq = _build(system, gradient)
    path = tmp_path / "rasters.seq"
    pio.write(seq, output=path)

    restored = pio.read(path)

    assert restored.system.adc_raster_time == pytest.approx(system.adc_raster_time)
    assert restored.system.grad_raster_time == pytest.approx(system.grad_raster_time)
    assert restored.system.rf_raster_time == pytest.approx(system.rf_raster_time)


def test_read_arrives_with_its_views_already_built(system, gradient, tmp_path):
    seq = _build(system, gradient)
    path = tmp_path / "views.seq"
    pio.write(seq, output=path)

    restored = pio.read(path)

    assert restored._view_cache is not None
    block = restored._seqplot.get_block(5)
    assert block.gy.amplitude == pytest.approx(gradient.amplitude * np.sin(ANGLES[1]), rel=1e-5)


def test_read_accepts_a_payload_and_survives_a_second_write(system, gradient):
    seq = _build(system, gradient)
    payload = pio.write(seq, output=None)

    restored = pio.read(payload)
    again = pio.read(pio.write(restored, output=None))

    assert again.extensions.keys() == seq.extensions.keys()
    assert np.allclose(again.extensions[5].rotation, seq.extensions[5].rotation)


# ── ESP lockout tables ───────────────────────────────────────────────


ESP_TABLE = """\
# gx
2
410 510 0.0
816 865 1.6
# gy
1
420 459 0.0
# gz
0
"""


def test_read_esp_bands_converts_echo_spacings_to_frequency(tmp_path):
    path = tmp_path / "epiesp.dat"
    path.write_text(ESP_TABLE)

    bands = pio.read_esp_bands(path)

    assert [b[3] for b in bands] == ["gx", "gx", "gy"]
    # f = 1 / (2 * ESP); the wider echo spacing gives the lower frequency.
    assert bands[0][0] == pytest.approx(1.0 / (2 * 510e-6))
    assert bands[0][1] == pytest.approx(1.0 / (2 * 410e-6))
    assert bands[1][2] == pytest.approx(16.0)  # 1.6 G/cm -> mT/m


@pytest.mark.parametrize(
    "table",
    [
        "2\n410 510 0.0\n",  # truncated inside an axis
        "1\n410 510\n0\n0\n",  # short row
        "1\n510 410 0.0\n0\n0\n",  # inverted range
        "1\n410 510 -1.0\n0\n0\n",  # negative limit
    ],
)
def test_read_esp_bands_fails_closed_on_a_malformed_table(tmp_path, table):
    path = tmp_path / "bad.dat"
    path.write_text(table)
    with pytest.raises(ValueError):
        pio.read_esp_bands(path)
