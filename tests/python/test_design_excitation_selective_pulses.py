"""Excitations selective in frequency, in several bands, or in two dimensions.

Each is checked against what the pulse actually does -- the spectral profile
from a Bloch simulation, the excited profile from the small-tip forward model --
rather than against the arithmetic that built it.
"""

from __future__ import annotations

import numpy as np
import pytest

import pulserver.design as design
import pulserver.pypulseq as pp


@pytest.fixture
def system():
    return pp.Opts(
        max_grad=40,
        grad_unit="mT/m",
        max_slew=180,
        slew_unit="T/m/s",
        rf_dead_time=100e-6,
        rf_ringdown_time=20e-6,
    )


def spectrum(module, **kwargs):
    """Longitudinal magnetisation against frequency, from a Bloch simulation."""
    response = module.sim_rf(compat=False, **kwargs)
    return np.asarray(response.frequency), np.asarray(response.mz_z)


def excited_profile(module, extent=0.08, n=161):
    """Transverse magnetisation over a plane, from the small-tip forward model."""
    gradient = np.stack([np.asarray(g.waveform) for g in module.gradients], axis=1)
    dwell = float(np.diff(np.asarray(module.gradients[0].tt))[0])
    kspace = -np.cumsum(gradient[::-1], axis=0)[::-1] * dwell
    axis = np.linspace(-extent, extent, n)
    grid = np.stack([m.ravel() for m in np.meshgrid(axis, axis, indexing="ij")], axis=1)
    profile = np.abs(
        np.exp(2j * np.pi * (grid @ kspace.T)) @ np.asarray(module.rf.signal)
    )
    return axis, profile.reshape(n, n) / profile.max()


# ----------------------------------------------------------------------
# FrequencySelectiveExcitation
# ----------------------------------------------------------------------


def test_a_frequency_selective_pulse_plays_no_gradient(system):
    water = design.FrequencySelectiveExcitation(system, 90.0, bandwidth_hz=200.0)
    assert len(water.blocks) == 1
    assert water.blocks[0] == (water.rf,)


def test_the_passband_fixes_the_duration(system):
    wide = design.FrequencySelectiveExcitation(system, 90.0, bandwidth_hz=400.0)
    narrow = design.FrequencySelectiveExcitation(system, 90.0, bandwidth_hz=100.0)
    assert wide.duration_s == pytest.approx(4.0 / 400.0, rel=1e-6)
    assert narrow.duration_s == pytest.approx(4 * wide.duration_s, rel=1e-6)


def test_it_inverts_inside_its_band_and_not_outside(system):
    """Probed at 180 degrees, where the response is a clean inversion."""
    band = design.FrequencySelectiveExcitation(
        system, 180.0, bandwidth_hz=500.0, pulse_type="inv"
    )
    frequency, mz = spectrum(band, df=10.0)
    inside = np.abs(frequency) < 100.0
    outside = np.abs(frequency) > 700.0
    assert mz[inside].max() < -0.8
    assert mz[outside].min() > 0.8


def test_the_band_can_be_pointed_off_water(system):
    fat = design.FrequencySelectiveExcitation(
        system, 90.0, bandwidth_hz=200.0, freq_offset_hz=-440.0
    )
    assert float(fat.rf.freq_offset) == pytest.approx(-440.0)


def test_a_band_with_no_width_is_refused(system):
    with pytest.raises(ValueError, match="bandwidth_hz"):
        design.FrequencySelectiveExcitation(system, 90.0, bandwidth_hz=0.0)


# ----------------------------------------------------------------------
# SpspExcitation
# ----------------------------------------------------------------------


def test_a_spectral_spatial_pulse_selects_a_slice_and_a_band(system):
    water = design.SpspExcitation(
        system, 30.0, thickness_m=10e-3, spectral_bandwidth_hz=300.0, n_subpulses=10
    )
    assert len(water.blocks) == 2
    assert water.gz.channel == "z"
    assert water.gz_reph.channel == "z"


def test_a_train_whose_lobes_already_cancel_gets_no_rephaser(system):
    """Half the lobes fall after the pulse's centre and they alternate, so an
    even number of them leaves nothing to undo -- and a rephaser carrying
    something anyway would be the thing that spoiled the slice."""
    water = design.SpspExcitation(
        system, 30.0, thickness_m=10e-3, spectral_bandwidth_hz=300.0, n_subpulses=12
    )
    assert len(water.blocks) == 1
    assert not hasattr(water.events, "gz_reph")


def test_its_selection_gradient_alternates(system):
    """One lobe per subpulse, and they cancel each other out along the train."""
    water = design.SpspExcitation(
        system, 30.0, thickness_m=10e-3, spectral_bandwidth_hz=300.0, n_subpulses=12
    )
    waveform = np.asarray(water.gz.waveform)
    assert waveform.min() < 0 < waveform.max()
    assert np.sum(np.diff(np.sign(waveform[waveform != 0])) != 0) >= 10


@pytest.mark.parametrize("thickness_m", [10e-3, 64e-3, 128e-3])
def test_the_pulse_leaves_k_where_it_found_it(system, thickness_m):
    """A slice or slab that does not refocus has no signal to give.

    The train's own lobes cancel along it, but the pulse excites at its centre
    and what the gradient accrues after that is what has to be undone. So the
    question is asked of k rather than of the waveform: where the excitation
    ends, k is back at the origin.
    """
    water = design.SpspExcitation(
        system, 30.0, thickness_m=thickness_m, spectral_bandwidth_hz=300.0
    )
    seq = pp.Sequence(system)
    for block in water.blocks:
        seq.add_block(*block)
    seq.add_block(pp.make_adc(num_samples=2, dwell=system.grad_raster_time))
    kz = float(np.asarray(seq.calculate_kspace()[0])[2][-1])
    assert abs(kz) < 1e-6 / thickness_m


def test_a_slab_carries_its_rephasing_in_the_selection_gradient(system):
    """A 3D readout encodes partitions on the selection axis, so a rephaser of
    its own would have nowhere to sit. Merged, it is one event and one block,
    and k still comes back to the origin."""
    slab = design.SpspExcitation(
        system, 30.0, thickness_m=128e-3, spectral_bandwidth_hz=300.0, is_slab=True
    )
    assert len(slab.blocks) == 1
    assert not hasattr(slab.events, "gz_reph")

    seq = pp.Sequence(system)
    seq.add_block(*slab.blocks[0])
    seq.add_block(pp.make_adc(num_samples=2, dwell=system.grad_raster_time))
    assert abs(float(np.asarray(seq.calculate_kspace()[0])[2][-1])) < 1e-6 / 128e-3


def test_the_rephaser_can_be_left_out(system):
    bare = design.SpspExcitation(
        system,
        30.0,
        thickness_m=10e-3,
        spectral_bandwidth_hz=300.0,
        n_subpulses=12,
        rephase=False,
    )
    assert len(bare.blocks) == 1
    assert not hasattr(bare.events, "gz_reph")


def test_a_spectral_spatial_pulse_on_a_channel_that_is_not_one_is_refused(system):
    with pytest.raises(ValueError, match="axis"):
        design.SpspExcitation(
            system, 30.0, thickness_m=10e-3, spectral_bandwidth_hz=300.0, axis="t"
        )


# ----------------------------------------------------------------------
# SmsExcitation
# ----------------------------------------------------------------------


def test_the_bands_land_where_the_slice_gap_says(system):
    sms = design.SmsExcitation(
        system, 60.0, thickness_m=3e-3, slice_gap_m=24e-3, n_bands=3
    )
    assert sms.band_positions_m == pytest.approx([-0.024, 0.0, 0.024], abs=1e-9)
    assert sms.band_offsets_hz[1] == 0.0


def test_the_bands_are_where_the_simulation_finds_them(system):
    """Measured against a Bloch simulation, not against the offsets asked for."""
    sms = design.SmsExcitation(
        system, 90.0, thickness_m=3e-3, slice_gap_m=24e-3, n_bands=3, duration_s=4e-3
    )
    frequency, mz = spectrum(sms, df=20.0, bandwidth_multiplier=12.0)
    for offset in sms.band_offsets_hz:
        near = np.abs(frequency - offset) < 200.0
        assert mz[near].min() < 0.2


def test_spreading_the_band_phases_lowers_the_peak(system):
    stacked = design.SmsExcitation(
        system, 60.0, thickness_m=3e-3, slice_gap_m=24e-3, n_bands=4, phases=None
    )
    spread = design.SmsExcitation(
        system, 60.0, thickness_m=3e-3, slice_gap_m=24e-3, n_bands=4
    )
    assert stacked.peak_ratio == pytest.approx(4.0, rel=0.02)
    assert spread.peak_ratio < 0.75 * stacked.peak_ratio


def test_one_band_is_the_pulse_it_was_made_from(system):
    single = design.SmsExcitation(
        system, 60.0, thickness_m=3e-3, slice_gap_m=24e-3, n_bands=1
    )
    assert single.peak_ratio == pytest.approx(1.0)
    assert single.band_positions_m == pytest.approx([0.0])


def test_a_gap_of_nothing_is_refused(system):
    with pytest.raises(ValueError, match="slice_gap_m"):
        design.SmsExcitation(system, 60.0, thickness_m=3e-3, slice_gap_m=0.0)


# ----------------------------------------------------------------------
# MultibandExcitation
# ----------------------------------------------------------------------


def test_a_multiband_saturation_pulse_plays_no_gradient(system):
    dual = design.MultibandExcitation(
        system, 7.0, duration_s=2e-3, band_offset_hz=7000.0, n_bands=3
    )
    assert len(dual.blocks) == 1
    assert dual.blocks[0] == (dual.rf,)
    assert dual.band_offsets_hz == pytest.approx([-7000.0, 0.0, 7000.0])


@pytest.mark.parametrize("n_bands", [2, 3])
def test_the_sideband_power_is_solved_for_the_requested_b1rms(system, n_bands):
    module = design.MultibandExcitation(
        system,
        7.0,
        duration_s=2e-3,
        band_offset_hz=7000.0,
        n_bands=n_bands,
        b1rms_ut=2.0,
        tr=30e-3,
    )
    assert module.b1rms_ut == pytest.approx(2.0, rel=1e-6)


def test_one_sideband_carries_what_two_carry_between_them(system):
    """Power-matched, not amplitude-matched: that is what makes ihMT ihMT."""
    single = design.MultibandExcitation(
        system,
        7.0,
        duration_s=2e-3,
        band_offset_hz=7000.0,
        n_bands=2,
        b1rms_ut=2.0,
        tr=30e-3,
    )
    dual = design.MultibandExcitation(
        system,
        7.0,
        duration_s=2e-3,
        band_offset_hz=7000.0,
        n_bands=3,
        b1rms_ut=2.0,
        tr=30e-3,
    )
    assert single.sideband_power == pytest.approx(2 * dual.sideband_power)
    centre = list(dual.band_offsets_hz).index(0.0)
    assert dual.band_power_fraction[centre] == pytest.approx(
        single.band_power_fraction[list(single.band_offsets_hz).index(0.0)]
    )
    assert dual.band_power_fraction.sum() == pytest.approx(1.0)


def test_the_bands_add_in_phase_so_two_of_them_cost_peak(system):
    single = design.MultibandExcitation(
        system,
        7.0,
        duration_s=2e-3,
        band_offset_hz=7000.0,
        n_bands=2,
        b1rms_ut=2.0,
        tr=30e-3,
    )
    dual = design.MultibandExcitation(
        system,
        7.0,
        duration_s=2e-3,
        band_offset_hz=7000.0,
        n_bands=3,
        b1rms_ut=2.0,
        tr=30e-3,
    )
    assert dual.peak_ut > single.peak_ut


def test_a_target_the_on_resonance_band_already_exceeds_is_refused(system):
    with pytest.raises(ValueError, match="already exceeds"):
        design.MultibandExcitation(
            system, 60.0, duration_s=2e-3, band_offset_hz=7000.0, b1rms_ut=0.1, tr=30e-3
        )


def test_a_root_mean_square_over_nothing_is_refused(system):
    with pytest.raises(ValueError, match="give tr too"):
        design.MultibandExcitation(
            system, 7.0, duration_s=2e-3, band_offset_hz=7000.0, b1rms_ut=2.0
        )


def test_a_single_band_is_not_a_multiband_pulse(system):
    with pytest.raises(ValueError, match="no sideband"):
        design.MultibandExcitation(
            system, 7.0, duration_s=2e-3, band_offset_hz=7000.0, n_bands=1
        )


# ----------------------------------------------------------------------
# SpatialSelective2DExcitation
# ----------------------------------------------------------------------


def test_a_two_dimensional_pulse_runs_two_gradients_in_its_own_block(system):
    pencil = design.SpatialSelective2DExcitation(
        system, 30.0, fov=0.256, matrix=32, selective_size=0.04
    )
    assert len(pencil.blocks) == 1
    assert [gradient.channel for gradient in pencil.gradients] == ["x", "y"]
    assert pencil.self_refocused
    assert pencil.rephasers == ()
    assert pencil.check_timing()[0]


def test_the_envelope_runs_alongside_the_trajectory_it_was_designed_for(system):
    """Sample n of the pulse belongs to sample n of the path, dead time and all."""
    pencil = design.SpatialSelective2DExcitation(
        system, 30.0, fov=0.256, matrix=32, selective_size=0.04
    )
    for gradient in pencil.gradients:
        assert float(gradient.delay) == pytest.approx(float(pencil.rf.delay))
        assert len(np.asarray(gradient.waveform)) == len(np.asarray(pencil.rf.signal))


def test_it_excites_a_disc_of_the_size_asked_for(system):
    """The forward model, not the design arithmetic."""
    pencil = design.SpatialSelective2DExcitation(
        system, 30.0, fov=0.256, matrix=32, selective_size=0.04
    )
    axis, profile = excited_profile(pencil)
    grid = np.hypot(*np.meshgrid(axis, axis, indexing="ij"))
    assert profile[grid < 0.012].mean() > 0.85
    assert profile[(grid > 0.030) & (grid < 0.08)].mean() < 0.15


def test_the_excited_disc_is_centred(system):
    pencil = design.SpatialSelective2DExcitation(
        system, 30.0, fov=0.256, matrix=32, selective_size=0.04
    )
    axis, profile = excited_profile(pencil)
    # The flat top ripples, so the brightest pixel is not the middle; where the
    # excited signal balances is.
    weight = profile.sum()
    assert (profile.sum(axis=1) @ axis) / weight == pytest.approx(0.0, abs=1e-3)
    assert (profile.sum(axis=0) @ axis) / weight == pytest.approx(0.0, abs=1e-3)


def test_fewer_arms_is_a_proportionally_shorter_pulse(system):
    whole = design.SpatialSelective2DExcitation(
        system, 30.0, fov=0.256, matrix=32, selective_size=0.04
    )
    half = design.SpatialSelective2DExcitation(
        system, 30.0, fov=0.256, matrix=32, selective_size=0.04, n_interleaves=8
    )
    assert float(half.rf.shape_dur) == pytest.approx(
        0.5 * float(whole.rf.shape_dur), rel=1e-9
    )


def test_a_bigger_spot_is_asked_for_and_delivered(system):
    small = design.SpatialSelective2DExcitation(
        system, 30.0, fov=0.256, matrix=32, selective_size=0.03
    )
    large = design.SpatialSelective2DExcitation(
        system, 30.0, fov=0.256, matrix=32, selective_size=0.06
    )
    _, small_profile = excited_profile(small)
    _, large_profile = excited_profile(large)
    assert (large_profile > 0.5).sum() > 2 * (small_profile > 0.5).sum()


def test_more_arms_than_the_trajectory_has_is_refused(system):
    with pytest.raises(ValueError, match="n_interleaves"):
        design.SpatialSelective2DExcitation(
            system, 30.0, fov=0.256, matrix=32, n_interleaves=999
        )


def test_a_target_that_does_not_fit_the_grid_is_refused(system):
    with pytest.raises(ValueError, match="does not match matrix"):
        design.SpatialSelective2DExcitation(
            system, 30.0, fov=0.256, matrix=32, target=np.ones((16, 16))
        )
