"""The magnetization preparations, checked by simulating what they leave behind.

A preparation's whole job is the state of the magnetization when it ends, so
that is what is measured: the module's RF is laid out on its own timeline and
integrated through the Bloch equation. Where a gradient carries the weighting
instead, the weighting is integrated from the played waveform.
"""

from __future__ import annotations

import numpy as np
import pytest

import pulserver.design as design
import pulserver.pypulseq as pp

#: Simulation raster: the finest any pulse here is written on.
RASTER = 1e-6


@pytest.fixture
def system():
    return pp.Opts(
        B0=3.0, max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s"
    )


def rf_timeline(module, b1_scale=1.0):
    """The module's RF as one complex field, on one raster.

    Gradients are left out: every module here is non-selective, so a gradient
    only spoils, and spoiling is not what these assertions are about. The
    pulses are resampled because they do not share a raster -- an SLR pulse
    comes out on the RF raster and an adiabatic one on the dwell it was
    designed with -- and the transmit offsets are put back, since they are
    fields on the event rather than part of the envelope.
    """
    b1 = np.zeros(round(module.duration / RASTER), dtype=complex)
    start = 0.0
    for block in module.blocks:
        for event in block:
            if getattr(event, "type", None) == "rf":
                times = np.asarray(event.t, dtype=float)
                signal = np.asarray(event.signal) * np.exp(
                    1j
                    * (
                        2.0 * np.pi * float(event.freq_offset) * times
                        + float(event.phase_offset)
                    )
                )
                # Resampled by interpolation, not by repeating: a block pulse
                # is stored as two samples a whole duration apart, so a repeat
                # count read off the spacing would play it for twice as long.
                span = pp.calc_duration(event) - float(event.delay)
                grid = (np.arange(round(span / RASTER)) + 0.5) * RASTER
                signal = np.interp(grid, times, signal.real) + 1j * np.interp(
                    grid, times, signal.imag
                )
                first = round((start + float(event.delay)) / RASTER)
                b1[first : first + signal.size] += b1_scale * signal
        start += pp.calc_duration(*block)
    return b1


def final_magnetization(module, *, b1_scale=1.0, off_resonance_hz=0.0):
    """Magnetization after the module's RF, starting from ``+z``."""
    b1 = rf_timeline(module, b1_scale)
    return pp.bloch(b1, np.full((1, 1), off_resonance_hz), RASTER)[0]


def spectral_profile(module, span_hz=6000.0, count=121):
    """Longitudinal magnetization against off-resonance, over ``span_hz``."""
    frequencies = np.linspace(-0.5 * span_hz, 0.5 * span_hz, count)
    magnetization = pp.bloch(rf_timeline(module), frequencies.reshape(-1, 1), RASTER)
    return frequencies, magnetization[:, 2]


# ======================================================================
# T2 and T1/T2 preparation
# ======================================================================


def test_a_t2_preparation_puts_the_magnetization_back_on_positive_z(system):
    assert final_magnetization(design.T2Preparation(system, 50e-3))[2] > 0.99


def test_the_hybrid_puts_it_on_negative_z(system):
    assert final_magnetization(design.T1T2Preparation(system, 50e-3))[2] < -0.99


@pytest.mark.parametrize("b1_scale", [0.85, 1.0, 1.15, 1.3])
def test_the_adiabatic_pulses_hold_across_transmit_amplitude(system, b1_scale):
    """The reason every pulse is adiabatic: the restoration must not track B1."""
    prep = design.T2Preparation(system, 50e-3)
    assert final_magnetization(prep, b1_scale=b1_scale)[2] > 0.9


def test_one_refocusing_pulse_is_refused(system):
    """An adiabatic full passage leaves a phase only a second one undoes."""
    with pytest.raises(ValueError, match="even"):
        design.T2Preparation(system, 50e-3, n_refocus=1)


def test_the_half_passages_are_exact_mirrors(system):
    prep = design.T2Preparation(system, 50e-3)
    down = np.asarray(prep.rf_prep.signal)
    up = np.asarray(prep.rf_store.signal)
    assert np.allclose(down, np.conj(up)[::-1])


def test_the_refocusing_pulse_is_published_once_however_many_play(system):
    prep = design.T2Preparation(system, 50e-3, n_refocus=4)
    assert not isinstance(prep.rf_ref, list)
    played = [block for block in prep.blocks if prep.rf_ref in block]
    assert len(played) == 4


def test_the_echo_time_is_measured_between_the_outer_pulse_centres(system):
    prep = design.T2Preparation(system, 50e-3)
    assert prep.echo_time == pytest.approx(50e-3, abs=2 * system.block_duration_raster)


@pytest.mark.parametrize("echo_time", [30e-3, 60e-3, 120e-3])
def test_the_echo_time_asked_for_is_the_echo_time_delivered(system, echo_time):
    prep = design.T2Preparation(system, echo_time)
    assert prep.echo_time == pytest.approx(
        echo_time, abs=2 * system.block_duration_raster
    )
    assert prep.check_timing()[0]


def test_an_echo_time_shorter_than_the_pulses_is_refused(system):
    with pytest.raises(ValueError, match="shorter than"):
        design.T2Preparation(system, 5e-3)


def test_the_hybrid_will_not_be_told_which_way_to_tip(system):
    with pytest.raises(ValueError, match="final_tip"):
        design.T1T2Preparation(system, 50e-3, final_tip="up")


def test_the_preparation_ends_with_a_three_axis_spoiler(system):
    prep = design.T2Preparation(system, 50e-3)
    channels = {event.channel for event in prep.blocks[-1] if hasattr(event, "channel")}
    assert channels == {"x", "y", "z"}


# ======================================================================
# Magnetization transfer
# ======================================================================


def test_mt_saturation_lands_off_resonance_and_leaves_water_alone(system):
    """Probed at 180 degrees, where the response is a clean inversion.

    At the working flip angle the pulse turns the pool through more than a
    full turn, so ``mz`` at the band says nothing simple -- 500 degrees reads
    back as ``cos(500)``. What the band placement has to guarantee is that the
    rotation happens there and nowhere near water, which one turn shows and
    several do not.
    """
    mt = design.MtPreparation(system, flip_angle_deg=180.0)
    frequencies, mz = spectral_profile(mt)
    assert np.interp(0.0, frequencies, mz) > 0.99
    assert np.interp(-1500.0, frequencies, mz) < -0.9


def test_mt_refuses_to_saturate_on_resonance(system):
    with pytest.raises(ValueError, match="freq_offset_hz"):
        design.MtPreparation(system, freq_offset_hz=0.0)


def test_the_mt_band_is_as_wide_as_the_time_bandwidth_product_says(system):
    """Bandwidth is the time-bandwidth product over the duration, and no wider."""
    narrow = design.MtPreparation(
        system, duration_s=16e-3, freq_offset_hz=-1500.0, flip_angle_deg=180.0
    )
    wide = design.MtPreparation(
        system, duration_s=4e-3, freq_offset_hz=-1500.0, flip_angle_deg=180.0
    )

    def width(module):
        frequencies, mz = spectral_profile(module, span_hz=8000.0, count=801)
        turned = frequencies[mz < 0.0]
        return turned.max() - turned.min()

    assert width(wide) == pytest.approx(4.0 * width(narrow), rel=0.2)


def test_ihmt_saturates_both_sides_of_water_at_once(system):
    ihmt = design.IhMtPreparation(system, flip_angle_deg=180.0)
    frequencies, mz = spectral_profile(ihmt)
    assert np.interp(0.0, frequencies, mz) > 0.99
    assert np.interp(-1500.0, frequencies, mz) < 0.0
    assert np.interp(+1500.0, frequencies, mz) < 0.0
    assert ihmt.band_offsets_hz.tolist() == [-1500.0, 1500.0]


def test_the_two_ihmt_bands_are_equally_deep(system):
    """Symmetric about water: neither side may be favoured."""
    ihmt = design.IhMtPreparation(system, flip_angle_deg=180.0)
    frequencies, mz = spectral_profile(ihmt)
    assert np.interp(-1500.0, frequencies, mz) == pytest.approx(
        np.interp(+1500.0, frequencies, mz), abs=1e-6
    )


def test_ihmt_deposits_the_same_power_as_the_single_offset_arm(system):
    """Power-matched, so the difference is inhomogeneous saturation and not power."""
    single = design.MtPreparation(system, freq_offset_hz=1500.0)
    dual = design.IhMtPreparation(system)

    def power(module):
        return float(np.sum(np.abs(np.asarray(module.rf_prep.signal)) ** 2))

    assert power(dual) == pytest.approx(power(single), rel=1e-3)


def test_ihmt_costs_peak_amplitude_for_that(system):
    """Two bands add in phase at the pulse centre; the peak goes up by root two."""
    single = design.MtPreparation(system, freq_offset_hz=1500.0)
    dual = design.IhMtPreparation(system)
    peak = np.abs(np.asarray(dual.rf_prep.signal)).max()
    assert peak == pytest.approx(
        np.sqrt(2.0) * np.abs(np.asarray(single.rf_prep.signal)).max(), rel=1e-3
    )


def test_ihmt_needs_a_positive_offset_because_it_builds_both(system):
    with pytest.raises(ValueError, match="positive"):
        design.IhMtPreparation(system, freq_offset_hz=-1500.0)


@pytest.mark.parametrize("n_pulses", [1, 3])
def test_a_saturation_train_plays_one_pulse_over_and_over(system, n_pulses):
    mt = design.MtPreparation(system, n_pulses=n_pulses)
    assert len(mt.blocks) == n_pulses + 1
    assert not isinstance(mt.rf_prep, list)


# ======================================================================
# Bloch-Siegert
# ======================================================================


def test_the_bloch_siegert_pulse_is_not_spoiled(system):
    """It runs between excitation and readout; spoiling would erase the signal."""
    pulse = design.BlochSiegertPreparation(system)
    assert len(pulse.blocks) == 1
    assert [event.type for event in pulse.blocks[0]] == ["rf"]


def test_it_barely_tips_anything(system):
    """Far off resonance: the effect is a phase, not an excitation."""
    magnetization = final_magnetization(design.BlochSiegertPreparation(system))
    assert magnetization[2] > 0.9


def test_its_envelope_is_a_fermi_plateau(system):
    """``flat_fraction`` is the width at half the peak, which is what a Fermi has."""
    pulse = design.BlochSiegertPreparation(system, flat_fraction=0.8)
    envelope = np.abs(np.asarray(pulse.rf_prep.signal))
    assert (envelope > 0.5 * envelope.max()).mean() == pytest.approx(0.8, abs=0.02)
    assert (envelope > 0.99 * envelope.max()).mean() > 0.5


def test_the_shift_constant_goes_as_one_over_the_offset(system):
    near = design.BlochSiegertPreparation(system, freq_offset_hz=4000.0)
    far = design.BlochSiegertPreparation(system, freq_offset_hz=8000.0)
    assert near.kbs == pytest.approx(2.0 * far.kbs, rel=1e-9)


def test_the_shift_constant_goes_as_the_square_of_the_amplitude(system):
    weak = design.BlochSiegertPreparation(system, peak_b1_hz=250.0)
    strong = design.BlochSiegertPreparation(system, peak_b1_hz=500.0)
    assert strong.kbs == pytest.approx(4.0 * weak.kbs, rel=1e-9)


def test_the_reported_constant_per_gauss_is_amplitude_independent(system):
    """kbs_per_gauss2 describes the shape, so the peak setting must not change it."""
    weak = design.BlochSiegertPreparation(system, peak_b1_hz=250.0)
    strong = design.BlochSiegertPreparation(system, peak_b1_hz=500.0)
    assert weak.kbs_per_gauss2 == pytest.approx(strong.kbs_per_gauss2, rel=1e-9)


def test_it_refuses_to_sit_on_resonance(system):
    with pytest.raises(ValueError, match="freq_offset_hz"):
        design.BlochSiegertPreparation(system, freq_offset_hz=0.0)


# ======================================================================
# Diffusion
# ======================================================================


def played_b_value(module, system):
    """b-value integrated from the waveform the module actually plays."""
    raster = system.grad_raster_time
    total = round(module.duration / raster)
    gradient = np.zeros(total)
    sign, start = 1.0, 0.0
    for block in module.blocks:
        for event in block:
            if getattr(event, "type", None) == "rf" and event.use == "refocusing":
                sign = -1.0
            if event is module.g_diff:
                first = round((start + float(event.delay)) / raster)
                count = round(pp.calc_duration(event) / raster)
                knots = np.cumsum(
                    [0.0, event.rise_time, event.flat_time, event.fall_time]
                )
                times = (np.arange(count) + 0.5) * raster
                gradient[first : first + count] += sign * np.interp(
                    times, knots, [0.0, event.amplitude, event.amplitude, 0.0]
                )
        start += pp.calc_duration(*block)
    q = 2.0 * np.pi * np.cumsum(gradient) * raster
    return float(np.sum(q**2) * raster / 1e6)


def test_the_played_waveform_carries_the_b_value_it_claims(system):
    prep = design.DiffusionPreparation(system, 500.0)
    assert played_b_value(prep, system) == pytest.approx(500.0, rel=1e-6)


def test_the_b_value_is_not_the_ideal_rectangle_formula(system):
    """The ramps are worth a few percent, which is why it is integrated."""
    prep = design.DiffusionPreparation(system, 500.0)
    amplitude = float(prep.g_diff.amplitude)
    flat = float(prep.g_diff.flat_time)
    ideal = (2.0 * np.pi * amplitude * flat) ** 2 * (30e-3 - flat / 3.0) / 1e6
    assert ideal == pytest.approx(500.0, rel=0.05)
    assert ideal != pytest.approx(500.0, rel=1e-3)


def test_a_lower_b_value_is_a_square_root_scaling(system):
    prep = design.DiffusionPreparation(system, 500.0)
    assert prep.scale_for(125.0) == pytest.approx(0.5)
    assert prep.scale_for(0.0) == 0.0
    assert prep.scale_for(500.0) == pytest.approx(1.0)


def test_scaling_beyond_the_design_b_value_is_refused(system):
    prep = design.DiffusionPreparation(system, 500.0)
    with pytest.raises(ValueError, match="between 0 and the design value"):
        prep.scale_for(600.0)


def test_a_b_value_beyond_the_hardware_says_what_is_reachable(system):
    with pytest.raises(ValueError, match="which reach"):
        design.DiffusionPreparation(system, 1e6)


def test_the_pair_is_one_event_played_twice(system):
    """So a direction is one rotation and a b-value one scaling, not two of each."""
    prep = design.DiffusionPreparation(system, 500.0)
    assert not isinstance(prep.g_diff, list)
    assert len([block for block in prep.blocks if prep.g_diff in block]) == 2


def test_the_diffusion_preparation_stores_on_positive_z(system):
    assert final_magnetization(design.DiffusionPreparation(system, 500.0))[2] > 0.99


def test_a_separation_that_cannot_hold_the_refocusing_pulse_is_refused(system):
    with pytest.raises(ValueError, match="cannot hold"):
        design.DiffusionPreparation(
            system, 100.0, gradient_duration_s=20e-3, gradient_separation_s=20.5e-3
        )


def test_lobes_that_overlap_are_refused(system):
    with pytest.raises(ValueError, match="gradient_separation_s"):
        design.DiffusionPreparation(
            system, 100.0, gradient_duration_s=20e-3, gradient_separation_s=10e-3
        )


# ======================================================================
# Shared shape
# ======================================================================


@pytest.mark.parametrize(
    "factory",
    [
        lambda system: design.T2Preparation(system, 50e-3),
        lambda system: design.T1T2Preparation(system, 50e-3),
        lambda system: design.MtPreparation(system),
        lambda system: design.IhMtPreparation(system),
        lambda system: design.BlochSiegertPreparation(system),
        lambda system: design.DiffusionPreparation(system, 500.0),
        lambda system: design.InversionPreparation(system),
        lambda system: design.FatSaturation(system),
    ],
)
def test_every_preparation_is_valid_pulseq(system, factory):
    module = factory(system)
    assert module.check_timing()[0]
    assert module.duration > 0
    assert 0.0 <= module.center <= module.duration


@pytest.mark.parametrize(
    "factory",
    [
        lambda system, labels: design.T2Preparation(system, 50e-3, labels=labels),
        lambda system, labels: design.MtPreparation(system, labels=labels),
        lambda system, labels: design.DiffusionPreparation(
            system, 500.0, labels=labels
        ),
    ],
)
def test_a_preparation_opens_a_slot_for_the_counter_it_was_given(system, factory):
    module = factory(system, ("REP",))
    assert module.prep_labels.label == "REP"
    assert module.prep_labels in module.blocks[0]
