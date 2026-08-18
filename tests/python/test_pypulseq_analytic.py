"""Routines whose correctness is an analytic property, asserted directly.

No reference data and no second implementation: what is checked is the
property each routine exists to guarantee — an area hit exactly within the
gradient and slew limits, an energy that scales as the square of the flip
angle, a hash that matches the bytes it was taken over.
"""

from __future__ import annotations

import numpy as np
import pypulseq as upstream
import pytest

import pulserver.pypulseq as pp


@pytest.fixture
def system():
    return pp.Opts(max_grad=30, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")


# -- make_hexagon_gradient_area ------------------------------------------


#: Endpoint amplitudes as a fraction of ``max_grad``, so the cases stay
#: meaningful whatever the system limits are.
ENDPOINTS = [(0.0, 0.0), (0.5, 0.0), (0.0, -0.6), (0.4, 0.4), (-0.3, 0.7)]


@pytest.mark.parametrize("area", [1000.0, -1000.0, 50.0, 20000.0])
@pytest.mark.parametrize(("start_fraction", "end_fraction"), ENDPOINTS)
def test_the_hexagon_hits_the_area_between_the_amplitudes_it_was_given(
    system, area, start_fraction, end_fraction
):
    grad_start = start_fraction * system.max_grad
    grad_end = end_fraction * system.max_grad
    _, times, amplitudes = pp.make_hexagon_gradient_area(
        area=area, channel="x", grad_start=grad_start, grad_end=grad_end, system=system
    )

    assert float(np.trapezoid(amplitudes, times)) == pytest.approx(area, abs=1e-3)
    assert amplitudes[0] == pytest.approx(grad_start)
    assert amplitudes[-1] == pytest.approx(grad_end)
    assert np.all(np.diff(times) > 0)


@pytest.mark.parametrize(("start_fraction", "end_fraction"), ENDPOINTS)
def test_the_hexagon_stays_inside_the_gradient_and_slew_limits(
    system, start_fraction, end_fraction
):
    """The limits are the point: an area reached by breaking them is no answer."""
    _, times, amplitudes = pp.make_hexagon_gradient_area(
        area=1000.0,
        channel="x",
        grad_start=start_fraction * system.max_grad,
        grad_end=end_fraction * system.max_grad,
        system=system,
    )
    slew = np.abs(np.diff(amplitudes) / np.diff(times))

    assert np.abs(amplitudes).max() <= system.max_grad
    assert slew.max() <= system.max_slew


def _durations(system, start_fraction, end_fraction, area):
    """How long each of the two shapes takes to accrue ``area``."""
    grad_start = start_fraction * system.max_grad
    grad_end = end_fraction * system.max_grad
    _, times, _ = pp.make_hexagon_gradient_area(
        area=area, channel="x", grad_start=grad_start, grad_end=grad_end, system=system
    )
    _, flat_times, _ = pp.make_extended_trapezoid_area(
        area=area, channel="x", grad_start=grad_start, grad_end=grad_end, system=system
    )
    return float(times[-1]), float(flat_times[-1])


def test_the_hexagon_is_never_longer_than_the_flat_topped_shape(system):
    """Freeing the middle vertices is only worth doing if it never costs time."""
    for start_fraction, end_fraction in ENDPOINTS:
        for area in (1000.0, 200.0):
            hexagon, flat = _durations(system, start_fraction, end_fraction, area)
            assert hexagon <= flat + 1e-12, (start_fraction, end_fraction, area)


def test_the_hexagon_is_sometimes_strictly_shorter(system):
    """If it never won, it would not be worth having."""
    wins = [
        (start, end)
        for start, end in ENDPOINTS
        for area in (1000.0, 200.0)
        if _durations(system, start, end, area)[0]
        < _durations(system, start, end, area)[1] - 1e-12
    ]
    assert wins


def test_the_hexagon_builds_a_playable_event(system):
    grad, _, _ = pp.make_hexagon_gradient_area(
        area=1000.0, channel="x", grad_start=0.0, grad_end=0.0, system=system
    )
    seq = pp.Sequence(system)
    seq.add_block(grad)
    assert seq.duration()[0] > 0


# -- calc_rf_power --------------------------------------------------------


def test_energy_scales_as_the_square_of_the_flip_angle(system):
    def energy(flip):
        rf = pp.make_sinc_pulse(
            flip_angle=flip, duration=3e-3, time_bw_product=4, system=system
        )
        return pp.calc_rf_power(rf)[0]

    assert energy(np.pi) / energy(np.pi / 2) == pytest.approx(4.0, rel=1e-6)


def test_the_rms_is_the_energy_spread_over_the_pulse(system):
    rf = pp.make_sinc_pulse(
        flip_angle=np.pi / 2, duration=3e-3, time_bw_product=4, system=system
    )
    energy, peak, rms = pp.calc_rf_power(rf)

    assert rms == pytest.approx(np.sqrt(energy / rf.shape_dur))
    assert peak >= rms**2


def test_a_block_pulse_is_integrated_from_two_samples(system):
    """Its ``t`` holds only the endpoints, which is why the pulse is resampled."""
    rf = pp.make_block_pulse(flip_angle=np.pi / 2, duration=1e-3, system=system)
    energy, peak, rms = pp.calc_rf_power(rf)

    assert len(rf.t) == 2
    # A rectangle: every sample is the peak, so energy is peak x duration.
    assert energy == pytest.approx(peak * rf.shape_dur, rel=1e-6)
    assert rms == pytest.approx(np.sqrt(peak), rel=1e-6)


def test_power_ignores_frequency_and_phase_modulation(system):
    rf = pp.make_sinc_pulse(
        flip_angle=np.pi / 2, duration=3e-3, time_bw_product=4, system=system
    )
    before = pp.calc_rf_power(rf)
    rf.freq_offset = 2000.0
    rf.phase_offset = 1.1
    assert pp.calc_rf_power(rf) == pytest.approx(before)


# -- calc_rf_bandwidth ----------------------------------------------------


def _sinc(system, **kwargs):
    return pp.make_sinc_pulse(
        flip_angle=np.pi / 2, duration=2e-3, time_bw_product=4, system=system, **kwargs
    )


def test_bandwidth_of_an_unshifted_pulse_is_what_upstream_answers(system):
    """Where upstream is right, the override must not move the number."""
    from pulserver.pypulseq._events import as_namespace

    rf = _sinc(system)
    assert pp.calc_rf_bandwidth(rf) == pytest.approx(
        float(np.ravel(upstream.calc_rf_bandwidth(as_namespace(rf)))[0])
    )


def test_bandwidth_is_the_time_bandwidth_product_over_the_duration(system):
    for time_bw_product, duration in [(4, 2e-3), (8, 4e-3), (4, 1e-3), (6, 3e-3)]:
        rf = pp.make_sinc_pulse(
            flip_angle=np.pi / 2,
            duration=duration,
            time_bw_product=time_bw_product,
            system=system,
        )
        assert pp.calc_rf_bandwidth(rf) == pytest.approx(
            time_bw_product / duration, rel=0.06
        )


def test_retuning_a_pulse_moves_its_band_without_widening_it(system):
    """The property upstream's estimator does not have.

    Upstream reports 20 Hz for the offset pulse below, against 1900 for the
    same pulse at zero offset -- so this is asserted here rather than diffed
    against it.
    """
    offset = 1500.0
    rf = _sinc(system)
    centred, spectrum, axis = pp.calc_rf_bandwidth(
        rf, return_spectrum=True, return_axis=True
    )
    band = axis[spectrum >= 0.5 * spectrum.max()]
    centre_before = 0.5 * (band[0] + band[-1])

    rf.freq_offset = offset
    shifted, spectrum, axis = pp.calc_rf_bandwidth(
        rf, return_spectrum=True, return_axis=True
    )
    band = axis[spectrum >= 0.5 * spectrum.max()]
    centre_after = 0.5 * (band[0] + band[-1])

    assert shifted == pytest.approx(centred, rel=1e-9)
    assert centre_after - centre_before == pytest.approx(offset, rel=1e-3)


def test_bandwidth_returns_what_each_flag_combination_promises(system):
    rf = _sinc(system)
    assert isinstance(pp.calc_rf_bandwidth(rf), float)

    bandwidth, axis = pp.calc_rf_bandwidth(rf, return_axis=True)
    bandwidth_again, spectrum = pp.calc_rf_bandwidth(rf, return_spectrum=True)
    both = pp.calc_rf_bandwidth(rf, return_spectrum=True, return_axis=True)

    assert bandwidth == bandwidth_again == both[0]
    np.testing.assert_array_equal(both[1], spectrum)
    np.testing.assert_array_equal(both[2], axis)


def test_bandwidth_keeps_upstreams_parameter_names_and_order():
    import inspect

    assert list(inspect.signature(pp.calc_rf_bandwidth).parameters) == list(
        inspect.signature(upstream.calc_rf_bandwidth).parameters
    )


# -- verify_file_signature ------------------------------------------------


def test_a_written_file_verifies_against_the_hash_write_returned(system, tmp_path):
    seq = pp.Sequence(system)
    seq.add_block(pp.make_trapezoid(channel="x", area=1000, system=system))
    path = tmp_path / "signed.seq"
    signature = seq.write(str(path), check_timing=False)

    ok, stored, computed = pp.verify_file_signature(path)
    assert ok
    assert stored == computed == signature


def test_an_edited_file_fails_verification(system, tmp_path):
    seq = pp.Sequence(system)
    seq.add_block(pp.make_trapezoid(channel="x", area=1000, system=system))
    path = tmp_path / "signed.seq"
    seq.write(str(path), check_timing=False)

    payload = path.read_bytes().replace(
        b"# Pulseq sequence file", b"# Pulseq sequence FILE"
    )
    path.write_bytes(payload)

    ok, stored, computed = pp.verify_file_signature(path)
    assert not ok
    assert stored != computed


def test_an_unsigned_file_says_so_rather_than_failing_verification(system, tmp_path):
    seq = pp.Sequence(system)
    seq.add_block(pp.make_trapezoid(channel="x", area=1000, system=system))
    path = tmp_path / "unsigned.seq"
    seq.write(str(path), create_signature=False, check_timing=False)

    with pytest.raises(ValueError, match=r"no \[SIGNATURE\] section"):
        pp.verify_file_signature(path)


def test_a_binary_file_is_recognised_by_its_bytes_not_its_name(system, tmp_path):
    seq = pp.Sequence(system)
    seq.add_block(pp.make_trapezoid(channel="x", area=1000, system=system))
    path = tmp_path / "looks_like_text.seq"
    seq.write_binary(str(path))

    with pytest.raises(ValueError, match="binary Pulseq file"):
        pp.verify_file_signature(path)


# -- the rest of the parity surface ---------------------------------------


def test_every_rf_use_tag_is_offered():
    assert pp.get_supported_rf_use() == pp.get_supported_rf_uses()
    assert set(pp.get_supported_rf_use()) == {
        "excitation",
        "refocusing",
        "inversion",
        "saturation",
        "preparation",
        "other",
        "undefined",
    }


def test_the_adiabatic_factory_is_reachable_and_builds_a_slotted_pulse(system):
    from pulserver._ext import pulseqpp as cxx

    for pulse_type in ("hypsec", "wurst"):
        rf = pp.make_adiabatic_pulse(
            pulse_type=pulse_type, system=system, duration=10e-3
        )
        assert isinstance(rf, cxx.Event)


def test_rotate3d_turns_a_gradient_onto_another_axis_and_returns_slotted_events(system):
    from pulserver._ext import pulseqpp as cxx

    gx = pp.make_trapezoid(channel="x", area=1000, system=system)
    about_z = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    (rotated,) = pp.rotate3D(gx, rotation_matrix=about_z, system=system)

    assert isinstance(rotated, cxx.Event)
    assert rotated.channel == "y"
    assert rotated.amplitude == pytest.approx(gx.amplitude)


@pytest.mark.parametrize(
    "name", ["add_custom_label", "addCustomLabel", "add_ramps", "addRamps"]
)
def test_a_matlab_name_that_is_not_ported_explains_itself(name):
    with pytest.raises(AttributeError, match="does not export"):
        getattr(pp, name)
