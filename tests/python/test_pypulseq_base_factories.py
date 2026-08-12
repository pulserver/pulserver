"""The base factories: events and plain arrays, stated in imaging terms.

What each is checked against is the quantity it was asked for -- an area, a
k-space cell count, a dwell that lands on two rasters at once -- rather than
against another implementation of the same arithmetic.
"""

from __future__ import annotations

import numpy as np
import pytest

import pulserver.pypulseq as pp
from pulserver._ext import _pulseqpp_wrapper as cxx


@pytest.fixture
def system():
    return pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")


# -- gradients ------------------------------------------------------------


def test_a_crusher_winds_the_cycles_it_was_asked_for(system):
    _, times, amplitudes = pp.make_crusher(4.0, 5e-3, "z", system=system)
    assert float(np.trapezoid(amplitudes, times)) == pytest.approx(4.0 / 5e-3)


@pytest.mark.parametrize("grad_start", [0.0, 3e5, -2e5])
def test_a_crusher_can_ride_a_gradient_that_is_already_running(system, grad_start):
    """The reason it wraps ``make_extended_trapezoid_area`` and not ``make_trapezoid``."""
    _, times, amplitudes = pp.make_crusher(
        4.0, 5e-3, "z", grad_start=grad_start, system=system
    )
    assert amplitudes[0] == pytest.approx(grad_start)
    assert float(np.trapezoid(amplitudes, times)) == pytest.approx(4.0 / 5e-3, rel=1e-6)


def test_a_crusher_refuses_a_meaningless_voxel(system):
    for cycles, voxel in ((4.0, 0.0), (4.0, -1e-3), (0.0, 5e-3)):
        with pytest.raises(ValueError):
            pp.make_crusher(cycles, voxel, "z", system=system)


def test_the_phase_encode_template_depends_only_on_resolution(system):
    """Field of view and matrix size cancel, so only their ratio may matter."""
    coarse = pp.make_phase_encoding("y", 0.22 / 64, system=system)
    same_ratio = pp.make_phase_encoding("y", 0.44 / 128, system=system)
    assert coarse.area == pytest.approx(same_ratio.area)
    assert coarse.area == pytest.approx(64 / (2 * 0.22))


def test_a_blip_of_r_cells_is_an_acceleration_of_r(system):
    one = pp.make_phase_blip("y", 0.24, steps=1, system=system)
    three = pp.make_phase_blip("y", 0.24, steps=3, system=system)
    assert one.area == pytest.approx(1 / 0.24)
    assert three.area == pytest.approx(3 * one.area)
    assert pp.make_phase_blip("y", 0.24, steps=-1, system=system).area == pytest.approx(-one.area)


def test_a_blip_refuses_to_stand_still(system):
    with pytest.raises(ValueError):
        pp.make_phase_blip("y", 0.24, steps=0, system=system)


# -- ADC timing -----------------------------------------------------------


@pytest.mark.parametrize("num_samples", [64, 96, 128, 200, 256])
@pytest.mark.parametrize(("grad_raster", "adc_raster"), [(10e-6, 100e-9), (20e-6, 2e-6), (4e-6, 2e-6)])
def test_the_adc_lands_on_both_rasters_at_once(num_samples, grad_raster, adc_raster):
    """The whole point: the readout can then sit exactly over the flat top."""
    dwell, duration = pp.calc_adc_timing(
        num_samples, 3.7e-6, grad_raster_time=grad_raster, adc_raster_time=adc_raster
    )
    assert dwell / adc_raster == pytest.approx(round(dwell / adc_raster))
    assert duration / grad_raster == pytest.approx(round(duration / grad_raster))
    assert duration == pytest.approx(num_samples * dwell)


def test_a_minimum_duration_is_respected():
    _, duration = pp.calc_adc_timing(
        64, 1e-6, grad_raster_time=10e-6, adc_raster_time=100e-9, min_readout_duration=1e-3
    )
    assert duration >= 1e-3


# -- RF -------------------------------------------------------------------


def test_an_slr_pulse_comes_back_shaped_like_a_sinc_pulse_does(system):
    alone = pp.make_slr_pulse(np.deg2rad(90), system=system)
    assert isinstance(alone, cxx.Event)

    rf, gz, gz_reph = pp.make_slr_pulse(
        np.deg2rad(90), slice_thickness=5e-3, return_gz=True, system=system
    )
    assert (rf.type, gz.channel, gz_reph.channel) == ("rf", "z", "z")
    assert pp.make_sigpy_pulse is pp.make_slr_pulse


def test_an_slr_pulse_needs_a_slice_to_select_before_it_will_select_one(system):
    with pytest.raises(ValueError):
        pp.make_slr_pulse(np.deg2rad(90), return_gz=True, system=system)


def test_an_slr_pulse_has_the_bandwidth_it_was_designed_for(system):
    for time_bw_product, duration in [(4, 4e-3), (6, 3e-3)]:
        rf = pp.make_slr_pulse(
            np.deg2rad(20), duration=duration, time_bw_product=time_bw_product, system=system
        )
        assert pp.calc_rf_bandwidth(rf) == pytest.approx(time_bw_product / duration, rel=0.15)


def test_a_spsp_pulse_rides_an_alternating_gradient_that_ends_balanced():
    fast = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=180, slew_unit="T/m/s")
    rf, gz, gz_reph = pp.make_spsp_pulse(
        np.deg2rad(30), 10e-3, 300.0, n_subpulses=12, system=fast
    )
    assert rf.type == "rf"
    # Alternating lobes over an even count cancel to zero net area.
    assert float(np.trapezoid(gz.waveform, gz.tt)) == pytest.approx(0.0, abs=1e-6 * abs(gz.waveform).max())
    assert gz_reph.channel == "z"


def test_a_spsp_pulse_refuses_what_the_hardware_cannot_do(system):
    with pytest.raises(ValueError):
        pp.make_spsp_pulse(np.deg2rad(30), 1e-4, 5000.0, n_subpulses=12, system=system)
    with pytest.raises(ValueError):
        pp.make_spsp_pulse(np.deg2rad(30), 10e-3, 300.0, n_subpulses=2, system=system)


def test_sms_bands_are_symmetric_and_include_zero(system):
    base = pp.make_sinc_pulse(flip_angle=np.deg2rad(30), duration=2e-3, system=system)
    for count, expected in ((3, [-1000.0, 0.0, 1000.0]), (2, [0.0, 1000.0]), (1, [0.0])):
        _, offsets, _ = pp.make_sms_pulse(base, count, 1000.0)
        assert offsets.tolist() == expected


def test_sms_power_is_power_not_amplitude(system):
    """A sideband at power p contributes sqrt(p) of amplitude."""
    base = pp.make_sinc_pulse(flip_angle=np.deg2rad(30), duration=2e-3, system=system)
    _, _, weights = pp.make_sms_pulse(base, 3, 1000.0, sideband_power=4.0)
    assert abs(weights[0]) == pytest.approx(2.0)
    assert abs(weights[1]) == pytest.approx(1.0)


def test_sms_leaves_the_pulse_it_was_given_alone(system):
    base = pp.make_sinc_pulse(flip_angle=np.deg2rad(30), duration=2e-3, system=system)
    before = np.array(base.signal, copy=True)
    banded, _, _ = pp.make_sms_pulse(base, 3, 1000.0)

    np.testing.assert_array_equal(base.signal, before)
    assert isinstance(banded, cxx.Event)
    assert not np.allclose(banded.signal, before)


# -- traj_to_grad ---------------------------------------------------------


def _spiral(turns=8, points=2000, radius=250.0):
    theta = np.linspace(0, turns * np.pi, points)
    r = np.linspace(0, radius, points)
    return np.stack([r * np.cos(theta), r * np.sin(theta)])


def test_the_time_optimal_gradient_obeys_both_limits_everywhere(system):
    k = _spiral()
    g, slew = pp.traj_to_grad(k, system=system)

    assert np.linalg.norm(g, axis=0).max() <= system.max_grad * 1.001
    assert np.linalg.norm(slew, axis=0).max() <= system.max_slew * 1.05


def test_resampling_the_same_path_does_not_change_the_gradient(system):
    """Under ``time_optimal`` the samples carry geometry, not timing."""
    coarse, _ = pp.traj_to_grad(_spiral(points=1000), system=system)
    fine, _ = pp.traj_to_grad(_spiral(points=4000), system=system)

    assert abs(coarse.shape[1] - fine.shape[1]) <= 2
    shared = min(coarse.shape[1], fine.shape[1])
    np.testing.assert_allclose(coarse[:, :shared], fine[:, :shared], rtol=0.02, atol=1e-3 * np.abs(coarse).max())


def test_differentiating_reproduces_upstream_exactly(system):
    import pypulseq as upstream

    k = _spiral(points=200)
    ours = pp.traj_to_grad(k, system.grad_raster_time, time_optimal=False, system=system)
    theirs = upstream.traj_to_grad(k, system.grad_raster_time)

    np.testing.assert_array_equal(ours[0], theirs[0])
    np.testing.assert_array_equal(ours[1], theirs[1])


def test_a_single_axis_trajectory_keeps_its_shape(system):
    k = np.linspace(0, 400, 500)
    g, slew = pp.traj_to_grad(k, system=system)
    assert g.ndim == slew.ndim == 1


def test_traj_to_grad_keeps_upstreams_parameter_names_first():
    import inspect

    import pypulseq as upstream

    ours = list(inspect.signature(pp.traj_to_grad).parameters)
    theirs = list(inspect.signature(upstream.traj_to_grad).parameters)
    assert ours[: len(theirs)] == theirs


# -- sampling -------------------------------------------------------------


def test_the_uniform_mask_samples_every_rth_line_plus_a_centre():
    mask = pp.make_uniform_mask(64, 2, calibration=8)
    assert int(mask.sum()) == 36
    assert mask[np.arange(0, 64, 2)].all()
    assert mask[28:36].all()


def test_the_uniform_mask_is_the_only_mode_a_single_axis_accepts():
    """The other three need two phase-encode axes to spread points over."""
    assert pp.make_uniform_mask(64, 2).ndim == 1
    for two_dimensional in (pp.make_random_mask, pp.make_poisson_disc_mask):
        with pytest.raises((ValueError, IndexError, TypeError)):
            two_dimensional(64, 2.0)


def test_every_traversal_order_visits_each_view_once():
    for order in ("sequential", "reverse", "interleaved", "center_out", "outside_in", "random"):
        visited = pp.calc_traversal_order(16, order)
        assert sorted(visited.tolist()) == list(range(16)), order


def test_centre_out_starts_at_the_centre():
    assert pp.calc_traversal_order(6, "center_out")[0] in (2, 3)


def test_golden_angles_never_repeat_and_tiny_ones_step_less_far():
    golden = pp.calc_golden_angles(64)
    tiny = pp.calc_tiny_golden_angles(64, index=4)

    assert len(np.unique(np.round(golden, 9))) == 64
    def first_step(angles):
        return abs(np.diff(angles)[0]) % (2 * np.pi)

    assert first_step(tiny) < first_step(golden)


def test_raga_angles_come_from_a_finite_equidistant_support():
    """What makes RAGA reproducible bin to bin, unlike a true golden angle."""
    angles = pp.calc_raga_angles(200, approximation_order=8)
    support = np.unique(np.round(angles, 9))
    assert len(support) < 200
    spacing = np.diff(np.sort(support))
    assert spacing.max() == pytest.approx(spacing.min(), rel=1e-6)


def test_uniform_angles_span_the_half_circle():
    angles = pp.calc_uniform_angles(4)
    np.testing.assert_allclose(np.rad2deg(angles), [0.0, 90.0, 180.0, 270.0])


def test_an_ordering_covers_the_coordinates_it_was_given_exactly_once():
    coords = np.stack(np.meshgrid(np.arange(8), np.arange(4), indexing="ij"), -1).reshape(-1, 2)
    for order in (
        pp.make_linear_order,
        pp.make_centric_order,
        pp.make_radial_order,
        pp.make_radial_adaptive_order,
        pp.make_shuffling_order,
    ):
        shots = order(coords, 4)
        flat = [index for shot in shots for index in shot]
        assert sorted(flat) == list(range(len(coords))), order.__name__


# -- arbitrary-gradient duration ------------------------------------------


@pytest.mark.parametrize("grad_raster", [10e-6, 20e-6])
def test_a_uniformly_sampled_gradient_lasts_one_raster_per_sample(grad_raster):
    system = pp.Opts(grad_raster_time=grad_raster)
    waveform = np.full(10, 1e5)
    gradient = pp.make_arbitrary_grad(channel="x", waveform=waveform, system=system)
    assert pp.calc_duration(gradient) == pytest.approx(waveform.size * grad_raster)


@pytest.mark.parametrize("grad_raster", [10e-6, 20e-6])
def test_an_extended_trapezoid_lasts_until_its_last_vertex(grad_raster):
    """Its times are vertices, not sample centres, so nothing follows the last.

    Reading them as sample centres invents half a step of tail out of the
    closing ramp, which lands the duration off the gradient raster -- and
    every delay `align` then computes against it is illegal.
    """
    system = pp.Opts(grad_raster_time=grad_raster)
    times = np.array([0.0, 2e-4, 1e-3, 1.2e-3])
    gradient = pp.make_extended_trapezoid(
        channel="x", amplitudes=np.array([0.0, 1e5, 1e5, 0.0]), times=times, system=system
    )
    assert pp.calc_duration(gradient) == pytest.approx(times[-1])


@pytest.mark.parametrize("grad_raster", [10e-6, 20e-6])
def test_a_solved_bridge_lands_on_the_gradient_raster(grad_raster):
    """The case the two conventions are told apart for: a bridged spoiler."""
    system = pp.Opts(
        max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s", grad_raster_time=grad_raster
    )
    gradient, _, _ = pp.make_extended_trapezoid_area(
        area=-5000.0, channel="x", grad_start=0.0, grad_end=4.5e5, system=system
    )
    steps = pp.calc_duration(gradient) / grad_raster
    assert steps == pytest.approx(round(steps))
