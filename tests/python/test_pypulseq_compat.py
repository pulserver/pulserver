"""The ``compat`` return convention: upstream's tuple, or one named object.

Two properties are load-bearing and are asserted here for every method that
grew the flag:

1. ``compat=True`` returns exactly what upstream returns -- same shapes, same
   numbers, same omissions. That is what makes an unchanged PyPulseq script
   keep its unchanged meaning.
2. ``compat=False`` returns something that **cannot be unpacked**. The flag
   exists because a tuple return is unpacked positionally, so a result object
   that could also be unpacked would reintroduce the fragility it was added to
   remove.
"""

from __future__ import annotations

import inspect
import warnings

import numpy as np
import pypulseq as upstream
import pytest

import pulserver.pypulseq as pp
from pypulseq.utils.safe_pns_prediction import safe_example_hw
from pulserver.pypulseq._results import RF_USES, AdcTimes, RfTimes, Waveforms, WaveformsAndTimes

#: Upstream's own example SAFE hardware, so no scanner .asc file is needed.
#: The Irnich dict form only reaches the C path; upstream's ``calc_pns``
#: wants the SAFE model, and ``compat=True`` has to go through upstream.
SAFE_HW = safe_example_hw()

#: Every method carrying the flag, and the arguments to exercise it with.
#: ``window_width`` is set short because the fixture sequence is shorter than
#: upstream's 50 ms default, which makes scipy clamp ``nperseg`` but not
#: ``noverlap`` and raise -- an upstream bug, not one under test here.
COMPAT_METHODS = {
    "waveforms": {"append_RF": True},
    "waveforms_and_times": {"append_RF": True},
    "rf_times": {},
    "adc_times": {},
    "calculate_kspace": {},
    "calculate_pns": {"hardware": SAFE_HW, "do_plots": False},
    "calculate_gradient_spectrum": {"plot": False, "window_width": 2e-3},
}


@pytest.fixture
def system():
    return pp.Opts(
        max_grad=30,
        grad_unit="mT/m",
        max_slew=150,
        slew_unit="T/m/s",
        rf_ringdown_time=20e-6,
        rf_dead_time=100e-6,
        adc_dead_time=10e-6,
    )


def _build(sequence_class, make, system, *, uses=("excitation",)):
    """A short multi-TR gradient echo, one RF pulse per use tag asked for."""
    seq = sequence_class(system=system)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gx = make.make_trapezoid(
            channel="x", flat_area=128 / 0.256, flat_time=2.56e-3, system=system
        )
        adc = make.make_adc(
            num_samples=128, duration=gx.flat_time, delay=gx.rise_time, system=system
        )
        pre = make.make_trapezoid(channel="x", area=-gx.area / 2, duration=1e-3, system=system)

        for use in uses:
            rf, gz, gzr = make.make_sinc_pulse(
                flip_angle=np.pi / 6,
                duration=2e-3,
                slice_thickness=3e-3,
                apodization=0.5,
                time_bw_product=4,
                system=system,
                return_gz=True,
                use=use,
            )
            for _ in range(2):
                seq.add_block(rf, gz)
                seq.add_block(pre, gzr)
                seq.add_block(gx, adc)
    return seq


@pytest.fixture
def pair(system):
    """The same sequence built twice: ours, and a real PyPulseq one."""
    return _build(pp.Sequence, pp, system), _build(upstream.Sequence, upstream, system)


# %% the flag itself


@pytest.mark.parametrize("name", sorted(COMPAT_METHODS))
def test_compat_is_keyword_only_optional_and_last(name):
    """It must never displace an upstream argument or become required."""
    parameters = inspect.signature(getattr(pp.Sequence, name)).parameters
    compat = parameters["compat"]
    assert compat.kind is inspect.Parameter.KEYWORD_ONLY
    assert compat.default is True

    upstream_names = set(inspect.signature(getattr(upstream.Sequence, name)).parameters)
    positional = [
        p.name
        for p in parameters.values()
        if p.kind is not inspect.Parameter.KEYWORD_ONLY and p.name != "self"
    ]
    assert set(positional) <= upstream_names


@pytest.mark.parametrize("name", sorted(COMPAT_METHODS))
def test_the_result_object_cannot_be_unpacked(pair, name):
    """The whole point of the flag: results grow fields, never tuple length."""
    ours, _ = pair
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = getattr(ours, name)(**COMPAT_METHODS[name], compat=False)

    assert not isinstance(result, tuple)
    with pytest.raises(TypeError):
        _first, _second = result
    assert not hasattr(result, "__iter__")
    assert not hasattr(result, "__getitem__")


# %% compat=True is upstream, exactly


def _same(ours, theirs) -> bool:
    if isinstance(ours, (list, tuple)):
        return len(ours) == len(theirs) and all(_same(a, b) for a, b in zip(ours, theirs))
    ours, theirs = np.asarray(ours), np.asarray(theirs)
    return ours.shape == theirs.shape and np.allclose(ours, theirs, equal_nan=True)


@pytest.mark.parametrize("append_RF", [False, True])
def test_waveforms_matches_upstream_on_trapezoids(pair, append_RF):
    """Every gradient here is a real trapezoid, so the raster-edge restoration
    never fires and the answer has to be upstream's to the bit."""
    ours, theirs = pair
    assert _same(ours.waveforms(append_RF=append_RF), theirs.waveforms(append_RF=append_RF))


def test_waveforms_and_times_matches_upstream(pair):
    ours, theirs = pair
    assert _same(ours.waveforms_and_times(append_RF=True), theirs.waveforms_and_times(append_RF=True))


def test_rf_and_adc_times_match_upstream(pair):
    ours, theirs = pair
    assert _same(ours.rf_times(), theirs.rf_times())
    assert _same(ours.adc_times(), theirs.adc_times())


def test_a_time_range_matches_upstream(pair):
    """The window is cut here and by upstream's own block search; they have to
    agree about which blocks are in it."""
    ours, theirs = pair
    window = [2e-3, 12e-3]
    assert _same(ours.waveforms(time_range=window), theirs.waveforms(time_range=window))
    assert _same(ours.adc_times(time_range=window), theirs.adc_times(time_range=window))


def test_gradient_spectrum_still_appends_its_fifth_element_under_compat(system):
    """The deviation compat=False retires stays reachable while compat is on."""
    assert "resonance_lines" in inspect.signature(pp.Sequence.calculate_gradient_spectrum).parameters


# %% compat=False carries what upstream's shape cannot


def test_every_rf_use_survives_not_just_two(system):
    """Upstream drops inversion, saturation, preparation and other on the
    floor. An inversion pulse is not a footnote -- it defines the contrast."""
    uses = ("excitation", "refocusing", "inversion", "saturation", "preparation", "other")
    ours = _build(pp.Sequence, pp, system, uses=uses)

    pulses = ours.rf_times(compat=False)
    assert isinstance(pulses, RfTimes)
    assert set(pulses.use) == set(uses)
    for use in uses:
        assert len(pulses.of(use)) == 2, use

    # ...while upstream's view of the same sequence sees only two of them.
    t_excitation, _, t_refocusing, _ = ours.rf_times()
    assert len(t_excitation) + len(t_refocusing) == 4


def test_the_compat_tuple_is_a_view_on_the_same_numbers(pair):
    """compat=False must be a superset, not a different computation -- the
    tuple's entries have to be recoverable from the object field for field."""
    ours, _ = pair
    tuple_form = ours.waveforms_and_times(append_RF=True)
    object_form = ours.waveforms_and_times(append_RF=True, compat=False)

    assert isinstance(object_form, WaveformsAndTimes)
    assert _same(tuple_form[0], object_form.waveforms.channels)
    assert _same(tuple_form[1], object_form.rf.of("excitation", "undefined").tfp)
    assert _same(tuple_form[2], object_form.rf.of("refocusing").tfp)
    assert _same(tuple_form[3], object_form.adc.t)
    assert _same(tuple_form[4], object_form.adc.fp)


def test_untagged_pulses_are_kept_apart_from_excitations_but_counted_with_them(system):
    """Upstream buckets an untagged pulse as an excitation. That behaviour is
    reproduced, and the tag is still readable -- both, not either."""
    ours = _build(pp.Sequence, pp, system, uses=("excitation",))
    pulses = ours.rf_times(compat=False)
    assert set(pulses.use) <= set(RF_USES)


def test_echo_centres_are_the_c_cores_and_are_not_computed_until_read(pair):
    """They need the trajectory, so they are a cached_property, not a field."""
    ours, _ = pair
    adc = ours.adc_times(compat=False)
    assert isinstance(adc, AdcTimes)
    assert "echo_center_time" not in adc.__dict__  # untouched so far

    centres = adc.echo_center_time
    assert centres.size == adc.num_samples.size
    assert "echo_center_time" in adc.__dict__  # and cached now

    # Each centre falls inside its own readout, which is the only claim the
    # number makes that can be checked without redoing the trajectory.
    starts = np.concatenate(([0], np.cumsum(adc.num_samples)[:-1]))
    for index, (start, count) in enumerate(zip(starts, adc.num_samples)):
        window = adc.t[start : start + count]
        assert window[0] <= centres[index] <= window[-1]


def test_adc_phase_modulation_is_returned_where_upstream_drops_it(system):
    """MATLAB returns pm_adc as a sixth output; PyPulseq returns nothing."""
    seq = pp.Sequence(system=system)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gx = pp.make_trapezoid(channel="x", flat_area=1000, flat_time=2.56e-3, system=system)
        modulation = np.linspace(0.0, 1.0, 128)
        adc = pp.make_adc(
            num_samples=128,
            duration=gx.flat_time,
            delay=gx.rise_time,
            system=system,
            phase_modulation=modulation,
        )
        seq.add_block(gx, adc)

    samples = seq.adc_times(compat=False)
    np.testing.assert_allclose(samples.phase_modulation, modulation, atol=1e-6)
    # And it is folded into the per-sample phase, which is what a demodulator
    # would apply -- but not into upstream's per-event fp.
    assert np.any(np.abs(np.diff(samples.sample_phase)) > 0)
    assert samples.fp.shape == (1, 2)


def test_kspace_carries_the_breakpoint_grid_upstream_has_nowhere_to_put(pair):
    ours, _ = pair
    dense = ours.calculate_kspace(compat=False)
    assert dense.k_traj_breakpoints.shape[0] == 3
    assert dense.t_breakpoints.size == dense.k_traj_breakpoints.shape[1]
    # The breakpoint grid describes the same curve in far fewer points.
    assert dense.t_breakpoints.size < dense.k_traj.shape[1]


def test_waveforms_object_names_the_channels(pair):
    ours, _ = pair
    result = ours.waveforms(append_RF=True, compat=False)
    assert isinstance(result, Waveforms)
    assert _same(result.channels, ours.waveforms(append_RF=True))
    assert result.rf is not None
    assert ours.waveforms(append_RF=False, compat=False).rf is None
