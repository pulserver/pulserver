"""The refocusing pulses, hard and slice-selective.

A refocusing pulse is judged by two things: how completely it inverts, and
whether its crushers are symmetric enough to leave the refocused pathway alone
while destroying the FID an imperfect inversion leaves behind. Both are
measured here -- the first by simulating the pulse, the second by integrating
the gradient either side of the pulse centre.
"""

from __future__ import annotations

import numpy as np
import pytest

import pulserver.design as design
import pulserver.pypulseq as pp


@pytest.fixture
def system():
    return pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")


def _moment(event, start, stop):
    """Zeroth moment of a gradient event between two times (s)."""
    if event.type == "trap":
        times = np.array(
            [
                0.0,
                event.rise_time,
                event.rise_time + event.flat_time,
                event.rise_time + event.flat_time + event.fall_time,
            ]
        )
        amplitudes = np.array([0.0, event.amplitude, event.amplitude, 0.0])
    else:
        times, amplitudes = (
            np.asarray(event.tt, dtype=float),
            np.asarray(event.waveform),
        )
    times = times + float(event.delay)

    grid = np.unique(np.concatenate([times, [start, stop]]))
    grid = grid[(grid >= start) & (grid <= stop)]
    return float(np.trapezoid(np.interp(grid, times, amplitudes), grid))


# ----------------------------------------------------------------------
# What the pulse does
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory",
    [
        lambda system: design.NonSelectiveRefocusing(system),
        lambda system: design.SpatialSelectiveRefocusing(system, 5e-3),
    ],
)
def test_a_refocusing_pulse_inverts(system, factory):
    response = factory(system).sim_rf(compat=False)
    assert float(np.min(response.mz_z)) < -0.99


@pytest.mark.parametrize(
    "factory",
    [
        lambda system: design.NonSelectiveRefocusing(system),
        lambda system: design.SpatialSelectiveRefocusing(system, 5e-3),
    ],
)
def test_it_is_phased_a_quarter_turn_from_the_excitation(system, factory):
    """CPMG: the refocusing axis is perpendicular to the excitation's."""
    assert float(factory(system).rf_ref.phase_offset) == pytest.approx(np.pi / 2)


def test_the_selective_one_says_it_is_a_refocusing_pulse(system):
    """The trajectory core negates accumulated k here, so ``use`` is not cosmetic."""
    module = design.SpatialSelectiveRefocusing(system, 5e-3)
    assert module.rf_ref.use == "refocusing"
    assert module.check_timing()[0]


def test_the_selective_profile_is_confined_to_the_slice(system):
    """Same bandwidth, so only the selection plateau tells the two apart."""
    thin = design.SpatialSelectiveRefocusing(system, 2e-3, spoiling_cycles=0.0)
    wide = design.SpatialSelectiveRefocusing(system, 8e-3, spoiling_cycles=0.0)
    assert float(thin.gz.amplitude) == pytest.approx(4.0 * float(wide.gz.amplitude))


# ----------------------------------------------------------------------
# The crushers
# ----------------------------------------------------------------------


def test_the_selective_crushers_are_bridged_into_one_gradient(system):
    module = design.SpatialSelectiveRefocusing(system, 5e-3)
    assert len(module.blocks) == 1
    assert module.blocks[0] == (module.rf_ref, module.gz)
    assert module.gz.type == "grad"


def test_the_selective_crushers_are_symmetric_about_the_pulse(system):
    module = design.SpatialSelectiveRefocusing(system, 5e-3, spoiling_cycles=4.0)
    centre = float(module.rf_ref.delay) + float(module.rf_ref.center)
    end = float(module.gz.delay) + float(np.asarray(module.gz.tt)[-1])
    before = _moment(module.gz, 0.0, centre)
    after = _moment(module.gz, centre, end)
    assert before == pytest.approx(after, rel=1e-9)


def test_the_selective_crusher_winds_the_cycles_it_was_asked_for(system):
    """Each side is one crusher plus half the selection plateau."""
    bare = design.SpatialSelectiveRefocusing(system, 5e-3, spoiling_cycles=0.0)
    module = design.SpatialSelectiveRefocusing(
        system, 5e-3, spoiling_cycles=4.0, voxel_size_m=1e-3
    )
    centre = float(module.rf_ref.delay) + float(module.rf_ref.center)
    half_plateau = 0.5 * float(bare.gz.flat_time) * float(bare.gz.amplitude)
    assert _moment(module.gz, 0.0, centre) == pytest.approx(4.0 / 1e-3 + half_plateau)


def test_the_hard_pulse_plays_one_crusher_twice(system):
    """One event, both sides -- which is what makes the two exactly equal."""
    module = design.NonSelectiveRefocusing(system)
    assert len(module.blocks) == 3
    assert module.blocks[0] == module.blocks[2] == (module.gz_spoil,)
    assert not isinstance(module.gz_spoil, list)


def test_asking_for_no_crushers_leaves_the_pulse_bare(system):
    assert len(design.NonSelectiveRefocusing(system, spoiling_cycles=0.0).blocks) == 1
    bare = design.SpatialSelectiveRefocusing(system, 5e-3, spoiling_cycles=0.0)
    assert bare.gz.type == "trap"


# ----------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------


def test_the_hard_pulse_centre_counts_from_the_module_not_the_block(system):
    """A leading crusher is part of the module, so it is part of the offset."""
    bare = design.NonSelectiveRefocusing(system, spoiling_cycles=0.0)
    module = design.NonSelectiveRefocusing(system)
    crusher = pp.calc_duration(module.gz_spoil)
    assert module.center == pytest.approx(bare.center + crusher, abs=1e-9)


def test_the_selective_pulse_sits_on_the_plateau_between_the_crushers(system):
    bare = design.SpatialSelectiveRefocusing(system, 5e-3, spoiling_cycles=0.0)
    module = design.SpatialSelectiveRefocusing(system, 5e-3)
    waveform = np.asarray(module.gz.waveform)
    times = np.asarray(module.gz.tt)
    centre = float(module.rf_ref.delay) + float(module.rf_ref.center)
    assert np.interp(centre, times, waveform) == pytest.approx(float(bare.gz.amplitude))
    assert float(np.max(waveform)) > float(bare.gz.amplitude)


# ----------------------------------------------------------------------
# Composition
# ----------------------------------------------------------------------


def test_a_readout_takes_it_as_the_pulse_that_opens_the_repetition(system):
    """Given a refocusing pulse instead of an excitation, a readout is a spin echo."""
    refocusing = design.SpatialSelectiveRefocusing(system, 5e-3)
    readout = design.LineReadout2D(
        system, refocusing.rf_ref, refocusing.gz, fov=0.22, matrix=128, te=8e-3
    )
    assert readout.echo_time == pytest.approx(8e-3)
    assert readout.check_timing()[0]


# ----------------------------------------------------------------------
# Refusals
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"spoiling_cycles": -1.0}, "spoiling_cycles"),
        ({"voxel_size_m": 0.0}, "voxel_size_m"),
        ({"axis": "w"}, "axis"),
    ],
)
def test_a_hard_refocusing_pulse_checks_its_arguments(system, kwargs, message):
    with pytest.raises(ValueError, match=message):
        design.NonSelectiveRefocusing(system, **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"thickness_m": 0.0}, "thickness_m"),
        ({"thickness_m": 5e-3, "duration_s": 0.0}, "duration_s"),
        ({"thickness_m": 5e-3, "axis": "w"}, "axis"),
    ],
)
def test_a_selective_refocusing_pulse_checks_its_arguments(system, kwargs, message):
    with pytest.raises(ValueError, match=message):
        design.SpatialSelectiveRefocusing(system, **kwargs)
