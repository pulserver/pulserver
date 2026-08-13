"""Fat saturation, and the placement that has to survive the scan's own transform.

The point of the module is not the pulse -- it is that a band put somewhere at
design time stays there when the finished sequence is repositioned. That is
checked end to end: build a scan, transform it, and read back where the band's
pulse and the imaging pulse each ended up.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import pulserver.design as design
import pulserver.pypulseq as pp

FAT_SHIFT_PPM = -3.45


@pytest.fixture
def system():
    return pp.Opts(B0=3.0, max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")


def rf_phase_ramp(reference, moved):
    """Hz of linear phase the placement wrote across a pulse."""
    times = np.asarray(moved.t, dtype=float)
    difference = np.unwrap(
        np.angle(np.asarray(moved.signal)) - np.angle(np.asarray(reference.signal))
    )
    return float(np.polyfit(times, difference, 1)[0] / (2 * np.pi))


# ----------------------------------------------------------------------
# The pulse
# ----------------------------------------------------------------------


def test_it_sits_on_the_fat_resonance(system):
    fatsat = design.FatSaturation(system)
    assert fatsat.freq_offset_hz == pytest.approx(
        FAT_SHIFT_PPM * 1e-6 * system.B0 * system.gamma
    )
    assert float(fatsat.rf_prep.freq_offset) == pytest.approx(fatsat.freq_offset_hz)


def test_the_offset_can_be_measured_rather_than_assumed(system):
    fatsat = design.FatSaturation(system, freq_offset_hz=-420.0)
    assert fatsat.freq_offset_hz == pytest.approx(-420.0)


def test_it_overshoots_the_transverse_plane(system):
    """110 degrees, so a transmit shortfall still leaves fat near null."""
    response = design.FatSaturation(system).sim_rf(compat=False)
    assert float(np.min(response.mz_z)) == pytest.approx(np.cos(np.deg2rad(110.0)), abs=0.02)


def test_it_leaves_water_alone(system):
    """The stopband has to reach the water peak, 3.45 ppm away."""
    fatsat = design.FatSaturation(system)
    response = fatsat.sim_rf(compat=False)
    water = float(np.interp(0.0, response.frequency, response.mz_z))
    assert water > 0.95


def test_the_pulse_duration_follows_the_bandwidth_asked_for(system):
    narrow = design.FatSaturation(system, bandwidth_hz=125.0)
    wide = design.FatSaturation(system, bandwidth_hz=500.0)
    assert pp.calc_duration(narrow.rf_prep) == pytest.approx(
        4.0 * pp.calc_duration(wide.rf_prep), rel=0.02
    )


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------


def test_it_is_a_pulse_and_a_three_axis_spoiler(system):
    fatsat = design.FatSaturation(system)
    assert len(fatsat.blocks) == 2
    assert {event.channel for event in fatsat.blocks[1] if hasattr(event, "channel")} == {
        "x",
        "y",
        "z",
    }


def test_a_global_saturation_needs_no_gradient(system):
    fatsat = design.FatSaturation(system)
    assert not hasattr(fatsat.events, "gz")
    assert fatsat.check_timing()[0]


def test_a_band_gets_a_selection_gradient(system):
    band = design.FatSaturation(system, thickness_m=0.08)
    assert band.gz.channel == "z"
    assert band.check_timing()[0]


def test_the_exemption_rides_the_pulse_block(system):
    """Not a block of its own: that would be a block of dead time."""
    fatsat = design.FatSaturation(system)
    assert [event.type for event in fatsat.blocks[0]] == ["rf", "labelset", "labelset"]
    assert fatsat.seq.get_block(1).labels == [("SET", "NOPOS", 1), ("SET", "NOROT", 1)]


def test_the_exemption_is_cleared_on_the_way_out(system):
    """Pulseq labels are sticky, so an uncleared flag would exempt the whole scan."""
    fatsat = design.FatSaturation(system)
    assert fatsat.seq.get_block(2).labels == [("SET", "NOPOS", 0), ("SET", "NOROT", 0)]


# ----------------------------------------------------------------------
# Placement
# ----------------------------------------------------------------------


def test_an_offset_band_carries_the_phase_that_moves_it(system):
    """A slab moves by a linear phase ramp of gamma * G * dz across the pulse."""
    here = design.FatSaturation(system, thickness_m=0.08)
    moved = design.FatSaturation(system, thickness_m=0.08, position_mm=(0.0, 0.0, 25.0))
    assert rf_phase_ramp(here.rf_prep, moved.rf_prep) == pytest.approx(
        float(here.gz.amplitude) * 0.025
    )


def test_a_tilted_band_carries_a_rotation_extension(system):
    band = design.FatSaturation(
        system, thickness_m=0.08, orientation=Rotation.from_euler("y", 30, degrees=True)
    )
    assert band.seq.get_block(1).rotation is not None
    assert np.allclose(
        band.seq.get_block(1).rotation,
        Rotation.from_euler("y", 30, degrees=True).as_quat(scalar_first=True),
    )


def test_an_orientation_may_be_given_as_a_matrix(system):
    turn = Rotation.from_euler("y", 30, degrees=True)
    as_object = design.FatSaturation(system, thickness_m=0.08, orientation=turn)
    as_matrix = design.FatSaturation(system, thickness_m=0.08, orientation=turn.as_matrix())
    assert np.allclose(
        as_object.seq.get_block(1).rotation, as_matrix.seq.get_block(1).rotation
    )


def test_the_spoiler_is_not_turned_with_the_band(system):
    """It dephases the same either way, and three full-slew lobes cannot be mixed."""
    band = design.FatSaturation(
        system, thickness_m=0.08, orientation=Rotation.from_euler("y", 30, degrees=True)
    )
    assert band.seq.get_block(2).rotation is None
    assert band.check_timing()[0]


# ----------------------------------------------------------------------
# What the exemption is for
# ----------------------------------------------------------------------


def _scan_with(fatsat, system):
    """One repetition: the saturation, then a slice excitation."""
    excitation = design.SpatialSelectiveExcitation(system, 15.0, 5e-3)
    seq = pp.Sequence(system)
    for block in fatsat.blocks:
        seq.add_block(*block)
    for block in excitation.blocks:
        seq.add_block(*block)
    return seq


def test_a_later_transform_moves_the_imaging_pulse_and_not_the_band(system):
    band = design.FatSaturation(system, thickness_m=0.08, position_mm=(0.0, 0.0, 25.0))
    scan = _scan_with(band, system)
    moved = pp.TransformFOV(translation=(0.0, 0.0, 40.0)).apply_to_sequence(scan)

    saturation_before, saturation_after = scan.get_block(1).rf, moved.get_block(1).rf
    imaging_before, imaging_after = scan.get_block(3).rf, moved.get_block(3).rf

    assert rf_phase_ramp(saturation_before, saturation_after) == pytest.approx(0.0, abs=1e-9)
    assert rf_phase_ramp(imaging_before, imaging_after) != pytest.approx(0.0, abs=1.0)


def test_a_later_rotation_turns_the_imaging_pulse_and_not_the_band(system):
    band = design.FatSaturation(
        system, thickness_m=0.08, orientation=Rotation.from_euler("y", 30, degrees=True)
    )
    scan = _scan_with(band, system)
    turned = pp.TransformFOV(
        rotation=Rotation.from_euler("x", 20, degrees=True).as_matrix()
    ).apply_to_sequence(scan)

    assert np.allclose(turned.get_block(1).rotation, scan.get_block(1).rotation)
    assert turned.get_block(3).rotation is not None
    assert scan.get_block(3).rotation is None


def test_the_exemption_does_not_outlive_the_module(system):
    """The clearing block is what stops the flag following the rest of the scan."""
    scan = _scan_with(design.FatSaturation(system), system)
    turned = pp.TransformFOV(
        rotation=Rotation.from_euler("x", 20, degrees=True).as_matrix()
    ).apply_to_sequence(scan)
    assert turned.get_block(3).rotation is not None


# ----------------------------------------------------------------------
# Refusals
# ----------------------------------------------------------------------


def test_a_saturation_of_everything_has_nowhere_to_be_put(system):
    with pytest.raises(ValueError, match="nowhere to be put"):
        design.FatSaturation(system, position_mm=(0.0, 0.0, 25.0))


def test_an_unspoiled_saturation_is_refused(system):
    with pytest.raises(ValueError, match="has to be spoiled"):
        design.FatSaturation(system, spoiling_cycles=0.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"bandwidth_hz": 0.0}, "bandwidth_hz"),
        ({"voxel_size_m": 0.0}, "voxel_size_m"),
        ({"axis": "w"}, "axis"),
        ({"thickness_m": -1.0}, "thickness_m"),
    ],
)
def test_it_checks_its_arguments(system, kwargs, message):
    with pytest.raises(ValueError, match=message):
        design.FatSaturation(system, **kwargs)


def test_baking_the_rotation_into_waveforms_is_not_offered(system):
    with pytest.raises(NotImplementedError):
        design.FatSaturation(
            system,
            thickness_m=0.08,
            orientation=Rotation.from_euler("y", 30, degrees=True),
            use_rotation_extension=False,
        )
