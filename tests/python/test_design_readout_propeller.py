"""PROPELLER readouts.

A blade is an EPI train, so what is tested here is only what makes it a blade:
that it straddles the centre of k-space without being placed there, that one
waveform serves every angle, and that the rotation is what puts each blade
where it belongs.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import pulserver.design as design
import pulserver.pypulseq as pp

FOV = 0.22
MATRIX = 64
FOV_Z = 0.12
MATRIX_Z = 16
WIDTH = 8


@pytest.fixture
def system():
    return pp.Opts(
        max_grad=50,
        grad_unit="mT/m",
        max_slew=180,
        slew_unit="T/m/s",
        rf_dead_time=100e-6,
        rf_ringdown_time=20e-6,
        adc_dead_time=10e-6,
    )


@pytest.fixture
def excitation(system):
    return design.SpatialSelectiveExcitation(system, 60.0, 3e-3)


@pytest.fixture
def slab(system):
    return design.SpatialSelectiveExcitation(system, 15.0, FOV_Z, is_slab=True)


def readout(system, excitation, **kwargs):
    kwargs.setdefault("fov", FOV)
    kwargs.setdefault("matrix", MATRIX)
    kwargs.setdefault("blade_width", WIDTH)
    return design.PropellerReadout2D(
        system, excitation.rf, excitation.gz, excitation.gz_reph, **kwargs
    )


def stack(system, slab, **kwargs):
    kwargs.setdefault("fov", FOV)
    kwargs.setdefault("matrix", MATRIX)
    kwargs.setdefault("blade_width", WIDTH)
    kwargs.setdefault("fov_z", FOV_Z)
    kwargs.setdefault("matrix_z", MATRIX_Z)
    return design.PropellerStackReadout(system, slab.rf, slab.gz, **kwargs)


def play(system, blade, angles, partitions=None):
    """A set of blades, written the way the docstring says to."""
    seq = pp.Sequence(system)
    for index, angle in enumerate(angles):
        rotation = pp.make_rotation(Rotation.from_euler("z", angle))
        encodes = [pp.scale_grad(blade.gy_pre, blade.blade_start)]
        if partitions is not None:
            encodes.append(pp.scale_grad(blade.gz_pre, partitions[index]))
        seq.add_block(blade.rf, blade.gz)
        seq.add_block(blade.gx_pre, *encodes, rotation)
        for line in range(blade.etl):
            step = blade.gy_blips[line]
            seq.add_block(
                blade.gx[line], blade.adc, *(() if step is None else (step,)), rotation
            )
        seq.add_block(blade.gx_spoil, rotation)
    return seq


def blades(module, seq, angles):
    """Each blade's k-space, shaped ``(blades, lines, samples, 3)``."""
    traj = np.asarray(seq.calculate_kspace(dense=False)[0]).T
    return traj.reshape(len(angles), module.etl, module.n_samples, 3)


# ----------------------------------------------------------------------
# The blade
# ----------------------------------------------------------------------


def test_a_blade_is_an_epi_train_as_wide_as_it_was_asked_for(system, excitation):
    blade = readout(system, excitation)
    assert blade.etl == WIDTH
    assert blade.blade_width == WIDTH
    assert list(blade.order[:, 0]) == list(range(WIDTH))
    assert blade.check_timing()[0]


def test_a_blade_samples_the_centre_of_k_space(system, excitation):
    """Every blade acquires the ky = 0 line, which is what makes it
    self-navigating -- and why nothing places a blade: only the angle varies.
    The lines follow the EPI convention, from -floor(w / 2) upward."""
    blade = readout(system, excitation)
    seq = play(system, blade, [0.0])
    offsets = blades(blade, seq, [0.0])[0, :, 0, 1] * FOV
    assert offsets == pytest.approx(np.arange(WIDTH) - WIDTH // 2, abs=1e-6)
    assert np.min(np.abs(offsets)) == pytest.approx(0.0, abs=1e-6)


def test_a_blade_is_read_across_its_full_width(system, excitation):
    blade = readout(system, excitation)
    seq = play(system, blade, [0.0])
    traced = blades(blade, seq, [0.0])[0]
    assert abs(traced[0, -1, 0]) == pytest.approx(0.5 * MATRIX / FOV, rel=0.05)
    assert traced[0, 0, 0] == pytest.approx(-traced[0, -1, 0], rel=1e-9)


def test_the_blade_start_is_the_same_for_every_blade(system, excitation):
    blade = readout(system, excitation)
    assert blade.blade_start == pytest.approx(-(WIDTH // 2) / (WIDTH / 2))


# ----------------------------------------------------------------------
# Turning it
# ----------------------------------------------------------------------


def test_a_rotation_turns_the_whole_blade(system, excitation):
    """One waveform, many angles: the blade is identical in its own frame."""
    blade = readout(system, excitation)
    angles = blade.blade_angles[:4]
    traced = blades(blade, play(system, blade, angles), angles)
    for index, angle in enumerate(angles):
        across = np.array([-np.sin(angle), np.cos(angle)])
        offsets = traced[index, :, 0, :2] @ across * FOV
        assert offsets == pytest.approx(np.arange(WIDTH) - WIDTH // 2, abs=1e-5)


def test_the_blades_point_where_their_angles_say(system, excitation):
    blade = readout(system, excitation)
    angles = blade.blade_angles[:4]
    traced = blades(blade, play(system, blade, angles), angles)
    for index, angle in enumerate(angles):
        sweep = traced[index, 0, -1, :2] - traced[index, 0, 0, :2]
        assert np.arctan2(sweep[1], sweep[0]) == pytest.approx(angle, abs=1e-6)


def test_a_rotation_leaves_the_partition_axis_alone(system, slab):
    """Turning about z is what lets a stack of propellers encode through it."""
    blade = stack(system, slab)
    angles = blade.blade_angles[:3]
    partitions = [-0.5, 0.0, 0.75]
    traced = blades(blade, play(system, blade, angles, partitions), angles)
    for index, scale in enumerate(partitions):
        expected = scale * 0.5 * MATRIX_Z / FOV_Z
        assert traced[index, :, 0, 2] == pytest.approx(expected, abs=1e-6)


# ----------------------------------------------------------------------
# The blade set
# ----------------------------------------------------------------------


def test_the_default_count_samples_the_rim_at_nyquist(system, excitation):
    blade = readout(system, excitation)
    assert blade.n_blades == int(np.ceil(np.pi * MATRIX / (2 * WIDTH)))


def test_a_wider_blade_needs_fewer_of_them(system, excitation):
    narrow = readout(system, excitation, blade_width=4)
    wide = readout(system, excitation, blade_width=16)
    assert narrow.n_blades > wide.n_blades
    assert narrow.etl < wide.etl


def test_uniform_blades_divide_half_a_turn(system, excitation):
    """Half, not a whole one: a blade turned by pi covers the strip it already did."""
    blade = readout(system, excitation, n_blades=6)
    assert blade.blade_angles == pytest.approx(np.arange(6) * np.pi / 6)


def test_golden_blades_stay_spread_out_over_any_prefix(system, excitation):
    whole = readout(system, excitation, n_blades=16, scheme="golden")
    prefix = readout(system, excitation, n_blades=8, scheme="golden")
    assert whole.blade_angles[:8] == pytest.approx(prefix.blade_angles)
    spread = np.sort(whole.blade_angles[:8] % np.pi)
    assert np.ptp(np.diff(spread)) < 0.5 * np.pi / 8


def test_a_supplied_set_of_angles_is_used_as_it_stands(system, excitation):
    blade = readout(system, excitation, angles=[0.0, 0.3, 1.1])
    assert blade.blade_angles == pytest.approx([0.0, 0.3, 1.1])
    assert blade.n_blades == 3


def test_supplying_angles_and_asking_for_a_count_is_refused(system, excitation):
    with pytest.raises(ValueError, match="alternatives"):
        readout(system, excitation, n_blades=4, angles=[0.0, 1.0])


def test_an_unknown_scheme_is_refused(system, excitation):
    with pytest.raises(ValueError, match="scheme must be"):
        readout(system, excitation, scheme="phyllotaxis")


def test_a_blade_narrower_than_one_line_is_refused(system, excitation):
    with pytest.raises(ValueError, match="blade_width"):
        readout(system, excitation, blade_width=0)


# ----------------------------------------------------------------------
# The stack
# ----------------------------------------------------------------------


def test_a_stack_encodes_partitions_and_blips_nothing_on_z(system, slab):
    blade = stack(system, slab)
    assert blade.gz_pre.channel == "z"
    assert set(blade.gz_blips) == {None}
    assert not blade.order[:, 1].any()


def test_a_stack_without_a_slab_to_encode_is_refused(system, slab):
    with pytest.raises(ValueError, match="fov_z and matrix_z"):
        design.PropellerStackReadout(
            system, slab.rf, slab.gz, fov=FOV, matrix=MATRIX, blade_width=WIDTH
        )
