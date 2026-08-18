"""The non-Cartesian readouts, and the interleaves they are built from.

The trajectory designers are checked against the geometry they claim -- where
the path starts, where it ends, how far out it reaches -- and the readout
modules against what a scan loop needs from them: one interleave, oriented by
the loop, with a TE that points at the k = 0 crossing.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import pulserver.design as design
import pulserver.pypulseq as pp
from pulserver.design.readout import _trajectories as trajectories

FOV = 0.22
MATRIX = 128
KMAX = MATRIX / (2 * FOV)


@pytest.fixture
def system():
    return pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")


@pytest.fixture
def excitation(system):
    return design.SpatialSelectiveExcitation(system, 15.0, 5e-3)


@pytest.fixture
def slab(system):
    return design.SpatialSelectiveExcitation(system, 8.0, 0.12, is_slab=True)


def _radius(path):
    return np.linalg.norm(np.asarray(path)[:, :2], axis=1)


# ----------------------------------------------------------------------
# The designed interleaves
# ----------------------------------------------------------------------


def test_a_radial_spoke_runs_edge_to_edge_through_the_centre(system):
    spoke = trajectories.Radial(system, FOV, MATRIX)
    path = spoke.trajectory
    assert path[0, 0] == pytest.approx(-KMAX)
    assert path[-1, 0] == pytest.approx(KMAX)
    assert spoke.has_prewinder and spoke.has_rewinder


@pytest.mark.parametrize(
    ("direction", "starts_at_centre", "ends_at_centre"),
    [("outward", True, False), ("inward", False, True), ("in_out", False, False)],
)
def test_a_spiral_runs_the_direction_it_was_asked_for(
    system, direction, starts_at_centre, ends_at_centre
):
    arm = trajectories.Spiral(system, FOV, MATRIX, 16, direction=direction)
    radius = _radius(arm.trajectory)
    assert bool(radius[0] < 0.05 * KMAX) is starts_at_centre
    assert bool(radius[-1] < 0.05 * KMAX) is ends_at_centre
    assert radius.max() == pytest.approx(KMAX, rel=1e-6)


def test_an_in_out_spiral_crosses_the_centre_halfway(system):
    arm = trajectories.Spiral(system, FOV, MATRIX, 16, direction="in_out")
    radius = _radius(arm.trajectory)
    assert int(np.argmin(radius)) == pytest.approx(radius.size // 2, abs=2)


def test_the_designer_reaches_and_leaves_k_zero_with_bridges(system):
    """Whichever end is off-centre gets a bridge; the centred end does not."""
    outward = trajectories.Spiral(system, FOV, MATRIX, 16, direction="outward")
    inward = trajectories.Spiral(system, FOV, MATRIX, 16, direction="inward")
    assert not outward.has_prewinder and outward.has_rewinder
    assert inward.has_prewinder and not inward.has_rewinder


def test_a_rosette_returns_to_where_it_started(system):
    petals = trajectories.Rosette(system, FOV, 64, petals=5)
    assert not petals.has_prewinder and not petals.has_rewinder
    radius = _radius(petals.trajectory)
    assert radius[0] == pytest.approx(0.0, abs=1e-6 * KMAX)
    assert radius[-1] == pytest.approx(0.0, abs=1e-6 * KMAX)


def test_design_interleaves_sets_the_pitch_not_the_shot_count(system):
    """More nominal interleaves is a more open arm, not more arms."""
    tight = trajectories.Spiral(system, FOV, MATRIX, 8)
    open_ = trajectories.Spiral(system, FOV, MATRIX, 32)
    assert open_.read_duration < tight.read_duration
    assert open_.design_interleaves == 32


# ----------------------------------------------------------------------
# Rotating an interleave
# ----------------------------------------------------------------------


@pytest.mark.parametrize("angle", [0.0, np.pi / 3, -0.7, np.pi])
def test_rotating_an_interleave_turns_its_path_and_nothing_else(system, angle):
    base = trajectories.Spiral(system, FOV, MATRIX, 16)
    turned = base.rotated(angle)

    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    assert np.allclose(
        turned.trajectory[:, :2], base.trajectory[:, :2] @ rotation.T, atol=1e-9
    )
    assert turned.read_duration == pytest.approx(base.read_duration)
    assert turned.adc is base.adc


def test_a_three_axis_interleave_has_no_plane_to_turn_in(system):
    path = trajectories.spiral_trajectory(FOV, 64, 8)
    volume = np.column_stack([path, np.linspace(-10.0, 10.0, path.shape[0])])
    with pytest.raises(ValueError, match="planar interleave"):
        trajectories.Arbitrary(system, volume, matrix=64).rotated(0.5)


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------


def _readout(cls, system, excitation, **kwargs):
    gz = getattr(excitation.events, "gz", None) or getattr(
        excitation.events, "gz_slab", None
    )
    return cls(system, excitation.rf, gz, fov=FOV, matrix=MATRIX, **kwargs)


FAMILIES = [
    (design.RadialReadout2D, {}),
    (design.RadialProjectionReadout, {}),
    (design.SpiralReadout2D, {"design_interleaves": 16}),
    (design.SpiralProjectionReadout, {"design_interleaves": 16}),
    (design.RosetteReadout2D, {"petals": 5}),
]
STACKS = [
    (design.RadialStackReadout, {}),
    (design.SpiralStackReadout, {"design_interleaves": 16}),
]


@pytest.mark.parametrize(
    ("cls", "kwargs"), FAMILIES, ids=lambda value: getattr(value, "__name__", "")
)
def test_every_family_lays_out_valid_pulseq(system, excitation, cls, kwargs):
    readout = _readout(cls, system, excitation, **kwargs)
    assert readout.check_timing()[0]
    assert readout.blocks[0][0] is readout.rf
    assert any(
        getattr(e, "type", "") == "adc" for block in readout.blocks for e in block
    )


@pytest.mark.parametrize(
    ("cls", "kwargs"), STACKS, ids=lambda value: getattr(value, "__name__", "")
)
def test_a_stack_adds_a_partition_encode_and_its_rewinder(system, slab, cls, kwargs):
    readout = _readout(cls, system, slab, fov_z=0.12, matrix_z=32, **kwargs)
    assert readout.gz_pre.channel == "z"
    assert readout.gz_pre.area == pytest.approx(1.0 / (2.0 * (0.12 / 32)))
    assert readout.gz_rew.area == pytest.approx(-readout.gz_pre.area)
    assert readout.check_timing()[0]


def test_a_projection_readout_has_no_partition_to_encode(system, excitation):
    """Dropping the argument would leave a 2D acquisition wearing a 3D protocol."""
    with pytest.raises(ValueError, match="encodes no partition axis"):
        _readout(
            design.RadialProjectionReadout, system, excitation, fov_z=0.12, matrix_z=32
        )


def test_the_echo_time_points_at_the_k_zero_crossing(system, excitation):
    """Read off the integrated gradient, so each family answers for itself."""
    for cls, kwargs in FAMILIES:
        readout = _readout(cls, system, excitation, **kwargs)
        k_traj_adc, _, t_excitation, _, t_adc = readout.calculate_kspace()
        count = int(readout.adc.num_samples)
        radius = np.linalg.norm(k_traj_adc[:2, :count], axis=0)
        nearest = int(np.argmin(radius))
        measured = float(t_adc[nearest]) - float(t_excitation[0])
        assert measured == pytest.approx(
            readout.echo_time, abs=2 * float(readout.adc.dwell) + 1e-9
        ), cls.__name__


def test_an_outward_spiral_starts_at_the_echo(system, excitation):
    readout = _readout(
        design.SpiralReadout2D, system, excitation, design_interleaves=16
    )
    k_traj_adc = readout.calculate_kspace()[0]
    assert np.linalg.norm(k_traj_adc[:2, 0]) < 0.05 * KMAX


# ----------------------------------------------------------------------
# Orientation
# ----------------------------------------------------------------------


def test_the_default_lays_out_one_interleave_for_the_loop_to_turn(system, excitation):
    """One waveform, however many arms the scan plays: the rotation is an event."""
    readout = _readout(
        design.SpiralReadout2D, system, excitation, design_interleaves=16
    )
    assert not isinstance(readout.gx, list)

    seq = pp.Sequence(system)
    angles = pp.calc_golden_angles(4)
    for angle in angles:
        seq.add_block(readout.rf, readout.gz)
        seq.add_block(
            readout.gx,
            readout.gy,
            readout.adc,
            pp.make_rotation(Rotation.from_euler("z", float(angle))),
        )
        seq.add_block(readout.gx_rew, readout.gy_rew)

    assert seq.check_timing()[0]
    count = int(readout.adc.num_samples)
    k_traj_adc = seq.calculate_kspace(dense=False)[0]
    first = k_traj_adc[:2, :count]
    for index, angle in enumerate(angles[1:], start=1):
        turn = np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )
        arm = k_traj_adc[:2, index * count : (index + 1) * count]
        assert np.allclose(arm, turn @ first, atol=1e-6)


def test_explicit_mode_publishes_one_waveform_per_arm(system, excitation):
    angles = np.deg2rad(np.arange(6) * 30.0)
    readout = _readout(
        design.SpiralReadout2D,
        system,
        excitation,
        design_interleaves=16,
        explicit=True,
        angles=angles,
        tr=10e-3,
    )
    assert isinstance(readout.gx, list)
    assert len(readout.gx) == len(readout.gy) == len(angles)
    assert readout.check_timing()[0]


def test_every_explicit_arm_is_the_base_arm_turned(system, excitation):
    angles = np.deg2rad(np.arange(4) * 45.0)
    readout = _readout(
        design.SpiralReadout2D,
        system,
        excitation,
        design_interleaves=16,
        explicit=True,
        angles=angles,
        tr=10e-3,
    )
    k_traj_adc = readout.calculate_kspace()[0]
    count = int(readout.adc.num_samples)
    first = k_traj_adc[:2, :count]
    for index, angle in enumerate(angles[1:], start=1):
        turn = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        arm = k_traj_adc[:2, index * count : (index + 1) * count]
        assert np.allclose(arm, turn @ first, atol=1e-6)


def test_every_explicit_arm_takes_the_same_repetition_time(system, excitation):
    """Re-solving a bridge per angle changes its length; a TR must not follow."""
    angles = np.deg2rad(np.arange(5) * 37.0)
    readout = _readout(
        design.SpiralReadout2D,
        system,
        excitation,
        design_interleaves=16,
        explicit=True,
        angles=angles,
        tr=10e-3,
    )
    assert readout.duration == pytest.approx(10e-3)
    assert readout.seq.duration()[0] == pytest.approx(len(angles) * 10e-3)


def test_a_sampling_pattern_is_not_the_readouts_to_hold(system, excitation):
    with pytest.raises(ValueError, match="sampling pattern"):
        _readout(
            design.SpiralReadout2D,
            system,
            excitation,
            design_interleaves=16,
            angles=np.zeros(4),
        )


def test_explicit_without_angles_says_what_is_missing(system, excitation):
    with pytest.raises(ValueError, match="needs the angles"):
        _readout(
            design.SpiralReadout2D,
            system,
            excitation,
            design_interleaves=16,
            explicit=True,
        )


# ----------------------------------------------------------------------
# Timing and spoiling
# ----------------------------------------------------------------------


def test_a_longer_te_or_tr_is_padded_with_a_wait(system, excitation):
    readout = _readout(
        design.SpiralReadout2D,
        system,
        excitation,
        design_interleaves=16,
        te=3e-3,
        tr=15e-3,
    )
    assert readout.echo_time == pytest.approx(3e-3)
    assert readout.duration == pytest.approx(15e-3)


@pytest.mark.parametrize(
    ("kwargs", "name"), [({"te": 1e-5}, "TE"), ({"tr": 1e-4}, "TR")]
)
def test_an_impossible_time_is_refused_by_name(system, excitation, kwargs, name):
    with pytest.raises(ValueError, match=f"requested {name}"):
        _readout(
            design.SpiralReadout2D, system, excitation, design_interleaves=16, **kwargs
        )


def test_the_spoiler_winds_the_cycles_it_was_asked_for(system, excitation):
    cycles, voxel = 4.0, 1e-3
    readout = _readout(
        design.SpiralReadout2D,
        system,
        excitation,
        design_interleaves=16,
        spoiling_cycles=cycles,
        voxel_size_m=voxel,
    )
    moment = float(np.trapezoid(readout.gz_spoil.waveform, readout.gz_spoil.tt))
    assert moment == pytest.approx(cycles / voxel, rel=1e-3)
    assert readout.gz_spoil.channel == "z"


def test_a_rewound_shot_ends_where_it_started(system, excitation):
    readout = _readout(
        design.SpiralReadout2D, system, excitation, design_interleaves=16
    )
    seq = pp.Sequence(system)
    for block in readout.blocks:
        seq.add_block(*block)
    assert np.allclose(seq.calculate_kspace()[1][:2, -1], 0.0, atol=0.02 * KMAX)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_echoes": 0}, "n_echoes must be >= 1"),
        ({"spoiling_cycles": -1.0}, "spoiling_cycles must be >= 0"),
        ({"spoiling_axis": "w"}, "spoiling_axis must be one of"),
    ],
)
def test_an_impossible_readout_is_refused(system, excitation, kwargs, message):
    with pytest.raises(ValueError, match=message):
        _readout(
            design.SpiralReadout2D, system, excitation, design_interleaves=16, **kwargs
        )


# ----------------------------------------------------------------------
# End to end
# ----------------------------------------------------------------------


def test_a_stack_of_spirals_is_a_plain_pypulseq_loop(system, slab, tmp_path):
    readout = _readout(
        design.SpiralStackReadout,
        system,
        slab,
        design_interleaves=16,
        fov_z=0.12,
        matrix_z=8,
        tr=12e-3,
        labels=("LIN", "PAR"),
    )
    arms, partitions = 6, 8
    angles = pp.calc_golden_angles(arms)
    phases = pp.make_rf_spoiling_schedule(arms * partitions)
    lin_label, par_label = readout.adc_labels

    seq = pp.Sequence(system)
    for shot, (arm, partition) in enumerate(
        (arm, partition) for arm in range(arms) for partition in range(partitions)
    ):
        readout.rf.phase_offset = readout.adc.phase_offset = phases[shot]
        lin_label.value, par_label.value = arm, partition
        kz = (partition - partitions / 2) / (partitions / 2)
        rotation = pp.make_rotation(Rotation.from_euler("z", float(angles[arm])))

        seq.add_block(readout.rf, readout.gz)
        seq.add_block(pp.scale_grad(readout.gz_pre, kz))
        seq.add_block(
            readout.gx, readout.gy, readout.adc, *readout.adc_labels, rotation
        )
        seq.add_block(
            readout.gx_rew,
            readout.gy_rew,
            pp.scale_grad(readout.gz_rew, kz),
            rotation,
        )
        seq.add_block(readout.wait_tr)

    assert seq.check_timing()[0]
    path = tmp_path / "spiral_stack.seq"
    seq.write(str(path))

    written = pp.Sequence()
    written.read(str(path))
    assert written.num_blocks == seq.num_blocks


# ----------------------------------------------------------------------
# Vector limits
# ----------------------------------------------------------------------


def _vector_slew(events):
    """Peak slew of a set of same-grid gradients taken as one vector."""
    times = np.asarray(events[0].tt, dtype=float)
    waveforms = np.column_stack([np.asarray(e.waveform, dtype=float) for e in events])
    return float(
        np.max(
            np.linalg.norm(np.diff(waveforms, axis=0) / np.diff(times)[:, None], axis=1)
        )
    )


@pytest.mark.parametrize("direction", ["outward", "inward", "in_out"])
def test_the_bridges_of_two_axes_obey_the_vector_slew_limit(
    system, excitation, direction
):
    """Two axes solved separately against the full limit combine past it.

    Per axis nothing is wrong, so no per-axis check reports it -- but the
    scanner limit is on the vector, and a rotation makes the point visible by
    mixing the combined slew onto a single axis.
    """
    readout = _readout(
        design.SpiralReadout2D,
        system,
        excitation,
        design_interleaves=16,
        direction=direction,
    )
    for bracket in (("gx_pre", "gy_pre"), ("gx_rew", "gy_rew")):
        events = [getattr(readout.events, name, None) for name in bracket]
        if any(event is None for event in events):
            continue
        assert _vector_slew(events) <= system.max_slew * 1.001, bracket


def test_a_rotated_interleave_is_still_playable(system, excitation):
    """The check the vector limit exists for, run the way a scan would."""
    readout = _readout(
        design.SpiralReadout2D, system, excitation, design_interleaves=16
    )
    seq = pp.Sequence(system)
    for angle in np.linspace(0.0, np.pi, 7):
        rotation = pp.make_rotation(Rotation.from_euler("z", float(angle)))
        seq.add_block(readout.rf, readout.gz)
        seq.add_block(readout.gx, readout.gy, readout.adc, rotation)
        seq.add_block(readout.gx_rew, readout.gy_rew, rotation)
    assert seq.check_timing()[0]


# ----------------------------------------------------------------------
# Sampling the caller asked for
# ----------------------------------------------------------------------


def test_a_radial_spoke_is_one_continuous_waveform(system, excitation):
    """Prephaser, readout and rewinder are the same lobe traversed once.

    Three separate blocks would need three consistent rotations to orient one
    spoke; one waveform needs one.
    """
    readout = _readout(design.RadialReadout2D, system, excitation)
    acquisition = [block for block in readout.blocks if readout.adc in block]
    assert len(acquisition) == 1
    assert set(acquisition[0]) == {readout.gx, readout.gy, readout.adc}
    assert not hasattr(readout.events, "gx_pre")
    assert not hasattr(readout.events, "gx_rew")


def test_the_radial_adc_sits_on_the_readout_plateau(system, excitation):
    """Shifted past the prephaser, so it acquires the flat part and nothing else."""
    readout = _readout(design.RadialReadout2D, system, excitation)
    kx = _first_line(readout)[0][0]
    half_width = MATRIX / (2 * FOV)
    assert kx.min() == pytest.approx(-half_width, rel=2e-2)
    assert kx.max() == pytest.approx(half_width, rel=2e-2)
    assert readout.adc.delay > 0


def _first_line(module):
    k_traj_adc, _, t_excitation, _, t_adc = module.calculate_kspace()
    count = int(module.adc.num_samples)
    return k_traj_adc[:, :count], np.asarray(t_adc)[:count], float(t_excitation[0])


@pytest.mark.parametrize(
    ("cls", "kwargs"),
    [
        (design.RadialReadout2D, {}),
        (design.SpiralReadout2D, {"design_interleaves": 16}),
    ],
    ids=["radial", "spiral"],
)
def test_a_lower_bandwidth_really_lowers_the_bandwidth(system, excitation, cls, kwargs):
    """The knob has to move something, or it is not a knob.

    A spiral cannot simply sample more slowly along a time-optimal arm --
    adjacent samples would land further than 1 / fov apart -- so the arm is
    stretched until the requested dwell is legal. That is the cost, and it is
    the caller's to accept.
    """
    fast = _readout(cls, system, excitation, readout_bandwidth_hz=250e3, **kwargs)
    slow = _readout(cls, system, excitation, readout_bandwidth_hz=62.5e3, **kwargs)

    assert slow.bandwidth_hz < fast.bandwidth_hz
    # At most the bandwidth asked for. It may be less: the dwell has to land on
    # the ADC raster while the readout duration lands on the gradient raster,
    # and the nearest pair that does both is sometimes slower than the request.
    assert slow.bandwidth_hz <= 62.5e3 * 1.05
    assert slow.bandwidth_hz > 0.5 * 62.5e3
    assert slow.adc.dwell > fast.adc.dwell
    assert slow.check_timing()[0]


def test_the_dwell_always_lands_on_the_adc_raster(system, excitation):
    for bandwidth in (250e3, 125e3, 90e3, 62.5e3):
        readout = _readout(
            design.SpiralReadout2D,
            system,
            excitation,
            design_interleaves=16,
            readout_bandwidth_hz=bandwidth,
        )
        steps = readout.adc.dwell / system.adc_raster_time
        assert steps == pytest.approx(round(steps)), bandwidth


def test_oversampling_a_spiral_samples_the_same_arm_more_densely(system, excitation):
    plain = _readout(design.SpiralReadout2D, system, excitation, design_interleaves=16)
    dense = _readout(
        design.SpiralReadout2D,
        system,
        excitation,
        design_interleaves=16,
        oversampling=2.0,
    )
    assert dense.n_samples > plain.n_samples
