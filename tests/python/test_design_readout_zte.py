"""The zero-echo-time readout.

Two things make ZTE what it is, and both are checked against the trajectory the
sequence reports: the gradient is already up when the pulse fires, so a spoke
starts at the centre of k-space, and it never returns to zero between views, so
a shell costs one ramp. The third is arithmetic on the directions -- every view
is the designed one turned -- and that is checked against the rotations.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import pulserver.design as design
import pulserver.pypulseq as pp

FOV = 0.24
MATRIX = 96


@pytest.fixture
def system():
    return pp.Opts(
        max_grad=40,
        grad_unit="mT/m",
        max_slew=150,
        slew_unit="T/m/s",
        rf_dead_time=100e-6,
        rf_ringdown_time=20e-6,
        adc_dead_time=10e-6,
    )


@pytest.fixture
def hard(system):
    return design.NonSelectiveExcitation(system, 4.0, duration_s=10e-6)


def readout(system, hard, **kwargs):
    kwargs.setdefault("fov", FOV)
    kwargs.setdefault("matrix", MATRIX)
    kwargs.setdefault("n_views", 32)
    return design.ZteReadout(system, hard.rf, **kwargs)


def spokes(module, seq=None):
    """Every acquisition's k-space, shaped ``(views, samples, 3)``."""
    k_traj_adc = (seq or module).calculate_kspace(dense=False)[0]
    count = int(module.adc.num_samples)
    return np.asarray(k_traj_adc).T.reshape(-1, count, 3)


def build_shell(system, zte, shot=None):
    """One shell, written the way the docstring says to."""
    seq = pp.Sequence(system)
    turns = [
        pp.make_rotation(
            Rotation.from_matrix(turn if shot is None else shot @ turn)
        )
        for turn in zte.view_rotations
    ]
    seq.add_block(*zte.g_ramp, turns[0])
    for view, turn in enumerate(turns):
        last = view == len(turns) - 1
        seq.add_block(zte.rf, *zte.g_hold, turn)
        seq.add_block(zte.adc, *(zte.g_end if last else zte.g_read), turn)
    return seq


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------


def test_the_module_lays_out_a_ramp_and_two_views(system, hard):
    zte = readout(system, hard)
    assert len(zte.blocks) == 5
    assert zte.blocks[0] == tuple(zte.g_ramp)
    assert zte.blocks[1] == (zte.rf, *zte.g_hold)
    assert zte.blocks[2] == (zte.adc, *zte.g_read)


def test_the_readout_is_valid_pulseq(system, hard):
    assert readout(system, hard).check_timing()[0]


def test_a_whole_shell_is_valid_pulseq(system, hard):
    zte = readout(system, hard)
    assert build_shell(system, zte).check_timing()[0]


def test_every_view_replays_the_same_events(system, hard):
    zte = readout(system, hard)
    assert zte.blocks[3][:-1] == zte.blocks[1]
    assert zte.blocks[4][0] == zte.blocks[2][0]


def test_labels_ride_the_acquisition_block(system, hard):
    zte = readout(system, hard, labels=("LIN",))
    assert zte.adc_labels in zte.blocks[2]
    assert zte.adc_labels.label == "LIN"


# ----------------------------------------------------------------------
# The gradient never comes down
# ----------------------------------------------------------------------


def test_the_pulse_plays_on_a_gradient_already_at_full_amplitude(system, hard):
    zte = readout(system, hard)
    for ramp, hold in zip(zte.g_ramp, zte.g_hold, strict=True):
        assert float(np.asarray(ramp.waveform)[0]) == pytest.approx(0.0)
        assert float(np.asarray(ramp.waveform)[-1]) == pytest.approx(
            float(np.asarray(hold.waveform)[0])
        )
        assert np.ptp(np.asarray(hold.waveform)) == pytest.approx(0.0)


def test_a_view_hands_the_gradient_to_the_next_without_passing_through_zero(system, hard):
    zte = readout(system, hard)
    plateau = np.array([float(np.asarray(g.waveform)[0]) for g in zte.g_read])
    handover = np.array([float(np.asarray(g.waveform)[-1]) for g in zte.g_read])
    assert np.linalg.norm(plateau) == pytest.approx(zte.gradient_amplitude)
    assert np.linalg.norm(handover) == pytest.approx(zte.gradient_amplitude)
    # It turned by the step rather than dropping and climbing again.
    cosine = np.dot(plateau, handover) / zte.gradient_amplitude**2
    assert np.arccos(np.clip(cosine, -1, 1)) == pytest.approx(zte.step_rad, rel=1e-6)


def test_only_the_last_view_ramps_down(system, hard):
    zte = readout(system, hard)
    assert np.linalg.norm(
        [float(np.asarray(g.waveform)[-1]) for g in zte.g_end]
    ) == pytest.approx(0.0, abs=1e-9)


def test_the_gradient_amplitude_is_one_step_of_k_per_dwell(system, hard):
    zte = readout(system, hard)
    assert zte.gradient_amplitude == pytest.approx(zte.delta_k * zte.bandwidth_hz)


# ----------------------------------------------------------------------
# The spoke
# ----------------------------------------------------------------------


def test_a_spoke_starts_near_the_centre_and_reaches_the_edge(system, hard):
    """It has to arrive at the edge, and it must not run past it."""
    zte = readout(system, hard)
    radius = np.linalg.norm(spokes(zte), axis=-1)
    assert np.all(radius[:, 0] < 5.0 * zte.delta_k)
    assert np.all(radius[:, -1] > zte.kmax - zte.delta_k)
    assert np.all(radius[:, -1] <= zte.kmax)


def test_the_reported_gap_is_the_time_to_the_first_sample(system, hard):
    """It counts the ADC's own delay and the half dwell to the sample centre."""
    for dead_time_s in (None, 40e-6):
        zte = readout(system, hard, dead_time_s=dead_time_s)
        first = np.linalg.norm(spokes(zte)[0, 0])
        assert first == pytest.approx(zte.gap * zte.gradient_amplitude, rel=1e-9)
        assert zte.gap >= float(zte.rf.ringdown_time) + (dead_time_s or system.adc_dead_time)


def test_a_spoke_is_a_straight_line_out_of_the_centre(system, hard):
    zte = readout(system, hard)
    first = spokes(zte)[0]
    direction = first[-1] / np.linalg.norm(first[-1])
    along = first @ direction
    assert np.linalg.norm(first - np.outer(along, direction)) == pytest.approx(0.0, abs=1e-6)
    assert np.all(np.diff(along) > 0.0)


def test_the_dropped_samples_are_counted_not_compressed(system, hard):
    zte = readout(system, hard)
    assert zte.n_samples + zte.n_missing == zte.n_nominal
    assert zte.n_missing >= 1
    # What survives is spaced by delta_k, not squeezed into the full extent.
    radius = np.linalg.norm(spokes(zte)[0], axis=-1)
    assert np.diff(radius) == pytest.approx(zte.delta_k, rel=1e-6)


def test_a_faster_readout_loses_fewer_samples(system, hard):
    slow = readout(system, hard, readout_bandwidth_hz=31.25e3)
    fast = readout(system, hard, readout_bandwidth_hz=125e3)
    assert fast.n_missing > slow.n_missing


def test_every_view_traces_the_same_spoke_from_the_centre(system, hard):
    """The trajectory has to restart at each pulse, not accumulate."""
    zte = readout(system, hard)
    radius = np.linalg.norm(spokes(zte), axis=-1)
    assert radius[1] == pytest.approx(radius[0], rel=1e-9)


# ----------------------------------------------------------------------
# Rotation encoding
# ----------------------------------------------------------------------


def test_the_view_rotations_carry_the_designed_view_onto_every_view(system, hard):
    """The design is a spoke along x, so that is what the rotations turn."""
    zte = readout(system, hard)
    placed = np.einsum("vij,j->vi", zte.view_rotations, np.array([1.0, 0.0, 0.0]))
    assert placed == pytest.approx(zte.directions, abs=1e-9)


def test_the_rotations_also_carry_the_transition(system, hard):
    """A view is played as the designed transition turned, so the rotation has
    to place where the gradient is *going* too, not only where it is."""
    zte = readout(system, hard)
    designed_step = np.array([np.cos(zte.step_rad), np.sin(zte.step_rad), 0.0])
    following = np.einsum("vij,j->vi", zte.view_rotations[:-1], designed_step)
    assert following == pytest.approx(zte.directions[1:], abs=1e-9)


def test_a_shell_plays_the_directions_it_was_given(system, hard):
    zte = readout(system, hard, n_views=12)
    seq = build_shell(system, zte)
    ends = spokes(zte, seq)[:, -1, :]
    played = ends / np.linalg.norm(ends, axis=1, keepdims=True)
    assert played == pytest.approx(zte.directions, abs=1e-4)


def test_a_rotated_shell_lands_where_the_shot_rotation_says(system, hard):
    zte = readout(system, hard, n_views=12, n_shots=5)
    shot = zte.shot_rotations[3]
    seq = build_shell(system, zte, shot=shot)
    ends = spokes(zte, seq)[:, -1, :]
    played = ends / np.linalg.norm(ends, axis=1, keepdims=True)
    assert played == pytest.approx(zte.directions @ shot.T, abs=1e-4)


def test_an_ordering_whose_step_wanders_is_refused(system, hard):
    golden = np.pi * (3.0 - np.sqrt(5.0))
    index = np.arange(24, dtype=float)
    height = 1.0 - 2.0 * (index + 0.5) / 24
    radius = np.sqrt(np.maximum(0.0, 1.0 - height**2))
    azimuth = index * golden
    phyllotaxis = np.column_stack(
        (radius * np.cos(azimuth), radius * np.sin(azimuth), height)
    )
    with pytest.raises(ValueError, match=r"constant angle|step by"):
        readout(system, hard, n_views=None, directions=phyllotaxis)


def test_a_shorter_shell_steps_further_between_views(system, hard):
    """The step is the polar gap at the pole, which ``n_views`` alone fixes."""
    short = readout(system, hard, n_views=32)
    long = readout(system, hard, n_views=128)
    assert short.step_rad == pytest.approx(np.arccos(1.0 - 2.0 / 31.0))
    assert short.step_rad > long.step_rad


# ----------------------------------------------------------------------
# The ordering: generated or supplied
# ----------------------------------------------------------------------


def test_it_generates_a_nyquist_matched_shell_when_given_none(system, hard):
    zte = design.ZteReadout(system, hard.rf, fov=FOV, matrix=MATRIX, n_shots=89)
    assert len(zte.directions) == -(-int(np.ceil(np.pi * MATRIX**2)) // 89)
    assert zte.shot_rotations.shape == (89, 3, 3)
    assert len(zte.directions) * 89 >= np.pi * MATRIX**2


def test_splitting_into_more_shots_shortens_the_shell(system, hard):
    """``n_shots`` divides a fixed sphere, so it buys segment length."""
    few = design.ZteReadout(system, hard.rf, fov=FOV, matrix=MATRIX, n_shots=32)
    many = design.ZteReadout(system, hard.rf, fov=FOV, matrix=MATRIX, n_shots=128)
    assert len(many.directions) * 4 == pytest.approx(len(few.directions), abs=4)


def test_the_default_split_matches_the_two_spacings(system, hard):
    """Ring to ring and within a ring agree at the equator: n_shots = pi n_views."""
    zte = design.ZteReadout(system, hard.rf, fov=FOV, matrix=MATRIX)
    n_views, n_shots = len(zte.directions), len(zte.shot_rotations)
    assert n_views * n_shots >= np.pi * MATRIX**2
    assert n_shots == pytest.approx(np.pi * n_views, rel=0.05)
    # Which is the same as saying the shell is one spoke per matrix step.
    assert n_views == pytest.approx(MATRIX, rel=0.05)


def test_the_default_sampling_is_isotropic_at_the_equator(system, hard):
    """Polar and azimuthal neighbours land the same distance apart."""
    zte = design.ZteReadout(system, hard.rf, fov=FOV, matrix=MATRIX)
    equator = len(zte.directions) // 2
    heights = np.arccos(np.clip(zte.directions[equator : equator + 2, 2], -1, 1))
    ring_gap = float(np.diff(heights)[0])
    azimuth_gap = 2.0 * np.pi / len(zte.shot_rotations)
    assert ring_gap == pytest.approx(azimuth_gap, rel=0.1)
    assert azimuth_gap == pytest.approx(2.0 / MATRIX, rel=0.1)
    # The shell's own step strides across many rings' worth of azimuth; it is
    # the shots, not the shell, that sample a ring.
    assert zte.step_rad > 5.0 * azimuth_gap


def test_a_generated_shell_runs_pole_to_pole(system, hard):
    """The ends sit on the shot rotation axis, so every shot shares them."""
    zte = readout(system, hard, n_shots=8)
    assert zte.directions[0] == pytest.approx([0.0, 0.0, 1.0], abs=1e-12)
    assert zte.directions[-1] == pytest.approx([0.0, 0.0, -1.0], abs=1e-12)
    ends = np.einsum("sij,vj->svi", zte.shot_rotations, zte.directions[[0, -1]])
    assert ends == pytest.approx(np.broadcast_to(zte.directions[[0, -1]], ends.shape))


def test_the_shots_put_even_azimuth_on_every_polar_ring(system, hard):
    """Full coverage: one shell per ring, turned by 2 pi / n_shots."""
    zte = readout(system, hard, n_views=9, n_shots=6)
    sphere = np.einsum("sij,vj->svi", zte.shot_rotations, zte.directions)
    for ring in sphere.transpose(1, 0, 2)[1:-1]:
        azimuth = np.sort(np.arctan2(ring[:, 1], ring[:, 0]))
        assert np.diff(azimuth) == pytest.approx(2.0 * np.pi / 6.0)


def test_a_supplied_ordering_is_used_as_it_stands(system, hard):
    directions, _ = pp.calc_projection_shell(20, scheme="meridian")
    zte = readout(system, hard, n_views=None, directions=directions)
    assert zte.directions == pytest.approx(directions)
    assert not hasattr(zte.events, "shot_rotations")


def test_a_supplied_ordering_silences_the_generator_arguments(system, hard):
    directions, _ = pp.calc_projection_shell(20, scheme="meridian")
    zte = readout(system, hard, n_views=7, n_shots=5, directions=directions)
    assert zte.directions == pytest.approx(directions)
    assert not hasattr(zte.events, "shot_rotations")


def test_the_directions_are_published_read_only(system, hard):
    zte = readout(system, hard)
    with pytest.raises(ValueError):
        zte.directions[0, 0] = 5.0


# ----------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------


def test_the_repetition_holds_the_gradient_until_the_next_view(system, hard):
    tight = readout(system, hard)
    padded = readout(system, hard, tr=tight.tr + 1e-3)
    assert padded.tr == pytest.approx(tight.tr + 1e-3)
    assert pp.calc_duration(padded.g_read[0]) > pp.calc_duration(tight.g_read[0])
    assert float(np.asarray(padded.g_read[0].waveform)[-1]) == pytest.approx(
        float(np.asarray(tight.g_read[0].waveform)[-1])
    )


def test_a_repetition_shorter_than_the_slew_is_refused(system, hard):
    tight = readout(system, hard)
    with pytest.raises(ValueError, match="TR"):
        readout(system, hard, tr=0.5 * tight.tr)


def test_a_shell_takes_one_ramp_and_a_repetition_per_view(system, hard):
    zte = readout(system, hard, n_views=12)
    seq = build_shell(system, zte)
    ramp = pp.calc_duration(zte.g_ramp[0])
    closing = zte.tr - pp.calc_duration(zte.g_read[0]) + pp.calc_duration(zte.g_end[0])
    assert seq.duration()[0] == pytest.approx(ramp + 11 * zte.tr + closing, abs=1e-9)


# ----------------------------------------------------------------------
# Arguments
# ----------------------------------------------------------------------


def test_a_pulse_too_long_for_the_dwell_is_refused(system):
    long_pulse = design.NonSelectiveExcitation(system, 4.0, duration_s=200e-6)
    with pytest.raises(ValueError, match="brief enough"):
        design.ZteReadout(system, long_pulse.rf, fov=FOV, matrix=MATRIX, n_views=8)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"fov": 0.0}, "fov"),
        ({"matrix": 1}, "matrix"),
        ({"oversampling": 0.5}, "oversampling"),
        ({"readout_bandwidth_hz": 0.0}, "readout_bandwidth_hz"),
        ({"dead_time_s": 0.0}, "dead_time_s"),
    ],
)
def test_an_impossible_request_is_refused(system, hard, kwargs, message):
    with pytest.raises(ValueError, match=message):
        readout(system, hard, **kwargs)


def test_a_single_direction_is_not_a_shell(system, hard):
    with pytest.raises(ValueError, match="at least two views"):
        readout(system, hard, n_views=None, directions=[[1.0, 0.0, 0.0]])
