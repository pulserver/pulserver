"""The balanced SSFP readouts.

What a bSSFP repetition has to get right is that every axis comes back to
k = 0, that the echo sits exactly halfway between two pulses, and that the read
axis never leaves the plateau between one acquisition and the next. All three
are checked against the trajectory the sequence reports rather than against
event fields.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import pulserver.design as design
import pulserver.pypulseq as pp

FOV = (0.28, 0.28, 0.12)
MATRIX = (128, 128, 32)


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
def slice_(system):
    return design.SpatialSelectiveExcitation(system, 40.0, 5e-3, 1e-3, rephase=False)


@pytest.fixture
def slab(system):
    return design.SpatialSelectiveExcitation(system, 40.0, FOV[2], 1e-3, rephase=False)


def readout2d(system, slice_, **kwargs):
    kwargs.setdefault("fov", FOV[:2])
    kwargs.setdefault("matrix", MATRIX[:2])
    return design.BssfpReadout2D(system, slice_.rf, slice_.gz, **kwargs)


def readout3d(system, slab, **kwargs):
    kwargs.setdefault("fov", FOV)
    kwargs.setdefault("matrix", MATRIX)
    return design.BssfpReadout3D(system, slab.rf, slab.gz, **kwargs)


def build_loop(system, bssfp, plan, partitions=None):
    """A scan loop over ``plan``, written the way the docstring says to."""
    seq = pp.Sequence(system)
    nominal = bssfp.rf.amplitude
    bssfp.rf.amplitude = 0.5 * nominal
    seq.add_block(bssfp.rf, bssfp.gz, *bssfp.prep_labels)
    seq.add_block(bssfp.wait_prep, *bssfp.train_labels)
    bssfp.rf.amplitude = nominal

    def z_lobes(index):
        if partitions is None:
            return bssfp.gz_pre, bssfp.gz_rew
        return (
            pp.add_gradients(
                [bssfp.gz_pre, pp.scale_grad(bssfp.gz_partition, partitions[index])],
                system=system,
            ),
            pp.add_gradients(
                [bssfp.gz_rew, pp.scale_grad(bssfp.gz_partition_rew, partitions[index])],
                system=system,
            ),
        )

    for shot, ky in enumerate(plan):
        bssfp.rf.phase_offset = bssfp.adc.phase_offset = np.pi * ((shot + 1) % 2)
        if shot:
            seq.add_block(
                bssfp.gx_rew, pp.scale_grad(bssfp.gy_rew, plan[shot - 1]), z_lobes(shot - 1)[1]
            )
        else:
            seq.add_block(bssfp.wait_rewind, bssfp.gz_rew)
        seq.add_block(bssfp.rf, bssfp.gz)
        seq.add_block(
            bssfp.gx,
            bssfp.adc,
            pp.scale_grad(bssfp.gy_pre, ky),
            z_lobes(shot)[0],
            *getattr(bssfp.events, "adc_labels", ()),
        )
    seq.add_block(
        bssfp.gx_rew, pp.scale_grad(bssfp.gy_rew, plan[-1]), z_lobes(len(plan) - 1)[1],
        *bssfp.end_labels,
    )
    return seq


def echo_times(module, seq):
    """When each echo crossed k = 0, and when each pulse was played."""
    _, _, t_excitation, _, t_adc = seq.calculate_kspace()
    count = int(module.adc.num_samples)
    samples = np.asarray(t_adc).reshape(-1, count)
    # `t_adc` reports sample centres; the crossing is half a dwell earlier.
    return samples[:, count // 2] - 0.5 / module.bandwidth_hz, np.asarray(t_excitation)


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------


def test_a_repetition_is_excite_read_rewind(system, slice_):
    bssfp = readout2d(system, slice_)
    assert len(bssfp.blocks) == 3
    assert bssfp.blocks[0] == (bssfp.rf, bssfp.gz)
    assert bssfp.blocks[1] == (bssfp.gx, bssfp.adc, bssfp.gy_pre, bssfp.gz_pre)
    assert bssfp.blocks[2] == (bssfp.gx_rew, bssfp.gy_rew, bssfp.gz_rew)


def test_the_repetition_is_valid_pulseq(system, slice_):
    assert readout2d(system, slice_).check_timing()[0]


def test_the_3d_repetition_is_valid_pulseq(system, slab):
    assert readout3d(system, slab).check_timing()[0]


def test_the_module_spans_exactly_one_repetition(system, slice_):
    bssfp = readout2d(system, slice_)
    assert bssfp.duration == pytest.approx(bssfp.tr)


def test_labels_ride_the_acquisition_block(system, slice_):
    bssfp = readout2d(system, slice_, labels=("LIN", "SLC"))
    assert all(label in bssfp.blocks[1] for label in bssfp.adc_labels)
    assert [label.label for label in bssfp.adc_labels] == ["LIN", "SLC"]


def test_a_trigger_rides_the_rewind_block(system, slice_):
    trigger = pp.make_trigger("physio1", duration=100e-6)
    assert trigger in readout2d(system, slice_, trigger=trigger).blocks[2]


def test_the_transients_are_marked_for_the_interpreter(system, slice_):
    bssfp = readout2d(system, slice_)
    values = [
        (labels[0].label, labels[0].value)
        for labels in (bssfp.prep_labels, bssfp.train_labels, bssfp.end_labels)
    ]
    assert values == [("ONCE", 1), ("ONCE", 0), ("ONCE", 2)]


def test_a_2d_readout_has_no_partition_encode(system, slice_):
    bssfp = readout2d(system, slice_)
    assert not hasattr(bssfp.events, "gz_partition")


# ----------------------------------------------------------------------
# Balance
# ----------------------------------------------------------------------


def test_the_read_axis_never_leaves_the_plateau(system, slice_):
    """The lobe that closes a repetition starts where the acquisition ended."""
    bssfp = readout2d(system, slice_)
    assert float(bssfp.gx.waveform[-1]) == pytest.approx(float(bssfp.gx_rew.waveform[0]))
    assert float(bssfp.gx.waveform[0]) == pytest.approx(0.0)
    assert float(bssfp.gx_rew.waveform[-1]) == pytest.approx(0.0)


def test_every_axis_is_rewound_across_a_repetition(system, slab):
    """Integrated pulse centre to pulse centre, which is the interval that has
    to net to zero for the steady state to be a steady state."""
    bssfp = readout3d(system, slab)
    seq = build_loop(system, bssfp, [0.4, -0.7, 1.0], partitions=[0.3, -1.0, 0.6])
    waveforms = seq.waveforms_and_times()[0]
    _, pulses = echo_times(bssfp, seq)
    for axis in range(3):
        times, amplitudes = waveforms[axis]
        for start, stop in itertools.pairwise(pulses[1:]):
            grid = np.linspace(start, stop, 20001)
            moment = np.trapezoid(np.interp(grid, times, amplitudes), grid)
            assert abs(moment) < 1e-3 * bssfp.delta_kx


def test_the_read_axis_crosses_zero_in_the_middle_of_every_echo(system, slice_):
    bssfp = readout2d(system, slice_)
    seq = build_loop(system, bssfp, [0.4, -0.7, 1.0])
    k_adc, _, _, _, _ = seq.calculate_kspace()
    k = np.asarray(k_adc).reshape(3, -1, bssfp.n_samples)
    middle = bssfp.n_samples // 2
    assert k[0, :, middle] == pytest.approx(0.5 * bssfp.delta_kx, rel=1e-6)
    assert k[0, :, middle - 1] == pytest.approx(-0.5 * bssfp.delta_kx, rel=1e-6)


def test_every_echo_reads_the_same_read_axis_line(system, slice_):
    bssfp = readout2d(system, slice_)
    seq = build_loop(system, bssfp, [0.4, -0.7, 1.0])
    k_adc, _, _, _, _ = seq.calculate_kspace()
    k = np.asarray(k_adc).reshape(3, -1, bssfp.n_samples)
    assert k[0] == pytest.approx(np.broadcast_to(k[0, 0], k[0].shape))


def test_the_loop_encodes_the_lines_it_asked_for(system, slab):
    bssfp = readout3d(system, slab)
    plan, partitions = [0.4, -0.7, 1.0], [0.3, -1.0, 0.6]
    seq = build_loop(system, bssfp, plan, partitions=partitions)
    k_adc, _, _, _, _ = seq.calculate_kspace()
    k = np.asarray(k_adc).reshape(3, -1, bssfp.n_samples)
    middle = bssfp.n_samples // 2
    assert k[1, :, middle] == pytest.approx(
        np.array(plan) * 0.5 * MATRIX[1] / FOV[1], rel=1e-6
    )
    assert k[2, :, middle] == pytest.approx(
        np.array(partitions) * 0.5 * MATRIX[2] / FOV[2], rel=1e-6
    )


# ----------------------------------------------------------------------
# The partition encode rides the rephaser and stays one trapezoid
# ----------------------------------------------------------------------


def test_the_partition_encode_shares_the_rephaser_window(system, slab):
    bssfp = readout3d(system, slab)
    for encode, rephaser in (
        (bssfp.gz_partition, bssfp.gz_pre),
        (bssfp.gz_partition_rew, bssfp.gz_rew),
    ):
        assert encode.type == rephaser.type == "trap"
        assert (encode.rise_time, encode.flat_time, encode.fall_time) == (
            rephaser.rise_time,
            rephaser.flat_time,
            rephaser.fall_time,
        )


def test_summing_them_stays_a_trapezoid_at_every_partition(system, slab):
    """Same timing, so the sum is one shape at an amplitude that varies."""
    bssfp = readout3d(system, slab)
    lobes = [
        pp.add_gradients(
            [bssfp.gz_pre, pp.scale_grad(bssfp.gz_partition, kz)], system=system
        )
        for kz in np.linspace(-1.0, 1.0, 17)
    ]
    assert {lobe.type for lobe in lobes} == {"trap"}
    assert len({(lobe.rise_time, lobe.flat_time, lobe.fall_time) for lobe in lobes}) == 1


def test_the_summed_lobe_stays_inside_the_gradient_limit(system, slab):
    bssfp = readout3d(system, slab)
    for sign in (1.0, -1.0):
        lobe = pp.add_gradients(
            [bssfp.gz_pre, pp.scale_grad(bssfp.gz_partition, sign)], system=system
        )
        assert abs(lobe.amplitude) <= system.max_grad * (1 + 1e-9)


def test_the_partition_encode_reaches_the_edge_of_k_space(system, slab):
    bssfp = readout3d(system, slab)
    assert bssfp.gz_partition.area == pytest.approx(0.5 * MATRIX[2] / FOV[2])
    assert bssfp.gz_partition_rew.area == pytest.approx(-0.5 * MATRIX[2] / FOV[2])


# ----------------------------------------------------------------------
# TE = TR/2
# ----------------------------------------------------------------------


def test_the_echo_time_is_half_the_repetition_time(system, slice_):
    bssfp = readout2d(system, slice_)
    assert bssfp.te == pytest.approx(0.5 * bssfp.tr)


def test_the_echo_really_lands_at_half_a_repetition(system, slice_):
    bssfp = readout2d(system, slice_)
    seq = build_loop(system, bssfp, [0.4, -0.7, 1.0, 0.0])
    echoes, pulses = echo_times(bssfp, seq)
    assert echoes - pulses[1:] == pytest.approx(bssfp.te, abs=1e-9)
    assert np.diff(pulses[1:]) == pytest.approx(bssfp.tr, abs=1e-9)


def test_the_half_flip_pulse_sits_half_a_repetition_before_the_train(system, slice_):
    bssfp = readout2d(system, slice_)
    seq = build_loop(system, bssfp, [0.4, -0.7])
    _, pulses = echo_times(bssfp, seq)
    assert pulses[1] - pulses[0] == pytest.approx(bssfp.te, abs=1e-9)


def test_a_longer_repetition_time_keeps_the_echo_at_the_middle(system, slice_):
    tight = readout2d(system, slice_)
    padded = readout2d(system, slice_, tr=tight.tr + 2e-3)
    assert padded.tr > tight.tr
    assert padded.te == pytest.approx(0.5 * padded.tr)
    seq = build_loop(system, padded, [0.4, -0.7, 1.0])
    echoes, pulses = echo_times(padded, seq)
    assert echoes - pulses[1:] == pytest.approx(padded.te, abs=1e-9)


def test_slack_goes_into_the_encodes_rather_than_into_dead_time(system, slice_):
    tight = readout2d(system, slice_)
    loose = readout2d(system, slice_, tr=tight.tr + 4e-3)
    assert pp.calc_duration(loose.gy_pre) > pp.calc_duration(tight.gy_pre)
    assert abs(loose.gy_pre.amplitude) < abs(tight.gy_pre.amplitude)
    assert loose.gy_pre.area == pytest.approx(tight.gy_pre.area)


def test_a_repetition_time_shorter_than_the_layout_is_refused(system, slice_):
    tight = readout2d(system, slice_)
    with pytest.raises(ValueError, match="TR"):
        readout2d(system, slice_, tr=0.5 * tight.tr)


# ----------------------------------------------------------------------
# The half-flip preparation
# ----------------------------------------------------------------------


def test_a_pulse_too_long_for_the_acquisition_is_refused(system):
    """Half a repetition has to hold the pulse tail, the rewind and the lead."""
    long_pulse = design.SpatialSelectiveExcitation(system, 40.0, 5e-3, 5e-3, rephase=False)
    with pytest.raises(ValueError, match="half-flip"):
        design.BssfpReadout2D(
            system, long_pulse.rf, long_pulse.gz, fov=FOV[:2], matrix=MATRIX[:2]
        )


def test_the_same_pulse_builds_without_the_half_flip_preparation(system):
    long_pulse = design.SpatialSelectiveExcitation(system, 40.0, 5e-3, 5e-3, rephase=False)
    bssfp = design.BssfpReadout2D(
        system,
        long_pulse.rf,
        long_pulse.gz,
        fov=FOV[:2],
        matrix=MATRIX[:2],
        half_flip_prep=False,
    )
    assert not hasattr(bssfp.events, "wait_prep")
    assert bssfp.te == pytest.approx(0.5 * bssfp.tr)
    assert bssfp.check_timing()[0]


def test_the_half_flip_pulse_must_oppose_the_first_excitation():
    """Simulated, because the phase convention is the whole point of the prep.

    A half-flip pulse leaves the magnetisation on one side of the steady-state
    cone; the first full pulse has to carry it to the other side, which is a
    rotation the other way. Getting it wrong is worse than no preparation.
    """
    alpha, tr = np.deg2rad(40.0), 3.44e-3

    def alternation(prep_phase, first_phase, t1=1.0, t2=0.08, echoes=40):
        magnetisation = np.array([0.0, 0.0, 1.0])
        if prep_phase is not None:
            magnetisation = _rotate(alpha / 2, prep_phase) @ magnetisation
            magnetisation = _relax(magnetisation, 0.5 * tr, t1, t2)
        signal = []
        for echo in range(echoes):
            magnetisation = _rotate(alpha, first_phase + np.pi * echo) @ magnetisation
            magnetisation = _relax(magnetisation, 0.5 * tr, t1, t2)
            signal.append(np.hypot(magnetisation[0], magnetisation[1]))
            magnetisation = _relax(magnetisation, 0.5 * tr, t1, t2)
        signal = np.asarray(signal)
        # How far each echo departs from the mean of its neighbours: the
        # echo-to-echo oscillation, with the smooth approach divided out.
        return np.max(np.abs(signal[1:-1] - 0.5 * (signal[:-2] + signal[2:]))) / signal[-1]

    opposed = alternation(0.0, np.pi)
    aligned = alternation(0.0, 0.0)
    bare = alternation(None, 0.0)
    assert opposed < 0.01
    assert aligned > 1.0
    assert bare > 1.0


def _rotate(flip, phase):
    axis = np.array([np.cos(phase), np.sin(phase), 0.0])
    cross = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + np.sin(flip) * cross + (1.0 - np.cos(flip)) * (cross @ cross)


def _relax(magnetisation, interval, t1, t2):
    decay = np.exp(-interval / t2)
    return np.array(
        [
            magnetisation[0] * decay,
            magnetisation[1] * decay,
            1.0 + (magnetisation[2] - 1.0) * np.exp(-interval / t1),
        ]
    )


# ----------------------------------------------------------------------
# Arguments
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"oversampling": 0.5}, "oversampling"),
        ({"readout_bandwidth_hz": 0.0}, "readout_bandwidth_hz"),
        ({"matrix": (1, 128)}, "matrix"),
        ({"fov": (0.0, 0.28)}, "fov"),
    ],
)
def test_an_impossible_request_is_refused(system, slice_, kwargs, message):
    with pytest.raises(ValueError, match=message):
        readout2d(system, slice_, **kwargs)


def test_a_hard_pulse_needs_no_rephasers(system):
    excitation = design.NonSelectiveExcitation(system, 40.0, duration_s=0.3e-3)
    bssfp = design.BssfpReadout3D(
        system, excitation.rf, fov=FOV, matrix=MATRIX
    )
    assert not hasattr(bssfp.events, "gz_pre")
    assert bssfp.gz_partition.type == "trap"
    assert bssfp.check_timing()[0]
    assert bssfp.te == pytest.approx(0.5 * bssfp.tr)
