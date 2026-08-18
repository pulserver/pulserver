"""The echo-planar readout.

What makes an EPI train an EPI train is checked against the trajectory the
sequence reports: every line lands on the phase-encode step the ordering asked
for, no line drifts while it is being acquired, and moving the shot origin
moves the whole train and nothing else.
"""

from __future__ import annotations

import numpy as np
import pytest

import pulserver.design as design
import pulserver.pypulseq as pp

FOV = 0.22
MATRIX = 32
FOV_Z = 0.12
MATRIX_Z = 12


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
    return design.EpiReadout2D(
        system, excitation.rf, excitation.gz, excitation.gz_reph, **kwargs
    )


def readout3d(system, slab, **kwargs):
    kwargs.setdefault("fov", (FOV, FOV, FOV_Z))
    kwargs.setdefault("matrix", (MATRIX, MATRIX, MATRIX_Z))
    return design.EpiReadout3D(system, slab.rf, slab.gz, **kwargs)


def read_residual(system, epi):
    """Where one whole repetition leaves the read axis (1/m)."""
    seq = play(system, epi, [(0, 0)])
    return float(np.asarray(seq.calculate_kspace()[1])[0, -1])


def lines(module, seq=None):
    """Every line's k-space, shaped ``(shots, etl, samples, 3)``."""
    traj = np.asarray((seq or module).calculate_kspace(dense=False)[0]).T
    return traj.reshape(-1, module.etl, module.n_samples, 3)


def play(system, epi, origins, ndim=2):
    """A sequence of shots, written the way the docstring says to."""
    seq = pp.Sequence(system)
    half = MATRIX / 2
    for origin in origins:
        seq.add_block(epi.rf, epi.gz)
        encodes = [pp.scale_grad(epi.gy_pre, origin[0] / half)]
        if ndim == 3:
            encodes.append(pp.scale_grad(epi.gz_pre, origin[1] / (MATRIX_Z / 2)))
        seq.add_block(epi.gx_pre, *encodes)
        for line in range(epi.etl):
            blips = [epi.gy_blips[line]]
            if ndim == 3:
                blips.append(epi.gz_blips[line])
            seq.add_block(epi.gx[line], epi.adc, *[b for b in blips if b is not None])
        seq.add_block(epi.gx_spoil, epi.gy_rew)
    return seq


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------


def test_the_module_lays_out_a_pulse_a_prewinder_a_train_and_a_close(
    system, excitation
):
    epi = readout(system, excitation, etl=8, acceleration=4)
    assert len(epi.blocks) == 3 + epi.etl
    assert epi.blocks[0] == (epi.rf, epi.gz)
    assert epi.blocks[2][:2] == (epi.gx[0], epi.adc)
    assert epi.blocks[-1][0] is epi.gx_spoil


def test_every_line_is_published_and_the_lists_line_up(system, excitation):
    epi = readout(system, excitation, etl=8, acceleration=4)
    assert len(epi.gx) == epi.etl
    assert len(epi.gy_blips) == epi.etl
    assert len(epi.line_labels) == epi.etl
    assert len(epi.order) == epi.etl


def test_the_read_lobes_alternate_polarity(system, excitation):
    epi = readout(system, excitation, etl=6, acceleration=4)
    signs = [float(np.sign(lobe.area)) for lobe in epi.gx]
    assert signs == [1.0, -1.0] * 3


def test_the_last_line_carries_no_outgoing_step(system, excitation):
    """There is nowhere to go, so the blip list ends on a half."""
    epi = readout(system, excitation, etl=4, acceleration=4)
    assert epi.gy_blips[0] is not None
    assert pp.calc_duration(epi.gy_blips[-1]) < pp.calc_duration(epi.gy_blips[1])


def test_a_single_line_train_needs_no_blips_at_all(system, excitation):
    epi = readout(system, excitation, order=[0])
    assert epi.etl == 1
    assert epi.gy_blips == [None]
    assert epi.blip_span == 0.0


# ----------------------------------------------------------------------
# The trajectory
# ----------------------------------------------------------------------


def test_each_line_lands_on_the_step_the_ordering_asked_for(system, excitation):
    epi = readout(system, excitation, etl=8, acceleration=4)
    measured = lines(epi)[0, :, 0, 1]
    steps = np.round((measured - measured[0]) * FOV).astype(int)
    assert list(steps) == list(epi.order[:, 0])


def test_the_partition_follows_its_own_column(system, slab):
    epi = readout3d(
        system,
        slab,
        scheme="caipi",
        acceleration=2,
        segments=2,
        partition_acceleration=3,
    )
    measured = lines(epi)[0, :, 0, 2]
    steps = np.round((measured - measured[0]) * FOV_Z).astype(int)
    assert list(steps) == list(epi.order[:, 1])


def test_no_line_drifts_while_it_is_being_acquired(system, slab):
    """The blips ride the ramps, so they are wholly outside the sampling window."""
    epi = readout3d(
        system, slab, scheme="caipi", acceleration=2, partition_acceleration=3
    )
    traced = lines(epi)[0]
    assert traced[:, -1, 1] == pytest.approx(traced[:, 0, 1], abs=1e-9)
    assert traced[:, -1, 2] == pytest.approx(traced[:, 0, 2], abs=1e-9)


def test_the_read_sweep_is_centred_and_reverses_line_by_line(system, excitation):
    epi = readout(system, excitation, etl=4, acceleration=4)
    traced = lines(epi)[0]
    for line in range(epi.etl):
        assert traced[line, 0, 0] == pytest.approx(-traced[line, -1, 0], rel=1e-9)
    assert traced[0, -1, 0] > 0 > traced[1, -1, 0]


def test_the_read_sweep_reaches_the_edge_of_k_space(system, excitation):
    epi = readout(system, excitation, etl=4, acceleration=4)
    traced = lines(epi)[0]
    assert abs(traced[0, -1, 0]) == pytest.approx(0.5 * MATRIX / FOV, rel=0.05)


def test_moving_the_shot_origin_moves_the_whole_train(system, excitation):
    epi = readout(system, excitation, etl=8, acceleration=4)
    seq = play(system, epi, [(-16, 0), (-14, 0), (2, 0)])
    traced = lines(epi, seq)[:, :, 0, 1] * FOV
    for shot, origin in enumerate((-16, -14, 2)):
        assert traced[shot] == pytest.approx(origin + epi.order[:, 0], abs=1e-6)


def test_the_origin_moves_the_partition_too(system, slab):
    epi = readout3d(
        system, slab, scheme="caipi", acceleration=2, partition_acceleration=3
    )
    seq = play(system, epi, [(-16, -6), (0, 3)], ndim=3)
    traced = lines(epi, seq)[:, :, 0, 2] * FOV_Z
    for shot, origin in enumerate((-6, 3)):
        assert traced[shot] == pytest.approx(origin + epi.order[:, 1], abs=1e-6)


def test_a_zigzag_train_returns_to_lines_it_has_already_read(system, excitation):
    """Which is what resolves a signal evolution rather than one image."""
    epi = readout(
        system, excitation, etl=15, scheme="zigzag", acceleration=4, extent=12
    )
    measured = np.round(lines(epi)[0, :, 0, 1] * FOV).astype(int)
    assert len(set(measured)) < epi.etl


# ----------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------


def test_the_train_is_legal_and_its_spacing_is_constant(system, excitation):
    epi = readout(system, excitation, etl=8, acceleration=4)
    assert epi.check_timing()[0]
    spacings = [pp.calc_duration(*block) for block in epi.blocks[2 : 2 + epi.etl]]
    assert spacings == pytest.approx([epi.esp] * epi.etl)


def test_ramp_sampling_buys_a_shorter_echo_spacing(system, excitation):
    ramped = readout(system, excitation, etl=8, acceleration=4)
    square = readout(system, excitation, etl=8, acceleration=4, ramp_sampling=False)
    assert ramped.esp < square.esp
    assert ramped.check_timing()[0]
    assert square.check_timing()[0]


def test_ramp_sampling_spaces_its_samples_unevenly_in_k(system, excitation):
    """The window is uniform in time, and the ramps make that non-uniform in k."""
    ramped = readout(system, excitation, etl=2, acceleration=4)
    square = readout(system, excitation, etl=2, acceleration=4, ramp_sampling=False)
    assert np.ptp(np.diff(lines(ramped)[0, 0, :, 0])) > 1e-3
    assert np.ptp(np.diff(lines(square)[0, 0, :, 0])) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("te", [None, 12e-3])
def test_the_echo_times_are_when_the_read_axis_crosses_zero(system, excitation, te):
    """Measured against the trajectory, not against the module's own arithmetic."""
    epi = readout(system, excitation, etl=4, acceleration=4, te=te)
    k_adc, _, _, _, t_adc = epi.calculate_kspace()
    read = np.asarray(k_adc)[0].reshape(epi.etl, epi.n_samples)
    times = np.asarray(t_adc).reshape(epi.etl, epi.n_samples)
    isodelay = float(epi.rf.delay) + float(epi.rf.center)
    for line in range(epi.etl):
        rising = read[line, 0] < read[line, -1]
        crossing = np.interp(
            0.0,
            read[line] if rising else read[line][::-1],
            times[line] if rising else times[line][::-1],
        )
        assert crossing - isodelay == pytest.approx(epi.echo_times[line], abs=1e-9)


def test_the_echoes_are_one_spacing_apart_from_the_first(system, excitation):
    epi = readout(system, excitation, etl=8, acceleration=4)
    assert epi.echo_times == pytest.approx(epi.echo_time + epi.esp * np.arange(epi.etl))


def test_a_longer_echo_time_is_waited_out(system, excitation):
    tight = readout(system, excitation, etl=8, acceleration=4)
    delayed = readout(
        system, excitation, etl=8, acceleration=4, te=tight.echo_time + 2e-3
    )
    assert delayed.echo_time == pytest.approx(tight.echo_time + 2e-3)
    assert delayed.duration > tight.duration


def test_an_echo_time_shorter_than_the_prewinder_is_refused(system, excitation):
    tight = readout(system, excitation, etl=8, acceleration=4)
    with pytest.raises(ValueError, match="TE"):
        readout(system, excitation, etl=8, acceleration=4, te=0.5 * tight.echo_time)


def test_a_repetition_time_shorter_than_the_train_is_refused(system, excitation):
    tight = readout(system, excitation, etl=8, acceleration=4)
    with pytest.raises(ValueError, match="TR"):
        readout(system, excitation, etl=8, acceleration=4, tr=0.5 * tight.duration)


def test_the_repetition_is_padded_to_the_time_asked_for(system, excitation):
    tight = readout(system, excitation, etl=8, acceleration=4)
    padded = readout(
        system, excitation, etl=8, acceleration=4, tr=tight.duration + 5e-3
    )
    assert padded.duration == pytest.approx(tight.duration + 5e-3)


# ----------------------------------------------------------------------
# Blips
# ----------------------------------------------------------------------


def test_every_step_takes_the_same_time_however_far_it_moves(system, slab):
    """One template scaled, so the echo spacing cannot depend on the ordering."""
    epi = readout3d(
        system,
        slab,
        scheme="caipi",
        acceleration=2,
        partition_acceleration=3,
        caipi_shift=1,
    )
    interior = [pp.calc_duration(event) for event in epi.gy_blips[1:-1]]
    assert interior == pytest.approx([epi.esp] * len(interior))


def test_a_wider_step_costs_a_longer_blip_window(system, excitation):
    narrow = readout(system, excitation, etl=8, acceleration=1)
    wide = readout(system, excitation, etl=8, acceleration=4)
    assert wide.blip_span > narrow.blip_span
    assert wide.esp > narrow.esp


# ----------------------------------------------------------------------
# Flyback
# ----------------------------------------------------------------------


def test_a_flyback_train_reads_every_line_the_same_way(system, excitation):
    epi = readout(system, excitation, etl=8, acceleration=4, flyback=True)
    traced = lines(epi)[0]
    assert np.all(traced[:, 0, 0] < 0)
    assert np.all(traced[:, -1, 0] > 0)
    assert traced[:, 0, 0] == pytest.approx(traced[0, 0, 0])


def test_a_flyback_train_puts_a_rewind_between_its_lines(system, excitation):
    epi = readout(system, excitation, etl=8, acceleration=4, flyback=True)
    assert len(epi.blocks) == 3 + epi.etl + (epi.etl - 1)
    assert epi.blocks[3][0] is epi.gx_flyback
    assert epi.esp == pytest.approx(
        pp.calc_duration(epi.gx[0]) + pp.calc_duration(epi.gx_flyback)
    )


def test_a_flyback_step_is_played_whole_in_the_rewind(system, excitation):
    """No ramp to hide on, so the whole step goes where the rewind already is."""
    epi = readout(system, excitation, etl=4, acceleration=4, flyback=True)
    assert epi.gy_blips[-1] is None
    assert epi.gy_blips[0] in epi.blocks[3]
    assert pp.calc_duration(epi.gy_blips[0]) <= pp.calc_duration(epi.gx_flyback)


def test_a_flyback_train_still_walks_its_ordering(system, slab):
    epi = readout3d(
        system,
        slab,
        scheme="caipi",
        acceleration=2,
        partition_acceleration=3,
        flyback=True,
    )
    traced = lines(epi)[0]
    assert np.round((traced[:, 0, 1] - traced[0, 0, 1]) * FOV).astype(
        int
    ).tolist() == list(epi.order[:, 0])
    assert np.round((traced[:, 0, 2] - traced[0, 0, 2]) * FOV_Z).astype(
        int
    ).tolist() == list(epi.order[:, 1])
    assert traced[:, -1, 1] == pytest.approx(traced[:, 0, 1], abs=1e-9)


def test_the_rewind_costs_what_a_blipped_train_saves(system, excitation):
    blipped = readout(system, excitation, etl=8, acceleration=4)
    flyback = readout(system, excitation, etl=8, acceleration=4, flyback=True)
    assert flyback.esp > blipped.esp
    assert flyback.check_timing()[0]


# ----------------------------------------------------------------------
# The sampling window
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"order": [0]},
        {"etl": 8, "acceleration": 4},
        {"etl": 8, "acceleration": 4, "flyback": True},
        {"etl": 4, "acceleration": 8, "readout_bandwidth_hz": 20e3},
        {
            "etl": 4,
            "acceleration": 8,
            "readout_bandwidth_hz": 20e3,
            "ramp_sampling": False,
        },
    ],
)
def test_the_window_clears_the_receiver_dead_time_and_stays_centred(
    system, excitation, kwargs
):
    """A window the dead time shunts forward is a sweep that no longer straddles k = 0."""
    epi = readout(system, excitation, **kwargs)
    assert float(epi.adc.delay) >= system.adc_dead_time
    traced = lines(epi)[0]
    assert traced[0, 0, 0] == pytest.approx(-traced[0, -1, 0], rel=1e-9)


# ----------------------------------------------------------------------
# Labels
# ----------------------------------------------------------------------


def test_the_shot_counter_is_set_once_and_stepped_per_line(system, slab):
    epi = readout3d(
        system,
        slab,
        scheme="caipi",
        acceleration=2,
        partition_acceleration=3,
        labels=("LIN", "PAR"),
    )
    assert [event.label for event in epi.shot_labels] == ["LIN", "PAR"]
    assert all(event.type == "labelset" for event in epi.shot_labels)
    assert epi.line_labels[0] == ()
    steps = np.diff(epi.order, axis=0)
    for line in range(1, epi.etl):
        assert [event.type for event in epi.line_labels[line]] == ["labelinc"] * 2
        assert [event.value for event in epi.line_labels[line]] == list(steps[line - 1])


def test_the_counters_ride_the_blocks_they_belong_to(system, excitation):
    epi = readout(system, excitation, etl=4, acceleration=4, labels=("LIN",))
    assert epi.shot_labels[0] in epi.blocks[1]
    assert epi.line_labels[2][0] in epi.blocks[2 + 2]


def test_naming_the_wrong_number_of_counters_is_refused(system, excitation):
    with pytest.raises(ValueError, match="counter for each encoded axis"):
        readout(system, excitation, etl=4, acceleration=4, labels=("LIN", "PAR"))


# ----------------------------------------------------------------------
# The ordering: generated or supplied
# ----------------------------------------------------------------------


def test_the_default_train_crosses_the_phase_encode_matrix(system, excitation):
    epi = readout(system, excitation)
    assert epi.etl == MATRIX
    assert epi.order[-1, 0] == MATRIX - 1


def test_segmenting_shortens_the_train(system, excitation):
    epi = readout(system, excitation, segments=4)
    assert epi.etl == MATRIX // 4
    assert epi.order[1, 0] == 4


def test_a_supplied_ordering_is_used_as_it_stands(system, excitation):
    epi = readout(system, excitation, order=[0, 3, 6, 9])
    assert list(epi.order[:, 0]) == [0, 3, 6, 9]
    assert epi.etl == 4


def test_a_supplied_ordering_may_be_two_columns_in_3d(system, slab):
    epi = readout3d(system, slab, order=[[0, 0], [2, 1], [4, 2]])
    assert epi.order.tolist() == [[0, 0], [2, 1], [4, 2]]


def test_a_two_dimensional_train_refuses_to_step_the_partition(system, excitation):
    with pytest.raises(ValueError, match="partition axis"):
        readout(system, excitation, order=[[0, 0], [1, 1]])


def test_an_ordering_wider_than_the_matrix_is_refused(system, excitation):
    with pytest.raises(ValueError, match="does not fit"):
        readout(system, excitation, order=[0, MATRIX])


def test_a_fractional_ordering_is_refused(system, excitation):
    with pytest.raises(ValueError, match="integers"):
        readout(system, excitation, order=[0.0, 1.5, 3.0])


def test_the_ordering_is_published_read_only(system, excitation):
    epi = readout(system, excitation, etl=4, acceleration=4)
    with pytest.raises(ValueError):
        epi.order[0, 0] = 5


# ----------------------------------------------------------------------
# Closing the repetition
# ----------------------------------------------------------------------


def test_the_read_axis_is_rewound_when_nothing_is_spoiled(system, excitation):
    epi = readout(system, excitation, etl=8, acceleration=4)
    assert abs(read_residual(system, epi)) == pytest.approx(0.0, abs=1e-6)


def test_spoiling_leaves_the_read_axis_where_it_was_asked_to(system, excitation):
    epi = readout(system, excitation, etl=8, acceleration=4, spoiling_cycles=4.0)
    assert abs(read_residual(system, epi)) == pytest.approx(
        4.0 / (FOV / MATRIX), rel=1e-6
    )
