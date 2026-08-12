"""The Cartesian line readouts.

Checked against k-space rather than against event fields wherever possible:
the module's job is to put the echo where it says it did, and
`calculate_kspace` is the only thing that answers that without re-deriving the
arithmetic under test.
"""

from __future__ import annotations

import numpy as np
import pytest

import pulserver.design as design
import pulserver.pypulseq as pp

FOV = (0.22, 0.22, 0.12)
MATRIX = (128, 128, 64)


@pytest.fixture
def system():
    return pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")


@pytest.fixture
def slab(system):
    return design.SpatialSelectiveExcitation(system, 8.0, FOV[2], is_slab=True)


def readout3d(system, slab, **kwargs):
    return design.LineReadout3D(
        system, slab.rf, slab.gz_slab, fov_m=FOV, matrix=MATRIX, **kwargs
    )


def _first_line(module):
    """K-space of the first acquisition window, and the times it was sampled at."""
    k_traj_adc, _, t_excitation, _, t_adc = module.calculate_kspace()
    count = int(module.adc.num_samples)
    return k_traj_adc[:, :count], np.asarray(t_adc)[:count], float(t_excitation[0])


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------


def test_a_repetition_is_pulse_prewinder_readout_rewinder(system, slab):
    readout = readout3d(system, slab)
    assert len(readout.blocks) == 4
    assert readout.blocks[0] == (readout.rf, readout.gz_select)
    assert readout.blocks[2] == (readout.gx_read, readout.adc)


def test_the_readout_is_valid_pulseq(system, slab):
    assert readout3d(system, slab).check_timing()[0]


def test_a_2d_readout_encodes_one_phase_axis(system):
    excitation = design.SpatialSelectiveExcitation(system, 15.0, 5e-3)
    readout = design.LineReadout2D(
        system, excitation.rf, excitation.gz_select, fov_m=0.22, matrix=128
    )
    assert readout.gy_phase.channel == "y"
    assert not hasattr(readout.events, "gz_phase")
    assert readout.check_timing()[0]


def test_a_hard_pulse_needs_no_selection_gradient(system, slab):
    excitation = design.NonSelectiveExcitation(system, 8.0)
    bare = design.LineReadout3D(system, excitation.rf, fov_m=FOV, matrix=MATRIX)
    assert bare.blocks[0] == (bare.rf,)
    assert len(bare.blocks) == len(readout3d(system, slab).blocks)
    assert bare.check_timing()[0]


# ----------------------------------------------------------------------
# Where the echo lands
# ----------------------------------------------------------------------


def test_the_line_is_centred_on_k_zero(system, slab):
    readout = readout3d(system, slab)
    kx, _, _ = _first_line(readout)
    half_width = MATRIX[0] / (2 * FOV[0])
    half_step = 0.5 * readout.sampling.delta_k
    # Samples sit at cell centres, so the extremes fall half a step inside.
    assert kx[0].min() == pytest.approx(-half_width + half_step)
    assert kx[0].max() == pytest.approx(half_width - half_step)


def test_the_echo_falls_at_the_echo_time_it_reports(system, slab):
    readout = readout3d(system, slab, te=5e-3)
    kx, t_adc, t_excitation = _first_line(readout)
    nearest = int(np.argmin(np.abs(kx[0])))
    # The k = 0 crossing lies between two sample centres, half a dwell from
    # whichever is nearer.
    assert t_adc[nearest] - t_excitation == pytest.approx(
        readout.echo_time, abs=0.5 * readout.adc.dwell + 1e-9
    )
    assert readout.echo_time == pytest.approx(5e-3)


def test_the_ramp_of_the_readout_lobe_is_prephased_too(system, slab):
    """Half a ramp of k accrues before the first sample.

    Left uncompensated it offsets the whole line -- which a reconstruction
    sees as a first-order phase, not as an error, so nothing downstream would
    report it.
    """
    readout = readout3d(system, slab)
    kx, _, _ = _first_line(readout)
    centre = 0.5 * (kx[0].min() + kx[0].max())
    assert centre == pytest.approx(0.0, abs=1e-9 * MATRIX[0] / FOV[0])


def test_partial_echo_truncates_the_leading_side_and_shortens_te(system, slab):
    full = readout3d(system, slab)
    partial = readout3d(system, slab, partial_echo=0.7)

    assert partial.echo_time < full.echo_time
    kx, _, _ = _first_line(partial)
    assert kx[0].max() == pytest.approx(_first_line(full)[0][0].max())
    assert kx[0].min() > _first_line(full)[0][0].min()


# ----------------------------------------------------------------------
# Encoding
# ----------------------------------------------------------------------


@pytest.mark.parametrize("axis", [1, 2])
def test_the_phase_encode_step_is_set_by_resolution_alone(system, slab, axis):
    readout = readout3d(system, slab)
    encode = readout.gy_phase if axis == 1 else readout.gz_phase
    resolution = FOV[axis] / MATRIX[axis]
    assert encode.area == pytest.approx(1.0 / (2.0 * resolution))


@pytest.mark.parametrize("scale", [-1.0, -0.25, 0.0, 0.75])
def test_scaling_the_encode_moves_the_line_in_k(system, slab, scale):
    """What a scan loop does per shot, and the only thing it has to do."""
    readout = readout3d(system, slab)
    seq = pp.Sequence(system)
    seq.add_block(readout.rf, readout.gz_select)
    seq.add_block(
        readout.gx_pre,
        pp.scale_grad(readout.gy_phase, scale),
        pp.scale_grad(readout.gz_phase, 0.0),
    )
    seq.add_block(readout.gx_read, readout.adc)

    k_traj_adc = seq.calculate_kspace()[0]
    assert np.allclose(k_traj_adc[1], scale * readout.gy_phase.area, atol=1e-6)
    assert np.allclose(k_traj_adc[2], 0.0, atol=1e-6)


def test_the_encodes_are_unwound_by_their_rewinders(system, slab):
    readout = readout3d(system, slab)
    assert readout.gy_rew.area == pytest.approx(-readout.gy_phase.area)
    assert readout.gz_rew.area == pytest.approx(-readout.gz_phase.area)


def test_a_balanced_readout_ends_where_it_started(system, slab):
    """Every axis returns to k = 0 by the end of the TR."""
    readout = readout3d(system, slab)
    seq = pp.Sequence(system)
    for block in readout.blocks:
        seq.add_block(*block)
    k_traj = seq.calculate_kspace()[1]
    assert np.allclose(k_traj[:, -1], 0.0, atol=1e-6 * MATRIX[0] / FOV[0])


# ----------------------------------------------------------------------
# Spoiling
# ----------------------------------------------------------------------


def test_post_spoiling_leaves_the_residual_it_was_asked_for(system, slab):
    cycles, voxel = 4.0, 1e-3
    readout = readout3d(system, slab, spoiling_cycles=cycles, voxel_size_m=voxel)
    seq = pp.Sequence(system)
    for block in readout.blocks:
        seq.add_block(*block)
    k_traj = seq.calculate_kspace()[1]
    assert k_traj[0, -1] == pytest.approx(cycles / voxel, rel=1e-3)


def test_post_spoiling_keeps_the_acquisition_centred(system, slab):
    """The dephasing follows the readout, so the line itself does not move."""
    balanced = readout3d(system, slab)
    spoiled = readout3d(system, slab, spoiling_cycles=4.0)
    assert np.allclose(_first_line(spoiled)[0][0], _first_line(balanced)[0][0], atol=1e-9)


def test_pre_spoiling_offsets_the_fid_pathway(system, slab):
    """A dephasing lobe before the readout must offset this TR's own k-space.

    What such a sequence reads is the echo refocused from the previous
    excitation, not this one's FID -- so a trajectory traced from rest never
    crossing k = 0 is the pathway distinction, not a design fault.
    """
    cycles, voxel = 4.0, 1e-3
    readout = readout3d(
        system, slab, spoiling_cycles=cycles, voxel_size_m=voxel, spoiling_position="pre"
    )
    kx, _, _ = _first_line(readout)
    assert kx[0].max() < 0
    assert kx[0].min() == pytest.approx(
        _first_line(readout3d(system, slab))[0][0].min() - cycles / voxel, rel=1e-3
    )


def test_the_spoiler_rides_the_readout_lobe_rather_than_waiting_for_it(system, slab):
    """Bridging is what keeps a short-TR steady-state sequence short."""
    spoiled = readout3d(system, slab, spoiling_cycles=4.0)
    assert spoiled.gx_rew.type == "grad"
    assert spoiled.gx_rew.first == pytest.approx(spoiled.gx_read.amplitude, rel=1e-6)
    assert spoiled.gx_rew.last == pytest.approx(0.0, abs=1e-6)


def test_the_default_voxel_is_the_readout_resolution(system, slab):
    explicit = readout3d(system, slab, spoiling_cycles=2.0, voxel_size_m=FOV[0] / MATRIX[0])
    implied = readout3d(system, slab, spoiling_cycles=2.0)
    assert implied.duration == pytest.approx(explicit.duration)


# ----------------------------------------------------------------------
# Echo trains
# ----------------------------------------------------------------------


def test_a_monopolar_train_reads_every_echo_the_same_way(system, slab):
    readout = readout3d(system, slab, num_echoes=3)
    assert readout.gx_flyback.area == pytest.approx(-readout.gx_read.area)
    lobes = [block[0] for block in readout.blocks if any(getattr(e, "type", "") == "adc" for e in block)]
    assert len(lobes) == 3
    assert all(lobe is readout.gx_read for lobe in lobes)


def test_a_bipolar_train_alternates_and_costs_no_rewinder(system, slab):
    monopolar = readout3d(system, slab, num_echoes=3)
    bipolar = readout3d(system, slab, num_echoes=3, flyback=False)

    assert bipolar.gx_read_reversed.amplitude == pytest.approx(-bipolar.gx_read.amplitude)
    assert len(bipolar.blocks) < len(monopolar.blocks)
    assert bipolar.duration < monopolar.duration


def test_every_echo_of_a_monopolar_train_traces_the_same_line(system, slab):
    readout = readout3d(system, slab, num_echoes=2)
    k_traj_adc = readout.calculate_kspace()[0]
    count = int(readout.adc.num_samples)
    assert np.allclose(k_traj_adc[0, :count], k_traj_adc[0, count : 2 * count], atol=1e-6)


# ----------------------------------------------------------------------
# Timing budget
# ----------------------------------------------------------------------


def test_a_longer_te_or_tr_is_padded_with_a_wait(system, slab):
    short = readout3d(system, slab)
    stretched = readout3d(system, slab, te=6e-3, tr=15e-3)
    assert stretched.echo_time == pytest.approx(6e-3)
    assert stretched.duration == pytest.approx(15e-3)
    assert len(stretched.blocks) == len(short.blocks) + 2


@pytest.mark.parametrize(("kwargs", "name"), [({"te": 1e-4}, "TE"), ({"tr": 1e-4}, "TR")])
def test_an_impossible_time_is_refused_by_name(system, slab, kwargs, name):
    with pytest.raises(ValueError, match=f"requested {name}"):
        readout3d(system, slab, **kwargs)


def test_the_achieved_bandwidth_is_reported_not_the_requested_one(system, slab):
    readout = readout3d(system, slab, bandwidth_hz=250e3)
    assert readout.bandwidth_hz == pytest.approx(1.0 / readout.adc.dwell)
    dwell_steps = readout.adc.dwell / system.adc_raster_time
    assert dwell_steps == pytest.approx(round(dwell_steps))
    flat_steps = readout.sampling.duration / system.grad_raster_time
    assert flat_steps == pytest.approx(round(flat_steps))


def test_oversampling_densifies_without_changing_resolution(system, slab):
    plain = readout3d(system, slab)
    dense = readout3d(system, slab, oversampling=2.0)
    assert dense.adc.num_samples == 2 * plain.adc.num_samples
    assert _first_line(dense)[0][0].max() == pytest.approx(
        _first_line(plain)[0][0].max(), rel=1e-2
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_echoes": 0}, "num_echoes must be >= 1"),
        ({"spoiling_position": "middle"}, "spoiling_position must be one of"),
        ({"spoiling_cycles": -1.0}, "spoiling_cycles must be >= 0"),
        ({"partial_echo": 0.4}, r"partial_echo must be in \(0.5, 1\]"),
        ({"oversampling": 0.5}, "oversampling must be >= 1"),
        ({"axes": ("x", "x", "z")}, "axes must be 3 distinct gradient channels"),
    ],
)
def test_an_impossible_readout_is_refused(system, slab, kwargs, message):
    with pytest.raises(ValueError, match=message):
        readout3d(system, slab, **kwargs)


def test_a_mismatched_fov_says_how_many_it_wanted(system, slab):
    with pytest.raises(ValueError, match="fov_m must be a scalar or 3 values"):
        design.LineReadout3D(
            system, slab.rf, slab.gz_slab, fov_m=(0.22, 0.22), matrix=MATRIX
        )


# ----------------------------------------------------------------------
# End to end
# ----------------------------------------------------------------------


def test_a_whole_3d_scan_is_a_plain_pypulseq_loop(system, slab, tmp_path):
    """Design once, index per shot, write. No module writes a scan loop."""
    readout = design.LineReadout3D(
        system, slab.rf, slab.gz_slab, fov_m=FOV, matrix=MATRIX,
        te=4e-3, tr=10e-3, spoiling_cycles=4.0, labels=("LIN", "PAR"),
    )
    lines, partitions = 8, 4
    phases = pp.make_rf_spoiling_schedule(lines * partitions)
    lin_label, par_label = readout.adc_labels

    seq = pp.Sequence(system)
    for shot, (line, partition) in enumerate(
        (line, partition) for line in range(lines) for partition in range(partitions)
    ):
        readout.rf.phase_offset = readout.adc.phase_offset = phases[shot]
        lin_label.value, par_label.value = line, partition
        ky = (line - lines / 2) / (lines / 2)
        kz = (partition - partitions / 2) / (partitions / 2)

        seq.add_block(readout.rf, readout.gz_select)
        seq.add_block(readout.wait_te)
        seq.add_block(
            readout.gx_pre,
            pp.scale_grad(readout.gy_phase, ky),
            pp.scale_grad(readout.gz_phase, kz),
        )
        seq.add_block(readout.gx_read, readout.adc, *readout.adc_labels)
        seq.add_block(
            readout.gx_rew,
            pp.scale_grad(readout.gy_rew, ky),
            pp.scale_grad(readout.gz_rew, kz),
        )
        seq.add_block(readout.wait_tr)

    assert seq.check_timing()[0]
    path = tmp_path / "gre3d.seq"
    seq.write(str(path))

    written = pp.Sequence()
    written.read(str(path))
    assert written.num_blocks == seq.num_blocks
    assert written.duration()[0] == pytest.approx(lines * partitions * 10e-3, rel=1e-6)
