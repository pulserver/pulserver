"""The fast spin-echo readouts.

Checked against k-space and against the refocusing times the sequence reports,
rather than against event fields: what an FSE train has to get right is that
every echo starts the same line, that the echoes sit halfway between the
pulses that make them, and that a scan loop replaying the published events
reproduces the timing the module claims.
"""

from __future__ import annotations

import numpy as np
import pytest

import pulserver.design as design
import pulserver.pypulseq as pp

FOV = (0.22, 0.22, 0.16)
MATRIX = (128, 128, 64)
ETL = 8


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
    return design.SpatialSelectiveExcitation(system, 90.0, 5e-3)


@pytest.fixture
def slab(system):
    return design.SpatialSelectiveExcitation(system, 90.0, FOV[2], is_slab=True)


@pytest.fixture
def selective(system):
    return design.SpatialSelectiveRefocusing(system, 5e-3)


@pytest.fixture
def hard(system):
    return design.NonSelectiveRefocusing(system, duration_s=0.6e-3, spoiling_cycles=0.0)


def readout2d(system, slice_, selective, **kwargs):
    kwargs.setdefault("fov", FOV[:2])
    kwargs.setdefault("matrix", MATRIX[:2])
    kwargs.setdefault("etl", ETL)
    return design.FseReadout2D(
        system,
        slice_.rf,
        slice_.gz,
        slice_.gz_reph,
        rf_ref=selective.rf_ref,
        gz_ref=selective.gz,
        **kwargs,
    )


def readout3d(system, slab, hard, **kwargs):
    kwargs.setdefault("fov", FOV)
    kwargs.setdefault("matrix", MATRIX)
    kwargs.setdefault("etl", ETL)
    kwargs.setdefault("spoiling_cycles", 2.0)
    return design.FseReadout3D(
        system, slab.rf, slab.gz, rf_ref=hard.rf_ref, **kwargs
    )


def _lines(module):
    """Every acquisition window's k-space, shaped ``(3, etl, samples)``."""
    k_traj_adc, _, t_excitation, t_refocusing, _ = module.calculate_kspace()
    count = int(module.adc.num_samples)
    return (
        np.asarray(k_traj_adc).reshape(3, module.etl, count),
        float(t_excitation[0]),
        np.asarray(t_refocusing, dtype=float),
    )


def _echo_sample(module, n_full=MATRIX[0]):
    """The sample the read gradient crosses k = 0 just before."""
    return module.n_samples - n_full // 2


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------


def test_a_repetition_is_an_excitation_a_prephaser_and_one_unit_per_echo(
    system, slice_, selective
):
    fse = readout2d(system, slice_, selective)
    stretched = 1 if fse.esp_first > fse.esp else 0
    assert len(fse.blocks) == 2 + 4 * ETL + stretched
    assert fse.blocks[0] == (fse.rf, fse.gz)
    assert fse.blocks[1] == (fse.gx_pre, fse.gz_reph)


def test_the_train_is_valid_pulseq(system, slice_, selective):
    assert readout2d(system, slice_, selective).check_timing()[0]


def test_the_3d_train_is_valid_pulseq(system, slab, hard):
    assert readout3d(system, slab, hard).check_timing()[0]


def test_every_echo_replays_the_same_events(system, slice_, selective):
    fse = readout2d(system, slice_, selective)
    units = [
        fse.blocks[-4 * (echo + 1) : len(fse.blocks) - 4 * echo] for echo in range(ETL - 1)
    ]
    assert all(unit == units[0] for unit in units)


def test_the_readout_block_is_the_lobe_and_the_adc(system, slice_, selective):
    fse = readout2d(system, slice_, selective)
    assert fse.blocks[-2] == (fse.gx, fse.adc)


def test_a_3d_train_encodes_a_partition_axis_as_well(system, slab, hard):
    fse = readout3d(system, slab, hard)
    assert (fse.gy_pre.channel, fse.gz_pre.channel) == ("y", "z")
    assert fse.blocks[-3] == (fse.gx_bridge_pre, fse.gy_pre, fse.gz_pre)


def test_a_2d_train_has_no_partition_encode(system, slice_, selective):
    fse = readout2d(system, slice_, selective)
    assert not hasattr(fse.events, "gz_pre")
    assert not hasattr(fse.events, "gz_rew")


def test_labels_ride_the_acquisition_block(system, slice_, selective):
    fse = readout2d(system, slice_, selective, labels=("LIN", "SLC"))
    assert fse.blocks[-2] == (fse.gx, fse.adc, *fse.adc_labels)
    assert [label.label for label in fse.adc_labels] == ["LIN", "SLC"]


def test_a_trigger_rides_the_prephaser_block(system, slice_, selective):
    trigger = pp.make_trigger("physio1", duration=100e-6)
    fse = readout2d(system, slice_, selective, trigger=trigger)
    assert trigger in fse.blocks[1]


# ----------------------------------------------------------------------
# The phase encodes are scalable trapezoids, not summed waveforms
# ----------------------------------------------------------------------


def test_every_phase_encode_is_a_trapezoid(system, slab, hard):
    fse = readout3d(system, slab, hard)
    assert [event.type for event in (fse.gy_pre, fse.gy_rew, fse.gz_pre, fse.gz_rew)] == [
        "trap"
    ] * 4


def test_scaling_a_partition_encode_keeps_one_shape(system, slab, hard):
    """A scaled trapezoid is the same waveform at a new amplitude.

    Summing the partition encode onto a crusher would make it an arbitrary
    waveform instead, and then every partition is a distinct shape.
    """
    fse = readout3d(system, slab, hard)
    scaled = [pp.scale_grad(fse.gz_pre, kz) for kz in np.linspace(-1.0, 1.0, 17)]
    spans = {(event.rise_time, event.flat_time, event.fall_time) for event in scaled}
    assert len(spans) == 1
    assert all(event.type == "trap" for event in scaled)


def test_the_partition_encode_reaches_the_edge_of_k_space(system, slab, hard):
    fse = readout3d(system, slab, hard)
    assert fse.gz_pre.area == pytest.approx(0.5 * MATRIX[2] / FOV[2])


def test_a_line_is_rewound_before_the_next_refocusing_pulse(system, slab, hard):
    fse = readout3d(system, slab, hard)
    assert fse.gy_rew.area == pytest.approx(-fse.gy_pre.area)
    assert fse.gz_rew.area == pytest.approx(-fse.gz_pre.area)


# ----------------------------------------------------------------------
# K-space: every echo starts and ends the same line
# ----------------------------------------------------------------------


def test_the_read_axis_crosses_zero_at_every_echo(system, slice_, selective):
    fse = readout2d(system, slice_, selective)
    k, _, _ = _lines(fse)
    sample = _echo_sample(fse)
    # The crossing falls between the two samples that straddle it, half a step
    # from each.
    assert k[0, :, sample] == pytest.approx(0.5 * fse.delta_kx, rel=1e-6)
    assert k[0, :, sample - 1] == pytest.approx(-0.5 * fse.delta_kx, rel=1e-6)


def test_every_echo_reads_the_same_line(system, slice_, selective):
    fse = readout2d(system, slice_, selective)
    k, _, _ = _lines(fse)
    assert k[0] == pytest.approx(np.broadcast_to(k[0, 0], k[0].shape))


def test_the_line_spans_the_matrix(system, slice_, selective):
    fse = readout2d(system, slice_, selective)
    k, _, _ = _lines(fse)
    assert k[0, 0, -1] - k[0, 0, 0] == pytest.approx(
        (fse.n_samples - 1) * fse.delta_kx, rel=1e-9
    )


def test_a_partial_echo_still_starts_every_line_at_the_same_place(system, slice_, selective):
    fse = readout2d(system, slice_, selective, partial_echo=0.7)
    k, _, _ = _lines(fse)
    assert fse.n_samples < MATRIX[0]
    assert k[0] == pytest.approx(np.broadcast_to(k[0, 0], k[0].shape))
    sample = _echo_sample(fse)
    assert k[0, :, sample] == pytest.approx(0.5 * fse.delta_kx, rel=1e-6)


def test_a_3d_train_holds_its_line_across_the_train(system, slab, hard):
    fse = readout3d(system, slab, hard)
    k, _, _ = _lines(fse)
    assert k == pytest.approx(np.broadcast_to(k[:, 0:1], k.shape))


# ----------------------------------------------------------------------
# CPMG timing
# ----------------------------------------------------------------------


def _cpmg_error(module):
    """How far each refocusing pulse sits from the midpoint it should be at."""
    _, excitation, refocusing = _lines(module)
    echoes = module.echo_times + excitation
    midpoints = 0.5 * (np.concatenate(([excitation], echoes[:-1])) + echoes)
    return np.abs(refocusing - midpoints)


def test_every_refocusing_pulse_sits_midway_between_the_echoes_it_makes(
    system, slice_, selective
):
    fse = readout2d(system, slice_, selective)
    assert _cpmg_error(fse).max() <= 0.5 * system.block_duration_raster


def test_the_3d_train_is_cpmg_too(system, slab, hard):
    assert _cpmg_error(readout3d(system, slab, hard)).max() <= (
        0.5 * system.block_duration_raster
    )


def test_the_echoes_are_evenly_spaced_after_the_first(system, slice_, selective):
    fse = readout2d(system, slice_, selective)
    assert np.diff(fse.echo_times) == pytest.approx(fse.esp)
    assert fse.echo_times[0] == pytest.approx(fse.esp_first, abs=system.block_duration_raster)


def test_the_train_lasts_one_spacing_per_echo(system, slice_, selective):
    short = readout2d(system, slice_, selective, etl=4)
    long = readout2d(system, slice_, selective, etl=12)
    assert long.duration - short.duration == pytest.approx(8 * short.esp)


# ----------------------------------------------------------------------
# Echo spacing
# ----------------------------------------------------------------------


def test_a_hard_refocusing_pulse_shortens_the_spacing(system, slab, hard, selective):
    """The reason single-slab 3D FSE refocuses non-selectively."""
    short = readout3d(system, slab, hard)
    long = design.FseReadout3D(
        system,
        slab.rf,
        slab.gz,
        rf_ref=selective.rf_ref,
        gz_ref=selective.gz,
        fov=FOV,
        matrix=MATRIX,
        etl=ETL,
    )
    assert short.esp < 0.75 * long.esp


def test_a_longer_spacing_than_asked_for_is_refused(system, slice_, selective):
    minimum = readout2d(system, slice_, selective).esp
    with pytest.raises(ValueError, match="echo spacing"):
        readout2d(system, slice_, selective, esp=0.5 * minimum)


def test_a_requested_spacing_is_delivered_exactly(system, slice_, selective):
    minimum = readout2d(system, slice_, selective).esp
    wanted = minimum + 2e-3
    fse = readout2d(system, slice_, selective, esp=wanted)
    assert fse.esp == pytest.approx(wanted)
    assert _cpmg_error(fse).max() <= 0.5 * system.block_duration_raster


def test_slack_goes_into_the_phase_encode_rather_than_into_dead_time(
    system, slice_, selective
):
    """A longer spacing buys a gentler encode, not an idle gradient axis."""
    tight = readout2d(system, slice_, selective)
    loose = readout2d(system, slice_, selective, esp=tight.esp + 4e-3)
    assert pp.calc_duration(loose.gy_pre) > pp.calc_duration(tight.gy_pre)
    assert abs(loose.gy_pre.amplitude) < abs(tight.gy_pre.amplitude)
    assert loose.gy_pre.area == pytest.approx(tight.gy_pre.area)


def test_crushing_the_read_axis_costs_echo_spacing(system, slab, hard):
    bare = readout3d(system, slab, hard, spoiling_cycles=0.0)
    crushed = readout3d(system, slab, hard, spoiling_cycles=6.0)
    assert crushed.esp > bare.esp


def test_a_bare_train_bridges_the_readout_with_its_own_ramps(system, slab, hard):
    """With nothing to crush, the read axis is the plateau's ramps and no more."""
    fse = readout3d(system, slab, hard, spoiling_cycles=0.0)
    plateau = pp.make_trapezoid(
        channel="x",
        flat_area=fse.n_samples * fse.delta_kx,
        flat_time=fse.readout_duration,
        system=system,
    )
    # The lobes fill their blocks, so it is the part of them that is not zero
    # that has to be the ramp and nothing else.
    entry = np.asarray(fse.gx_bridge_pre.tt)
    exit_ = np.asarray(fse.gx_bridge_post.tt)
    exit_amplitudes = np.abs(np.asarray(fse.gx_bridge_post.waveform))
    assert entry[-1] == pytest.approx(plateau.rise_time, abs=system.grad_raster_time)
    assert exit_[int(np.argmax(exit_amplitudes <= 1e-6))] == pytest.approx(
        plateau.fall_time, abs=system.grad_raster_time
    )


# ----------------------------------------------------------------------
# The first echo spacing
# ----------------------------------------------------------------------


def test_a_long_excitation_lengthens_only_the_first_spacing(system, slab, hard):
    fse = readout3d(system, slab, hard)
    assert fse.esp_first > fse.esp
    assert np.diff(fse.echo_times) == pytest.approx(fse.esp)


def test_the_first_spacing_is_still_cpmg(system, slab, hard):
    fse = readout3d(system, slab, hard)
    _, excitation, refocusing = _lines(fse)
    first_echo = fse.echo_times[0] + excitation
    assert refocusing[0] - excitation == pytest.approx(
        first_echo - refocusing[0], abs=system.block_duration_raster
    )


def test_a_short_excitation_needs_no_extra_first_spacing(system, hard):
    excitation = design.NonSelectiveExcitation(system, 90.0, duration_s=0.5e-3)
    fse = design.FseReadout3D(
        system, excitation.rf, rf_ref=hard.rf_ref, fov=FOV, matrix=MATRIX, etl=ETL
    )
    assert fse.esp_first == pytest.approx(fse.esp)
    assert not hasattr(fse.events, "wait_esp1")


def test_a_first_spacing_shorter_than_the_excitation_is_refused(system, slab, hard):
    fse = readout3d(system, slab, hard)
    with pytest.raises(ValueError, match="first echo spacing"):
        readout3d(system, slab, hard, esp=fse.esp, esp_first=fse.esp)


def test_a_wider_spacing_absorbs_the_excitation(system, slab, hard):
    fse = readout3d(system, slab, hard)
    wide = readout3d(system, slab, hard, esp=fse.esp_first)
    assert wide.esp_first == pytest.approx(wide.esp)


# ----------------------------------------------------------------------
# FID displacement: the net moment over one spacing
# ----------------------------------------------------------------------


def test_the_phase_encode_axes_are_balanced_over_a_spacing(system, slab, hard):
    fse = readout3d(system, slab, hard)
    k, _, _ = _lines(fse)
    # Read at the same sample of consecutive echoes: the encode axes come back
    # to where they were, so an FID picks up no displacement from them.
    assert np.diff(k[1, :, 0]) == pytest.approx(0.0, abs=1e-9)
    assert np.diff(k[2, :, 0]) == pytest.approx(0.0, abs=1e-9)


def _area(event):
    return float(np.trapezoid(np.asarray(event.waveform), np.asarray(event.tt)))


def test_crushing_adds_its_asked_for_area_to_both_read_lobes(system, slab, hard):
    """The net read moment is what pushes the FID out of the sampled window."""
    cycles, voxel = 6.0, 0.8e-3
    bare = readout3d(system, slab, hard, spoiling_cycles=0.0, voxel_size_m=voxel)
    crushed = readout3d(system, slab, hard, spoiling_cycles=cycles, voxel_size_m=voxel)
    for lobe in ("gx_bridge_pre", "gx_bridge_post"):
        added = _area(getattr(crushed, lobe)) - _area(getattr(bare, lobe))
        assert added == pytest.approx(cycles / voxel, rel=1e-6)


# ----------------------------------------------------------------------
# What the scan loop sees
# ----------------------------------------------------------------------


def test_the_refocusing_pulse_is_one_event_at_one_amplitude(system, slice_, selective):
    fse = readout2d(system, slice_, selective)
    assert fse.rf_ref is selective.rf_ref
    assert all(fse.rf_ref in block for block in fse.blocks if fse.rf_ref in block)


def test_a_scan_loop_reproduces_the_timing_the_module_reports(system, slab, hard):
    """The published events, replayed, are the module -- to the nanosecond."""
    fse = readout3d(system, slab, hard)
    seq = pp.Sequence(system)
    seq.add_block(fse.rf, fse.gz)
    seq.add_block(fse.gx_pre)
    for echo in range(fse.etl):
        ky = (echo - 0.5 * fse.etl) / (0.5 * fse.etl)
        kz = ((echo * 7) % MATRIX[2] - 0.5 * MATRIX[2]) / (0.5 * MATRIX[2])
        seq.add_block(fse.rf_ref)
        if echo == 0 and fse.esp_first > fse.esp:
            seq.add_block(fse.wait_esp1)
        seq.add_block(
            fse.gx_bridge_pre,
            pp.scale_grad(fse.gy_pre, ky),
            pp.scale_grad(fse.gz_pre, kz),
        )
        seq.add_block(fse.gx, fse.adc)
        seq.add_block(
            fse.gx_bridge_post,
            pp.scale_grad(fse.gy_rew, ky),
            pp.scale_grad(fse.gz_rew, kz),
        )
    assert seq.duration()[0] == pytest.approx(fse.duration, abs=1e-12)
    assert seq.check_timing()[0]


def test_a_scan_loop_can_vary_the_refocusing_flip(system, slice_, selective):
    fse = readout2d(system, slice_, selective)
    nominal = fse.rf_ref.amplitude
    schedule = pp.make_traps_schedule(fse.etl, np.deg2rad(120.0), variable=True)
    try:
        seq = pp.Sequence(system)
        for flip in schedule:
            fse.rf_ref.amplitude = nominal * float(flip) / np.pi
            seq.add_block(fse.rf_ref, fse.gz_ref)
        assert seq.check_timing()[0]
        assert len(seq.block_durations) == fse.etl
    finally:
        fse.rf_ref.amplitude = nominal


def test_the_repetition_time_pads_the_end(system, slice_, selective):
    tight = readout2d(system, slice_, selective)
    padded = readout2d(system, slice_, selective, tr=tight.duration + 30e-3)
    assert padded.duration == pytest.approx(tight.duration + 30e-3)
    assert padded.wait_tr.delay == pytest.approx(30e-3)


def test_a_repetition_time_shorter_than_the_train_is_refused(system, slice_, selective):
    tight = readout2d(system, slice_, selective)
    with pytest.raises(ValueError, match="TR"):
        readout2d(system, slice_, selective, tr=0.5 * tight.duration)


# ----------------------------------------------------------------------
# Arguments
# ----------------------------------------------------------------------


def test_the_refocusing_pulse_must_say_it_refocuses(system, slice_):
    excitation = design.NonSelectiveExcitation(system, 180.0, duration_s=0.6e-3)
    with pytest.raises(ValueError, match="use='refocusing'"):
        design.FseReadout2D(
            system,
            slice_.rf,
            slice_.gz,
            slice_.gz_reph,
            rf_ref=excitation.rf,
            fov=FOV[:2],
            matrix=MATRIX[:2],
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"etl": 0}, "etl"),
        ({"spoiling_cycles": -1.0}, "spoiling_cycles"),
        ({"oversampling": 0.5}, "oversampling"),
        ({"partial_echo": 0.4}, "partial_echo"),
        ({"readout_bandwidth_hz": 0.0}, "readout_bandwidth_hz"),
        ({"matrix": (1, 128)}, "matrix"),
        ({"fov": (0.0, 0.22)}, "fov"),
        ({"voxel_size_m": 0.0}, "voxel_size_m"),
    ],
)
def test_an_impossible_request_is_refused(system, slice_, selective, kwargs, message):
    with pytest.raises(ValueError, match=message):
        readout2d(system, slice_, selective, **kwargs)


def test_a_rephaser_on_the_read_axis_has_nowhere_to_sit(system, selective):
    on_x = design.SpatialSelectiveExcitation(system, 90.0, 5e-3, axis="x")
    with pytest.raises(ValueError, match="already plays a gradient on x"):
        design.FseReadout2D(
            system,
            on_x.rf,
            on_x.gz,
            on_x.gz_reph,
            rf_ref=selective.rf_ref,
            gz_ref=selective.gz,
            fov=FOV[:2],
            matrix=MATRIX[:2],
        )


def test_a_single_echo_train_is_a_spin_echo(system, slice_, selective):
    fse = readout2d(system, slice_, selective, etl=1)
    assert fse.etl == 1
    assert fse.echo_times.size == 1
    assert fse.check_timing()[0]


def test_a_train_with_no_phase_encode_keeps_its_echo_spacing(system, slice_, selective):
    """A multi-echo spin echo is this train with the encodes left out."""
    fse = readout2d(system, slice_, selective, labels=("ECO",))
    seq = pp.Sequence(system)
    seq.add_block(fse.rf, fse.gz)
    seq.add_block(fse.gx_pre, fse.gz_reph)
    for echo in range(fse.etl):
        fse.adc_labels.value = echo
        seq.add_block(fse.rf_ref, fse.gz_ref)
        if echo == 0 and fse.esp_first > fse.esp:
            seq.add_block(fse.wait_esp1)
        seq.add_block(fse.gx_bridge_pre)
        seq.add_block(fse.gx, fse.adc, fse.adc_labels)
        seq.add_block(fse.gx_bridge_post)
    assert seq.duration()[0] == pytest.approx(fse.duration, abs=1e-12)
    assert seq.check_timing()[0]
