"""TransformFOV: scaling, moving and turning the volume of a built sequence."""

from __future__ import annotations

import numpy as np
import pulserver.pypulseq as pp
import pytest

from scipy.spatial.transform import Rotation


@pytest.fixture
def system():
    return pp.Opts(
        max_grad=32,
        grad_unit="mT/m",
        max_slew=130,
        slew_unit="T/m/s",
        rf_ringdown_time=20e-6,
        rf_dead_time=100e-6,
        adc_dead_time=10e-6,
    )


@pytest.fixture
def readout(system):
    """A flat readout: constant gradient, so a shift is two scalars and no shape."""
    return (
        pp.make_trapezoid(
            channel="x", flat_area=581.8, flat_time=3.2e-3, system=system
        ),
        pp.make_adc(num_samples=128, dwell=2.4e-5, delay=1e-4, system=system),
    )


@pytest.fixture
def seq(system, readout):
    """Three identical readouts, so a block range has something to leave alone.

    Deduplicated, which is to say held at the precision the file writes: the
    transform works there, because the phase it bakes has to agree with the
    gradients a consumer will read back rather than with the ones that were
    briefly in memory. A readout built at 581.8/3.2e-3 is 181812.49999999997
    in memory and 181812 in the file.
    """
    gx, adc = readout
    built = pp.Sequence(system=system)
    for _ in range(3):
        built.add_block(gx, adc)
    return built.remove_duplicates(in_place=True)


def _flag(label, value=1):
    return pp.make_label(type="SET", label=label, value=value)


def _phases(seq):
    return [seq.get_block(i).adc.phase_offset for i in range(1, seq.num_blocks + 1)]


def _amplitudes(seq):
    return [seq.get_block(i).gx.amplitude for i in range(1, seq.num_blocks + 1)]


def _over(seq, first, last):
    """``time_range`` covering blocks ``first..last``, 1-based inclusive."""
    edges = np.concatenate(([0.0], np.cumsum(seq.block_durations)))
    return [float(edges[first - 1]), float(edges[last])]


# %% what it refuses


def test_a_transform_that_does_nothing_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        pp.TransformFOV()


def test_a_homogeneous_transform_cannot_be_combined_with_its_parts():
    with pytest.raises(ValueError, match="not both"):
        pp.TransformFOV(transform=np.eye(4), translation=(1.0, 0.0, 0.0))


def test_baking_the_rotation_into_the_gradients_says_it_is_not_implemented():
    with pytest.raises(NotImplementedError, match="rotate3D"):
        pp.TransformFOV(rotation=np.eye(3), use_rotation_extension=False)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"rotation": np.eye(2)}, r"\(3, 3\)"),
        ({"transform": np.eye(3)}, r"\(4, 4\)"),
        ({"translation": (1.0, 2.0)}, "three numbers"),
        ({"scale": (1.0, 1.0, 1.0, 1.0)}, "three numbers"),
    ],
)
def test_a_misshapen_transform_is_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        pp.TransformFOV(**kwargs)


def test_a_backwards_time_range_is_rejected(seq):
    with pytest.raises(ValueError, match="after begin time"):
        pp.TransformFOV(translation=(1.0, 0.0, 0.0)).apply_to_sequence(
            seq, time_range=[0.02, 0.0]
        )


def test_a_time_range_past_the_end_clamps_to_the_last_block(seq):
    """Unlike a block range, a window is not a claim about how long the
    sequence is -- asking past the end asks for what is there."""
    whole = pp.TransformFOV(translation=(10.0, 0.0, 0.0)).apply_to_sequence(seq)
    past = pp.TransformFOV(translation=(10.0, 0.0, 0.0)).apply_to_sequence(
        seq, time_range=[0.0, 9.0]
    )
    assert _phases(past) == _phases(whole)


# %% translation


def test_a_shift_along_the_readout_becomes_a_frequency_and_a_phase(seq):
    """A constant gradient: the shift is exactly `dr . g`, and needs no shape."""
    shift_mm = 10.0
    amplitude = seq.get_block(1).gx.amplitude  # Hz/m

    shapes_before = seq._native.num_shapes()
    pp.TransformFOV(translation=(shift_mm, 0.0, 0.0)).apply_to_sequence(
        seq, in_place=True
    )

    assert seq.get_block(1).adc.freq_offset == pytest.approx(
        amplitude * shift_mm * 1e-3
    )
    assert seq._native.num_shapes() == shapes_before  # no residual was needed


def test_a_shift_along_an_axis_with_no_gradient_does_nothing(seq):
    before = _phases(seq)
    pp.TransformFOV(translation=(0.0, 12.0, 0.0)).apply_to_sequence(seq, in_place=True)
    assert _phases(seq) == before


def test_a_time_range_leaves_the_blocks_outside_it_alone(seq):
    before = _phases(seq)
    pp.TransformFOV(translation=(10.0, 0.0, 0.0)).apply_to_sequence(
        seq, time_range=_over(seq, 2, 2), in_place=True
    )
    after = _phases(seq)
    assert after[1] != before[1]
    assert (after[0], after[2]) == (before[0], before[2])


def test_a_ranged_shift_gives_the_same_phase_as_shifting_everything(seq):
    """Absolute k is accumulated from block 1, so the boundary does not matter."""
    whole = pp.TransformFOV(translation=(10.0, 0.0, 0.0)).apply_to_sequence(seq)
    part = pp.TransformFOV(translation=(10.0, 0.0, 0.0)).apply_to_sequence(
        seq, time_range=_over(seq, 3, 3)
    )
    assert part.get_block(3).adc.phase_offset == pytest.approx(
        whole.get_block(3).adc.phase_offset
    )


def test_server_mode_on_a_cartesian_sequence_equals_native_mode(seq):
    """Unrotated constant-gradient readouts are exact as two scalars, so
    server mode bakes them like native and has nothing to defer."""
    server = pp.TransformFOV(
        translation=(10.0, 0.0, 0.0), server_mode=True
    ).apply_to_sequence(seq)
    native = pp.TransformFOV(translation=(10.0, 0.0, 0.0)).apply_to_sequence(seq)

    assert _phases(server) == pytest.approx(_phases(native))
    assert (
        server.get_block(1).adc.freq_offset
        == native.get_block(1).adc.freq_offset
        != 0.0
    )
    assert not server._native.has_base_trajectory()


def test_server_mode_defers_a_rotated_readout_and_attaches_its_trajectory(seq):
    """Under a rotation the shift phase is orientation-dependent, so the ADC
    side is left to the consumer, which gets the trajectory to do it with."""
    turn = Rotation.from_euler("z", 45.0, degrees=True)
    rotated = pp.TransformFOV(rotation=turn.as_matrix()).apply_to_sequence(seq)
    before = _phases(rotated)

    shifted = pp.TransformFOV(
        translation=(10.0, 0.0, 0.0), server_mode=True
    ).apply_to_sequence(rotated)

    assert _phases(shifted) == before
    assert shifted.get_block(1).adc.freq_offset == 0.0
    assert shifted._native.has_base_trajectory()


def test_a_rotation_carried_by_the_same_server_transform_still_defers(seq):
    """The rotation is composed before the shift looks, so a transform that
    both turns and moves does not bake scalars the consumer would repeat."""
    turn = Rotation.from_euler("y", 30.0, degrees=True)
    out = pp.TransformFOV(
        rotation=turn.as_matrix(), translation=(10.0, 0.0, 0.0), server_mode=True
    ).apply_to_sequence(seq)

    assert out.get_block(1).adc.freq_offset == 0.0
    assert out._native.has_base_trajectory()


def test_native_mode_bakes_the_adc_and_attaches_nothing(seq):
    pp.TransformFOV(translation=(10.0, 0.0, 0.0)).apply_to_sequence(seq, in_place=True)
    assert seq.get_block(1).adc.freq_offset != 0.0
    assert not seq._native.has_base_trajectory()


def test_a_shift_through_a_slice_select_becomes_an_rf_frequency(system):
    """The gradient is constant across the pulse, so the shift is the classic
    slice-offset pair -- a frequency, a phase zeroed at the centre, no shape."""
    rf, gz, gzr = pp.make_sinc_pulse(
        flip_angle=np.pi / 2,
        duration=1e-3,
        slice_thickness=5e-3,
        apodization=0.5,
        time_bw_product=4,
        delay=1e-4,
        system=system,
        return_gz=True,
        use="excitation",
    )
    built = pp.Sequence(system=system)
    built.add_block(rf, gz)
    built.add_block(gzr)

    shift_mm = 10.0
    shapes_before = built._native.num_shapes()
    pp.TransformFOV(translation=(0.0, 0.0, shift_mm)).apply_to_sequence(
        built, in_place=True
    )

    expected_hz = gz.amplitude * shift_mm * 1e-3
    assert built.get_block(1).rf.freq_offset == pytest.approx(expected_hz)
    assert built.get_block(1).rf.phase_offset == pytest.approx(
        -2.0 * np.pi * expected_hz * rf.center
    )
    assert built._native.num_shapes() == shapes_before


# %% scale


def test_a_scale_multiplies_the_amplitude_and_registers_no_shape(seq):
    before = _amplitudes(seq)
    shapes_before = seq._native.num_shapes()

    pp.TransformFOV(scale=(0.5, 1.0, 1.0)).apply_to_sequence(seq, in_place=True)

    assert _amplitudes(seq) == pytest.approx([0.5 * a for a in before])
    assert seq._native.num_shapes() == shapes_before


def test_a_scale_of_zero_flattens_the_axis(seq):
    pp.TransformFOV(scale=(0.0, 1.0, 1.0)).apply_to_sequence(seq, in_place=True)
    assert _amplitudes(seq) == pytest.approx([0.0, 0.0, 0.0])


def test_scaling_an_arbitrary_gradient_keeps_its_waveform(system):
    wave = np.sin(np.linspace(0, np.pi, 200)) * 1e5
    built = pp.Sequence(system=system)
    built.add_block(pp.make_arbitrary_grad(channel="y", waveform=wave, system=system))
    built.remove_duplicates(in_place=True)

    before = built.get_block(1).gy.waveform
    pp.TransformFOV(scale=(1.0, 2.0, 1.0)).apply_to_sequence(built, in_place=True)

    np.testing.assert_allclose(built.get_block(1).gy.waveform, 2.0 * before, rtol=1e-12)


def test_scale_is_applied_before_the_translation(seq):
    """Halving the gradient halves the phase a given shift produces."""
    plain = pp.TransformFOV(translation=(10.0, 0.0, 0.0)).apply_to_sequence(seq)
    scaled = pp.TransformFOV(
        translation=(10.0, 0.0, 0.0), scale=(0.5, 1.0, 1.0)
    ).apply_to_sequence(seq)
    assert scaled.get_block(1).adc.freq_offset == pytest.approx(
        0.5 * plain.get_block(1).adc.freq_offset
    )


# %% rotation


def test_a_rotation_is_attached_as_an_extension(seq):
    turn = Rotation.from_euler("z", 30.0, degrees=True)
    pp.TransformFOV(rotation=turn.as_matrix()).apply_to_sequence(seq, in_place=True)

    for index in range(1, seq.num_blocks + 1):
        stored = Rotation.from_quat(seq.get_block(index).rotation, scalar_first=True)
        assert stored.approx_equal(turn, atol=1e-12)


def test_a_rotation_composes_with_one_the_block_already_carried(system, readout):
    gx, adc = readout
    built = pp.Sequence(system=system)
    built.add_block(
        gx, adc, pp.make_rotation(Rotation.from_euler("z", 30.0, degrees=True))
    )
    built.remove_duplicates(in_place=True)

    # What the block carries, at the precision the file holds it -- composing
    # against the exact 30 degrees would be asserting the rounding rather than
    # the composition.
    carried = Rotation.from_quat(built.get_block(1).rotation, scalar_first=True)
    turn = Rotation.from_euler("z", 20.0, degrees=True)

    pp.TransformFOV(rotation=turn.as_matrix()).apply_to_sequence(built, in_place=True)

    stored = Rotation.from_quat(built.get_block(1).rotation, scalar_first=True)
    assert stored.approx_equal(turn * carried, atol=1e-12)


def test_composing_leaves_the_blocks_other_extensions_in_place(system, readout):
    gx, adc = readout
    built = pp.Sequence(system=system)
    built.add_block(
        gx, adc, _flag("LIN", 7), pp.make_trigger(channel="physio1", duration=1e-3)
    )

    pp.TransformFOV(rotation=np.eye(3)).apply_to_sequence(built, in_place=True)

    block = built.get_block(1)
    assert block.label_sets == [("LIN", 7)]
    assert [trigger.channel for trigger in block.triggers] == ["physio1"]
    assert block.rotation is not None


def test_a_rotation_registers_no_shape(seq):
    """It is an annotation, not a redesign: no waveform is built for it."""
    shapes_before = seq._native.num_shapes()
    pp.TransformFOV(
        rotation=Rotation.from_euler("y", 45.0, degrees=True).as_matrix()
    ).apply_to_sequence(seq, in_place=True)
    assert seq._native.num_shapes() == shapes_before


# %% the labels that exempt a block


@pytest.mark.parametrize(
    ("label", "kwargs", "read"),
    [
        ("NOPOS", {"translation": (10.0, 0.0, 0.0)}, _phases),
        ("NOSCL", {"scale": (0.5, 1.0, 1.0)}, _amplitudes),
    ],
)
def test_a_flag_exempts_the_block_that_sets_it_and_every_block_after(
    system, readout, label, kwargs, read
):
    gx, adc = readout
    built = pp.Sequence(system=system)
    built.add_block(gx, adc)
    built.add_block(gx, adc, _flag(label))
    built.add_block(gx, adc)
    built.remove_duplicates(in_place=True)

    before = read(built)
    pp.TransformFOV(**kwargs).apply_to_sequence(built, in_place=True)
    after = read(built)

    assert after[0] != before[0]  # ahead of the flag
    assert after[1] == before[1]  # the block that set it
    assert after[2] == before[2]  # and it stays set


def test_norot_exempts_the_block_that_sets_it(system, readout):
    gx, adc = readout
    built = pp.Sequence(system=system)
    built.add_block(gx, adc)
    built.add_block(gx, adc, _flag("NOROT"))

    pp.TransformFOV(
        rotation=Rotation.from_euler("z", 30.0, degrees=True).as_matrix()
    ).apply_to_sequence(built, in_place=True)

    assert built.get_block(1).rotation is not None
    assert built.get_block(2).rotation is None


def test_clearing_a_flag_lets_the_transformation_resume(system, readout):
    gx, adc = readout
    built = pp.Sequence(system=system)
    built.add_block(gx, adc, _flag("NOPOS"))
    built.add_block(gx, adc)
    built.add_block(gx, adc, _flag("NOPOS", 0))

    before = _phases(built)
    pp.TransformFOV(translation=(10.0, 0.0, 0.0)).apply_to_sequence(
        built, in_place=True
    )
    after = _phases(built)

    assert (after[0], after[1]) == (before[0], before[1])
    assert after[2] != before[2]


def test_the_flags_start_clear_for_every_call(system, readout):
    """State is per call, matching MATLAB's reset at the top of applyToSeq."""
    gx, adc = readout
    built = pp.Sequence(system=system)
    built.add_block(gx, adc, _flag("NOPOS"))
    built.add_block(gx, adc)

    shift = pp.TransformFOV(translation=(10.0, 0.0, 0.0))
    once = shift.apply_to_sequence(built)
    twice = shift.apply_to_sequence(built)

    assert once.get_block(1).adc.phase_offset == twice.get_block(1).adc.phase_offset


# %% copying


def test_the_source_is_untouched_unless_asked_for_in_place(seq):
    before = _phases(seq)
    moved = pp.TransformFOV(translation=(10.0, 0.0, 0.0)).apply_to_sequence(seq)

    assert _phases(seq) == before
    assert _phases(moved) != before
    assert moved is not seq


def test_in_place_hands_back_the_same_object(seq):
    assert (
        pp.TransformFOV(translation=(10.0, 0.0, 0.0)).apply_to_sequence(
            seq, in_place=True
        )
        is seq
    )


def test_a_transformed_copy_still_writes(seq, tmp_path):
    moved = pp.TransformFOV(
        translation=(10.0, 0.0, 0.0), scale=(0.5, 1.0, 1.0)
    ).apply_to_sequence(seq)
    moved.remove_duplicates(in_place=True)
    path = tmp_path / "moved.seq"
    moved.write(path)

    back = pp.Sequence(system=seq.system)
    back.read(path)
    assert back.num_blocks == moved.num_blocks


# %% the homogeneous form, and the Sequence shorthand


def test_a_homogeneous_transform_is_its_rotation_and_translation():
    turn = Rotation.from_euler("z", 30.0, degrees=True)
    offset_world = np.array([10.0, 0.0, 0.0])

    matrix = np.eye(4)
    matrix[:3, :3] = turn.as_matrix()
    matrix[:3, 3] = offset_world

    combined = pp.TransformFOV(transform=matrix)
    # The homogeneous form states its offset in the rotated frame.
    np.testing.assert_allclose(
        combined.translation, turn.as_matrix().T @ offset_world, atol=1e-12
    )
    np.testing.assert_allclose(combined.rotation, turn.as_matrix(), atol=1e-12)


@pytest.mark.parametrize("server_mode", [False, True])
def test_applying_in_place_matches_the_copy_it_would_have_returned(seq, server_mode):
    expected = pp.TransformFOV(
        translation=(0.0, 0.0, 10.0), server_mode=server_mode
    ).apply_to_sequence(seq)

    pp.TransformFOV(
        translation=(0.0, 0.0, 10.0), server_mode=server_mode
    ).apply_to_sequence(seq, in_place=True)

    assert _phases(seq) == pytest.approx(_phases(expected))
    assert seq._native.has_base_trajectory() == expected._native.has_base_trajectory()


def test_transforming_in_place_invalidates_what_was_derived_from_the_waveforms(seq):
    """The scan structure is cached until the sequence changes underneath it.

    A transform changes it -- amplitudes, phases, sometimes a rotation
    extension -- so a safety view taken afterwards must not be the one
    computed before.
    """
    before = seq._revision
    pp.TransformFOV(translation=(0.0, 0.0, 10.0)).apply_to_sequence(seq, in_place=True)
    assert seq._revision > before


# %% nothing to do


def test_a_zero_shift_leaves_every_block_exactly_as_it_was(seq):
    """The ordinary case for a prescription, so it has to cost nothing."""
    before = (_phases(seq), _amplitudes(seq), seq._native.num_shapes())

    pp.TransformFOV(translation=(0.0, 0.0, 0.0)).apply_to_sequence(seq, in_place=True)

    assert (_phases(seq), _amplitudes(seq), seq._native.num_shapes()) == before


def test_a_unit_scale_leaves_every_block_exactly_as_it_was(seq):
    before = _amplitudes(seq)

    pp.TransformFOV(scale=(1.0, 1.0, 1.0)).apply_to_sequence(seq, in_place=True)

    assert _amplitudes(seq) == before


def test_a_zero_shift_beside_a_real_scale_still_scales(seq):
    before = _amplitudes(seq)[0]
    pp.TransformFOV(
        translation=(0.0, 0.0, 0.0), scale=(2.0, 1.0, 1.0)
    ).apply_to_sequence(seq, in_place=True)

    assert _amplitudes(seq)[0] == pytest.approx(2.0 * before)


def test_an_identity_rotation_is_still_attached(seq):
    """Not the same kind of nothing: the extension is part of a block's identity."""
    pp.TransformFOV(rotation=np.eye(3)).apply_to_sequence(seq, in_place=True)

    assert seq.get_block(1).rotation is not None
