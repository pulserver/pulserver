"""The public Sequence over the C++ pulseq library: build, decode, window, files."""

from __future__ import annotations

import inspect
import io

import matplotlib
import numpy as np
import pulserver.pypulseq as pp
import pypulseq as upstream
import pytest
from pulserver.pypulseq._extensions import strip_extensions
from scipy.spatial.transform import Rotation

matplotlib.use("Agg")


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
def events(system):
    """One repetition's worth of events, covering every kind that survives a file."""
    rf, gz, gzr = pp.make_sinc_pulse(
        flip_angle=np.pi / 6,
        duration=3e-3,
        slice_thickness=5e-3,
        apodization=0.5,
        time_bw_product=4,
        system=system,
        return_gz=True,
        use="excitation",
    )
    return {
        "rf": rf,
        "gz": gz,
        "gzr": gzr,
        "gx": pp.make_trapezoid(channel="x", flat_area=581.8, flat_time=3.2e-3, system=system),
        "gy": pp.make_arbitrary_grad(
            channel="y", waveform=np.sin(np.linspace(0, np.pi, 200)) * 1e5, system=system
        ),
        "adc": pp.make_adc(num_samples=128, dwell=2.4e-5, delay=1e-4, system=system),
        "trigger": pp.make_trigger(channel="physio1", duration=1e-3, system=system),
    }


@pytest.fixture
def seq(system, events):
    """Four repetitions, each rotated a little further and labelled with its line."""
    built = pp.Sequence(system=system)
    for line in range(4):
        events["rf"].phase_offset = 0.5 * line
        built.add_block(events["rf"], events["gz"])
        built.add_block(events["gzr"])
        built.add_block(
            events["gx"],
            events["gy"],
            events["adc"],
            pp.make_rotation(Rotation.from_euler("z", 30.0 * line, degrees=True)),
            *([events["trigger"]] if line == 0 else []),
            pp.make_label(type="SET", label="LIN", value=line),
        )
        built.add_block(pp.make_delay(5e-3))
    return built


def test_a_sequence_counts_its_blocks_and_their_duration(seq):
    assert seq.num_blocks == len(seq) == 16
    duration, blocks, per_column = seq.duration()
    assert blocks == 16
    assert duration == pytest.approx(float(np.sum(seq.block_durations)))
    assert per_column.tolist() == [4, 4, 4, 8, 4, 4]  # rf, gx, gy, gz, adc, ext


def test_a_decoded_block_holds_the_events_that_went_into_it(seq, events):
    block = seq.get_block(3)
    assert block.rf is None and block.adc.num_samples == 128
    assert block.gx.type == "trap" and block.gy.type == "grad"
    assert block.label_sets == [("LIN", 0)]
    assert [trigger.channel for trigger in block.triggers] == ["physio1"]
    np.testing.assert_allclose(block.gy.waveform, events["gy"].waveform, rtol=1e-12)
    np.testing.assert_allclose(block.gx.amplitude, events["gx"].amplitude, rtol=1e-12)


def test_deduplication_leaves_the_columns_at_the_precision_the_file_writes(seq, events):
    seq.remove_duplicates()
    # Six significant digits on a gradient amplitude is what `%12g` records, so
    # that -- not double precision -- is what survives a round trip.
    np.testing.assert_allclose(seq.get_block(3).gy.waveform, events["gy"].waveform, rtol=2e-6)
    seq.remove_duplicates()
    assert seq.num_blocks == 16


def test_the_version_is_the_one_the_writers_emit(seq):
    assert (seq.version_major, seq.version_minor, seq.version_revision) == (1, 5, 0)


def test_a_clone_and_its_source_diverge_independently(seq):
    copy = seq._clone()
    assert copy.num_blocks == seq.num_blocks

    seq.add_block(pp.make_delay(1e-3))
    assert copy.num_blocks == 16 and seq.num_blocks == 17

    copy.set_definition("Name", "the copy")
    assert seq.get_definition("Name") is None


def test_a_written_sequence_reads_back_to_the_same_bytes(seq, tmp_path):
    seq.remove_duplicates()
    seq.set_definition("Name", "round_trip")
    seq.set_definition("FOV", [0.22, 0.22, 0.005])

    path = tmp_path / "sequence.seq"
    seq.write(path)
    text = path.read_bytes()

    back = pp.Sequence(system=seq.system)
    back.read(path)
    assert back.num_blocks == seq.num_blocks
    assert back.get_definition("Name") == "round_trip"

    again = tmp_path / "again.seq"
    back.write(again)
    assert again.read_bytes() == text


def test_write_is_text_whatever_the_file_is_called(seq, tmp_path):
    """The reference toolbox's write writes .seq text; the suffix decides nothing."""
    seq.remove_duplicates()
    path = tmp_path / "misleading.bin"
    seq.write(path)
    assert path.read_bytes().lstrip().startswith(b"# Pulseq sequence file")


def test_the_binary_format_carries_the_same_sequence(seq, tmp_path):
    seq.remove_duplicates()
    path = tmp_path / "sequence.bin"
    seq.write_binary(path)

    back = pp.Sequence(system=seq.system)
    back.read(path)
    assert back.num_blocks == seq.num_blocks
    np.testing.assert_allclose(back.block_durations, seq.block_durations, rtol=1e-12)


def test_only_the_binary_writer_takes_a_stream(seq, tmp_path):
    """It is the format meant to be handed on without touching the filesystem."""
    seq.remove_duplicates()
    path = tmp_path / "sequence.bin"
    seq.write_binary(path)

    stream = io.BytesIO()
    seq.write_binary(stream)
    assert stream.getvalue() == path.read_bytes()

    with pytest.raises(TypeError):
        seq.write(io.BytesIO())


def test_upstream_pypulseq_reads_what_was_written(seq, tmp_path):
    # Once ROTATIONS is stripped: upstream 1.5.0 has no reader for it.
    seq.remove_duplicates()
    written = tmp_path / "written.seq"
    seq.write(written)
    path = tmp_path / "plain.seq"
    path.write_bytes(strip_extensions(written.read_bytes()))
    native = upstream.Sequence(system=seq.system)
    native.read(str(path))
    assert len(native.block_events) == seq.num_blocks


def test_set_block_replaces_a_block_in_place(seq):
    seq.set_block(4, pp.make_delay(9e-3))
    assert seq.get_block(4).block_duration == pytest.approx(9e-3)
    assert seq.num_blocks == 16


def test_a_slotted_and_a_namespace_event_describe_the_same_block(system):
    slotted = pp.Sequence(system=system)
    plain = pp.Sequence(system=system)
    slotted.add_block(pp.make_trapezoid(channel="x", area=1000, system=system))
    plain.add_block(upstream.make_trapezoid(channel="x", area=1000, system=system))
    assert slotted._to_text(create_signature=False) == plain._to_text(create_signature=False)


def test_scaling_an_rf_amplitude_costs_no_extra_shape(system):
    """A variable flip angle train is one set of shapes at many amplitudes."""

    def shapes_for(amplitudes):
        built = pp.Sequence(system=system)
        rf = pp.make_block_pulse(flip_angle=np.pi / 2, duration=1e-3, system=system)
        for amplitude in amplitudes:
            rf.amplitude = amplitude
            built.add_block(rf)
        built.remove_duplicates()
        return built._native.num_shapes(), built._native.num_rf()

    one_shape_count, one_row = shapes_for([1e3])
    many_shape_count, many_rows = shapes_for([1e3, 5e2, 2.5e2])
    assert many_shape_count == one_shape_count
    assert (one_row, many_rows) == (1, 3)


@pytest.mark.parametrize(
    "name",
    [
        "payload",
        "tr_info",
        "tr_duration",
        "tr_block_range",
        "segments",
        "segment",
        "pns",
        "mech_resonances",
        "grad_spectrum",
        "plot_kspace",
        "block_range_of",
        "_collection",
    ],
)
def test_the_scan_structure_placeholders_are_gone_rather_than_stubbed(name):
    """They belong to the layer above; this class does not stand in for it.

    ``num_trs``, ``tr_size`` and ``num_segments`` were once on this list and
    are deliberately off it: they answer about the structure the C core
    *recovered*, which is this class's own knowledge, where the entries above
    stood in for a scan-definition layer that lives elsewhere. The rest of
    that layer's vocabulary stays gone.
    """
    assert not hasattr(pp.Sequence, name)


#: Methods that raise on purpose, and the reason each states. A stub here is a
#: decision, not a gap, and the message has to say which -- so the test matches
#: on the reason and not merely on the exception type.
_DELIBERATE_STUBS = [
    ("install", (), {}, "Siemens scanner transfer"),
    ("check_timing", (), {"print_errors": True}, "predownload"),
    ("sound", (), {}, "acoustic preview"),
]


@pytest.mark.parametrize(
    ("name", "args", "kwargs", "reason"),
    _DELIBERATE_STUBS,
    ids=[n for n, _, _, _ in _DELIBERATE_STUBS],
)
def test_the_deliberate_stubs_refuse_and_say_why(seq, name, args, kwargs, reason):
    with pytest.raises(NotImplementedError, match=reason):
        getattr(seq, name)(*args, **kwargs)


#: Methods that *are* implemented, and still have to answer to upstream's
#: signature -- a PyPulseq script has to run here unchanged whether or not the
#: body underneath it is ours.
_PORTED = [
    "plot",
    "calculate_pns",
    "calculate_gradient_spectrum",
    "calculate_kspace",
    "waveforms",
    "waveforms_and_times",
    "rf_times",
    "adc_times",
    "get_gradients",
    "paper_plot",
    "install",
    "get_extension_type_ID",
    "get_extension_type_string",
    "set_extension_string_ID",
    "get_raw_block_content_IDs",
    "rf_from_lib_data",
    "flip_grad_axis",
    "mod_grad_axis",
    "find_block_by_time",
    "check_timing",
    "test_report",
    "evaluate_labels",
    "apply_soft_delay",
]


def test_every_public_upstream_method_exists_here():
    """Absent is worse than unimplemented: it is an AttributeError at the call
    site instead of a message saying what is missing and why."""
    missing = [
        name
        for name in dir(upstream.Sequence)
        if not name.startswith("_")
        and callable(getattr(upstream.Sequence, name, None))
        and not hasattr(pp.Sequence, name)
    ]
    assert missing == []


@pytest.mark.parametrize("name", _PORTED)
def test_the_methods_take_upstreams_arguments_the_way_upstream_takes_them(name):
    """Upstream's parameters, in order, with upstream's defaults, and first.

    A method may take *more* than upstream's -- ``calculate_pns`` grows a
    ``tr`` -- but only keyword-only, only after every one of upstream's, and
    without disturbing a single name or default before it. That is what makes
    an unchanged PyPulseq call reach an unchanged PyPulseq meaning, whichever
    way it was written: positionally or by keyword.
    """
    ours = list(inspect.signature(getattr(pp.Sequence, name)).parameters.values())
    theirs = list(inspect.signature(getattr(upstream.Sequence, name)).parameters.values())

    assert [(p.name, p.default) for p in ours[: len(theirs)]] == [
        (p.name, p.default) for p in theirs
    ]
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in ours[len(theirs) :])
    # An added parameter has to be optional, or upstream's own call breaks.
    assert all(p.default is not inspect.Parameter.empty for p in ours[len(theirs) :])


# %% upstream's register_*_event protocol


def test_the_upstream_registration_idiom_runs_unchanged(system, events):
    """`gx.id = seq.register_grad_event(gx)` -- writeGradientEcho3D.m's opening."""
    built = pp.Sequence(system=system)
    gx, gy, rf, adc = events["gx"], events["gy"], events["rf"], events["adc"]

    gx.id = built.register_grad_event(gx)
    gy.id = built.register_grad_event(gy)
    _, rf.shape_IDs = built.register_rf_event(rf)
    adc.id = built.register_adc_event(adc)

    built.add_block(gx, gy, adc)
    assert built.num_blocks == 1


@pytest.mark.parametrize("name", ["id", "shape_IDs"])
def test_the_registration_attributes_are_absent_until_assigned(events, name):
    """`hasattr` must answer False: upstream's own registration branches on it,
    and would otherwise trust shape ids belonging to another sequence."""
    event = events["gx"]
    assert not hasattr(event, name)

    setattr(event, name, [3, 4])
    assert getattr(event, name) == [3, 4]  # and reads back what was set

    delattr(event, name)
    assert not hasattr(event, name)


def test_pre_registering_registers_the_shape_exactly_once(system, events):
    """The point: the block loop must not register the waveform a second time."""
    built = pp.Sequence(system=system)
    gy = events["gy"]

    _, shapes = built.register_grad_event(gy)
    after_registering = built._native.num_shapes()
    assert shapes[0] > 0  # a real shape id, not a placeholder

    for _ in range(4):
        built.add_block(gy)
    assert built._native.num_shapes() == after_registering


def test_pre_registering_does_not_change_the_sequence(system, events):
    """Ids are issued in a different order; nothing a block plays moves."""

    def build(preregister):
        built = pp.Sequence(system=system)
        gx, gy, rf, adc = events["gx"], events["gy"], events["rf"], events["adc"]
        if preregister:
            gx.id = built.register_grad_event(gx)
            _, rf.shape_IDs = built.register_rf_event(rf)
        for _ in range(3):
            built.add_block(rf)
            built.add_block(gx, gy, adc)
        built.remove_duplicates()
        return built

    plain, pre = build(False), build(True)
    assert (plain.num_blocks, plain._native.num_shapes()) == (
        pre.num_blocks,
        pre._native.num_shapes(),
    )
    for index in range(1, plain.num_blocks + 1):
        one, other = plain.get_block(index), pre.get_block(index)
        assert one.block_duration == other.block_duration
        np.testing.assert_allclose(one.gy.waveform if one.gy else 0, other.gy.waveform if other.gy else 0)
        assert (one.rf is None) == (other.rf is None)
        if one.rf is not None:
            np.testing.assert_allclose(one.rf.signal, other.rf.signal, rtol=1e-12)


def test_a_trapezoid_and_a_label_have_no_shape_to_register(system, events):
    built = pp.Sequence(system=system)
    assert built.register_grad_event(events["gx"]) == 0  # trapezoid: bare id
    assert built.register_label_event(pp.make_label(type="SET", label="LIN", value=0)) == 0
    assert built._native.num_shapes() == 0


def test_an_upstream_namespace_event_can_be_pre_registered_too(system):
    """The other event flavour: memoized on the identity of its array."""
    built = pp.Sequence(system=system)
    plain = upstream.make_arbitrary_grad(
        channel="y", waveform=np.sin(np.linspace(0, np.pi, 64)) * 1e5, system=system
    )

    _, shapes = built.register_grad_event(plain)
    assert shapes[0] > 0
    registered = built._native.num_shapes()

    built.add_block(plain)
    assert built._native.num_shapes() == registered


def test_calculate_kspace_pp_is_deprecated_outright(seq):
    """Upstream warns and forwards; here the deprecated spelling is an error."""
    with pytest.raises(DeprecationWarning, match="use calculate_kspace instead"):
        seq.calculate_kspacePP()


# %% extension registration (MATLAB's registerRotationEvent / registerRfShimEvent)


def test_registering_a_rotation_or_shim_twice_returns_the_same_id(system):
    """Unlike ``add_block``, which appends per block and lets dedup collapse
    them, registration looks for an equal row first -- so an orientation has
    one stable name."""
    seq = pp.Sequence(system=system)
    turn = pp.make_rotation(Rotation.from_euler("z", 30, degrees=True))
    other = pp.make_rotation(Rotation.from_euler("z", 60, degrees=True))
    shim = pp.make_rf_shim(np.array([1 + 0j, 0.5 * np.exp(1j * 0.7)]))

    assert seq.register_rotation_event(turn) == seq.register_rotation_event(turn)
    assert seq.register_rotation_event(other) != seq.register_rotation_event(turn)
    assert seq._native.num_rotations() == 2

    assert seq.register_rf_shim_event(shim) == seq.register_rf_shim_event(shim)
    assert seq._native.num_rf_shims() == 1


def test_pre_registering_an_extension_changes_nothing_in_the_written_file(system, events):
    """The registration idiom has to be free of consequence, or it is a trap."""
    turn = pp.make_rotation(Rotation.from_euler("z", 30, degrees=True))
    shim = pp.make_rf_shim(np.array([1 + 0j, 0.5 * np.exp(1j * 0.7)]))

    def build(*, preregister):
        built = pp.Sequence(system=system)
        if preregister:
            built.register_rotation_event(turn)
            built.register_rf_shim_event(shim)
            built.register_control_event(events["trigger"])
        built.add_block(events["gx"], turn)
        built.add_block(events["rf"], shim)
        built.remove_duplicates()
        return built

    assert build(preregister=True)._to_text(create_signature=False) == build(
        preregister=False
    )._to_text(create_signature=False)


def test_a_rotation_must_be_a_unit_quaternion(system):
    seq = pp.Sequence(system=system)
    with pytest.raises(ValueError, match="unit quaternion"):
        seq.register_rotation_event(pp.make_rotation(np.array([1.0, 1.0, 1.0, 1.0])))
    with pytest.raises(ValueError, match="four numbers"):
        seq.register_rotation_event(pp.make_rotation(np.array([1.0, 0.0])))


# %% gradient axis scaling, on the TransformFOV path


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_mod_grad_axis_scales_only_that_axis(seq, axis):
    before = [seq.get_block(n) for n in range(1, seq.num_blocks + 1)]
    seq.mod_grad_axis(axis, 0.5)
    after = [seq.get_block(n) for n in range(1, seq.num_blocks + 1)]

    for old, new in zip(before, after):
        for name in ("gx", "gy", "gz"):
            was, now = getattr(old, name), getattr(new, name)
            if was is None:
                assert now is None
                continue
            factor = 0.5 if name[-1] == axis else 1.0
            if was.type == "trap":
                assert now.amplitude == pytest.approx(was.amplitude * factor)
            else:
                np.testing.assert_allclose(now.waveform, was.waveform * factor, rtol=1e-9)


def test_flip_grad_axis_is_a_negative_scale(seq):
    before = seq.get_block(3).gx.amplitude
    seq.flip_grad_axis("x")
    assert seq.get_block(3).gx.amplitude == pytest.approx(-before)


def test_mod_grad_axis_refuses_an_axis_that_is_not_one(seq):
    with pytest.raises(ValueError, match="must be 'x', 'y' or 'z'"):
        seq.mod_grad_axis("w", 2.0)


# %% the rest of the batch


def test_find_block_by_time_finds_the_block_being_played(seq):
    """Including the first block, where upstream's own version raises."""
    edges = np.concatenate(([0.0], np.cumsum(seq.block_durations)))
    for number in range(1, seq.num_blocks + 1):
        middle = 0.5 * (edges[number - 1] + edges[number])
        assert seq.find_block_by_time(middle) == number
    assert seq.find_block_by_time(edges[-1] + 1.0) is None
    assert seq.find_block_by_time(-1.0) is None
    # A block boundary belongs to the block that starts there.
    assert seq.find_block_by_time(edges[1]) == 2


def test_find_block_by_time_does_not_reproduce_upstreams_off_by_one(seq):
    """Upstream indexes a one-based dict with a zero-based searchsorted, so it
    is off by one and raises KeyError on the first block. Pinned so nobody
    "restores parity" by reintroducing the bug."""
    native = upstream.Sequence(system=seq.system)
    for _ in range(4):
        native.add_block(upstream.make_delay(1e-3))
    with pytest.raises(KeyError):
        native.find_block_by_time(0.0005)


def test_copy_definitions_takes_another_sequences_definitions(seq, system):
    seq.set_definition("Name", "source")
    seq.set_definition("FOV", [0.22, 0.22, 0.005])

    other = pp.Sequence(system=system)
    other.copy_definitions(seq)
    assert other.get_definition("Name") == "source"
    assert other.get_definition("FOV") == [0.22, 0.22, 0.005]

    # A copy, not a view.
    seq.set_definition("Name", "changed")
    assert other.get_definition("Name") == "source"


def test_sound_refuses_and_points_at_the_measured_answer(seq):
    with pytest.raises(NotImplementedError, match="calculate_gradient_spectrum"):
        seq.sound()
