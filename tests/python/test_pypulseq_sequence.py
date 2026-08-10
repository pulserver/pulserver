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
        "num_trs",
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
    """They belong to the layer above; this class does not stand in for it."""
    assert not hasattr(pp.Sequence, name)


#: Every method that carries upstream's signature but not its implementation
#: yet, with arguments named the way upstream names them -- so the call itself
#: proves the signature matches, and the exception proves it is still a stub.
_NOT_PORTED = [
    ("calculate_kspace", (), {"trajectory_delay": 0.0, "gradient_offset": 0.0}),
    ("waveforms", (), {"append_RF": True, "time_range": [0.0, 1.0]}),
    ("waveforms_and_times", (), {"append_RF": False, "time_range": None}),
    ("check_timing", (), {"print_errors": True}),
    ("test_report", (), {}),
    ("evaluate_labels", (), {"init": None, "evolution": "adc"}),
    ("apply_soft_delay", (), {"TE": 40e-3}),
    ("flip_grad_axis", ("x",), {}),
    ("mod_grad_axis", (), {"axis": "y", "modifier": 0.5}),
    ("find_block_by_time", (), {"t": 0.01}),
]


@pytest.mark.parametrize(("name", "args", "kwargs"), _NOT_PORTED, ids=[n for n, _, _ in _NOT_PORTED])
def test_the_unported_methods_take_upstreams_arguments_and_say_they_are_stubs(
    seq, name, args, kwargs
):
    with pytest.raises(NotImplementedError, match="no implementation"):
        getattr(seq, name)(*args, **kwargs)


#: Methods that *are* implemented, and still have to answer to upstream's
#: signature -- a PyPulseq script has to run here unchanged whether or not the
#: body underneath it is ours.
_PORTED = ["plot", "calculate_pns", "calculate_gradient_spectrum"]


@pytest.mark.parametrize("name", [name for name, _, _ in _NOT_PORTED] + _PORTED)
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
