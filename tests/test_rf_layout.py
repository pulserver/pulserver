"""The RF layout an evaluation states, and the blocks that carry it to a scanner."""

import logging
import math

import numpy as np
import pypulseqpp as pp
import pytest
from _host import LIMITS, PLUGINS, generate, value_block

from pulserver.design import Evaluation, FloatParam, RfLayout, SequencePlugin
from pulserver.host import DesignStore, call
from pulserver.protocol import (
    ConfigKey,
    UIParam,
    UserKey,
    format_rf_definitions,
    format_rf_layout,
    parse_listing,
    parse_rf_definitions,
    parse_rf_layout,
    parse_validation,
)

SYSTEM = pp.Opts(max_grad=40.0, grad_unit="mT/m", max_slew=150.0, slew_unit="T/m/s")
ENTRIES = {
    UIParam.FLIP: FloatParam("flip", unit="deg", range_max=180.0),
    UserKey.USER0: FloatParam("refocusing", unit="deg", range_max=180.0),
}
# The controls of the excitation and the two refocusing pulses of _echo_app.
SCALED = [UIParam.FLIP, UserKey.USER0, UserKey.USER0]


def _train(refocusing, excitation=90.0, system=SYSTEM):
    """One 1 ms block pulse per flip angle in degrees, an excitation first, 4 ms apart."""
    seq = pp.Sequence(system)
    plays = [(flip, "refocusing") for flip in refocusing]
    if excitation is not None:
        plays.insert(0, (excitation, "excitation"))
    for flip, use in plays:
        seq.add_block(
            pp.make_block_pulse(np.deg2rad(flip), duration=1e-3, system=system, use=use)
        )
        seq.add_block(pp.make_delay(4e-3))
    return seq


def _controls(echoes):
    """The excitation follows the flip angle, every refocusing pulse the first user entry."""
    return [UIParam.FLIP, *[UserKey.USER0] * echoes]


def _gre(lines, tr=10e-3):
    """RF-spoiled repetitions of a 1 ms excitation, its phase offset stepping quadratically."""
    seq = pp.Sequence(SYSTEM)
    for line in range(lines):
        phase = np.deg2rad(117.0 * line * (line + 1) / 2)
        seq.add_block(
            pp.make_block_pulse(
                np.deg2rad(15.0),
                duration=1e-3,
                system=SYSTEM,
                phase_offset=phase,
                use="excitation",
            )
        )
        seq.add_block(pp.make_delay(tr - 1e-3))
    return seq


def _pulse(kind, system=SYSTEM):
    """A sinc pulse of 2 ms, a block pulse of 0.5 ms, or a two-channel complex pTx pulse."""
    if kind == "sinc":
        return pp.make_sinc_pulse(
            np.deg2rad(30.0),
            duration=2e-3,
            time_bw_product=4,
            system=system,
            use="excitation",
        )
    if kind == "block":
        return pp.make_block_pulse(
            np.deg2rad(30.0), duration=0.5e-3, system=system, use="excitation"
        )
    t = np.linspace(-1.0, 1.0, 200)
    gauss = np.exp(-4.0 * t**2)
    channels = np.stack([300.0 * gauss + 0j, 200j * gauss * np.exp(3j * t)])
    return pp.make_ptx_pulse(channels, system=system, use="excitation")


def _runs(layout):
    return [
        line
        for line in format_rf_layout(layout).splitlines()
        if line.startswith("run ")
    ]


def _echo_app(system, flip=90.0, refocusing=150.0):
    return _train([refocusing] * 2, flip, system)


def _plugin(stated, entries=ENTRIES, change=None):
    """A plugin over :func:`_echo_app` whose evaluation returns ``stated(seq)`` as its layout.

    The evaluated protocol is the request with the values of ``change`` replaced.
    """

    def evaluate(self, system, protocol):
        seq = _echo_app(system, **protocol.arguments)
        if change is not None:
            protocol = protocol.replace(change)
        return Evaluation(protocol, seq.duration()[0], rf_layout=stated(seq))

    return type(
        "Stated",
        (SequencePlugin,),
        {"app": _echo_app, "protocol": entries, "evaluate": evaluate},
    )()


def _scaled(controls, **kwargs):
    return _plugin(lambda seq: RfLayout.of(seq, controls), **kwargs)


STATED_PLUGIN = """\
import pypulseqpp as pp

from pulserver.design import Evaluation, FloatParam, RfLayout, SequencePlugin
from pulserver.protocol import UIParam


def app(system, flip=20.0):
    seq = pp.Sequence(system)
    seq.add_block(pp.make_delay(1e-3))
    return seq


class Stated(SequencePlugin):
    app = app
    protocol = {UIParam.FLIP: FloatParam("flip", unit="deg", range_max=180.0)}

    def evaluate(self, system, protocol):
        BODY
"""


def _stated_plugins(tmp_path, body):
    """A directory holding the plugin ``stated``, whose evaluation runs the line ``body``."""
    plugins = tmp_path / "plugins"
    plugins.mkdir(exist_ok=True)
    (plugins / "stated.py").write_text(STATED_PLUGIN.replace("BODY", body))
    return plugins


def _list(rf_definitions=True, plugin="rf_train", plugins=PLUGINS):
    status, reply = call(
        "list",
        plugins=plugins,
        plugin=plugin,
        limits=LIMITS,
        rf_definitions=rf_definitions,
    )
    assert status == 0, reply
    return reply


def _validate(values, rf_layout=True, plugin="rf_train", plugins=PLUGINS):
    status, reply = call(
        "validate",
        plugins=plugins,
        plugin=plugin,
        limits=LIMITS,
        block=value_block(values),
        rf_layout=rf_layout,
    )
    assert status == 0, reply
    return reply


@pytest.mark.parametrize(
    "stated",
    ["()", "[]", "RfLayout.of(app(system), None)"],
    ids=["tuple", "list", "no-rf"],
)
def test_an_empty_layout_sends_no_rf_block(tmp_path, stated):
    plugins = _stated_plugins(
        tmp_path, f"return Evaluation(protocol, 1e-3, rf_layout={stated})"
    )

    listed = _list(plugins=plugins, plugin="stated")
    validated = _validate({}, plugins=plugins, plugin="stated")

    assert listed == _list(False, plugins=plugins, plugin="stated")
    assert validated.startswith("VALID ")
    assert "[RfLayout" not in validated


def test_a_fixed_instance_has_no_control():
    controls = [UIParam.FLIP, None, None]

    layout = RfLayout.of(_train([120.0, 120.0]), controls)

    assert layout.control == (UIParam.FLIP, None, None)
    assert _runs(layout) == ["run 0 1 flip 1", "run 1 1 - 2"]
    assert parse_rf_layout(format_rf_layout(layout)).control == layout.control


def test_one_definition_carries_none_and_two_controls():
    controls = [None, UIParam.FLIP, UserKey.USER0]

    layout = RfLayout.of(_train([150.0] * 3, excitation=None), controls)

    assert len(layout.instances.definitions) == 1
    assert _runs(layout) == [
        "run 0 1 - 1",
        "run 0 1 flip 1",
        "run 0 1 user0_value 1",
    ]
    read = parse_rf_layout(format_rf_layout(layout))
    assert read.definition == (0, 0, 0)
    assert read.control == (None, UIParam.FLIP, UserKey.USER0)


def test_a_gre_layout_holds_one_excitation_whatever_the_matrix():
    layouts = {lines: RfLayout.of(_gre(lines), UIParam.FLIP) for lines in (1, 8, 16)}

    for lines, layout in layouts.items():
        assert len(layout.instances.definitions) == 1
        assert _runs(layout) == [f"run 0 1 flip {lines}"]
        assert layout.period == pytest.approx(lines * 10e-3)
    listed = {format_rf_definitions(layout.instances) for layout in layouts.values()}
    assert len(listed) == 1


def test_a_layout_needs_one_control_per_instance():
    seq = _train([150.0] * 2)

    with pytest.raises(ValueError, match="one control per instance"):
        RfLayout.of(seq, [UIParam.FLIP])
    with pytest.raises(ValueError, match="one control per instance"):
        RfLayout(seq.rf_instances(), (UIParam.FLIP,) * 4, 1.0)


@pytest.mark.parametrize(
    "control",
    [
        UIParam.TE,
        UIParam.RF_SPOILING,
        UIParam.SEQUENCE_TYPE,
        ConfigKey.ENABLE_SAR_BURST_MODE,
        UIParam.user_name(0),
        "flip",
    ],
    ids=["time", "bool", "choice", "config", "user-name", "plain-string"],
)
def test_a_control_must_scale_rf(control):
    with pytest.raises(TypeError, match="flip angle or to a user entry"):
        RfLayout.of(_train([150.0]), control)


@pytest.mark.parametrize(
    "control",
    [UIParam.FLIP, UIParam.user_value(18), None],
    ids=["flip", "user18", "none"],
)
def test_one_control_applies_to_every_instance(control):
    layout = RfLayout.of(_train([150.0] * 2), control)

    assert layout.control == (control,) * 3


def test_the_period_defaults_to_the_repetition_and_can_be_stated():
    seq = _train([150.0] * 2)

    assert RfLayout.of(seq, UIParam.FLIP).period == seq.duration()[0]
    assert RfLayout.of(seq, UIParam.FLIP, period=0.25).period == 0.25


@pytest.mark.parametrize("period", [0.0, -1.0, math.nan])
def test_a_period_must_be_positive(period):
    seq = _train([150.0])

    with pytest.raises(ValueError, match="period"):
        RfLayout.of(seq, UIParam.FLIP, period=period)
    with pytest.raises(ValueError, match="period"):
        RfLayout(seq.rf_instances(), (UIParam.FLIP,) * 2, period)


def test_a_valid_reply_carries_the_layout_of_the_evaluation():
    reply = _scaled(SCALED).validate(SYSTEM, {})

    assert reply.valid
    assert reply.rf_layout.control == tuple(SCALED)
    assert reply.rf_layout.instances.definition.tolist() == [0, 1, 1]


@pytest.mark.parametrize("stated", [(), []], ids=["tuple", "list"])
def test_an_evaluation_stating_no_layout_is_valid_and_carries_none(stated):
    reply = _plugin(lambda seq: stated).validate(SYSTEM, {})

    assert reply.valid
    assert reply.rf_layout is None


def test_an_undeclared_control_is_invalid(caplog):
    plugin = _scaled(SCALED, entries={UIParam.FLIP: ENTRIES[UIParam.FLIP]})

    with caplog.at_level(logging.WARNING, logger="pulserver.design"):
        reply = plugin.validate(SYSTEM, {})

    assert not reply.valid
    assert reply.rf_layout is None
    assert "user0_value" in reply.info
    assert [record.levelno for record in caplog.records] == [logging.WARNING]


@pytest.mark.parametrize("value", [0.0, -5.0])
def test_a_non_positive_control_is_invalid(value):
    reply = _scaled(SCALED, change={UIParam.FLIP: value}).validate(SYSTEM, {})

    assert not reply.valid
    assert reply.rf_layout is None
    assert "flip" in reply.info
    assert "not positive" in reply.info


def test_a_stated_layout_that_is_not_an_rf_layout_is_an_error_of_the_plugin(caplog):
    plugin = _plugin(lambda seq: seq)

    with caplog.at_level(logging.ERROR, logger="pulserver.design"):
        reply = plugin.validate(SYSTEM, {})

    assert (reply.valid, reply.info) == (False, "TypeError in Stated.evaluate")


def test_the_rf_layout_round_trips_the_instance_table_in_order():
    flips = [180.0, 120.0, 120.0, 180.0, 90.0]
    layout = RfLayout.of(_train(flips), _controls(len(flips)))

    read = parse_rf_layout(format_rf_layout(layout))

    assert read.period == pytest.approx(layout.period)
    assert read.definition == tuple(layout.instances.definition.tolist())
    assert read.amplitude == pytest.approx(tuple(layout.instances.amplitude), rel=1e-8)
    assert read.control == layout.control
    assert _runs(layout) == [
        "run 0 1 flip 1",
        "run 1 1 user0_value 1",
        "run 1 0.666666667 user0_value 2",
        "run 1 1 user0_value 1",
        "run 1 0.5 user0_value 1",
    ]


def test_a_constant_train_is_one_run():
    layout = RfLayout.of(_train([150.0] * 64), _controls(64))

    assert _runs(layout) == ["run 0 1 flip 1", "run 1 1 user0_value 64"]
    assert len(parse_rf_layout(format_rf_layout(layout)).definition) == 65


def test_amplitudes_that_print_alike_are_one_run():
    flips = [150.0 * (1.0 + k * 1e-12) for k in range(8)]

    layout = RfLayout.of(_train(flips), _controls(8))

    assert _runs(layout) == ["run 0 1 flip 1", "run 1 1 user0_value 8"]


def test_the_rf_layout_carries_no_samples():
    seq = pp.Sequence(SYSTEM)
    for _ in range(4):
        seq.add_block(_pulse("sinc"))
        seq.add_block(pp.make_delay(5e-3))

    layout = RfLayout.of(seq, UIParam.FLIP)

    (definition,) = layout.instances.definitions
    assert definition.waveform.shape == (1, 1000)
    assert format_rf_layout(layout).splitlines() == [
        "[RfLayout]",
        "period 0.028",
        "run 0 1 flip 4",
        "[RfLayout End]",
    ]


def test_the_rf_blocks_are_one_list_per_line_of_nine_significant_digits():
    seq = _train([120.0], excitation=None)
    layout = RfLayout.of(seq, UserKey.USER0, period=2 / 3)
    bandwidth = f"{pp.calc_rf_bandwidth(seq.get_block(1).rf):.9g}"

    assert format_rf_layout(layout) == (
        "[RfLayout]\nperiod 0.666666667\nrun 0 1 user0_value 1\n[RfLayout End]\n"
    )
    assert format_rf_definitions(layout.instances) == (
        "[RfDefinitions]\n"
        f"definition 0 refocusing 120 333.333333 {bandwidth} 0 0.0005 0.001 1 2\n"
        "0 0.001\n"
        "1 1\n"
        "0 0\n"
        "[RfDefinitions End]\n"
    )


def test_the_rf_definitions_round_trip_complex_channels():
    seq = pp.Sequence(SYSTEM)
    seq.add_block(_pulse("ptx"))
    layout = RfLayout.of(seq, UIParam.FLIP)
    (written,) = layout.instances.definitions

    (read,) = parse_rf_definitions(format_rf_definitions(layout.instances))

    assert (read.index, read.use) == (0, "excitation")
    assert read.flip_deg == pytest.approx(written.flip_deg, rel=1e-8)
    assert read.peak_hz == pytest.approx(written.peak_hz, rel=1e-8)
    assert (read.delay, read.center) == pytest.approx((written.delay, written.center))
    assert read.waveform.shape == written.waveform.shape == (2, 200)
    assert read.waveform.imag.any()
    np.testing.assert_allclose(read.waveform, written.waveform, rtol=0, atol=1e-8)
    np.testing.assert_allclose(read.time, written.time, rtol=1e-8)


@pytest.mark.parametrize("raster", [1e-6, 2e-6, 10e-6])
@pytest.mark.parametrize("kind", ["sinc", "block", "ptx"])
def test_the_listed_duration_is_the_shape_duration_of_the_event(kind, raster):
    system = pp.Opts(
        max_grad=40.0,
        grad_unit="mT/m",
        max_slew=150.0,
        slew_unit="T/m/s",
        rf_raster_time=raster,
    )
    seq = pp.Sequence(system)
    seq.add_block(_pulse(kind, system))

    (listed,) = parse_rf_definitions(format_rf_definitions(seq.rf_instances()))

    assert listed.duration == pytest.approx(seq.get_block(1).rf.shape_dur, rel=1e-8)


@pytest.mark.parametrize("kind", ["sinc", "block", "ptx"])
def test_the_listed_bandwidth_is_the_one_pypulseqpp_measures_on_the_event(kind):
    seq = pp.Sequence(SYSTEM)
    seq.add_block(_pulse(kind))

    (listed,) = parse_rf_definitions(format_rf_definitions(seq.rf_instances()))

    measured = pp.calc_rf_bandwidth(seq.get_block(1).rf)
    assert listed.bandwidth_hz == pytest.approx(measured, rel=1e-8)


def test_a_definition_played_at_zero_amplitude_lists_no_bandwidth():
    seq = _train([0.0], excitation=None)

    (listed,) = parse_rf_definitions(format_rf_definitions(seq.rf_instances()))

    assert (listed.peak_hz, listed.bandwidth_hz) == (0.0, 0.0)
    assert not listed.waveform.any()


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("[RfLayout]\nrun 0 1 flip 1\n[RfLayout End]\n", "period"),
        ("[RfLayout]\nperiod 1\nrun 0 1 flip 0\n[RfLayout End]\n", "not a line"),
        ("[RfLayout]\nperiod 1\nrun 0 1 flip\n[RfLayout End]\n", "not a line"),
        ("[RfLayout]\nperiod 1\nrun 0 1 flip 1\n", "not closed"),
    ],
    ids=["no-period", "empty-run", "short-run", "unclosed"],
)
def test_a_malformed_rf_layout_is_refused(text, message):
    with pytest.raises(ValueError, match=message):
        parse_rf_layout(text)


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda lines: lines[:-1], "not closed"),
        (lambda lines: [*lines[:4], lines[-1]], "cut short"),
        (
            lambda lines: [lines[0], lines[1][:-1] + "3", *lines[2:]],
            "does not hold 3 samples",
        ),
        (lambda lines: [lines[0], "pulse 0", *lines[2:]], "not a definition line"),
    ],
    ids=["unclosed", "cut-short", "wrong-sample-count", "unknown-line"],
)
def test_a_malformed_rf_definition_block_is_refused(edit, message):
    block = format_rf_definitions(_train([150.0], excitation=None).rf_instances())

    with pytest.raises(ValueError, match=message):
        parse_rf_definitions("\n".join(edit(block.splitlines())) + "\n")


def test_the_rf_layout_is_sent_only_when_asked():
    plain = _validate({"etl": 4}, rf_layout=False)
    asked = _validate({"etl": 4})
    invalid = _validate({"flip": 0})

    assert "[RfLayout" not in plain
    assert asked.startswith(plain)
    assert asked[len(plain) :] == (
        "[RfLayout]\nperiod 0.023\nrun 0 1 flip 1\nrun 1 1 user0_value 4\n[RfLayout End]\n"
    )
    assert invalid.startswith("INVALID\n")
    assert "[RfLayout" not in invalid


def test_the_rf_definitions_are_listed_only_when_asked():
    plain = _list(False)
    asked = _list()

    assert "[RfDefinitions" not in plain
    assert asked.startswith(plain)
    excitation, refocusing = parse_rf_definitions(asked)
    assert (excitation.index, excitation.use) == (0, "excitation")
    assert (refocusing.index, refocusing.use) == (1, "refocusing")
    assert (excitation.flip_deg, refocusing.flip_deg) == pytest.approx((90.0, 150.0))
    assert refocusing.peak_hz == pytest.approx(150.0 / 360.0 / 1e-3)


def test_the_rf_definitions_are_evaluated_under_limits():
    status, reply = call(
        "list", plugins=PLUGINS, plugin="rf_train", rf_definitions=True
    )

    assert status == 1
    assert reply.startswith("ERROR ")
    assert "limits" in reply


def test_a_listing_whose_default_evaluation_is_invalid_is_replied_without_rf_definitions(
    tmp_path, caplog
):
    plugins = _stated_plugins(tmp_path, 'raise ValueError("no default design")')

    with caplog.at_level(logging.WARNING, logger="pulserver.host"):
        asked = _list(plugins=plugins, plugin="stated")

    assert asked == _list(False, plugins=plugins, plugin="stated")
    warned = [r.getMessage() for r in caplog.records if r.name == "pulserver.host"]
    assert len(warned) == 1
    assert "no default design" in warned[0]


def test_a_reader_ignores_the_blocks_it_does_not_read():
    listed, plain_listing = _list(), _list(False)
    validated, plain_validation = _validate({"etl": 4}), _validate({"etl": 4}, False)
    listing = parse_listing(plain_listing)

    assert parse_listing(listed) == listing
    assert parse_validation(validated, listing) == parse_validation(
        plain_validation, listing
    )
    assert parse_rf_layout(listed) is None
    assert parse_rf_definitions(validated) == []


def test_the_rf_layout_is_the_one_the_evaluation_states():
    read = parse_rf_layout(_validate({"etl": 4}))

    assert read.period == pytest.approx(0.023)
    assert read.definition == (0, 1, 1, 1, 1)
    assert read.amplitude == (1.0,) * 5
    assert read.control == (UIParam.FLIP, *[UserKey.USER0] * 4)


def test_a_train_length_change_keeps_the_definition_ids():
    short, long = (
        RfLayout.of(_train([150.0] * echoes), _controls(echoes)) for echoes in (1, 16)
    )

    assert short.instances.definition.tolist() == [0, 1]
    assert long.instances.definition.tolist() == [0] + [1] * 16
    assert format_rf_definitions(short.instances) == format_rf_definitions(
        long.instances
    )


def test_a_flip_schedule_keeps_the_refocusing_id():
    schedule = [180.0, 120.0, 90.0, 90.0, 120.0, 150.0]
    constant = RfLayout.of(_train([150.0] * 6), _controls(6))
    varied = RfLayout.of(_train(schedule), _controls(6))

    assert constant.instances.definition.tolist() == [0] + [1] * 6
    assert varied.instances.definition.tolist() == [0] + [1] * 6
    assert varied.instances.amplitude[1:] == pytest.approx(np.array(schedule) / 180.0)
    assert _runs(varied) == [
        "run 0 1 flip 1",
        "run 1 1 user0_value 1",
        "run 1 0.666666667 user0_value 1",
        "run 1 0.5 user0_value 2",
        "run 1 0.666666667 user0_value 1",
        "run 1 0.833333333 user0_value 1",
    ]


@pytest.mark.parametrize(
    "values",
    [{}, {"etl": 1}, {"etl": 64}, {"flip": 30, "user0_value": 90, "etl": 3}],
    ids=["default", "one-echo", "longest", "other-angles"],
)
def test_every_layout_row_names_a_listed_definition(values):
    listed = {record.index for record in parse_rf_definitions(_list())}

    layout = parse_rf_layout(_validate(values))

    assert listed == {0, 1}
    assert set(layout.definition) <= listed
    assert len(layout.definition) == 1 + values.get("etl", 8)


def test_a_follows_declaration_warns_and_has_no_effect(tmp_path):
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    source = (PLUGINS / "rf_train.py").read_text()
    declared = "class RfTrain(SequencePlugin):\n"
    assert declared in source
    (plugins / "followed.py").write_text(
        source.replace(declared, declared + "    follows = (UIParam.FLIP,)\n")
    )
    plain, followed = (
        DesignStore(tmp_path / "plain"),
        DesignStore(tmp_path / "followed"),
    )

    with pytest.warns(DeprecationWarning, match="follows") as warned:
        design = generate(followed, "followed", {}, plugins=plugins)
    reference = generate(plain, "rf_train", {})

    assert warned[0].filename == str(plugins / "followed.py")
    assert followed.manifest(design)["files"] == plain.manifest(reference)["files"]
    assert _validate({}, plugin="followed", plugins=plugins) == _validate({})


def test_a_design_stores_the_sequence_and_its_protocol_only(tmp_path):
    store = DesignStore(tmp_path / "store")

    design = generate(store, "rf_train", {})

    assert sorted(store.manifest(design)["files"]) == [
        "resolved.protocol",
        "sequence.pseg",
        "sequence.seq",
    ]
