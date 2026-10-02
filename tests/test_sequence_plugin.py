"""The sequence plugin: its hooks, its validation boundary and the design of a request."""

import functools
import logging
import warnings
from pathlib import Path

import pypulseqpp as pp
import pytest
from pypulseqpp import sequences

from pulserver import _plugins
from pulserver.design import (
    Evaluation,
    IntParam,
    Protocol,
    ScannerSequence,
    SequencePlugin,
    TimeParam,
    load_plugin,
)
from pulserver.protocol import UIParam

SYSTEM = pp.Opts(max_grad=40.0, grad_unit="mT/m", max_slew=150.0, slew_unit="T/m/s")
PLUGINS = Path(__file__).parent / "plugins"
ENTRIES = {
    UIParam.TE: TimeParam("te", range_max=80000),
    UIParam.NX: IntParam("nx", range_max=100),
}


class ScannedApp(sequences.SequenceApp):
    """A 1 ms prescan, then three repetitions; every construction is counted."""

    MAX_GRAD = 40.0
    MAX_SLEW = 150.0
    constructions = 0

    def init_sequence(self, tr: float = 10e-3) -> None:
        type(self).constructions += 1
        self.tr = tr

    def prescans(self):
        return {"calibration": self.calibration}

    def calibration(self) -> None:
        self.seq.add_block(pp.make_delay(1e-3))

    def loop(self) -> None:
        for _ in range(3):
            self.kernel()

    def kernel(self) -> None:
        self.seq.add_block(pp.make_delay(self.tr))


@pytest.fixture
def calls():
    """The arguments of every call of the app made by :func:`_delays`."""
    return []


@pytest.fixture
def counted():
    """:class:`ScannedApp`, its construction count reset."""
    ScannedApp.constructions = 0
    return ScannedApp


def _delays(calls):
    def delays(system, te=2.5e-3, nx=3):
        calls.append({"te": te, "nx": nx})
        return _sequence(system, te, nx)

    return delays


def _sequence(system, duration, count):
    seq = pp.Sequence(system)
    for _ in range(count):
        seq.add_block(pp.make_delay(duration))
    return seq


def _plugin(app, **attributes):
    """A plugin over ``app`` with the entries of :data:`ENTRIES` and ``attributes``."""
    attributes = {"app": app, "protocol": ENTRIES, **attributes}
    return type("Delays", (SequencePlugin,), attributes)()


def _durations(paths):
    return [pp.io.read(path).duration()[0] for path in paths]


def _declared(reply):
    """The wire values of the entries :data:`ENTRIES` declares."""
    return {key: reply.values[key] for key in ENTRIES}


def test_a_function_app_lists_the_defaults_of_its_signature(calls):
    listing = _plugin(_delays(calls)).listing()

    assert (listing[UIParam.TE].value, listing[UIParam.NX].value) == (2500, 3)
    assert calls == []


def test_a_partial_app_lists_the_defaults_of_the_partial(calls):
    listing = _plugin(functools.partial(_delays(calls), te=4e-3, nx=5)).listing()

    assert (listing[UIParam.TE].value, listing[UIParam.NX].value) == (4000, 5)


def test_a_function_app_without_a_protocol_is_designed_at_its_defaults(calls, tmp_path):
    plugin = _plugin(_delays(calls), protocol={})

    validation, paths = plugin.design(SYSTEM, {}, tmp_path)

    assert validation.valid
    assert calls == [{"te": 2.5e-3, "nx": 3}]
    assert _durations(paths) == [pytest.approx(3 * 2.5e-3)]


def test_the_default_evaluation_of_a_function_app_never_calls_the_app(calls):
    plugin = _plugin(_delays(calls))

    reply = plugin.validate(SYSTEM, {UIParam.NX: 5})

    assert (reply.valid, reply.duration, reply.info) == (True, None, "")
    assert _declared(reply) == {UIParam.TE: 2500, UIParam.NX: 5}
    assert calls == []


def test_an_evaluation_returning_none_is_the_protocol_unchanged(calls):
    def evaluate(self, system, protocol):
        return None

    reply = _plugin(_delays(calls), evaluate=evaluate).validate(SYSTEM, {UIParam.NX: 5})

    assert (reply.valid, reply.duration, reply.info) == (True, None, "")
    assert _declared(reply) == {UIParam.TE: 2500, UIParam.NX: 5}


def test_a_valid_reply_carries_the_protocol_duration_and_note_of_the_evaluation(calls):
    def evaluate(self, system, protocol):
        achieved = protocol.replace({UIParam.TE: 3e-3})
        return Evaluation(achieved, duration=0.0123, info="TE raised to 3 ms")

    reply = _plugin(_delays(calls), evaluate=evaluate).validate(
        SYSTEM, {UIParam.TE: 2000}
    )

    assert (reply.valid, reply.duration, reply.info) == (
        True,
        0.0123,
        "TE raised to 3 ms",
    )
    assert _declared(reply) == {UIParam.TE: 3000, UIParam.NX: 3}


@pytest.mark.parametrize(
    ("error", "info"),
    [(ValueError("no such design"), "no such design"), (ValueError(), "ValueError")],
)
def test_an_invalid_reply_carries_the_request_and_the_message_of_the_error(
    calls, error, info
):
    def evaluate(self, system, protocol):
        raise error

    reply = _plugin(_delays(calls), evaluate=evaluate).validate(SYSTEM, {UIParam.NX: 7})

    assert (reply.valid, reply.duration, reply.info) == (False, None, info)
    assert _declared(reply) == {UIParam.TE: 2500, UIParam.NX: 7}


def test_a_pypulseq_feasibility_assertion_is_an_expected_rejection(calls, caplog):
    def evaluate(self, system, protocol):
        pp.make_trapezoid("x", system=system, area=1e5, duration=1e-4)

    plugin = _plugin(_delays(calls), evaluate=evaluate)

    with caplog.at_level(logging.WARNING, logger="pulserver.design"):
        reply = plugin.validate(SYSTEM, {})

    assert not reply.valid
    assert "Requested area is too large" in reply.info
    assert [record.levelno for record in caplog.records] == [logging.WARNING]


def test_the_default_generate_of_a_function_app_passes_the_arguments_of_the_protocol(
    calls,
):
    plugin = _plugin(_delays(calls))
    protocol = Protocol.from_wire(
        plugin.protocol, {UIParam.TE: 4000, UIParam.NX: 2}, SYSTEM
    )

    built = plugin.generate(SYSTEM, protocol)

    assert isinstance(built, pp.Sequence)
    assert calls == [{"te": 4e-3, "nx": 2}]


def test_design_writes_one_sequence_as_sequence_seq_in_the_binary_form(calls, tmp_path):
    validation, paths = _plugin(_delays(calls)).design(
        SYSTEM, {UIParam.TE: 4000, UIParam.NX: 2}, tmp_path
    )

    assert validation.valid, validation.info
    assert paths == [str(tmp_path / "sequence.seq")]
    assert b"[BLOCKS]" not in Path(paths[0]).read_bytes()
    assert _durations(paths) == [pytest.approx(2 * 4e-3)]
    assert pp.io.read(paths[0]).get_definition("NextSequence") == ""


def test_design_accepts_the_directory_as_text(calls, tmp_path):
    _, paths = _plugin(_delays(calls)).design(SYSTEM, {}, str(tmp_path))

    assert paths == [str(tmp_path / "sequence.seq")]


def test_design_writes_a_chain_linked_by_next_sequence_in_play_order(tmp_path):
    def chain(system, nx=3):
        return [_sequence(system, 1e-3, 2), _sequence(system, 2e-3, nx)]

    plugin = _plugin(chain, protocol={UIParam.NX: ENTRIES[UIParam.NX]})

    validation, paths = plugin.design(SYSTEM, {UIParam.NX: 4}, tmp_path)

    assert validation.valid, validation.info
    assert [Path(p).name for p in paths] == ["sequence.seq", "sequence_main.seq"]
    assert _durations(paths) == [pytest.approx(2e-3), pytest.approx(4 * 2e-3)]
    prescan, scan = (pp.io.read(path) for path in paths)
    assert prescan.get_definition("NextSequence") == "sequence_main.seq"
    assert scan.get_definition("NextSequence") == ""


def test_design_names_the_files_of_a_longer_chain_by_position(tmp_path):
    def chain(system):
        return [_sequence(system, 1e-3, n) for n in (1, 2, 3, 4)]

    validation, paths = _plugin(chain, protocol={}).design(SYSTEM, {}, tmp_path)

    assert validation.valid, validation.info
    names = [Path(p).name for p in paths]
    assert names == [
        "sequence.seq",
        "sequence_prescan2.seq",
        "sequence_prescan3.seq",
        "sequence_main.seq",
    ]
    linked = [pp.io.read(path).get_definition("NextSequence") for path in paths]
    assert linked == [*names[1:], ""]
    assert _durations(paths) == [pytest.approx(n * 1e-3) for n in (1, 2, 3, 4)]


def test_design_writes_a_sequence_application_as_the_application_writes_it(
    counted, tmp_path
):
    plugin = type(
        "Scanned",
        (SequencePlugin,),
        {"app": ScannedApp, "protocol": {UIParam.TR: TimeParam("tr")}},
    )()

    validation, paths = plugin.design(SYSTEM, {UIParam.TR: 20000}, tmp_path)

    assert validation.valid, validation.info
    assert [Path(p).name for p in paths] == ["sequence.seq", "sequence_main.seq"]
    assert _durations(paths) == [pytest.approx(1e-3), pytest.approx(3 * 20e-3)]
    assert pp.io.read(paths[0]).get_definition("NextSequence") == "sequence_main.seq"
    assert b"[BLOCKS]" not in Path(paths[0]).read_bytes()


def test_design_of_an_invalid_request_neither_generates_nor_writes(calls, tmp_path):
    def evaluate(self, system, protocol):
        raise ValueError("no such design")

    plugin = _plugin(_delays(calls), evaluate=evaluate)

    validation, paths = plugin.design(SYSTEM, {}, tmp_path)

    assert (validation.valid, validation.info) == (False, "no such design")
    assert paths == []
    assert calls == []
    assert list(tmp_path.iterdir()) == []


def test_an_error_while_generating_is_not_an_invalid_protocol(tmp_path):
    def failing(system):
        raise RuntimeError("the design failed")

    plugin = _plugin(failing, protocol={})

    assert plugin.validate(SYSTEM, {}).valid
    with pytest.raises(RuntimeError, match="the design failed"):
        plugin.design(SYSTEM, {}, tmp_path)


@pytest.mark.parametrize("returned", [None, "sequence.seq", [], [None]])
def test_a_function_app_returning_neither_a_sequence_nor_a_list_is_a_type_error(
    tmp_path, returned
):
    plugin = _plugin(lambda system: returned, protocol={})

    with pytest.raises(TypeError, match="neither a Sequence nor a list"):
        plugin.design(SYSTEM, {}, tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_a_validation_and_the_design_after_it_construct_the_application_once(
    counted, tmp_path
):
    plugin = type("Scanned", (SequencePlugin,), {"app": ScannedApp})()

    reply = plugin.validate(SYSTEM, {})
    plugin.design(SYSTEM, reply.values, tmp_path)

    assert counted.constructions == 1


def test_a_design_does_not_reuse_the_application_of_an_earlier_design(
    counted, tmp_path
):
    plugin = type("Scanned", (SequencePlugin,), {"app": ScannedApp})()

    for name in ("first", "second"):
        (tmp_path / name).mkdir()
        plugin.design(SYSTEM, {}, tmp_path / name)

    assert counted.constructions == 2


def test_a_subclass_of_scanner_sequence_warns_at_its_class_statement(calls):
    with pytest.warns(DeprecationWarning, match="ScannerSequence") as warned:

        class Old(ScannerSequence):
            app = _delays(calls)
            protocol = ENTRIES

    assert warned[0].filename == __file__
    assert issubclass(Old, SequencePlugin)
    assert Old().validate(SYSTEM, {}).valid


def test_a_ui_declaration_warns_and_is_the_protocol(calls):
    with pytest.warns(DeprecationWarning, match="ui") as warned:

        class Declared(SequencePlugin):
            app = _delays(calls)
            ui = ENTRIES

    assert warned[0].filename == __file__
    assert Declared.protocol == Declared.ui == ENTRIES
    assert list(Declared().listing())[:2] == [UIParam.TE, UIParam.NX]


def test_a_class_declaring_protocol_and_ui_is_refused(calls):
    with pytest.raises(ValueError, match="both protocol and ui"):

        class Both(SequencePlugin):
            app = _delays(calls)
            protocol = ENTRIES
            ui = ENTRIES


def test_a_plugin_file_may_subclass_the_deprecated_name(tmp_path):
    path = tmp_path / "old.py"
    path.write_text(
        "from pulserver.design import ScannerSequence\n\n"
        "def app(system):\n    raise AssertionError\n\n"
        "class Old(ScannerSequence):\n    app = app\n"
    )

    with pytest.warns(DeprecationWarning, match="ScannerSequence"):
        plugin = load_plugin(path)

    assert isinstance(plugin, SequencePlugin)


@pytest.mark.parametrize(
    "path",
    sorted([*_plugins.SEQUENCES.glob("*.py"), *PLUGINS.glob("*.py")]),
    ids=lambda path: path.stem,
)
def test_a_shipped_or_test_plugin_declares_no_deprecated_name(path):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        plugin = load_plugin(path)

    assert isinstance(plugin, SequencePlugin)
    assert not {"ui", "recon", "follows"} & set(vars(type(plugin)))
