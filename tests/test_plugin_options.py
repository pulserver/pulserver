"""Plugin options: user entries a request states, at which the design calls load the plugin."""

import pytest
from _host import LIMITS, value_block

from pulserver.design import (
    Description,
    OptionError,
    OptionParam,
    SequencePlugin,
    load_plugin,
    option,
)
from pulserver.host import DesignStore, _command, _server, _service
from pulserver.protocol import UIParam, parse_listing

REPEATED = '''"""Delays, as many as an option says; another option adds a user entry."""

import pypulseqpp as pp
from pypulseqpp import sequences

from pulserver.design import (
    Description,
    Evaluation,
    IntParam,
    SequencePlugin,
    TimeParam,
    option,
)
from pulserver.protocol import UIParam

REPETITIONS = option("repetitions", 4, choices=(2, 4, 6), user=40, label="Repeats")
SPACING = option(
    "spacing", "short", choices=("short", "very_long"), user=41, label="Spacing"
)
EXTRA = option("extra", False, user=42, label="Extra delays")


def repeated(system, *, te: float = 8e-3, n_extra: int = 0):
    seq = pp.Sequence(system)
    delay = te if SPACING == "short" else 2 * te
    for _ in range(REPETITIONS + n_extra):
        seq.add_block(pp.make_delay(delay))
    return seq


class Repeated(SequencePlugin):
    app = repeated
    protocol = {UIParam.TE: TimeParam("te", range_min=1000, range_max=80000)}
    if EXTRA:
        protocol |= {
            UIParam.user_name(12): Description("Extra delays"),
            UIParam.user_value(12): IntParam("n_extra", range_min=0, range_max=8),
        }

    def evaluate(self, system, protocol):
        sequence = self.app(system, **protocol.arguments)
        return Evaluation(protocol, sequences.duration(sequence))
'''

REPEATS = UIParam.user_value(40)
SPACING = UIParam.user_value(41)
EXTRA = UIParam.user_value(42)


@pytest.fixture
def installed(tmp_path):
    """A scanner layout: ``sequences/sequence3.py`` linked to ``sequences/src/repeated.py``."""
    sequences = tmp_path / "sequences"
    (sequences / "src").mkdir(parents=True)
    (sequences / "src" / "repeated.py").write_text(REPEATED)
    (sequences / "sequence3.py").symlink_to("src/repeated.py")
    return sequences


def _listed(plugins, values=None):
    status, reply = _service.call(
        "list",
        plugins=plugins,
        plugin="sequence3",
        block=value_block(values) if values else "",
    )
    assert status == 0, reply
    return parse_listing(reply)


def _generated(store, plugins, options=None):
    status, reply = _service.call(
        "generate",
        plugins=plugins,
        plugin="sequence3",
        limits=LIMITS,
        block=value_block({"TE": 5000, **(options or {})}),
        store=store,
    )
    assert status == 0, reply
    return reply.split()[1]


def test_an_option_outside_a_plugin_load_is_its_default():
    assert option("repetitions", 4, choices=(4, 6), user=40, label="Repeats") == 4


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"default": True, "choices": (False, True)}, "a flag takes no choices"),
        ({"default": 4}, "names its choices"),
        ({"default": 3, "choices": (4, 6)}, "default 3 is not one of"),
        ({"default": 1, "choices": tuple(range(6))}, "at most 5"),
        ({"default": True, "user": 46}, "user46 is not a user entry"),
    ],
)
def test_an_option_the_console_cannot_list_is_refused(arguments, message):
    with pytest.raises(OptionError, match=message):
        option("repetitions", **{"user": 40, "label": "Repeats", **arguments})


def test_a_plugin_records_every_option_at_the_value_it_was_loaded_with(installed):
    plugin = load_plugin(installed / "sequence3.py", {"repetitions": 6})

    assert plugin.options == {"repetitions": 6, "spacing": "short", "extra": False}
    assert load_plugin(installed / "sequence3.py").options["repetitions"] == 4


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"repetitons": 6}, "does not declare the options \\['repetitons'\\]"),
        ({"repetitions": "6"}, "repetitions is one of"),
        ({"repetitions": 5}, "repetitions is one of"),
        ({"extra": 1}, "extra is one of"),
        ({"spacing": "medium"}, "spacing is one of"),
    ],
)
def test_an_option_the_plugin_does_not_declare_or_take_is_refused(
    installed, options, message
):
    with pytest.raises(OptionError, match=message):
        load_plugin(installed / "sequence3.py", options)


def test_an_option_is_listed_at_its_user_entry_as_the_index_of_its_value(installed):
    plugin = load_plugin(installed / "sequence3.py", {"spacing": "very_long"})
    listing = plugin.listing()

    assert listing[REPEATS].value == 1
    assert listing[REPEATS].options == (0, 1, 2)
    assert listing[SPACING].value == 1
    assert listing[EXTRA].value == 0
    assert (listing[EXTRA].range_min, listing[EXTRA].range_max) == (0, 1)
    assert listing[UIParam.user_name(41)].value == "Spacing: 0 short, 1 very long"
    assert listing[UIParam.user_name(42)].value == "Extra delays: 0 off, 1 on"


def test_an_option_shapes_the_protocol_the_plugin_declares(installed):
    plain = load_plugin(installed / "sequence3.py")
    extra = load_plugin(installed / "sequence3.py", {"extra": True})

    assert UIParam.user_value(12) not in plain.protocol
    assert UIParam.user_value(12) in extra.protocol


def test_a_request_for_another_choice_than_the_loaded_one_is_invalid(installed):
    from pulserver.host._limits import design_system

    plugin = load_plugin(installed / "sequence3.py")

    validation = plugin.validate(design_system(LIMITS), {REPEATS: 2})

    assert not validation.valid
    assert "repetitions 6" in validation.info


def test_an_entry_at_the_user_entry_of_an_option_is_refused():
    option("navigator", False, user=44, label="Navigator")

    with pytest.raises(ValueError, match="holds the option navigator"):

        class Clashing(SequencePlugin):
            app = staticmethod(lambda system: None)
            protocol = {UIParam.user_name(44): Description("Something else")}


def test_two_options_at_one_user_entry_are_refused(tmp_path):
    plugin = tmp_path / "clashing.py"
    plugin.write_text(
        "from pulserver.design import option\n"
        "A = option('a', False, user=40, label='A')\n"
        "B = option('b', False, user=40, label='B')\n"
    )

    with pytest.raises(OptionError, match="both at user40"):
        load_plugin(plugin)


def test_a_listing_is_that_of_the_plugin_at_the_options_of_its_block(installed):
    assert UIParam.user_value(12) not in _listed(installed)
    assert UIParam.user_value(12) in _listed(installed, {EXTRA: 1})
    # Other lines, of entries the plugin does or does not list, are not read.
    assert _listed(installed, {EXTRA: 1, "TE": 9000, "user12_value": 3})[
        EXTRA
    ].value == (1)


def test_an_option_entry_that_holds_no_choice_is_an_error_naming_it(installed):
    status, reply = _service.call(
        "list", plugins=installed, plugin="sequence3", block=value_block({EXTRA: 2})
    )

    assert status == 1
    assert "user42_value" in reply and "extra" in reply


def test_the_options_of_a_request_are_a_design_of_their_own_and_are_recorded(
    installed, tmp_path
):
    store = DesignStore(tmp_path / "designs", "scan")
    plain = _generated(store, installed)
    six = _generated(store, installed, {REPEATS: 2})

    assert plain != six
    assert store.manifest(six)["options"]["repetitions"] == 6
    assert _generated(store, installed, {REPEATS: 2}) == six
    assert _generated(store, installed, {REPEATS: 1}) == plain


def test_a_request_that_shows_an_entry_its_options_hide_is_refused(installed):
    status, reply = _service.call(
        "validate",
        plugins=installed,
        plugin="sequence3",
        limits=LIMITS,
        block=value_block({"TE": 5000, "user12_value": 3}),
    )

    assert status == 1
    assert "user12_value" in reply


def test_a_request_that_shows_an_entry_its_options_add_reaches_the_plugin(installed):
    status, reply = _service.call(
        "validate",
        plugins=installed,
        plugin="sequence3",
        limits=LIMITS,
        block=value_block({"TE": 5000, EXTRA: 1, "user12_value": 2}),
    )

    assert status == 0, reply
    assert reply.startswith("VALID 0.03")


def test_the_command_and_the_server_carry_a_listing_s_values(tmp_path, installed):
    values = tmp_path / "values.protocol"
    values.write_text(value_block({EXTRA: 1}))
    args = _command._parser().parse_args(
        [
            "list",
            "--plugins",
            str(installed),
            "--plugin",
            "sequence3",
            "--values",
            str(values),
        ]
    )
    request = _command.request(args, "")

    assert request["input"] == value_block({EXTRA: 1})
    status, reply = _server.answer(request)
    assert status == 0, reply
    assert UIParam.user_value(12) in parse_listing(reply)


def test_an_option_entry_is_an_option_param_of_the_plugin_s_protocol(installed):
    plugin = load_plugin(installed / "sequence3.py")

    assert plugin.protocol[EXTRA] == OptionParam("extra", (False, True), False)
