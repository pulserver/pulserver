"""Plugin options and the parameter files that set them, by numbered set."""

import importlib.util
import os

import pytest
from _host import LIMITS

from pulserver import _plugins
from pulserver.design import OptionError, load_plugin, option, read_options
from pulserver.design._options import loading_options
from pulserver.host import DesignStore, _command, _service
from pulserver.protocol import PROTOCOL_BEGIN, PROTOCOL_END, UIParam, parse_listing

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

REPETITIONS = option("repetitions", 4)
SPACING = option("spacing", "short", choices=("short", "long"))
EXTRA = option("extra", False)


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


@pytest.fixture
def installed(tmp_path):
    """A scanner layout: ``sequences/sequence3.py`` linked to ``sequences/src/repeated.py``, and set 1 (``repetitions = 6``) and set 2 (``extra = true``) of ``parameters/repeated``."""
    sequences = tmp_path / "sequences"
    (sequences / "src").mkdir(parents=True)
    (sequences / "src" / "repeated.py").write_text(REPEATED)
    (sequences / "sequence3.py").symlink_to("src/repeated.py")
    parameters = tmp_path / "parameters" / "repeated"
    parameters.mkdir(parents=True)
    (parameters / "01.toml").write_text("repetitions = 6\n")
    (parameters / "02.toml").write_text("extra = true\n")
    return sequences


def _block(values):
    lines = [f"{key}: {value}" for key, value in values.items()]
    return "\n".join([PROTOCOL_BEGIN, *lines, PROTOCOL_END]) + "\n"


def _generated(store, plugins, parameter_set=0):
    status, reply = _service.call(
        "generate",
        plugins=plugins,
        plugin="sequence3",
        limits=LIMITS,
        block=_block({"TE": 5000}),
        store=store,
        parameter_set=parameter_set,
    )
    assert status == 0, reply
    return reply.split()[1]


def test_an_option_outside_a_plugin_load_is_its_default():
    assert option("repetitions", 4) == 4


def test_a_float_option_takes_an_int_and_holds_a_float():
    with loading_options({"scale": 2}) as resolved:
        assert option("scale", 1.5) == 2.0
    assert resolved == {"scale": 2.0}
    assert isinstance(resolved["scale"], float)


def test_a_plugin_records_every_option_at_the_value_it_was_loaded_with(installed):
    plugin = load_plugin(installed / "sequence3.py", {"repetitions": 6})

    assert plugin.options == {"repetitions": 6, "spacing": "short", "extra": False}
    assert load_plugin(installed / "sequence3.py").options["repetitions"] == 4


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"repetitons": 6}, "does not declare the options \\['repetitons'\\]"),
        ({"repetitions": "6"}, "repetitions is a int"),
        ({"extra": 1}, "extra is a bool"),
        ({"repetitions": True}, "repetitions is a int"),
        ({"spacing": "medium"}, "spacing is one of"),
    ],
)
def test_an_option_the_plugin_does_not_declare_or_take_is_refused(
    installed, options, message
):
    with pytest.raises(OptionError, match=message):
        load_plugin(installed / "sequence3.py", options)


def test_an_option_shapes_the_protocol_the_plugin_declares(installed):
    plain = load_plugin(installed / "sequence3.py")
    extra = load_plugin(installed / "sequence3.py", {"extra": True})

    assert UIParam.user_value(12) not in plain.protocol
    assert UIParam.user_value(12) in extra.protocol


def test_a_parameter_file_that_is_not_toml_is_refused(tmp_path):
    (tmp_path / "01.toml").write_text("repetitions 6\n")

    with pytest.raises(OptionError, match="not TOML"):
        read_options(tmp_path / "01.toml")


def test_a_parameter_set_is_the_file_of_the_script_its_slot_links_to(installed):
    slot = installed / "sequence3.py"
    parameters = installed.parent / "parameters"

    assert _service.parameter_file(slot, 0) is None
    assert _service.parameter_file(slot, 1) == parameters / "repeated" / "01.toml"
    assert _service.parameter_file(slot, 2, parameters) == (
        parameters / "repeated" / "02.toml"
    )
    with pytest.raises(_service.CallError, match="no file"):
        _service.parameter_file(slot, 3)
    with pytest.raises(_service.CallError, match="negative"):
        _service.parameter_file(slot, -1)


def test_a_listing_is_that_of_the_plugin_loaded_with_its_parameter_set(installed):
    def listed(parameter_set):
        status, reply = _service.call(
            "list", plugins=installed, plugin="sequence3", parameter_set=parameter_set
        )
        assert status == 0, reply
        return parse_listing(reply)

    assert UIParam.user_value(12) not in listed(0)
    assert UIParam.user_value(12) in listed(2)


def test_a_parameter_set_is_a_design_of_its_own_and_its_options_are_recorded(
    installed, tmp_path
):
    store = DesignStore(tmp_path / "designs", "scan")
    plain = _generated(store, installed)
    six = _generated(store, installed, parameter_set=1)

    assert plain != six
    assert store.manifest(six)["options"]["repetitions"] == 6
    assert _generated(store, installed, parameter_set=1) == six


def test_an_edited_parameter_file_is_another_design(installed, tmp_path):
    store = DesignStore(tmp_path / "designs", "scan")
    first = _generated(store, installed, parameter_set=1)
    file = installed.parent / "parameters" / "repeated" / "01.toml"
    file.write_text('repetitions = 6\nspacing = "long"\n')
    stat = file.stat()
    os.utime(file, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))

    assert _generated(store, installed, parameter_set=1) != first


def test_a_parameter_file_the_plugin_does_not_take_is_an_error_naming_it(installed):
    file = installed.parent / "parameters" / "repeated" / "01.toml"
    file.write_text("navigator = true\n")

    status, reply = _service.call(
        "list", plugins=installed, plugin="sequence3", parameter_set=1
    )

    assert status == 1
    assert str(file) in reply and "navigator" in reply


def test_the_command_forwards_the_parameter_set_and_its_directory(tmp_path):
    args = _command._parser().parse_args(
        [
            "list",
            "--plugins",
            str(tmp_path),
            "--plugin",
            "sequence3",
            "--parameter-set",
            "2",
            "--parameters",
            str(tmp_path / "parameters"),
        ]
    )
    request = _command.request(args, "")

    assert request["parameter_set"] == 2
    assert request["parameters"] == str((tmp_path / "parameters").absolute())


SHIPPED_PRESETS = sorted(_plugins.PARAMETERS.glob("*/*.toml"))


@pytest.mark.parametrize(
    "preset",
    SHIPPED_PRESETS,
    ids=[f"{p.parent.name}/{p.stem}" for p in SHIPPED_PRESETS],
)
def test_every_shipped_parameter_file_loads_in_its_shipped_sequence(preset):
    options = read_options(preset)
    if (
        options.get("refocusing", "constant") != "constant"
        and importlib.util.find_spec("blochsim") is None
    ):
        pytest.skip("needs blochsim")
    plugin = load_plugin(_plugins.SEQUENCES / f"{preset.parent.name}.py", options)

    assert options.items() <= plugin.options.items()


def test_a_parameter_set_asks_for_the_field_camera_train_for_the_psd_not_the_plugin(
    installed, tmp_path
):
    """``field_camera`` is no option of the plugin: it reaches the PSD in ``psd.cvs``."""
    store = DesignStore(tmp_path / "designs", "scan")
    six = _generated(store, installed, parameter_set=1)
    file = installed.parent / "parameters" / "repeated" / "03.toml"
    file.write_text("repetitions = 6\nfield_camera = true\n")

    camera = _generated(store, installed, parameter_set=3)

    assert camera != six
    assert store.manifest(camera)["options"]["repetitions"] == 6
    assert store.manifest(camera)["psd"] == {"sfc_enable": 1}
    assert (store.directory(camera) / "psd.cvs").read_text() == "sfc_enable 1\n"
    assert not (store.directory(six) / "psd.cvs").exists()


def test_a_field_camera_that_is_not_true_or_false_is_refused(installed):
    file = installed.parent / "parameters" / "repeated" / "01.toml"
    file.write_text("field_camera = 1\n")

    status, reply = _service.call(
        "list", plugins=installed, plugin="sequence3", parameter_set=1
    )

    assert status == 1
    assert "field_camera" in reply and "true or false" in reply
