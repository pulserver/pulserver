"""The protocol key enums against the interpreter's parameter table."""

import re
from pathlib import Path

import pypulseqpp as pp
import pytest

from pulserver.design import IntParam, Protocol, SequencePlugin, load_plugin
from pulserver.protocol import (
    BoolKey,
    ConfigKey,
    EnumKey,
    FloatKey,
    ImagingMode,
    IntKey,
    Kind,
    Parameter,
    ProtocolKey,
    TEPreset,
    UIParam,
    UserKey,
    UserNameKey,
    format_listing,
)
from pulserver.protocol._keys import NUM_USER_ENTRIES

TABLE = Path(__file__).parents[1] / "src" / "c" / "io" / "pulseg_protocol.c"
USER = re.compile(r"user\d+_(value|name)")
PLUGINS = Path(__file__).parent / "plugins"
SYSTEM = pp.Opts(max_grad=40.0, grad_unit="mT/m", max_slew=150.0, slew_unit="T/m/s")


def _table():
    """``{type: {wire name, ...}}`` from ``g_param_table``."""
    entries = re.findall(
        r'\{"(\w+)",\s*PULSEG_PARAM_\w+,\s*PULSEG_PTYPE_(\w+)\}', TABLE.read_text()
    )
    table = {}
    for name, kind in entries:
        table.setdefault(kind, set()).add(name)
    return table


@pytest.mark.parametrize(
    ("kind", "keys"),
    [
        ("FLOAT", FloatKey),
        ("INT", IntKey),
        ("BOOL", BoolKey),
        ("STRINGLIST", EnumKey),
        ("CONFIG", ConfigKey),
    ],
)
def test_each_key_enum_is_its_type_in_the_interpreter_table(kind, keys):
    declared = {name for name in _table()[kind] if not USER.fullmatch(name)}
    assert {key.value for key in keys} == declared


def test_the_user_entries_are_the_interpreter_tables():
    table = _table()
    assert {key.value for key in UserKey} == {
        name for name in table["FLOAT"] if USER.fullmatch(name)
    }
    assert {key.value for key in UserNameKey} == table["DESCRIPTION"]


def test_a_user_entry_key_is_the_member_of_its_enum_that_the_function_returns():
    for n in range(NUM_USER_ENTRIES):
        assert UIParam.user_value(n) is UserKey[f"USER{n}"]
        assert UIParam.user_value(n) == f"user{n}_value"
        assert UIParam.user_name(n) is UserNameKey[f"USER{n}"]
        assert UIParam.user_name(n) == f"user{n}_name"


def test_every_key_enum_is_a_protocol_key():
    enums = (FloatKey, IntKey, BoolKey, EnumKey, ConfigKey, UserKey, UserNameKey)
    assert all(isinstance(key, ProtocolKey) for keys in enums for key in keys)
    assert not isinstance("TE", ProtocolKey)


def test_ui_param_carries_every_typed_key():
    members = {
        value for name, value in vars(UIParam).items() if not name.startswith("_")
    } - {UIParam.__dict__["user_value"], UIParam.__dict__["user_name"]}
    assert members == {*FloatKey, *IntKey, *BoolKey, *EnumKey}


@pytest.mark.parametrize("n", [-1, NUM_USER_ENTRIES])
def test_a_user_entry_outside_the_table_is_refused(n):
    with pytest.raises(ValueError, match="numbered"):
        UIParam.user_value(n)


def test_a_scanner_sequence_refuses_a_name_the_interpreter_does_not_know():
    with pytest.raises(ValueError, match=r"\['Nx'\]"):

        class Misspelled(SequencePlugin):
            protocol = {"Nx": IntParam("n_x")}


def test_a_scanner_sequence_stores_its_entries_under_key_members():
    class Keyed(SequencePlugin):
        protocol = {
            UIParam.NX: IntParam("n_x"),
            UIParam.user_value(0): IntParam("n_y"),
        }

    assert [type(key) for key in Keyed.protocol] == [IntKey, UserKey]
    assert list(Keyed.protocol) == ["nx", "user0_value"]


def test_a_plain_string_naming_an_entry_is_stored_as_its_member():
    class Plain(SequencePlugin):
        protocol = {"nx": IntParam("n_x"), "user0_name": IntParam("n_y")}

    assert [type(key) for key in Plain.protocol] == [IntKey, UserNameKey]
    assert Plain.protocol[IntKey.NX] == IntParam("n_x")


@pytest.fixture(scope="module")
def typed():
    return load_plugin(PLUGINS / "typed.py")


def test_a_protocol_keeps_its_typed_keys(typed):
    listing = typed.listing()
    wire = {key: p.value for key, p in listing.items() if p.editable}
    protocol = Protocol.from_wire(typed.protocol, wire, SYSTEM)
    stages = {
        "entries": typed.protocol,
        "listing": listing,
        "protocol": protocol,
        "replaced": protocol.replace({UIParam.NX: 3}),
        "to_wire": protocol.to_wire(),
        "validation": typed.validate(SYSTEM, {UIParam.TE: TEPreset.MINIMUM}).values,
    }
    for stage, keyed in stages.items():
        assert keyed, stage
        assert all(isinstance(key, ProtocolKey) for key in keyed), stage


def test_string_list_options_travel_as_their_values():
    parameter = Parameter(
        Kind.STRINGLIST, ImagingMode.THREE_D, options=tuple(ImagingMode)
    )
    assert parameter.options == ("2d", "3d")
    line = format_listing({UIParam.IMAGING_MODE: parameter}).splitlines()[1]
    # The value travels as the index of the chosen option.
    assert line == "imaging_mode: stringlist|1|2d|3d"
