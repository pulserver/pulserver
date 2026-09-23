"""The protocol key enums against the interpreter's parameter table."""

import re
from pathlib import Path

import pytest

from pulserver.design import IntParam, ScannerSequence
from pulserver.protocol import (
    BoolKey,
    ConfigKey,
    EnumKey,
    FloatKey,
    ImagingMode,
    IntKey,
    Kind,
    Parameter,
    UIParam,
    format_listing,
)
from pulserver.protocol._keys import NUM_USER_ENTRIES

TABLE = Path(__file__).parents[1] / "src" / "c" / "io" / "pulseg_protocol.c"
USER = re.compile(r"user\d+_(value|name)")


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
    values = {UIParam.user_value(n) for n in range(NUM_USER_ENTRIES)}
    names = {UIParam.user_name(n) for n in range(NUM_USER_ENTRIES)}
    assert values == {name for name in table["FLOAT"] if USER.fullmatch(name)}
    assert names == table["DESCRIPTION"]


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

        class Misspelled(ScannerSequence):
            ui = {"Nx": IntParam("n_x")}


def test_an_enum_key_is_stored_and_sent_as_its_wire_name():
    class Keyed(ScannerSequence):
        ui = {UIParam.NX: IntParam("n_x"), UIParam.user_value(0): IntParam("n_y")}

    assert [type(name) for name in Keyed.ui] == [str, str]
    assert list(Keyed.ui) == ["nx", "user0_value"]


def test_string_list_options_travel_as_their_values():
    parameter = Parameter(
        Kind.STRINGLIST, ImagingMode.THREE_D, options=tuple(ImagingMode)
    )
    assert parameter.options == ("2d", "3d")
    line = format_listing({UIParam.IMAGING_MODE: parameter}).splitlines()[1]
    # The value travels as the index of the chosen option.
    assert line == "imaging_mode: stringlist|1|2d|3d"
