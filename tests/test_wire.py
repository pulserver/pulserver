import pytest

from pulserver.protocol import (
    PROTOCOL_BEGIN,
    PROTOCOL_END,
    InputMode,
    Kind,
    Parameter,
    TEPreset,
    Validation,
    format_listing,
    format_validation,
    format_values,
    parse_listing,
    parse_validation,
    parse_values,
)

LISTING = {
    "TE": Parameter(
        Kind.INT,
        8000,
        InputMode.DROPDOWN,
        1000,
        80000,
        100,
        "us",
        (TEPreset.MINIMUM, 5000, 8000),
    ),
    "TR": Parameter(Kind.INT, 250000, InputMode.TYPEIN, 5000, 5000000, 1000, "us"),
    "fov": Parameter(Kind.FLOAT, 220.0, InputMode.TYPEIN, 50.0, 500.0, 1.0, "mm"),
    "flip": Parameter(
        Kind.FLOAT, 12.0, InputMode.DROPDOWN, 1.0, 90.0, 1.0, "deg", (5.0, 12.0)
    ),
    "swap_phase_freq": Parameter(Kind.BOOL, False),
    "sequence_type": Parameter(
        Kind.STRINGLIST, "gre", InputMode.DROPDOWN, options=("gre", "se")
    ),
    "enable_sar_burst_mode": Parameter(Kind.CONFIG, 1, InputMode.OFF),
    "user0_name": Parameter(Kind.DESCRIPTION, "Line one\nline two"),
}

# One line per kind, in the grammar pulseg_protocol_parse reads.
LISTED_LINES = [
    "TE: int|dropdown|8000|1000|80000|100|us|-2|5000|8000",
    "TR: int|typein|250000|5000|5000000|1000|us",
    "fov: float|typein|220.0|50.0|500.0|1.0|mm",
    "flip: float|dropdown|12.0|1.0|90.0|1.0|deg|5.0|12.0",
    "swap_phase_freq: bool|false",
    "sequence_type: stringlist|0|gre|se",
    "enable_sar_burst_mode: config|1",
    "user0_name: description|Line one\\nline two",
]


def test_every_listed_line_matches_the_interpreter_grammar():
    lines = format_listing(LISTING).splitlines()
    assert lines == [PROTOCOL_BEGIN, *LISTED_LINES, PROTOCOL_END]


def test_a_listed_protocol_parses_back_to_the_same_schema():
    assert parse_listing(format_listing(LISTING)) == LISTING


def test_a_preset_travels_as_its_negative_dropdown_value():
    block = format_values({"TE": TEPreset.MINIMUM}, LISTING)
    assert "TE: -2" in block.splitlines()
    assert parse_values(block, LISTING)["TE"] == TEPreset.MINIMUM


def test_a_stringlist_value_is_read_by_option_or_by_index():
    by_index = f"{PROTOCOL_BEGIN}\nsequence_type: 1\n{PROTOCOL_END}"
    by_option = f"{PROTOCOL_BEGIN}\nsequence_type: se\n{PROTOCOL_END}"
    assert parse_values(by_index, LISTING) == {"sequence_type": "se"}
    assert parse_values(by_option, LISTING) == {"sequence_type": "se"}
    assert "sequence_type: 1" in format_values({"sequence_type": "se"}, LISTING)


def test_read_only_entries_travel_in_listings_but_not_in_values():
    values = {"TR": 10000, "enable_sar_burst_mode": 1, "user0_name": "x"}
    assert format_values(values, LISTING).splitlines() == [
        PROTOCOL_BEGIN,
        "TR: 10000",
        PROTOCOL_END,
    ]
    sent = f"{PROTOCOL_BEGIN}\nTR: 10000\nuser0_name: anything\n{PROTOCOL_END}"
    assert parse_values(sent, LISTING) == {"TR": 10000}


def test_a_value_for_an_undeclared_parameter_is_refused():
    with pytest.raises(ValueError, match="nex"):
        parse_values(f"{PROTOCOL_BEGIN}\nnex: 2\n{PROTOCOL_END}", LISTING)


def test_a_value_its_entry_cannot_take_is_an_error_naming_the_entry():
    with pytest.raises(ValueError, match=r"^TR: "):
        parse_values(f"{PROTOCOL_BEGIN}\nTR: 0.6\n{PROTOCOL_END}", LISTING)


@pytest.mark.parametrize(
    "validation",
    [
        Validation(True, 1.5, "TA = 0:02", {"TE": 2800, "TR": 6460, "fov": 180.0}),
        Validation(True, None, "", {"swap_phase_freq": True}),
        Validation(False, None, "the requested TR is too short", {"TR": 1000}),
    ],
    ids=["valid", "no-duration", "invalid"],
)
def test_a_validation_reply_round_trips(validation):
    reply = format_validation(validation, LISTING)
    assert parse_validation(reply, LISTING) == validation


def test_a_dropdown_without_options_is_refused():
    with pytest.raises(ValueError, match="dropdown"):
        Parameter(Kind.FLOAT, 1.0, InputMode.DROPDOWN)
