"""The typed protocol: keys, choices and values are objects until the wire."""

from pathlib import Path

import numpy as np
import pypulseqpp as pp
import pytest
from _host import LIMITS, PLUGINS, value_block

from pulserver.design import (
    ChoiceParam,
    Protocol,
    SequencePlugin,
    StringListParam,
    TimeParam,
    load_plugin,
)
from pulserver.host import DesignStore, call, design_identity
from pulserver.host._blocks import format_import, parse_import
from pulserver.protocol import (
    FloatKey,
    ImagingMode,
    Kind,
    ProtocolKey,
    SequenceType,
    TEPreset,
    TRPreset,
    UIParam,
    UserKey,
    format_listing,
    format_prescription,
    format_validation,
    parse_prescription,
    parse_values,
)

SYSTEM = pp.Opts(max_grad=40.0, grad_unit="mT/m", max_slew=150.0, slew_unit="T/m/s")


def block(lines):
    return "\n".join(["[Protocol]", *lines, "[Protocol End]"]) + "\n"


# The wire text of the interpreter's grammar, as the design calls reply it.
PRESCRIPTION_LISTING = [
    "fov_offset_x: float|off|0.0|-1000.0|1000.0|0.1|mm",
    "fov_offset_y: float|off|0.0|-1000.0|1000.0|0.1|mm",
    "fov_offset_z: float|off|0.0|-1000.0|1000.0|0.1|mm",
    "fov_rotation_11: float|off|1.0|-1.0|1.0|1e-06|",
    "fov_rotation_12: float|off|0.0|-1.0|1.0|1e-06|",
    "fov_rotation_13: float|off|0.0|-1.0|1.0|1e-06|",
    "fov_rotation_21: float|off|0.0|-1.0|1.0|1e-06|",
    "fov_rotation_22: float|off|1.0|-1.0|1.0|1e-06|",
    "fov_rotation_23: float|off|0.0|-1.0|1.0|1e-06|",
    "fov_rotation_31: float|off|0.0|-1.0|1.0|1e-06|",
    "fov_rotation_32: float|off|0.0|-1.0|1.0|1e-06|",
    "fov_rotation_33: float|off|1.0|-1.0|1.0|1e-06|",
]
PRESCRIPTION_VALUES = [
    "fov_offset_x: 0.0",
    "fov_offset_y: 0.0",
    "fov_offset_z: 0.0",
    "fov_rotation_11: 1.0",
    "fov_rotation_12: 0.0",
    "fov_rotation_13: 0.0",
    "fov_rotation_21: 0.0",
    "fov_rotation_22: 1.0",
    "fov_rotation_23: 0.0",
    "fov_rotation_31: 0.0",
    "fov_rotation_32: 0.0",
    "fov_rotation_33: 1.0",
]
TYPED_LISTING = block(
    [
        "TE: int|dropdown|8000|1000|80000|100|us|-2",
        "TR: int|dropdown|-1|1000|5000000|1|us|-1",
        "imaging_mode: stringlist|0|2d|3d",
        "nx: int|typein|4|1|64|1|",
        "FatSat: bool|false",
        "user0_name: description|Thickness",
        "user0_value: float|typein|5.0|1.0|50.0|1.0|mm",
        "enable_sar_burst_mode: config|1",
        "nex: float|typein|1.0|1.0|16.0|1.0|",
        *PRESCRIPTION_LISTING,
    ]
)
TYPED_VALUES = block(
    [
        "TE: 2500",
        "TR: -1",
        "imaging_mode: 1",
        "nx: 6",
        "FatSat: true",
        "user0_value: 7.5",
        "nex: 1.0",
        *PRESCRIPTION_VALUES,
    ]
)
TYPED_INVALID_VALUES = block(
    [
        "TE: 1000",
        "TR: -1",
        "imaging_mode: 1",
        "nx: 4",
        "FatSat: false",
        "user0_value: 5.0",
        "nex: 1.0",
        *PRESCRIPTION_VALUES,
    ]
)
GRE2D_LISTING = block(
    [
        "imaging_mode: stringlist|0|2d|3d",
        "flip: float|typein|12.0|1.0|90.0|1.0|deg",
        "TE: int|dropdown|8000|1000|80000|10|us|-2|5000|8000",
        "TR: int|dropdown|250000|1000|5000000|1|us|-1",
        "bandwidth: float|typein|250000.0|1000.0|1000000.0|1.0|Hz",
        "fov: float|typein|220.0|50.0|500.0|1.0|mm",
        "phase_fov: float|typein|220.0|50.0|500.0|1.0|mm",
        "nx: int|typein|128|32|512|2|",
        "ny: int|typein|128|32|512|2|",
        "nslices: int|typein|1|1|64|1|",
        "slice_spacing: float|typein|0.0|0.0|100.0|0.1|mm",
        "slice_thickness: float|typein|5.0|1.0|20.0|1.0|mm",
        "Ry: int|typein|1|1|4|1|",
        "enable_saturation_ui: config|1",
        "exsat_mask: config|3",
        "exsat1_normal_x: float|typein|1.0|-1.0|1.0|0.001|",
        "exsat1_normal_y: float|typein|0.0|-1.0|1.0|0.001|",
        "exsat1_normal_z: float|typein|0.0|-1.0|1.0|0.001|",
        "exsat1_loc: float|typein|150.0|-500.0|500.0|1.0|mm",
        "exsat1_thickness: float|typein|40.0|5.0|200.0|1.0|mm",
        "exsat2_normal_x: float|typein|0.0|-1.0|1.0|0.001|",
        "exsat2_normal_y: float|typein|1.0|-1.0|1.0|0.001|",
        "exsat2_normal_z: float|typein|0.0|-1.0|1.0|0.001|",
        "exsat2_loc: float|typein|150.0|-500.0|500.0|1.0|mm",
        "exsat2_thickness: float|typein|40.0|5.0|200.0|1.0|mm",
        "user0_name: description|Dummy scans, -1 until steady state",
        "user0_value: int|typein|-1|-1|4096|1|",
        "user1_name: description|Partial echo",
        "user1_value: float|typein|1.0|0.75|1.0|0.01|",
        "user2_name: description|Partial Fourier y",
        "user2_value: float|typein|1.0|0.75|1.0|0.01|",
        "user4_name: description|ACS lines y",
        "user4_value: int|typein|24|0|128|1|",
        "user6_name: description|Readout oversampling",
        "user6_value: float|typein|2.0|1.0|4.0|0.1|",
        "nex: float|typein|1.0|1.0|16.0|1.0|",
        *PRESCRIPTION_LISTING,
    ]
)

# What a request for the shortest TE and TR, three-dimensional mode, six
# repetitions, fat saturation and a 7.5 mm user entry asks of the typed plugin.
REQUEST = {
    UIParam.TE: TEPreset.MINIMUM,
    UIParam.TR: TRPreset.MINIMUM,
    UIParam.IMAGING_MODE: ImagingMode.THREE_D,
    UIParam.NX: 6,
    UIParam.FAT_SAT: True,
    UserKey.USER0: 7.5,
}
WIRE_REQUEST = {
    "TE": TEPreset.MINIMUM,
    "TR": -1,
    "imaging_mode": 1,
    "nx": 6,
    "FatSat": "true",
    "user0_value": 7.5,
}


@pytest.fixture(scope="module")
def typed():
    return load_plugin(PLUGINS / "typed.py")


def _bound(typed, entries):
    """A plugin over the application of ``typed`` with ``entries`` as its protocol."""
    attributes = {"app": type(typed).app, "protocol": entries}
    return type("Bound", (SequencePlugin,), attributes)()


def _wire(typed, changes=None):
    """The wire values the protocol of ``typed`` starts at, with ``changes``."""
    wire = {key: p.value for key, p in typed.listing().items() if p.editable}
    wire.update(changes or {})
    return wire


def _protocol(plugin, changes=None):
    """The :class:`Protocol` the wire values of ``plugin`` stand for, with ``changes``."""
    return Protocol.from_wire(plugin.protocol, _wire(plugin, changes), SYSTEM)


def test_only_the_wire_turns_keys_into_strings(typed):
    listing = typed.listing()
    reply = typed.validate(SYSTEM, REQUEST)
    assert reply.valid, reply.info

    for keyed in (typed.protocol, listing, reply.values):
        assert all(isinstance(key, ProtocolKey) for key in keyed)

    assert format_listing(listing) == TYPED_LISTING
    assert format_validation(reply, listing) == "VALID 0.015\nINFO \n" + TYPED_VALUES
    parsed = parse_values(TYPED_VALUES, listing)
    assert parsed == reply.values
    assert all(isinstance(key, ProtocolKey) for key in parsed)


def test_the_design_calls_reply_the_wire_text_the_interpreter_parses(typed, tmp_path):
    asked = {"plugins": PLUGINS, "plugin": "typed"}
    limits = {"limits": LIMITS}

    assert call("list", **asked) == (0, "PROTOCOL\n" + TYPED_LISTING)
    assert call("validate", **asked, **limits, block=value_block(WIRE_REQUEST)) == (
        0,
        "VALID 0.015\nINFO \n" + TYPED_VALUES,
    )
    invalid = value_block({"TE": 1000, "imaging_mode": 1})
    assert call("validate", **asked, **limits, block=invalid) == (
        0,
        "INVALID\nINFO the requested TE of 1.000 ms is shorter than 2.500 ms\n"
        + TYPED_INVALID_VALUES,
    )

    store = DesignStore(tmp_path / "designs")
    status, reply = call(
        "generate", **asked, **limits, block=value_block(WIRE_REQUEST), store=store
    )
    assert status == 0, reply
    stored = store.directory(reply.split()[1]) / "resolved.protocol"
    assert stored.read_text() == TYPED_VALUES


def test_a_shipped_sequence_is_listed_in_the_wire_text_the_interpreter_parses():
    assert call("list", plugins=[], plugin="gre2d") == (
        0,
        "PROTOCOL\n" + GRE2D_LISTING,
    )


def test_a_design_identity_does_not_depend_on_the_types_of_keys_and_values(typed):
    reply = typed.validate(SYSTEM, REQUEST)
    plain = {
        key.value: getattr(value, "value", value) for key, value in reply.values.items()
    }
    assert plain["imaging_mode"] == "3d"
    assert all(type(key) is str for key in plain)
    assert design_identity("typed", LIMITS, reply.values, "x") == design_identity(
        "typed", LIMITS, plain, "x"
    )


def test_a_choice_is_a_member_of_its_enum_until_the_wire(typed):
    listing = typed.listing()
    mode = listing[UIParam.IMAGING_MODE]
    assert type(mode.value) is ImagingMode
    assert all(type(option) is ImagingMode for option in mode.options)

    request = {UIParam.IMAGING_MODE: "3d"}
    reply = typed.validate(SYSTEM, request)
    assert reply.valid, reply.info
    arguments = _protocol(typed, request).arguments
    assert type(arguments["mode"]) is ImagingMode
    assert reply.values[UIParam.IMAGING_MODE] is ImagingMode.THREE_D

    text = format_validation(reply, listing)
    assert "imaging_mode: 1\n" in text
    parsed = parse_values(text, listing)
    assert parsed[UIParam.IMAGING_MODE] is ImagingMode.THREE_D


def test_a_choice_default_is_a_member_of_the_choices_when_the_listing_is_built(typed):
    explicit = ChoiceParam("mode", ImagingMode, default=ImagingMode.THREE_D)
    listing = _bound(typed, {UIParam.IMAGING_MODE: explicit}).listing()
    assert listing[UIParam.IMAGING_MODE].value is ImagingMode.THREE_D

    foreign = ChoiceParam("mode", ImagingMode, default="4d")
    with pytest.raises(ValueError, match="imaging_mode: '4d' is not one of 2d, 3d"):
        _bound(typed, {UIParam.IMAGING_MODE: foreign}).listing()

    # The application's own default is "2d".
    unnamed = ChoiceParam("mode", SequenceType)
    with pytest.raises(ValueError, match="'2d' is not one of spin_echo, gradient_echo"):
        _bound(typed, {UIParam.IMAGING_MODE: unnamed}).listing()


@pytest.mark.parametrize("text", ["bogus", "2", "-1"])
def test_a_requested_choice_that_names_no_member_is_refused(typed, text):
    listing = typed.listing()
    with pytest.raises(ValueError, match="is not one of 2d, 3d"):
        parse_values(block([f"imaging_mode: {text}"]), listing)

    status, answer = call(
        "validate",
        plugins=PLUGINS,
        plugin="typed",
        limits=LIMITS,
        block=value_block({"imaging_mode": text}),
    )
    assert (status, answer[:5]) == (1, "ERROR")
    reply = typed.validate(SYSTEM, {UIParam.IMAGING_MODE: text})
    assert not reply.valid
    assert "is not one of 2d, 3d" in reply.info


def test_a_string_list_param_is_a_choice_param_over_an_enum_of_its_options(typed):
    with pytest.warns(DeprecationWarning, match="ChoiceParam"):
        entry = StringListParam("mode", options=("2d", "3d"), default="3d")
    assert isinstance(entry, ChoiceParam)
    assert [member.value for member in entry.choices] == ["2d", "3d"]
    assert type(entry.default) is entry.choices

    listed = _bound(typed, {UIParam.IMAGING_MODE: entry})
    mode = listed.listing()[UIParam.IMAGING_MODE]
    assert (mode.kind, mode.value, mode.options) == (
        Kind.STRINGLIST,
        "3d",
        ("2d", "3d"),
    )
    assert format_listing({UIParam.IMAGING_MODE: mode}).splitlines()[1] == (
        "imaging_mode: stringlist|1|2d|3d"
    )
    arguments = _protocol(listed, {UIParam.IMAGING_MODE: "2d"}).arguments
    assert arguments["mode"] == "2d"
    assert isinstance(arguments["mode"], str)


def test_a_protocol_holds_the_values_in_the_units_of_the_application_arguments(typed):
    wire = _wire(
        typed,
        {
            UIParam.TE: 2500,
            UIParam.IMAGING_MODE: "3d",
            UIParam.NX: 6,
            UIParam.FAT_SAT: True,
            UserKey.USER0: 7.5,
        },
    )
    protocol = Protocol.from_wire(typed.protocol, wire, SYSTEM)

    assert protocol[UIParam.TE] == 2.5e-3
    assert protocol[UIParam.TR] is None
    assert protocol[UserKey.USER0] == 7.5e-3
    assert protocol[UIParam.IMAGING_MODE] is ImagingMode.THREE_D
    assert protocol.arguments == {
        "te": 2.5e-3,
        "tr": None,
        "mode": ImagingMode.THREE_D,
        "n_repetitions": 6,
        "fat_sat": True,
        "thickness": 7.5e-3,
    }
    assert list(protocol) == list(wire)


def test_the_prescription_entries_keep_the_units_of_the_wire_and_bind_no_argument(
    typed,
):
    wire = _wire(typed, {FloatKey.FOV_OFFSET_X: 12.5})
    protocol = Protocol.from_wire(typed.protocol, wire, SYSTEM)
    assert protocol[FloatKey.FOV_OFFSET_X] == 12.5
    assert protocol[FloatKey.FOV_ROTATION_22] == 1.0
    assert protocol.to_wire()[FloatKey.FOV_OFFSET_X] == 12.5
    assert not {"fov_offset_x", FloatKey.FOV_OFFSET_X} & set(protocol.arguments)
    assert len(protocol.arguments) == 6


def test_a_callable_preset_is_a_function_of_the_scanner_limits(typed):
    shortest = {TEPreset.MINIMUM: lambda system: 250 * system.grad_raster_time}
    raster = _bound(typed, {UIParam.TE: TimeParam("te", presets=shortest)})
    protocol = Protocol.from_wire(
        raster.protocol, {UIParam.TE: TEPreset.MINIMUM}, SYSTEM
    )
    assert protocol[UIParam.TE] == pytest.approx(250 * SYSTEM.grad_raster_time)
    assert protocol.preset(UIParam.TE) is TEPreset.MINIMUM


def test_a_protocol_converts_back_to_the_wire_values_it_was_made_from(typed):
    wire = _wire(typed, {UIParam.TE: TEPreset.MINIMUM, UIParam.NX: 9})
    protocol = Protocol.from_wire(typed.protocol, wire, SYSTEM)
    assert protocol.to_wire() == wire


def test_a_float_is_carried_to_six_significant_digits_and_a_time_to_a_microsecond(
    typed,
):
    wire = _wire(typed, {UIParam.TE: 2500, UserKey.USER0: 7.1234567})
    protocol = Protocol.from_wire(typed.protocol, wire, SYSTEM)
    assert protocol.to_wire()[UserKey.USER0] == 7.12346
    assert protocol.replace({UIParam.TE: 2.5004e-3}).to_wire()[UIParam.TE] == 2500
    assert protocol.replace({UIParam.TE: 2.5006e-3}).to_wire()[UIParam.TE] == 2501


def test_a_protocol_names_the_preset_a_time_shows(typed):
    protocol = Protocol.from_wire(typed.protocol, _wire(typed), SYSTEM)
    assert protocol.preset(UIParam.TR) is TRPreset.MINIMUM
    assert protocol.preset(UIParam.TE) is None
    assert protocol.preset(UIParam.NX) is None

    shortest = Protocol.from_wire(
        typed.protocol, _wire(typed, {UIParam.TE: TEPreset.MINIMUM}), SYSTEM
    )
    assert shortest.preset(UIParam.TE) is TEPreset.MINIMUM
    assert shortest[UIParam.TE] is None


def test_a_time_given_in_seconds_shows_no_preset(typed):
    protocol = Protocol.from_wire(typed.protocol, _wire(typed), SYSTEM)
    replaced = protocol.replace({UIParam.TR: 10e-3})
    assert replaced[UIParam.TR] == 10e-3
    assert replaced.preset(UIParam.TR) is None
    assert replaced.to_wire()[UIParam.TR] == 10000
    assert protocol.preset(UIParam.TR) is TRPreset.MINIMUM


def test_replacing_values_returns_a_new_protocol_and_leaves_the_old_one(typed):
    protocol = Protocol.from_wire(typed.protocol, _wire(typed), SYSTEM)
    changed = protocol.replace(
        {UIParam.NX: 9, UIParam.IMAGING_MODE: "3d", UserKey.USER0: 0.02}
    )

    assert changed is not protocol
    assert (protocol[UIParam.NX], protocol[UIParam.IMAGING_MODE]) == (
        4,
        ImagingMode.TWO_D,
    )
    assert changed[UIParam.NX] == 9
    assert changed[UserKey.USER0] == 0.02
    assert changed[UIParam.IMAGING_MODE] is ImagingMode.THREE_D
    assert list(changed) == list(protocol)


def test_a_protocol_is_an_immutable_mapping(typed):
    protocol = Protocol.from_wire(typed.protocol, _wire(typed), SYSTEM)
    with pytest.raises(TypeError):
        protocol[UIParam.NX] = 9
    protocol.arguments["n_repetitions"] = 99
    assert protocol.arguments["n_repetitions"] == 4
    assert protocol == dict(protocol)
    assert list(protocol) == [k for k, p in typed.listing().items() if p.editable]
    assert UIParam.NX in protocol
    assert "nx" in protocol


def test_a_protocol_refuses_what_it_does_not_hold(typed):
    protocol = Protocol.from_wire(typed.protocol, _wire(typed), SYSTEM)
    with pytest.raises(ValueError, match="flip"):
        protocol.replace({FloatKey.FLIP: 10.0})
    with pytest.raises(ValueError, match="flip"):
        Protocol.from_wire(typed.protocol, {FloatKey.FLIP: 10.0}, SYSTEM)
    with pytest.raises(ValueError, match="is not one of 2d, 3d"):
        protocol.replace({UIParam.IMAGING_MODE: "4d"})
    with pytest.raises(ValueError, match="does not offer preset -3"):
        Protocol.from_wire(typed.protocol, {UIParam.TE: TEPreset.IN_PHASE}, SYSTEM)


def test_the_prescription_lines_of_a_block_name_the_entries_by_their_wire_names():
    values = {
        FloatKey.FOV_OFFSET_X: 12.5,
        FloatKey.FOV_ROTATION_11: 0.0,
        FloatKey.FOV_ROTATION_12: -1.0,
    }
    lines = format_prescription(values)
    assert lines == [
        "fov_offset_x: 12.5",
        "fov_rotation_11: 0.0",
        "fov_rotation_12: -1.0",
    ]

    found = parse_prescription("\n".join(["[Import]", "file: a.seq", *lines, "TE: 3"]))
    assert found == values
    assert all(isinstance(key, ProtocolKey) for key in found)
    with pytest.raises(ValueError, match="could not convert"):
        parse_prescription("fov_offset_x: wide")


def test_an_import_block_is_the_file_and_its_prescription_lines():
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    text = format_import(Path("/data/a.seq"), (1.0, 2.5, -3.0), rotation)
    assert text.splitlines() == [
        "[Import]",
        "file: /data/a.seq",
        "fov_offset_x: 1.0",
        "fov_offset_y: 2.5",
        "fov_offset_z: -3.0",
        "fov_rotation_11: 0.0",
        "fov_rotation_12: -1.0",
        "fov_rotation_13: 0.0",
        "fov_rotation_21: 1.0",
        "fov_rotation_22: 0.0",
        "fov_rotation_23: 0.0",
        "fov_rotation_31: 0.0",
        "fov_rotation_32: 0.0",
        "fov_rotation_33: 1.0",
        "[Import End]",
    ]
    path, offset, turned = parse_import(text)
    assert (path, offset) == (Path("/data/a.seq"), (1.0, 2.5, -3.0))
    np.testing.assert_array_equal(turned, rotation)
    assert format_import("a.seq").splitlines() == [
        "[Import]",
        "file: a.seq",
        "[Import End]",
    ]
