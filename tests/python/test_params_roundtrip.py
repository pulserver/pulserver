from __future__ import annotations

import pulserver
import pytest
from pulserver import (
    BoolKey,
    BoolParam,
    DropdownFloatParam,
    DropdownIntParam,
    EnumKey,
    FloatKey,
    ImagingMode,
    InputMode,
    IntKey,
    ParamKind,
    PreparationType,
    SequenceType,
    StringListParam,
    TriggerType,
    TypeinFloatParam,
    TypeinIntParam,
    UIParam,
    Validate,
    dict_to_protocol,
    enum_options,
    expected_param_kind,
    make_enum_param,
    protocol_to_dict,
    validate_protocol_entry,
)


def test_protocol_roundtrip(sample_protocol):
    encoded = protocol_to_dict(sample_protocol)
    decoded = dict_to_protocol(encoded)

    assert set(decoded.keys()) == {UIParam.TE.value, UIParam.TR.value}
    assert decoded[UIParam.TE.value].value == pytest.approx(sample_protocol[UIParam.TE].value)
    assert decoded[UIParam.TR.value].value == pytest.approx(sample_protocol[UIParam.TR].value)


def test_roundtrip_preserves_units(sample_protocol):
    decoded = dict_to_protocol(protocol_to_dict(sample_protocol))
    assert decoded[UIParam.TE.value].unit == "ms"
    assert decoded[UIParam.TR.value].unit == "ms"


def test_numeric_params_default_to_no_auto_validation():
    fp = TypeinFloatParam(value=1.0, min=0.0, max=2.0, incr=0.1)
    ip = TypeinIntParam(value=1, min=0, max=2, incr=1)

    assert fp.validate == Validate.NONE
    assert ip.validate == Validate.NONE


def test_dict_to_protocol_keeps_keys_as_is():
    encoded = {
        UIParam.FLIP.value: {
            "type": "float",
            "value": 10.0,
            "min": 0.0,
            "max": 180.0,
            "incr": 1.0,
            "unit": "deg",
            "validate": "none",
        },
        UIParam.user_value(1): {
            "type": "float",
            "value": 2.0,
            "min": 0.0,
            "max": 5.0,
            "incr": 0.1,
            "unit": "",
            "validate": "none",
        },
    }

    decoded = dict_to_protocol(encoded)

    assert UIParam.FLIP.value in decoded
    assert UIParam.user_value(1) in decoded


def test_protocol_to_dict_emits_keys_as_given():
    protocol = {
        UIParam.FLIP: TypeinFloatParam(value=10.0, min=0.0, max=180.0, incr=1.0, unit="deg"),
        UIParam.user_value(1): TypeinFloatParam(value=2.0, min=0.0, max=5.0, incr=0.1),
    }

    encoded = protocol_to_dict(protocol)

    assert UIParam.FLIP.value in encoded
    assert UIParam.user_value(1) in encoded


def test_float_ui_metadata_roundtrip():
    protocol = {
        UIParam.TE: DropdownFloatParam(
            value=12.0,
            min=5.0,
            max=60.0,
            incr=1.0,
            unit="ms",
            options=[8.0, 12.0, 16.0],
        )
    }

    decoded = dict_to_protocol(protocol_to_dict(protocol))
    te = decoded[UIParam.TE.value]
    assert te.mode == InputMode.DROPDOWN
    assert te.options == [8.0, 12.0, 16.0]
    assert te.num_entries == 4


def test_int_ui_num_entries_semantics():
    typein = TypeinIntParam(value=1, min=1, max=8, incr=1)
    dropdown = DropdownIntParam(value=1, min=1, max=8, incr=1, options=[1, 2, 4, 8])

    assert typein.num_entries == 1
    assert dropdown.num_entries == 5


def test_dropdown_requires_between_one_and_five_options():
    # Empty options are now inferred from bounds; invalid bounds still error.
    with pytest.raises(ValueError):
        DropdownIntParam(value=1, min=8, max=1, incr=1, options=[])

    with pytest.raises(ValueError):
        DropdownFloatParam(
            value=1.0,
            min=0.0,
            max=10.0,
            incr=1.0,
            options=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )


def test_value_only_numeric_params_default_to_typein_and_wire_bounds():
    fp = TypeinFloatParam(value=42.0)
    ip = TypeinIntParam(value=7)

    assert fp.mode == InputMode.TYPEIN
    assert fp.num_entries == 1
    assert fp.min == pytest.approx(0.0)
    assert fp.max == pytest.approx(float("inf"))
    assert fp.incr == pytest.approx(1.0)

    assert ip.mode == InputMode.TYPEIN
    assert ip.num_entries == 1
    assert ip.min == 0
    assert ip.max > ip.value
    assert ip.incr == 1


def test_dropdown_infers_options_from_bounds_when_missing():
    ip = DropdownIntParam(value=1, min=1, max=16, incr=3)
    fp = DropdownFloatParam(value=1.0, min=0.5, max=3.0, incr=0.5)

    assert ip.options == [1, 4, 7, 10]
    assert ip.num_entries == 5
    assert fp.options == [0.5, 1.0, 1.5, 2.0]
    assert fp.num_entries == 5


def test_bool_and_stringlist_support_validate_field():
    encoded = protocol_to_dict(
        {
            "flag": BoolParam(value=True, validate=Validate.NONE),
            "choice": StringListParam(options=["A", "B"], value="B", validate=Validate.SEARCH),
        }
    )
    decoded = dict_to_protocol(encoded)
    assert decoded["flag"].validate == Validate.NONE
    assert decoded["choice"].validate == Validate.SEARCH
    assert decoded["choice"].value == "B"
    assert decoded["choice"].index == 1


def test_dict_to_protocol_recovers_specialized_numeric_classes():
    decoded = dict_to_protocol(
        {
            UIParam.TE.value: {
                "type": "float",
                "value": 12.0,
                "min": 5.0,
                "max": 60.0,
                "incr": 1.0,
                "mode": "dropdown",
                "options": [8.0, 12.0, 16.0],
                "validate": "none",
                "unit": "ms",
            },
            UIParam.TR.value: {
                "type": "float",
                "value": 500.0,
                "validate": "none",
                "unit": "ms",
            },
            UIParam.NX.value: {
                "type": "int",
                "value": 128,
                "min": 64,
                "max": 512,
                "incr": 64,
                "mode": "typein",
                "validate": "none",
                "unit": "",
            },
        }
    )

    assert isinstance(decoded[UIParam.TE.value], DropdownFloatParam)
    assert isinstance(decoded[UIParam.TR.value], TypeinFloatParam)
    assert isinstance(decoded[UIParam.NX.value], TypeinIntParam)


def test_expected_param_kind_maps_core_keys():
    assert expected_param_kind(UIParam.NX) == ParamKind.INT
    assert expected_param_kind(UIParam.TE) == ParamKind.FLOAT
    assert expected_param_kind(UIParam.ENABLE_SATURATION_UI) == ParamKind.BOOL
    assert expected_param_kind(UIParam.SEQUENCE_TYPE) == ParamKind.STRINGLIST
    assert expected_param_kind(UIParam.user_value(1)) == ParamKind.FLOAT
    assert expected_param_kind(UIParam.user_enabled(1)) == ParamKind.BOOL


def test_uiparam_is_convenience_namespace_over_typed_keys():
    assert UIParam.TE is FloatKey.TE
    assert UIParam.NX is IntKey.NX
    assert UIParam.ENABLE_SATURATION_UI is BoolKey.ENABLE_SATURATION_UI
    assert UIParam.SEQUENCE_TYPE is EnumKey.SEQUENCE_TYPE


def test_enum_backed_controls_build_stringlist_params():
    seq = make_enum_param(UIParam.SEQUENCE_TYPE, SequenceType.SPIN_ECHO)
    imode = make_enum_param(UIParam.IMAGING_MODE, ImagingMode.THREE_D)
    prep = make_enum_param(UIParam.PREPARATION_TYPE, PreparationType.T2_PREP)
    trig = make_enum_param(UIParam.TRIGGER_TYPE, TriggerType.ECG)

    assert seq.options == enum_options(UIParam.SEQUENCE_TYPE)
    assert seq.index == 0
    assert seq.value == "spin_echo"
    assert imode.options == ["2d", "3d"]
    assert imode.index == 1
    assert imode.value == "3d"
    assert prep.options == ["inversion", "t2_prep"]
    assert prep.index == 1
    assert prep.value == "t2_prep"
    assert trig.options == ["none", "physio1", "physio2"]
    assert trig.index == 2
    assert trig.value == "physio2"


def test_stringlist_param_can_resolve_value_from_index():
    param = StringListParam(options=["none", "physio1", "physio2"], index=1)
    assert param.value == "physio1"
    assert param.index == 1


def test_protocol_validation_rejects_invalid_key_value_pairs():
    with pytest.raises(TypeError):
        validate_protocol_entry(UIParam.TE, TypeinIntParam(value=10))

    with pytest.raises(TypeError):
        protocol_to_dict({UIParam.SEQUENCE_TYPE: BoolParam(value=True)})


def test_top_level_package_re_exports_core_api():
    assert pulserver.SequencePlugin is not None
    assert pulserver.UIParam.TE == UIParam.TE
    assert pulserver.TypeinFloatParam is TypeinFloatParam
    assert not hasattr(pulserver, "FloatParam")
    assert not hasattr(pulserver, "IntParam")


def test_typed_key_sets_are_disjoint():
    key_sets = [
        {member.value for member in FloatKey},
        {member.value for member in IntKey},
        {member.value for member in BoolKey},
        {member.value for member in EnumKey},
    ]
    for idx, left in enumerate(key_sets):
        for right in key_sets[idx + 1 :]:
            assert left.isdisjoint(right)
