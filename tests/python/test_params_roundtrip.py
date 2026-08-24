from __future__ import annotations

import warnings

import pulserver
import pytest
from pulserver.design import (
    BoolParam,
    DropdownFloatParam,
    DropdownIntParam,
    ImagingMode,
    PreparationType,
    SequenceType,
    StringListParam,
    TriggerType,
    TypeinFloatParam,
    TypeinIntParam,
    UIParam,
    dict_to_protocol,
    make_enum_param,
    protocol_to_dict,
)
from pulserver.design._params import (
    BoolKey,
    EnumKey,
    FloatKey,
    InputMode,
    IntKey,
    ParamKind,
    Validate,
    enum_options,
    expected_param_kind,
    validate_protocol_entry,
)


def test_protocol_roundtrip(sample_protocol):
    encoded = protocol_to_dict(sample_protocol)
    decoded = dict_to_protocol(encoded)

    assert set(decoded.keys()) == {UIParam.TE.value, UIParam.TR.value}
    assert decoded[UIParam.TE.value].value == pytest.approx(
        sample_protocol[UIParam.TE].value
    )
    assert decoded[UIParam.TR.value].value == pytest.approx(
        sample_protocol[UIParam.TR].value
    )


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
        UIParam.FLIP: TypeinFloatParam(
            value=10.0, min=0.0, max=180.0, incr=1.0, unit="deg"
        ),
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
            "choice": StringListParam(
                options=["A", "B"], value="B", validate=Validate.SEARCH
            ),
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


def test_the_root_holds_the_namespaces_and_nothing_else():
    """One name per role at the root; everything else is inside one of them."""
    assert pulserver.__all__ == ["app", "design", "mrd", "pypulseq", "recon"]
    assert sorted(pulserver.__dir__()) == pulserver.__all__
    for name in pulserver.__all__:
        assert getattr(pulserver, name).__name__ == f"pulserver.{name}"
    for moved in ("SequencePlugin", "UIParam", "TypeinFloatParam", "ReconPlugin"):
        assert not hasattr(pulserver, moved), moved


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


# ----------------------------------------------------------------------
# main() arguments from a protocol
# ----------------------------------------------------------------------


def test_main_kwargs_fills_only_what_main_declares():
    """A plugin's signature decides; the protocol only supplies."""
    from pulserver.design import main_kwargs

    def main(*, system=None, te=8e-3, n_x=128, n_acs=24): ...

    protocol = protocol_to_dict(
        {
            UIParam.TE: TypeinFloatParam(value=5.0, unit="ms"),
            UIParam.NX: TypeinIntParam(value=64),
            UIParam.NY: TypeinIntParam(value=96),
            UIParam.FLIP: TypeinFloatParam(value=12.0, unit="deg"),
        }
    )
    kwargs = main_kwargs(main, "the-system", protocol)

    assert kwargs == {"system": "the-system", "te": 0.005, "n_x": 64}


def test_main_kwargs_lets_the_plugin_have_the_last_word():
    from pulserver.design import main_kwargs

    def main(*, n_x=128, n_acs=24): ...

    protocol = protocol_to_dict({UIParam.NX: TypeinIntParam(value=64)})
    assert main_kwargs(main, None, protocol, n_x=32, n_acs=8) == {"n_x": 32, "n_acs": 8}


def test_main_kwargs_refuses_an_override_main_cannot_take():
    from pulserver.design import main_kwargs

    def main(*, n_x=128): ...

    with pytest.raises(TypeError, match="names no parameter"):
        main_kwargs(main, None, {}, n_slices=3)


def test_a_timing_preset_asks_for_whatever_the_sequence_can_reach():
    """Both presets are negative, and `None` is what a readout module reads."""
    from pulserver.design import TEPreset, TRPreset, main_kwargs

    def main(*, te=8e-3, tr=250e-3): ...

    protocol = protocol_to_dict(
        {
            UIParam.TE: TypeinFloatParam(value=float(TEPreset.MINIMUM), unit="ms"),
            UIParam.TR: TypeinFloatParam(value=float(TRPreset.MINIMUM), unit="ms"),
        }
    )
    assert main_kwargs(main, None, protocol) == {"te": None, "tr": None}


def test_a_preset_survives_a_dropdown_round_trip():
    from pulserver.design import DropdownFloatParam, TEPreset

    te = DropdownFloatParam(
        value=8.0, min=1.0, max=80.0, unit="ms", options=[TEPreset.MINIMUM, 5.0, 8.0]
    )
    wire = protocol_to_dict({UIParam.TE: te})
    assert wire["TE"]["options"][0] == float(TEPreset.MINIMUM)
    assert dict_to_protocol(wire)["TE"].options[0] == float(TEPreset.MINIMUM)


# ----------------------------------------------------------------------
# Writing, in the form the destination reads
# ----------------------------------------------------------------------


def test_the_scanner_form_is_binary_and_unchecked(tmp_path):
    """Everything skipped here the interpreter does at predownload."""
    import pulserver.pypulseq as pp
    from pulserver.design import write_sequence

    seq = pp.Sequence()
    seq.add_block(pp.make_trapezoid(channel="x", area=1000, system=pp.Opts()))

    path = tmp_path / "scanner.seq"
    assert write_sequence(seq, str(path), offline=False) is None
    assert path.read_bytes().startswith(b"\x01pulseq")


def test_the_offline_form_is_text_and_signed(tmp_path):
    import pulserver.pypulseq as pp
    from pulserver.design import write_sequence

    seq = pp.Sequence()
    seq.add_block(pp.make_trapezoid(channel="x", area=1000, system=pp.Opts()))

    path = tmp_path / "offline.seq"
    signature = write_sequence(seq, str(path), offline=True)
    text = path.read_text()

    assert signature
    assert text.startswith("# Pulseq sequence file")
    assert "[SIGNATURE]" in text


def test_the_offline_form_says_when_a_gradient_is_beyond_the_system(tmp_path):
    """The check nothing downstream will run, because nothing downstream is."""
    import pulserver.pypulseq as pp
    from pulserver.design import write_sequence

    modest = pp.Opts(max_grad=10, grad_unit="mT/m", max_slew=100, slew_unit="T/m/s")
    seq = pp.Sequence(modest)
    seq.add_block(pp.make_trapezoid(channel="x", area=8000, system=pp.Opts()))

    assert not seq.check_hardware_limits()[0]
    with pytest.warns(UserWarning, match="exceeds the system limit"):
        write_sequence(seq, str(tmp_path / "loud.seq"), offline=True)

    # The same file goes to the scanner without a word: it checks it itself.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write_sequence(seq, str(tmp_path / "quiet.seq"), offline=False)
