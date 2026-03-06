"""Tests for the sequence plugin contract, typed params, and GRE plugin."""

from __future__ import annotations

import pypulseq as pp
import pytest

from pulserver.sequences import (
    BoolParam,
    Description,
    FloatParam,
    IntParam,
    Protocol,
    ProtocolValue,
    PulseqSequence,
    StringListParam,
    UIParam,
    Validate,
    dict_to_param,
    dict_to_protocol,
    param_to_dict,
    protocol_to_dict,
)


# ── fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def default_opts() -> pp.Opts:
    """Default pypulseq hardware opts."""
    return pp.Opts()


# ── UIParam ─────────────────────────────────────────────────────────────────


class TestUIParam:
    def test_is_str(self):
        assert isinstance(UIParam.TE, str)
        assert UIParam.TE == "TE"

    def test_user_cv(self):
        assert UIParam.user(0) == "User0"
        assert UIParam.user(47) == "User47"

    def test_usable_as_dict_key(self):
        d: dict[UIParam | str, int] = {UIParam.TE: 1, "custom": 2}
        assert d[UIParam.TE] == 1
        assert d["TE"] == 1  # StrEnum interop

    def test_membership(self):
        assert "TE" in UIParam.__members__.values()


# ── Validate ────────────────────────────────────────────────────────────────


class TestValidate:
    def test_values(self):
        assert Validate.SEARCH == "search"
        assert Validate.CLIP == "clip"
        assert Validate.NONE == "none"


# ── Param dataclasses ──────────────────────────────────────────────────────


class TestParamDataclasses:
    def test_float_param_defaults(self):
        p = FloatParam(5.0, 1.0, 100.0, 0.1)
        assert p.validate == Validate.SEARCH
        assert p.type == "float"
        assert p.unit == ""

    def test_int_param(self):
        p = IntParam(256, 32, 512, 32, unit="mm")
        assert p.value == 256
        assert p.unit == "mm"

    def test_bool_param(self):
        p = BoolParam(True)
        assert p.type == "bool"

    def test_stringlist_param(self):
        p = StringListParam(["Linear", "Centric"], 0)
        assert p.options[p.index] == "Linear"
        assert p.type == "stringlist"

    def test_description(self):
        p = Description("Example GRE")
        assert p.text == "Example GRE"

    def test_validate_override(self):
        p = FloatParam(5.0, 1.0, 100.0, 0.1, validate=Validate.NONE)
        assert p.validate == Validate.NONE


# ── Serialization roundtrip ────────────────────────────────────────────────


class TestSerialization:
    @pytest.mark.parametrize(
        "param",
        [
            FloatParam(5.0, 1.0, 100.0, 0.1, unit="ms"),
            IntParam(256, 32, 512, 32),
            BoolParam(False),
            StringListParam(["A", "B", "C"], 1),
            Description("test"),
        ],
    )
    def test_roundtrip_single(self, param: ProtocolValue):
        d = param_to_dict(param)
        assert isinstance(d, dict)
        restored = dict_to_param(d)
        assert restored == param

    def test_roundtrip_protocol(self):
        protocol: Protocol = {
            UIParam.TE: FloatParam(5.0, 1.0, 100.0, 0.1, unit="ms"),
            UIParam.FOV: IntParam(256, 64, 512, 1, unit="mm"),
            UIParam.RF_SPOILING: BoolParam(True),
            "Custom": Description("hello"),
        }
        d = protocol_to_dict(protocol)
        assert all(isinstance(v, dict) for v in d.values())
        restored = dict_to_protocol(d)
        # Keys become plain strings after roundtrip (expected — StrEnum is str)
        for key in protocol:
            assert restored[str(key)] == protocol[key]

    def test_dict_to_param_does_not_mutate_input(self):
        d = {"type": "float", "value": 1.0, "min": 0.0, "max": 10.0, "incr": 0.5}
        original = dict(d)
        dict_to_param(d)
        assert d == original


# ── ABC enforcement ────────────────────────────────────────────────────────


class TestABCEnforcement:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            PulseqSequence()

    def test_incomplete_subclass_fails(self):
        class BadPlugin(PulseqSequence):
            def get_default_protocol(self, opts):
                return {}

        with pytest.raises(TypeError):
            BadPlugin()

    def test_complete_subclass_ok(self):
        class MinimalPlugin(PulseqSequence):
            def get_default_protocol(self, opts):
                return {}

            def validate_protocol(self, opts, protocol):
                return {"valid": True, "duration": None, "info": None}

            def make_sequence(self, opts, protocol):
                return ""

        p = MinimalPlugin()
        assert p.validate_protocol({}, {})["valid"] is True


# ── GRE2D plugin ───────────────────────────────────────────────────────────


class TestGRE2D:
    @pytest.fixture
    def gre(self):
        from pulserver.sequences.gre_2d import GRE2D

        return GRE2D()

    @pytest.fixture
    def protocol(self, gre, default_opts):
        return gre.get_default_protocol(default_opts)

    def test_is_pulseq_sequence(self, gre):
        assert isinstance(gre, PulseqSequence)

    def test_default_protocol_keys(self, protocol):
        assert UIParam.TE in protocol
        assert UIParam.TR in protocol
        assert UIParam.FOV in protocol
        assert UIParam.FLIP_ANGLE in protocol

    def test_default_protocol_types(self, protocol):
        assert isinstance(protocol[UIParam.TE], FloatParam)
        assert isinstance(protocol[UIParam.FOV], IntParam)
        assert isinstance(protocol[UIParam.RF_SPOILING], BoolParam)

    def test_validate_default_is_valid(self, gre, default_opts, protocol):
        result = gre.validate_protocol(default_opts, protocol)
        assert result["valid"] is True
        assert result["duration"] is not None
        assert result["duration"] > 0

    def test_validate_te_too_short(self, gre, default_opts, protocol):
        protocol[UIParam.TE] = FloatParam(0.01, 0.01, 100.0, 0.1, unit="ms")
        result = gre.validate_protocol(default_opts, protocol)
        assert result["valid"] is False
        assert "TE" in result["info"]

    def test_validate_tr_too_short(self, gre, default_opts, protocol):
        protocol[UIParam.TR] = FloatParam(0.1, 0.1, 1000.0, 0.1, unit="ms")
        result = gre.validate_protocol(default_opts, protocol)
        assert result["valid"] is False

    def test_module_level_functions(self):
        from pulserver.sequences.gre_2d import (
            get_default_protocol,
            make_sequence,
            validate_protocol,
        )

        assert callable(get_default_protocol)
        assert callable(validate_protocol)
        assert callable(make_sequence)

    def test_protocol_serialization_roundtrip(self, protocol):
        d = protocol_to_dict(protocol)
        restored = dict_to_protocol(d)
        for key in protocol:
            assert restored[str(key)] == protocol[key]
