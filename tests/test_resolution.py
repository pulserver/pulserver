from pathlib import Path

import numpy as np
import pypulseqpp as pp
import pytest

from pulserver.design import FloatParam, ScannerSequence, load_plugin
from pulserver.protocol import (
    PRESCRIPTION,
    FloatKey,
    InputMode,
    Kind,
    TEPreset,
    TRPreset,
    prescribed_offset,
)

PLUGINS = Path(__file__).parent / "plugins"
SYSTEM = pp.Opts(max_grad=40.0, grad_unit="mT/m", max_slew=150.0, slew_unit="T/m/s")

PRESCRIPTIONS = [
    ("tiny", {}),
    ("tiny", {"TE": TEPreset.MINIMUM}),
    ("tiny", {"TE": 12500, "nx": 3}),
    ("gre2d", {}),
    ("gre2d", {"TE": TEPreset.MINIMUM, "TR": TRPreset.MINIMUM}),
    ("gre2d", {"TE": 5000, "TR": 30000, "bandwidth": 130e3}),
    ("gre2d", {"fov": 180.0, "nx": 96}),
]


@pytest.fixture(scope="module")
def tiny():
    return load_plugin(PLUGINS / "tiny.py")


@pytest.fixture(scope="module")
def gre2d():
    return load_plugin(PLUGINS / "gre2d.py")


def test_a_time_is_listed_in_integer_microseconds(tiny):
    te = tiny.listing()["TE"]
    assert (te.kind, te.value, te.unit, te.mode) == (
        Kind.INT,
        8000,
        "us",
        InputMode.DROPDOWN,
    )
    assert te.options == (TEPreset.MINIMUM,)


def test_a_minimum_request_resolves_to_the_designed_value(tiny, gre2d):
    assert tiny.validate(SYSTEM, {"TE": TEPreset.MINIMUM}).values["TE"] == 2500
    reply = gre2d.validate(SYSTEM, {"TE": TEPreset.MINIMUM})
    assert reply.valid, reply.info
    assert isinstance(reply.values["TE"], int)
    assert 0 < reply.values["TE"] < 8000


def test_an_infeasible_protocol_is_invalid_with_the_design_error_as_info(tiny, gre2d):
    reply = tiny.validate(SYSTEM, {"TE": 1000})
    assert not reply.valid
    assert "shorter than" in reply.info
    assert reply.values["TE"] == 1000
    assert "TR" in gre2d.validate(SYSTEM, {"TR": 1000}).info


def test_a_preset_the_entry_does_not_offer_is_invalid(tiny):
    reply = tiny.validate(SYSTEM, {"TE": TEPreset.IN_PHASE})
    assert not reply.valid
    assert "preset" in reply.info


def test_a_request_for_an_undeclared_entry_is_refused(tiny):
    with pytest.raises(ValueError, match="flip"):
        tiny.validate(SYSTEM, {"flip": 10.0})


@pytest.mark.parametrize(("plugin", "prescription"), PRESCRIPTIONS)
def test_resolving_a_resolved_protocol_changes_nothing(plugin, prescription, request):
    sequence = request.getfixturevalue(plugin)
    first = sequence.validate(SYSTEM, prescription)
    assert first.valid, first.info
    assert sequence.validate(SYSTEM, first.values) == first


@pytest.mark.parametrize(("plugin", "prescription"), PRESCRIPTIONS)
def test_a_resolved_protocol_survives_cv_storage(plugin, prescription, request):
    sequence = request.getfixturevalue(plugin)
    first = sequence.validate(SYSTEM, prescription)
    stored = {
        name: float(np.float32(value)) if isinstance(value, float) else value
        for name, value in first.values.items()
    }
    second = sequence.validate(SYSTEM, stored)
    assert second.valid, second.info
    assert {n: np.float32(v) for n, v in second.values.items()} == {
        n: np.float32(v) for n, v in first.values.items()
    }


def test_generate_writes_the_resolved_design(tiny, tmp_path):
    validation, paths = tiny.generate(SYSTEM, {"TE": TEPreset.MINIMUM}, tmp_path)
    assert validation.values["TE"] == 2500
    assert [Path(p).name for p in paths] == ["sequence.seq"]
    seq = pp.Sequence()
    seq.read(paths[0])
    assert seq.duration()[0] == pytest.approx(4 * 2.5e-3)


def test_nothing_is_written_for_an_invalid_request(tiny, tmp_path):
    validation, paths = tiny.generate(SYSTEM, {"TE": 1000}, tmp_path)
    assert not validation.valid
    assert paths == []
    assert list(tmp_path.iterdir()) == []


def test_a_plugin_file_must_define_exactly_one_scanner_sequence(tmp_path):
    empty = tmp_path / "empty.py"
    empty.write_text("VALUE = 1\n")
    with pytest.raises(ValueError, match="defines 0"):
        load_plugin(empty)


def test_the_listing_ends_with_the_prescription_at_zero_and_not_editable(tiny):
    listing = tiny.listing()
    assert list(listing)[-3:] == list(PRESCRIPTION)
    for name in PRESCRIPTION:
        entry = listing[name]
        assert (entry.kind, entry.value, entry.mode, entry.unit) == (
            Kind.FLOAT,
            0.0,
            InputMode.OFF,
            "mm",
        )


def test_the_prescription_travels_through_resolution_unchanged(gre2d):
    request = {"nx": 32, "ny": 32, "fov_offset_x": 12.5, "fov_offset_z": -4.0}
    values = gre2d.validate(SYSTEM, request).values
    assert prescribed_offset(values) == pytest.approx((0.0125, 0.0, -0.004))


def test_a_scanner_sequence_may_not_bind_a_prescription_entry(tiny):
    with pytest.raises(ValueError, match="prescription entries"):
        type(
            "Moved",
            (ScannerSequence,),
            {"app": type(tiny).app, "ui": {FloatKey.FOV_OFFSET_X: FloatParam("te")}},
        )
