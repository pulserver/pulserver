from pathlib import Path

import numpy as np
import pypulseqpp as pp
import pytest
from pypulseqpp import sequences

from pulserver.design import FloatParam, ScannerSequence, TimeParam, load_plugin
from pulserver.protocol import (
    PRESCRIPTION,
    FloatKey,
    InputMode,
    Kind,
    TEPreset,
    TRPreset,
    UIParam,
    prescribed_offset,
    prescribed_rotation,
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


class StatedApp(sequences.SequenceApp):
    """Records its echo time under another name and states its scan time."""

    MAX_GRAD = 40.0
    MAX_SLEW = 150.0

    def init_sequence(self, te: float | None = None, tr: float = 10e-3) -> None:
        self.echo = 2.5e-3 if te is None else te
        self.duration = 4 * tr
        self.resolve(te=self.echo)

    def loop(self) -> None:
        raise AssertionError("the scan was played")

    def kernel(self) -> None:
        pass


class PrescannedApp(sequences.SequenceApp):
    """States no scan time: a 1 ms prescan, then three 10 ms repetitions."""

    MAX_GRAD = 40.0
    MAX_SLEW = 150.0

    def init_sequence(self, tr: float = 10e-3) -> None:
        self.tr = tr

    def prescans(self):
        return {"calibration": self.calibration}

    def calibration(self) -> None:
        if self.tr > 1.0:
            raise ValueError("the calibration cannot play a TR over 1 s")
        self.seq.add_block(pp.make_delay(1e-3))

    def loop(self) -> None:
        for _ in range(3):
            self.kernel()

    def kernel(self) -> None:
        self.seq.add_block(pp.make_delay(self.tr))


class Stated(ScannerSequence):
    app = StatedApp
    ui = {
        UIParam.TE: TimeParam("te", range_max=80000, presets={TEPreset.MINIMUM: None}),
        UIParam.TR: TimeParam("tr", range_max=5_000_000),
    }


class Prescanned(ScannerSequence):
    app = PrescannedApp
    ui = {UIParam.TR: TimeParam("tr", range_max=5_000_000)}


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


def test_an_entry_resolves_to_the_value_the_application_records():
    values = Stated().validate(SYSTEM, {"TE": TEPreset.MINIMUM}).values
    assert (values["TE"], values["TR"]) == (2500, 10000)


def test_a_stated_scan_time_is_reported_without_playing_the_scan():
    reply = Stated().validate(SYSTEM, {"TR": 20000})
    assert reply.valid, reply.info
    assert reply.duration == pytest.approx(80e-3)


def test_an_unstated_scan_time_is_that_of_the_chain_with_its_prescans():
    reply = Prescanned().validate(SYSTEM, {"TR": 10000})
    assert reply.valid, reply.info
    assert reply.duration == pytest.approx(1e-3 + 3 * 10e-3)


def test_a_design_refused_while_it_is_timed_is_invalid():
    reply = Prescanned().validate(SYSTEM, {"TR": 2_000_000})
    assert (reply.valid, reply.info) == (
        False,
        "the calibration cannot play a TR over 1 s",
    )


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


def test_the_listing_ends_with_the_prescription_as_the_identity_and_not_editable(
    tiny,
):
    listing = tiny.listing()
    assert list(listing)[-len(PRESCRIPTION) :] == list(PRESCRIPTION)
    expected = [(0.0, "mm")] * 3 + [(float(v), "") for v in np.eye(3).ravel()]
    for name, (value, unit) in zip(PRESCRIPTION, expected, strict=True):
        entry = listing[name]
        assert (entry.kind, entry.value, entry.mode, entry.unit) == (
            Kind.FLOAT,
            value,
            InputMode.OFF,
            unit,
        )


def test_the_prescription_travels_through_resolution_unchanged(gre2d):
    quarter = {"fov_rotation_11": 0.0, "fov_rotation_12": -1.0}
    quarter |= {"fov_rotation_21": 1.0, "fov_rotation_22": 0.0}
    request = {"nx": 32, "ny": 32, "fov_offset_x": 12.5, "fov_offset_z": -4.0}
    values = gre2d.validate(SYSTEM, {**request, **quarter}).values
    assert prescribed_offset(values) == pytest.approx((0.0125, 0.0, -0.004))
    np.testing.assert_allclose(
        prescribed_rotation(values), [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12
    )


def test_a_request_whose_rotation_is_not_orthonormal_is_invalid(tiny):
    reply = tiny.validate(SYSTEM, {"fov_rotation_33": 1.01})
    assert not reply.valid
    assert "not orthonormal" in reply.info


def test_a_scanner_sequence_may_not_bind_a_prescription_entry(tiny):
    with pytest.raises(ValueError, match="prescription entries"):
        type(
            "Moved",
            (ScannerSequence,),
            {"app": type(tiny).app, "ui": {FloatKey.FOV_OFFSET_X: FloatParam("te")}},
        )
