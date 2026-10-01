"""The RF a sequence states it plays, so a scanner can cost it while prescribing."""

import numpy as np
import pytest

from pulserver.protocol import RfPulse, format_pulses, parse_pulses

_SHAPE = np.abs(np.sinc(np.linspace(-2, 2, 64)))
SINC = tuple((_SHAPE / _SHAPE.max()).tolist())


def excitation(**kwargs):
    return RfPulse(
        envelope=SINC,
        duration_us=2000.0,
        flip_deg=30.0,
        follows="flip_angle",
        bandwidth_hz=1250.0,
        **kwargs,
    )


def test_a_pulse_states_what_its_angle_follows_and_comes_back_the_same():
    """The wire carries six significant digits, which is a float32's worth."""
    pulses = [
        excitation(),
        RfPulse(envelope=(1.0, 1.0), duration_us=500.0, flip_deg=180.0),
    ]
    read = parse_pulses(format_pulses(pulses))
    assert len(read) == len(pulses)
    for got, sent in zip(read, pulses, strict=True):
        assert got.follows == sent.follows
        assert got.factor == pytest.approx(sent.factor)
        assert got.flip_deg == pytest.approx(sent.flip_deg)
        assert got.duration_us == pytest.approx(sent.duration_us)
        assert got.bandwidth_hz == pytest.approx(sent.bandwidth_hz)
        # Six significant digits against a peak of one: a part in 1e5 of the
        # peak bounds every sample, large or small.
        assert got.envelope == pytest.approx(sent.envelope, abs=1e-5)


def test_a_pulse_the_operator_does_not_move_follows_nothing():
    fixed = RfPulse(envelope=(1.0,), duration_us=500.0, flip_deg=180.0)
    assert fixed.follows == ""
    assert parse_pulses(format_pulses([fixed]))[0].follows == ""


def test_a_pulse_may_follow_a_control_scaled():
    """A refocusing at four fifths of the excitation, an inversion at twice it."""
    scaled = excitation(factor=0.8)
    read = parse_pulses(format_pulses([scaled]))[0]
    assert read.follows == "flip_angle"
    assert read.factor == pytest.approx(0.8)


def test_text_without_a_block_states_no_pulses():
    """A sequence the scanner is not given the RF of in advance."""
    assert parse_pulses("[Protocol]\nTE: 8000\n[Protocol End]\n") == []


def test_an_envelope_shorter_than_it_says_is_refused():
    text = "[RfPulses]\nflip_angle 1 30 2000 0 1 8 1 1 1\n[RfPulses End]\n"
    with pytest.raises(ValueError, match="states 8 samples"):
        parse_pulses(text)


def test_a_line_too_short_to_be_a_pulse_is_refused():
    with pytest.raises(ValueError, match="at least seven values"):
        parse_pulses("[RfPulses]\nflip_angle 1 30\n[RfPulses End]\n")


def test_the_envelope_is_normalised_to_a_peak_of_one():
    """Every statistic a scanner costs from the shape is read at unit peak."""
    read = parse_pulses(format_pulses([excitation()]))[0]
    assert max(read.envelope) == pytest.approx(1.0, abs=1e-6)


def test_the_pulses_of_a_design_reach_the_listing(tmp_path):
    """The whole point: a scanner is told the RF without asking for a design."""
    from pulserver.host._service import PULSES_FILE, list_protocol
    from pulserver.host._store import DesignStore

    plugins = tmp_path / "plugins"
    plugins.mkdir()
    (plugins / "linked.py").write_text(
        "from pypulseqpp.sequences.sequence.gre2D_sequence import Gre2DApp\n"
        "from pulserver.design import FloatParam, ScannerSequence\n"
        "from pulserver.protocol import UIParam\n"
        "class Linked(ScannerSequence):\n"
        "    app = Gre2DApp\n"
        "    ui = {UIParam.FLIP: FloatParam('flip_angle_deg', unit='deg',\n"
        "                                   range_min=1.0, range_max=180.0)}\n"
        "    follows = {'excitation': UIParam.FLIP}\n"
        "PLUGIN = Linked()\n"
    )
    store = DesignStore(tmp_path / "store")
    staged = store.stage()
    (staged / PULSES_FILE).write_text(format_pulses([excitation(count=144)]))
    store.commit("identity-1", staged, {"plugin": "linked"})

    read = parse_pulses(list_protocol(plugins, "linked", store))
    assert len(read) == 1
    assert read[0].follows == "flip_angle"
    assert read[0].count == 144


def test_a_sequence_no_design_has_been_made_of_states_no_rf(tmp_path):
    """Nothing is designed to answer a listing; the scanner costs it later."""
    from pulserver.host._service import list_protocol
    from pulserver.host._store import DesignStore

    plugins = tmp_path / "plugins"
    plugins.mkdir()
    (plugins / "bare.py").write_text(
        "from pypulseqpp.sequences.sequence.gre2D_sequence import Gre2DApp\n"
        "from pulserver.design import ScannerSequence\n"
        "class Bare(ScannerSequence):\n"
        "    app = Gre2DApp\n"
        "    ui = {}\n"
        "PLUGIN = Bare()\n"
    )
    assert (
        parse_pulses(list_protocol(plugins, "bare", DesignStore(tmp_path / "s"))) == []
    )
