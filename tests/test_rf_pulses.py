"""The RF a sequence states it plays, so a scanner can cost it while prescribing."""

import pytest

from pulserver.protocol import RfPulse, format_pulses, parse_pulses

EXCITATION = RfPulse(flip_deg=90.0, use="excitation")
REFOCUSING = RfPulse(
    flip_deg=120.0, follows="flip", factor=0.8, use="refocusing", count=16
)


def test_a_pulse_states_what_its_angle_follows_and_comes_back_the_same():
    design, read = parse_pulses(format_pulses([EXCITATION, REFOCUSING], "abc123"))
    assert design == "abc123"
    assert read == [EXCITATION, REFOCUSING]


def test_a_pulse_the_operator_does_not_move_follows_nothing():
    assert EXCITATION.follows == ""
    assert parse_pulses(format_pulses([EXCITATION]))[1][0].follows == ""


def test_a_pulse_may_follow_a_knob_scaled():
    """A refocusing at four fifths of the excitation, an inversion at twice it."""
    _, read = parse_pulses(format_pulses([REFOCUSING]))
    assert read[0].follows == "flip"
    assert read[0].factor == pytest.approx(0.8)


def test_a_pulse_says_how_many_times_it_is_played():
    """A cost is per occurrence, so the count is part of the statement."""
    _, read = parse_pulses(format_pulses([REFOCUSING]))
    assert read[0].count == 16


def test_text_without_a_block_states_no_pulses():
    assert parse_pulses("[Protocol]\nTE: 8000\n[Protocol End]\n") == ("", [])


def test_a_line_too_short_to_be_a_pulse_is_refused():
    with pytest.raises(ValueError, match="five values"):
        parse_pulses("[RfPulses]\nflip 1 30\n[RfPulses End]\n")


def test_the_shape_does_not_travel():
    """It is in the design's cache, which the scanner reads for itself."""
    assert not hasattr(EXCITATION, "envelope")
    assert len(format_pulses([EXCITATION, REFOCUSING], "abc123")) < 200


def test_a_design_names_where_its_shapes_are():
    design, _ = parse_pulses(format_pulses([EXCITATION], "d7"))
    assert design == "d7"


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
    design = store.commit("identity-1", staged, {"plugin": "linked"})
    (store.directory(design) / PULSES_FILE).write_text(
        format_pulses([REFOCUSING], design)
    )

    named, read = parse_pulses(list_protocol(plugins, "linked", store))
    assert named == design
    assert [p.follows for p in read] == ["flip"]
    assert read[0].count == 16


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
    assert parse_pulses(
        list_protocol(plugins, "bare", DesignStore(tmp_path / "s"))
    ) == ("", [])
