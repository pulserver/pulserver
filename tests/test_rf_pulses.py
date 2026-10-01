"""The RF a sequence states it plays, so a scanner can cost it while prescribing."""

import pytest

from pulserver.protocol import RfPulse, format_pulses, parse_pulses

EXCITATION = RfPulse(definition=0, flip_deg=90.0)
REFOCUSING = RfPulse(definition=1, flip_deg=120.0, follows="flip", factor=0.8)


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


def test_a_pulse_is_named_by_the_definition_it_is_an_instance_of():
    """The number a scanner reads the same pulse by, out of the design's cache."""
    _, read = parse_pulses(format_pulses([EXCITATION, REFOCUSING]))
    assert [p.definition for p in read] == [0, 1]


def test_text_without_a_block_states_no_pulses():
    assert parse_pulses("[Protocol]\nTE: 8000\n[Protocol End]\n") == ("", [])


def test_a_line_too_short_to_be_a_pulse_is_refused():
    with pytest.raises(ValueError, match="four values"):
        parse_pulses("[RfPulses]\nflip 1 30\n[RfPulses End]\n")


def test_the_shape_does_not_travel():
    """It is in the design's cache, which the scanner reads for itself."""
    assert not hasattr(EXCITATION, "envelope")
    assert len(format_pulses([EXCITATION, REFOCUSING], "abc123")) < 200


def test_a_design_names_where_its_shapes_are():
    design, _ = parse_pulses(format_pulses([EXCITATION], "d7"))
    assert design == "d7"


def _plugin(tmp_path, body):
    plugins = tmp_path / "plugins"
    plugins.mkdir(exist_ok=True)
    (plugins / "linked.py").write_text(
        "from pypulseqpp.sequences.sequence.gre2D_sequence import Gre2DApp\n"
        "from pulserver.design import FloatParam, ScannerSequence\n"
        "from pulserver.protocol import UIParam\n"
        "class Linked(ScannerSequence):\n"
        "    app = Gre2DApp\n"
        "    ui = {UIParam.FLIP: FloatParam('flip_angle_deg', unit='deg',\n"
        "                                   range_min=1.0, range_max=180.0)}\n"
        f"    {body}\n"
        "PLUGIN = Linked()\n"
    )
    return plugins


def test_the_pulses_of_a_design_reach_the_listing(tmp_path):
    """The whole point: a scanner is told the RF without asking for a design."""
    from pulserver.host._service import PULSES_FILE, list_protocol
    from pulserver.host._store import DesignStore

    plugins = _plugin(tmp_path, "follows = (UIParam.FLIP,)")
    store = DesignStore(tmp_path / "store")
    staged = store.stage()
    design = store.commit("identity-1", staged, {"plugin": "linked"})
    (store.directory(design) / PULSES_FILE).write_text(
        format_pulses([REFOCUSING], design)
    )

    named, read = parse_pulses(list_protocol(plugins, "linked", store))
    assert named == design
    assert [p.follows for p in read] == ["flip"]
    assert read[0].definition == 1


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


def _designed(plugins, tmp_path):
    """Design the plugin's sequence and return it beside its resolved protocol."""
    import sys

    sys.path.insert(0, str(plugins))
    try:
        import linked

        plugin = linked.PLUGIN
    finally:
        sys.path.remove(str(plugins))
        sys.modules.pop("linked", None)
    app = plugin.app()
    written = app.write(str(tmp_path / "out"))
    paths = [written] if isinstance(written, str) else list(written)
    return plugin, paths[0], app.resolved


def test_a_declared_pulse_takes_its_share_of_the_control_that_drives_it(tmp_path):
    """A GRE excites once: one pulse, following flip, at the angle it was designed."""
    plugin, seq_path, designed = _designed(
        _plugin(tmp_path, "follows = (UIParam.FLIP,)"), tmp_path
    )
    pulses = plugin.rf_pulses(seq_path, designed)
    assert [p.follows for p in pulses] == ["flip"]
    assert pulses[0].factor == pytest.approx(1.0, abs=1e-3)
    assert pulses[0].flip_deg == pytest.approx(
        float(designed["flip_angle_deg"]), rel=1e-3
    )


def test_a_pulse_left_undeclared_is_played_at_the_angle_it_was_designed(tmp_path):
    plugin, seq_path, designed = _designed(
        _plugin(tmp_path, "follows = ('',)"), tmp_path
    )
    pulses = plugin.rf_pulses(seq_path, designed)
    assert [p.follows for p in pulses] == [""]
    assert pulses[0].factor == 1.0


def test_declaring_a_different_number_of_pulses_than_the_sequence_plays_is_refused(
    tmp_path,
):
    """Otherwise a pulse is silently costed against another pulse's control."""
    plugin, seq_path, designed = _designed(
        _plugin(tmp_path, "follows = (UIParam.FLIP, UIParam.FLIP)"), tmp_path
    )
    with pytest.raises(ValueError, match="declares 2 pulses but its sequence plays 1"):
        plugin.rf_pulses(seq_path, designed)


def test_declaring_nothing_states_no_rf(tmp_path):
    """The default: a scanner costs the RF at download, from the cache."""
    plugin, seq_path, designed = _designed(_plugin(tmp_path, "recon = ''"), tmp_path)
    assert plugin.rf_pulses(seq_path, designed) == []
