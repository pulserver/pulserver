"""The shipped scanner sequences' evaluations: their scan time, the values they read back and the RF they state."""

import functools
import inspect

import numpy as np
import pypulseqpp as pp
import pytest
from _host import LIMITS, value_block

from pulserver import _plugins
from pulserver._zoo._evaluation import achieved, rf_layout
from pulserver.design import Protocol, load_plugin
from pulserver.host import call
from pulserver.protocol import (
    TEPreset,
    TRPreset,
    UIParam,
    parse_rf_definitions,
    parse_rf_layout,
)

SYSTEM = pp.Opts(**LIMITS)
SHIPPED = sorted(path.stem for path in _plugins.SEQUENCES.glob("*.py"))
# The spin-echo sequences fix their flip angles; the others take one.
SPIN_ECHO = [name for name in SHIPPED if name.startswith(("se", "fse"))]
FLIPPED = [name for name in SHIPPED if name not in SPIN_ECHO]

# The layout integrates the squared magnitude of each unit-peak waveform with
# the trapezoidal rule over the sample times the listing carries;
# ``calc_rf_power`` interpolates the waveform linearly onto a 1 us grid and sums
# it. The two differ by a term of second order in the ratio of the sample
# spacing to the pulse duration, which is below 1e-4 for the shortest sinc pulse
# of the shipped sequences, the balanced SSFP's, and smaller for the others.
POWER_RTOL = 1e-3

# A protocol small enough to design in a moment, whose flip angle differs from
# the default and which changes the matrix, the number of interleaves or of
# echoes, whichever the sequence has; by wire name.
CHANGED = {
    "bssfp2d": {"flip": 30.0, "ny": 48},
    "epi2d": {"flip": 60.0, "nx": 64, "ny": 32},
    "gre2d": {"flip": 20.0, "nx": 64, "ny": 48},
    "gre3d": {"flip": 20.0, "nx": 64, "ny": 32, "nslices": 16},
    "gre_multiecho2d": {"flip": 20.0, "num_echoes": 2, "nx": 64, "ny": 48},
    "gre_radial2d": {"flip": 20.0, "nx": 64},
    "gre_spiral2d": {"flip": 20.0, "nx": 64, "num_shots": 8},
    "gre_stack_of_spirals3d": {"flip": 20.0, "nx": 64, "nslices": 8},
    "gre_stack_of_stars3d": {"flip": 20.0, "nx": 64, "nslices": 8},
    "se2d": {"nx": 64, "ny": 48},
    "se3d": {"nx": 64, "ny": 32, "nslices": 16},
    "se_radial2d": {"nx": 64},
    "se_spiral2d": {"nx": 64, "num_shots": 8},
    "se_stack_of_spirals3d": {"nx": 64, "nslices": 8},
    "se_stack_of_stars3d": {"nx": 64, "nslices": 8},
}
REQUESTS = [pytest.param(name, {}, id=f"{name}-default") for name in SHIPPED] + [
    pytest.param(name, CHANGED[name], id=f"{name}-changed") for name in SHIPPED
]

# The defaults, and prescriptions whose slices fall into packets of unequal
# size at a requested TR or into one packet at the shortest; by wire name.
SCANS = [pytest.param(name, {}, id=name) for name in SHIPPED] + [
    pytest.param(name, changes, id=f"{name}-{label}")
    for name, label, changes in (
        ("bssfp2d", "slices", {"nslices": 3, "TR": 6000, "ny": 64}),
        ("epi2d", "undersampled", {"Ry": 2}),
        ("epi2d", "packets", {"nslices": 7, "TR": 900000, "Ry": 2, "num_shots": 2}),
        ("epi2d", "frames", {"nslices": 12, "TR": 5000000, "num_frames": 3}),
        ("epi2d", "shortest", {"nslices": 6, "TR": TRPreset.MINIMUM}),
        ("gre3d", "partitions", {"nslices": 16, "TR": 20000, "nx": 64, "ny": 32}),
        (
            "gre_multiecho2d",
            "shortest",
            {"nslices": 9, "TR": TRPreset.MINIMUM, "num_echoes": 6, "ny": 48},
        ),
        ("gre_radial2d", "packets", {"nslices": 10, "TR": 40000, "nx": 64}),
        ("se2d", "packets", {"nslices": 7, "TR": 60000, "Ry": 3}),
        ("se_spiral2d", "packets", {"nslices": 25, "TR": 300000, "num_shots": 8}),
        (
            "se_stack_of_stars3d",
            "shortest",
            {"nslices": 8, "TR": TRPreset.MINIMUM, "nx": 64},
        ),
    )
]

# Requested TRs shorter than one slice takes, and an EPI time series whose
# volume one TR cannot hold.
REJECTED = [
    pytest.param("epi2d", {"nslices": 5, "TR": 300000}, id="epi2d-short"),
    pytest.param(
        "epi2d", {"nslices": 12, "TR": 2000000, "num_frames": 5}, id="epi2d-frames"
    ),
    pytest.param("gre_radial2d", {"nslices": 4, "TR": 3000}, id="gre_radial2d-short"),
    pytest.param("se2d", {"nslices": 4, "TR": 12000}, id="se2d-short"),
]

# Many slices, lines, partitions, interleaves and echoes, at the shortest TR;
# by wire name.
LARGE = {
    "nx": 128,
    "ny": 128,
    "nslices": 32,
    "num_shots": 16,
    "num_echoes": 8,
    "TR": TRPreset.MINIMUM,
}

# The RF uses of one TR at the default protocol, in play order, where a TR is
# not one excitation followed, in a spin echo, by one refocusing pulse; by name.
TRAINS: dict[str, list[str]] = {}


@pytest.fixture(scope="module")
def zoo():
    """The shipped scanner sequences by name."""
    return {name: load_plugin(_plugins.SEQUENCES / f"{name}.py") for name in SHIPPED}


def _protocol(plugin, changes=None):
    """The protocol of ``plugin`` at its listed values, with the wire values ``changes``."""
    wire = {
        key: entry.value for key, entry in plugin.listing().items() if entry.editable
    }
    wire.update(changes or {})
    return Protocol.from_wire(plugin.protocol, wire, SYSTEM)


def _chain(designed):
    """The sequences an app returned, the main sequence last."""
    return [designed] if isinstance(designed, pp.Sequence) else list(designed)


def _main(plugin, protocol):
    """The main sequence of the chain ``plugin`` designs for ``protocol``."""
    return _chain(plugin.generate(SYSTEM, protocol))[-1]


def _excitations(sequence):
    """The number of excitation pulses ``sequence`` plays."""
    instances = sequence.rf_instances()
    uses = [definition.use for definition in instances.definitions]
    return sum(uses[number] == "excitation" for number in instances.definition)


def _unit_energy(time, waveform):
    """The integral of the squared magnitude of a unit-peak waveform over its channels, in s."""
    return float(sum(np.trapezoid(np.abs(channel) ** 2, time) for channel in waveform))


def _mean_power(period, amplitude, peak_hz, energy):
    """The RF power averaged over ``period``, in Hz²: the energy of the instances over the period.

    Each argument but ``period`` holds one entry per instance: ``amplitude``
    over ``peak_hz``, and the ``energy`` of the definition's unit-peak waveform.
    """
    played = np.asarray(amplitude) * np.asarray(peak_hz)
    return float(np.sum(played**2 * np.asarray(energy)) / period)


def _layout_power(layout):
    """The mean RF power in Hz² an RF layout states."""
    instances = layout.instances
    peak = np.array([definition.peak_hz for definition in instances.definitions])
    energy = np.array([_unit_energy(d.time, d.waveform) for d in instances.definitions])
    which = instances.definition
    return _mean_power(layout.period, instances.amplitude, peak[which], energy[which])


def _wire_power(listed, layout):
    """The mean RF power in Hz² of an RF layout read from the wire, over the definitions listed."""
    listed = sorted(listed, key=lambda record: record.index)
    assert [record.index for record in listed] == list(range(len(listed)))
    peak = np.array([record.peak_hz for record in listed])
    energy = np.array([_unit_energy(r.time, r.waveform) for r in listed])
    which = list(layout.definition)
    return _mean_power(layout.period, layout.amplitude, peak[which], energy[which])


def _last_tr_power(main):
    """The RF energy of the blocks of ``main`` that end within a ``TR`` of its end, over that ``TR``, in Hz²."""
    tr = main.definitions["TR"][0]
    ends = np.cumsum(main.libraries().block_durations)
    first = int(np.flatnonzero(ends > ends[-1] - tr)[0]) + 1
    return main.calc_rf_power(block_range=(first, ends.size))[3] / tr


def _played_peak(layout):
    """The peak RF amplitude in Hz each instance of a layout plays."""
    instances = layout.instances
    peak = np.array([definition.peak_hz for definition in instances.definitions])
    return instances.amplitude * peak[instances.definition]


def _reply(command, name, **asked):
    status, reply = call(command, plugins=[], plugin=name, limits=LIMITS, **asked)
    assert status == 0, reply
    return reply


def test_the_shipped_sequences_with_a_flip_entry_are_the_ones_that_are_not_spin_echoes(
    zoo,
):
    assert FLIPPED
    assert SPIN_ECHO
    assert {name for name in SHIPPED if UIParam.FLIP in zoo[name].protocol} == set(
        FLIPPED
    )


@pytest.mark.parametrize("name", SHIPPED)
def test_a_zoo_sequence_has_a_flip_entry_exactly_where_its_function_takes_a_flip_angle(
    zoo, name
):
    plugin = zoo[name]
    app = plugin.app
    function = app.func if isinstance(app, functools.partial) else app

    takes_flip = "flip_angle_deg" in inspect.signature(function).parameters

    assert (UIParam.FLIP in plugin.protocol) == takes_flip
    if takes_flip:
        assert plugin.protocol[UIParam.FLIP].argument == "flip_angle_deg"


@pytest.mark.parametrize(("name", "changes"), SCANS)
def test_an_evaluation_states_the_values_and_the_scan_time_of_the_design(
    zoo, name, changes
):
    plugin = zoo[name]
    protocol = _protocol(plugin, changes)

    evaluation = plugin.evaluate(SYSTEM, protocol)
    chain = _chain(plugin.app(SYSTEM, **protocol.arguments))

    expected = protocol.replace(achieved(plugin, chain[-1]))
    assert evaluation.protocol.to_wire() == expected.to_wire()
    # The delay closing a TR is rounded up to the block raster, so each TR the
    # design plays lasts the TR it states or one raster more.
    played = sum(sequence.duration()[0] for sequence in chain)
    rounding = SYSTEM.block_duration_raster * sum(map(_excitations, chain))
    assert abs(evaluation.duration - played) <= rounding + 1e-9


@pytest.mark.parametrize(("name", "changes"), SCANS)
def test_an_evaluation_states_as_its_layout_a_regular_tr_of_the_design(
    zoo, name, changes
):
    plugin = zoo[name]
    protocol = _protocol(plugin, changes)

    layout = plugin.evaluate(SYSTEM, protocol).rf_layout

    main = _main(plugin, protocol)
    # The first TR holds the largest packet of slices; a balanced steady state
    # opens with its half-angle pulse, so its last TR is the regular one.
    balanced = name.startswith("bssfp")
    start = main.duration()[0] - main.definitions["TR"][0] if balanced else 0
    expected = rf_layout(main, UIParam.FLIP in plugin.protocol, start=start)
    assert layout.period == pytest.approx(expected.period, rel=1e-12)
    assert layout.instances.definition.tolist() == (
        expected.instances.definition.tolist()
    )
    assert layout.control == expected.control
    assert layout.instances.amplitude == pytest.approx(
        expected.instances.amplitude, rel=1e-9
    )
    definitions = layout.instances.definitions
    assert [d.use for d in definitions] == [
        d.use for d in expected.instances.definitions
    ]
    assert [d.peak_hz for d in definitions] == pytest.approx(
        [d.peak_hz for d in expected.instances.definitions], rel=1e-9
    )


@pytest.mark.parametrize(("name", "changes"), REJECTED)
def test_an_evaluation_rejects_a_tr_the_design_rejects(zoo, name, changes):
    plugin = zoo[name]
    protocol = _protocol(plugin, changes)

    with pytest.raises(ValueError, match="TR"):
        plugin.app(SYSTEM, **protocol.arguments)
    with pytest.raises(ValueError, match="TR"):
        plugin.evaluate(SYSTEM, protocol)


@pytest.mark.parametrize("name", SHIPPED)
def test_a_zoo_evaluation_designs_two_trs_at_most_however_large_the_prescription(
    zoo, name, monkeypatch
):
    plugin = zoo[name]
    app = plugin.app
    designed = []

    @functools.wraps(app)
    def recording(system, **arguments):
        designed.append(app(system, **arguments))
        return designed[-1]

    monkeypatch.setattr(plugin, "app", recording)
    changes = {key: value for key, value in LARGE.items() if key in plugin.protocol}
    plugin.evaluate(SYSTEM, _protocol(plugin, changes))

    [chain] = designed
    main = _chain(chain)[-1]
    rounding = SYSTEM.block_duration_raster * _excitations(main)
    assert main.duration()[0] <= 2 * main.definitions["TR"][0] + rounding


@pytest.mark.parametrize("name", SHIPPED)
def test_resolving_a_resolved_zoo_protocol_changes_nothing(zoo, name):
    plugin = zoo[name]
    listing = plugin.listing()
    shortest = {
        key: preset
        for key, preset in (
            (UIParam.TE, TEPreset.MINIMUM),
            (UIParam.TR, TRPreset.MINIMUM),
        )
        if key in listing and preset in listing[key].options
    }

    first = plugin.validate(SYSTEM, shortest)

    assert first.valid, first.info
    assert plugin.validate(SYSTEM, first.values) == first


@pytest.mark.parametrize("name", SHIPPED)
def test_each_zoo_layout_plays_the_rf_power_of_the_last_tr_of_the_design(zoo, name):
    plugin = zoo[name]
    protocol = _protocol(plugin)

    layout = plugin.evaluate(SYSTEM, protocol).rf_layout

    assert _layout_power(layout) == pytest.approx(
        _last_tr_power(_main(plugin, protocol)), rel=POWER_RTOL
    )


@pytest.mark.parametrize("name", SHIPPED)
def test_each_zoo_layout_is_one_tr_of_an_excitation_and_a_spin_echo_refocusing_pulse(
    zoo, name
):
    plugin = zoo[name]
    protocol = _protocol(plugin)

    layout = plugin.evaluate(SYSTEM, protocol).rf_layout

    instances = layout.instances
    uses = [instances.definitions[number].use for number in instances.definition]
    assert layout.period == pytest.approx(
        _main(plugin, protocol).definitions["TR"][0], rel=1e-12
    )
    single = ["excitation", "refocusing"] if name in SPIN_ECHO else ["excitation"]
    assert uses == TRAINS.get(name, single)


@pytest.mark.parametrize("name", FLIPPED)
def test_changing_the_flip_angle_scales_exactly_the_instances_a_zoo_layout_marks(
    zoo, name
):
    plugin = zoo[name]
    flip = plugin.listing()[UIParam.FLIP].value / 2

    before = plugin.evaluate(SYSTEM, _protocol(plugin)).rf_layout
    after = plugin.evaluate(SYSTEM, _protocol(plugin, {UIParam.FLIP: flip})).rf_layout

    assert after.control == before.control
    assert after.instances.definition.tolist() == before.instances.definition.tolist()
    played = _played_peak(before)
    assert (played > 0).all()
    ratio = _played_peak(after) / played
    marked = np.array([control == UIParam.FLIP for control in before.control])
    assert marked.any()
    assert ratio[marked] == pytest.approx(0.5, rel=1e-9)
    assert ratio[~marked] == pytest.approx(1.0, rel=1e-9)


@pytest.mark.parametrize("name", SPIN_ECHO)
def test_a_zoo_sequence_without_a_flip_entry_marks_no_instance(zoo, name):
    plugin = zoo[name]

    layout = plugin.evaluate(SYSTEM, _protocol(plugin)).rf_layout

    assert set(layout.control) == {None}


@pytest.mark.parametrize(("name", "changes"), REQUESTS)
def test_the_listed_definitions_and_the_validated_layout_play_the_rf_power_of_the_design(
    zoo, name, changes
):
    listing = _reply("list", name, rf_definitions=True)
    validated = _reply("validate", name, block=value_block(changes), rf_layout=True)

    assert validated.startswith("VALID"), validated
    listed = parse_rf_definitions(listing)
    layout = parse_rf_layout(validated)

    assert set(layout.definition) <= {record.index for record in listed}
    plugin = zoo[name]
    main = _main(plugin, _protocol(plugin, changes))
    assert _wire_power(listed, layout) == pytest.approx(
        _last_tr_power(main), rel=POWER_RTOL
    )
