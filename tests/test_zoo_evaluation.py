"""The shipped scanner sequences' evaluations: their scan time, the values they read back and the RF they state."""

import functools
import inspect

import numpy as np
import pypulseqpp as pp
import pytest
from _host import LIMITS, value_block

from pulserver import _plugins
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
SPIN_ECHO = [name for name in SHIPPED if name.startswith("se")]
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


def _main(plugin, protocol):
    """The main sequence of the chain ``plugin`` designs for ``protocol``."""
    designed = plugin.generate(SYSTEM, protocol)
    return designed if isinstance(designed, pp.Sequence) else list(designed)[-1]


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


@pytest.mark.parametrize(
    ("name", "changes", "files"),
    [
        *[pytest.param(name, {}, 1, id=name) for name in SHIPPED],
        pytest.param("epi2d", {"Ry": 2}, 3, id="epi2d-undersampled"),
    ],
)
def test_an_evaluation_states_the_duration_of_every_file_the_design_writes(
    zoo, tmp_path, name, changes, files
):
    plugin = zoo[name]

    stated = plugin.evaluate(SYSTEM, _protocol(plugin, changes)).duration
    validation, paths = plugin.design(SYSTEM, changes, tmp_path)

    assert validation.valid, validation.info
    assert len(paths) >= files
    written = sum(pp.io.read(path).duration()[0] for path in paths)
    assert stated == pytest.approx(written, rel=1e-2)


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
def test_each_zoo_layout_matches_the_scan_mean_rf_power(zoo, name):
    plugin = zoo[name]
    protocol = _protocol(plugin)

    layout = plugin.evaluate(SYSTEM, protocol).rf_layout

    main = _main(plugin, protocol)
    assert _layout_power(layout) == pytest.approx(
        main.calc_rf_power()[0], rel=POWER_RTOL
    )


@pytest.mark.parametrize("name", SHIPPED)
def test_each_zoo_layout_holds_every_rf_event_of_the_main_sequence_over_its_duration(
    zoo, name
):
    plugin = zoo[name]
    protocol = _protocol(plugin)

    layout = plugin.evaluate(SYSTEM, protocol).rf_layout

    main = _main(plugin, protocol)
    assert layout.period == pytest.approx(main.duration()[0])
    assert len(layout.instances.definition) == np.count_nonzero(
        main.libraries().blocks[:, 0]
    )


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
        main.calc_rf_power()[0], rel=POWER_RTOL
    )
