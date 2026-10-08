"""The shipped scanner sequences' evaluations: their scan time, the values they read back and the RF they state."""

import functools
import importlib.util
import inspect
from types import SimpleNamespace

import numpy as np
import pypulseqpp as pp
import pytest
from _host import LIMITS, value_block

from pulserver import _plugins
from pulserver._zoo import ZOO_PAIRS
from pulserver._zoo._evaluation import achieved, rf_layout
from pulserver._zoo._saturation import explicit_bands
from pulserver._zoo._slab import slab
from pulserver._zoo._user import SHARED
from pulserver.design import Protocol, StatedParam, load_plugin
from pulserver.host import call
from pulserver.protocol import (
    ConfigKey,
    ImagingMode,
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
    "bssfp3d": {"flip": 30.0, "ny": 32, "nslices": 8},
    "epi2d": {"flip": 60.0, "nx": 64, "ny": 32},
    "epi3d": {"flip": 60.0, "nx": 64, "ny": 32, "nslices": 8},
    "fse3d": {"nx": 64, "ny": 32, "nslices": 16, "etl": 16},
    "gre2d": {"flip": 20.0, "nx": 64, "ny": 48},
    "gre3d": {"flip": 20.0, "nx": 64, "ny": 32, "nslices": 16},
    "gre_multiecho2d": {"flip": 20.0, "num_echoes": 2, "nx": 64, "ny": 48},
    "gre_multiecho3d": {
        "flip": 20.0,
        "num_echoes": 2,
        "nx": 64,
        "ny": 32,
        "nslices": 16,
    },
    "gre_propeller2d": {"flip": 20.0, "nx": 64},
    "gre_radial2d": {"flip": 20.0, "nx": 64},
    "gre_spiral2d": {"flip": 20.0, "nx": 64, "num_shots": 8},
    "gre_stack_of_blades3d": {"flip": 20.0, "nx": 64, "nslices": 8},
    "gre_stack_of_spirals3d": {"flip": 20.0, "nx": 64, "nslices": 8},
    "gre_stack_of_stars3d": {"flip": 20.0, "nx": 64, "nslices": 8},
    "mprage3d": {"flip": 12.0, "nx": 64, "ny": 32, "nslices": 16},
    "mprage_stack_of_spirals3d": {
        "flip": 20.0,
        "nx": 64,
        "nslices": 8,
        "num_shots": 8,
    },
    "mprage_stack_of_stars3d": {"flip": 20.0, "nx": 64, "nslices": 8},
    "se2d": {"nx": 64, "ny": 48},
    "se3d": {"nx": 64, "ny": 32, "nslices": 16},
    "se_epi_propeller2d": {"nx": 64, "etl": 8},
    "se_propeller2d": {"nx": 64},
    "se_radial2d": {"nx": 64},
    "se_spiral2d": {"nx": 64, "num_shots": 8},
    "se_stack_of_blades3d": {"nx": 64, "nslices": 8},
    "se_stack_of_spirals3d": {"nx": 64, "nslices": 8},
    "se_stack_of_stars3d": {"nx": 64, "nslices": 8},
    "zte3d": {"flip": 20.0, "nx": 64},
}
REQUESTS = [pytest.param(name, {}, id=f"{name}-default") for name in SHIPPED] + [
    pytest.param(name, CHANGED[name], id=f"{name}-changed") for name in SHIPPED
]

#: The toggled sequences whose refocusing trains blochsim designs.
OPTIMIZED = ("fse3d+OPTIMIZED", "fse3d+DUAL_REGION")
WITH_BLOCHSIM = pytest.mark.skipif(
    importlib.util.find_spec("blochsim") is None, reason="needs blochsim"
)

# The defaults, and prescriptions that change what the repetitions of a scan
# are counted from: slices in packets of unequal size at a requested TR or in
# one packet at the shortest, undersampling, partitions, shots, frames, blades,
# echo trains and inversion shots; by wire name.
SCANS = [pytest.param(name, {}, id=name) for name in SHIPPED] + [
    pytest.param(
        name,
        changes,
        id=f"{name}-{label}",
        marks=WITH_BLOCHSIM if name in OPTIMIZED else (),
    )
    for name, label, changes in (
        ("bssfp2d", "slices", {"nslices": 3, "TR": 6000, "ny": 64}),
        ("bssfp3d", "longer-tr", {"TR": 9000, "nslices": 8, "ny": 32}),
        ("bssfp3d", "undersampled", {"nslices": 12, "ny": 48, "Ry": 2, "Rz": 2}),
        ("epi2d", "undersampled", {"Ry": 2}),
        ("epi2d+MULTIBAND", "multiband", {"nslices": 6, "multiband": 2}),
        (
            "epi2d+MULTIBAND",
            "multiband-packets",
            {"nslices": 8, "multiband": 2, "TR": 900000, "num_shots": 2},
        ),
        (
            "fse3d+DUAL_REGION",
            "periphery",
            {
                "nslices": 16,
                "etl": 10,
                "nx": 64,
                "ny": 32,
                "TR": 1000000,
                "user12_value": 1500.0,
                "user13_value": 14,
            },
        ),
        (
            "fse3d+OPTIMIZED",
            "optimized",
            {"nslices": 16, "etl": 10, "nx": 64, "ny": 32, "flip": 120.0},
        ),
        (
            "fse3d+NAVIGATOR",
            "navigator",
            {"nslices": 16, "etl": 16, "nx": 64, "ny": 32},
        ),
        ("mprage3d+NAVIGATOR", "navigator", {"nslices": 16, "nx": 64, "ny": 32}),
        ("gre3d+WAVE", "wave", {"nslices": 24, "ny": 32, "Ry": 2, "Rz": 2}),
        (
            "gre_multiecho3d+WAVE",
            "wave",
            {"nslices": 24, "ny": 32, "Ry": 2, "Rz": 2, "num_echoes": 2},
        ),
        ("mprage3d+WAVE", "wave", {"nslices": 24, "nx": 64, "ny": 32, "Ry": 2}),
        (
            "fse3d+WAVE",
            "wave",
            {"nslices": 24, "nx": 64, "ny": 32, "etl": 16, "Ry": 2, "Rz": 2},
        ),
        ("gre_spiral2d+VARIABLE_DENSITY", "variable", {"user12_value": 3.0}),
        (
            "mprage_stack_of_spirals3d+VARIABLE_DENSITY",
            "variable",
            {"nslices": 16, "user12_value": 1.5},
        ),
        ("epi2d", "packets", {"nslices": 7, "TR": 900000, "Ry": 2, "num_shots": 2}),
        ("epi2d", "frames", {"nslices": 12, "TR": 5000000, "num_frames": 3}),
        ("epi2d", "shortest", {"nslices": 6, "TR": TRPreset.MINIMUM}),
        ("gre2d", "saturation", {"nslices": 3, "exsat2_loc": 60.0}),
        ("se2d", "saturation", {"nslices": 3, "exsat1_thickness": 30.0}),
        (
            "epi3d",
            "undersampled",
            {"Ry": 2, "Rz": 2, "nx": 64, "ny": 64, "nslices": 16},
        ),
        (
            "epi3d",
            "shots",
            {"Ry": 3, "Rz": 2, "num_shots": 2, "nx": 64, "ny": 96, "nslices": 12},
        ),
        (
            "epi3d",
            "frames",
            {"num_frames": 3, "TR": 3000000, "nx": 64, "ny": 32, "nslices": 8},
        ),
        (
            "fse3d",
            "shortest",
            {
                "nslices": 16,
                "TE": TEPreset.MINIMUM,
                "TR": TRPreset.MINIMUM,
                "etl": 10,
                "nx": 64,
                "ny": 32,
            },
        ),
        (
            "fse3d",
            "undersampled",
            {"nslices": 24, "Ry": 2, "Rz": 2, "etl": 12, "nx": 64, "ny": 48},
        ),
        ("gre3d", "partitions", {"nslices": 16, "TR": 20000, "nx": 64, "ny": 32}),
        (
            "gre_multiecho2d",
            "shortest",
            {"nslices": 9, "TR": TRPreset.MINIMUM, "num_echoes": 6, "ny": 48},
        ),
        (
            "gre_multiecho3d",
            "undersampled",
            {"nslices": 12, "Ry": 2, "Rz": 2, "num_echoes": 3, "nx": 64, "ny": 48},
        ),
        (
            "gre_propeller2d",
            "packets",
            {"nslices": 10, "TR": 40000, "nx": 64, "Ry": 2},
        ),
        ("gre_radial2d", "packets", {"nslices": 10, "TR": 40000, "nx": 64}),
        (
            "gre_stack_of_blades3d",
            "blades",
            {"nslices": 5, "TR": 12000, "nx": 100, "bandwidth": 62500.0},
        ),
        (
            "mprage3d",
            "shortest",
            {"nslices": 16, "TE": TEPreset.MINIMUM, "TR": TRPreset.MINIMUM, "ny": 32},
        ),
        (
            "mprage3d",
            "undersampled",
            {"nslices": 12, "Ry": 2, "Rz": 2, "prep_time": 600000, "nx": 64, "ny": 48},
        ),
        (
            "mprage_stack_of_spirals3d",
            "recovery",
            {"nslices": 9, "num_shots": 8, "TR": 2500000, "prep_time": 700000},
        ),
        (
            "mprage_stack_of_stars3d",
            "recovery",
            {"nslices": 9, "nx": 64, "TR": 3000000, "prep_time": 500000},
        ),
        (
            "mprage_stack_of_stars3d",
            "shortest",
            {"nslices": 8, "nx": 64, "TR": TRPreset.MINIMUM},
        ),
        ("se2d", "packets", {"nslices": 7, "TR": 60000, "Ry": 3}),
        (
            "se_epi_propeller2d",
            "passes",
            {"nslices": 7, "TR": 300000, "nx": 64, "etl": 8},
        ),
        ("se_propeller2d", "packets", {"nslices": 7, "TR": 60000, "Ry": 3}),
        ("se_spiral2d", "packets", {"nslices": 25, "TR": 300000, "num_shots": 8}),
        ("se_stack_of_blades3d", "blades", {"nslices": 5, "TR": 30000, "nx": 100}),
        (
            "se_stack_of_blades3d",
            "shortest",
            {"nslices": 8, "TR": TRPreset.MINIMUM, "nx": 64},
        ),
        (
            "se_stack_of_stars3d",
            "shortest",
            {"nslices": 8, "TR": TRPreset.MINIMUM, "nx": 64},
        ),
        ("zte3d", "undersampled", {"Ry": 4, "nx": 64}),
        ("zte3d", "longer", {"TR": 900, "Ry": 3, "nx": 48}),
    )
]

# Requested TRs shorter than one slice, partition, echo train or shot takes,
# and an EPI time series whose volume one TR cannot hold.
REJECTED = [
    pytest.param("bssfp3d", {"TR": 4000, "nslices": 8, "ny": 32}, id="bssfp3d-short"),
    pytest.param("epi2d", {"nslices": 5, "TR": 300000}, id="epi2d-short"),
    pytest.param(
        "epi2d", {"nslices": 12, "TR": 2000000, "num_frames": 5}, id="epi2d-frames"
    ),
    pytest.param(
        "epi3d", {"nx": 64, "ny": 32, "nslices": 8, "TR": 100000}, id="epi3d-short"
    ),
    pytest.param(
        "fse3d",
        {"TR": 30000, "etl": 16, "nslices": 16, "nx": 64, "ny": 32},
        id="fse3d-short",
    ),
    pytest.param(
        "gre_multiecho3d",
        {"TR": 8000, "num_echoes": 8, "nslices": 16, "nx": 64, "ny": 32},
        id="gre_multiecho3d-short",
    ),
    pytest.param(
        "gre_propeller2d", {"nslices": 4, "TR": 3000}, id="gre_propeller2d-short"
    ),
    pytest.param("gre_radial2d", {"nslices": 4, "TR": 3000}, id="gre_radial2d-short"),
    pytest.param(
        "gre_stack_of_blades3d", {"TR": 3000}, id="gre_stack_of_blades3d-short"
    ),
    pytest.param(
        "mprage3d",
        {"TR": 500000, "nslices": 16, "nx": 64, "ny": 32},
        id="mprage3d-short",
    ),
    pytest.param(
        "mprage_stack_of_spirals3d",
        {"num_shots": 8, "TR": 500000},
        id="mprage_stack_of_spirals3d-short",
    ),
    pytest.param(
        "mprage_stack_of_stars3d",
        {"nx": 64, "TR": 1000000},
        id="mprage_stack_of_stars3d-short",
    ),
    pytest.param("se2d", {"nslices": 4, "TR": 12000}, id="se2d-short"),
    pytest.param(
        "se_epi_propeller2d", {"nslices": 4, "TR": 20000}, id="se_epi_propeller2d-short"
    ),
    pytest.param(
        "se_propeller2d", {"nslices": 4, "TR": 12000}, id="se_propeller2d-short"
    ),
    pytest.param(
        "se_stack_of_blades3d", {"TR": 10000}, id="se_stack_of_blades3d-short"
    ),
    pytest.param("zte3d", {"nx": 64, "TR": 300}, id="zte3d-short"),
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
TRAINS: dict[str, list[str]] = {
    "epi2d": ["saturation", "excitation"],
    "epi3d": ["excitation"] * 32,
    "fse3d": ["excitation"] + ["refocusing"] * 45,
    "gre2d": ["saturation"] * 2 + ["excitation"],
    "mprage3d": ["inversion"] + ["excitation"] * 256,
    "mprage_stack_of_spirals3d": ["inversion"] + ["excitation"] * 16,
    "mprage_stack_of_stars3d": ["inversion"] + ["excitation"] * 403,
    "se2d": ["saturation"] * 2 + ["excitation", "refocusing"],
}

# The blocks in the last TR of a design whose last view outlasts the TR it
# states, by name. A zero echo time view is a block that holds its pulse and a
# block that reads; the last view of a shell reads until the gradient has
# slewed to zero, so the final TR seconds of the design hold no pulse.
LAST_TR_BLOCKS = {"zte3d": 2}


#: Shipped sequences with a module constant switched on, by the name the scans
#: below use.
TOGGLED = {
    "bssfp2d+RETROSPECTIVE": ("bssfp2d", "RETROSPECTIVE"),
    "epi2d+MULTIBAND": ("epi2d", "MULTIBAND"),
    "fse3d+OPTIMIZED": ("fse3d", "OPTIMIZED"),
    "fse3d+DUAL_REGION": ("fse3d", "DUAL_REGION"),
    "fse3d+NAVIGATOR": ("fse3d", "NAVIGATOR"),
    "mprage3d+NAVIGATOR": ("mprage3d", "NAVIGATOR"),
    "gre3d+WAVE": ("gre3d", "WAVE"),
    "gre_multiecho3d+WAVE": ("gre_multiecho3d", "WAVE"),
    "mprage3d+WAVE": ("mprage3d", "WAVE"),
    "fse3d+WAVE": ("fse3d", "WAVE"),
    "gre_spiral2d+VARIABLE_DENSITY": ("gre_spiral2d", "VARIABLE_DENSITY"),
    "mprage_stack_of_spirals3d+VARIABLE_DENSITY": (
        "mprage_stack_of_spirals3d",
        "VARIABLE_DENSITY",
    ),
}


@pytest.fixture(scope="module")
def zoo(tmp_path_factory):
    """The shipped scanner sequences by name, and those of ``TOGGLED`` as a copy of the file with the constant set."""
    plugins = {name: load_plugin(_plugins.SEQUENCES / f"{name}.py") for name in SHIPPED}
    for key, (name, constant) in TOGGLED.items():
        if key in OPTIMIZED and importlib.util.find_spec("blochsim") is None:
            continue
        source = (_plugins.SEQUENCES / f"{name}.py").read_text()
        assert f"\n{constant} = False\n" in source
        path = tmp_path_factory.mktemp(key.replace("+", "-")) / f"{name}.py"
        path.write_text(
            source.replace(f"\n{constant} = False\n", f"\n{constant} = True\n")
        )
        plugins[key] = load_plugin(path)
    return plugins


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


def _last_tr_power(main, blocks=None):
    """The RF energy of the last TR of ``main``, over its ``TR``, in Hz².

    The last TR is the last ``blocks`` blocks, or those that end within a
    ``TR`` of the end.
    """
    tr = main.definitions["TR"][0]
    ends = np.cumsum(main.libraries().block_durations)
    first = (
        ends.size - blocks + 1
        if blocks
        else int(np.flatnonzero(ends > ends[-1] - tr)[0]) + 1
    )
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
    chain = _chain(plugin.generate(SYSTEM, protocol))

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

    if name in OPTIMIZED:
        # The layout plays the train at a constant refocusing angle.
        main = _chain(
            plugin.app(SYSTEM, **(protocol.arguments | {"flip_modulation": "constant"}))
        )[-1]
    else:
        main = _main(plugin, protocol)
    # The first TR holds the largest packet of slices; a balanced steady state
    # opens with its half-angle pulse, so its last TR is the regular one.
    balanced = name.startswith("bssfp")
    start = main.duration()[0] - main.definitions["TR"][0] if balanced else 0
    # An optimized refocusing train is designed around its flip angle rather
    # than scaled by it.
    scaled = UIParam.FLIP in plugin.protocol and not name.startswith("fse3d")
    expected = rf_layout(main, scaled, start=start)
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


@pytest.mark.parametrize(
    ("name", "changes"),
    [
        ("mprage_stack_of_spirals3d", {"nslices": 16, "num_shots": 8}),
        ("mprage_stack_of_stars3d", {"nslices": 16, "nx": 64}),
    ],
)
def test_an_inversion_train_evaluation_counts_the_shots_and_excitations_of_an_undersampled_scan(
    zoo, monkeypatch, name, changes
):
    plugin = zoo[name]
    monkeypatch.setattr(
        plugin,
        "app",
        functools.partial(
            plugin.app, ry=2, rz=2, n_acs_z=4, partial_fourier_z=0.875, n_dummy=2
        ),
    )
    protocol = _protocol(plugin, changes)

    evaluation = plugin.evaluate(SYSTEM, protocol)

    main = _main(plugin, protocol)
    expected = rf_layout(main, scaled=True)
    rounding = SYSTEM.block_duration_raster * _excitations(main)
    assert abs(evaluation.duration - main.duration()[0]) <= rounding + 1e-9
    assert evaluation.protocol.to_wire() == (
        protocol.replace(achieved(plugin, main)).to_wire()
    )
    assert evaluation.rf_layout.period == pytest.approx(expected.period, rel=1e-12)
    assert evaluation.rf_layout.instances.definition.tolist() == (
        expected.instances.definition.tolist()
    )
    assert len(expected.instances.definition) > 2


@pytest.mark.parametrize(("name", "changes"), REJECTED)
def test_an_evaluation_rejects_a_tr_the_design_rejects(zoo, name, changes):
    plugin = zoo[name]
    protocol = _protocol(plugin, changes)

    with pytest.raises(ValueError, match="TR"):
        plugin.generate(SYSTEM, protocol)
    with pytest.raises(ValueError, match="TR"):
        plugin.evaluate(SYSTEM, protocol)


# Arguments of the stack-of-blades functions that no entry binds, set on the
# function: the blades, partitions and dummies they leave out or add.
BLADES = {
    "shipped": {},
    "fewer-blades": {"ry": 3, "blade_width": 12, "n_dummy": 5},
    "undersampled-partitions": {"rz": 2, "n_acs_z": 4, "partial_fourier_z": 0.8},
    "both": {"ry": 2, "rz": 3, "n_acs_z": 6, "blade_width": 8, "n_dummy": 1},
}


@pytest.mark.parametrize("name", ["gre_stack_of_blades3d", "se_stack_of_blades3d"])
@pytest.mark.parametrize("bound", BLADES.values(), ids=BLADES)
def test_a_stack_of_blades_evaluation_plays_every_repetition_of_the_design(
    zoo, name, bound, monkeypatch
):
    plugin = zoo[name]
    monkeypatch.setattr(plugin, "app", functools.partial(plugin.app, **bound))
    protocol = _protocol(plugin, {"nx": 64, "nslices": 12})

    evaluation = plugin.evaluate(SYSTEM, protocol)

    # Every repetition plays the same blocks, so the durations agree to
    # round-off; the raster per excitation the other scans allow would admit a
    # repetition more or fewer.
    designed = plugin.app(SYSTEM, **protocol.arguments)
    assert evaluation.duration == pytest.approx(designed.duration()[0], rel=1e-9)


@pytest.mark.parametrize("name", SHIPPED)
def test_a_zoo_evaluation_designs_two_trs_at_most_however_large_the_prescription(
    zoo, name, monkeypatch
):
    plugin = zoo[name]
    app = plugin.app
    designed, built = [], []

    @functools.wraps(app)
    def recording(system, **arguments):
        designed.append(app(system, **arguments))
        return designed[-1]

    initialise = pp.Sequence.__init__

    def recording_init(self, *args, **kwargs):
        initialise(self, *args, **kwargs)
        built.append(self)

    changes = {key: value for key, value in LARGE.items() if key in plugin.protocol}
    protocol = _protocol(plugin, changes)
    monkeypatch.setattr(plugin, "app", recording)
    monkeypatch.setattr(pp.Sequence, "__init__", recording_init)
    plugin.evaluate(SYSTEM, protocol)

    # An evaluation designs through the app once, two TRs at most, or builds
    # the modules of a repetition itself, two excitations at most.
    assert len(designed) <= 1
    if designed:
        main = _chain(designed[0])[-1]
        tr = main.definitions["TR"][0]
        rounding = SYSTEM.block_duration_raster * _excitations(main)
        assert main.duration()[0] <= 2 * tr + rounding
    else:
        assert built
        assert max(map(_excitations, built)) <= 2


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
        _last_tr_power(_main(plugin, protocol), LAST_TR_BLOCKS.get(name)),
        rel=POWER_RTOL,
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
        _last_tr_power(main, LAST_TR_BLOCKS.get(name)), rel=POWER_RTOL
    )


@pytest.mark.parametrize("name", ["fse3d+NAVIGATOR", "mprage3d+NAVIGATOR"])
def test_a_sequence_with_navigators_asks_for_motion_correction(zoo, name):
    plugin = zoo[name]
    protocol = _protocol(plugin, {"nslices": 16, "nx": 64, "ny": 32})

    main = _main(plugin, protocol)

    assert main.definitions["EnablePmc"][0] == 1
    assert ZOO_PAIRS[TOGGLED[name][0]] == "pmc"


def _turned(changes, rotation, offset_mm):
    from pulserver.protocol import FOV_OFFSET, FOV_ROTATION

    return {
        **changes,
        **dict(zip(FOV_ROTATION, np.asarray(rotation).ravel(), strict=True)),
        **dict(zip(FOV_OFFSET, offset_mm, strict=True)),
    }


def test_the_console_bands_reach_the_sequence_function_as_placed_in_the_physical_frame():
    def app(
        *,
        sat1_normal_x=0.0,
        sat1_normal_y=0.0,
        sat1_normal_z=0.0,
        sat1_position=0.0,
        sat1_thickness=0.0,
        n_x=64,
    ):
        return locals()

    banded = explicit_bands(app, 1)
    received = banded(
        exsat1_normal_x=0.0,
        exsat1_normal_y=0.6,
        exsat1_normal_z=0.8,
        exsat1_loc=-0.03,
        exsat1_thickness=0.02,
        n_x=32,
    )

    assert "sat1_position" not in inspect.signature(banded).parameters
    assert received == {
        "sat1_normal_x": 0.0,
        "sat1_normal_y": 0.6,
        "sat1_normal_z": 0.8,
        "sat1_position": -0.03,
        "sat1_thickness": 0.02,
        "n_x": 32,
    }


BANDED = ["gre2d", "se2d"]


@pytest.mark.parametrize("name", BANDED)
def test_a_banded_sequence_declares_every_band_it_plays_on(name, zoo):
    listing = zoo[name].listing()

    assert listing[ConfigKey.EXSAT_MASK].value == 3
    assert not listing[ConfigKey.EXSAT_MASK].editable
    assert UIParam.exsat_loc(2) in listing and UIParam.exsat_loc(3) not in listing


def _saturations(seq):
    return [
        seq.get_block(index).rf
        for index in seq.block_events
        if (rf := seq.get_block(index).rf) is not None and rf.use == "saturation"
    ]


@pytest.mark.parametrize("name", BANDED)
def test_every_band_is_played_before_every_excitation_exempt_from_the_prescription(
    name, zoo
):
    plugin = zoo[name]
    request = {
        "nx": 64,
        "ny": 32,
        "exsat2_normal_x": 0.0,
        "exsat2_normal_y": 1.0,
        "exsat2_normal_z": 0.0,
        "exsat2_loc": 30.0,
        "exsat2_thickness": 20.0,
    }
    main = _main(plugin, _protocol(plugin, request))
    pulses = [
        index
        for index in main.block_events
        if (rf := main.get_block(index).rf) is not None and rf.use == "saturation"
    ]
    exempt = {
        (label.label, int(label.value))
        for index in pulses
        for label in main.get_block(index).label or ()
    }

    assert len(pulses) == 2 * _excitations(main)
    # A 3 ms sinc of time-bandwidth 4 over 20 mm, centred 30 mm along y.
    assert main.get_block(pulses[1]).rf.freq_offset == pytest.approx(
        4.0 / (3e-3 * 20e-3) * 30e-3, rel=1e-6
    )
    assert {("NOPOS", 1), ("NOROT", 1)} <= exempt


def test_the_2d_epi_saturates_fat_and_plays_no_band(zoo):
    plugin = zoo["epi2d"]
    *_, main = plugin.generate(SYSTEM, _protocol(plugin, {"nx": 64, "ny": 64}))

    assert ConfigKey.EXSAT_MASK not in plugin.protocol
    assert UIParam.FAT_SAT not in plugin.protocol
    assert len(_saturations(main)) > 0
    assert all(rf.freq_ppm != 0.0 for rf in _saturations(main))


def test_a_cine_is_gated_at_the_heart_rate_asked_for(zoo):
    plugin = zoo["bssfp2d"]

    assert plugin.listing()[UIParam.HEART_RATE].value == 60
    assert _protocol(plugin, {"heart_rate": 80}).arguments["heart_rate_bpm"] == 80


@pytest.mark.parametrize(
    ("name", "gating"),
    [("bssfp2d", "prospective"), ("bssfp2d+RETROSPECTIVE", "retrospective")],
)
def test_an_ecg_trigger_gates_the_cine_with_a_segment_per_heartbeat(zoo, name, gating):
    plugin = zoo[name]
    module = SimpleNamespace(**type(plugin).generate.__globals__)
    changes = {"trigger_type": "physio2", "Ry": 2, "num_frames": 20}
    protocol = _protocol(plugin, changes)

    evaluation = plugin.evaluate(SYSTEM, protocol)
    gated = module._gated(plugin, SYSTEM, protocol)
    main = _main(plugin, protocol)

    assert gated["gating"] == gating
    lines = main.definitions["TR"][0] * gated["views_per_segment"] * 20
    assert lines <= 1.0 - gated["trigger_delay"]
    phases = _phase_labels(main)
    if gating == "prospective":
        segments = len(
            module._segments(_arguments(plugin, protocol), gated["views_per_segment"])
        )
        assert evaluation.duration == pytest.approx(segments * 1.0)
        assert phases == set(range(20))
    else:
        assert (
            abs(evaluation.duration - main.duration()[0])
            <= 2 * SYSTEM.block_duration_raster
        )


def _arguments(plugin, protocol):
    from pulserver._zoo._evaluation import arguments

    return arguments(plugin, protocol)


def _phase_labels(sequence):
    """The values ``PHS`` holds at the readouts of ``sequence``."""
    return set(np.asarray(sequence.evaluate_labels(evolution="adc")["PHS"]).tolist())


def test_the_cine_refuses_respiratory_triggering(zoo):
    validation = zoo["bssfp2d"].validate(SYSTEM, {"trigger_type": "physio1"})
    assert not validation.valid
    assert "respiratory" in validation.info


@pytest.mark.parametrize("name", SHIPPED)
def test_every_shipped_sequence_states_its_imaging_mode_and_refuses_the_other(
    name, zoo
):
    plugin = zoo[name]
    stated = plugin.protocol[UIParam.IMAGING_MODE]
    other = next(mode for mode in ImagingMode if mode != stated.value)

    assert isinstance(stated, StatedParam)
    assert stated.value == (
        ImagingMode.THREE_D if name.endswith("3d") else ImagingMode.TWO_D
    )
    assert "imaging_mode" not in _protocol(plugin).arguments
    with pytest.raises(ValueError, match=f"imaging_mode is {stated.value}"):
        _protocol(plugin, {UIParam.IMAGING_MODE: other})


TWO_D = [name for name in SHIPPED if name.endswith("2d")]
SLABS = [name for name in SHIPPED if name.endswith("3d") and name != "zte3d"]


@pytest.mark.parametrize("name", TWO_D)
def test_every_2d_sequence_takes_its_slice_thickness_and_spacing(name, zoo):
    protocol = zoo[name].protocol

    assert UIParam.SLICE_THICKNESS in protocol and UIParam.SLICE_SPACING in protocol


def test_a_slab_sequence_takes_the_thickness_of_a_location_in_place_of_the_slabs():
    def app(system, *, fov_z=0.128, n_z=64, other=1):
        return fov_z, n_z, other

    wrapped = slab(app)

    assert inspect.signature(wrapped).parameters["slice_thickness"].default == 0.002
    assert wrapped(None, n_z=12, slice_thickness=1.5e-3) == (
        pytest.approx(0.018),
        12,
        1,
    )


@pytest.mark.parametrize("name", SLABS)
def test_every_slab_sequence_is_prescribed_by_its_locations_and_their_thickness(
    name, zoo
):
    plugin = zoo[name]
    parameters = inspect.signature(plugin.app).parameters

    assert "slice_thickness" in parameters and "fov_z" not in parameters
    assert (
        UIParam.SLICE_THICKNESS in plugin.protocol
        and UIParam.NSLICES in plugin.protocol
    )


def test_a_3d_sequence_achieves_the_thickness_of_one_location_whatever_their_number(
    zoo,
):
    plugin = zoo["gre3d"]
    for locations in (16, 32):
        evaluation = plugin.evaluate(
            SYSTEM,
            _protocol(
                plugin,
                {"nslices": locations, "slice_thickness": 2.0, "nx": 64, "ny": 32},
            ),
        )
        assert evaluation.protocol[UIParam.SLICE_THICKNESS] == pytest.approx(2e-3)


@pytest.mark.parametrize("name", SHIPPED)
def test_every_shared_control_a_sequence_takes_is_named_at_its_own_user_cv(name, zoo):
    plugin = zoo[name]
    taken = inspect.signature(plugin.app).parameters
    bound = {
        getattr(entry, "argument", None)
        for key, entry in plugin.protocol.items()
        if not key.startswith("user")
    }

    for index, (arguments, label, _) in enumerate(SHARED):
        free = [a for a in arguments if a in taken and a not in bound]
        listed = UIParam.user_value(index) in plugin.protocol
        assert listed == bool(free), arguments
        if listed:
            assert plugin.protocol[UIParam.user_value(index)].argument == free[0]
            assert plugin.protocol[UIParam.user_name(index)].text == label


def test_a_dummy_scan_count_set_through_its_user_cv_reaches_the_design(zoo):
    plugin = zoo["gre2d"]
    request = {"nx": 32, "ny": 32}

    lengths = [
        len(
            plugin.app(
                SYSTEM, **_protocol(plugin, request | {"user0_value": n}).arguments
            ).block_events
        )
        for n in (0, 8)
    ]

    assert lengths[1] > lengths[0]


def test_an_echo_spacing_of_zero_is_the_shortest_one(zoo):
    plugin = zoo["fse3d"]
    request = {"nx": 64, "ny": 32, "nslices": 16, "etl": 16}

    shortest = plugin.app(
        SYSTEM, **_protocol(plugin, request | {"user7_value": 0.0}).arguments
    )

    assert shortest.definitions["EchoSpacing"][0] > 0.0


@pytest.mark.parametrize("name", ["gre_multiecho2d", "gre_multiecho3d"])
def test_a_multi_echo_gradient_echo_spaces_its_echoes_by_the_shared_echo_spacing(
    name, zoo
):
    plugin = zoo[name]
    request = {"nx": 64, "ny": 32, "nslices": 8, "num_echoes": 3}

    def echo_times(spacing_ms):
        designed = plugin.app(
            SYSTEM, **_protocol(plugin, request | {"user7_value": spacing_ms}).arguments
        )
        main = designed if isinstance(designed, pp.Sequence) else designed[-1]
        return np.asarray(main.definitions["TE"])

    shortest, spaced = echo_times(0.0), echo_times(10.0)

    assert UIParam.TE2 not in plugin.protocol
    assert len(spaced) == 3
    assert np.diff(spaced) == pytest.approx(10e-3, abs=1e-5)
    assert np.diff(shortest).max() < 10e-3
