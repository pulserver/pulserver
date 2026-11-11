"""A design past the nerve or slew limit is made again at a lower slew rate, unless the plugin is strict."""

import json

import pypulseqpp as pp
import pytest
from _host import LIMITS

from pulserver import ir
from pulserver.host import DesignStore, _slew
from pulserver.host import _service as service
from pulserver.protocol import PROTOCOL_BEGIN, PROTOCOL_END

#: A nerve model under which a 36 mT/m step ramped at the design's full slew
#: rate stimulates past the limit, and ramped at about a third of it does not.
SITE = {
    **LIMITS,
    "design_max_grad": LIMITS["max_grad"],
    "design_max_slew": LIMITS["max_slew"],
    "pns_chronaxie": 360e-6,
    "pns_rheobase": 38.0,
    "pns_limit": 1.0,
}

#: Bipolar lobes at nine tenths of the amplitude limit, ramped as fast as
#: the design's slew rate allows. ``{strict}`` makes the plugin strict,
#: ``{sets}`` makes its sequence set StrictLimits itself, ``{evaluate}``
#: states the design as the evaluation's representative sequence.
PLUGIN = '''"""Bipolar lobes ramped at the design's slew rate."""

import pypulseqpp as pp

from pulserver.design import Evaluation, IntParam, SequencePlugin
from pulserver.protocol import UIParam


def lobes(system, nx=8):
    seq = pp.Sequence(system)
    lobe = pp.make_trapezoid("x", amplitude=0.9 * system.max_grad, flat_time=1e-3, system=system)
    for _ in range(4):
        seq.add_block(lobe)
        seq.add_block(pp.scale_grad(lobe, -1.0))
    seq.add_block(pp.make_delay(5e-3))
    if {sets}:
        seq.set_definition(key="StrictLimits", value=1)
    return seq


class Lobes(SequencePlugin):
    app = lobes
    protocol = {{UIParam.NX: IntParam("nx", range_max=100)}}
    strict_limits = {strict}

    def evaluate(self, system, protocol):
        if not {evaluate}:
            return Evaluation(protocol)
        return Evaluation(protocol, sequence=lobes(system, **protocol.arguments))
'''

BLOCK = f"{PROTOCOL_BEGIN}\nnx: 8\n{PROTOCOL_END}\n"


@pytest.fixture
def plugins(tmp_path):
    folder = tmp_path / "plugins"
    folder.mkdir()
    variants = {
        "auto": ("False", "False", "False"),
        "evaluated": ("False", "False", "True"),
        "strict": ("True", "False", "False"),
        "unannounced": ("False", "True", "False"),
    }
    for name, (strict, sets, evaluate) in variants.items():
        (folder / f"{name}.py").write_text(
            PLUGIN.format(strict=strict, sets=sets, evaluate=evaluate)
        )
    return folder


@pytest.fixture(autouse=True)
def _forget():
    service._LOWERED = _slew.Remembered()
    service._RESPONSES = _slew.Remembered(kind="nerve")
    yield
    service._LOWERED = _slew.Remembered()
    service._RESPONSES = _slew.Remembered(kind="nerve")


def _generate(tmp_path, plugins, plugin):
    store = DesignStore(tmp_path / "designs", f"scan-{plugin}")
    status, text = service.call(
        "generate",
        plugins=plugins,
        plugin=plugin,
        limits=SITE,
        block=BLOCK,
        store=store,
    )
    if status != 0:
        return None, None, text
    directory = store.directory(text.split()[1])
    manifest = json.loads((directory / "manifest.json").read_text())
    sequence = pp.Sequence()
    sequence.read(str(directory / "sequence.seq"))
    return manifest, sequence, text


def _pns(sequence):
    model = pp.safety.ChronaxieModel(SITE["pns_chronaxie"], SITE["pns_rheobase"])
    return pp.safety.check_pns(sequence, model, system=pp.Opts(**LIMITS))[1].peak.value


def test_a_design_past_the_nerve_limit_is_designed_again_at_a_lower_slew_rate(
    tmp_path, plugins
):
    design = service.design_system(SITE)
    assert _pns(service._plugin(service._variant(plugins, "auto")).app(design)) > 1.0
    manifest, sequence, _ = _generate(tmp_path, plugins, "auto")
    assert manifest["slew_passes"] >= 1
    assert manifest["strict_limits"] is False
    assert _pns(sequence) <= SITE["pns_limit"]
    designed = manifest["design_max_slew_t_per_m_per_s"]
    assert designed < design.max_slew / design.gamma
    assert sequence.definitions["DesignMaxSlew"][0] == pytest.approx(designed)
    assert "StrictLimits" not in sequence.definitions


def test_a_strict_plugin_is_refused_as_designed_and_says_so_in_its_files(
    tmp_path, plugins
):
    manifest, _, problem = _generate(tmp_path, plugins, "strict")
    assert manifest is None
    assert "PNS" in problem
    assert not list((tmp_path / "designs").rglob("sequence.seq"))
    relaxed = {**SITE, "pns_limit": 10.0}
    store = DesignStore(tmp_path / "relaxed", "scan")
    status, text = service.call(
        "generate",
        plugins=plugins,
        plugin="strict",
        limits=relaxed,
        block=BLOCK,
        store=store,
    )
    assert status == 0
    sequence = pp.Sequence()
    sequence.read(str(store.directory(text.split()[1]) / "sequence.seq"))
    assert sequence.definitions["StrictLimits"][0] == 1
    manifest = json.loads(
        (store.directory(text.split()[1]) / "manifest.json").read_text()
    )
    assert manifest["strict_limits"] is True
    assert manifest["slew_passes"] == 0


def test_a_sequence_setting_strict_limits_unannounced_is_refused_as_designed(
    tmp_path, plugins
):
    manifest, _, problem = _generate(tmp_path, plugins, "unannounced")
    assert manifest is None
    assert "PNS" in problem


def test_validate_reports_the_slew_rate_generate_designs_at(tmp_path, plugins):
    status, _ = service.call(
        "validate", plugins=plugins, plugin="evaluated", limits=SITE, block=BLOCK
    )
    assert status == 0
    manifest, sequence, _ = _generate(tmp_path, plugins, "evaluated")
    assert manifest["slew_passes"] == 0
    assert manifest["design_max_slew_t_per_m_per_s"] < LIMITS["max_slew"] * 0.9
    assert _pns(sequence) <= SITE["pns_limit"]


def test_a_lower_slew_rate_is_not_tried_for_an_amplitude_past_its_limit():
    assert _slew.exceeds({"pns": 1.2})
    assert not _slew.exceeds({"pns": 1.2, "grad": 1.1})
    assert not _slew.exceeds({"pns": 0.9, "slew": 0.8})


def test_two_designs_fix_the_nerve_response_of_a_ramp():
    # A response A S / (S + x): the proportional first step does not lower it
    # enough, and the line through two designs lands on the target.
    def response(slew):
        return 2.0 * slew / (slew + 100.0)

    first = 200.0
    second = _slew.lowered([(first, {"pns": response(first)})])
    assert response(second) > _slew.TARGET
    third = _slew.lowered(
        [(first, {"pns": response(first)}), (second, {"pns": response(second)})]
    )
    assert response(third) == pytest.approx(_slew.TARGET)


def test_the_check_reports_its_gradient_readings(tmp_path):
    seq = pp.Sequence(pp.Opts(**LIMITS))
    lobe = pp.make_trapezoid(
        "x", amplitude=0.5 * seq.system.max_grad, flat_time=1e-3, system=seq.system
    )
    seq.add_block(lobe)
    path = tmp_path / "lobe.seq"
    seq.write(str(path))
    readings = {}
    limits = ir.CheckLimits(pns=pp.safety.ChronaxieModel(360e-6, 38.0), pns_limit=0.5)
    ir.check(path, pp.Opts(**LIMITS), limits=limits, readings=readings)
    assert readings["grad"] == pytest.approx(0.5, rel=1e-3)
    assert 0.0 < readings["slew"] <= 1.0
    assert readings["pns"] > 0.0
    assert "vector_slew" not in readings


#: The limits of a 7 T scanner whose interpreter sends the hardware slew rate
#: as the design limit: a 3D FSE readout at full slew rate stimulates past
#: the nerve limit.
SEVEN_T = {
    "B0": 7.0,
    "max_grad": 100.0,
    "grad_unit": "mT/m",
    "max_slew": 200.0,
    "slew_unit": "T/m/s",
    "design_max_grad": 100.0,
    "design_max_slew": 200.0,
    "rf_raster_time": 2e-6,
    "grad_raster_time": 4e-6,
    "adc_raster_time": 2e-6,
    "block_duration_raster": 4e-6,
    "max_b1": 25.0,
    "b1_unit": "uT",
    "design_rf_dead_time": 232e-6,
    "design_rf_ringdown_time": 4e-6,
    "pns_chronaxie": 359e-6,
    "pns_rheobase": 71.62162,
    "pns_limit": 0.98,
}


def test_a_shipped_fse_at_the_hardware_slew_rate_is_designed_once_within_the_nerve_limit(
    tmp_path,
):
    """No validation searched the request: generate searches its echo train, then designs."""
    from pulserver import _plugins

    block = (
        f"{PROTOCOL_BEGIN}\nTE: 54000\nTR: 3431000\netl: 80\nbandwidth: 500000.0\n"
        "fov: 210.0\nphase_fov: 159.6\nnx: 276\nny: 32\nnslices: 16\n"
        f"slice_thickness: 0.8\nRy: 2\nRz: 2\n{PROTOCOL_END}\n"
    )
    store = DesignStore(tmp_path / "designs", "scan")
    status, text = service.call(
        "generate",
        plugins=(_plugins.SEQUENCES,),
        plugin="fse3d",
        limits=SEVEN_T,
        block=block,
        store=store,
    )
    assert status == 0, text
    directory = store.directory(text.split()[1])
    manifest = json.loads((directory / "manifest.json").read_text())
    sequence = pp.Sequence()
    sequence.read(str(directory / "sequence.seq"))
    model = pp.safety.ChronaxieModel(SEVEN_T["pns_chronaxie"], SEVEN_T["pns_rheobase"])
    assert manifest["slew_passes"] == 0
    assert manifest["design_max_slew_t_per_m_per_s"] < 0.9 * SEVEN_T["max_slew"]
    assert pp.safety.check_pns(sequence, model)[1].peak.value <= SEVEN_T["pns_limit"]


@pytest.mark.parametrize("plugin", ["fse3d", "fse_pd3d"])
def test_a_shipped_fse_is_lowered_at_validation_from_its_echo_train(tmp_path, plugin):
    """The evaluation's echo train is checked, and generate designs once where it left off.

    The train plays the views at the edges of k-space, whose phase encodes
    are the largest of the scan.
    """
    from pulserver import _plugins

    block = (
        f"{PROTOCOL_BEGIN}\nTR: 3431000\netl: 80\nbandwidth: 500000.0\n"
        "fov: 210.0\nphase_fov: 159.6\nnx: 276\nny: 32\nnslices: 16\n"
        f"slice_thickness: 0.8\nRy: 2\nRz: 2\n{PROTOCOL_END}\n"
    )
    common = {"plugins": (_plugins.SEQUENCES,), "plugin": plugin, "limits": SEVEN_T}
    status, text = service.call("validate", block=block, **common)
    assert status == 0, text
    variant = service._variant(common["plugins"], plugin, block)
    request = service._request(block, service._listing(variant))
    validated = service._LOWERED.get(_slew.key(variant, SEVEN_T, request))
    assert validated is not None
    assert validated < 0.9 * SEVEN_T["max_slew"] * 42.576e6
    store = DesignStore(tmp_path / "designs", "scan")
    status, text = service.call("generate", block=block, store=store, **common)
    assert status == 0, text
    directory = store.directory(text.split()[1])
    manifest = json.loads((directory / "manifest.json").read_text())
    sequence = pp.Sequence()
    sequence.read(str(directory / "sequence.seq"))
    model = pp.safety.ChronaxieModel(SEVEN_T["pns_chronaxie"], SEVEN_T["pns_rheobase"])
    assert manifest["slew_passes"] == 0
    # The manifest states the slew rate to four decimals.
    assert manifest["design_max_slew_t_per_m_per_s"] == pytest.approx(
        validated / 42.576e6, abs=1e-4
    )
    # The nerve response the search modelled its steps on leaves little unused.
    peak = pp.safety.check_pns(sequence, model)[1].peak.value
    assert peak >= 0.85 * SEVEN_T["pns_limit"]
    assert pp.safety.check_pns(sequence, model)[1].peak.value <= SEVEN_T["pns_limit"]


def _ramp(step, rheobase=70.0, chronaxie=360e-6, limit=0.98):
    """The nerve response of one ramp of ``step`` (T/m) over its limit, by slew rate (T/m/s)."""
    return lambda slew: step / (rheobase * (chronaxie + step / slew)) / limit


def test_one_design_fixes_the_nerve_response_of_one_ramp_under_the_nerve_model():
    # One ramp responds as A S / (S + x) with x = A times the rheobase and
    # the limit: the model lands on the target from a single reading.
    response = _ramp(80e-3)
    first = 180.0
    assert response(first) > 1.0
    one = [(first, {"pns": response(first)})]
    second = _slew.fitted(one, first * _slew.TARGET, nerve=(70.0, 0.98))
    assert second < first
    assert response(second) == pytest.approx(_slew.TARGET, rel=1e-9)
    # Without the nerve model the step is proportional, and falls short.
    assert response(_slew.lowered([(first, {"pns": response(first)})])) > 1.0


def test_one_design_fixes_the_nerve_response_where_an_earlier_search_found_it():
    def response(slew):
        return 2.0 * slew / (slew + 100.0)

    searched = [(200.0, {"pns": response(200.0)}), (150.0, {"pns": response(150.0)})]
    found = _slew.response_of(searched)
    assert found == pytest.approx(100.0)
    assert _slew.response_of(searched[:1]) is None
    # A later design is lowered, or raised, onto the target at once.
    lower = _slew.fitted([(180.0, {"pns": response(180.0)})], 171.0, response=found)
    assert response(lower) == pytest.approx(_slew.TARGET)
    raised = _slew.fitted([(50.0, {"pns": response(50.0)})], 180.0, response=found)
    assert 50.0 < raised < 180.0
    assert response(raised) == pytest.approx(_slew.TARGET)
    # A response that never reaches the target leaves the slew rate at its limit.
    assert _slew.fitted([(50.0, {"pns": 0.2})], 180.0, response=10.0) == 180.0


def test_a_request_past_the_nerve_limit_is_evaluated_twice(tmp_path, plugins):
    """At the design limits, then at the slew rate the nerve model gives for its reading."""
    asked = []
    real = service.design_system

    def recording(limits, **kwargs):
        system = real(limits, **kwargs)
        asked.append(round(system.max_slew / system.gamma, 3))
        return system

    plugin = service._plugin(service._variant(plugins, "evaluated"))
    evaluated = []
    hook = type(plugin).evaluate

    def counted(self, system, protocol):
        evaluated.append(round(system.max_slew / system.gamma, 3))
        return hook(self, system, protocol)

    type(plugin).evaluate = counted
    try:
        status, text = service.call(
            "validate", plugins=plugins, plugin="evaluated", limits=SITE, block=BLOCK
        )
    finally:
        type(plugin).evaluate = hook
    assert status == 0, text
    assert text.startswith("VALID")
    assert len(evaluated) == 2
    assert evaluated[1] < evaluated[0]
    sequence = plugin.app(service.design_system(SITE, max_slew=evaluated[1] * 42.576e6))
    assert 0.85 <= _pns(sequence) <= SITE["pns_limit"]


@pytest.mark.parametrize(
    ("plugin", "values"),
    [
        ("gre2d", {"nx": 64, "ny": 64}),
        ("gre3d", {"nx": 64, "ny": 64, "nslices": 16}),
        ("gre_multiecho3d", {"nx": 64, "ny": 64, "nslices": 16}),
        ("se3d", {"nx": 64, "ny": 64, "nslices": 16}),
        ("mprage3d", {"nx": 64, "ny": 64, "nslices": 32}),
        ("fse3d", {"nx": 64, "ny": 64, "nslices": 32, "etl": 32}),
        ("gre_stack_of_stars3d", {"nx": 64, "nslices": 16}),
        ("gre_radial2d", {"nx": 64}),
        ("se_propeller2d", {"nx": 64}),
        ("gre_spiral2d", {"nx": 64}),
    ],
)
def test_the_representative_sequence_of_a_shipped_plugin_bounds_the_nerve_response_of_its_scan(
    plugin, values
):
    """The views at the edges of k-space play the largest phase encodes of the scan."""
    from pulserver import _plugins
    from pulserver.design import load_plugin
    from pulserver.host._limits import split_limits

    system, _, checked = split_limits(SEVEN_T)
    design = service.design_system(SEVEN_T, max_slew=100.0 * system.gamma)
    scanner = load_plugin(_plugins.SEQUENCES / f"{plugin}.py")
    validation, protocol, representative = scanner._evaluated(design, values)
    assert validation.valid, validation.info
    built = scanner.generate(design, protocol)
    scan = built if isinstance(built, pp.Sequence) else built[-1]

    def read(sequence, rotation=None):
        return service._representative_readings(
            sequence, system, checked, rotation, False
        )

    assert read(representative)["pns"] >= 0.99 * read(scan)["pns"]
    # Oblique, either way round: the scan reads no more than the margin the
    # search holds a reading of the representative sequence to.
    for angles in ((35.0, -40.0, 25.0), (-35.0, 40.0, -25.0)):
        rotation = _rotation(*angles)
        bound, played = read(representative, rotation), read(scan, rotation)
        # A blade is designed as one line, without the phase encodes of its
        # others: its scan reads a fraction more than the margin.
        margin = 1.035 if "propeller" in plugin else 1.0005 / service._HELD
        for reading in ("pns", "slew", "grad"):
            assert played[reading] <= bound[reading] * margin, reading


def _rotation(about_x, about_y, about_z):
    """The rotation by the three angles, in degrees, about x, then y, then z."""
    import numpy as np

    a, b, c = np.radians([about_x, about_y, about_z])
    turn_x = [[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]
    turn_y = [[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]]
    turn_z = [[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]]
    return np.array(turn_z) @ np.array(turn_y) @ np.array(turn_x)
