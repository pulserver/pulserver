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
    yield
    service._LOWERED = _slew.Remembered()


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


def test_a_shipped_fse_at_the_hardware_slew_rate_is_designed_within_the_nerve_limit(
    tmp_path,
):
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
    assert manifest["slew_passes"] >= 1
    assert manifest["design_max_slew_t_per_m_per_s"] < 0.9 * SEVEN_T["max_slew"]
    assert pp.safety.check_pns(sequence, model)[1].peak.value <= SEVEN_T["pns_limit"]


@pytest.mark.parametrize("plugin", ["fse3d", "fse_pd3d"])
def test_a_shipped_fse_is_lowered_at_validation_from_its_echo_train(tmp_path, plugin):
    """The evaluation's echo train is checked, and generate starts where it left off.

    The train acquires fewer peripheral views than the scan, so the scan's
    partition encodes may lower the slew rate once more at generation.
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
    lowered = list(service._LOWERED._slews.values())
    assert lowered and max(lowered) < 0.9 * SEVEN_T["max_slew"] * 42.576e6
    store = DesignStore(tmp_path / "designs", "scan")
    status, text = service.call("generate", block=block, store=store, **common)
    assert status == 0, text
    directory = store.directory(text.split()[1])
    manifest = json.loads((directory / "manifest.json").read_text())
    sequence = pp.Sequence()
    sequence.read(str(directory / "sequence.seq"))
    model = pp.safety.ChronaxieModel(SEVEN_T["pns_chronaxie"], SEVEN_T["pns_rheobase"])
    assert manifest["slew_passes"] <= 1
    assert manifest["design_max_slew_t_per_m_per_s"] * 42.576e6 <= max(lowered)
    assert pp.safety.check_pns(sequence, model)[1].peak.value <= SEVEN_T["pns_limit"]
