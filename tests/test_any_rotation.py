"""A scan that follows motion-correction poses is designed and checked for every rotation."""

import json
import math

import pypulseqpp as pp
import pytest
from _host import LIMITS

from pulserver import ir
from pulserver.host import DesignStore
from pulserver.host import _service as service
from pulserver.protocol import PROTOCOL_BEGIN, PROTOCOL_END

#: Each logical axis at full scale for the prescribed rotation, at 1/sqrt(3)
#: of it for any rotation.
SITE = {
    **LIMITS,
    "design_max_grad": LIMITS["max_grad"],
    "design_max_slew": LIMITS["max_slew"],
    "design_max_grad_any_rotation": LIMITS["max_grad"] / math.sqrt(3.0),
    "design_max_slew_any_rotation": LIMITS["max_slew"] / math.sqrt(3.0),
}

#: Three axes at nine tenths of the design's limit at once: within every
#: axis's own limit whatever the limits, but summed over the axes past them
#: unless designed for every rotation. ``{wrap}`` decides whether the
#: function states, before designing, that it sets EnablePmc.
PLUGIN = '''"""Three gradient axes at once{doc}."""

import pypulseqpp as pp

from pulserver._zoo._pmc import navigated
from pulserver.design import IntParam, SequencePlugin
from pulserver.protocol import UIParam


def lobes(system, nx=8, navigator=False):
    seq = pp.Sequence(system)
    amplitude = 0.9 * system.max_grad
    lobe = [pp.make_trapezoid(axis, amplitude=amplitude, flat_time=1e-3, system=system) for axis in "xyz"]
    seq.add_block(*lobe)
    seq.add_block(*[pp.scale_grad(g, -1.0) for g in lobe])
    if {sets}:
        seq.set_definition(key="EnablePmc", value=1)
    return seq


class Lobes(SequencePlugin):
    app = {wrap}
    protocol = {{UIParam.NX: IntParam("nx", range_max=100)}}
'''


@pytest.fixture
def plugins(tmp_path):
    folder = tmp_path / "plugins"
    folder.mkdir()
    (folder / "navigated.py").write_text(
        PLUGIN.format(doc=", navigated", sets="False", wrap="navigated(lobes)")
    )
    (folder / "plain.py").write_text(PLUGIN.format(doc="", sets="False", wrap="lobes"))
    (folder / "undeclared.py").write_text(
        PLUGIN.format(
            doc=", setting EnablePmc without saying so", sets="True", wrap="lobes"
        )
    )
    return folder


def _generate(tmp_path, plugins, plugin, pmc):
    store = DesignStore(tmp_path / "designs", f"scan-{plugin}-{pmc}")
    status, text = service.call(
        "generate",
        plugins=plugins,
        plugin=plugin,
        limits={**SITE, "pmc_enable": pmc},
        block=f"{PROTOCOL_BEGIN}\nnx: 8\n{PROTOCOL_END}\n",
        store=store,
    )
    if status != 0:
        return None, text
    design = text.split()[1]
    directory = store.directory(design)
    manifest = json.loads((directory / "manifest.json").read_text())
    sequence = pp.Sequence()
    sequence.read(str(directory / "sequence.seq"))
    return manifest, sequence


def _peak_per_axis(sequence):
    _, report = pp.safety.check_max_grad(sequence, pp.Opts(**LIMITS))
    return report.per_axis.value


def test_a_navigated_scan_at_a_site_following_poses_is_designed_for_every_rotation(
    tmp_path, plugins
):
    manifest, sequence = _generate(tmp_path, plugins, "navigated", 1)
    assert manifest["design_rotation"] == "any"
    any_rotation = pp.Opts(
        max_grad=SITE["design_max_grad_any_rotation"], grad_unit="mT/m"
    ).max_grad
    assert _peak_per_axis(sequence) <= any_rotation


def test_the_same_scan_at_a_site_not_following_poses_keeps_the_prescribed_limits(
    tmp_path, plugins
):
    manifest, sequence = _generate(tmp_path, plugins, "navigated", 0)
    assert manifest["design_rotation"] == "prescribed"
    any_rotation = pp.Opts(
        max_grad=SITE["design_max_grad_any_rotation"], grad_unit="mT/m"
    ).max_grad
    assert _peak_per_axis(sequence) > any_rotation


def test_a_scan_without_motion_correction_keeps_the_prescribed_limits_at_a_site_following_poses(
    tmp_path, plugins
):
    manifest, _ = _generate(tmp_path, plugins, "plain", 1)
    assert manifest["design_rotation"] == "prescribed"


def test_a_sequence_setting_enable_pmc_unannounced_is_refused_where_poses_could_turn_it(
    tmp_path, plugins
):
    manifest, problem = _generate(tmp_path, plugins, "undeclared", 1)
    assert manifest is None
    assert "summed over the axes" in problem
    assert (
        _generate(tmp_path, plugins, "undeclared", 0)[0]["design_rotation"]
        == "prescribed"
    )


def _three_axes(enable_pmc):
    system = pp.Opts(**LIMITS)
    design = pp.Opts(**{**LIMITS, "max_slew": 0.8 * LIMITS["max_slew"]})
    seq = pp.Sequence(system)
    lobe = [
        pp.make_trapezoid(
            a, amplitude=0.9 * system.max_grad, flat_time=1e-3, system=design
        )
        for a in "xyz"
    ]
    seq.add_block(*lobe)
    seq.add_block(*[pp.scale_grad(g, -1.0) for g in lobe])
    if enable_pmc:
        seq.set_definition(key="EnablePmc", value=1)
    return seq


@pytest.mark.parametrize(
    ("follows_poses", "enable_pmc", "refused"),
    [(True, True, True), (False, True, False), (True, False, False)],
)
def test_the_vector_is_held_to_the_axis_limit_only_where_poses_may_turn_the_scan(
    tmp_path, follows_poses, enable_pmc, refused
):
    seq = _three_axes(enable_pmc)
    path = tmp_path / "three.seq"
    seq.write(str(path))
    problems = ir.check(
        path, pp.Opts(**LIMITS), limits=ir.CheckLimits(follows_poses=follows_poses)
    )
    summed = [p for p in problems if "summed over the axes" in p]
    assert bool(summed) == refused
    assert not [p for p in problems if "exceeds" in p and "summed" not in p]
