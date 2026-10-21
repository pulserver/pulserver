"""Every shipped sequence, in every parameter set, is feasible at its default protocol on realistic scanners."""

import math
import sys

import pypulseqpp as pp
import pytest

from pulserver import _plugins
from pulserver.design import Protocol, load_plugin
from pulserver.host._limits import design_system
from pulserver.protocol import TRPreset

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

SHIPPED = sorted(path.stem for path in _plugins.SEQUENCES.glob("*.py"))
PARAMETERS = _plugins.SEQUENCES.parent / "parameters"

#: Each plugin with its shipped parameter sets, by ``<plugin>/<set>``.
VARIANTS = {f"{name}/default": (name, {}) for name in SHIPPED} | {
    f"{name}/{path.stem}": (name, tomllib.loads(path.read_text()))
    for name in SHIPPED
    for path in sorted((PARAMETERS / name).glob("*.toml"))
}


def _oblique(limits):
    """``limits`` with each logical axis designed for any prescription rotation."""
    return {
        **limits,
        "design_max_grad": limits["max_grad"] / math.sqrt(3.0),
        "design_max_slew": limits["max_slew"] / math.sqrt(3.0),
    }


#: A 7 T head gradient and a 3 T body gradient, designed for any orientation.
#: The 7 T transmit chain allows a peak B1 below the 30 uT of a 3 ms
#: refocusing pulse, which the spin echoes lengthen to fit under 0.9 of it.
SCANNERS = {
    "7T-head": _oblique(
        {
            "max_grad": 80.0,
            "grad_unit": "mT/m",
            "max_slew": 200.0,
            "slew_unit": "T/m/s",
            "B0": 7.0,
            "max_b1": 11.2,
        }
    ),
    "3T-body": _oblique(
        {
            "max_grad": 40.0,
            "grad_unit": "mT/m",
            "max_slew": 150.0,
            "slew_unit": "T/m/s",
            "B0": 3.0,
        }
    ),
}


@pytest.mark.parametrize(
    ("variant", "scanner"),
    [
        pytest.param(variant, scanner, id=f"{variant}-{scanner}")
        for variant in VARIANTS
        for scanner in SCANNERS
    ],
)
def test_the_default_protocol_of_every_variant_is_feasible(variant, scanner):
    name, options = VARIANTS[variant]
    plugin = load_plugin(_plugins.SEQUENCES / f"{name}.py", options)

    validation = plugin.validate(design_system(SCANNERS[scanner]), {})

    assert validation.valid, validation.info


@pytest.mark.parametrize("name", ["mprage3d", "fse3d"])
def test_a_navigated_scan_at_its_shortest_tr_plays_its_navigators(name):
    """The shortest TR of a scan asking for motion correction holds a navigator."""
    system = design_system(SCANNERS["3T-body"])
    small = {"nx": 64, "ny": 32, "nslices": 16, "TR": TRPreset.MINIMUM}

    def excitations(options):
        plugin = load_plugin(_plugins.SEQUENCES / f"{name}.py", options)
        wire = {k: e.value for k, e in plugin.listing().items() if e.editable}
        protocol = Protocol.from_wire(plugin.protocol, wire | small, system)
        designed = plugin.generate(system, protocol)
        main = designed if isinstance(designed, pp.Sequence) else designed[-1]
        instances = main.rf_instances()
        uses = [d.use for d in instances.definitions]
        return sum(uses[n] == "excitation" for n in instances.definition)

    assert excitations({"navigator": True}) > excitations({})
