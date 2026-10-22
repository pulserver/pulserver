"""Every shipped sequence, at every choice of each option, is feasible at its default protocol on realistic scanners."""

import itertools
import math

import pypulseqpp as pp
import pytest
from _host import OPTION_VARIANTS, needs_blochsim, shipped_options

from pulserver import _plugins
from pulserver.design import Protocol, load_plugin
from pulserver.host._limits import design_system
from pulserver.protocol import TRPreset

SHIPPED = sorted(path.stem for path in _plugins.SEQUENCES.glob("*.py"))

#: Each plugin at its defaults and at each other choice of each option, by
#: ``<plugin>/default`` and ``<plugin>/<option>=<choice>``.
VARIANTS = {f"{name}/default": (name, {}) for name in SHIPPED} | OPTION_VARIANTS


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
    if needs_blochsim(options):
        pytest.skip("needs blochsim")
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


@pytest.mark.parametrize("name", [n for n in SHIPPED if shipped_options(n)])
def test_a_user_entry_means_the_same_in_every_combination_of_options(name):
    """An option shows or hides a user entry, and never gives one another meaning."""
    entries = shipped_options(name).values()
    meanings = {}
    for combination in itertools.product(*(entry.choices for entry in entries)):
        options = {
            entry.name: choice
            for entry, choice in zip(entries, combination, strict=True)
        }
        if needs_blochsim(options):
            continue
        plugin = load_plugin(_plugins.SEQUENCES / f"{name}.py", options)
        for key, entry in plugin.protocol.items():
            if key.value.startswith("user"):
                meaning = getattr(entry, "argument", None) or getattr(
                    entry, "name", None
                )
                if meaning is not None:
                    meanings.setdefault(key, set()).add(meaning)

    assert {key: m for key, m in meanings.items() if len(m) > 1} == {}
