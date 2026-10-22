"""Limits and plugins the design tests share, the shipped plugins' option choices, and a design generated in-process."""

import importlib.util
import math
from pathlib import Path

from pulserver import _plugins
from pulserver.design import OptionParam, load_plugin
from pulserver.host import DesignStore, call
from pulserver.protocol import PROTOCOL_BEGIN, PROTOCOL_END

PLUGINS = Path(__file__).parent / "plugins"
FIXTURES = Path(__file__).parent / "fixtures" / "sequences"
# The limits the IR fixtures convert under.
FIXTURE_LIMITS = {
    "max_grad": 40.0,
    "grad_unit": "mT/m",
    "max_slew": 170.0,
    "slew_unit": "T/m/s",
    "B0": 3.0,
    "rf_raster_time": 1e-6,
    "grad_raster_time": 1e-5,
    "adc_raster_time": 1e-7,
    "block_duration_raster": 1e-5,
}
VENDOR_IR = {"ir_vendor": 5, "ir_label_column_map": "8 7 6", "ir_cache_ext": ".cache"}
LIMITS = {
    "max_grad": 40.0,
    "grad_unit": "mT/m",
    "max_slew": 150.0,
    "slew_unit": "T/m/s",
    "B0": 3.0,
}
# Each logical axis designed at 1/sqrt(3) of LIMITS keeps every physical axis
# within them after any rotation.
ANY_ORIENTATION = {
    **LIMITS,
    "design_max_grad": LIMITS["max_grad"] / math.sqrt(3.0),
    "design_max_slew": LIMITS["max_slew"] / math.sqrt(3.0),
}

# A matrix small enough to scan here, at the reference's field of view. The
# balanced SSFP is read at a bandwidth whose readout outlasts the half of its
# excitation that follows the pulse centre. The gradient-echo spiral is read in
# enough interleaves that its image is not limited by undersampling. The zero
# echo time sequence is read at a bandwidth at which the dead-time gap after
# each pulse leaves no sample at the centre of k-space unacquired.
SMALL = {"nx": 32, "ny": 32, "fov": 220.0, "phase_fov": 220.0}
NONCARTESIAN = {"ny": None, "phase_fov": None}
SMALLER = {
    "bssfp2d": {"bandwidth": 25e3},
    "bssfp3d": {"nslices": 8, "bandwidth": 25e3},
    "gre_propeller2d": NONCARTESIAN,
    "gre_radial2d": NONCARTESIAN,
    "gre_spiral2d": {**NONCARTESIAN, "num_shots": 32},
    "se_epi_propeller2d": {**NONCARTESIAN, "bandwidth": 50e3},
    "se_propeller2d": NONCARTESIAN,
    "se_radial2d": NONCARTESIAN,
    "se_spiral2d": NONCARTESIAN,
    "epi2d": {"bandwidth": 50e3},
    "epi3d": {"nslices": 8},
    "fse3d": {"nslices": 8},
    "gre3d": {"nslices": 8},
    "gre_multiecho3d": {"nslices": 8, "num_echoes": 2},
    "se3d": {"nslices": 8},
    "gre_stack_of_blades3d": {**NONCARTESIAN, "nslices": 8},
    "gre_stack_of_stars3d": {**NONCARTESIAN, "nslices": 8},
    "gre_stack_of_spirals3d": {**NONCARTESIAN, "nslices": 8},
    "mprage3d": {"nslices": 8},
    "se_stack_of_blades3d": {**NONCARTESIAN, "nslices": 8},
    "se_stack_of_stars3d": {**NONCARTESIAN, "nslices": 8},
    "se_stack_of_spirals3d": {**NONCARTESIAN, "nslices": 8},
    "mprage_stack_of_stars3d": {**NONCARTESIAN, "nslices": 8},
    "mprage_stack_of_spirals3d": {**NONCARTESIAN, "nslices": 8},
    "zte3d": {**NONCARTESIAN, "bandwidth": 25e3},
}


def value_block(values):
    lines = [f"{name}: {value}" for name, value in values.items()]
    return "\n".join([PROTOCOL_BEGIN, *lines, PROTOCOL_END]) + "\n"


def shipped_options(name):
    """The option entries of the shipped plugin ``name`` at its defaults, by key."""
    plugin = load_plugin(_plugins.SEQUENCES / f"{name}.py")
    return {
        key: entry
        for key, entry in plugin.protocol.items()
        if isinstance(entry, OptionParam)
    }


#: Each shipped plugin with every other choice of each of its options, one
#: option at a time, by ``<plugin>/<option>=<choice>``.
OPTION_VARIANTS = {
    f"{name}/{entry.name}={choice}": (name, {entry.name: choice})
    for name in sorted(path.stem for path in _plugins.SEQUENCES.glob("*.py"))
    for entry in shipped_options(name).values()
    for choice in entry.choices
    if choice != entry.value
}


def needs_blochsim(options):
    """Whether a shipped plugin loaded with ``options`` designs its refocusing trains with blochsim, and blochsim is missing."""
    return (
        options.get("refocusing", "constant") != "constant"
        and importlib.util.find_spec("blochsim") is None
    )


def generate(
    store: DesignStore,
    plugin: str,
    values,
    limits=LIMITS,
    plugins=PLUGINS,
    push=None,
):
    """Return the identifier of the design a request generates, pushed to ``push``.

    Raises
    ------
    AssertionError
        If the call replies ``ERROR``.
    """
    status, reply = call(
        "generate",
        plugins=plugins,
        plugin=plugin,
        limits=limits,
        block=value_block(values),
        store=store,
        push=push,
    )
    assert status == 0, reply
    return reply.split()[1]
