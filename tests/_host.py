"""Limits and plugins the design tests share, and a design generated in-process."""

from pathlib import Path

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
GE_IR = {"ir_vendor": 2, "ir_label_column_map": "8 0 6", "ir_cache_ext": ".pge"}
LIMITS = {
    "max_grad": 40.0,
    "grad_unit": "mT/m",
    "max_slew": 150.0,
    "slew_unit": "T/m/s",
}


def value_block(values):
    lines = [f"{name}: {value}" for name, value in values.items()]
    return "\n".join([PROTOCOL_BEGIN, *lines, PROTOCOL_END]) + "\n"


def generate(store: DesignStore, plugin: str, values, limits=LIMITS, plugins=PLUGINS):
    """Return the identifier of the design a request generates.

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
    )
    assert status == 0, reply
    return reply.split()[1]
