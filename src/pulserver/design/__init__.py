"""Binding of pypulseqpp sequence applications to the scanner protocol."""

# The protocol names a plugin's ``ui`` is written with, so that a plugin imports
# from one module. Documented with :mod:`pulserver.protocol`.
from ..protocol import TEPreset, TRPreset, UIParam
from ._scanner import (
    BoolParam,
    ChoiceParam,
    ConfigParam,
    Description,
    FloatParam,
    IntParam,
    Protocol,
    ScannerSequence,
    StringListParam,
    TimeParam,
    load_plugin,
)

__all__ = [
    "BoolParam",
    "ChoiceParam",
    "ConfigParam",
    "Description",
    "FloatParam",
    "IntParam",
    "Protocol",
    "ScannerSequence",
    "StringListParam",
    "TimeParam",
    "load_plugin",
]
