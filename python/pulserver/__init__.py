"""Pulserver package root.

The package re-exports the most common protocol-building types at the top level
so sequence plugins can be written with short imports.

Examples
--------
>>> from pulserver import UIParam, TypeinFloatParam, PulseqSequence
>>> protocol = {UIParam.TR: TypeinFloatParam(value=500.0, unit="ms")}
>>> UIParam.TR in protocol
True

Notes
-----
Use ``pulserver.pulseq`` for sequence/event helpers and ``pulserver.io`` for
I/O utilities.

The authoring modules (``io``, ``pulseq``, ``core``) require the *optional*
``pypulseq`` dependency, so they are imported lazily (PEP 562): ``import
pulserver`` — and hence ``pulserver.recon``, which runs in the scanner recon
env without ``pypulseq`` — stays import-clean; accessing an authoring name pulls
it in (raising a clear error if the extra is absent).
"""

from __future__ import annotations

import importlib

__all__ = [
    "pulseq",
    "io",
    "arbgrad",
    "design",
    "PulseqSequence",
    "UIParam",
    "Validate",
    "ParamKind",
    "InputMode",
    "FloatKey",
    "IntKey",
    "BoolKey",
    "EnumKey",
    "SequenceType",
    "ImagingMode",
    "PreparationType",
    "TriggerType",
    "TypeinFloatParam",
    "DropdownFloatParam",
    "TypeinIntParam",
    "DropdownIntParam",
    "BoolParam",
    "StringListParam",
    "Description",
    "Protocol",
    "ProtocolValue",
    "expected_param_kind",
    "enum_options",
    "make_enum_param",
    "validate_protocol_entry",
    "validate_protocol",
    "param_to_dict",
    "dict_to_param",
    "protocol_to_dict",
    "dict_to_protocol",
]


def __getattr__(name: str):
    if name in ("io", "pulseq", "arbgrad", "design"):
        return importlib.import_module(f"{__name__}.{name}")
    if name in __all__:
        return getattr(importlib.import_module(f"{__name__}.core"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
