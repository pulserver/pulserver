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
"""

from . import (
    io,  # noqa: F401
    pulseq,  # noqa: F401
)
from .core import (  # noqa: F401
    BoolKey,
    BoolParam,
    Description,
    DropdownFloatParam,
    DropdownIntParam,
    EnumKey,
    FloatKey,
    ImagingMode,
    InputMode,
    IntKey,
    ParamKind,
    PreparationType,
    Protocol,
    ProtocolValue,
    PulseqSequence,
    SequenceType,
    StringListParam,
    TriggerType,
    TypeinFloatParam,
    TypeinIntParam,
    UIParam,
    Validate,
    dict_to_param,
    dict_to_protocol,
    enum_options,
    expected_param_kind,
    make_enum_param,
    param_to_dict,
    protocol_to_dict,
    validate_protocol,
    validate_protocol_entry,
)

__all__ = [
    "pulseq",
    "io",
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
