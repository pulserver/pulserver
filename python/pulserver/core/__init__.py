"""Core public API for pulserver plugin contracts and protocol types.

Examples
--------
>>> from pulserver.core import UIParam, TypeinFloatParam
>>> protocol = {UIParam.TR: TypeinFloatParam(value=500.0, unit="ms")}
>>> str(next(iter(protocol)))
'TR'
"""

from __future__ import annotations

from ._base import PulseqSequence
from ._params import (
    BoolParam,
    BoolKey,
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
    SequenceType,
    StringListParam,
    TypeinFloatParam,
    TypeinIntParam,
    TriggerType,
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
