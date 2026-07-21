"""Core public API for pulserver plugin contracts and protocol types.

Examples
--------
>>> from pulserver._core import UIParam, TypeinFloatParam
>>> protocol = {UIParam.TR: TypeinFloatParam(value=500.0, unit="ms")}
>>> str(next(iter(protocol)))
'TR'
"""

from __future__ import annotations

from ._base import PulseqSequence
from ._cli import run_cli
from ._module import SequenceModule
from ._params import (
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
    set_protocol_value,
    validate_protocol,
    validate_protocol_entry,
)

Sequence = PulseqSequence

__all__ = [
    "PulseqSequence",
    "Sequence",
    "SequenceModule",
    "run_cli",
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
    "set_protocol_value",
]
