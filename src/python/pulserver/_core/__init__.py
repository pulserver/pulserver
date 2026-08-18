"""Core public API for pulserver plugin contracts and protocol types.

Examples
--------
>>> from pulserver._core import UIParam, TypeinFloatParam
>>> protocol = {UIParam.TR: TypeinFloatParam(value=500.0, unit="ms")}
>>> str(next(iter(protocol)))
'TR'
"""

from __future__ import annotations

from ._base import SequencePlugin
from ._cli import run_cli, write_sequence
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
    OffFloatParam,
    OffIntParam,
    ParamKind,
    PreparationType,
    Protocol,
    ProtocolValue,
    SequenceType,
    StringListParam,
    TEPreset,
    TriggerType,
    TRPreset,
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

# Last: it reads `pulserver.UIParam`, which resolves back through this module.
from ._protocol import main_kwargs

__all__ = [
    "SequencePlugin",
    "SequenceModule",
    "run_cli",
    "write_sequence",
    "main_kwargs",
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
    "TEPreset",
    "TRPreset",
    "TypeinFloatParam",
    "DropdownFloatParam",
    "TypeinIntParam",
    "DropdownIntParam",
    "OffFloatParam",
    "OffIntParam",
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
