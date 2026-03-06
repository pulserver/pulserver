"""Sequence plugin contract and typed protocol helpers.

Public API::

    PulseqSequence      ABC that every plugin must subclass.
    UIParam             StrEnum of standard protocol keys.
    Validate            Per-parameter validation strategy.
    FloatParam          Numeric float parameter descriptor.
    IntParam            Numeric int parameter descriptor.
    BoolParam           Boolean toggle descriptor.
    StringListParam     Dropdown descriptor.
    Description         Read-only section header.
"""

__all__ = [
    "PulseqSequence",
    "UIParam",
    "Validate",
    "FloatParam",
    "IntParam",
    "BoolParam",
    "StringListParam",
    "Description",
    "Protocol",
    "ProtocolValue",
    "param_to_dict",
    "dict_to_param",
    "protocol_to_dict",
    "dict_to_protocol",
]

from ..core._base import PulseqSequence
from ..core._params import (
    BoolParam,
    Description,
    FloatParam,
    IntParam,
    Protocol,
    ProtocolValue,
    StringListParam,
    UIParam,
    Validate,
    dict_to_param,
    dict_to_protocol,
    param_to_dict,
    protocol_to_dict,
)
