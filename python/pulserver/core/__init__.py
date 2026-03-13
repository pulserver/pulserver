"""Core public API for pulserver plugin contracts and protocol types."""

from __future__ import annotations

from ._base import PulseqSequence
from ._params import (
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

try:
    from ._extension._pulseqlib_wrapper import SequenceCollection, deserialize, serialize
except Exception as exc:  # pragma: no cover - environment-dependent import guard
    _IMPORT_ERROR = exc

    def _raise_extension_import_error():
        raise ImportError(
            "pulserver C extension is unavailable for this Python ABI. "
            "Install a compatible wheel or rebuild from source."
        ) from _IMPORT_ERROR

    class SequenceCollection:  # type: ignore[no-redef]
        def __init__(self, *_args, **_kwargs):
            _raise_extension_import_error()

    def serialize(*_args, **_kwargs):  # type: ignore[no-redef]
        _raise_extension_import_error()

    def deserialize(*_args, **_kwargs):  # type: ignore[no-redef]
        _raise_extension_import_error()


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
    "SequenceCollection",
    "serialize",
    "deserialize",
]
