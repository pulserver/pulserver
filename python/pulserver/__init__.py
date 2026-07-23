"""Pulserver package root.

This namespace holds the *plugin contract* only: the base classes a sequence
plugin subclasses, the abstract types those classes exchange, the typed
protocol parameters its UI is built from, the helpers that serialise and
validate a protocol, and the offline CLI entry point.

Everything used to *build* a sequence is deliberately **not** re-exported
here, so that the three roles stay visibly separate::

    import pulserver.pypulseq as pp          # events, Sequence, Opts
    import pulserver.design as design        # modules and scan loops
    from pulserver import Sequence, UIParam  # plugin contract

:mod:`pulserver.pypulseq` is the event layer — upstream PyPulseq re-exported
whole, plus Pulserver's replacements for a few of its objects.
:mod:`pulserver.design` is the toolbox above it: every factory returning a
:class:`SequenceModule` or a :class:`ScanLoop`.

Examples
--------
>>> from pulserver import Sequence, UIParam, TypeinFloatParam
>>> protocol = {UIParam.TR: TypeinFloatParam(value=500.0, unit="ms")}
>>> UIParam.TR in protocol
True

Notes
-----
The authoring modules (``io``, ``pypulseq`` and ``design``) require the
*optional* ``pypulseq`` dependency, so they are imported lazily (PEP 562):
``import pulserver`` — and hence ``pulserver.recon``, which runs in the scanner
recon env without ``pypulseq`` — stays import-clean; accessing an authoring name
pulls it in (raising a clear error if the extra is absent). ``ScanLoop`` and
``EncodingAxis`` are authoring types and load the same way.
"""

from __future__ import annotations

import importlib

__all__ = [
    "pypulseq",
    "design",
    "io",
    "sequence",
    "sequences",
    "params",
    "Sequence",
    "PulseqSequence",
    "SequenceModule",
    "ScanLoop",
    "EncodingAxis",
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
    "make_enum_param",
    "validate_protocol",
    "protocol_to_dict",
    "dict_to_protocol",
]


_CORE_MODULES = {"params"}
#: Authoring data types defined under ``design`` but named by the contract.
_AUTHORING_TYPES = {"ScanLoop", "EncodingAxis"}
_SUBMODULES = {"io", "pypulseq", "design", "sequence", "sequences"}


def __getattr__(name: str):
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    if name in _CORE_MODULES:
        return importlib.import_module(f"{__name__}._core._protocol")
    if name in _AUTHORING_TYPES:
        return getattr(importlib.import_module(f"{__name__}.design._sampling"), name)
    try:
        return getattr(importlib.import_module(f"{__name__}._core"), name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
