"""Pulserver package root.

This namespace holds the *plugin contracts* only: the base classes sequence
and reconstruction plugins subclass, the abstract types those classes
exchange, the typed protocol parameters the scanner UI is built from, the
helpers that serialise and validate a protocol, and the offline CLI entry
point.

Everything used to *build* a sequence is deliberately **not** re-exported
here, so that the three roles stay visibly separate::

    import pulserver.pypulseq as pp          # events, Sequence, Opts
    import pulserver.design as design        # modules and scan loops
    from pulserver import Sequence, UIParam  # plugin contract
    from pulserver import ReconApp           # reconstruction plugin contract

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
    "AcquisitionBucket",
    "AcquisitionBucketStats",
    "BoolKey",
    "BoolParam",
    "Description",
    "DropdownFloatParam",
    "DropdownIntParam",
    "EncodingAxis",
    "EnumKey",
    "ExamCache",
    "FloatKey",
    "ImagingMode",
    "InputMode",
    "IntKey",
    "ParamKind",
    "PreparationType",
    "Protocol",
    "ProtocolValue",
    "PulseqSequence",
    "ReconApp",
    "ReconContext",
    "ReconResult",
    "ScanLoop",
    "Sequence",
    "SequenceModule",
    "SequenceType",
    "StringListParam",
    "TriggerType",
    "TypeinFloatParam",
    "TypeinIntParam",
    "UIParam",
    "Validate",
    "design",
    "dict_to_protocol",
    "examples",
    "io",
    "make_enum_param",
    "params",
    "protocol_to_dict",
    "pypulseq",
    "run_cli",
    "sequence",
    "sequences",
    "validate_protocol",
]


_CORE_MODULES = {"params"}
#: Authoring data types defined under ``design`` but named by the contract.
_AUTHORING_TYPES = {"ScanLoop", "EncodingAxis"}
_RECON_APP_TYPES = {
    "AcquisitionBucket",
    "AcquisitionBucketStats",
    "ExamCache",
    "ReconApp",
    "ReconContext",
    "ReconResult",
}
_SUBMODULES = {"io", "pypulseq", "design", "sequence", "sequences", "examples"}


def __getattr__(name: str):
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    if name in _CORE_MODULES:
        return importlib.import_module(f"{__name__}._core._protocol")
    if name in _AUTHORING_TYPES:
        return getattr(importlib.import_module(f"{__name__}.design._sampling"), name)
    if name in _RECON_APP_TYPES:
        return getattr(importlib.import_module(f"{__name__}.recon.app"), name)
    try:
        return getattr(importlib.import_module(f"{__name__}._core"), name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
