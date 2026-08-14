"""Pulserver package root.

This namespace holds the *plugin contracts* only: the base classes sequence
and reconstruction plugins subclass, the abstract types those classes
exchange, the typed protocol parameters the scanner UI is built from, the
helpers that serialise and validate a protocol, and the offline CLI entry
point.

Everything used to *build* a sequence is deliberately **not** re-exported
here, so that the three roles stay visibly separate::

    import pulserver.pypulseq as pp       # events, Sequence, Opts
    import pulserver.design as design     # reusable sequence modules
    from pulserver import SequencePlugin  # sequence plugin contract
    from pulserver import ReconPlugin     # reconstruction plugin contract

:mod:`pulserver.pypulseq` is the event layer — upstream PyPulseq re-exported
whole, plus Pulserver's replacements for a few of its objects.
:mod:`pulserver.design` is the toolbox above it: every
:class:`SequenceModule` — an excitation, a preparation, one readout TR.

The two zoos are the worked examples built on those contracts:
:mod:`pulserver.seqzoo` holds one module per sequence, each with a ``main``
that returns a :class:`~pulserver.pypulseq.Sequence`, and
:mod:`pulserver.reczoo` the reconstruction that matches it, module name for
module name.

Examples
--------
>>> from pulserver import SequencePlugin, UIParam, TypeinFloatParam
>>> protocol = {UIParam.TR: TypeinFloatParam(value=500.0, unit="ms")}
>>> UIParam.TR in protocol
True

Notes
-----
The authoring modules (``io``, ``pypulseq`` and ``design``) require the
*optional* ``pypulseq`` dependency, so they are imported lazily (PEP 562):
``import pulserver`` — and hence ``pulserver.recon``, which runs in the scanner
recon env without ``pypulseq`` — stays import-clean; accessing an authoring name
pulls it in (raising a clear error if the extra is absent).
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
    "ReconContext",
    "ReconPlugin",
    "ReconResult",
    "SequenceModule",
    "SequencePlugin",
    "SequenceType",
    "StringListParam",
    "TriggerType",
    "TypeinFloatParam",
    "TypeinIntParam",
    "UIParam",
    "Validate",
    "design",
    "dict_to_protocol",
    "io",
    "make_enum_param",
    "params",
    "protocol_to_dict",
    "pypulseq",
    "reczoo",
    "run_cli",
    "seqzoo",
    "validate_protocol",
]


_CORE_MODULES = {"params"}
_RECON_PLUGIN_TYPES = {
    "AcquisitionBucket",
    "AcquisitionBucketStats",
    "ExamCache",
    "ReconContext",
    "ReconPlugin",
    "ReconResult",
}
_SUBMODULES = {"io", "pypulseq", "design", "seqzoo", "reczoo"}


def __getattr__(name: str):
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    if name in _CORE_MODULES:
        return importlib.import_module(f"{__name__}._core._protocol")
    if name in _RECON_PLUGIN_TYPES:
        return getattr(importlib.import_module(f"{__name__}.recon.plugin"), name)
    try:
        return getattr(importlib.import_module(f"{__name__}._core"), name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
