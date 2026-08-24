"""Pulserver: a Pulseq sequence from design to acquisition to image.

The package is five namespaces, one per role, and nothing else at the root::

    import pulserver.pypulseq as pp    # events, Sequence, Opts, files
    import pulserver.design as design  # sequence modules and the scanner protocol
    import pulserver.app as app        # whole sequences and their reconstructions
    import pulserver.recon as recon    # the reconstruction stack
    import pulserver.mrd as mrd        # the data model both ends share

:mod:`pulserver.pypulseq` is the event layer -- upstream PyPulseq re-exported
whole, plus Pulserver's replacements for a few of its objects.
:mod:`pulserver.design` is the toolbox above it: every
:class:`~pulserver.design.SequenceModule` -- an excitation, a preparation, one
readout TR -- together with the contract a scanner drives a plugin through.
:mod:`pulserver.app` is what those build: one module per sequence, each callable
and each with a ``main``, beside the reconstruction that matches it under the
same name. :mod:`pulserver.recon` reconstructs the stream a scan produces, in
the vocabulary :mod:`pulserver.mrd` holds for both ends of it.

Examples
--------
>>> import pulserver
>>> pulserver.__all__
['app', 'design', 'mrd', 'pypulseq', 'recon']

Notes
-----
The authoring namespaces require the *optional* ``pypulseq`` dependency, so
every one of them is imported lazily (PEP 562): ``import pulserver`` -- and
hence :mod:`pulserver.recon`, which runs in the scanner recon environment
without ``pypulseq`` -- stays import-clean, and reaching for an authoring name
pulls it in, raising a clear error if the extra is absent.
"""

from __future__ import annotations

import importlib

__all__ = ["app", "design", "mrd", "pypulseq", "recon"]


def __getattr__(name: str):
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
