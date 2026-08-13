"""Pulserver's reconstruction zoo: the recon each sequence is designed for.

``pyproject.toml`` packages this directory as :mod:`pulserver.reczoo`. A module
here holds one :class:`pulserver.ReconApp` subclass and the module-level
``PLUGIN`` instance the inline runtime discovers, so the same code runs offline
on arrays and online on an MRD stream::

    import numpy as np
    from pulserver import AcquisitionBucket, ReconContext
    from pulserver.reczoo import gre_2d

    bucket = AcquisitionBucket.from_arrays(kspace, labels={"kspace_encode_step_1": lines})
    image = gre_2d.PLUGIN(bucket, ReconContext.offline()).data

Modules named after a :mod:`pulserver.seqzoo` sequence reconstruct that
sequence and read what it encoded — its counters, its calibration flags, the
echo position it recorded. The rest are single-purpose examples of one
reconstruction primitive.
"""

from __future__ import annotations

import importlib

__all__ = ["gre_2d", "inline_fft", "sms_epi", "subspace_basis"]


def __getattr__(name: str):
    """Import one reconstruction module on first use."""
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return the reconstructions this zoo ships."""
    return sorted(__all__)
