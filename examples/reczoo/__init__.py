"""Pulserver's reconstruction zoo: the recon each sequence is designed for.

``pyproject.toml`` packages this directory as :mod:`pulserver.reczoo`. A module
here holds one :class:`pulserver.ReconPlugin` subclass and the module-level
``PLUGIN`` instance the inline runtime discovers, so the same code runs offline
on arrays and online on an MRD stream::

    from pulserver import AcquisitionBucket, ReconContext
    from pulserver.reczoo import gre_2d

    bucket = AcquisitionBucket.from_arrays(kspace, labels={"kspace_encode_step_1": lines})
    image = gre_2d.PLUGIN(bucket, ReconContext.offline(header)).data

Calling the plugin replays the bucket through the lifecycle the inline runtime
drives, so the offline result is the streamed one. The MRD header comes along
because that is what sizes the buffers a plugin allocates.

Modules named after a :mod:`pulserver.seqzoo` sequence reconstruct that
sequence and read what it encoded — its counters and the flags marking where
its calibration block and its slices end. The rest are single-purpose examples
of one reconstruction primitive.
"""

from __future__ import annotations

import importlib

__all__ = [
    "bssfp_2d",
    "bssfp_3d",
    "fse_2d",
    "fse_3d",
    "gre_2d",
    "gre_3d",
    "gre_multiecho_2d",
    "gre_multiecho_3d",
    "inline_fft",
    "mprage_3d",
    "se_2d",
    "se_3d",
    "sms_epi",
    "subspace_basis",
]


def __getattr__(name: str):
    """Import one reconstruction module on first use."""
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return the reconstructions this zoo ships."""
    return sorted(__all__)
