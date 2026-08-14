"""Torch-first MRI reconstruction algorithms and numerical building blocks.

The public API is organized by scientific role, following the module-oriented
style used by DeepInverse and MRI-NUFFT::

    import pulserver.recon as recon

    physics = recon.physics.NonCartesian2D(trajectory, (256, 256))
    image = recon.pics(kspace, physics)

Submodules are imported lazily so ``import pulserver.recon`` does not require
an MRD server environment or load optional numerical backends. Gadgetron/MRD
transport machinery is intentionally private under :mod:`pulserver.recon._mrd`.
"""

from __future__ import annotations

__all__ = [
    "AcquisitionBucket",
    "AcquisitionBucketStats",
    "ExamCache",
    "ReconContext",
    "ReconPlugin",
    "ReconResult",
    "calibration",
    "datasets",
    "denoisers",
    "diffusion_table",
    "execution",
    "has_acquisition_flag",
    "learned",
    "models",
    "motion",
    "optim",
    "physics",
    "pics",
    "plugin",
    "postprocessing",
    "preprocessing",
    "simulation",
    "weights",
]

import importlib
from typing import TYPE_CHECKING, Any

_SUBMODULES = (
    "calibration",
    "datasets",
    "denoisers",
    "execution",
    "learned",
    "models",
    "motion",
    "optim",
    "physics",
    "plugin",
    "postprocessing",
    "preprocessing",
    "simulation",
    "weights",
)
_PLUGIN_TYPES = {
    "AcquisitionBucket",
    "has_acquisition_flag",
    "AcquisitionBucketStats",
    "ExamCache",
    "ReconContext",
    "ReconPlugin",
    "ReconResult",
}


def __getattr__(name: str) -> Any:
    """Lazily resolve public reconstruction modules and entry points."""
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    if name in _PLUGIN_TYPES:
        return getattr(importlib.import_module(f"{__name__}.plugin"), name)
    if name == "pics":
        return importlib.import_module(f"{__name__}.optim").pics
    if name == "diffusion_table":
        # In _mrd because it reads an MRD header, public because reading one
        # is what a diffusion pipeline does first.
        return importlib.import_module(f"{__name__}._mrd.metadata").diffusion_table
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return the deliberately small public reconstruction namespace."""
    return sorted(__all__)


if TYPE_CHECKING:
    from . import calibration as calibration
    from . import datasets as datasets
    from . import denoisers as denoisers
    from . import execution as execution
    from . import learned as learned
    from . import models as models
    from . import motion as motion
    from . import optim as optim
    from . import physics as physics
    from . import plugin as plugin
    from . import postprocessing as postprocessing
    from . import preprocessing as preprocessing
    from . import simulation as simulation
    from . import weights as weights
    from .optim import pics as pics
    from .plugin import AcquisitionBucket as AcquisitionBucket
    from .plugin import AcquisitionBucketStats as AcquisitionBucketStats
    from .plugin import ExamCache as ExamCache
    from .plugin import has_acquisition_flag as has_acquisition_flag
    from .plugin import ReconContext as ReconContext
    from .plugin import ReconPlugin as ReconPlugin
    from .plugin import ReconResult as ReconResult
