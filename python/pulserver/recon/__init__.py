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
    "ReconApp",
    "ReconContext",
    "ReconResult",
    "algorithms",
    "app",
    "calibration",
    "corrections",
    "denoisers",
    "density",
    "diffusion_table",
    "execution",
    "motion",
    "optim",
    "physics",
    "pics",
    "preprocessing",
    "simulation",
]

import importlib
from typing import TYPE_CHECKING, Any

_SUBMODULES = (
    "algorithms",
    "app",
    "calibration",
    "corrections",
    "denoisers",
    "density",
    "execution",
    "motion",
    "optim",
    "physics",
    "preprocessing",
    "simulation",
)
_APP_TYPES = {
    "AcquisitionBucket",
    "AcquisitionBucketStats",
    "ExamCache",
    "ReconApp",
    "ReconContext",
    "ReconResult",
}


def __getattr__(name: str) -> Any:
    """Lazily resolve public reconstruction modules and entry points."""
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    if name in _APP_TYPES:
        return getattr(importlib.import_module(f"{__name__}.app"), name)
    if name == "pics":
        return importlib.import_module(f"{__name__}.algorithms").pics
    if name == "diffusion_table":
        # In _mrd because it reads an MRD header, public because reading one
        # is what a diffusion pipeline does first.
        return importlib.import_module(f"{__name__}._mrd.metadata").diffusion_table
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return the deliberately small public reconstruction namespace."""
    return sorted(__all__)


if TYPE_CHECKING:
    from . import app as app
    from . import algorithms as algorithms
    from . import calibration as calibration
    from . import corrections as corrections
    from . import denoisers as denoisers
    from . import density as density
    from . import execution as execution
    from . import motion as motion
    from . import optim as optim
    from . import physics as physics
    from . import preprocessing as preprocessing
    from . import simulation as simulation
    from .algorithms import pics as pics
    from .app import AcquisitionBucket as AcquisitionBucket
    from .app import AcquisitionBucketStats as AcquisitionBucketStats
    from .app import ExamCache as ExamCache
    from .app import ReconApp as ReconApp
    from .app import ReconContext as ReconContext
    from .app import ReconResult as ReconResult
