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

import importlib
from typing import TYPE_CHECKING, Any

_SUBMODULES = (
    "algorithms",
    "calibration",
    "corrections",
    "denoisers",
    "density",
    "execution",
    "physics",
    "preprocessing",
    "simulation",
)

__all__ = [*_SUBMODULES, "pics"]


def __getattr__(name: str) -> Any:
    """Lazily resolve public reconstruction modules and entry points."""
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    if name == "pics":
        return importlib.import_module(f"{__name__}.algorithms").pics
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return the deliberately small public reconstruction namespace."""
    return sorted(__all__)


if TYPE_CHECKING:
    from . import algorithms as algorithms
    from . import calibration as calibration
    from . import corrections as corrections
    from . import denoisers as denoisers
    from . import density as density
    from . import execution as execution
    from . import physics as physics
    from . import preprocessing as preprocessing
    from . import simulation as simulation
    from .algorithms import pics as pics
