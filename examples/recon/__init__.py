"""The reconstruction half of :mod:`pulserver.app`: one recon per sequence.

``pyproject.toml`` packages this directory as :mod:`pulserver.app.recon`, and
every module here is reachable from :mod:`pulserver.app` directly. Each takes
what the scanner sends back for one :mod:`pulserver.app.sequence` module and
turns it into images, built out of the :mod:`pulserver.recon` toolbox::

    from pulserver.app import gre2D_recon

Not to be confused with :mod:`pulserver.recon`, which is that toolbox rather
than the plugins written against it.
"""

from __future__ import annotations

import importlib

__all__ = [
    "bssfp2D_recon",
    "bssfp3D_recon",
    "epi2D_recon",
    "epi3D_recon",
    "fse2D_recon",
    "fse3D_recon",
    "gre2D_recon",
    "gre3D_recon",
    "gre_multiecho2D_recon",
    "gre_multiecho3D_recon",
    "gre_radial2D_recon",
    "gre_spiral2D_recon",
    "gre_stack_of_spirals3D_recon",
    "gre_stack_of_stars3D_recon",
    "mprage3D_recon",
    "mprage_stack_of_spirals3D_recon",
    "se2D_recon",
    "se3D_recon",
    "se_propeller2D_recon",
    "subspace_basis_recon",
    "zte3D_recon",
]


def __getattr__(name: str):
    """Import one reconstruction on first use."""
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return the reconstructions this package ships."""
    return sorted(__all__)
