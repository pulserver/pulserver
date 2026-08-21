"""The reconstruction half of the zoo: one recon per sampling.

Every module here is installed into :mod:`pulserver.app` and reached from
there. Each is callable, and calling it reconstructs an MRD file::

    from pulserver.app import cartesian2D_recon

    images = cartesian2D_recon("scan.h5")

There is deliberately no module per sequence. What a reconstruction needs to
know is how k-space was sampled, not what contrast the sequence was after: a
spin echo, a gradient echo and a balanced SSFP all leave one Cartesian grid
per slice, so one plugin serves all three. The modules are named for the
sampling they undo, and are built out of the :mod:`pulserver.recon` toolbox.

Every module here is one complete plugin and nothing else: one
:class:`pulserver.recon.ReconPlugin` subclass, its ``PLUGIN`` instance, and
the hooks. No module-level helper, no private method, no module that
demonstrates a step rather than reconstructing a scan -- a step that a plugin
needs is a name in :mod:`pulserver.recon`, general enough to compose in a hook,
because a local subroutine obscures which of the code is the mandatory hook and
which is this plugin's own detour.

Not to be confused with :mod:`pulserver.recon`, which is that toolbox rather
than the plugins written against it.
"""

from __future__ import annotations

import importlib

__all__ = [
    "cartesian2D_recon",
    "cartesian3D_recon",
    "epi2D_recon",
    "epi3D_recon",
    "noncartesian2D_recon",
    "noncartesian3D_recon",
    "noncartesian_stack_recon",
]


def __getattr__(name: str):
    """Import one reconstruction on first use."""
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return the reconstructions this package ships."""
    return sorted(__all__)
