"""Pulserver's shipped plugins: one complete, self-contained module each.

A plugin is a worked example of a whole stack, written the way you would write
it yourself. A sequence module composes :mod:`pulserver.design` modules into an
encoding plan and loops over it writing blocks; a reconstruction module takes
what the scanner sends back and turns it into images. Each carries two entry
points over one implementation:

``main(...)``
    A sequence module's whole design, under explicit keyword controls, written
    in the style of a PyPulseq example script: what to read, copy and edit.
``PLUGIN``
    The same thing behind the scanner protocol contract, so the bridge can
    offer it in the UI, and running the module as a script does the job
    offline. A reconstruction is stateful across the acquisitions it is fed,
    so a ``_recon`` module carries only this -- a
    :class:`pulserver.ReconPlugin`.

Either way the module itself is callable, and calling it does the module's
job: a sequence module designs a sequence, a reconstruction module
reconstructs a file. So the two families read the same way::

    from pulserver.app import gre3D_sequence, cartesian3D_recon

    seq = gre3D_sequence(n_x=128, n_y=128, n_z=64, slab_thickness=0.128)
    seq.write("gre3D.seq")

    images = cartesian3D_recon("scan.h5")

The reconstruction is not a different code path from the inline one: the file
holds what the scanner sent, and :meth:`pulserver.ReconPlugin.run` streams it
to the plugin in this process, through the same lifecycle hooks. No plugin
implements it and none overrides it.

The two families are also reachable as :mod:`pulserver.app.sequence` and
:mod:`pulserver.app.recon`, which is how the API reference groups them and how
``examples/sequence`` and ``examples/recon`` are laid out in the repository.
:mod:`pulserver.recon` is a different thing: the reconstruction toolbox these
plugins are built out of.
"""

from __future__ import annotations

import importlib
from types import ModuleType

__all__ = ["PluginModule", "recon", "sequence"]

_FAMILIES = {"_sequence": "sequence", "_recon": "recon"}


class PluginModule(ModuleType):
    """A plugin module, callable as the job it does.

    A sequence module defines ``main`` and calling it designs the sequence; a
    reconstruction module defines ``PLUGIN`` and calling it reconstructs an MRD
    file through :meth:`pulserver.ReconPlugin.run`.
    """

    def __call__(self, *args, **kwargs):
        main = getattr(self, "main", None)
        if main is not None:
            return main(*args, **kwargs)
        plugin = getattr(self, "PLUGIN", None)
        if plugin is not None:
            return plugin.run(*args, **kwargs)
        raise TypeError(f"{self.__name__!r} defines neither main nor PLUGIN")


def _family_of(name: str) -> str | None:
    """The subpackage a plugin name belongs to, by its suffix."""
    for suffix, family in _FAMILIES.items():
        if name.endswith(suffix):
            return family
    return None


def __getattr__(name: str):
    """Import one plugin, or one family, on first use."""
    if name in ("sequence", "recon"):
        return importlib.import_module(f"{__name__}.{name}")

    family = _family_of(name)
    if family is not None:
        package = importlib.import_module(f"{__name__}.{family}")
        if name in package.__all__:
            module = importlib.import_module(f"{__name__}.{family}.{name}")
            if type(module) is not PluginModule and (
                hasattr(module, "main") or hasattr(module, "PLUGIN")
            ):
                module.__class__ = PluginModule
            return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return every plugin, both families together."""
    names = list(__all__)
    for family in ("sequence", "recon"):
        names += importlib.import_module(f"{__name__}.{family}").__all__
    return sorted(names)
