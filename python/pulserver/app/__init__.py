"""Pulserver's shipped plugins: one complete, self-contained module each.

A plugin is a worked example of a whole stack, written the way you would write
it yourself. A sequence module composes :mod:`pulserver.design` modules into an
encoding plan and loops over it writing blocks; a reconstruction module takes
what the scanner sends back and turns it into images. Each carries two entry
points over one implementation:

``main(...)``
    Explicit keyword controls. This is the whole plugin, written in the style
    of a PyPulseq example script, and it is what to read, copy and edit. The
    module itself is callable, so ``gre2D_sequence(...)`` is ``main(...)``.
``PLUGIN``
    A :class:`pulserver.SequencePlugin` or :class:`pulserver.ReconPlugin`
    wrapping that same ``main`` in the scanner protocol contract, so the bridge
    can offer it in the UI. Running the module as a script does the same job
    offline.

Everything lives in one namespace, named for what it does::

    from pulserver.app import gre3D_sequence, gre3D_recon

    seq = gre3D_sequence(n_x=128, n_y=128, n_z=64, slab_thickness=0.128)
    seq.write("gre3D.seq")

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
    """A plugin module, callable as the ``main`` it defines."""

    def __call__(self, *args, **kwargs):
        return self.main(*args, **kwargs)


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
            if hasattr(module, "main") and type(module) is not PluginModule:
                module.__class__ = PluginModule
            return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return every plugin, both families together."""
    names = list(__all__)
    for family in ("sequence", "recon"):
        names += importlib.import_module(f"{__name__}.{family}").__all__
    return sorted(names)
