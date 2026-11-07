"""Import of a reconstruction plugin file."""

from __future__ import annotations

__all__ = ["load_plugin"]

import importlib.util
import sys
from pathlib import Path

from .._site_packages import manifest
from .plugin import ReconPlugin


def load_plugin(path: Path | str) -> ReconPlugin:
    """Import a plugin file and return the ``PLUGIN`` instance it defines.

    The module is registered as ``pulserver_recon_<file stem>``; loading another
    file with the same stem replaces it.

    Raises
    ------
    ValueError
        If the file defines no module-level ``PLUGIN``, or one that is not a
        :class:`ReconPlugin`.
    """
    path = Path(path)
    dependencies = manifest(path)
    name = f"pulserver_recon_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    plugin = getattr(module, "PLUGIN", None)
    if not isinstance(plugin, ReconPlugin):
        raise ValueError(f"{path} defines no module-level PLUGIN reconstruction")
    plugin._dependency_manifest = dependencies
    return plugin
