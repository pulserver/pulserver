"""Loading the compiled kernels that ship inside the wheel.

``pulserver._ext`` is one extension module, built for each interpreter
Pulserver supports and holding every kernel as a submodule. A kernel that
will not import therefore means a broken installation -- most often a wheel
built for one interpreter imported from another -- and never an optional
extra a caller may reasonably do without.

:func:`require` says so, naming the ABI mismatch, rather than letting a
caller fall through to a slower Python path whose only symptom is
unexplained runtime.
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path
from typing import Any

__all__ = ["require"]

_MODULE = "pulserver._ext"


def _installed_tags() -> list[str]:
    """ABI tags of the builds of the extension present beside the package."""
    suffix = ".pyd" if sys.platform == "win32" else ".so"
    package = import_module("pulserver")
    tags = set()
    for directory in getattr(package, "__path__", ()):
        for path in Path(directory).glob(f"_ext.*{suffix}"):
            parts = path.name.split(".")
            if len(parts) >= 3:
                tags.add(parts[1])
    return sorted(tags)


def require(name: str, *, attribute: str | None = None) -> Any:
    """Load a bundled accelerator, or raise naming the mismatch.

    Parameters
    ----------
    name
        Submodule of ``pulserver._ext``, e.g. ``"recon_cpu"``.
    attribute
        Symbol the caller needs. Checked here, so a binary that predates
        the symbol fails at the load rather than at the call.

    Returns
    -------
    module or object
        The submodule, or the named attribute when ``attribute`` is given.

    Raises
    ------
    ImportError
        The accelerator is absent, was built for another interpreter, or
        does not provide ``attribute``.
    """
    try:
        module = import_module(f"{_MODULE}.{name}")
    except ImportError as error:
        installed = _installed_tags()
        found = ", ".join(installed) if installed else "none"
        raise ImportError(
            f"the bundled accelerator {name!r} did not load. Running "
            f"{sys.implementation.cache_tag}, builds present: {found}. "
            f"Pulserver ships this kernel for every interpreter it supports, "
            f"so install the distribution matching this interpreter; it will "
            f"not silently select a slower Python path."
        ) from error

    if attribute is None:
        return module

    member = getattr(module, attribute, None)
    if member is None:
        raise ImportError(
            f"the bundled accelerator {name!r} loaded but provides no "
            f"{attribute!r}: the binary is stale with respect to the Python "
            f"package. Rebuild or reinstall Pulserver."
        )
    return member
