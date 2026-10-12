"""Load the compiled extension shipped inside the wheel."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path
from typing import Any

__all__ = ["require"]

_MODULE = "pulserver._ext"


def _installed_tags() -> list[str]:
    """Return the ABI tags of the extension builds present beside the package."""
    suffix = ".pyd" if sys.platform == "win32" else ".so"
    package = import_module("pulserver")
    tags = set()
    for directory in getattr(package, "__path__", ()):
        for path in Path(directory).glob(f"_ext.*{suffix}"):
            parts = path.name.split(".")
            if len(parts) >= 3:
                tags.add(parts[1])
    return sorted(tags)


def require(attribute: str | None = None) -> Any:
    """Load the bundled extension, or raise naming the mismatch.

    The wheel ships the extension for every supported interpreter, so a failed
    import means a broken installation, never an optional feature.

    Parameters
    ----------
    attribute : str, default=None
        Symbol the caller needs, checked at load so a binary that predates it
        fails here rather than at the call.

    Returns
    -------
    module or object
        The extension module, or the named attribute when one is given.

    Raises
    ------
    ImportError
        The extension is absent, was built for another interpreter, or does not
        provide ``attribute``.
    """
    try:
        module = import_module(_MODULE)
    except ImportError as error:
        installed = _installed_tags()
        found = ", ".join(installed) if installed else "none"
        raise ImportError(
            f"the bundled extension did not load. Running "
            f"{sys.implementation.cache_tag}, builds present: {found}."
        ) from error
    if attribute is None:
        return module
    try:
        return getattr(module, attribute)
    except AttributeError as error:
        raise ImportError(
            f"the bundled extension does not provide {attribute!r}; "
            f"the installed binary predates it."
        ) from error
