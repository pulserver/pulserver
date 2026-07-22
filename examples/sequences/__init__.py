"""Public REPL shims and self-contained bridge sequence plugins.

``pyproject.toml`` packages this directory directly as
``pulserver.sequences``.  Individual plugin modules remain independently
callable by the bridge; this initializer only re-exports the explicit-keyword
REPL callbacks from the shared thin-shim implementation.
"""

import importlib.util

if importlib.util.find_spec(f"{__name__}._shim") is not None:
    from . import _shim

    __all__ = _shim.__all__
    globals().update({name: getattr(_shim, name) for name in __all__})
else:
    # In an uninstalled source checkout this directory has no local _shim;
    # the bridge imports its individual plugin modules directly instead.
    __all__ = []
