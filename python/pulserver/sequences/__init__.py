"""Source-checkout forwarding module for :mod:`pulserver.sequences`.

Installed wheels map ``examples/sequences`` directly to this package path.
The implementation lives in ``_shim`` so this source-tree package and the
installed external-source package expose exactly the same public callbacks.
"""

from . import _shim

__all__ = _shim.__all__
globals().update({name: getattr(_shim, name) for name in __all__})
