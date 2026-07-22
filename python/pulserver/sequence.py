"""Compatibility alias for :mod:`pulserver.sequences`.

Use the plural namespace in new code::

    from pulserver.sequences import design_gre_2d
"""

from . import sequences as _sequences

__all__ = _sequences.__all__
globals().update({name: getattr(_sequences, name) for name in __all__})
