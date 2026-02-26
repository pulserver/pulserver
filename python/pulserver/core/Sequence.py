"""Backwards-compatible re-export.

All public symbols have moved to dedicated modules:
  _sequence, _analysis, _acoustics, _pns
"""

__all__ = [
    "SequenceCollection",
    "pns",
    "grad_spectrum",
]

from ._sequence import SequenceCollection
from ._pns import pns
from ._acoustics import grad_spectrum
