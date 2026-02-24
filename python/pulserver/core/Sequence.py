"""Backwards-compatible re-export.

All public symbols have moved to dedicated modules:
  _sequence, _analysis, _acoustics, _pns
"""

__all__ = [
    "PulserverSequence",
    "find_tr",
    "find_segments_in_tr",
    "get_tr_gradient_waveforms",
    "get_tr_acoustic_spectra",
    "get_pns",
]

from ._sequence import PulserverSequence
from ._analysis import find_tr, find_segments_in_tr, get_tr_gradient_waveforms
from ._acoustics import get_tr_acoustic_spectra
from ._pns import get_pns
