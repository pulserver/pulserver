"""
"""

__all__ = []

import copy

import pypulseq as pp

from ._extension._pulseqlib_wrapper import _PulserverSeqFile

class PulserverSequence(pp.Sequence):
    """
    """
    def __init__(self, seq: pp.Sequence):
        object.__setattr__(self, "_seq", copy.deepcopy(seq))
        object.__setattr__(self, "_cseq", _PulserverSeqFile(seq))

    def __getattribute__(self, name):
        # Always return PulserverSequence's own attributes/methods first
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            # Delegate to the underlying _seq
            return getattr(self._seq, name)

    def __setattr__(self, name, value):
        # Set PulserverSequence's own attributes, else delegate
        if name in ("_seq", "_cseq"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._seq, name, value)
            
    def __str__(self):
        return str(self._seq)
