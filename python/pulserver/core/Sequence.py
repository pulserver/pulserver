"""
"""

__all__ = []

import copy

import pypulseq as pp

from ._extension._pulseqlib_wrapper import _PulserverSeqFile
from ._extension._pulseqlib_wrapper import _get_unique_blocks
from ._iostream import write_to_stream


class PulserverSequence(pp.Sequence):
    """ """

    def __init__(self, seq: pp.Sequence):
        object.__setattr__(self, '_seq', copy.deepcopy(seq))
        sys = seq.system
        cseq = _PulserverSeqFile(
            write_to_stream(seq),
            float(sys.B0),
            float(sys.max_grad),
            float(sys.max_slew),
            float(sys.rf_raster_time),
            float(sys.grad_raster_time),
            float(sys.adc_raster_time),
            float(sys.block_duration_raster),
        )
        object.__setattr__(self, '_cseq', cseq)

    def __getattribute__(self, name):
        # Always return PulserverSequence's own attributes/methods first
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            # Delegate to the underlying _seq
            return getattr(self._seq, name)

    def __setattr__(self, name, value):
        # Set PulserverSequence's own attributes, else delegate
        if name in ('_seq', '_cseq'):
            object.__setattr__(self, name, value)
        else:
            setattr(self._seq, name, value)

    def __str__(self):
        return str(self._seq)
    
def get_unique_blocks(seq: PulserverSequence):
    return _get_unique_blocks(seq._cseq)
