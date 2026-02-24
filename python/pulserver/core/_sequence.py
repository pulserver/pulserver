"""PulserverSequence class — thin wrapper around pypulseq with C analysis backend."""

__all__ = ['PulserverSequence']

import copy

import pypulseq as pp

from ._extension._pulseqlib_wrapper import _PulseqCollection
from ._iostream import write_to_stream


class PulserverSequence(pp.Sequence):
    """
    Extended Sequence that provides TR / segment / safety analysis.

    Wraps a :class:`pypulseq.Sequence` and creates the C-backed
    ``_PulseqCollection`` used by all analysis functions.

    Parameters
    ----------
    seq : pp.Sequence
        Source pypulseq sequence (deep-copied internally).
    parse_labels : bool
        If ``True`` (default), parse label extensions during loading.
    num_averages : int
        Number of averages; influences scan-time calculations.
    """

    def __init__(
        self,
        seq: pp.Sequence,
        parse_labels: bool = True,
        num_averages: int = 1,
    ):
        object.__setattr__(self, '_seq', copy.deepcopy(seq))
        sys = seq.system
        cseq = _PulseqCollection(
            write_to_stream(seq),
            float(sys.gamma),
            float(sys.B0),
            float(sys.max_grad),
            float(sys.max_slew),
            float(sys.rf_raster_time),
            float(sys.grad_raster_time),
            float(sys.adc_raster_time),
            float(sys.block_duration_raster),
            parse_labels,
            num_averages,
        )
        object.__setattr__(self, '_cseq', cseq)

    def __getattribute__(self, name):
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            return getattr(self._seq, name)

    def __setattr__(self, name, value):
        if name in ('_seq', '_cseq'):
            object.__setattr__(self, name, value)
        else:
            setattr(self._seq, name, value)

    def __str__(self):
        return str(self._seq)
