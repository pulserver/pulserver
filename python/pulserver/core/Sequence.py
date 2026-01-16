""" """

__all__ = []

import copy
from types import SimpleNamespace

import pypulseq as pp

from ._extension._pulseqlib_wrapper import (
    _find_tr_in_sequence,
    _find_segments_in_tr,
    _get_unique_blocks,
    _PulserverSeqFile,
)
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
    unique_blocks, unique_table, _, _, _, _ = _get_unique_blocks(seq._cseq)
    return unique_blocks, unique_table


def find_tr(seq: PulserverSequence, num_reps: int = 1) -> SimpleNamespace:
    _, unique_table, block_durations_us, pure_delay_block, num_prep, num_cooldown = (
        _get_unique_blocks(seq._cseq)
    )
    tr_size, num_trs, degenerate_prep, degenerate_cooldown = _find_tr_in_sequence(unique_table, block_durations_us, pure_delay_block, num_prep, num_cooldown)

    # Prepare result
    result = SimpleNamespace()
        
    # Special case: only one TR
    if num_reps == 1 and num_trs == 1:
        result.main_tr = seq._seq
        result.first_rep_first_tr = None
        result.last_rep_last_tr = None
        return result
        
    # Prepare main TR
    result.main_tr = pp.Sequence(system=seq.system)
    
    # Prepare first and last repetitions header and footer blocks
    if degenerate_prep or num_prep == 0:
        result.first_rep_first_tr = None
    else:
        result.first_rep_first_tr = pp.Sequence(system=seq.system)
    if degenerate_cooldown or num_cooldown == 0:
        result.last_rep_last_tr = None
    else:
        result.last_rep_last_tr = pp.Sequence(system=seq.system)
        
    # Add first repetition header blocks
    if result.first_rep_first_tr is not None:
        for n in range(num_prep):
            block = seq._seq.get_block(n + 1)
            result.first_rep_first_tr.add_block(block)
    
    # Add main TR blocks            
    for n in range(num_prep, num_prep + tr_size):
        block = seq._seq.get_block(n + 1)
        result.main_tr.add_block(block)
        if result.first_rep_first_tr is not None:
            result.first_rep_first_tr.add_block(block)
        if result.last_rep_last_tr is not None:
            result.last_rep_last_tr.add_block(block)
    
    # Add last repetition footer blocks
    if result.last_rep_last_tr is not None:
        for n in range(len(seq.block_events) - num_cooldown, len(seq.block_events)):
            block = seq._seq.get_block(n + 1)
            result.last_rep_last_tr.add_block(block)
                 
    return result


def find_segments_in_tr(seq: PulserverSequence) -> tuple[list[pp.Sequence], SimpleNamespace]:
    """
    Find segment definitions within the TR structure of a sequence.
    
    Parameters
    ----------
    seq : PulserverSequence
        The sequence to analyze.
        
    Returns
    -----
    list[pp.Sequence]: 
        List of unique segments as pp.Sequence objects.

    SimpleNamespace:
        Structured result containing:
        - prep_segment_table: Maps prep section segments to unique segment IDs
        - main_segment_table: Maps main TR segments to unique segment IDs
        - cooldown_segment_table: Maps cooldown section segments to unique segment IDs
    """
    # First get unique blocks and TR info
    _, unique_table, block_durations_us, pure_delay_block, num_prep, num_cooldown = (
        _get_unique_blocks(seq._cseq)
    )
    
    # Find TR pattern
    tr_size, num_trs, degenerate_prep, degenerate_cooldown = _find_tr_in_sequence(
        unique_table, block_durations_us, pure_delay_block, num_prep, num_cooldown
    )
    
    # If no valid TR found, return empty result
    if tr_size == 0:
        return [], SimpleNamespace(
            prep_segment_table=[],
            main_segment_table=[],
            cooldown_segment_table=[],
        )
    
    # Get segments in TR (now returns a dict)
    raw_result = _find_segments_in_tr(
        seq._cseq,
        tr_size,
        num_trs,
        num_prep,
        num_cooldown,
        degenerate_prep,
        degenerate_cooldown,
        unique_table,
    )
    
    # Build SegmentInfo objects with pp.Sequence for each unique segment
    unique_segments = []
    for seg_dict in raw_result["unique_segments"]:
        start = seg_dict["start_block"]
        count = seg_dict["num_blocks"]
        
        # Build a pp.Sequence for this segment
        segment_seq = pp.Sequence(system=seq.system)
        for n in range(start, start + count):
            # pypulseq uses 1-based block indexing
            block = seq._seq.get_block(n + 1)
            segment_seq.add_block(block)
        unique_segments.append(segment_seq)
    
    return unique_segments, SimpleNamespace(
        prep_segment_table=raw_result["prep_segment_table"],
        main_segment_table=raw_result["main_segment_table"],
        cooldown_segment_table=raw_result["cooldown_segment_table"],
    )