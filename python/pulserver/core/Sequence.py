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


def _make_diagnostic(diag_dict: dict) -> SimpleNamespace:
    """Create a diagnostic SimpleNamespace from a dict."""
    diag = SimpleNamespace(
        code=diag_dict["code"],
        message=diag_dict["message"],
        hint=diag_dict["hint"],
        block_index=diag_dict["block_index"],
        channel=diag_dict["channel"],
        num_unique_blocks=diag_dict["num_unique_blocks"],
        imaging_region_length=diag_dict["imaging_region_length"],
        candidate_pattern_length=diag_dict["candidate_pattern_length"],
        mismatch_position=diag_dict["mismatch_position"],
    )
    # Add convenience property
    diag.success = diag.code > 0
    return diag


def _format_diagnostic(diag: SimpleNamespace) -> str:
    """Format diagnostic info as a human-readable string."""
    if diag.success:
        return "Success"
    
    lines = [f"Error: {diag.message}"]
    
    if diag.block_index >= 0:
        lines.append(f"  Block index: {diag.block_index}")
    if diag.mismatch_position >= 0:
        lines.append(f"  Mismatch at position: {diag.mismatch_position}")
    if diag.num_unique_blocks > 0:
        lines.append(f"  Unique blocks found: {diag.num_unique_blocks}")
    if diag.imaging_region_length > 0:
        lines.append(f"  Imaging region length: {diag.imaging_region_length}")
    if diag.candidate_pattern_length > 0:
        lines.append(f"  Best candidate pattern length: {diag.candidate_pattern_length}")
    
    if diag.hint:
        lines.append(f"\nHint: {diag.hint}")
    
    return "\n".join(lines)


class SequenceAnalysisError(Exception):
    """Exception raised when sequence analysis fails."""
    
    def __init__(self, diagnostic: SimpleNamespace):
        self.diagnostic = diagnostic
        super().__init__(_format_diagnostic(diagnostic))


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


def get_unique_blocks(seq: PulserverSequence) -> tuple[list[int], list[int]]:
    """
    Get unique blocks in sequence.

    Parameters
    ----------
    seq : PulserverSequence
        Input sequence.

    Returns
    -------
    unique_blocks : list[int]
        Set of block ID containing the first instance of each unique block.
    unique_table : list[int]
        Table containing the unique block index corresponding to each block in sequence.
        
    Notes
    -----
    A unique block is determined by the tuple of unique (rf, gx, gy, gz),
    i.e., the tuple of blocks whose rf and gradient events has the same normalized
    waveforms on all channels. For gradients, waveforms are uniquely determined by
    class (trapezoids vs arbitrary grads) and timing.

    """
    unique_blocks, unique_table, _, _, _, _ = _get_unique_blocks(seq._cseq)
    return unique_blocks, unique_table


def find_tr(seq: PulserverSequence, num_reps: int = 1, raise_on_error: bool = True) -> SimpleNamespace:
    """
    Find TR structure in a sequence.
    
    Parameters
    ----------
    seq : PulserverSequence
        The sequence to analyze.
    num_reps : int
        Number of repetitions (for output structure).
    raise_on_error : bool
        If True, raise SequenceAnalysisError on failure.
        If False, return result with diagnostic info.
    
    Returns
    -------
    SimpleNamespace
        Result containing:
        - main_tr: The main TR as a pp.Sequence
        - first_rep_first_tr: First TR with prep blocks (or None)
        - last_rep_last_tr: Last TR with cooldown blocks (or None)
    
    Raises
    ------
    SequenceAnalysisError
        If raise_on_error=True and TR detection fails.
    """
    _, unique_table, block_durations_us, pure_delay_block, num_prep, num_cooldown = (
        _get_unique_blocks(seq._cseq)
    )
    
    tr_result = _find_tr_in_sequence(
        unique_table, block_durations_us, pure_delay_block, num_prep, num_cooldown
    )
    
    diagnostic = _make_diagnostic(tr_result["diagnostic"])
    
    # Check for failure
    if not tr_result["success"]:
        if raise_on_error:
            raise SequenceAnalysisError(diagnostic)
        else:
            result = SimpleNamespace()
            result.main_tr = None
            result.first_rep_first_tr = None
            result.last_rep_last_tr = None
            result.diagnostic = diagnostic
            return result
    
    tr_size = tr_result["tr_size"]
    num_trs = tr_result["num_trs"]
    degenerate_prep = tr_result["degenerate_prep"]
    degenerate_cooldown = tr_result["degenerate_cooldown"]
    
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


def find_segments_in_tr(seq: PulserverSequence, raise_on_error: bool = True) -> tuple[list[pp.Sequence], SimpleNamespace]:
    """
    Find segment definitions within the TR structure of a sequence.
    
    Parameters
    ----------
    seq : PulserverSequence
        The sequence to analyze.
    raise_on_error : bool
        If True, raise SequenceAnalysisError on failure.
        If False, return result with diagnostic info.
        
    Returns
    -------
    list[pp.Sequence]: 
        List of pp.Sequences representing the unique sequence Segments.
    SimpleNamespace
        Result containing:
          unique_block_indices, and sequence (pp.Sequence)
        - prep_segment_table: Maps prep section segments to unique segment IDs
        - main_segment_table: Maps main TR segments to unique segment IDs
        - cooldown_segment_table: Maps cooldown section segments to unique segment IDs
    """
    # First get unique blocks and TR info
    _, unique_table, block_durations_us, pure_delay_block, num_prep, num_cooldown = (
        _get_unique_blocks(seq._cseq)
    )
    
    # Find TR pattern
    tr_result = _find_tr_in_sequence(
        unique_table, block_durations_us, pure_delay_block, num_prep, num_cooldown
    )
    
    diagnostic = _make_diagnostic(tr_result["diagnostic"])
    
    # If no valid TR found, return empty result or raise
    if not tr_result["success"]:
        if raise_on_error:
            raise SequenceAnalysisError(diagnostic)
        result = SimpleNamespace()
        result.unique_segments = []
        result.prep_segment_table = []
        result.main_segment_table = []
        result.cooldown_segment_table = []
        result.diagnostic = diagnostic
        return result
    
    tr_size = tr_result["tr_size"]
    num_trs = tr_result["num_trs"]
    degenerate_prep = tr_result["degenerate_prep"]
    degenerate_cooldown = tr_result["degenerate_cooldown"]
    
    # Get segments in TR
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
    
    # Build SimpleNamespace for each unique segment with pp.Sequence
    unique_segments = []
    for seg_dict in raw_result["unique_segments"]:
        start = seg_dict["start_block"]
        count = seg_dict["num_blocks"]
        
        # Build a pp.Sequence for this segment
        segment_seq = pp.Sequence(system=seq.system)
        for n in range(start, start + count):
            block = seq._seq.get_block(n + 1)
            segment_seq.add_block(block)
        unique_segments.append(segment_seq)
    
    result = SimpleNamespace()
    result.prep_segment_table = raw_result["prep_segment_table"]
    result.main_segment_table = raw_result["main_segment_table"]
    result.cooldown_segment_table = raw_result["cooldown_segment_table"]
    
    return unique_segments, result