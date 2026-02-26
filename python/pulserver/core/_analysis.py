"""TR and segment analysis for SequenceCollection."""

__all__ = [
    'find_tr',
    'find_segments_in_tr',
    'get_tr_gradient_waveforms',
]

from types import SimpleNamespace

import numpy as np
import pypulseq as pp

from ._extension._pulseqlib_wrapper import (
    _find_tr,
    _find_segments,
    _get_tr_gradient_waveforms,
)
from ._sequence import SequenceCollection


def find_tr(
    seq: SequenceCollection,
    num_reps: int = 1,
) -> SimpleNamespace:
    """
    Find TR structure in a sequence.

    Analyses the block table to identify a repeating TR pattern with
    optional preparation and cooldown blocks.

    Parameters
    ----------
    seq : SequenceCollection
        The sequence to analyze.
    num_reps : int
        Number of repetitions (for output structure).

    Returns
    -------
    SimpleNamespace
        Result containing:

        - ``main_tr`` : pp.Sequence — The main TR.
        - ``first_rep_first_tr`` : pp.Sequence | None — First TR with prep blocks
          (``None`` if prep is degenerate or absent).
        - ``last_rep_last_tr`` : pp.Sequence | None — Last TR with cooldown blocks
          (``None`` if cooldown is degenerate or absent).

    Raises
    ------
    RuntimeError
        If the C library fails to identify a TR structure.
    """
    tr_result = _find_tr(seq._cseq)

    tr_size = tr_result["tr_size"]
    num_trs = tr_result["num_trs"]
    num_prep = tr_result["num_prep_blocks"]
    num_cooldown = tr_result["num_cooldown_blocks"]
    degenerate_prep = tr_result["degenerate_prep"]
    degenerate_cooldown = tr_result["degenerate_cooldown"]

    result = SimpleNamespace()

    # Special case: only one TR
    if num_reps == 1 and num_trs == 1:
        result.main_tr = seq._seq
        result.first_rep_first_tr = None
        result.last_rep_last_tr = None
        return result

    # Main TR
    result.main_tr = pp.Sequence(system=seq.system)

    # First and last repetition header / footer
    if degenerate_prep or num_prep == 0:
        result.first_rep_first_tr = None
    else:
        result.first_rep_first_tr = pp.Sequence(system=seq.system)
    if degenerate_cooldown or num_cooldown == 0:
        result.last_rep_last_tr = None
    else:
        result.last_rep_last_tr = pp.Sequence(system=seq.system)

    # Prep blocks
    if result.first_rep_first_tr is not None:
        for n in range(num_prep):
            block = seq._seq.get_block(n + 1)
            result.first_rep_first_tr.add_block(block)

    # Main TR blocks
    for n in range(num_prep, num_prep + tr_size):
        block = seq._seq.get_block(n + 1)
        result.main_tr.add_block(block)
        if result.first_rep_first_tr is not None:
            result.first_rep_first_tr.add_block(block)
        if result.last_rep_last_tr is not None:
            result.last_rep_last_tr.add_block(block)

    # Cooldown blocks
    if result.last_rep_last_tr is not None:
        for n in range(len(seq.block_events) - num_cooldown, len(seq.block_events)):
            block = seq._seq.get_block(n + 1)
            result.last_rep_last_tr.add_block(block)

    return result


def find_segments_in_tr(
    seq: SequenceCollection,
) -> tuple[list[pp.Sequence], SimpleNamespace]:
    """
    Find segment definitions within the TR structure of a sequence.

    Parameters
    ----------
    seq : SequenceCollection
        The sequence to analyze.

    Returns
    -------
    list[pp.Sequence]
        List of :class:`pp.Sequence` representing the unique segments.
    SimpleNamespace
        Result containing:

        - ``prep_segment_table`` : list[int]
        - ``main_segment_table`` : list[int]
        - ``cooldown_segment_table`` : list[int]

    Raises
    ------
    RuntimeError
        If the C library fails to identify segments.
    """
    raw_result = _find_segments(seq._cseq)

    unique_segments = []
    for seg_dict in raw_result["unique_segments"]:
        start = seg_dict["start_block"]
        count = seg_dict["num_blocks"]

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


def get_tr_gradient_waveforms(seq: SequenceCollection) -> SimpleNamespace:
    """
    Extract concatenated gradient waveforms for a single TR.

    Parameters
    ----------
    seq : SequenceCollection
        The sequence to analyze.

    Returns
    -------
    SimpleNamespace
        Result containing:

        - ``time`` : np.ndarray of time points (microseconds)
        - ``waveform_gx`` : np.ndarray of Gx amplitude (Hz/m)
        - ``waveform_gy`` : np.ndarray of Gy amplitude (Hz/m)
        - ``waveform_gz`` : np.ndarray of Gz amplitude (Hz/m)

    Notes
    -----
    Timing conventions:

    - Trapezoids: corner points (0, rise, rise+flat, rise+flat+fall)
    - Extended trapezoids: samples at raster edges (from time shape)
    - Arbitrary gradients: samples at raster centres (0.5*raster, 1.5*raster, ...)

    The waveforms are concatenated across all blocks in the TR, with time
    offsets adjusted so each block's waveform starts where the previous ended.
    """
    result_dict = _get_tr_gradient_waveforms(seq._cseq)

    result = SimpleNamespace(
        time=np.asarray(result_dict["time_gx"], dtype=np.float32),
        waveform_gx=np.asarray(result_dict["waveform_gx"], dtype=np.float32),
        waveform_gy=np.asarray(result_dict["waveform_gy"], dtype=np.float32),
        waveform_gz=np.asarray(result_dict["waveform_gz"], dtype=np.float32),
    )

    return result
