"""Sequence collection report for SequenceCollection."""

__all__ = ['report']

from types import SimpleNamespace

from ._extension._pulseqlib_wrapper import _get_report
from ._sequence import SequenceCollection


def report(
    seq: SequenceCollection,
    *,
    do_print: bool = False,
) -> list[SimpleNamespace]:
    """Generate a structured report of the sequence collection.

    Returns one :class:`~types.SimpleNamespace` per subsequence with:

    - ``unique_block_ids`` — list of (segment_idx, block_idx) for the
      first instance of each unique block in the sequence.
    - ``segments`` — list of ``(start_block, num_blocks)`` tuples, where
      ``start_block`` is the 0-based index in the original Sequence of
      the segment instance with maximum energy.
    - ``num_prep_blocks`` — preparation blocks before first TR.
    - ``num_cooldown_blocks`` — cooldown blocks after last TR.
    - ``tr_size`` — number of blocks per TR.
    - ``tr_duration_s`` — TR duration in seconds.
    - ``segment_order`` — ordered list of segment IDs composing one TR.

    Parameters
    ----------
    seq : SequenceCollection
        The sequence to report on.
    do_print : bool
        If ``True``, return a formatted string instead of a list of
        SimpleNamespaces.

    Returns
    -------
    list[SimpleNamespace] or str
        One entry per subsequence (or a formatted string if *do_print*).
    """
    raw = _get_report(seq._cseq)

    num_ss = raw["num_subsequences"]
    results = []

    for ss_dict in raw["subsequences"]:
        ns = SimpleNamespace()

        # -- unique block IDs (segment_idx, block_idx) --
        block_ids = []
        seg_offset = ss_dict["segment_offset"]
        for i, seg in enumerate(ss_dict["segments"]):
            for blk in range(seg["num_blocks"]):
                block_ids.append((seg_offset + i, blk))
        ns.unique_block_ids = block_ids

        # -- segments: (start_block, num_blocks) --
        ns.segments = [
            (seg["start_block"], seg["num_blocks"])
            for seg in ss_dict["segments"]
        ]

        ns.num_prep_blocks = ss_dict["num_prep_blocks"]
        ns.num_cooldown_blocks = ss_dict["num_cooldown_blocks"]
        ns.tr_size = ss_dict["tr_size"]
        ns.tr_duration_s = ss_dict["tr_duration_us"] * 1e-6

        # -- ordered segment IDs composing one TR --
        ns.segment_order = list(ss_dict["main_segment_table"])

        # -- extra tables --
        ns.prep_segment_table = list(ss_dict["prep_segment_table"])
        ns.cooldown_segment_table = list(ss_dict["cooldown_segment_table"])

        results.append(ns)

    if not do_print:
        return results

    # ── formatted output ──────────────────────────────────────────
    lines = []
    total_dur_s = raw["total_duration_us"] * 1e-6
    lines.append(f"Sequence length: {total_dur_s:.6f} s  |  "
                 f"Subsequences: {num_ss}  |  "
                 f"Total unique segments: {raw['num_segments']}")
    lines.append("")

    for idx, ns in enumerate(results):
        lines.append(f"--- Subsequence {idx} ---")
        lines.append(f"  TR size:            {ns.tr_size} blocks")
        lines.append(f"  TR duration:        {ns.tr_duration_s * 1e3:.3f} ms")
        lines.append(f"  Prep blocks:        {ns.num_prep_blocks}")
        lines.append(f"  Cooldown blocks:    {ns.num_cooldown_blocks}")
        lines.append(f"  Unique blocks:      {len(ns.unique_block_ids)}")
        lines.append(f"  Unique segments:    {len(ns.segments)}")
        lines.append(f"  Segment order (TR): {ns.segment_order}")
        for si, (start, nblk) in enumerate(ns.segments):
            lines.append(f"    seg {si}: start_block={start}, "
                         f"num_blocks={nblk}")
        lines.append("")

    return "\n".join(lines)
