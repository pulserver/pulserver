"""Per-block waveform access for SequenceCollection."""

__all__ = ['get_block']

from types import SimpleNamespace

import numpy as np

from ._extension._pulseqlib_wrapper import _get_block_info


def get_block(seq, segment_idx: int, block_idx: int) -> SimpleNamespace:
    """Return metadata for a single base block (normalized amplitudes).

    Convenience wrapper — delegates to ``seq.get_block()``.

    Parameters
    ----------
    seq : SequenceCollection
        The loaded sequence collection.
    segment_idx : int
        Segment index (global, 0-based).
    block_idx : int
        Block index within the segment (0-based).

    Returns
    -------
    SimpleNamespace
        See :meth:`SequenceCollection.get_block`.
    """
    return seq.get_block(segment_idx, block_idx)


def _get_block_impl(seq, segment_idx: int, block_idx: int) -> SimpleNamespace:
    """Internal implementation called by SequenceCollection.get_block()."""
    raw = _get_block_info(seq._cseq, segment_idx, block_idx)

    ns = SimpleNamespace()
    ns.duration_us   = raw["duration_us"]
    ns.start_time_us = raw["start_time_us"]

    for ax in ("x", "y", "z"):
        setattr(ns, f"has_g{ax}", raw[f"{ax}_has_grad"])
        setattr(ns, f"g{ax}_is_trapezoid", raw[f"{ax}_is_trapezoid"])
        setattr(ns, f"g{ax}_delay_us", raw[f"{ax}_grad_delay_us"])
        setattr(ns, f"g{ax}_num_samples", raw[f"{ax}_num_samples"])

    ns.has_rf         = raw["has_rf"]
    ns.rf_delay_us    = raw["rf_delay_us"]
    ns.rf_num_samples = raw["rf_num_samples"]
    ns.has_adc        = raw["has_adc"]
    ns.adc_delay_us   = raw["adc_delay_us"]

    return ns
