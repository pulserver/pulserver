"""PNS visualisation for SequenceCollection (plotting only, no pass/fail)."""

__all__ = ['pns']

import numpy as np

from ._extension._pulseqlib_wrapper import _calc_pns
from ._sequence import SequenceCollection

# Matplotlib imported lazily to avoid hard dependency at import time.


def pns(
    seq: SequenceCollection,
    *,
    sequence_idx: int = 0,
    stim_threshold: float,
    decay_constant_us: float,
    threshold_percent: float = 80.0,
) -> None:
    """Plot convolved PNS waveforms for a representative TR.

    Displays per-axis (X, Y, Z) and combined PNS percentage waveforms
    together with a horizontal threshold line.  No pass/fail check is
    performed — use :meth:`SequenceCollection.check` for that.

    Parameters
    ----------
    seq : SequenceCollection
        The sequence to analyse.
    sequence_idx : int
        Subsequence index (0-based, default 0).
    stim_threshold : float
        PNS stimulation threshold in Hz/m/s.  Equals
        ``rheobase / alpha`` in the SAFE nerve model.
    decay_constant_us : float
        PNS decay constant (chronaxie) in microseconds.
    threshold_percent : float
        Threshold line to draw on the plot (default 80 %).
    """
    import matplotlib.pyplot as plt

    # Internally map to C-level parameters: alpha = 1 → rheobase = stim_threshold
    result_dict = _calc_pns(
        seq._cseq,
        subsequence_idx=sequence_idx,
        chronaxie_us=decay_constant_us,
        rheobase=stim_threshold,
        alpha=1.0,
    )

    num_samples = result_dict["num_samples"]
    pns_x = np.asarray(result_dict["slew_x"], dtype=np.float32)
    pns_y = np.asarray(result_dict["slew_y"], dtype=np.float32)
    pns_z = np.asarray(result_dict["slew_z"], dtype=np.float32)
    pns_total = np.sqrt(pns_x ** 2 + pns_y ** 2 + pns_z ** 2)

    grad_raster_time = seq.system.grad_raster_time  # seconds
    time_ms = np.arange(num_samples) * 0.5 * grad_raster_time * 1e3

    # ── Single-panel figure ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(time_ms, pns_total, color='C3', linewidth=2, label='PNS Total')
    ax.plot(time_ms, pns_x, color='C0', linewidth=1.2, label='PNS X')
    ax.plot(time_ms, pns_y, color='C1', linewidth=1.2, label='PNS Y')
    ax.plot(time_ms, pns_z, color='C2', linewidth=1.2, label='PNS Z')

    ax.axhline(
        threshold_percent, color='red', linestyle='--', linewidth=2,
        label=f'{threshold_percent:.0f}% threshold',
    )

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('PNS (%)')
    ax.set_title('Peripheral Nerve Stimulation — Convolved Slew Rate')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
