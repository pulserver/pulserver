"""PNS visualisation for SequenceCollection (plotting only, no pass/fail)."""

__all__ = ['pns']

import numpy as np

from ._extension._pulseqlib_wrapper import _calc_pns, _find_tr
from ._sequence import SequenceCollection

# Matplotlib imported lazily to avoid hard dependency at import time.


def pns(
    seq: SequenceCollection,
    *,
    sequence_idx: int | None = None,
    stim_threshold: float,
    decay_constant_us: float,
    threshold_percent: float | list[float] | tuple[float, ...] | None = None,
) -> None:
    """Plot convolved PNS waveforms for canonical TRs.

    Displays per-axis (X, Y, Z) and combined PNS percentage waveforms
    together with a horizontal threshold line.  No pass/fail check is
    performed — use :meth:`SequenceCollection.check` for that.

    Canonical-TR selection follows the C safety backend per shot-ID
    combination, using shot-filtered ``AMP_MAX_POS`` for each group.

    Parameters
    ----------
    seq : SequenceCollection
        The sequence to analyse.
    sequence_idx : int, optional
        Subsequence index (0-based). If ``None`` (default), all subsequences
        are plotted.
    stim_threshold : float
        PNS stimulation threshold in Hz/m/s.  Equals
        ``rheobase / alpha`` in the SAFE nerve model.
    decay_constant_us : float
        PNS decay constant (chronaxie) in microseconds.
    threshold_percent : float or sequence of float
        Threshold line(s) to draw on the plot (default 80 % and 100%).
    """
    import matplotlib.pyplot as plt

    # Internally map to C-level parameters: alpha = 1 → rheobase = stim_threshold
    if threshold_percent is None:
        threshold_percent = [80.0, 100.0]
    if isinstance(threshold_percent, (list, tuple)):
        thresholds = [float(v) for v in threshold_percent]
    else:
        thresholds = [float(threshold_percent)]

    # Color thresholds from left-to-right: lighter gray to black.
    # The rightmost (largest) threshold is always black.
    sorted_thresholds = sorted(thresholds)
    n_thr = len(sorted_thresholds)
    if n_thr == 0:
        threshold_styles = []
    elif n_thr == 1:
        threshold_styles = [(sorted_thresholds[0], (0.0, 0.0, 0.0))]
    else:
        threshold_styles = []
        for i, thr in enumerate(sorted_thresholds):
            gray = 0.75 * float(n_thr - 1 - i) / float(n_thr - 1)
            threshold_styles.append((thr, (gray, gray, gray)))

    if sequence_idx is None:
        sequence_indices = list(range(seq.num_sequences))
    else:
        if sequence_idx < 0 or sequence_idx >= seq.num_sequences:
            raise ValueError(
                f'sequence_idx={sequence_idx} out of range for {seq.num_sequences} subsequences'
            )
        sequence_indices = [int(sequence_idx)]

    grad_raster_time = seq.system.grad_raster_time  # seconds

    # Base colors (opaque) for the four PNS traces.
    _BASE_COLORS = ['C3', 'C0', 'C1', 'C2']

    for ss_idx in sequence_indices:
        tr_info = _find_tr(seq._cseq, subsequence_idx=ss_idx)
        num_canonical = int(tr_info.get('num_canonical_trs', 1))

        fig, ax = plt.subplots(figsize=(12, 5))

        for canonical_tr_idx in range(num_canonical):
            result_dict = _calc_pns(
                seq._cseq,
                subsequence_idx=ss_idx,
                canonical_tr_idx=canonical_tr_idx,
                chronaxie_us=decay_constant_us,
                rheobase=stim_threshold,
                alpha=1.0,
            )

            num_samples = result_dict['num_samples']
            pns_x = np.asarray(result_dict['slew_x'], dtype=np.float32)
            pns_y = np.asarray(result_dict['slew_y'], dtype=np.float32)
            pns_z = np.asarray(result_dict['slew_z'], dtype=np.float32)
            pns_total = np.sqrt(pns_x**2 + pns_y**2 + pns_z**2)
            time_ms = np.arange(num_samples, dtype=np.float32) * grad_raster_time * 1e3

            # First canonical TR is fully opaque; subsequent ones are shaded.
            alpha = 1.0 if canonical_tr_idx == 0 else 0.35
            ctr_label = f' (CTR{canonical_tr_idx})' if num_canonical > 1 else ''

            ax.plot(
                time_ms,
                pns_total,
                color=_BASE_COLORS[0],
                linewidth=2,
                alpha=alpha,
                label=f'PNS Total{ctr_label}',
            )
            ax.plot(
                time_ms,
                pns_x,
                color=_BASE_COLORS[1],
                linewidth=1.2,
                alpha=alpha,
                label=f'PNS X{ctr_label}',
            )
            ax.plot(
                time_ms,
                pns_y,
                color=_BASE_COLORS[2],
                linewidth=1.2,
                alpha=alpha,
                label=f'PNS Y{ctr_label}',
            )
            ax.plot(
                time_ms,
                pns_z,
                color=_BASE_COLORS[3],
                linewidth=1.2,
                alpha=alpha,
                label=f'PNS Z{ctr_label}',
            )

        for thr, color in threshold_styles:
            ax.axhline(
                thr,
                color=color,
                linestyle=':',
                linewidth=2,
                label=f'{thr:.0f}% threshold',
            )

        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('PNS (%)')
        ax.set_title(f'Peripheral Nerve Stimulation - Convolved Slew Rate [SS{ss_idx}]')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        fig.tight_layout()
