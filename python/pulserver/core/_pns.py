"""Peripheral Nerve Stimulation (PNS) analysis for SequenceCollection."""

__all__ = ['pns']

import warnings
from types import SimpleNamespace

import numpy as np

import matplotlib.pyplot as plt

from ._extension._pulseqlib_wrapper import _calc_pns
from ._helpers import _circular_pad
from ._sequence import SequenceCollection
from ._analysis import get_tr_gradient_waveforms


def pns(
    seq: SequenceCollection,
    chronaxie_us: float | None = None,
    rheobase: float | None = None,
    alpha: float | None = None,
    do_plot: bool = False,
) -> SimpleNamespace:
    """
    Compute Peripheral Nerve Stimulation (PNS) levels for gradient waveforms.

    Uses convolution with a vendor-specific nerve response kernel to estimate
    PNS as a percentage of the stimulation threshold.  Circular padding is
    automatically applied based on the kernel length.

    Parameters
    ----------
    seq : SequenceCollection
        The sequence to analyze.
    chronaxie_us : float | None
        Chronaxie time constant in microseconds.
        Typical value: ~360 us (IEC 60601-2-33:2022).
    rheobase : float | None
        Rheobase — minimum slew rate for stimulation in T/m/s.
        Typical value: ~20 T/m/s.
    alpha : float | None
        Effective coil length in metres.
        The stimulation threshold Smin = rheobase / alpha.
        Typical value: ~0.333 m (IEC 60601-2-33:2022).
    do_plot : bool
        If ``True``, plot the PNS waveforms.  Default is ``False``.

    Returns
    -------
    SimpleNamespace
        Result containing:

        - ``max_pns`` : float — Maximum PNS value (%).  Values > 100 %
          indicate stimulation.
        - ``max_pns_index`` : int — Sample index of maximum.
        - ``max_pns_time_us`` : float — Time of maximum in microseconds.
        - ``num_samples`` : int — Number of output samples.
        - ``pns_total`` : np.ndarray — Combined PNS waveform
          sqrt(X^2 + Y^2 + Z^2).
        - ``pns_x``, ``pns_y``, ``pns_z`` : np.ndarray — Per-axis PNS
          waveforms (%).

    Raises
    ------
    ValueError
        If required GE parameters are not provided.
    RuntimeError
        If PNS computation fails.

    Notes
    -----
    **Currently Supported: GE/GEHC Model** (IEC 60601-2-33:2022 Eq. AA.21)

    The nerve response kernel is::

        h(tau) = (dt / Smin) * c / (c + tau)^2

    where *c* = chronaxie (us) and *Smin* = rheobase / alpha (T/m/s).

    PNS is computed as the convolution of the gradient slew rate with
    this kernel, with circular padding automatically applied.

    Examples
    --------
    >>> result = pns(seq, chronaxie_us=360.0, rheobase=20.0, alpha=0.333)
    >>> print(f"Max PNS: {result.max_pns:.1f}%")
    """
    if chronaxie_us is None or rheobase is None or alpha is None:
        raise ValueError(
            "GE PNS model requires 'chronaxie_us', 'rheobase', and 'alpha' "
            "parameters. Typical values: chronaxie_us=360.0, rheobase=20.0, "
            "alpha=0.333"
        )

    # Call C++ extension — returns per-axis PNS percentages
    result_dict = _calc_pns(seq._cseq, chronaxie_us, rheobase, alpha)

    num_samples = result_dict["num_samples"]
    pns_x = np.asarray(result_dict["slew_x"], dtype=np.float32)
    pns_y = np.asarray(result_dict["slew_y"], dtype=np.float32)
    pns_z = np.asarray(result_dict["slew_z"], dtype=np.float32)

    # Combined PNS = sqrt(x² + y² + z²)
    pns_total = np.sqrt(pns_x ** 2 + pns_y ** 2 + pns_z ** 2)

    max_pns = float(np.max(pns_total))
    max_pns_index = int(np.argmax(pns_total))
    grad_raster_time = seq.system.grad_raster_time  # seconds
    max_pns_time_us = max_pns_index * 0.5 * grad_raster_time * 1e6

    result = SimpleNamespace(
        max_pns=max_pns,
        max_pns_index=max_pns_index,
        max_pns_time_us=max_pns_time_us,
        num_samples=num_samples,
        chronaxie_us=chronaxie_us,
        pns_total=pns_total,
        pns_x=pns_x,
        pns_y=pns_y,
        pns_z=pns_z,
    )

    if do_plot:
        _plot_pns(result, seq)

    # Threshold check
    _check_pns_thresholds(result)

    return result


# ── plotting ─────────────────────────────────────────────────────────

def _plot_pns(
    pns: SimpleNamespace,
    seq: SequenceCollection | None = None,
) -> tuple:
    """
    Plot PNS waveforms with TR gradient waveforms circularly padded to match.

    Parameters
    ----------
    pns : SimpleNamespace
        Output from :func:`pns`.
    seq : SequenceCollection | None
        The sequence object (needed to get gradient waveforms).

    Returns
    -------
    tuple
        ``(fig, axes)`` for further customisation.
    """
    if not hasattr(pns, 'pns_x') or not hasattr(pns, 'pns_y') or not hasattr(pns, 'pns_z'):
        raise ValueError("pns must have pns_x, pns_y, pns_z arrays")

    num_pns_samples = pns.num_samples
    grad_raster_time = seq.system.grad_raster_time
    chronaxie_us = pns.chronaxie_us

    # Kernel length matching C library: PNS_KERNEL_DURATION_FACTOR = 20.0
    kernel_len = int(20.0 * chronaxie_us / (grad_raster_time * 1e6)) + 1
    padded_len = num_pns_samples + kernel_len

    fig = plt.figure()

    # ── Panel 1: PNS Waveforms ──
    ax_pns = plt.subplot(2, 1, 1)
    time_pns_ms = np.arange(num_pns_samples) * 0.5 * grad_raster_time * 1000.0

    colors = {'x': 'C0', 'y': 'C1', 'z': 'C2', 'total': 'C3'}
    labels = {'x': 'PNS_X', 'y': 'PNS_Y', 'z': 'PNS_Z', 'total': 'PNS_Total'}

    ax_pns.plot(time_pns_ms, pns.pns_total, color=colors['total'],
                linewidth=2, label=labels['total'])
    ax_pns.plot(time_pns_ms, pns.pns_x, color=colors['x'],
                linewidth=2, label=labels['x'])
    ax_pns.plot(time_pns_ms, pns.pns_y, color=colors['y'],
                linewidth=2, label=labels['y'])
    ax_pns.plot(time_pns_ms, pns.pns_z, color=colors['z'],
                linewidth=2, label=labels['z'])

    ax_pns.axhline(80.0, color='gray', linestyle=':', linewidth=2,
                   label='80% limit')
    ax_pns.axhline(100.0, color='red', linestyle='--', linewidth=2,
                   label='100% threshold')

    ax_pns.set_xlabel('Time (ms)', fontsize=12)
    ax_pns.set_ylabel('PNS (%)', fontsize=12)
    ax_pns.set_title('Peripheral Nerve Stimulation (PNS)', fontsize=14,
                     fontweight='bold')
    ax_pns.legend(loc='upper right', fontsize=11)
    ax_pns.grid(True, alpha=0.3)
    ax_pns.set_ylim(bottom=0)

    # ── Panel 2: Gradient Waveforms (Circularly Padded) ──
    if seq is not None:
        ax_wf = plt.subplot(2, 1, 2)
        gamma = seq.system.gamma

        waveforms = get_tr_gradient_waveforms(seq)

        wf_gx_mtpm = waveforms.waveform_gx / gamma * 1000.0
        wf_gy_mtpm = waveforms.waveform_gy / gamma * 1000.0
        wf_gz_mtpm = waveforms.waveform_gz / gamma * 1000.0

        wf_gx_padded = _circular_pad(wf_gx_mtpm, padded_len)
        wf_gy_padded = _circular_pad(wf_gy_mtpm, padded_len)
        wf_gz_padded = _circular_pad(wf_gz_mtpm, padded_len)

        time_wf_padded_ms = np.arange(padded_len) * 0.5 * grad_raster_time * 1000.0

        ax_wf.plot(time_wf_padded_ms, wf_gx_padded, color=colors['x'],
                   linewidth=2, label='Gx')
        ax_wf.plot(time_wf_padded_ms, wf_gy_padded, color=colors['y'],
                   linewidth=2, label='Gy')
        ax_wf.plot(time_wf_padded_ms, wf_gz_padded, color=colors['z'],
                   linewidth=2, label='Gz')

        ax_wf.axvline(waveforms.time[-1] / 1000.0, color='gray',
                      linestyle=':', linewidth=1.5, alpha=0.7,
                      label='Padding start')

        ax_wf.set_xlabel('Time (ms)', fontsize=12)
        ax_wf.set_ylabel('Gradient Amplitude (mT/m)', fontsize=12)
        ax_wf.set_title(
            f'TR Gradient Waveforms (Circularly Padded by {kernel_len} samples)',
            fontsize=14, fontweight='bold',
        )
        ax_wf.legend(loc='upper right', fontsize=11)
        ax_wf.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, fig.axes


# ── threshold check ──────────────────────────────────────────────────

def _check_pns_thresholds(pns: SimpleNamespace) -> None:
    """Check PNS waveforms for threshold violations and emit warnings."""
    if not hasattr(pns, 'pns_total'):
        return

    max_val = np.max(pns.pns_total)

    for threshold in (80.0, 100.0):
        if max_val > threshold:
            warnings.warn(
                f"PNS Total exceeds {threshold}% threshold: "
                f"max = {max_val:.1f}%",
                UserWarning,
            )
