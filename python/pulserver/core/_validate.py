"""Validation: compare pulserver waveforms against pypulseq reference."""

__all__ = ['validate']

from typing import Literal

import numpy as np
import pypulseq as pp

from ._sequence import SequenceCollection
from ._waveforms import get_tr_waveforms


def _interp_to_ref(t_ref, a_ref, t_test, a_test):
    """Interpolate *test* waveform onto *ref* time base, return aligned pair."""
    if len(t_ref) == 0 or len(t_test) == 0:
        return np.empty(0), np.empty(0)
    a_interp = np.interp(t_ref, t_test, a_test, left=0.0, right=0.0)
    return a_ref, a_interp


def _rms_error(ref, test):
    """Percentage RMS error relative to ref (0 if ref is silent)."""
    norm = np.sqrt(np.mean(ref ** 2))
    if norm < 1e-30:
        return 0.0
    return 100.0 * np.sqrt(np.mean((ref - test) ** 2)) / norm


def validate(
    seq: SequenceCollection,
    *,
    block_range: tuple[int, int] | None = None,
    grad_atol: float | None = None,
    rf_rms_percent: float = 10.0,
    amplitude_mode: Literal['max_pos', 'min_abs', 'actual'] = 'actual',
    tr_index: int = 0,
    plot: bool = False,
) -> dict:
    """Compare pulserver waveform extraction against pypulseq reference.

    For every gradient axis and the RF envelope, the waveforms produced
    by pulserver's C backend are compared against pypulseq's built-in
    ``seq.waveforms_and_times()`` output.  This gives confidence that
    the C library interprets the ``.seq`` file identically to the
    reference Python implementation.

    Parameters
    ----------
    seq : SequenceCollection
        Sequence to validate.
    block_range : (start, end) or None
        1-based block range to compare (None = full TR).
    grad_atol : float or None
        Absolute gradient error tolerance in mT/m.  If ``None``,
        defaults to ``3 * max_slew * grad_raster_time`` (one slew
        step) to account for raster-edge interpolation differences.
    rf_rms_percent : float
        RF magnitude percent-RMS error threshold.
    amplitude_mode : str
        Amplitude mode for pulserver extraction.
    tr_index : int
        TR instance for ``'actual'`` mode.
    plot : bool
        If ``True``, show a comparison plot (requires matplotlib).

    Returns
    -------
    dict
        Keys:

        - ``'ok'`` : bool — ``True`` if all channels pass.
        - ``'errors'`` : dict[str, float] — per-channel max error.
        - ``'messages'`` : list[str] — human-readable violation messages.
    """
    sys = seq.system

    # Default gradient tolerance: 3 slew steps (mT/m)
    if grad_atol is None:
        # max_slew in T/m/s, grad_raster in s → T/m per raster step → mT/m
        grad_atol = 3.0 * sys.max_slew * sys.grad_raster_time * 1e3

    # --- pypulseq reference waveforms ---
    if block_range is not None:
        ref_waves = seq._seq.waveforms_and_times(append_zero=True,
                                                  block_range=block_range)
    else:
        ref_waves = seq._seq.waveforms_and_times(append_zero=True)

    # ref_waves is (gx, gy, gz, rf, adc) where each is (2, N): row0=time(s), row1=value
    gamma = sys.gamma  # Hz/T

    ref_gx_t = ref_waves[0][0] * 1e6  # s → µs
    ref_gx_a = ref_waves[0][1] / (gamma * 1e-3)  # Hz/m → mT/m
    ref_gy_t = ref_waves[1][0] * 1e6
    ref_gy_a = ref_waves[1][1] / (gamma * 1e-3)
    ref_gz_t = ref_waves[2][0] * 1e6
    ref_gz_a = ref_waves[2][1] / (gamma * 1e-3)

    # RF magnitude
    ref_rf_t = ref_waves[3][0] * 1e6  # s → µs
    ref_rf_mag = np.abs(ref_waves[3][1]) / gamma * 1e6  # Hz → µT

    # --- pulserver waveforms ---
    wf = get_tr_waveforms(seq, amplitude_mode=amplitude_mode, tr_index=tr_index)

    # --- compare gradients ---
    errors = {}
    messages = []

    for axis, ref_t, ref_a, test_ch in [
        ('gx', ref_gx_t, ref_gx_a, wf.gx),
        ('gy', ref_gy_t, ref_gy_a, wf.gy),
        ('gz', ref_gz_t, ref_gz_a, wf.gz),
    ]:
        r, t = _interp_to_ref(ref_t, ref_a, test_ch.time_us, test_ch.amplitude)
        if len(r) > 0:
            err = float(np.max(np.abs(r - t)))
        else:
            err = 0.0
        errors[axis] = err
        if err > grad_atol:
            messages.append(
                f'{axis} waveform mismatch: max diff {err:.4f} mT/m '
                f'(tolerance {grad_atol:.4f} mT/m)')

    # --- compare RF ---
    r, t = _interp_to_ref(ref_rf_t, ref_rf_mag,
                           wf.rf_mag.time_us, wf.rf_mag.amplitude)
    rf_err = _rms_error(r, t)
    errors['rf_mag'] = rf_err
    if rf_err > rf_rms_percent:
        messages.append(
            f'RF magnitude mismatch: {rf_err:.1f}% RMS error '
            f'(tolerance {rf_rms_percent:.1f}%)')

    ok = len(messages) == 0

    # --- optional plot ---
    if plot:
        _plot_comparison(
            ref_gx_t, ref_gx_a,
            ref_gy_t, ref_gy_a,
            ref_gz_t, ref_gz_a,
            ref_rf_t, ref_rf_mag,
            wf, ok, messages,
        )

    return {'ok': ok, 'errors': errors, 'messages': messages}


def _plot_comparison(
    ref_gx_t, ref_gx_a,
    ref_gy_t, ref_gy_a,
    ref_gz_t, ref_gz_a,
    ref_rf_t, ref_rf_mag,
    wf, ok, messages,
):
    """Side-by-side comparison plot."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(14, 8), sharex=True)
    status = 'PASS' if ok else 'FAIL'

    axes[0].set_title(f'validate() — {status}', fontweight='bold')

    # RF magnitude
    axes[0].plot(ref_rf_t / 1e3, ref_rf_mag, 'k-', label='pypulseq', linewidth=0.8)
    axes[0].plot(wf.rf_mag.time_us / 1e3, wf.rf_mag.amplitude, 'r.', label='pulserver',
                 markersize=2)
    axes[0].set_ylabel('|RF| (µT)')
    axes[0].legend(loc='upper right', fontsize=7)

    for ax_idx, (label, ref_t, ref_a, ch) in enumerate([
        ('Gx', ref_gx_t, ref_gx_a, wf.gx),
        ('Gy', ref_gy_t, ref_gy_a, wf.gy),
        ('Gz', ref_gz_t, ref_gz_a, wf.gz),
    ], start=1):
        axes[ax_idx].plot(ref_t / 1e3, ref_a, 'k-', linewidth=0.8)
        axes[ax_idx].plot(ch.time_us / 1e3, ch.amplitude, 'r.', markersize=2)
        axes[ax_idx].set_ylabel(f'{label} (mT/m)')

    axes[-1].set_xlabel('Time (ms)')

    if messages:
        fig.text(0.5, 0.01, '\n'.join(messages), ha='center', fontsize=8,
                 color='red')

    plt.tight_layout(rect=[0, 0.05 if messages else 0, 1, 1])
    plt.show()
