"""Internal helper utilities for pulserver sequence analysis."""

__all__ = []

import numpy as np


def _circular_pad(waveform: np.ndarray, target_length: int) -> np.ndarray:
    """
    Circularly pad a waveform to a target length.

    Uses circular (modulo) indexing so that when extended, the waveform
    wraps around to the beginning (true circular padding).

    If target_length == len(waveform), returns a copy.
    If target_length > len(waveform), uses modulo indexing to repeat cyclically.
    If target_length < len(waveform), truncates the waveform.

    Parameters
    ----------
    waveform : np.ndarray
        Input waveform.
    target_length : int
        Desired output length.

    Returns
    -------
    np.ndarray
        Circularly padded waveform of length target_length.
    """
    current_length = len(waveform)

    if current_length == target_length:
        return waveform.copy()
    elif current_length > target_length:
        return waveform[:target_length].copy()
    else:
        indices = np.arange(target_length) % current_length
        return waveform[indices].copy()


def _add_echo_spacing_axis(ax, freq_min, freq_max):
    """
    Add a secondary x-axis showing echo spacing (us) = 1/(2*frequency).
    Uses proper inverse transformation so tick values are correct.
    """
    ax_top = ax.twiny()

    ax_top.set_xlim(ax.get_xlim())

    def freq_to_es(freq_hz):
        if freq_hz <= 0:
            return np.inf
        return 1e6 / (2 * freq_hz)

    primary_ticks = ax.get_xticks()
    primary_ticks = primary_ticks[primary_ticks >= 0]

    ax_top.set_xticks(primary_ticks)
    ax_top.set_xticklabels(
        [f'{freq_to_es(f):.1f}' if f > 0 else '\u221e' for f in primary_ticks]
    )
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xlabel('Echo Spacing (\u00b5s)', fontsize=12)

    return ax_top
