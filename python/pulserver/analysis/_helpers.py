"""Internal helper utilities for pulserver sequence analysis."""

__all__ = []

import numpy as np


def _add_echo_spacing_axis(ax, freq_min, freq_max):
    """Add secondary x-axis showing echo spacing (µs) dual to frequency (Hz).

    Creates a second x-axis at the top of the provided Axes object that
    displays echo spacing values (µs) corresponding to frequency ticks on
    the primary axis. Uses the relationship: echo_spacing = 1 / (2 * frequency).

    This is useful for spectral plots where both frequency and echo-spacing
    interpretations are relevant (e.g., gradient spectrum analysis).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to which the secondary axis is added.
    freq_min : float
        Minimum frequency (Hz) for the primary axis (currently unused,
        kept for potential future enhancements).
    freq_max : float
        Maximum frequency (Hz) for the primary axis (currently unused).

    Returns
    -------
    matplotlib.axes.Axes
        The secondary (top) x-axis object, ready for further customization.

    Notes
    -----
    - The secondary axis shares the same x-limits as the primary axis.
    - Frequency values ≤ 0 are displayed as ∞ (infinity symbol).
    - Echo spacing is computed as 1 / (2 * frequency_Hz) in microseconds.
    - The secondary axis y-label is set to ``'Echo Spacing (µs)'``.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> ax.set_xlabel('Frequency (Hz)')
    >>> _add_echo_spacing_axis(ax, freq_min=0, freq_max=3000)
    >>> # Now the top x-axis shows echo spacing values
    """
    ax_top = ax.twiny()

    ax_top.set_xlim((freq_min, freq_max))

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
    ax_top.set_xlim((freq_min, freq_max))
    ax_top.tick_params(axis='x', pad=6)
    ax_top.set_xlabel('Echo Spacing (\u00b5s)', fontsize=12)

    return ax_top
