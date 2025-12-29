"""Simultaneous Multislice pulse design."""

import numpy as np

from ._slr import dzrf


def dz_pins(
    sl_sep,
    sl_thick,
    g_max,
    g_slew,
    dt,
    b1_max=0.18,
    gambar=4258,
):
    r"""
    PINS multiband pulse design.

    Args:
        tb (float): time-bandwidth product.
        sl_sep (float): slice separation in cm.
        sl_thick (float): slice thickness in cm.
        g_max (float): max gradient amplitude in gauss/cm
        g_slew (float): max gradient sliew in gauss/cm/s
        dt (float): RF + gradient dwell time in s.
        b1_max (float): Maximum RF amplitude
        ptype (string): pulse type, 'st' (small-tip excitation), 'ex' (pi/2
            excitation pulse), 'se' (spin-echo pulse), 'inv' (inversion), or
            'sat' (pi/2 saturation pulse).
        ftype (string): type of filter to use in pulse design
        d1 (float): passband ripple level in :math:'M_0^{-1}'.
        d2 (float): stopband ripple level in :math:'M_0^{-1}'.
        gambar (float): Appropriate gyromagnetic ratio in Hz/gauss.

    Returns:
        2-element tuple containing:

        - **rf** (*array*): RF Pulse in Gauss
        - **g** (*array*): Gradient waveform in Gauss/cm

    References:
        Norris, D.G. and Koopmans, P.J. and Boyacioglu, R and Barth, M (2011).
        'Power independent of number of slices (PINS) radiofrequency Pulses
        for low-power simultaneous multislice excitation'.
        Magn. Reson. Med., 66(5):1234-1240.
    """

    kz_width = tb / sl_thick  # 1/cm, width in k-space we must go

    # calculate number of subpulses (odd)
    n_pulses = int(2 * np.floor(np.ceil(kz_width / (1 / sl_sep)) / 2))

    # call SLR to get envelope
    rf_soft = dzrf(n_pulses, tb, ptype, ftype, d1, d2)

    # design the blip trapezoid
    g_area = 1 / sl_sep / gambar
    gz_blip, _ = trap_grad(g_area, g_max, g_slew, dt)

    # Calculate the block/hard RF pulse width based on
    b1_scaled = 2 * np.pi * gambar * b1_max * dt
    hpw = int(np.ceil(np.max(np.abs(rf_soft)) / b1_scaled))

    # interleave RF subpusles with gradient subpulses to form full pulses
    rf = np.kron(
        rf_soft[:-1],
        np.concatenate((np.ones(hpw), np.zeros((np.size(gz_blip))))),
    )
    rf = np.concatenate((rf, rf_soft[-1] * np.ones(hpw)))
    rf = rf / (np.sum(rf) * 2 * np.pi * gambar * dt) * np.sum(rf_soft)

    g = np.concatenate([np.zeros(hpw), np.squeeze(gz_blip)])
    g = np.tile(g, n_pulses - 1)
    g = np.concatenate((g, np.zeros(hpw)))

    return rf, g
