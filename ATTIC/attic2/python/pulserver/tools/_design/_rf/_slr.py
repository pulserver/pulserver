"""Base Shinnar-LeRoux pulse design."""

__all__ = ['design_slr_pulse']

from types import SimpleNamespace
from typing import Union

import numpy as np
import scipy.signal
from pypulseq import make_arbitrary_rf
from pypulseq.opts import Opts
from pypulseq.supported_labels_rf_use import get_supported_rf_uses


def design_slr_pulse(
    flip_angle: float,
    apodization: float = 0,
    bandwidth: float = 0,
    delay: float = 0,
    dwell: float = 0,
    duration: float = 4e-3,
    freq_offset: float = 0,
    phase_offset: float = 0,
    system: Union[Opts, None] = None,
    time_bw_product: float = 4,
    use: str = 'undefined',
    freq_ppm: float = 0,
    phase_ppm: float = 0,
    filter_type: str = 'ls',
    passband_ripple_lvl: float = 0.01,
    stopband_ripple_lvl: float = 0.01,
    cancel_alpha_phs: bool = False,
) -> SimpleNamespace:
    r"""
    Basic Shinnar-LeRoux RF pulse design interface.

    Parameters
    ----------
    flip_angle : float
        Flip angle in radians.
    apodization : float, default=0
        Apodization.
    bandwidth : float, default=0
        Bandwidth in Hertz (Hz).
    delay : float, default=0
        Delay in seconds (s).
    dwell : float, default=0
    duration : float, default=4e-3
        Duration in seconds (s).
    freq_offset : float, default=0
        Frequency offset in Hertz (Hz).
    phase_offset : float, default=0
        Phase offset in Hertz (Hz).
    system : Opts, default=Opts()
        System limits.
    time_bw_product : int, default=4
        Time-bandwidth product.
    use : str, default='undefined'
        Use of radio-frequency Gauss pulse event.
        Must be one of 'excitation', 'refocusing', 'inversion',
        'saturation', 'preparation', 'other', 'undefined'.
    freq_ppm : float, default=0
        PPM frequency offset.
    phase_ppm : float, default=0
        PPM phase offset.
    filter_type : str, default="ls"
        Type of filter to use: ``"ms"`` (sinc),
        ``"pm``, (Parks-McClellan equal-ripple),
        ``"min"`` (minphase using factored pm),
        ``"max"`` (maxphase using factored pm), ``"ls"`` (least squares).
    passband_ripple_lvl : float, default=0.01
        Passband ripple level in :math:'M_0^{-1}'.
    stopband_ripple_lvl : float, default=0.01
        Stopband ripple level in :math:'M_0^{-1}'.
    cancel_alpha_phs : bool, default=False
        For 'ex' pulses, absorb the alpha phase
        profile from beta's profile, so they cancel for a flatter
        total phase.

    Returns
    -------
    NDArray[complex]
        Radio-frequency pulse event with Shinnar-LeRoux pulse shape.

    Reference
    ---------
    Pauly, J., Le Roux, Patrick., Nishimura, D., and Macovski, A.(1991).
        Parameter Relations for the Shinnar-LeRoux Selective Excitation
        Pulse Design Algorithm.
        IEEE Transactions on Medical Imaging, Vol 10, No 1, 53-65.

    Notes
    -----
    Adapted from https://github.com/mikgroup/sigpy/blob/main/sigpy/mri/rf/slr.py

    """
    if system is None:
        system = Opts.default

    valid_pulse_uses = get_supported_rf_uses()
    if use != '' and use not in valid_pulse_uses:
        raise ValueError(
            f'Invalid use parameter. Must be one of {valid_pulse_uses}. Passed: {use}'
        )

    if dwell == 0:
        dwell = system.rf_raster_time

    if bandwidth == 0 and time_bw_product == 0:
        raise ValueError('User must provide bandwidth or time-bandwidth product.')
    if bandwidth != 0 and time_bw_product != 0:
        raise ValueError(
            'User must provide either bandwidth or time-bandwidth product, not both.'
        )
    if bandwidth == 0:
        bandwidth = time_bw_product / duration
    else:
        time_bw_product = bandwidth * duration

    # Calculate number of samples
    n_samples = round(duration / dwell)

    # Calculate envelope
    signal = dzrf(
        n_samples,
        time_bw_product,
        use2ptype(use),
        filter_type,
        passband_ripple_lvl,
        stopband_ripple_lvl,
        cancel_alpha_phs,
    )

    # Pass to PyPulseq
    return make_arbitrary_rf(
        signal,
        flip_angle,
        bandwidth,
        delay,
        dwell,
        freq_offset,
        False,  # no_signal_scaling
        0,  # max_grad, not used w/o slice selection
        0,  # max_slew, not used w/o slice selection
        phase_offset,
        False,  # return_gz
        0,  # slice_thickness
        system,
        time_bw_product,
        use,
        freq_ppm,
        phase_ppm,
    )


# %% Utils
def use2ptype(use, flip_angle):
    small_tip = np.rad2deg(flip_angle) <= 45.0
    if use == 'refocusing':
        return 'se'
    if use == 'preparation' or use == 'inversion':
        return 'inv'

    # excitation, other and undefined
    if small_tip:
        return 'st'
    return 'ex'


def dzrf(n, tb, ptype, ftype, d1, d2, cancel_alpha_phs):
    bsf, d1, d2 = calc_ripples(ptype, d1, d2)

    if ftype == 'ms':  # sinc
        b = msinc(n, tb / 4)
    elif ftype == 'pm':  # linphase
        b = dzlp(n, tb, d1, d2)
    elif ftype == 'min':  # minphase
        b = dzmp(n, tb, d1, d2)
        b = b[::-1]
    elif ftype == 'max':  # maxphase
        b = dzmp(n, tb, d1, d2)
    elif ftype == 'ls':  # least squares
        b = dzls(n, tb, d1, d2)
    else:
        raise Exception('Filter type ("{}") is not recognized.'.format(ftype))

    if ptype == 'st':
        rf = b
    elif ptype == 'ex':
        b = bsf * b
        rf = b2rf(b, cancel_alpha_phs)
    else:
        b = bsf * b
        rf = b2rf(b)

    return rf


def calc_ripples(ptype, d1, d2):
    if ptype == 'st':
        bsf = 1
    elif ptype == 'ex':
        bsf = np.sqrt(1 / 2)
        d1 = np.sqrt(d1 / 2)
        d2 = d2 / np.sqrt(2)
    elif ptype == 'se':
        bsf = 1
        d1 = d1 / 4
        d2 = np.sqrt(d2)
    elif ptype == 'inv':
        bsf = 1
        d1 = d1 / 8
        d2 = np.sqrt(d2 / 2)
    elif ptype == 'sat':
        bsf = np.sqrt(1 / 2)
        d1 = d1 / 2
        d2 = np.sqrt(d2)
    else:
        raise Exception('Pulse type ("{}") is not recognized.'.format(ptype))

    return bsf, d1, d2


def dzls(n, tb, d1, d2):
    di = dinf(d1, d2)
    w = di / tb
    f = np.asarray([0, (1 - w) * (tb / 2), (1 + w) * (tb / 2), (n / 2)])
    f = f / (n / 2)
    m = [1, 1, 0, 0]
    w = [1, d1 / d2]

    # Create filter
    h = scipy.signal.firls(n + 1, f, m, weight=w)

    # Shift the filter half a sample to make it symmetric, like in MATLAB
    c = np.exp(
        1j
        * 2
        * np.pi
        / (2 * (n + 1))
        * np.concatenate([np.arange(0, n / 2 + 1, 1), np.arange(-n / 2, 0, 1)])
    )
    h = np.fft.ifft(np.fft.fft(h, norm='ortho') * c, norm='ortho').real

    # Lop off extra sample
    h = h[:n]

    return h


def dzmp(n, tb, d1, d2):
    n2 = 2 * n - 1
    di = 0.5 * dinf(2 * d1, 0.5 * d2 * d2)
    w = di / tb
    f = np.asarray([0, (1 - w) * (tb / 2), (1 + w) * (tb / 2), (n / 2)]) / n
    m = [1, 0]
    w = [1, 2 * d1 / (0.5 * d2 * d2)]

    # Create filter
    hl = scipy.signal.remez(n2, f, m, weight=w)
    h = fmp(hl)

    return h


def fmp(h):
    ll = h.size
    lp = 128 * np.exp(np.ceil(np.log(ll) / np.log(2)) * np.log(2))
    padwidths = np.asarray([np.ceil((lp - ll) / 2), np.floor((lp - ll) / 2)])
    hp = np.pad(h, padwidths.astype(int), 'constant')
    hpf = np.fft.fft(hp)
    hpfs = hpf - np.min(np.real(hpf)) * 1.000001
    hpfmp = mag2mp(np.sqrt(np.abs(hpfs)))
    hpmp = np.fft.ifft(np.fft.ifftshift(hpfmp.conj()))
    hmp = hpmp[: int((ll + 1) / 2)]

    return hmp


def dzlp(n, tb, d1, d2):
    di = dinf(d1, d2)
    w = di / tb
    f = np.asarray([0, (1 - w) * (tb / 2), (1 + w) * (tb / 2), (n / 2)]) / n
    m = [1, 0]
    w = [1, d1 / d2]

    # Create filter
    h = scipy.signal.remez(n, f, m, weight=w)

    return h


def msinc(n, m):
    x = np.arange(-n / 2, n / 2, 1) / (n / 2)
    snc = np.sin(m * 2 * np.pi * x + 0.00001) / (m * 2 * np.pi * x + 0.00001)

    ms = snc * (0.54 + 0.46 * np.cos(np.pi * x))
    ms = ms * 4 * m / n

    return ms


def b2rf(b, cancel_alpha_phs=False):
    a = b2a(b)
    if cancel_alpha_phs:
        b_a_phase = np.fft.fft(b) * np.exp(
            -1j * np.angle(np.fft.fft(a[np.size(a) :: -1]))
        )
        b = np.fft.ifft(b_a_phase)
    rf = ab2rf(a, b)

    return rf


def b2a(b):
    n = np.size(b)
    npad = n * 16
    bcp = np.zeros(npad, dtype=np.complex128)
    bcp[0:n:1] = b
    bf = np.fft.fft(bcp)
    bfmax = np.abs(bf).max()
    if bfmax >= 1:
        bf = bf / (1e-7 + bfmax)
    afa = mag2mp(np.sqrt(1 - np.abs(bf) ** 2))
    a = np.fft.fft(afa) / npad
    a = a[0:n:1]
    a = a[::-1]

    return a


def mag2mp(x):
    n = x.size
    xl = np.log(np.abs(x))  # Log of mag spectrum
    xlf = np.fft.fft(xl)
    xlfp = xlf
    xlfp[0] = xlf[0]  # Keep DC the same
    xlfp[1 : (n // 2) : 1] = 2 * xlf[1 : (n // 2) : 1]  # Double positive frequencies
    xlfp[n // 2] = xlf[n // 2]  # keep half Nyquist the same
    xlfp[n // 2 + 1 : n : 1] = 0  # zero negative frequencies
    xlaf = np.fft.ifft(xlfp)
    a = np.exp(xlaf)  # Complex exponentiation

    return a


def ab2rf(a, b):
    n = a.size
    rf = np.zeros(n, dtype=np.complex128)

    a = a.astype(np.complex128)
    b = b.astype(np.complex128)

    for ii in range(n - 1, -1, -1):
        cj = np.sqrt(1 / (1 + np.abs(b[ii] / a[ii]) ** 2))
        sj = (cj * b[ii] / a[ii]).conj()
        theta = np.arctan2(np.abs(sj), cj)
        psi = np.angle(sj)
        rf[ii] = 2 * theta * np.exp(1j * psi)

        # Remove this rotation from polynomials
        if ii > 0:
            at = cj * a + sj * b
            bt = -sj.conj() * a + cj * b
            a = at[1 : ii + 1 : 1]
            b = bt[0:ii:1]

    return rf


def dinf(d1, d2):
    a1 = 5.309e-3
    a2 = 7.114e-2
    a3 = -4.761e-1
    a4 = -2.66e-3
    a5 = -5.941e-1
    a6 = -4.278e-1

    l10d1 = np.log10(d1)
    l10d2 = np.log10(d2)

    d = (a1 * l10d1 * l10d1 + a2 * l10d1 + a3) * l10d2 + (
        a4 * l10d1 * l10d1 + a5 * l10d1 + a6
    )

    return d
