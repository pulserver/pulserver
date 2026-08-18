"""Small, NumPy/SciPy-only subset of SigPy's SLR design implementation.

This module is derived from ``sigpy.mri.rf.slr`` and the original Pauly
``rf_tools`` toolbox.  Only the functions needed by pulserver's public SLR
pulse factory are retained.  Keeping the implementation here avoids making
SigPy (and its array/optimization stack) a runtime dependency.
"""

from __future__ import annotations

import numpy as np
from scipy import signal


def _dinf(d1: float, d2: float) -> float:
    a1, a2, a3 = 5.309e-3, 7.114e-2, -4.761e-1
    a4, a5, a6 = -2.66e-3, -5.941e-1, -4.278e-1
    l10d1 = np.log10(d1)
    l10d2 = np.log10(d2)
    return float(
        (a1 * l10d1 * l10d1 + a2 * l10d1 + a3) * l10d2
        + (a4 * l10d1 * l10d1 + a5 * l10d1 + a6)
    )


def _calc_ripples(pulse_type: str, d1: float, d2: float) -> tuple[float, float, float]:
    if pulse_type == "st":
        scale = 1.0
    elif pulse_type == "ex":
        scale = np.sqrt(0.5)
        d1, d2 = np.sqrt(d1 / 2.0), d2 / np.sqrt(2.0)
    elif pulse_type == "se":
        scale = 1.0
        d1, d2 = d1 / 4.0, np.sqrt(d2)
    elif pulse_type == "inv":
        scale = 1.0
        d1, d2 = d1 / 8.0, np.sqrt(d2 / 2.0)
    elif pulse_type == "sat":
        scale = np.sqrt(0.5)
        d1, d2 = d1 / 2.0, np.sqrt(d2)
    else:
        raise ValueError("pulse_type must be one of 'st', 'ex', 'se', 'inv', or 'sat'")
    return float(scale), float(d1), float(d2)


def _least_squares(n: int, tbw: float, d1: float, d2: float) -> np.ndarray:
    transition = _dinf(d1, d2) / tbw
    bands = np.asarray(
        [0.0, (1.0 - transition) * tbw / 2.0, (1.0 + transition) * tbw / 2.0, n / 2.0]
    ) / (n / 2.0)
    h = signal.firls(n + 1, bands, [1.0, 1.0, 0.0, 0.0], weight=[1.0, d1 / d2])
    # MATLAB's firls result is shifted by half a sample relative to SciPy.
    phase = np.exp(
        1j
        * 2.0
        * np.pi
        / (2.0 * (n + 1))
        * np.concatenate((np.arange(0, n / 2 + 1), np.arange(-n / 2, 0)))
    )
    return np.real(np.fft.ifft(np.fft.fft(h) * phase))[:n]


def _linear_phase(n: int, tbw: float, d1: float, d2: float) -> np.ndarray:
    transition = _dinf(d1, d2) / tbw
    bands = (
        np.asarray(
            [
                0.0,
                (1.0 - transition) * tbw / 2.0,
                (1.0 + transition) * tbw / 2.0,
                n / 2.0,
            ]
        )
        / n
    )
    return signal.remez(n, bands, [1.0, 0.0], weight=[1.0, d1 / d2], fs=1.0)


def _minimum_phase_spectrum(magnitude: np.ndarray) -> np.ndarray:
    log_magnitude = np.log(np.maximum(np.abs(magnitude), np.finfo(float).tiny))
    spectrum = np.fft.fft(log_magnitude)
    causal = np.zeros_like(spectrum)
    causal[0] = spectrum[0]
    causal[1 : magnitude.size // 2] = 2.0 * spectrum[1 : magnitude.size // 2]
    causal[magnitude.size // 2] = spectrum[magnitude.size // 2]
    return np.exp(np.fft.ifft(causal))


def _factor_minimum_phase(h: np.ndarray) -> np.ndarray:
    length = h.size
    padded_length = int(128 * 2 ** np.ceil(np.log2(length)))
    pad_left = int(np.ceil((padded_length - length) / 2.0))
    pad_right = int(np.floor((padded_length - length) / 2.0))
    spectrum = np.fft.fft(np.pad(h, (pad_left, pad_right)))
    positive = spectrum - np.min(np.real(spectrum)) * 1.000001
    minimum_phase = _minimum_phase_spectrum(np.sqrt(np.abs(positive)))
    impulse = np.fft.ifft(np.fft.ifftshift(np.conj(minimum_phase)))
    return impulse[: (length + 1) // 2]


def _minimum_phase(n: int, tbw: float, d1: float, d2: float) -> np.ndarray:
    n2 = 2 * n - 1
    transition = 0.5 * _dinf(2.0 * d1, 0.5 * d2 * d2) / tbw
    bands = (
        np.asarray(
            [
                0.0,
                (1.0 - transition) * tbw / 2.0,
                (1.0 + transition) * tbw / 2.0,
                n / 2.0,
            ]
        )
        / n
    )
    linear = signal.remez(
        n2, bands, [1.0, 0.0], weight=[1.0, 2.0 * d1 / (0.5 * d2 * d2)], fs=1.0
    )
    return _factor_minimum_phase(linear)


def _windowed_sinc(n: int, lobes: float) -> np.ndarray:
    x = np.arange(-n / 2, n / 2) / (n / 2)
    arg = lobes * 2.0 * np.pi * x + 1e-5
    result = np.sin(arg) / arg
    result *= 0.54 + 0.46 * np.cos(np.pi * x)
    return result * (4.0 * lobes / n)


def _beta_to_alpha(beta: np.ndarray) -> np.ndarray:
    n = beta.size
    padded = np.zeros(n * 16, dtype=np.complex128)
    padded[:n] = beta
    beta_spectrum = np.fft.fft(padded)
    peak = np.max(np.abs(beta_spectrum))
    if peak >= 1.0:
        beta_spectrum /= 1e-7 + peak
    alpha_spectrum = _minimum_phase_spectrum(np.sqrt(1.0 - np.abs(beta_spectrum) ** 2))
    alpha = np.fft.fft(alpha_spectrum) / padded.size
    return alpha[:n][::-1]


def _alpha_beta_to_rf(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    n = alpha.size
    rf = np.zeros(n, dtype=np.complex128)
    alpha = alpha.astype(np.complex128, copy=True)
    beta = beta.astype(np.complex128, copy=True)
    for index in range(n - 1, -1, -1):
        cosine = np.sqrt(1.0 / (1.0 + np.abs(beta[index] / alpha[index]) ** 2))
        sine = np.conj(cosine * beta[index] / alpha[index])
        rf[index] = 2.0 * np.arctan2(np.abs(sine), cosine) * np.exp(1j * np.angle(sine))
        if index > 0:
            alpha_tmp = cosine * alpha + sine * beta
            beta_tmp = -np.conj(sine) * alpha + cosine * beta
            alpha = alpha_tmp[1 : index + 1]
            beta = beta_tmp[:index]
    return rf


def _beta_to_rf(beta: np.ndarray, cancel_alpha_phase: bool) -> np.ndarray:
    alpha = _beta_to_alpha(beta)
    if cancel_alpha_phase:
        beta_spectrum = np.fft.fft(beta) * np.exp(
            -1j * np.angle(np.fft.fft(alpha[::-1]))
        )
        beta = np.fft.ifft(beta_spectrum)
    return _alpha_beta_to_rf(alpha, beta)


def design_slr(
    n: int,
    time_bandwidth_product: float,
    *,
    pulse_type: str = "st",
    filter_type: str = "ls",
    passband_ripple: float = 0.01,
    stopband_ripple: float = 0.01,
    cancel_alpha_phase: bool = False,
) -> np.ndarray:
    """Return a dimensionless SLR RF waveform."""
    if n < 8 or n % 2:
        raise ValueError("n must be an even integer >= 8")
    if time_bandwidth_product <= 0:
        raise ValueError("time_bandwidth_product must be > 0")
    if not 0 < passband_ripple < 1 or not 0 < stopband_ripple < 1:
        raise ValueError("passband_ripple and stopband_ripple must lie in (0, 1)")

    scale, d1, d2 = _calc_ripples(pulse_type, passband_ripple, stopband_ripple)
    if filter_type == "ls":
        beta = _least_squares(n, time_bandwidth_product, d1, d2)
    elif filter_type == "pm":
        beta = _linear_phase(n, time_bandwidth_product, d1, d2)
    elif filter_type == "min":
        beta = _minimum_phase(n, time_bandwidth_product, d1, d2)[::-1]
    elif filter_type == "max":
        beta = _minimum_phase(n, time_bandwidth_product, d1, d2)
    elif filter_type == "ms":
        beta = _windowed_sinc(n, time_bandwidth_product / 4.0)
    else:
        raise ValueError("filter_type must be one of 'ls', 'pm', 'min', 'max', or 'ms'")

    if pulse_type == "st":
        return np.asarray(beta, dtype=np.complex128)
    return _beta_to_rf(scale * beta, cancel_alpha_phase)
