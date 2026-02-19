"""Adiabatic pulse routine."""

__all__ = ['as_adiabatic']

from types import SimpleNamespace
from typing import Union

import numpy as np
from pypulseq import Opts, make_arbitrary_rf

from ._slr import ab2rf, b2a


def as_adiabatic(
    pulse: SimpleNamespace,
    duration: Union[float, None] = None,
    quadforce: float = 1.4e4,
    threshold: float = 0.1,
) -> SimpleNamespace:
    """
    Create an adiabatic pulse from input linear phased pulse.

    Input pulse can be generated using SLR design algorithm.

    Parameters
    ----------
    pulse : SimpleNamespace
        Input RF pulse event.
    duration : Union[float, None], default=None
        Target RF pulse duration. If not provided,
        retain duration of input RF.
    quadforce: float, default=1.4e4
        Quadratic phase rate.
    threshold: float, default=0.1
        Amplitude threshold to truncate final RF profile.

    Returns
    -------
    SimpleNamespace
        Radio-frequency pulse event with quadratic phase.

    """
    # Hardcoded params
    quadforce = 1.4e4
    threshold = 0.1

    # Extract parameters from input pulse
    signal = pulse.signal

    # Get rf dead time and ringdown time
    system = Opts.default  # Dummy
    system.dead_time = pulse.rf_dead_time
    system.ringdown_time = pulse.rf_ringdown_time

    # Get dwell time
    dwell = pulse.shape_dur / signal.size

    # Pad to target duration
    if duration is not None:
        pad_time = duration - pulse.shape_dur
        pad_size = int(pad_time / dwell)
        if pad_size > 0:
            signal = np.pad(signal, (0, pad_size))

    # Create quadratic phase increment
    signal_spec = np.fft.ifftshift(np.fft.fft(np.fft.fftshift(signal)))

    # Build filter
    filt = np.linspace(-1.0, 1.0, len(signal_spec))

    # Apply filter
    signal_spec = signal_spec * np.exp(1j * quadforce * filt**2)

    # SLR for envelope
    # beta polynomial of quadratic profile
    bz_adiabatic = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(signal_spec)))

    # Normalize so the max magnetization is 1
    bz_adiabatic = bz_adiabatic / max(abs(np.fft.fft(bz_adiabatic)))

    # Here comes the inverse SLR
    az_adiabatic = b2a(bz_adiabatic)
    signal = ab2rf(az_adiabatic, bz_adiabatic)

    # Truncate the rf
    signal_norm = signal / max(abs(signal))
    idx = np.where(abs(signal_norm) > threshold)[0]
    signal = signal[idx[0] : idx[-1] + 1]

    # Scale to get waveform in Hz
    signal = signal / (2 * np.pi * dwell)

    # Pass to PyPulseq
    return make_arbitrary_rf(
        signal,
        np.pi,  # flip_angle, not used when no_signal_scaling=True
        1e3,  # bandwidth, not used w/o slice selection
        pulse.delay,
        dwell,
        pulse.freq_offset,
        True,  # no_signal_scaling
        0,  # max_grad, not used w/o slice selection
        0,  # max_slew, not used w/o slice selection
        pulse.phase_offset,
        False,  # return_gz
        0,  # slice_thickness
        system,
        1,  # time bandwidth product, not used w/o slice selection
        pulse.use,
        pulse.freq_ppm,
        pulse.phase_ppm,
    )
