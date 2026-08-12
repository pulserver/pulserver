"""Bloch simulation of an RF pulse.

A port of MATLAB Pulseq's ``mr.simRf``, which upstream PyPulseq has no
equivalent of. Rotations are accumulated as quaternions over the pulse and
applied once at the end, so the cost is one rotation per time step per
frequency rather than a matrix product.

Relaxation is ignored, which is what makes a single accumulated rotation
valid: over a pulse of a few milliseconds T1 and T2 do nothing a profile
plot would show.
"""

from __future__ import annotations

__all__ = ["bloch", "sim_rf"]

import numpy as np
import pypulseq as pp

from . import _results
from ._opts import default_system


def _quat_multiply(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Hamilton product of two ``(N, 4)`` scalar-first quaternion arrays."""
    qs, qv = q[:, :1], q[:, 1:]
    rs, rv = r[:, :1], r[:, 1:]
    return np.concatenate(
        [qs * rs - np.sum(qv * rv, axis=1, keepdims=True), qs * rv + rs * qv + np.cross(qv, rv)],
        axis=1,
    )


def _rotate(q: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Apply ``q* v q`` to one ``(3,)`` vector, for every row of ``q``."""
    conjugate = q * np.array([1.0, -1.0, -1.0, -1.0])
    m = np.zeros((q.shape[0], 4))
    m[:, 1:] = vector
    return _quat_multiply(conjugate, _quat_multiply(m, q))[:, 1:]


def bloch(b1_hz, bz_hz, dt: float, *, initial=None) -> np.ndarray:
    """Integrate the Bloch equation over a hard-pulse approximation.

    Parameters
    ----------
    b1_hz : array_like
        Complex transverse field per time step, shape ``(T,)``, in Hz.
    bz_hz : array_like
        Longitudinal field, shape ``(P, T)`` or ``(P, 1)``, in Hz.
    dt : float
        Raster step, in seconds.
    initial : array_like, optional
        Starting magnetisation, ``(3,)``. Defaults to ``+z``.

    Returns
    -------
    numpy.ndarray
        Final magnetisation, shape ``(P, 3)``.
    """
    b1_hz = np.asarray(b1_hz, dtype=complex)
    bz_hz = np.atleast_2d(np.asarray(bz_hz, dtype=float))
    n_pos = bz_hz.shape[0]

    magnetisation = np.zeros((n_pos, 3))
    magnetisation[:] = np.array([0.0, 0.0, 1.0]) if initial is None else np.asarray(initial, dtype=float)

    two_pi_dt = 2.0 * np.pi * dt
    for step in range(len(b1_hz)):
        omega = np.empty((n_pos, 3))
        omega[:, 0] = two_pi_dt * b1_hz[step].real
        omega[:, 1] = two_pi_dt * b1_hz[step].imag
        omega[:, 2] = two_pi_dt * bz_hz[:, step if bz_hz.shape[1] > 1 else 0]

        angle = np.linalg.norm(omega, axis=1)
        active = angle > 1e-15
        if not np.any(active):
            continue
        axis = np.zeros_like(omega)
        axis[active] = omega[active] / angle[active, None]

        cosine, sine = np.cos(angle)[:, None], np.sin(angle)[:, None]
        dot = np.sum(axis * magnetisation, axis=1)[:, None]
        magnetisation = (
            magnetisation * cosine + np.cross(axis, magnetisation) * sine + axis * dot * (1.0 - cosine)
        )
    return magnetisation


def _adapted_step(bandwidth: float, dt: float | None) -> float:
    """MATLAB's step-size ladder: roughly ``dt ~ 1 / bandwidth / 50``."""
    if dt is not None:
        return float(dt)
    if bandwidth > 20e3:
        return 1e-6
    if bandwidth > 10e3:
        return 2e-6
    if bandwidth > 4e3:
        return 5e-6
    return 10e-6


def _bandwidth_and_centre(rf, freq_offset: float, cutoff: float, df: float) -> tuple[float, float]:
    """Width at ``cutoff`` of the peak, and the midpoint of the two flanks, in Hz.

    Not :func:`pypulseq.calc_rf_bandwidth`: that one mis-measures a pulse
    carrying a frequency offset -- on a 2 ms TBW-4 sinc shifted by 1.5 kHz it
    reports 20 Hz instead of ~1900, and leaves the peak at zero. The offset
    matters here because it decides where the simulated axis has to sit.
    """
    times = np.asarray(rf.t, dtype=float)
    signal = np.asarray(rf.signal, dtype=complex) * np.exp(2j * np.pi * freq_offset * times)
    if times.size < 2 or not np.any(signal):
        return 0.0, freq_offset

    step = float(np.min(np.diff(times)))
    grid = np.arange(times[0], times[-1] + 0.5 * step, step)
    sampled = np.interp(grid, times, signal.real) + 1j * np.interp(grid, times, signal.imag)

    # Zero-pad until one bin is no wider than the resolution asked for.
    n = max(int(round(1.0 / (df * step))), sampled.size)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(sampled, n)))
    frequency = np.fft.fftshift(np.fft.fftfreq(n, step))

    peak = int(np.argmax(spectrum))
    threshold = cutoff * spectrum[peak]
    # Walk outward from the peak rather than taking the outermost sample over
    # threshold anywhere, so a side lobe cannot widen the answer.
    left = peak
    while left > 0 and spectrum[left - 1] >= threshold:
        left -= 1
    right = peak
    while right < n - 1 and spectrum[right + 1] >= threshold:
        right += 1

    return float(frequency[right] - frequency[left]), float(0.5 * (frequency[left] + frequency[right]))


def sim_rf(
    rf,
    rephase_factor: float | None = None,
    prephase_factor: float = 0.0,
    *,
    df: float = 1.0,
    bandwidth_multiplier: float = 4.0,
    dt: float | None = None,
    system: pp.Opts | None = None,
    compat: bool = True,
):
    """Simulate an RF pulse across off-resonance, as MATLAB's ``mr.simRf`` does.

    Parameters
    ----------
    rf : SimpleNamespace
        The RF event.
    rephase_factor : float, optional
        Free precession after the pulse, as a fraction of its duration.
        Defaults to ``0`` for a refocusing pulse and to
        ``-(shape_dur - center) / shape_dur`` otherwise, which is the
        rephasing a slice-selective excitation is followed by -- without it
        the phase across the profile is the linear ramp the pulse leaves,
        not the one the sequence plays.
    prephase_factor : float, default 0.0
        The same, before the pulse. Useful for refocusing and spoiling.
    df : float, default 1.0
        Spectral resolution, in Hz.
    bandwidth_multiplier : float, default 4.0
        Width of the simulated axis, in pulse bandwidths.
    dt : float, optional
        Simulation raster, in seconds. Defaults to a step chosen from the
        bandwidth, as MATLAB does.
    system : pypulseq.Opts, optional
        Only read when the pulse carries a ppm offset, which needs ``B0``
        and ``gamma`` to become Hz.
    compat : bool, default True
        Return MATLAB's six-tuple ``(mz_z, mz_xy, f, ref_eff, mx_xy,
        my_xy)`` rather than :class:`~._results.RfResponse`.

    Returns
    -------
    mz_z : numpy.ndarray
        ``Mz`` after the pulse, starting from ``+z``. The inversion or
        saturation profile.
    mz_xy : numpy.ndarray
        Complex ``Mxy`` after the pulse, starting from ``+z``. The excitation
        profile.
    f : numpy.ndarray
        The frequency axis, in Hz.
    ref_eff : numpy.ndarray
        Refocusing efficiency, complex: its magnitude tracks the refocused
        fraction and its phase the axis of the flip.
    mx_xy, my_xy : numpy.ndarray
        Complex ``Mxy`` starting from ``+x`` and from ``+y``, which is what
        separates a good refocusing pulse from one that merely inverts.

    Examples
    --------
    >>> import numpy as np, pulserver.pypulseq as pp
    >>> rf = pp.make_sinc_pulse(flip_angle=np.pi / 2, duration=2e-3, use="excitation")
    >>> mz_z, mz_xy, f, *_ = pp.sim_rf(rf)
    >>> bool(abs(mz_xy).max() > 0.99)          # a 90 deg pulse tips fully
    True
    """
    if rephase_factor is None:
        rephase_factor = (
            0.0
            if str(getattr(rf, "use", "")) == "refocusing"
            else -(float(rf.shape_dur) - float(rf.center)) / float(rf.shape_dur)
        )

    # The axis has to span the pulse's own offset as well as its width, and
    # the step follows from that total -- measured once, as MATLAB does.
    freq_offset = float(getattr(rf, "freq_offset", 0.0))
    phase_offset = float(getattr(rf, "phase_offset", 0.0))
    freq_ppm = float(getattr(rf, "freq_ppm", 0.0) or 0.0)
    phase_ppm = float(getattr(rf, "phase_ppm", 0.0) or 0.0)
    if freq_ppm or phase_ppm:
        limits = default_system(system)
        per_ppm = 1e-6 * float(limits.gamma) * float(limits.B0)
        freq_offset += freq_ppm * per_ppm
        phase_offset += phase_ppm * per_ppm

    # The axis has to span the pulse's own offset as well as its width.
    width, centre = _bandwidth_and_centre(rf, freq_offset, 0.5, df * 10.0)
    bandwidth = abs(width) + abs(centre)
    step = _adapted_step(bandwidth, dt)

    n_steps = round(float(rf.shape_dur) / step)
    times = (np.arange(1, n_steps + 1) * step) - 0.5 * step
    n_freq = max(round(bandwidth / df), 2)
    omega = 2.0 * np.pi * np.linspace(
        centre - bandwidth_multiplier * bandwidth / 2.0,
        centre + bandwidth_multiplier * bandwidth / 2.0,
        n_freq,
    )

    source_t = np.asarray(rf.t, dtype=float)
    signal = np.asarray(rf.signal, dtype=complex)
    envelope = 2.0 * np.pi * signal * np.exp(1j * (phase_offset + 2.0 * np.pi * freq_offset * source_t))
    shape = np.interp(times, source_t, envelope.real, left=0.0, right=0.0) + 1j * np.interp(
        times, source_t, envelope.imag, left=0.0, right=0.0
    )

    quaternion = np.zeros((n_freq, 4))
    quaternion[:, 0] = 1.0

    def free_precession(factor: float) -> np.ndarray:
        angle = -omega * step * n_steps * factor
        turn = np.zeros((n_freq, 4))
        turn[:, 0] = np.cos(angle / 2.0)
        turn[:, 3] = np.sin(angle / 2.0)
        return turn

    quaternion = _quat_multiply(quaternion, free_precession(prephase_factor))

    for value in shape:
        magnitude = np.sqrt(abs(value) ** 2 + omega**2)
        angle = -step * magnitude
        # magnitude is zero only where the pulse and the offset both vanish,
        # and there the rotation is the identity whatever axis is named.
        safe = np.where(magnitude > 0.0, magnitude, 1.0)
        half = np.sin(angle / 2.0) / safe
        turn = np.empty((n_freq, 4))
        turn[:, 0] = np.cos(angle / 2.0)
        turn[:, 1] = half * value.real
        turn[:, 2] = half * value.imag
        turn[:, 3] = half * omega
        quaternion = _quat_multiply(quaternion, turn)

    quaternion = _quat_multiply(quaternion, free_precession(rephase_factor))

    from_z = _rotate(quaternion, np.array([0.0, 0.0, 1.0]))
    from_x = _rotate(quaternion, np.array([1.0, 0.0, 0.0]))
    from_y = _rotate(quaternion, np.array([0.0, 1.0, 0.0]))

    frequency = omega / (2.0 * np.pi)
    mz_z = from_z[:, 2]
    mz_xy = from_z[:, 0] + 1j * from_z[:, 1]
    mx_xy = from_x[:, 0] + 1j * from_x[:, 1]
    my_xy = from_y[:, 0] + 1j * from_y[:, 1]
    ref_eff = (mx_xy + 1j * my_xy) / 2.0

    if compat:
        return mz_z, mz_xy, frequency, ref_eff, mx_xy, my_xy
    return _results.RfResponse.of(
        mz_z=mz_z,
        mz_xy=mz_xy,
        frequency=frequency,
        ref_eff=ref_eff,
        mx_xy=mx_xy,
        my_xy=my_xy,
    )
