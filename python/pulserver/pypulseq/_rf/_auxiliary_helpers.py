"""Special-purpose pulses: fat saturation and Bloch-Siegert B1 mapping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pypulseq as pp

from .._gradients import make_spoiler
from .._make_label import make_label

FATSAT_DEFAULT_FLIP_DEG = 110.0
FATSAT_DEFAULT_DURATION_S = 8.0e-3
FAT_SHIFT_PPM = -3.45

BS_DEFAULT_DURATION_S = 8.0e-3
BS_DEFAULT_FREQ_OFFSET_HZ = 4000.0
BS_DEFAULT_PEAK_B1_HZ = 500.0


def fat_shift_hz(b0_tesla: float, gamma_hz_per_t: float = 42.576e6) -> float:
    """Return the fat resonance offset (Hz) at ``b0_tesla`` (-3.45 ppm).

    Parameters
    ----------
    b0_tesla : float
        Main field strength (T).
    gamma_hz_per_t : float, optional
        Gyromagnetic ratio (Hz/T).

    Returns
    -------
    float
        Fat-water shift in Hz (negative).

    Examples
    --------
    >>> from pulserver.pypulseq._rf import _auxiliary_helpers as pulses
    >>> round(pulses.fat_shift_hz(3.0))
    -441
    """
    return FAT_SHIFT_PPM * 1e-6 * b0_tesla * gamma_hz_per_t


@dataclass(frozen=True)
class FatSatModule:
    """Fat-saturation module: spectral pulse + 3-axis spoiler + exemption labels."""

    rf: object
    spoiler: tuple
    labels: tuple
    duration_s: float


def fat_saturation(
    opts: pp.Opts,
    *,
    freq_offset_hz: float,
    flip_deg: float = FATSAT_DEFAULT_FLIP_DEG,
    duration_s: float = FATSAT_DEFAULT_DURATION_S,
    voxel_size_m: float,
    spoil_cycles: float = 4.0,
) -> FatSatModule:
    """Build a spectrally selective fat-saturation module.

    Follows the pulseq demo convention (``makeGaussPulse(110 deg, 8 ms,
    bandwidth = |freq_offset|, freqOffset = freq_offset)``) plus a 3-axis
    spoiler. The module carries ``NOPOS`` and ``NOROT`` label events
    (standard pulseq FOV-transform exemptions): the pulse takes its
    frequency shift from this interface only and must not be repositioned
    or rotated with the imaging FOV — add ``*module.labels`` to the RF
    block so the interpreter tags it exempt.

    Parameters
    ----------
    opts : pypulseq.Opts
        System limits.
    freq_offset_hz : float
        Fat resonance offset (Hz); see :func:`fat_shift_hz`.
    flip_deg : float, optional
        Saturation flip angle (deg).
    duration_s : float, optional
        Pulse duration (s).
    voxel_size_m : float
        Spoiling reference length for the post-pulse spoiler.
    spoil_cycles : float, optional
        Dephasing cycles per ``voxel_size_m``.

    Returns
    -------
    FatSatModule
        Bundled RF, spoiler trapezoids, ``(NOPOS, NOROT)`` label events, and
        total duration.

    Examples
    --------
    Build a 3 T fat-sat module and emit it with its exemption labels:

    >>> import pypulseq as pp
    >>> from pulserver.pypulseq._rf import _auxiliary_helpers as pulses
    >>> fs = pulses.fat_saturation(
    ...     pp.Opts(), freq_offset_hz=pulses.fat_shift_hz(3.0), voxel_size_m=3e-3)
    >>> [lab.label for lab in fs.labels]
    ['NOPOS', 'NOROT']
    """
    bandwidth_hz = abs(freq_offset_hz)
    rf = pp.make_gauss_pulse(
        flip_angle=np.deg2rad(flip_deg),
        duration=duration_s,
        bandwidth=bandwidth_hz,
        freq_offset=freq_offset_hz,
        system=opts,
        use="saturation",
    )
    spoiler = make_spoiler(opts, voxel_size_m, dephasing_cycles=spoil_cycles)
    labels = (
        make_label(label="NOPOS", type="SET", value=1),
        make_label(label="NOROT", type="SET", value=1),
    )
    return FatSatModule(
        rf=rf,
        spoiler=spoiler,
        labels=labels,
        duration_s=pp.calc_duration(rf) + pp.calc_duration(*spoiler),
    )


@dataclass(frozen=True)
class BlochSiegertModule:
    """Off-resonant Fermi pulse for Bloch-Siegert B1 mapping + its K_BS."""

    rf: object
    kbs_rad_per_gauss2: float
    kbs_rad: float


def bloch_siegert(
    opts: pp.Opts,
    *,
    freq_offset_hz: float = BS_DEFAULT_FREQ_OFFSET_HZ,
    duration_s: float = BS_DEFAULT_DURATION_S,
    peak_b1_hz: float = BS_DEFAULT_PEAK_B1_HZ,
    t0_fraction: float = 0.4,
    a_fraction: float = 0.02,
    dwell: float = 1e-5,
) -> BlochSiegertModule:
    """Build an off-resonant Fermi pulse for Bloch-Siegert B1 mapping.

    Sacolick et al., MRM 2010: a symmetric Fermi envelope
    ``B1(t) = B1_peak / (1 + exp((|t - T/2| - t0) / a))`` applied at
    ``+/- freq_offset_hz`` induces a B1^2-proportional phase shift
    ``phi_BS = K_BS * B1_peak^2``. The module reports

    ``kbs_rad = integral (2*pi*B1(t))^2 / (2 * 2*pi*freq_offset) dt``

    for the built pulse, plus the normalized ``K_BS`` per Gauss^2 for
    B1-map scaling. Acquire two measurements at ``+`` and ``-`` offsets
    (flip ``rf.freq_offset``) and use the phase difference.

    Parameters
    ----------
    opts : pypulseq.Opts
        System limits.
    freq_offset_hz : float, optional
        Off-resonance offset (Hz); must be well outside the water/fat band.
    duration_s : float, optional
        Pulse duration (s).
    peak_b1_hz : float, optional
        Peak B1 amplitude in Hz (``gamma * B1 / 2pi``).
    t0_fraction : float, optional
        Fermi flat-top half-width as a fraction of the duration.
    a_fraction : float, optional
        Fermi transition width as a fraction of the duration.
    dwell : float, optional
        RF sample spacing (s).

    Returns
    -------
    BlochSiegertModule
        The RF event and its Bloch-Siegert phase constants.

    Examples
    --------
    >>> import pypulseq as pp
    >>> from pulserver.pypulseq._rf import _auxiliary_helpers as pulses
    >>> bs = pulses.bloch_siegert(pp.Opts())
    >>> bs.kbs_rad > 0
    True
    """
    if freq_offset_hz == 0.0:
        raise ValueError("freq_offset_hz must be non-zero (off-resonant pulse)")

    n = max(2, int(round(duration_s / dwell)))
    t = (np.arange(n) + 0.5) * dwell
    t0 = t0_fraction * duration_s
    a = a_fraction * duration_s
    envelope_hz = peak_b1_hz / (1.0 + np.exp((np.abs(t - duration_s / 2.0) - t0) / a))

    rf = pp.make_arbitrary_rf(
        signal=envelope_hz,
        flip_angle=1.0,  # ignored with no_signal_scaling: envelope stays in Hz
        no_signal_scaling=True,
        dwell=dwell,
        freq_offset=freq_offset_hz,
        system=opts,
        use="preparation",
    )

    omega1_rad_s = 2.0 * np.pi * envelope_hz
    kbs_rad = float(np.sum(omega1_rad_s**2 / (2.0 * 2.0 * np.pi * freq_offset_hz)) * dwell)
    gamma_hz_per_gauss = opts.gamma * 1e-4
    kbs_rad_per_gauss2 = float(
        np.sum((2.0 * np.pi * gamma_hz_per_gauss * (envelope_hz / peak_b1_hz)) ** 2
               / (2.0 * 2.0 * np.pi * freq_offset_hz)) * dwell
    )
    return BlochSiegertModule(rf=rf, kbs_rad_per_gauss2=kbs_rad_per_gauss2, kbs_rad=kbs_rad)
