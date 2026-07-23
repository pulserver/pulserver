"""RF excitation builders shared by the sequence zoo."""

from __future__ import annotations

import numpy as np
import pypulseq as pp

RF_SPOILING_INC_DEG = 117.0

DEFAULT_SLICE_RF_DURATION_S = 2.00e-3
DEFAULT_SLICE_RF_TBW = 4.0
DEFAULT_SLICE_RF_APODIZATION = 0.5
DEFAULT_HARD_RF_DURATION_S = 1.0e-3
DEFAULT_ADIABATIC_DURATION_S = 10.24e-3


def slice_selective(
    opts: pp.Opts,
    flip_deg: float,
    thickness_m: float,
    *,
    duration_s: float = DEFAULT_SLICE_RF_DURATION_S,
    tbw: float = DEFAULT_SLICE_RF_TBW,
    apodization: float = DEFAULT_SLICE_RF_APODIZATION,
    use: str | None = None,
):
    """Build a slice- or slab-selective sinc excitation.

    Parameters
    ----------
    opts : pypulseq.Opts
        System limits.
    flip_deg : float
        Flip angle in degrees.
    thickness_m : float
        Slice (or slab) thickness in metres.
    duration_s : float, optional
        RF pulse duration (s).
    tbw : float, optional
        Time-bandwidth product.
    apodization : float, optional
        Sinc apodization factor.
    use : str or None, optional
        Pulse use tag; defaults to ``"excitation"``.

    Returns
    -------
    rf : SimpleNamespace
        The sinc RF event.
    gz : SimpleNamespace
        Slice-select gradient.
    gz_reph : SimpleNamespace
        Slice-rephase gradient.

    Examples
    --------
    Build a slice-selective 90° excitation:

    >>> import pypulseq as pp
    >>> from pulserver.design._rf import _excitation_helpers as excitation
    >>> opts = pp.Opts()
    >>> rf, gz, gz_reph = excitation.slice_selective(opts, flip_deg=90, thickness_m=5e-3)
    """
    return pp.make_sinc_pulse(
        flip_angle=np.deg2rad(flip_deg),
        duration=duration_s,
        slice_thickness=thickness_m,
        apodization=apodization,
        time_bw_product=tbw,
        system=opts,
        use=use if use is not None else "excitation",
        return_gz=True,
    )


def nonselective(
    opts: pp.Opts,
    flip_deg: float,
    *,
    duration_s: float = DEFAULT_HARD_RF_DURATION_S,
    use: str | None = None,
):
    """Build a non-selective rectangular (hard) pulse.

    Parameters
    ----------
    opts : pypulseq.Opts
        System limits.
    flip_deg : float
        Flip angle in degrees.
    duration_s : float, optional
        Pulse duration (s).
    use : str or None, optional
        Pulse use tag; defaults to ``"excitation"``.

    Returns
    -------
    rf : SimpleNamespace
        The block RF event.

    Examples
    --------
    Build a hard 180° inversion:

    >>> import pypulseq as pp
    >>> from pulserver.design._rf import _excitation_helpers as excitation
    >>> rf = excitation.nonselective(pp.Opts(), flip_deg=180, use="inversion")
    """
    return pp.make_block_pulse(
        flip_angle=np.deg2rad(flip_deg),
        duration=duration_s,
        system=opts,
        use=use if use is not None else "excitation",
    )


def adiabatic_inversion(
    opts: pp.Opts,
    *,
    duration_s: float = DEFAULT_ADIABATIC_DURATION_S,
    dwell: float = 1e-5,
):
    """Build a hyperbolic-secant adiabatic inversion pulse.

    Parameters
    ----------
    opts : pypulseq.Opts
        System limits.
    duration_s : float, optional
        Pulse duration (s).
    dwell : float, optional
        RF sample spacing (s).

    Returns
    -------
    rf : SimpleNamespace
        The adiabatic RF event.

    Examples
    --------
    >>> import pypulseq as pp
    >>> from pulserver.design._rf import _excitation_helpers as excitation
    >>> rf = excitation.adiabatic_inversion(pp.Opts())
    """
    return pp.make_adiabatic_pulse(
        pulse_type="hypsec",
        duration=duration_s,
        dwell=dwell,
        system=opts,
        use="inversion",
    )


# --- Multiband phase tables, ported from refcode/sigpy sigpy/mri/rf/multiband.py ---
# Wong ISMRM 2012 p.2209 ("wong"), Malik ISMRM 2015 p.2398 ("malik"),
# Grissom quadratic ("quadratic").
_WONG_PHASES = {
    3: [0.0, 0.73, 4.602],
    4: [0.0, 3.875, 5.94, 6.197],
    5: [0.0, 3.778, 5.335, 0.872, 0.471],
    6: [0.0, 2.005, 1.674, 5.012, 5.736, 4.123],
    7: [0.0, 3.002, 5.998, 5.909, 2.624, 2.528, 2.440],
    8: [0.0, 1.036, 3.414, 3.778, 3.215, 1.756, 4.555, 2.467],
    9: [0.0, 1.250, 1.783, 3.558, 0.739, 3.319, 1.296, 0.521, 5.332],
    10: [0.0, 4.418, 2.360, 0.677, 2.253, 3.472, 3.040, 3.974, 1.192, 2.510],
    11: [0.0, 5.041, 4.285, 3.001, 5.765, 4.295, 0.056, 4.213, 6.040, 1.078, 2.759],
    12: [0.0, 2.755, 5.491, 4.447, 0.231, 2.499, 3.539, 2.931, 2.759, 5.376, 4.554, 3.479],
    13: [0.0, 0.603, 0.009, 4.179, 4.361, 4.837, 0.816, 5.995, 4.150, 0.417, 1.520, 4.517, 1.729],
    14: [0.0, 3.997, 0.830, 5.712, 3.838, 0.084, 1.685, 5.328, 0.237, 0.506, 1.356, 4.025, 4.483, 4.084],
    15: [0.0, 4.126, 2.266, 0.957, 4.603, 0.815, 3.475, 0.977, 1.449, 1.192, 0.148, 0.939, 2.531, 3.612, 4.801],
    16: [0.0, 4.359, 3.510, 4.410, 1.750, 3.357, 2.061, 5.948, 3.000, 2.822, 0.627, 2.768, 3.875, 4.173, 4.224, 5.941],
}

_MALIK_PHASES = {
    4: [0.0, np.pi, np.pi, 0.0],
    5: [0.0, 0.0, np.pi, 0.0, 0.0],
    6: [1.691, 2.812, 1.157, -1.157, -2.812, -1.691],
    7: [2.582, -0.562, 0.102, 0.0, -0.102, 0.562, -2.582],
    8: [2.112, 0.220, 1.464, 1.992, -1.992, -1.464, -0.220, -2.112],
    9: [0.479, -2.667, -0.646, -0.419, 0.0, 0.419, 0.646, 2.667, -0.479],
    10: [1.683, -2.395, 2.913, 0.304, 0.737, -0.737, -0.304, -2.913, 2.395, -1.683],
    11: [1.405, 0.887, -1.854, 0.070, -1.494, 0.0, 1.494, -0.070, 1.854, -0.887, -1.405],
    12: [1.729, 0.444, 0.722, 2.190, -2.196, 0.984, -0.984, 2.196, -2.190, -0.722, -0.444, -1.729],
}


def multiband_phases(n_bands: int, phase_scheme: str = "quadratic") -> np.ndarray:
    """Return per-band phases minimizing peak B1 of a multiband pulse.

    Ported from ``refcode/sigpy`` (``sigpy.mri.rf.multiband.mb_phs_tab``).

    Parameters
    ----------
    n_bands : int
        Number of simultaneous bands.
    phase_scheme : str, optional
        ``"quadratic"`` (Grissom, any band count), ``"wong"`` (Wong ISMRM
        2012, 3-16 bands), ``"malik"`` (Malik ISMRM 2015, 4-12 bands,
        Hermitian/amplitude-modulated), or ``"none"`` (all zero).

    Returns
    -------
    numpy.ndarray
        Per-band phases (rad), length ``n_bands``.

    Examples
    --------
    >>> from pulserver.design._rf import _excitation_helpers as excitation
    >>> len(excitation.multiband_phases(4, "quadratic"))
    4
    """
    if phase_scheme == "none":
        return np.zeros(n_bands)
    if phase_scheme == "quadratic":
        k = 3.4 / n_bands
        return k * (np.arange(n_bands) - (n_bands - 1) / 2.0) ** 2
    if phase_scheme == "wong":
        if n_bands not in _WONG_PHASES:
            raise ValueError("Wong phases valid for 3 <= n_bands <= 16")
        return np.asarray(_WONG_PHASES[n_bands])
    if phase_scheme == "malik":
        if n_bands not in _MALIK_PHASES:
            raise ValueError("Malik phases valid for 4 <= n_bands <= 12")
        return np.asarray(_MALIK_PHASES[n_bands])
    raise ValueError(f"Unknown phase_scheme '{phase_scheme}'")


def multiband(base_rf, n_bands: int, band_separation: float, *, phase_scheme: str = "quadratic"):
    """Multiband (SMS) modulate a slice-selective RF pulse.

    Ported from ``refcode/sigpy`` (``sigpy.mri.rf.multiband.mb_rf``): the
    base pulse's envelope is multiplied by a sum of complex exponentials,
    one per band, symmetric about the base slice position.

    Parameters
    ----------
    base_rf : SimpleNamespace
        Single-band RF event (e.g. from :func:`slice_selective`); modified
        copy returned, original untouched.
    n_bands : int
        Number of simultaneous slices.
    band_separation : float
        Normalized band separation ``slice_sep / slice_thickness * tbw``
        (sigpy's ``band_sep`` convention), where ``tbw`` is the base
        pulse's time-bandwidth product.
    phase_scheme : str, optional
        Per-band phase set (see :func:`multiband_phases`).

    Returns
    -------
    SimpleNamespace
        The multibanded RF event (complex envelope).

    Examples
    --------
    Build a 3-band SMS excitation from a slice-selective base:

    >>> import pypulseq as pp
    >>> from pulserver.design._rf import _excitation_helpers as excitation
    >>> rf, gz, _ = excitation.slice_selective(pp.Opts(), 90, 5e-3)
    >>> rf_mb = excitation.multiband(rf, n_bands=3, band_separation=12.0)
    >>> rf_mb.signal.shape == rf.signal.shape
    True
    """
    import copy as _copy

    phases = multiband_phases(n_bands, phase_scheme)
    signal = np.asarray(base_rf.signal, dtype=np.complex128)
    n = signal.size
    modulation = np.zeros(n, dtype=np.complex128)
    sample_idx = np.arange(-n / 2, n / 2, 1)
    for band in range(n_bands):
        offset = band - (n_bands - 1) / 2.0
        modulation += np.exp(1j * 2 * np.pi / n * band_separation * sample_idx * offset) * np.exp(
            1j * phases[band]
        )

    rf_out = _copy.deepcopy(base_rf)
    rf_out.signal = modulation * signal
    return rf_out


def frequency_selective(
    opts: pp.Opts,
    flip_deg: float,
    bandwidth_hz: float,
    *,
    center_hz: float = 0.0,
    tbw: float = 4.0,
    use: str | None = None,
    n_bands: int = 1,
    band_separation_hz: float = 0.0,
    phase_scheme: str = "quadratic",
):
    """Build a frequency-selective (non-spatial) Gaussian pulse.

    The duration follows from the requested spectral bandwidth via the
    time-bandwidth product (``duration = tbw / bandwidth_hz``); no gradient
    is played, so selectivity is purely spectral (e.g. fat/water). With
    ``n_bands > 1`` the envelope is multiband-modulated to excite several
    frequency bands simultaneously, ``band_separation_hz`` apart.

    Parameters
    ----------
    opts : pypulseq.Opts
        System limits.
    flip_deg : float
        Flip angle (deg).
    bandwidth_hz : float
        Spectral full bandwidth of each band (Hz).
    center_hz : float, optional
        Center frequency offset (Hz).
    tbw : float, optional
        Time-bandwidth product (sets the duration).
    use : str or None, optional
        Pulse use tag; defaults to ``"saturation"``.
    n_bands : int, optional
        Number of frequency bands.
    band_separation_hz : float, optional
        Band center-to-center separation (Hz); required when ``n_bands > 1``.
    phase_scheme : str, optional
        Per-band phase set for the multiband case.

    Returns
    -------
    SimpleNamespace
        The RF event (``freq_offset`` set to ``center_hz``).

    Examples
    --------
    Build a 90° water-selective pulse of 200 Hz bandwidth:

    >>> import pypulseq as pp
    >>> from pulserver.design._rf import _excitation_helpers as excitation
    >>> rf = excitation.frequency_selective(pp.Opts(), 90, 200.0)
    >>> rf.freq_offset
    0.0
    """
    if bandwidth_hz <= 0.0:
        raise ValueError("bandwidth_hz must be > 0")
    if n_bands > 1 and band_separation_hz <= 0.0:
        raise ValueError("band_separation_hz must be > 0 for n_bands > 1")

    duration_s = tbw / bandwidth_hz
    rf = pp.make_gauss_pulse(
        flip_angle=np.deg2rad(flip_deg),
        duration=duration_s,
        freq_offset=center_hz,
        time_bw_product=tbw,
        system=opts,
        use=use if use is not None else "saturation",
    )
    if n_bands > 1:
        # sigpy's normalized band_sep: cycles across the pulse per band step.
        band_sep_normalized = band_separation_hz * duration_s
        rf = multiband(rf, n_bands, band_sep_normalized, phase_scheme=phase_scheme)
    return rf
