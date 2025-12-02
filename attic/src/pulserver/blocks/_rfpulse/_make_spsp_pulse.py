"""Spatial-Spectral RF pulses design routines."""

__all__ = ["make_spsp_pulse"]

from types import SimpleNamespace
from collections.abc import Iterable

import numpy as np

from pypulseq import make_arbitrary_rf
from pypulseq import make_arbitrary_grad
from pypulseq.convert import convert
from pypulseq.opts import Opts
from pypulseq.supported_labels_rf_use import get_supported_rf_uses

from ._spsp import ss_design, ss_globals
from ._plot import display_pulse

USE_TABLE = {"excitation": "ex", "refocusing": "se", "inversion": "inv"}


def make_spsp_pulse(
    system: Opts,
    flip_angle: float | Iterable[float],
    slice_thickness: float,
    spectral_band_edges: Iterable[float] | str = "water",
    freq_offset: float | None = None,
    spatial_time_bw_product: float = 4.0,
    spatial_ripples: Iterable[float] = (0.01, 0.01),
    spatial_filter_type: str = "ls",
    spectral_ripples: tuple[float] = (0.02, 0.005),
    spectral_filter_type: str = "min",
    use: str = "excitation",
    verbose: bool = False,
    plot: bool = False,
) -> dict:
    """
    Design a spectral-spatial RF pulse.

    This function designs a pulse for simultaneous spatial and spectral selectivity, such as used in
    MR imaging or spectroscopy sequences. The function returns the gradient and RF waveforms necessary
    to create the specified spectral-spatial pulse.

    Parameters
    ----------
    system : Opts
        System limits.
    flip_angle : float | Iterable[float]
        Flip angle specification for each spectral band, in ``[deg]``.
        For water selective pulses (``spectral_band_edges="water"``),
        if the user provides a scalar, we assume that the flip angle refers
        to water and fat flip angle is ``0.0 [deg]``.
    slice_thickness : float
        Thickness of the slice in ``[mm]``.
    spectral_band_edges : Iterable[float] | str, optional
        Frequency band edge specification in ``[Hz]``.
        Can also be a string (e.g., ``"water"``),
        in which case a water-selective pulse is designed and the
        spectral band is set according to the ``B0`` value.
    freq_offset : float | None, optional
        Frequency offset in Hertz (Hz).
    use : str, optional
        Use of radio-frequency pulse. Must be one of ``"excitation"`` (default),
        ``"refocusing"`` or ``"inversion"``.
    spatial_time_bw_product : float
        Time-bandwidth product for the spatial dimension.
    spatial_ripples : Iterable[float]
        Spatial passband and stopband ripples, in the form ``(pass_ripple, stop_ripple)``.
    spatial_filter_type : str, optional
        Type of filter for the spatial dimension: ``"ms"``, ``"ls"`` (default), ``"pm"``, ``"min"``, ``"max"``.
    spectral_ripples : Iterable[float]
        Spectral passband and stopband ripples.
    spectral_filter_type : str, optional
        Type of filter for the spectral dimension: ``"min"`` (default), ``"max"``, ``"lin"``.
    verbose : bool, optional
        If ``True``, print debug messages. Default is ``False``.

    Returns
    -------
    rf_block : dict
        Dictionary with the following keys:

        * rf : SimpleNamespace
            Radio-frequency block pulse event.
        * gz : SimpleNamespace
            Slice (or slab) selection event,
            including (hopefully!) the rephasing lobe.

    Raises
    ------
    ValueError
        If invalid ``use`` parameter was passed. Must be one of ``"excitation"``, ``"refocusing"`` or ``"inversion"``.

    """
    spatial_ripples = np.asarray(spatial_ripples)
    spectral_ripples = np.asarray(spectral_ripples)

    if system is None:
        system = Opts.default

    valid_pulse_uses = get_supported_rf_uses()
    if use != "" and use not in valid_pulse_uses:
        raise ValueError(
            f"Invalid use parameter. Must be one of {valid_pulse_uses}. Passed: {use}"
        )

    slice_thickness *= 1e-3
    flip_angle = np.deg2rad(flip_angle)
    gamma = system.gamma
    raster_time = system.grad_raster_time
    system.rf_raster_time = raster_time
    max_grad = convert(
        from_value=system.max_grad,
        from_unit="Hz/m",
        to_unit="mT/m",
        gamma=abs(gamma),
    )

    max_slew = convert(
        from_value=system.max_slew,
        from_unit="Hz/m/s",
        to_unit="T/m/s",
        gamma=abs(gamma),
    )

    if use != "" and use not in get_supported_rf_uses():
        raise ValueError(
            f"Invalid use parameter. Must be one of {get_supported_rf_uses()}. Passed: {use}"
        )
    use = USE_TABLE[use]
    ss_system = ss_globals()
    ss_system.SS_MXG = max_grad / 10.0  # G/cm
    ss_system.SS_MXS = max_slew / 10.0  # G/cm/ms
    ss_system.SS_TS = raster_time  # Sampling time (s)

    # set parameters for design
    if isinstance(spectral_band_edges, str) and spectral_band_edges == "water":
        df = 0.5e-6  # Conservative shim requirement
        water = 4.7e-6
        fat2 = 1.3e-6
        fat1 = 0.9e-6
        spectral_band_edges = (
            gamma
            * system.B0
            * (np.asarray((water - df, water + df, fat1 - df, fat2 + df)) - water)
        )
        if freq_offset is None:
            freq_offset = 0.5 * (
                spectral_band_edges[0] + spectral_band_edges[1]
            )  # center around water
    else:
        spectral_band_edges = np.asarray(spectral_band_edges)

    # adjust flip angle
    if np.isscalar(flip_angle):
        flip_angle = [flip_angle]
    flip_angle = list(flip_angle)
    n_bands = int(len(spectral_band_edges) // 2)
    while len(flip_angle) < n_bands:
        flip_angle += [0.0]
    flip_angle = np.asarray(flip_angle)

    # actual design
    grad, rf, _ = ss_design(
        slice_thickness * 1e2,
        spatial_time_bw_product,
        spatial_ripples,
        spectral_band_edges,
        flip_angle,
        spectral_ripples,
        use,
        spatial_filter_type,
        spectral_filter_type,
        "Flyback Half",
        freq_offset,
        ss_system,
        verbose,
    )

    if plot:
        freq_range = 2 * max(abs(spectral_band_edges))
        display_pulse(raster_time, rf * 1e2, grad * 10.0, slice_thickness, freq_range)

    # convert units for pulseq
    rf = gamma * rf * 1e-4  # (G -> T * gamma[Hz/T] = [Hz])
    grad = gamma * grad * 1e-2  # (G/cm -> T/m * gamma[Hz/T] = [Hz/m])

    # prepare structure
    rf = make_arbitrary_rf(rf, flip_angle=1.0, no_signal_scaling=True, system=system)
    grad = make_arbitrary_grad(channel="z", waveform=grad, system=system)

    # following 2 lines of code are workarounds for numpy returning 3.14... for np.angle(-0.00...)
    negative_zero_indices = np.where(rf.signal == -0.0)
    rf.signal[negative_zero_indices] = 0

    return {"rf": rf, "gz": grad}
