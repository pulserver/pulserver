"""Blocks sub-package.

This sub-package contains PyPulseq-based sub-routines to generate RF pulses blocks.
"""

__all__ = []


import numpy as _np

from pypulseq import Opts as _Opts
from pypulseq.sigpy_pulse_opts import SigpyPulseOpts as _SigpyPulseOpts

from pypulseq import make_block_pulse as _make_block_pulse

from ._make_slr_pulse import pulpy_n_seq as _make_pulpy_pulse
from ._make_spsp_pulse import make_spsp_pulse

_USE_TABLE = {"excitation": "ex", "refocusing": "se", "inversion": "inv"}


def make_hard_pulse(
    system: _Opts,
    flip_angle: float,
    duration: float = 0.5e-3,
    bandwidth: float = None,
    time_bw_product: float = None,
    freq_offset: float = 0,
    phase_offset: float = 0,
    use: str = "excitation",
) -> dict:
    """
    Create a block (RECT or hard) pulse.

    Define duration, or bandwidth, or bandwidth and time_bw_product.
    If none are provided a default 4 ms pulse will be generated.

    Parameters
    ----------
    system : Opts
        System limits.
    flip_angle : float
        Flip angle in degrees.
    duration : float, optional
        Duration in seconds (s).
        The default is ``0.5e-3 s``.
    bandwidth : float, optional
        Bandwidth in Hertz (Hz).
        If supplied without time_bw_product ``duration = 1 / (4 * bandwidth)``
    time_bw_product : float, optional
        Time-bandwidth product.
        If supplied with bandwidth, ``duration = time_bw_product / bandwidth``
    freq_offset : float, optional
        Frequency offset in Hertz (Hz).
        The default is ``0.0``.
    phase_offset : float, optional
        Phase offset Hertz (Hz).
        The default is ``0.0``.
    use : str, optional
        Use of radio-frequency pulse. Must be one of ``"excitation"`` (default),
        ``"refocusing"`` or ``"inversion"``.

    Returns
    -------
    rf_block : dict
        Dictionary with the following keys:

        * rf : SimpleNamespace
            Radio-frequency block pulse event.

    Raises
    ------
    ValueError
        If invalid `use` parameter is passed.
        One of bandwidth or duration must be defined, but not both.
        One of bandwidth or duration must be defined and be > 0.

    """
    rf, _ = _make_block_pulse(
        _np.deg2rad(flip_angle),
        0.0,
        duration,
        bandwidth,
        time_bw_product,
        freq_offset,
        phase_offset,
        system,
        use,
    )

    return {"rf": rf}


def make_slr_pulse(
    system: _Opts,
    flip_angle: float,
    slice_thickness: float,
    duration: float = 4e-3,
    freq_offset: float = 0,
    n_bands: int = 1,
    slice_sep: float = 2,
    use: str = "excitation",
    time_bw_product: float = 4.0,
    ripples: tuple[float] = (0.01, 0.01),
    filter_type: str = "ls",
    center_pos: float = 0.5,
    cancel_alpha_phs: bool = False,
    phs_0_pt: str | None = None,
    plot: bool = False,
) -> tuple[dict, dict]:
    """
    Creates a radio-frequency SLR pulse event using the sigpy rf pulse library.
    Also design the accompanying slice/slab select and slice/slab rephasing
    trapezoidal gradient events.

    Parameters
    ----------
    system : Opts
        System limits.
    flip_angle : float
        Flip angle in degrees.
    slice_thickness : float
        Slice thickness in (mm) of accompanying slice select trapezoidal event.
        The slice thickness determines the area of the slice select event.
    duration : float, optional
        Duration in seconds (s).
        The default is ``4e-3``.
    freq_offset : float, optional
        Frequency offset in Hertz (Hz).
        The default is ``0``.
    n_bands : int, optional
        Number of bands. The default is ``1`` (no SMS pulse).
    slice_sep : float, optional
        Normalized slice separation. The default is ``2``.
        Ignored if ``n_bands=1``.
    use : str, optional
        Use of radio-frequency pulse. Must be one of ``"excitation"`` (default),
        ``"refocusing"`` or ``"inversion"``.
    time_bw_product : float, optional
        Time-bandwidth product.
        The default is ``4.'0``.
    ripples : Iterable[float]
        Passband and stopband ripples, in the form
        ``(pass_ripple, stop_ripple)``.
    filter_type : str, optional
        Type of filter for the spatial dimension:
        ``"ms"``, ``"ls"`` (default), ``"pm"``, ``"min"``, ``"max"``.
    center_pos : float, optional
        Relative position of rf peak.
        The default is ``0.5`` (midway).
    cancel_alpha_phs : bool, optional
        For ``‘excitation’`` pulses, absorb the alpha phase profile from beta’s profile,
        so they cancel for a flatter total phase.
    phs_0_pt : str, optional
        Set of phases to use for SMS pulse.
        Can be 'phs_mod' (Wong), ``"amp_mod"`` (Malik), ``"quad_mod"`` (Grissom),
        or ``None``. The default is ``None``.
    plot: bool, optional
        Show sigpy plot outputs.
        The default is ``False``.

    Returns
    -------
    rf_block : dict
        Dictionary with the following keys:

        * rf : SimpleNamespace
            Radio-frequency block pulse event.
        * gz : SimpleNamespace
            Slice (or slab) selection event.

    greph_block : dict
        Dictionary with the following keys:

        * gz : SimpleNamespace
            Slice (or slab) rephasing event.

    Raises
    ------
    ValueError
        If invalid `use` parameter was passed. Must be one of 'excitation', 'refocusing' or 'inversion'.
        If `return_gz=True` and `slice_thickness` was not provided.

    """
    _pulse_cfg = {}
    if n_bands == 1:
        _pulse_cfg["pulse_type"] = "slr"
    else:
        _pulse_cfg["pulse_type"] = "sms"
    _pulse_cfg["ptype"] = _USE_TABLE[use]
    _pulse_cfg["ftype"] = filter_type
    _pulse_cfg["d1"] = ripples[0]
    _pulse_cfg["d2"] = ripples[1]
    _pulse_cfg["cancel_alpha_phs"] = cancel_alpha_phs
    _pulse_cfg["n_bands"] = n_bands
    _pulse_cfg["band_sep"] = slice_sep * time_bw_product
    _pulse_cfg["phs_0_pt"] = phs_0_pt
    pulse_cfg = _SigpyPulseOpts(**_pulse_cfg)

    rf, gz, gzr, _ = _make_pulpy_pulse(
        _np.deg2rad(flip_angle),
        0,
        duration,
        freq_offset,
        center_pos,
        0,
        0,
        0,
        True,
        slice_thickness * 1e-3,
        system,
        time_bw_product,
        pulse_cfg,
        use,
        plot,
    )

    return {"rf": rf, "gz": gz}, {"gz": gzr}


__all__ = ["make_hard_pulse", "make_slr_pulse", "make_spsp_pulse"]
