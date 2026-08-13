"""One pulse modulated into several spectral bands, with a gradient or without."""

from __future__ import annotations

__all__ = ["MultibandExcitation", "SmsExcitation"]

import numpy as np

from ... import pypulseq as pp
from ._base import RfModule, rf_reference

_AXES = ("x", "y", "z")


class SmsExcitation(RfModule):
    """A slice-selective pulse modulated to excite several slices at once.

    A gradient turns frequency into position, so multiplying one designed pulse
    by a sum of complex exponentials excites a comb of slices for the price of
    one. What it costs is peak B1: the bands add in phase at the centre of the
    pulse, so a ``n``-band pulse peaks at ``n`` times the single-band one unless
    the phases are spread. ``phases='quadratic'`` does that, and is what makes a
    high band count feasible at all.

    The band spacing is stated as a **slice gap** and converted through the
    selection gradient the module built, so the answer follows the pulse rather
    than having to be recomputed whenever the thickness or duration changes.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    flip_angle_deg : float
        Nominal flip angle (degrees), per band.
    thickness_m : float
        Slice thickness (m).
    slice_gap_m : float
        Centre-to-centre spacing between excited slices (m).
    n_bands : int, optional
        Slices excited at once.
    duration_s : float, optional
        Pulse duration (s).
    phases : {'quadratic'} or sequence of float, optional
        Per-band phase (rad). ``'quadratic'`` spreads the peak; ``None`` leaves
        every band in phase, which is the worst case for peak B1.
    rephase : bool, optional
        Build a rephaser at all.
    time_bw_product : float, optional
        Time-bandwidth product of the underlying slice profile.
    axis : {'z', 'x', 'y'}, optional
        Selection axis.
    use : str, optional
        What the pulse is for; the trajectory core reads it.

    Attributes
    ----------
    rf : RfEvent
        The modulated pulse.
    gz : TrapEvent
        The selection gradient.
    gz_reph : TrapEvent
        Its rephaser, when ``rephase``.
    band_offsets_hz : numpy.ndarray
        Where each band sits (Hz), zero among them.
    band_positions_m : numpy.ndarray
        The same, as slice positions (m).
    peak_ratio : float
        Peak B1 relative to the single-band pulse the bands were made from.

    Raises
    ------
    ValueError
        If a size is not positive, ``n_bands`` is below one, or ``axis`` is not
        a gradient channel.

    Examples
    --------
    >>> import pulserver.design as design
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=180, slew_unit="T/m/s")
    >>> sms = design.SmsExcitation(
    ...     system, 60.0, thickness_m=3e-3, slice_gap_m=24e-3, n_bands=3
    ... )
    >>> sms.band_positions_m.round(6).tolist()
    [-0.024, 0.0, 0.024]

    Spreading the band phases is what keeps the peak down:

    >>> stacked = design.SmsExcitation(
    ...     system, 60.0, thickness_m=3e-3, slice_gap_m=24e-3, n_bands=3, phases=None
    ... )
    >>> round(stacked.peak_ratio, 1), bool(sms.peak_ratio < stacked.peak_ratio)
    (3.0, True)
    """

    def init_module(
        self,
        system: pp.Opts,
        flip_angle_deg: float,
        *,
        thickness_m: float,
        slice_gap_m: float,
        n_bands: int = 2,
        duration_s: float = 3e-3,
        phases: object = "quadratic",
        rephase: bool = True,
        time_bw_product: float = 4.0,
        axis: str = "z",
        use: str = "excitation",
    ) -> None:
        if thickness_m <= 0 or duration_s <= 0:
            raise ValueError("thickness_m and duration_s must be positive")
        if slice_gap_m <= 0:
            raise ValueError("slice_gap_m must be positive")
        if int(n_bands) < 1:
            raise ValueError("n_bands must be >= 1")
        if axis not in _AXES:
            raise ValueError(f"axis must be one of {_AXES}, got {axis!r}")

        single, gz, gz_reph = pp.make_slr_pulse(
            np.deg2rad(flip_angle_deg),
            duration=duration_s,
            slice_thickness=thickness_m,
            time_bw_product=time_bw_product,
            return_gz=True,
            use=use,
            system=system,
        )
        gz.channel = axis
        gz_reph.channel = axis

        # A slice sits where its frequency matches the gradient, so the band
        # spacing the modulation needs is the gap read through the lobe.
        band_offset_hz = float(gz.amplitude) * slice_gap_m
        rf, band_offsets_hz, _ = pp.make_sms_pulse(
            single, int(n_bands), band_offset_hz, phases=phases
        )

        self.seq = pp.Sequence(system)
        self.seq.add_block(rf, gz)
        if rephase:
            self.seq.add_block(gz_reph)

        self.register(band_offsets_hz=band_offsets_hz)
        self.center = rf_reference(rf)
        self.band_positions_m = band_offsets_hz / float(gz.amplitude)
        self.peak_ratio = _peak_ratio(rf, single)
        self.n_bands = int(n_bands)


class MultibandExcitation(RfModule):
    """A pulse with off-resonance sidebands and no gradient at all.

    The same modulation :class:`SmsExcitation` uses, with nothing to turn
    frequency into position -- so the sidebands saturate wherever their offset
    reaches rather than exciting a slice. That is what an inhomogeneous
    magnetisation transfer experiment wants: the on-resonance band tips the
    free pool by ``flip_angle_deg`` while the sidebands deposit power into the
    bound pool, and comparing one sideband against two symmetric ones at the
    **same total power** is the ihMT measurement.

    Power, not amplitude, is what has to match. Two sidebands each at
    ``sqrt(beta/2)`` of the centre band's amplitude carry the same total
    ``|B1|^2`` as one at ``sqrt(beta)``, and halving an amplitude instead --
    which looks natural -- delivers half the power and turns the ihMT
    difference into a power difference.

    Give ``b1rms_ut`` and ``tr`` to have the sideband power solved for a total
    root-mean-square B1 over the repetition, which is how the experiment is
    normally specified; give ``sideband_power`` to state it directly.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    flip_angle_deg : float
        On-resonance flip angle (degrees).
    duration_s : float
        Pulse duration (s).
    band_offset_hz : float
        Sideband offset from the on-resonance band (Hz).
    n_bands : int, optional
        Bands counting the on-resonance one: two for a single sideband, three
        for a symmetric pair.
    sideband_power : float, optional
        Power of *each* sideband relative to the on-resonance band. Ignored
        when ``b1rms_ut`` is given.
    b1rms_ut : float, optional
        Target root-mean-square B1 over the repetition (uT), from which the
        sideband power is solved. Needs ``tr``.
    tr : float, optional
        The repetition the RMS is taken over (s).
    time_bw_product : float, optional
        Time-bandwidth product of the underlying envelope.
    use : str, optional
        What the pulse is for; the trajectory core reads it.

    Attributes
    ----------
    rf : RfEvent
        The modulated pulse.
    band_offsets_hz : numpy.ndarray
        Where each band sits (Hz), zero among them.
    band_power_fraction : numpy.ndarray
        Share of the total ``|B1|^2`` each band carries.
    sideband_power : float
        Power of each sideband relative to the on-resonance band, whether it
        was given or solved.
    b1rms_ut : float
        Root-mean-square B1 achieved over ``tr``, when one was given.
    peak_ut : float
        Peak B1 (uT), which is what a transmit chain limits.

    Raises
    ------
    ValueError
        If a size is out of range, ``b1rms_ut`` is given without ``tr``, or the
        on-resonance band alone already exceeds the requested ``b1rms_ut``.

    Examples
    --------
    >>> import pulserver.design as design
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts()
    >>> dual = design.MultibandExcitation(
    ...     system, 7.0, duration_s=2e-3, band_offset_hz=7000.0, n_bands=3,
    ...     b1rms_ut=2.0, tr=30e-3,
    ... )
    >>> dual.band_offsets_hz.tolist()
    [-7000.0, 0.0, 7000.0]
    >>> round(dual.b1rms_ut, 6)
    2.0

    Two sidebands or one, the total power is the same -- which is what makes
    the difference between them an ihMT measurement rather than a power one:

    >>> single = design.MultibandExcitation(
    ...     system, 7.0, duration_s=2e-3, band_offset_hz=7000.0, n_bands=2,
    ...     b1rms_ut=2.0, tr=30e-3,
    ... )
    >>> round(single.b1rms_ut, 6)
    2.0

    One sideband therefore carries exactly what two carry between them:

    >>> single.sideband_power == 2 * dual.sideband_power
    True
    """

    def init_module(
        self,
        system: pp.Opts,
        flip_angle_deg: float,
        *,
        duration_s: float,
        band_offset_hz: float,
        n_bands: int = 3,
        sideband_power: float = 1.0,
        b1rms_ut: float | None = None,
        tr: float | None = None,
        time_bw_product: float = 4.0,
        use: str = "saturation",
    ) -> None:
        n_bands = int(n_bands)
        if duration_s <= 0:
            raise ValueError("duration_s must be positive")
        if n_bands < 2:
            raise ValueError("n_bands must be >= 2: with one band there is no sideband")
        if band_offset_hz <= 0:
            raise ValueError("band_offset_hz must be positive")
        if sideband_power < 0:
            raise ValueError("sideband_power must be >= 0")

        single = pp.make_gauss_pulse(
            np.deg2rad(flip_angle_deg),
            duration=duration_s,
            time_bw_product=time_bw_product,
            use=use,
            system=system,
        )
        centre_squared = _b1_squared_integral(single, system)

        if b1rms_ut is not None:
            if tr is None:
                raise ValueError("b1rms_ut is a root-mean-square over a repetition; give tr too")
            if tr <= 0:
                raise ValueError("tr must be positive")
            # The sidebands make up whatever power the on-resonance band leaves
            # short of the target, shared equally between them.
            target_squared = (float(b1rms_ut) * 1e-6) ** 2 * float(tr)
            if target_squared < centre_squared:
                raise ValueError(
                    f"the on-resonance band alone is {np.sqrt(centre_squared / tr) * 1e6:.3f} uT "
                    f"rms over this TR, which already exceeds the {b1rms_ut:.3f} uT asked for; "
                    f"lower flip_angle_deg, lengthen duration_s, or raise b1rms_ut"
                )
            beta = (target_squared - centre_squared) / centre_squared
            sideband_power = beta / (n_bands - 1)

        rf, band_offsets_hz, weights = pp.make_sms_pulse(
            single, n_bands, band_offset_hz, sideband_power=sideband_power
        )

        self.seq = pp.Sequence(system)
        self.seq.add_block(rf)

        powers = np.abs(weights) ** 2
        self.register(band_offsets_hz=band_offsets_hz, band_power_fraction=powers / powers.sum())
        self.center = rf_reference(rf)
        self.sideband_power = float(sideband_power)
        self.n_bands = n_bands
        self.peak_ut = float(np.max(np.abs(np.asarray(rf.signal)))) / system.gamma * 1e6
        if tr:
            self.b1rms_ut = float(np.sqrt(_b1_squared_integral(rf, system) / tr)) * 1e6


def _b1_squared_integral(rf, system: pp.Opts) -> float:
    """``int |B1|^2 dt`` over a pulse, in tesla squared seconds."""
    signal = np.asarray(rf.signal, dtype=complex) / system.gamma
    time = np.asarray(rf.t, dtype=float)
    dwell = float(np.median(np.diff(time))) if len(time) > 1 else float(rf.shape_dur)
    return float(np.sum(np.abs(signal) ** 2) * dwell)


def _peak_ratio(banded, single) -> float:
    """Peak amplitude of a modulated pulse against the one it was made from."""
    return float(
        np.max(np.abs(np.asarray(banded.signal))) / np.max(np.abs(np.asarray(single.signal)))
    )
