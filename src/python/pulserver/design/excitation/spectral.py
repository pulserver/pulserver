"""Excitation selective in frequency: a spectral band, or a band and a slice."""

from __future__ import annotations

__all__ = ["FrequencySelectiveExcitation", "SpspExcitation"]

import numpy as np

from ... import pypulseq as pp
from ._base import RfModule, rf_reference

_AXES = ("x", "y", "z")


class FrequencySelectiveExcitation(RfModule):
    """A pulse that tips one spectral band and leaves the rest alone.

    No gradient, so nothing is spatially selective: the envelope alone sets the
    profile, and the duration follows from the passband --
    ``time_bw_product / bandwidth_hz``. That is the whole design, and it is why
    a narrow band costs a long pulse.

    The offset is carried in **ppm** rather than hertz, so the interpreter
    resolves it against the field the scan actually runs at. Pass
    ``freq_offset_hz`` instead to pin it to a frequency.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    flip_angle_deg : float
        Nominal flip angle (degrees).
    bandwidth_hz : float
        Spectral passband (Hz).
    freq_offset_ppm : float, optional
        Centre of the band, relative to water (ppm). Water itself by default.
    freq_offset_hz : float, optional
        Centre of the band as a frequency (Hz), added to whatever
        ``freq_offset_ppm`` asks for.
    time_bw_product : float, optional
        Time-bandwidth product. Higher is a squarer band and a longer pulse.
    pulse_type : {'st', 'ex', 'se', 'inv', 'sat'}, optional
        SLR design family.
    use : str, optional
        What the pulse is for; the trajectory core reads it.
    passband_ripple, stopband_ripple : float, optional
        Ripple allowed in each band of the spectral profile.

    Attributes
    ----------
    rf : RfEvent
        The pulse.
    duration_s : float
        Its length (s), which the passband fixed.
    bandwidth_hz : float
        The passband asked for.

    Raises
    ------
    ValueError
        If ``bandwidth_hz`` or ``time_bw_product`` is not positive.

    Examples
    --------
    >>> import pulserver.design as design
    >>> import pulserver.pypulseq as pp
    >>> water = design.FrequencySelectiveExcitation(pp.Opts(), 90.0, bandwidth_hz=200.0)
    >>> len(water.blocks), round(water.duration_s * 1e3, 3)
    (1, 20.0)

    A narrower band is a longer pulse, and nothing else changes:

    >>> narrow = design.FrequencySelectiveExcitation(pp.Opts(), 90.0, bandwidth_hz=100.0)
    >>> narrow.duration_s == 2 * water.duration_s
    True

    The band it tips, and the rest of the spectrum it leaves where it was:

    .. plot::

       import pulserver.design as design
       import pulserver.pypulseq as pp
       from _figures import rf_profile

       rf_profile(
           design.FrequencySelectiveExcitation(pp.Opts(), 90.0, bandwidth_hz=200.0),
           title="water excitation, 200 Hz passband",
       )
    """

    def init_module(
        self,
        system: pp.Opts,
        flip_angle_deg: float,
        *,
        bandwidth_hz: float,
        freq_offset_ppm: float = 0.0,
        freq_offset_hz: float = 0.0,
        time_bw_product: float = 4.0,
        pulse_type: str = "st",
        use: str = "excitation",
        passband_ripple: float = 0.01,
        stopband_ripple: float = 0.01,
    ) -> None:
        if bandwidth_hz <= 0:
            raise ValueError("bandwidth_hz must be positive")
        if time_bw_product <= 0:
            raise ValueError("time_bw_product must be positive")

        duration_s = pp.ceil_to_raster(
            time_bw_product / bandwidth_hz, system.rf_raster_time
        )
        rf = pp.make_slr_pulse(
            np.deg2rad(flip_angle_deg),
            duration=duration_s,
            time_bw_product=time_bw_product,
            pulse_type=pulse_type,
            passband_ripple=passband_ripple,
            stopband_ripple=stopband_ripple,
            freq_offset=freq_offset_hz,
            freq_ppm=freq_offset_ppm,
            use=use,
            system=system,
        )

        self.seq = pp.Sequence(system)
        self.seq.add_block(rf)

        self.center = rf_reference(rf)
        self.duration_s = duration_s
        self.bandwidth_hz = bandwidth_hz


class SpspExcitation(RfModule):
    """A spectral-spatial pulse: one slice and one spectral band at once.

    A train of short slice-selective subpulses rides an alternating selection
    gradient, and the subpulse envelope -- sampled at the subpulse repetition
    rate -- sets the spectral profile. Water-only excitation therefore costs no
    separate fat-saturation module and no extra TR time, which is what makes it
    worth the design.

    The two selectivities trade against each other through ``n_subpulses``:
    more subpulses widen the spectral profile's free range and lengthen the
    pulse.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    flip_angle_deg : float
        Nominal flip angle (degrees).
    thickness_m : float
        Slice thickness (m).
    spectral_bandwidth_hz : float
        Spectral passband (Hz). With ``spectral_time_bw_product`` it fixes the
        total duration.
    freq_offset_hz : float, optional
        Centre of the spectral band (Hz).
    rephase : bool, optional
        Build a rephaser at all.
    spatial_time_bw_product : float, optional
        Time-bandwidth product of each spatial subpulse.
    spectral_time_bw_product : float, optional
        Time-bandwidth product of the spectral envelope.
    n_subpulses : int, optional
        Subpulses in the train.
    axis : {'z', 'x', 'y'}, optional
        Selection axis.
    use : str, optional
        What the pulse is for; the trajectory core reads it.

    Attributes
    ----------
    rf : RfEvent
        The pulse.
    gz : GradEvent
        The alternating selection gradient.
    gz_reph : TrapEvent
        Its rephaser, when ``rephase``.

    Raises
    ------
    ValueError
        If a size is not positive, ``axis`` is not a gradient channel, or the
        requested selectivity exceeds a gradient limit.

    Examples
    --------
    >>> import pulserver.design as design
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=180, slew_unit="T/m/s")
    >>> water = design.SpspExcitation(
    ...     system, 30.0, thickness_m=10e-3, spectral_bandwidth_hz=300.0, n_subpulses=12
    ... )
    >>> len(water.blocks), water.gz.channel
    (2, 'z')

    The selection gradient alternates, one lobe per subpulse:

    >>> water.gz.type
    'grad'

    The subpulse train and its spectral response. The passband repeats at the
    subpulse rate, and where the repeats fall is what the subpulse count
    buys:

    .. plot::

       import pulserver.design as design
       import pulserver.pypulseq as pp
       from _figures import rf_profile

       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=180, slew_unit="T/m/s")
       rf_profile(
           design.SpspExcitation(
               system,
               30.0,
               thickness_m=10e-3,
               spectral_bandwidth_hz=300.0,
               n_subpulses=12,
           ),
           title="spectral-spatial, 10 mm and 300 Hz",
           spatial=False,
           extent=2500,
       )
    """

    def init_module(
        self,
        system: pp.Opts,
        flip_angle_deg: float,
        *,
        thickness_m: float,
        spectral_bandwidth_hz: float,
        freq_offset_hz: float = 0.0,
        rephase: bool = True,
        spatial_time_bw_product: float = 4.0,
        spectral_time_bw_product: float = 3.0,
        n_subpulses: int = 10,
        axis: str = "z",
        use: str = "excitation",
    ) -> None:
        if thickness_m <= 0:
            raise ValueError("thickness_m must be positive")
        if spectral_bandwidth_hz <= 0:
            raise ValueError("spectral_bandwidth_hz must be positive")
        if axis not in _AXES:
            raise ValueError(f"axis must be one of {_AXES}, got {axis!r}")

        rf, gz, gz_reph = pp.make_spsp_pulse(
            np.deg2rad(flip_angle_deg),
            slice_thickness=thickness_m,
            spectral_bandwidth=spectral_bandwidth_hz,
            freq_offset=freq_offset_hz,
            spatial_time_bandwidth_product=spatial_time_bw_product,
            spectral_time_bandwidth_product=spectral_time_bw_product,
            n_subpulses=n_subpulses,
            system=system,
            use=use,
        )
        gz.channel = axis
        gz_reph.channel = axis

        self.seq = pp.Sequence(system)
        self.seq.add_block(rf, gz)
        if rephase:
            self.seq.add_block(gz_reph)

        self.center = rf_reference(rf)
        self.spectral_bandwidth_hz = spectral_bandwidth_hz
        self.n_subpulses = n_subpulses
