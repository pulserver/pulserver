"""Off-resonance saturation: MT, inhomogeneous MT, and the Bloch-Siegert pulse.

Three modules that share one shape -- a strong pulse placed away from water,
played a few times, and (usually) spoiled. What separates them is where the
pulse sits in frequency and what envelope carries it, so that is all the
subclasses do.
"""

from __future__ import annotations

__all__ = [
    "BlochSiegertPreparation",
    "IhMtPreparation",
    "MtPreparation",
    "OffResonanceSaturation",
]

from typing import Any

import numpy as np

from ... import pypulseq as pp
from ..excitation._base import RfModule, rf_reference
from ._common import spoiler_gradients


class OffResonanceSaturation(RfModule):
    """A train of off-resonance pulses, optionally spoiled.

    The layout every off-resonance preparation has: the same pulse played
    ``n_pulses`` times back to back, then a spoiler if one was asked for. One
    event, replayed, so it publishes as one ``rf_prep`` however long the train
    is.

    Subclass this to add a family: design the pulse and hand it over. What
    makes each family is the envelope and the offset -- an SLR passband for
    magnetization transfer, a Fermi plateau for Bloch-Siegert -- and neither
    changes anything below.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    rf_prep : RfEvent
        The pulse to play.
    n_pulses : int, optional
        Times to play it.
    spoiling_cycles : float, optional
        Cycles of dephasing the closing spoiler winds across ``voxel_size_m``.
        Zero omits the spoiler, which is what a pulse whose effect is a phase
        rather than a saturation needs.
    voxel_size_m : float, optional
        Length the dephasing is counted over (m).
    labels : sequence of str, optional
        Counters emitted on the first pulse's block.

    Attributes
    ----------
    rf_prep : RfEvent
        The pulse, published once however many times it is played.
    gx_spoil, gy_spoil, gz_spoil : GradEvent
        The closing spoiler, when there is one.
    prep_labels : LabelSetEvent or list of LabelSetEvent
        One per name in ``labels``.

    Raises
    ------
    ValueError
        If ``n_pulses`` is below one, or a spoiling argument is out of range.

    Examples
    --------
    The pulse is the family's; the train and the spoiler are this class's.

    >>> import numpy as np
    >>> import pulserver.design as design
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts()
    >>> pulse = pp.make_gauss_pulse(
    ...     flip_angle=np.deg2rad(500), duration=8e-3, freq_offset=-1500.0,
    ...     use="saturation", system=system,
    ... )
    >>> prep = design.OffResonanceSaturation(system, pulse, n_pulses=3)
    >>> len(prep.blocks)
    4

    One event replayed, so three pulses cost one waveform:

    >>> prep.blocks[0] == prep.blocks[1] == prep.blocks[2]
    True

    Three pulses back to back, and what they leave along z:

    .. plot::

       import numpy as np
       import pulserver.design as design
       import pulserver.pypulseq as pp
       from _figures import rf_profile

       system = pp.Opts()
       pulse = pp.make_gauss_pulse(
           flip_angle=np.deg2rad(500), duration=8e-3, freq_offset=-1500.0,
           use="saturation", system=system,
       )
       rf_profile(
           design.OffResonanceSaturation(system, pulse, n_pulses=3),
           title="three saturation pulses, 1.5 kHz below resonance",
           whole=True,
           extent=3000,
       )
    """

    def init_module(
        self,
        system: pp.Opts,
        rf_prep: Any,
        *,
        n_pulses: int = 1,
        spoiling_cycles: float = 4.0,
        voxel_size_m: float = 1e-3,
        labels: tuple[str, ...] | None = None,
    ) -> None:
        n_pulses = int(n_pulses)
        if n_pulses < 1:
            raise ValueError("n_pulses must be >= 1")
        if spoiling_cycles < 0:
            raise ValueError("spoiling_cycles must be >= 0")
        if voxel_size_m <= 0:
            raise ValueError("voxel_size_m must be positive")

        prep_labels = [
            pp.make_label(type="SET", label=name, value=0) for name in labels or ()
        ]

        self.seq = pp.Sequence(system)
        for index in range(n_pulses):
            self.seq.add_block(rf_prep, *(prep_labels if index == 0 else ()))
        if spoiling_cycles:
            gx_spoil, gy_spoil, gz_spoil = spoiler_gradients(
                system, spoiling_cycles, voxel_size_m
            )
            self.seq.add_block(gx_spoil, gy_spoil, gz_spoil)

        self.center = rf_reference(rf_prep)


def saturation_pulse(
    system: pp.Opts,
    flip_angle_deg: float,
    duration_s: float,
    freq_offset_hz: float,
    time_bw_product: float,
) -> Any:
    """The SLR envelope both MT families saturate with.

    SLR rather than a Gaussian: what an MT experiment is trying to control is
    *where in frequency* the power lands, and an SLR design states the
    passband, the stopband and the ripple between them instead of leaving them
    to the tails of a window function. The pulse is non-selective in space --
    there is no gradient -- so its only profile is the spectral one.
    """
    return pp.make_slr_pulse(
        np.deg2rad(flip_angle_deg),
        duration=duration_s,
        slice_thickness=0.0,
        time_bw_product=time_bw_product,
        pulse_type="sat",
        freq_offset=freq_offset_hz,
        use="saturation",
        system=system,
    )


class MtPreparation(OffResonanceSaturation):
    """Off-resonance saturation of the bound pool, then a spoiler.

    A long, high-flip pulse placed a kilohertz or two from water saturates the
    broad macromolecular resonance without exciting free water directly. That
    saturation transfers to water over the following milliseconds, and the
    readout sees the reduced signal. The spoiler removes whatever transverse
    magnetization the pulse did produce.

    Compare an acquisition with this module against one without to get the MT
    ratio; the module is played or not played, rather than scaled, because
    scaling the envelope changes the saturation non-linearly.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    flip_angle_deg : float, optional
        Nominal flip angle (degrees). Far above 90: the point is deposited
        power, not a tip.
    freq_offset_hz : float, optional
        Offset from water (Hz); must not be zero.
    duration_s : float, optional
        Pulse duration (s).
    time_bw_product : float, optional
        Time-bandwidth product, which with ``duration_s`` sets how wide a band
        of the bound pool is saturated.
    n_pulses : int, optional
        Pulses in the train.
    spoiling_cycles, voxel_size_m, labels
        As :class:`OffResonanceSaturation`.

    Attributes
    ----------
    As :class:`OffResonanceSaturation`.

    Raises
    ------
    ValueError
        If ``freq_offset_hz`` is zero or a duration is not positive.

    Examples
    --------
    >>> import pulserver.design as design
    >>> import pulserver.pypulseq as pp
    >>> mt = design.MtPreparation(pp.Opts())
    >>> len(mt.blocks), round(float(mt.rf_prep.freq_offset))
    (2, -1500)

    One band, off resonance, and the free pool left alone at zero:

    .. plot::

       import pulserver.design as design
       import pulserver.pypulseq as pp
       from _figures import rf_profile

       rf_profile(
           design.MtPreparation(pp.Opts()),
           title="MT saturation, 1.5 kHz below resonance",
           whole=True,
           extent=3000,
       )
    """

    def init_module(
        self,
        system: pp.Opts,
        *,
        flip_angle_deg: float = 500.0,
        freq_offset_hz: float = -1500.0,
        duration_s: float = 8e-3,
        time_bw_product: float = 4.0,
        **kwargs: Any,
    ) -> None:
        if freq_offset_hz == 0:
            raise ValueError(
                "freq_offset_hz must not be zero: an on-resonance saturation excites water "
                "rather than the bound pool"
            )
        if duration_s <= 0 or time_bw_product <= 0:
            raise ValueError("duration_s and time_bw_product must be positive")

        rf_prep = saturation_pulse(
            system, flip_angle_deg, duration_s, freq_offset_hz, time_bw_product
        )
        super().init_module(system, rf_prep, **kwargs)


class IhMtPreparation(OffResonanceSaturation):
    """Saturation at both offsets at once, for the inhomogeneous MT difference.

    The bound pool has a component whose lineshape is inhomogeneously
    broadened, and saturating one side of water leaves it partly unsaturated
    while saturating both sides at once does not. Acquiring both and taking
    the difference isolates that component:
    ``ihMT = (MT+ + MT-) / 2 - MT_dual``, where the two single-offset arms are
    :class:`MtPreparation` at plus and minus ``freq_offset_hz`` and this module
    is the dual-offset one.

    Dual-offset means *simultaneously*, not alternately, so the pulse is built
    by modulating one envelope into two bands with
    :func:`~pulserver.pypulseq.make_sms_pulse` -- the same wrapper that turns
    one slice into several -- and shifting the pair so neither band lands on
    water.

    **The two arms are matched on power, not on amplitude.** Deposited power
    goes as the integral of the squared envelope, and two bands each at
    ``A / sqrt(2)`` integrate to the same as one band at ``A`` -- so that is
    the scaling used, and what the subtraction removes is the inhomogeneous
    saturation rather than a power difference. It costs peak amplitude: the
    two bands add in phase at the centre of the pulse, so the dual pulse peaks
    about 1.41 times higher than the single-offset one at the same setting.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    flip_angle_deg : float, optional
        Flip angle (degrees) of the single-offset arm this one is matched
        against. Each band gets ``1 / sqrt(2)`` of it.
    freq_offset_hz : float, optional
        Distance of each band from water (Hz), positive. Both signs are built.
    duration_s : float, optional
        Pulse duration (s).
    time_bw_product : float, optional
        Time-bandwidth product of each band.
    n_pulses : int, optional
        Pulses in the train.
    spoiling_cycles, voxel_size_m, labels
        As :class:`OffResonanceSaturation`.

    Attributes
    ----------
    rf_prep : RfEvent
        The dual-band pulse.
    band_offsets_hz : numpy.ndarray
        Where the two bands actually landed (Hz), for a spectral check.

    Raises
    ------
    ValueError
        If ``freq_offset_hz`` is not positive or a duration is not positive.

    Examples
    --------
    >>> import pulserver.design as design
    >>> import pulserver.pypulseq as pp
    >>> ihmt = design.IhMtPreparation(pp.Opts())
    >>> ihmt.band_offsets_hz.tolist()
    [-1500.0, 1500.0]

    Both bands at once, which is the measurement: the same total power as one
    band, delivered symmetrically:

    .. plot::

       import pulserver.design as design
       import pulserver.pypulseq as pp
       from _figures import rf_profile

       rf_profile(
           design.IhMtPreparation(pp.Opts()),
           title="ihMT saturation, both sidebands",
           whole=True,
           extent=3000,
       )
    """

    def init_module(
        self,
        system: pp.Opts,
        *,
        flip_angle_deg: float = 500.0,
        freq_offset_hz: float = 1500.0,
        duration_s: float = 8e-3,
        time_bw_product: float = 4.0,
        **kwargs: Any,
    ) -> None:
        if freq_offset_hz <= 0:
            raise ValueError(
                "freq_offset_hz must be positive; both signs are built from it"
            )
        if duration_s <= 0 or time_bw_product <= 0:
            raise ValueError("duration_s and time_bw_product must be positive")

        # Two bands and no on-resonance one: the modulation puts them at 0 and
        # +2f, and the pulse's own offset then slides the pair to -f and +f.
        # Each band carries 1/sqrt(2) of the single-offset amplitude, so the two
        # together deposit the same power -- see the class docstring.
        single = saturation_pulse(
            system, flip_angle_deg / np.sqrt(2.0), duration_s, 0.0, time_bw_product
        )
        rf_prep, offsets, _ = pp.make_sms_pulse(single, 2, 2.0 * freq_offset_hz)
        rf_prep.freq_offset = -freq_offset_hz

        super().init_module(system, rf_prep, **kwargs)

        self.band_offsets_hz = np.asarray(offsets, dtype=float) - freq_offset_hz


class BlochSiegertPreparation(OffResonanceSaturation):
    """An off-resonance pulse that writes a B1-squared phase shift, and nothing else.

    A strong pulse far from resonance barely tips anything, but it shifts the
    water resonance by an amount proportional to the square of the transmit
    amplitude. Acquiring one image with the pulse at ``+f`` and one at ``-f``
    turns that shift into a phase difference, and the phase difference into a
    B1 map. Nothing about the readout changes.

    Two things follow from what the pulse is for. There is **no spoiler** --
    the module runs between the excitation and the readout, and spoiling would
    destroy the very signal that is meant to carry the phase. And the envelope
    is a **Fermi** function: a flat top with smooth shoulders, which keeps the
    time-integrated squared amplitude high (the shift is proportional to it)
    while keeping the spectral tails away from water.

    The calibration constants a reconstruction needs come back on the module:
    ``kbs`` is the shift per unit squared Pulseq amplitude and
    ``kbs_per_gauss2`` the same in the conventional rad/G^2.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    freq_offset_hz : float, optional
        Offset from water (Hz); flip its sign for the second acquisition.
    duration_s : float, optional
        Pulse duration (s).
    peak_b1_hz : float, optional
        Plateau amplitude, in Pulseq units.
    flat_fraction : float, optional
        Fraction of the duration held at ``peak_b1_hz``, in ``(0, 1)``.
    transition_fraction : float, optional
        Width of each shoulder, as a fraction of the duration.
    dwell_s : float, optional
        RF raster (s).
    n_pulses, spoiling_cycles, voxel_size_m, labels
        As :class:`OffResonanceSaturation`. ``spoiling_cycles`` defaults to
        zero here, for the reason above.

    Attributes
    ----------
    rf_prep : RfEvent
        The Fermi pulse.
    kbs : float
        Phase shift per unit squared Pulseq amplitude (rad).
    kbs_per_gauss2 : float
        The same constant in rad/G^2.

    Raises
    ------
    ValueError
        If ``freq_offset_hz`` is zero, a duration or amplitude is not
        positive, or the envelope fractions are out of range.

    References
    ----------
    Sacolick et al., *B1 mapping by Bloch-Siegert shift*, Magn Reson Med
    2010;63:1315-1322, ``10.1002/mrm.22357``.

    Examples
    --------
    >>> import pulserver.design as design
    >>> import pulserver.pypulseq as pp
    >>> pulse = design.BlochSiegertPreparation(pp.Opts())
    >>> len(pulse.blocks), round(pulse.kbs_per_gauss2, 1)
    (1, 86.6)

    The pulse sits far enough off resonance to shift the phase without
    tipping much, and the residual dip is what "far enough" costs:

    .. plot::

       import pulserver.design as design
       import pulserver.pypulseq as pp
       from _figures import rf_profile

       rf_profile(
           design.BlochSiegertPreparation(pp.Opts()),
           title="Bloch-Siegert probe, 4 kHz off resonance",
           whole=True,
           extent=(-1000, 8000),
       )
    """

    def init_module(
        self,
        system: pp.Opts,
        *,
        freq_offset_hz: float = 4000.0,
        duration_s: float = 8e-3,
        peak_b1_hz: float = 500.0,
        flat_fraction: float = 0.8,
        transition_fraction: float = 0.02,
        dwell_s: float = 10e-6,
        spoiling_cycles: float = 0.0,
        **kwargs: Any,
    ) -> None:
        if freq_offset_hz == 0:
            raise ValueError(
                "freq_offset_hz must not be zero: the shift is inversely in it"
            )
        if duration_s <= 0 or peak_b1_hz <= 0 or dwell_s <= 0:
            raise ValueError("duration_s, peak_b1_hz and dwell_s must be positive")
        if not 0 < flat_fraction < 1 or transition_fraction <= 0:
            raise ValueError(
                "flat_fraction must lie in (0, 1) and transition_fraction must be positive"
            )

        n_samples = max(2, round(duration_s / dwell_s))
        duration_s = n_samples * dwell_s
        times = (np.arange(n_samples) + 0.5) * dwell_s
        half_flat = 0.5 * flat_fraction * duration_s
        transition = transition_fraction * duration_s
        envelope = peak_b1_hz / (
            1.0 + np.exp((np.abs(times - 0.5 * duration_s) - half_flat) / transition)
        )
        rf_prep = pp.make_arbitrary_rf(
            signal=envelope,
            flip_angle=1.0,
            no_signal_scaling=True,
            dwell=dwell_s,
            freq_offset=freq_offset_hz,
            system=system,
            use="preparation",
        )

        super().init_module(system, rf_prep, spoiling_cycles=spoiling_cycles, **kwargs)

        # The shift is (gamma B1)^2 / (2 * 2 pi * offset), integrated over the
        # pulse. Reported twice: once in the amplitude units the sequence uses,
        # once per gauss squared, which is how the literature quotes it.
        denominator = 2.0 * 2.0 * np.pi * freq_offset_hz
        gamma_hz_per_gauss = system.gamma * 1e-4
        self.kbs = float(np.sum((2.0 * np.pi * envelope) ** 2 / denominator) * dwell_s)
        self.kbs_per_gauss2 = float(
            np.sum(
                (2.0 * np.pi * gamma_hz_per_gauss * envelope / peak_b1_hz) ** 2
                / denominator
            )
            * dwell_s
        )
