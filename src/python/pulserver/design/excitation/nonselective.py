"""Excitation, refocusing and inversion acting on everything in the transmit coil."""

from __future__ import annotations

__all__ = ["Inversion", "NonSelectiveExcitation", "NonSelectiveRefocusing"]

import numpy as np

from ... import pypulseq as pp
from ._base import RfModule, rf_reference

_AXES = ("x", "y", "z")


class NonSelectiveExcitation(RfModule):
    """A hard pulse: one rectangular envelope, no gradient.

    The simplest excitation there is, and the right one whenever the slab is
    the coil — 3D encoding, a whole-volume preparation, a calibration TR.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    flip_angle_deg : float, optional
        Nominal flip angle (degrees).
    duration_s : float, optional
        Pulse duration (s). Shorter is broader in frequency; the bandwidth is
        ``1 / duration_s``.
    use : str, optional
        What the pulse is for. The trajectory and label cores read this to
        find where a readout period begins, so leaving it ``"undefined"``
        makes the module unanalysable.
    freq_offset_hz, phase_offset_rad : float, optional
        Transmit offsets designed into the pulse. A scan loop moves
        ``module.rf.freq_offset`` and ``module.rf.phase_offset`` per shot
        instead of rebuilding the module.

    Attributes
    ----------
    rf : RfEvent
        The pulse.

    Examples
    --------
    >>> import pulserver.design as design
    >>> import pulserver.pypulseq as pp
    >>> excitation = design.NonSelectiveExcitation(pp.Opts(), flip_angle_deg=10.0)
    >>> round(excitation.center * 1e6)
    500

    With no gradient there is nothing to be selective in but frequency, and
    the profile is the transform of a rectangle:

    .. plot::
       :include-source:

       import pulserver.design as design
       import pulserver.pypulseq as pp

       design.NonSelectiveExcitation(pp.Opts(), 90.0, 0.5e-3).plot_rf(
           title="hard 90, 0.5 ms",
           plot_now=False,
       )
    """

    def init_module(
        self,
        system: pp.Opts,
        flip_angle_deg: float = 10.0,
        duration_s: float = 1e-3,
        *,
        use: str = "excitation",
        freq_offset_hz: float = 0.0,
        phase_offset_rad: float = 0.0,
    ) -> None:
        rf = pp.make_block_pulse(
            flip_angle=np.deg2rad(flip_angle_deg),
            duration=duration_s,
            freq_offset=freq_offset_hz,
            phase_offset=phase_offset_rad,
            use=use,
            system=system,
        )

        self.seq = pp.Sequence(system)
        self.seq.add_block(rf)

        self.center = rf_reference(rf)


class NonSelectiveRefocusing(RfModule):
    """A hard 180 between two crushers.

    The refocusing half of a spin echo, applied to everything the coil
    reaches. Hard rather than shaped because a non-selective refocusing pulse
    has no profile to design: what matters is that it is broadband enough to
    turn the whole spectrum, which is what a short rectangle is.

    The crushers are what make it usable. No 180 is perfect, and the part of
    the magnetisation it tips *into* the transverse plane rather than through
    it becomes an FID that would otherwise be read alongside the echo. A pair
    of identical lobes, one each side, leaves the refocused pathway untouched
    -- it accrues one lobe's phase before the pulse and unwinds it after --
    while everything the pulse newly excited sees only the second lobe and
    dephases. Being one event played twice, they are exactly identical, which
    is the condition that makes the cancellation exact.

    The pulse is phased a quarter turn from the excitation, which is the CPMG
    condition: a train of such pulses then refocuses its own flip-angle error
    on alternate echoes instead of accumulating it.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    flip_angle_deg : float, optional
        Nominal flip angle (degrees).
    duration_s : float, optional
        Pulse duration (s).
    spoiling_cycles : float, optional
        Cycles of dephasing each crusher winds across ``voxel_size_m``. Zero
        leaves the pulse bare, which is what an echo train that crushes
        elsewhere wants.
    voxel_size_m : float, optional
        Length the dephasing is counted over (m).
    axis : {'z', 'x', 'y'}, optional
        Crusher axis.
    phase_offset_rad : float, optional
        RF phase. The CPMG quarter turn by default.
    use : str, optional
        What the pulse is for; the trajectory core negates accumulated k at a
        refocusing pulse, so this is not cosmetic.

    Attributes
    ----------
    rf_ref : RfEvent
        The refocusing pulse.
    gz_spoil : GradEvent
        The crusher, published once because both sides play the same event.

    Raises
    ------
    ValueError
        If ``spoiling_cycles`` is negative, ``voxel_size_m`` is not positive,
        or ``axis`` is not a gradient channel.

    Examples
    --------
    >>> import pulserver.design as design
    >>> import pulserver.pypulseq as pp
    >>> refocusing = design.NonSelectiveRefocusing(pp.Opts())
    >>> len(refocusing.blocks), refocusing.blocks[0] == refocusing.blocks[2]
    (3, True)

    Without crushers it is the pulse alone:

    >>> len(design.NonSelectiveRefocusing(pp.Opts(), spoiling_cycles=0.0).blocks)
    1

    The pulse and both crushers together, integrated as one:

    .. plot::
       :include-source:

       import pulserver.design as design
       import pulserver.pypulseq as pp

       design.NonSelectiveRefocusing(pp.Opts(), duration_s=0.5e-3).plot_rf(
           title="hard 180 between crushers",
           whole=True,
           plot_now=False,
       )
    """

    def init_module(
        self,
        system: pp.Opts,
        flip_angle_deg: float = 180.0,
        duration_s: float = 1e-3,
        *,
        spoiling_cycles: float = 4.0,
        voxel_size_m: float = 1e-3,
        axis: str = "z",
        phase_offset_rad: float = np.pi / 2,
        use: str = "refocusing",
    ) -> None:
        if spoiling_cycles < 0:
            raise ValueError("spoiling_cycles must be >= 0")
        if voxel_size_m <= 0:
            raise ValueError("voxel_size_m must be positive")
        if axis not in _AXES:
            raise ValueError(f"axis must be one of {_AXES}, got {axis!r}")

        rf_ref = pp.make_block_pulse(
            flip_angle=np.deg2rad(flip_angle_deg),
            duration=duration_s,
            phase_offset=phase_offset_rad,
            use=use,
            system=system,
        )

        self.seq = pp.Sequence(system)

        if spoiling_cycles:
            gz_spoil, _, _ = pp.make_crusher(
                spoiling_cycles, voxel_size_m, axis, system=system
            )
            self.seq.add_block(gz_spoil)

        lead = self.seq.duration()[0]
        self.seq.add_block(rf_ref)
        if spoiling_cycles:
            self.seq.add_block(gz_spoil)

        self.center = lead + rf_reference(rf_ref)


class Inversion(RfModule):
    """An adiabatic inversion pulse, alone.

    Adiabatic because an inversion is worth doing right: above a threshold B1
    the flip stops depending on B1 at all, so the inversion holds across a
    transmit field that a hard pulse would leave partially inverted.

    This is the pulse on its own — no crusher, no inversion time. Those belong
    to the preparation the pulse is used in; see
    :class:`~pulserver.design.InversionPreparation`.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    duration_s : float, optional
        Pulse duration (s). Adiabaticity is a condition on sweeping slowly
        enough, so this is not free to shorten.
    pulse_type : str, optional
        Sweep family, as :func:`pypulseq.make_adiabatic_pulse` names them
        (``"hypsec"``, ``"wurst"``).
    bandwidth_hz : float, optional
        Frequency width of the sweep (Hz).
    adiabaticity : int, optional
        Sweep-rate margin over the adiabatic condition.
    use : str, optional
        What the pulse is for.

    Attributes
    ----------
    rf_prep : RfEvent
        The pulse.

    Examples
    --------
    >>> import pulserver.design as design
    >>> import pulserver.pypulseq as pp
    >>> inversion = design.Inversion(pp.Opts(), duration_s=8e-3)
    >>> round(inversion.duration * 1e3, 1)
    8.0

    The sweep, and the band it inverts. The plateau is flat because an
    adiabatic pulse inverts everything above its threshold equally:

    .. plot::
       :include-source:

       import pulserver.design as design
       import pulserver.pypulseq as pp

       design.Inversion(pp.Opts(), 8e-3).plot_rf(
           title="hyperbolic secant inversion, 8 ms",
           plot_now=False,
       )
    """

    def init_module(
        self,
        system: pp.Opts,
        duration_s: float = 10e-3,
        *,
        pulse_type: str = "hypsec",
        bandwidth_hz: float = 40e3,
        adiabaticity: int = 4,
        use: str = "inversion",
    ) -> None:
        rf_prep = pp.make_adiabatic_pulse(
            pulse_type=pulse_type,
            duration=duration_s,
            bandwidth=bandwidth_hz,
            adiabaticity=adiabaticity,
            use=use,
            system=system,
        )

        self.seq = pp.Sequence(system)
        self.seq.add_block(rf_prep)

        self.center = rf_reference(rf_prep)
