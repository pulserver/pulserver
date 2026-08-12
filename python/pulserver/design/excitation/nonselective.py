"""Excitation and inversion that act on everything in the transmit coil."""

from __future__ import annotations

__all__ = ["Inversion", "NonSelectiveExcitation"]

import numpy as np

from ... import pypulseq as pp
from ._base import RfModule, rf_reference


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
    rf : RfEvent
        The pulse.

    Examples
    --------
    >>> import pulserver.design as design
    >>> import pulserver.pypulseq as pp
    >>> inversion = design.Inversion(pp.Opts(), duration_s=8e-3)
    >>> round(inversion.duration * 1e3, 1)
    8.0
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
        rf = pp.make_adiabatic_pulse(
            pulse_type=pulse_type,
            duration=duration_s,
            bandwidth=bandwidth_hz,
            adiabaticity=adiabaticity,
            use=use,
            system=system,
        )

        self.seq = pp.Sequence(system)
        self.seq.add_block(rf)

        self.center = rf_reference(rf)
