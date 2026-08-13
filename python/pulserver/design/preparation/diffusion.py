"""Diffusion preparation: weight the magnetization, then store it on z."""

from __future__ import annotations

__all__ = ["DiffusionPreparation"]

import math

import numpy as np

from ... import pypulseq as pp
from ..excitation._base import RfModule, rf_reference
from ._common import spoiler_gradients


class DiffusionPreparation(RfModule):
    """A 90 - G - 180 - G - 90 sandwich that leaves diffusion weighting on z.

    What survives to be stored on ``z`` is attenuated by ``exp(-b D)``, so any
    readout can follow and the weighting need not come from the readout's own
    echo. Non-selective: a slice-selective diffusion experiment is a spin-echo
    readout whose imaging pulses carry the gradients, not a preparation.

    The 90s are hard rather than adiabatic half passages, which cannot survive
    the single refocusing pulse a diffusion pair needs -- see
    :class:`~pulserver.design.T2Preparation`, which insists on an even number
    for the same reason. The cost is B1 robustness.

    One canonical design along z at ``b_value``; the loop scales it with
    :meth:`scale_for` and turns it with a rotation event::

        for direction, b in plan:
            g = pp.scale_grad(prep.g_diff, prep.scale_for(b))
            rotation = pp.make_rotation(direction)

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    b_value : float
        Design b-value (s/mm^2). This is the largest the module can produce;
        everything else is reached by scaling down.
    gradient_duration_s : float, optional
        Duration of each diffusion lobe (s).
    gradient_separation_s : float, optional
        Onset-to-onset separation of the two lobes (s). Must exceed
        ``gradient_duration_s`` by enough to hold the refocusing pulse.
    pulse_duration_s : float, optional
        Duration of each 90 (s).
    refocusing_duration_s : float, optional
        Duration of the 180 (s).
    axis : {'z', 'x', 'y'}, optional
        Axis the canonical pair is designed on.
    spoiling_cycles : float, optional
        Cycles of dephasing the closing spoiler winds across ``voxel_size_m``.
    voxel_size_m : float, optional
        Length the dephasing is counted over (m).
    labels : sequence of str, optional
        Counters emitted on the first pulse's block.

    Attributes
    ----------
    rf_prep : RfEvent
        The 90 that tips down.
    rf_ref : RfEvent
        The 180 between the lobes.
    rf_store : RfEvent
        The 90 that stores what is left.
    g_diff : TrapEvent
        The diffusion lobe, published once because both sides play the same
        event. Scale it per b-value, rotate it per direction.
    wait_te : DelayEvent
        The gap either side of the refocusing pulse, when the separation
        leaves one.
    gx_spoil, gy_spoil, gz_spoil : GradEvent
        The closing spoiler.
    prep_labels : LabelSetEvent or list of LabelSetEvent
        One per name in ``labels``.
    b_value : float
        The design b-value achieved (s/mm^2).
    max_b_value : float
        What this timing could reach at full gradient amplitude (s/mm^2).

    Raises
    ------
    ValueError
        If a duration is out of order, the separation cannot hold the
        refocusing pulse, or the b-value exceeds what the system can reach --
        the message says what it can.

    Examples
    --------
    >>> import pulserver.design as design
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    >>> prep = design.DiffusionPreparation(system, 500.0)
    >>> round(prep.b_value)
    500
    >>> round(prep.scale_for(125.0), 3)
    0.5
    """

    def init_module(
        self,
        system: pp.Opts,
        b_value: float,
        *,
        gradient_duration_s: float = 20e-3,
        gradient_separation_s: float = 30e-3,
        pulse_duration_s: float = 0.5e-3,
        refocusing_duration_s: float = 1e-3,
        axis: str = "z",
        spoiling_cycles: float = 4.0,
        voxel_size_m: float = 1e-3,
        labels: tuple[str, ...] | None = None,
    ) -> None:
        if b_value <= 0:
            raise ValueError("b_value must be positive")
        if gradient_duration_s <= 0 or gradient_separation_s <= gradient_duration_s:
            raise ValueError(
                "require gradient_separation_s > gradient_duration_s > 0: the lobes have to "
                "be separated by more than their own length for anything to sit between them"
            )
        if spoiling_cycles < 0:
            raise ValueError("spoiling_cycles must be >= 0")
        if voxel_size_m <= 0:
            raise ValueError("voxel_size_m must be positive")

        rf_prep = pp.make_block_pulse(
            flip_angle=np.pi / 2.0,
            duration=pulse_duration_s,
            use="preparation",
            system=system,
        )
        rf_ref = pp.make_slr_pulse(
            np.pi,
            duration=refocusing_duration_s,
            slice_thickness=0.0,
            pulse_type="se",
            phase_offset=np.pi / 2.0,
            use="refocusing",
            system=system,
        )
        # Half a turn from the tip-down: the refocusing pulse is about the
        # perpendicular axis, which leaves the tipped magnetization where it is,
        # so undoing the first rotation is what carries it back to +z.
        rf_store = pp.make_block_pulse(
            flip_angle=np.pi / 2.0,
            duration=pulse_duration_s,
            phase_offset=np.pi,
            use="preparation",
            system=system,
        )

        g_full = pp.make_trapezoid(
            channel=axis,
            amplitude=system.max_grad,
            duration=gradient_duration_s,
            system=system,
        )
        remaining = (
            gradient_separation_s - pp.calc_duration(g_full) - pp.calc_duration(rf_ref)
        )
        if remaining < -1e-12:
            raise ValueError(
                f"a separation of {gradient_separation_s * 1e3:.3f} ms cannot hold a "
                f"{pp.calc_duration(g_full) * 1e3:.3f} ms lobe and a "
                f"{pp.calc_duration(rf_ref) * 1e3:.3f} ms refocusing pulse"
            )
        raster = system.block_duration_raster
        first_gap = pp.round_to_raster(0.5 * remaining, raster)
        second_gap = pp.round_to_raster(remaining - first_gap, raster)

        max_b_value = _pair_b_value(
            g_full,
            first_gap + pp.calc_duration(rf_ref) + second_gap,
            system.grad_raster_time,
        )
        if b_value > max_b_value * (1.0 + 1e-9):
            raise ValueError(
                f"a b-value of {b_value:.1f} s/mm^2 is beyond this timing and gradient "
                f"strength, which reach {max_b_value:.1f} s/mm^2; lengthen the lobes or "
                f"their separation"
            )
        g_diff = pp.scale_grad(g_full, math.sqrt(b_value / max_b_value))

        gx_spoil, gy_spoil, gz_spoil = (
            spoiler_gradients(system, spoiling_cycles, voxel_size_m)
            if spoiling_cycles
            else (None, None, None)
        )
        prep_labels = [pp.make_label(type="SET", label=name, value=0) for name in labels or ()]

        self.seq = pp.Sequence(system)
        self.seq.add_block(rf_prep, *prep_labels)
        self.seq.add_block(g_diff)
        if first_gap:
            wait_te = pp.make_delay(first_gap)
            self.seq.add_block(wait_te)
        self.seq.add_block(rf_ref)
        if second_gap:
            wait_te = pp.make_delay(second_gap)
            self.seq.add_block(wait_te)
        self.seq.add_block(g_diff)
        self.seq.add_block(rf_store)
        if spoiling_cycles:
            self.seq.add_block(gx_spoil, gy_spoil, gz_spoil)

        self.center = rf_reference(rf_prep)
        self.b_value = float(b_value)
        self.max_b_value = float(max_b_value)

    def scale_for(self, b_value: float) -> float:
        """Amplitude factor for ``g_diff`` that gives ``b_value``.

        The b-value goes as the square of the gradient amplitude, so this is a
        square root rather than a ratio -- the thing worth having a method for.

        Parameters
        ----------
        b_value : float
            Wanted b-value (s/mm^2), between zero and the design value.

        Returns
        -------
        float
            Factor to pass to :func:`pypulseq.scale_grad`.

        Raises
        ------
        ValueError
            If ``b_value`` is negative or above the design value, which the
            module cannot reach by scaling down.
        """
        if not 0.0 <= b_value <= self.b_value:
            raise ValueError(
                f"b_value must lie between 0 and the design value of {self.b_value:.1f} "
                f"s/mm^2, got {b_value:.1f}"
            )
        return math.sqrt(b_value / self.b_value) if self.b_value else 0.0


def _pair_b_value(lobe, middle_duration: float, raster: float) -> float:
    """b-value of two identical lobes with a refocusing pulse between them.

    Integrated from the rasterized trapezoid, and with the second lobe's sign
    reversed: in the frame the refocusing pulse leaves behind, the two lobes
    dephase in the same direction rather than cancelling.
    """
    count = round(pp.calc_duration(lobe) / raster)
    times = (np.arange(count) + 0.5) * raster
    knots = np.cumsum([0.0, lobe.rise_time, lobe.flat_time, lobe.fall_time])
    samples = np.interp(times, knots, [0.0, lobe.amplitude, lobe.amplitude, 0.0])

    gap = np.zeros(round(middle_duration / raster))
    gradient = np.concatenate((samples, gap, -samples))
    # Amplitudes are Hz/m, so 2 pi turns the integral into rad/m; the result is
    # s/m^2 and s/mm^2 is a million times smaller.
    q_rad_per_m = 2.0 * np.pi * np.cumsum(gradient) * raster
    return float(np.sum(q_rad_per_m**2) * raster / 1e6)
