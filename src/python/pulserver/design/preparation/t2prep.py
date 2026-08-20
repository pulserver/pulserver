"""T2 and joint T1/T2 preparation: weight the magnetization, then store it."""

from __future__ import annotations

__all__ = ["T1T2Preparation", "T2Preparation"]

import numpy as np

from ... import pypulseq as pp
from ..excitation._base import RfModule, rf_reference
from ._common import half_passages, spoiler_gradients

_FINAL_TIPS = ("up", "down")


class T2Preparation(RfModule):
    """A 90 - 180 - 90 sandwich that leaves T2 weighting on the z axis.

    Magnetization is tipped into the transverse plane, left to decay for the
    preparation echo time, and put back on ``z``. What the readout then sees is
    already T2-weighted, whatever its own echo time is -- which is the point:
    the contrast is decoupled from the acquisition, so a bSSFP or a short-TE
    gradient echo can carry T2 contrast it could never generate itself.

    Every pulse is adiabatic and non-selective. The 90s are half passages, run
    forwards to tip down and time-reversed to store; see
    :func:`~pulserver.design.preparation._common.half_passages` for why the
    mirror matters. The 180s in the middle are full passages.

    **Refocusing pulses come in pairs, and the count must be even.** An
    adiabatic full passage inverts z cleanly but leaves transverse
    magnetization with a phase that depends on transmit amplitude, so it is
    not a refocusing pulse on its own; a second one undoes what the first did.
    Simulated over 0.7 to 1.3 of nominal B1, a pair restores 0.80 to 1.00 of
    the magnetization while a single pulse ranges from -0.69 to +0.98. An odd
    count is refused rather than corrected: the leftover phase is not half a
    turn -- it measures about 202 degrees for these pulses -- so there is no
    phase to hard-code that would stay right for another duration or sweep.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    echo_time_s : float
        Preparation echo time (s), first pulse centre to last.
    final_tip : {'up', 'down'}, optional
        Store on ``+z`` (pure T2 weighting) or ``-z``, which starts a T1
        recovery as well -- see :class:`T1T2Preparation`.
    n_refocus : int, optional
        Adiabatic refocusing pulses, spread evenly through the echo time. Must
        be even; see above.
    half_passage_duration_s : float, optional
        Duration of each 90 (s).
    refocusing_duration_s : float, optional
        Duration of each 180 (s).
    adiabaticity : int, optional
        Sweep-rate margin over the adiabatic condition, for every pulse.
    dwell_s : float, optional
        RF raster (s).
    spoiling_cycles : float, optional
        Cycles of dephasing the closing spoiler winds across ``voxel_size_m``.
        Zero leaves the module unspoiled, which is only right if something
        downstream spoils.
    voxel_size_m : float, optional
        Length the dephasing is counted over (m).
    labels : sequence of str, optional
        Counters emitted on the first pulse's block. A preparation is where a
        shot begins, so it is the natural place to say which shot this is.

    Attributes
    ----------
    rf_prep : RfEvent
        The half passage that tips down.
    rf_ref : RfEvent
        The refocusing pulse, published once because every repeat plays the
        same event.
    rf_store : RfEvent
        The reverse half passage that stores what is left.
    wait_te : DelayEvent
        The gap around each refocusing pulse. Absent when the pulses already
        fill the echo time.
    gx_spoil, gy_spoil, gz_spoil : GradEvent
        The closing spoiler.
    prep_labels : LabelSetEvent or list of LabelSetEvent
        One per name in ``labels``.
    echo_time : float
        The preparation echo time achieved (s).

    Raises
    ------
    ValueError
        If ``final_tip`` is not a pole, ``n_refocus`` is not a positive even
        number, a size is out of range, or the echo time is shorter than the
        pulses themselves.

    Examples
    --------
    >>> import pulserver.design as design
    >>> import pulserver.pypulseq as pp
    >>> prep = design.T2Preparation(pp.Opts(), 50e-3)
    >>> round(prep.echo_time * 1e3, 2)
    49.99

    Sweep the preparation echo time to fit T2::

        for te in (0.0, 30e-3, 60e-3):
            prep = design.T2Preparation(system, te)
            for block in prep.blocks:
                seq.add_block(*block)

    Tipped down, refocused twice, tipped back: over the band the refocusing
    covers, the whole module returns the magnetisation to z, and what it lost
    on the way is T2 weighting:

    .. plot::

       import pulserver.design as design
       import pulserver.pypulseq as pp
       from _figures import rf_profile

       rf_profile(
           design.T2Preparation(pp.Opts(), 40e-3),
           title="T2 preparation, TE 40 ms",
           whole=True,
           extent=600,
       )
    """

    def init_module(
        self,
        system: pp.Opts,
        echo_time_s: float,
        *,
        final_tip: str = "up",
        n_refocus: int = 2,
        half_passage_duration_s: float = 4e-3,
        refocusing_duration_s: float = 10.24e-3,
        adiabaticity: int = 8,
        dwell_s: float = 10e-6,
        spoiling_cycles: float = 4.0,
        voxel_size_m: float = 1e-3,
        labels: tuple[str, ...] | None = None,
    ) -> None:
        if final_tip not in _FINAL_TIPS:
            raise ValueError(
                f"final_tip must be one of {_FINAL_TIPS}, got {final_tip!r}"
            )
        n_refocus = int(n_refocus)
        if n_refocus < 2 or n_refocus % 2:
            raise ValueError(
                "n_refocus must be a positive even number: an adiabatic full passage leaves "
                "a transmit-dependent phase that only a second one undoes"
            )
        if echo_time_s <= 0:
            raise ValueError("echo_time_s must be positive")
        if spoiling_cycles < 0:
            raise ValueError("spoiling_cycles must be >= 0")
        if voxel_size_m <= 0:
            raise ValueError("voxel_size_m must be positive")

        rf_prep, rf_store = half_passages(
            system, half_passage_duration_s, adiabaticity=adiabaticity, dwell_s=dwell_s
        )
        rf_ref = pp.make_adiabatic_pulse(
            pulse_type="hypsec",
            duration=refocusing_duration_s,
            dwell=dwell_s,
            adiabaticity=adiabaticity,
            system=system,
            use="refocusing",
        )

        # With the refocusing pulses paired, the transverse phase arrives back
        # where the tip-down left it, so the storing pulse needs no phase of its
        # own -- except the half turn that stores on -z instead of +z.
        if final_tip == "down":
            rf_store.phase_offset += np.pi

        # The echo time runs centre to centre, and the refocusing pulses sit at
        # the centre of each of the n equal intervals it is cut into.
        interval = echo_time_s / n_refocus
        lead = pp.calc_duration(rf_prep) - pp.calc_rf_center(rf_prep)[0]
        centre_ref = pp.calc_rf_center(rf_ref)[0]
        span_ref = pp.calc_duration(rf_ref)
        trail = pp.calc_rf_center(rf_store)[0]

        first_gap = pp.round_to_raster(
            0.5 * interval - lead - centre_ref, system.block_duration_raster
        )
        inner_gap = pp.round_to_raster(
            interval - span_ref, system.block_duration_raster
        )
        last_gap = pp.round_to_raster(
            0.5 * interval - (span_ref - centre_ref) - trail,
            system.block_duration_raster,
        )
        if min(first_gap, inner_gap, last_gap) < -1e-12:
            shortest = lead + n_refocus * span_ref + trail
            raise ValueError(
                f"the requested preparation echo time of {echo_time_s * 1e3:.3f} ms is "
                f"shorter than the {shortest * 1e3:.3f} ms its pulses already occupy"
            )

        gx_spoil, gy_spoil, gz_spoil = (
            spoiler_gradients(system, spoiling_cycles, voxel_size_m)
            if spoiling_cycles
            else (None, None, None)
        )
        prep_labels = [
            pp.make_label(type="SET", label=name, value=0) for name in labels or ()
        ]

        self.seq = pp.Sequence(system)
        self.seq.add_block(rf_prep, *prep_labels)
        for index in range(n_refocus):
            gap = first_gap if index == 0 else inner_gap
            if gap:
                wait_te = pp.make_delay(gap)
                self.seq.add_block(wait_te)
            self.seq.add_block(rf_ref)
        if last_gap:
            wait_te = pp.make_delay(last_gap)
            self.seq.add_block(wait_te)

        # Read off the sequence rather than re-summed from the parts: the
        # blocks have been rounded onto the block raster by now, and the echo
        # time is where the pulses ended up, not where they were asked for.
        store_start = self.seq.duration()[0]
        self.seq.add_block(rf_store)
        if spoiling_cycles:
            self.seq.add_block(gx_spoil, gy_spoil, gz_spoil)

        self.center = rf_reference(rf_prep)
        self.echo_time = store_start + rf_reference(rf_store) - self.center
        self.final_tip = final_tip


class T1T2Preparation(T2Preparation):
    """T2 preparation that stores on ``-z``, so T1 recovery starts too.

    The same sandwich as :class:`T2Preparation` with the storing pulse turned
    over. The magnetization that lands on ``-z`` carries the T2 weighting the
    echo time gave it *and* then recovers through the null, so one module
    produces joint T1/T2 contrast and no separate inversion is needed. The
    inversion time is the gap to the readout, which belongs to the loop.

    Takes everything :class:`T2Preparation` does except ``final_tip``, which is
    what this class fixes.

    Examples
    --------
    >>> import pulserver.design as design
    >>> import pulserver.pypulseq as pp
    >>> prep = design.T1T2Preparation(pp.Opts(), 50e-3)
    >>> prep.final_tip
    'down'

    The same module tipping down instead of up, so the readout that follows
    sees an inversion recovering as well as a T2 decay:

    .. plot::

       import pulserver.design as design
       import pulserver.pypulseq as pp
       from _figures import rf_profile

       rf_profile(
           design.T1T2Preparation(pp.Opts(), 40e-3),
           title="T1-T2 preparation, TE 40 ms",
           whole=True,
           extent=600,
       )
    """

    def init_module(self, system: pp.Opts, echo_time_s: float, **kwargs) -> None:
        if "final_tip" in kwargs:
            raise ValueError(
                "T1T2Preparation is the final_tip='down' case; use T2Preparation to choose"
            )
        super().init_module(system, echo_time_s, final_tip="down", **kwargs)
