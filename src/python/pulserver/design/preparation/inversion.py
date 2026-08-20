"""Inversion-recovery preparation."""

from __future__ import annotations

__all__ = ["InversionPreparation"]

from ... import pypulseq as pp
from ..excitation._base import RfModule, rf_reference

_AXES = ("x", "y", "z")


class InversionPreparation(RfModule):
    """A non-selective adiabatic inversion followed by a crusher.

    Inversion recovery in the form a scan loop wants it: the pulse that
    inverts, and the gradient that destroys whatever transverse magnetisation
    the pulse left behind. The inversion time itself is **not** here — TI is
    the gap between this module and the readout that samples the recovery, so
    it belongs to the loop that plays both, which is also the only place that
    knows how many readouts share one inversion.

    The crusher is stated as cycles of dephasing across a voxel rather than as
    an area, so the same numbers mean the same thing at any resolution.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    duration_s : float, optional
        Inversion pulse duration (s).
    spoiling_cycles : float, optional
        Cycles of dephasing the crusher winds across ``voxel_size_m``.
    voxel_size_m : float, optional
        Length the dephasing is counted over (m) — the smallest voxel
        dimension, since that is the one that has to be spoiled.
    axis : {'z', 'x', 'y'}, optional
        Crusher axis.
    pulse_type : str, optional
        Adiabatic sweep family.
    bandwidth_hz : float, optional
        Frequency width of the sweep (Hz).
    labels : sequence of str, optional
        Counters emitted on the inversion block. An inversion is where a shot
        begins, so it is the natural place to say which shot this is; the loop
        writes the values, this only makes the slots.

    Attributes
    ----------
    rf_prep : RfEvent
        The inversion pulse.
    gz_spoil : GradEvent
        The crusher.
    prep_labels : LabelSetEvent or list of LabelSetEvent
        One per name in ``labels``, in order. Absent when ``labels`` is
        empty, and a bare event rather than a list when there is one.

    Raises
    ------
    ValueError
        If ``voxel_size_m`` is not positive, ``spoiling_cycles`` is negative,
        or ``axis`` is not a gradient channel.

    Examples
    --------
    >>> import pulserver.design as design
    >>> import pulserver.pypulseq as pp
    >>> prep = design.InversionPreparation(pp.Opts(), duration_s=8e-3)
    >>> len(prep.blocks)
    2

    Tag every inversion with the shot it opens::

        prep = design.InversionPreparation(system, labels=("REP",))
        for shot in range(num_shots):
            prep.prep_labels[0].value = shot
            for block in prep.blocks:
                seq.add_block(*block)

    The pulse and its spoiler together, which is what the TI is measured
    from:

    .. plot::

       import pulserver.design as design
       import pulserver.pypulseq as pp
       from _figures import rf_profile

       rf_profile(
           design.InversionPreparation(pp.Opts(), 8e-3),
           title="inversion preparation, 8 ms",
           whole=True,
           extent=2000,
       )
    """

    def init_module(
        self,
        system: pp.Opts,
        duration_s: float = 10e-3,
        *,
        spoiling_cycles: float = 4.0,
        voxel_size_m: float = 1e-3,
        axis: str = "z",
        pulse_type: str = "hypsec",
        bandwidth_hz: float = 40e3,
        labels: tuple[str, ...] | None = None,
    ) -> None:
        if voxel_size_m <= 0:
            raise ValueError("voxel_size_m must be positive")
        if spoiling_cycles < 0:
            raise ValueError("spoiling_cycles must be >= 0")
        if axis not in _AXES:
            raise ValueError(f"axis must be one of {_AXES}, got {axis!r}")

        rf_prep = pp.make_adiabatic_pulse(
            pulse_type=pulse_type,
            duration=duration_s,
            bandwidth=bandwidth_hz,
            use="inversion",
            system=system,
        )
        gz_spoil, _, _ = pp.make_crusher(
            spoiling_cycles, voxel_size_m, axis, system=system
        )
        # A slot per label, so the loop has somewhere to put one. An iteration
        # can leave it out or say something different with it, but it cannot
        # add a slot the module never built.
        prep_labels = [
            pp.make_label(type="SET", label=name, value=0) for name in labels or ()
        ]

        self.seq = pp.Sequence(system)
        self.seq.add_block(rf_prep, *prep_labels)
        self.seq.add_block(gz_spoil)

        self.center = rf_reference(rf_prep)
