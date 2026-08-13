"""Non-Cartesian readouts: one designed interleave, played as a whole repetition."""

from __future__ import annotations

__all__ = [
    "NonCartesianReadout",
    "RadialProjectionReadout",
    "RadialReadout2D",
    "RadialStackReadout",
    "RosetteProjectionReadout",
    "RosetteReadout2D",
    "RosetteStackReadout",
    "SpiralProjectionReadout",
    "SpiralReadout2D",
    "SpiralStackReadout",
]

from typing import Any

import numpy as np

from ... import pypulseq as pp
from ..._core._module import SequenceModule
from ._common import AXES, solve_delay
from ._trajectories import NonCartesianGradient, Rosette, Spiral

_READOUT_GRAD_MARGIN = 0.8


# ======================================================================
# Radial
# ======================================================================


class _RadialReadout(SequenceModule):
    """A full radial spoke, prephaser and rewinder merged into one waveform.

    A spoke runs from -kmax through the centre to +kmax and comes back, so the
    same lobe serves as prephaser and as rewinder and the whole traversal is
    one continuous gradient. Building it that way -- rather than as three
    blocks -- is what lets the scan loop play a spoke as a single
    ``add_block``, and what lets a rotation event orient it without having to
    orient three separate things consistently.

    The ADC is shifted onto the readout plateau, so it acquires the flat part
    and nothing else.

    Orientation is the loop's business. The default publishes one spoke along
    x for a rotation event to turn::

        for angle in pp.calc_golden_angles(n_spokes):
            rotation = pp.make_rotation(Rotation.from_euler("z", angle))
            seq.add_block(readout.gx, readout.adc, rotation)

    ``explicit=True`` with ``angles`` instead writes every spoke out, as a
    cosine and sine scaling of the base waveform, so ``gx`` and ``gy`` come
    back as **lists** with one entry per angle. That costs a registered
    waveform per spoke, which is what the rotation extension exists to avoid,
    so it is opt-in and ``angles`` is then required.

    Attributes
    ----------
    rf : RfEvent
        The pulse the module was given.
    gz : GradEvent
        Its selection gradient, if one was given.
    gx, gy : GradEvent
        The spoke, prephaser through rewinder. ``gy`` is the same waveform at
        zero amplitude, so a block always has a y slot for a rotation to fill.
        Lists of one entry per angle when ``explicit``.
    gz_pre, gz_rew : TrapEvent
        Partition encode and its rewinder. Stacks only.
    gz_spoil : GradEvent
        End-of-TR spoiler, when ``spoiling_cycles`` is nonzero.
    adc : AdcEvent
        The acquisition window, delayed onto the readout plateau.
    adc_labels : LabelSetEvent or list of LabelSetEvent
        One per name in ``labels``; a bare event when there is one.
    wait_te, wait_tr : DelayEvent
        Present only when a TE or TR longer than the minimum was asked for.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    rf : RfEvent
        The pulse that opens the repetition.
    gz : GradEvent, optional
        A selection gradient played in the same block as ``rf``.
    fov : float
        Isotropic in-plane field of view (m).
    matrix : int
        In-plane matrix size. Sets ``kmax = matrix / (2 * fov)``.
    fov_z, matrix_z : float, int
        Partition field of view (m) and count. Stacks only.
    te, tr : float, optional
        Echo time (s) from the RF isodelay to the centre crossing, and
        repetition time (s). ``None`` is as short as possible.
    oversampling : float, optional
        Readout oversampling: more samples along the same spoke.
    readout_bandwidth_hz : float, optional
        Requested receiver bandwidth. Read ``bandwidth_hz`` for what the two
        rasters actually allowed.
    spoiling_cycles : float, optional
        Dephasing left at the end of the TR, in cycles across
        ``voxel_size_m``. Zero leaves the spoke rewound.
    voxel_size_m : float, optional
        Length the spoiling is counted over (m); the resolution by default.
    spoiling_axis : {'z', 'x', 'y'}, optional
        Axis the spoiler is played on.
    n_echoes : int, optional
        Times the spoke is replayed per repetition.
    explicit : bool, optional
        Write out one spoke per entry of ``angles`` instead of one base spoke.
    angles : array-like, optional
        In-plane rotations (rad). Required when ``explicit``, refused
        otherwise -- a sampling pattern is not the readout's to hold.
    labels : sequence of str, optional
        Counters emitted on the acquisition block.
    trigger : event, optional
        A trigger or digital output armed on the block that opens the readout.

    Raises
    ------
    ValueError
        If a count is out of range, ``angles`` and ``explicit`` disagree, or
        the requested TE or TR is shorter than the module can achieve.
    """

    #: Set by the stack variants to the axis they encode partitions on.
    _phase_axis: str | None = None

    def init_module(
        self,
        system: pp.Opts,
        rf: Any,
        gz: Any = None,
        *,
        fov: float,
        matrix: int,
        fov_z: float | None = None,
        matrix_z: int | None = None,
        te: float | None = None,
        tr: float | None = None,
        oversampling: float = 1.0,
        readout_bandwidth_hz: float = 250e3,
        spoiling_cycles: float = 0.0,
        voxel_size_m: float | None = None,
        spoiling_axis: str = "z",
        n_echoes: int = 1,
        explicit: bool = False,
        angles: Any = None,
        labels: tuple[str, ...] | None = None,
        trigger: Any = None,
    ) -> None:
        n_echoes = _checked_layout(
            n_echoes, spoiling_cycles, spoiling_axis, explicit, angles
        )
        if fov <= 0 or int(matrix) < 2:
            raise ValueError("fov must be positive and matrix must be >= 2")
        if oversampling < 1.0:
            raise ValueError("oversampling must be >= 1")
        if readout_bandwidth_hz <= 0:
            raise ValueError("readout_bandwidth_hz must be positive")

        # A spoke spans 2 * kmax, sampled n_samples times.
        delta_kx = 1.0 / fov
        n_samples = max(2, round(oversampling * int(matrix)))
        readout_area = int(matrix) * delta_kx
        dwell, readout_duration = pp.calc_adc_timing(
            n_samples,
            1.0 / readout_bandwidth_hz,
            grad_raster_time=system.grad_raster_time,
            adc_raster_time=system.adc_raster_time,
            min_readout_duration=readout_area / (_READOUT_GRAD_MARGIN * system.max_grad),
        )

        gx = pp.make_trapezoid(
            channel="x", flat_area=readout_area, flat_time=readout_duration, system=system
        )
        # The spoke runs out through the centre and back, so one lobe of half
        # the readout area serves as both prephaser and rewinder.
        gx_pre = pp.make_trapezoid(channel="x", area=-0.5 * gx.area, system=system)
        gx_rew = pp.make_trapezoid(channel="x", area=-0.5 * gx.area, system=system)
        rise_time = gx.rise_time
        adc = pp.make_adc(
            num_samples=n_samples, dwell=dwell, delay=rise_time, system=system
        )

        # Shift each piece to where it is played, then sum them into one
        # continuous waveform.
        gx_pre_duration = pp.calc_duration(gx_pre)
        gx.delay += gx_pre_duration
        adc.delay += gx_pre_duration
        gx_rew.delay += pp.calc_duration(gx)
        gx = pp.add_gradients(grads=[gx_pre, gx, gx_rew], system=system)
        gy = pp.scale_grad(gx, 0.0)
        gy.channel = "y"

        echo_offset = gx_pre_duration + rise_time + 0.5 * readout_duration
        if explicit:
            angles = np.atleast_1d(np.asarray(angles, dtype=float))
            base = gx
            gx = [pp.scale_grad(base, float(np.cos(angle))) for angle in angles]
            gy = []
            for angle in angles:
                lobe = pp.scale_grad(base, float(np.sin(angle)))
                lobe.channel = "y"
                gy.append(lobe)

        gz_pre, gz_rew = _partition_encode(
            self._phase_axis, fov_z, matrix_z, type(self).__name__, system
        )

        gz_spoil = None
        if spoiling_cycles:
            if voxel_size_m is None:
                voxel_size_m = fov / int(matrix)
            if voxel_size_m <= 0:
                raise ValueError("voxel_size_m must be positive")
            gz_spoil, _, _ = pp.make_crusher(
                spoiling_cycles, voxel_size_m, spoiling_axis, system=system
            )

        adc_labels = [pp.make_label(type="SET", label=name, value=0) for name in labels or ()]
        n_arms = len(gx) if isinstance(gx, list) else 1

        self.seq = pp.Sequence(system)

        rf_center = float(rf.delay) + float(rf.center)
        rf_block = pp.ceil_to_raster(
            pp.calc_duration(rf, gz) if gz is not None else pp.calc_duration(rf),
            system.block_duration_raster,
        )
        pre_span = (
            pp.ceil_to_raster(pp.calc_duration(gz_pre), system.block_duration_raster)
            if gz_pre is not None
            else 0.0
        )
        te_min = rf_block - rf_center + pre_span + echo_offset
        te_delay = solve_delay(te, te_min, "TE", system)
        wait_te = pp.make_delay(te_delay) if te_delay else None

        for i_arm in range(n_arms):
            if gz is not None:
                self.seq.add_block(rf, gz)
            else:
                self.seq.add_block(rf)
            if wait_te is not None:
                self.seq.add_block(wait_te)
            if gz_pre is not None:
                self.seq.add_block(gz_pre, *_armed(trigger))
            for _ in range(n_echoes):
                self.seq.add_block(
                    _at(gx, i_arm), _at(gy, i_arm), adc, *adc_labels,
                    *(_armed(trigger) if gz_pre is None else ()),
                )
            if gz_rew is not None or gz_spoil is not None:
                self.seq.add_block(*(event for event in (gz_rew, gz_spoil) if event is not None))

        tr_min = self.seq.duration()[0] / n_arms
        tr_delay = solve_delay(tr, tr_min, "TR", system)
        if tr_delay:
            wait_tr = pp.make_delay(tr_delay)
            for _ in range(n_arms):
                self.seq.add_block(wait_tr)

        self.echo_time = te_min + te_delay
        self.center = self.echo_time + rf_center
        self.duration = tr_min + tr_delay
        self.bandwidth_hz = 1.0 / dwell
        self.n_samples = n_samples
        self.readout_duration = readout_duration


class RadialReadout2D(_RadialReadout):
    """A full radial spoke through the centre of a plane.

    Examples
    --------
    >>> import pulserver.design as design
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts()
    >>> excitation = design.SpatialSelectiveExcitation(system, 15.0, 5e-3)
    >>> readout = design.RadialReadout2D(
    ...     system, excitation.rf, excitation.gz, fov=0.22, matrix=128
    ... )
    >>> len(readout.blocks), readout.gx.channel
    (2, 'x')
    """


class RadialStackReadout(_RadialReadout):
    """Radial spokes in-plane, Cartesian partitions along z: stack of stars."""

    _phase_axis = "z"


class RadialProjectionReadout(_RadialReadout):
    """Radial spokes turned over a sphere: a koosh-ball acquisition.

    Its blocks are a 2D readout's: what makes it a projection acquisition is
    the rotations the loop applies, and those belong to the loop. The class
    exists so the intent is stated where the readout is built, and so that a
    partition encode -- which such an acquisition has no place for -- is
    refused rather than quietly accepted.
    """


# ======================================================================
# Solved trajectories: spiral and rosette
# ======================================================================


class NonCartesianReadout(SequenceModule):
    """A solved interleave, bracketed by the bridges that reach it and leave it.

    Where a radial spoke is one continuous lobe, a spiral or a rosette is a
    waveform solved against the vector limits, and whichever of its endpoints
    is away from k = 0 needs a bridge of its own. So the repetition is the
    pulse, a prewinder block when the path does not start at the centre, the
    acquisition, and a rewinder block when it does not end there.

    Subclass this to add a family: design a :class:`NonCartesianGradient` and
    hand it over. Everything below -- the bracket alignment, the TE and TR
    budget, the spoiler, the ``explicit`` path -- is inherited, and the events
    a subclass builds are published alongside the ones built here.

    Orientation is the loop's business; see :class:`_RadialReadout` for the
    two ways to apply it, which are the same here.

    Attributes
    ----------
    rf : RfEvent
        The pulse the module was given.
    gz : GradEvent
        Its selection gradient, if one was given.
    gx, gy : GradEvent
        The interleave. Lists of one entry per angle when ``explicit``.
    gx_pre, gy_pre : GradEvent
        Bridges reaching the start of the path, when it is not k = 0.
    gx_rew, gy_rew : GradEvent
        Bridges returning to k = 0, when the path does not end there.
    gz_pre, gz_rew : TrapEvent
        Partition encode and its rewinder. Stacks only.
    gz_spoil : GradEvent
        End-of-TR spoiler, when ``spoiling_cycles`` is nonzero.
    adc : AdcEvent
        The acquisition window.
    adc_labels : LabelSetEvent or list of LabelSetEvent
        One per name in ``labels``; a bare event when there is one.
    wait_te, wait_tr : DelayEvent
        Present only when a TE or TR longer than the minimum was asked for.
    trajectory : NonCartesianGradient
        The designed interleave, for its ``trajectory`` array and timings.

    Parameters
    ----------
    Shared with :class:`_RadialReadout`, except that the interleave arrives as
    ``trajectory`` rather than being built from ``fov`` and ``matrix``.
    """

    _phase_axis: str | None = None

    def init_module(
        self,
        system: pp.Opts,
        rf: Any,
        gz: Any = None,
        *,
        trajectory: NonCartesianGradient,
        fov_z: float | None = None,
        matrix_z: int | None = None,
        te: float | None = None,
        tr: float | None = None,
        spoiling_cycles: float = 0.0,
        voxel_size_m: float | None = None,
        spoiling_axis: str = "z",
        n_echoes: int = 1,
        explicit: bool = False,
        angles: Any = None,
        labels: tuple[str, ...] | None = None,
        trigger: Any = None,
    ) -> None:
        n_echoes = _checked_layout(
            n_echoes, spoiling_cycles, spoiling_axis, explicit, angles
        )
        arms = (
            [trajectory.rotated(float(angle)) for angle in np.atleast_1d(angles)]
            if explicit
            else [trajectory]
        )

        gz_pre, gz_rew = _partition_encode(
            self._phase_axis, fov_z, matrix_z, type(self).__name__, system
        )

        gz_spoil = None
        if spoiling_cycles:
            if voxel_size_m is None:
                voxel_size_m = _resolution(trajectory)
            if voxel_size_m <= 0:
                raise ValueError("voxel_size_m must be positive")
            gz_spoil, _, _ = pp.make_crusher(
                spoiling_cycles, voxel_size_m, spoiling_axis, system=system
            )

        # Rotating re-solves the bridges, so arm to arm they differ by a raster
        # or two. A repetition whose length depended on which arm it played
        # would not have one TR, so every arm is given the longest.
        raster = system.block_duration_raster
        pre_span = pp.ceil_to_raster(
            max(
                (
                    pp.calc_duration(*arm.prewinders, *_present(gz_pre))
                    for arm in arms
                    if arm.prewinders or gz_pre is not None
                ),
                default=0.0,
            ),
            raster,
        )
        rew_span = pp.ceil_to_raster(
            max(
                (
                    pp.calc_duration(*arm.rewinders, *_present(gz_rew), *_present(gz_spoil))
                    for arm in arms
                    if arm.rewinders or gz_rew is not None or gz_spoil is not None
                ),
                default=0.0,
            ),
            raster,
        )

        gx_pre, gy_pre = _bracket(arms, "prewinders", "right", pre_span, system)
        gx, gy = _bracket(arms, "gradients", None, 0.0, system)
        gx_rew, gy_rew = _bracket(arms, "rewinders", "left", rew_span, system)
        _pre_floor = pp.make_delay(pre_span) if pre_span else None
        _rew_floor = pp.make_delay(rew_span) if rew_span else None

        adc = trajectory.adc
        adc_labels = [pp.make_label(type="SET", label=name, value=0) for name in labels or ()]

        self.seq = pp.Sequence(system)

        rf_center = float(rf.delay) + float(rf.center)
        rf_block = pp.ceil_to_raster(
            pp.calc_duration(rf, gz) if gz is not None else pp.calc_duration(rf), raster
        )
        te_min = (
            rf_block
            - rf_center
            + pre_span
            + _echo_offset_of(trajectory, system.grad_raster_time)
        )
        te_delay = solve_delay(te, te_min, "TE", system)
        wait_te = pp.make_delay(te_delay) if te_delay else None

        for i_arm in range(len(arms)):
            if gz is not None:
                self.seq.add_block(rf, gz)
            else:
                self.seq.add_block(rf)
            if wait_te is not None:
                self.seq.add_block(wait_te)
            if _pre_floor is not None:
                self.seq.add_block(
                    *_present(_at(gx_pre, i_arm)), *_present(_at(gy_pre, i_arm)),
                    *_present(gz_pre), *_armed(trigger), _pre_floor,
                )
            for _ in range(n_echoes):
                self.seq.add_block(_at(gx, i_arm), _at(gy, i_arm), adc, *adc_labels)
            if _rew_floor is not None:
                self.seq.add_block(
                    *_present(_at(gx_rew, i_arm)), *_present(_at(gy_rew, i_arm)),
                    *_present(gz_rew), *_present(gz_spoil), _rew_floor,
                )

        tr_min = self.seq.duration()[0] / len(arms)
        tr_delay = solve_delay(tr, tr_min, "TR", system)
        if tr_delay:
            wait_tr = pp.make_delay(tr_delay)
            for _ in arms:
                self.seq.add_block(wait_tr)

        self.trajectory = trajectory
        self.echo_time = te_min + te_delay
        self.center = self.echo_time + rf_center
        self.duration = tr_min + tr_delay
        self.bandwidth_hz = 1.0 / float(adc.dwell)
        self.n_samples = int(adc.num_samples)


class _SpiralReadout(NonCartesianReadout):
    """One spiral arm, solved for the requested pitch and bandwidth."""

    def init_module(
        self,
        system: pp.Opts,
        rf: Any,
        gz: Any = None,
        *,
        fov: float,
        matrix: int,
        design_interleaves: int = 16,
        direction: str = "outward",
        density: str = "constant",
        inner_design_interleaves: float | None = None,
        outer_design_interleaves: float | None = None,
        variable_density_power: float = 2.0,
        transition_radius: float = 0.5,
        transition_speed: float = 12.0,
        oversampling: float = 1.0,
        readout_bandwidth_hz: float = 250e3,
        n_points: int = 1024,
        derate: bool = True,
        **kwargs: Any,
    ) -> None:
        trajectory = Spiral(
            system,
            fov,
            matrix,
            design_interleaves,
            direction=direction,
            density=density,
            inner_design_interleaves=inner_design_interleaves,
            outer_design_interleaves=outer_design_interleaves,
            variable_density_power=variable_density_power,
            transition_radius=transition_radius,
            transition_speed=transition_speed,
            num_points=n_points,
            bandwidth_hz_px=readout_bandwidth_hz,
            oversamp=oversampling,
            derate=derate,
        )
        super().init_module(system, rf, gz, trajectory=trajectory, **kwargs)


class SpiralReadout2D(_SpiralReadout):
    """One spiral arm in a plane.

    ``direction`` runs the arm centre-to-edge (``'outward'``), edge-to-centre
    (``'inward'``) or edge-to-centre-to-edge (``'in_out'``); ``density``
    chooses a constant, variable or dual pitch. ``design_interleaves`` sets
    that pitch and is **not** how many arms the loop plays.

    ``readout_bandwidth_hz`` is the requested sample spacing along the arm; the
    solver stretches the waveform to hold it when the time-optimal traversal
    would sample faster. Read ``bandwidth_hz`` for what was achieved.

    Examples
    --------
    >>> import pulserver.design as design
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts()
    >>> excitation = design.SpatialSelectiveExcitation(system, 15.0, 5e-3)
    >>> readout = design.SpiralReadout2D(
    ...     system, excitation.rf, excitation.gz,
    ...     fov=0.22, matrix=128, design_interleaves=16, direction="in_out",
    ... )
    >>> readout.trajectory.direction
    'in_out'
    """


class SpiralStackReadout(_SpiralReadout):
    """Spiral arms in-plane, Cartesian partitions along z."""

    _phase_axis = "z"


class SpiralProjectionReadout(_SpiralReadout):
    """Spiral arms turned over a sphere."""


class _RosetteReadout(NonCartesianReadout):
    """One multi-petal rosette interleave."""

    def init_module(
        self,
        system: pp.Opts,
        rf: Any,
        gz: Any = None,
        *,
        fov: float,
        matrix: int,
        petals: int = 5,
        angular_frequency_ratio: float = 3.0 / 5.0,
        echo_spacing_s: float | None = None,
        oversampling: float = 1.0,
        readout_bandwidth_hz: float = 250e3,
        derate: bool = True,
        **kwargs: Any,
    ) -> None:
        trajectory = Rosette(
            system,
            fov,
            matrix,
            petals=petals,
            angular_frequency_ratio=angular_frequency_ratio,
            echo_spacing_s=echo_spacing_s,
            bandwidth_hz_px=readout_bandwidth_hz,
            oversamp=oversampling,
            derate=derate,
        )
        super().init_module(system, rf, gz, trajectory=trajectory, **kwargs)


class RosetteReadout2D(_RosetteReadout):
    """One multi-petal rosette interleave in a plane."""


class RosetteStackReadout(_RosetteReadout):
    """Rosette petals in-plane, Cartesian partitions along z."""

    _phase_axis = "z"


class RosetteProjectionReadout(_RosetteReadout):
    """Rosette petals turned over a sphere."""


# ======================================================================
# Shared arithmetic
# ======================================================================


def _checked_layout(n_echoes, spoiling_cycles, spoiling_axis, explicit, angles) -> int:
    """Validate the arguments every non-Cartesian readout shares."""
    n_echoes = int(n_echoes)
    if n_echoes < 1:
        raise ValueError("n_echoes must be >= 1")
    if spoiling_cycles < 0:
        raise ValueError("spoiling_cycles must be >= 0")
    if spoiling_axis not in AXES:
        raise ValueError(f"spoiling_axis must be one of {AXES}, got {spoiling_axis!r}")
    if explicit and angles is None:
        raise ValueError("explicit=True needs the angles to lay out")
    if angles is not None and not explicit:
        raise ValueError(
            "angles are a sampling pattern, which a readout does not hold; rotate the "
            "interleave in the scan loop, or pass explicit=True to write every arm out"
        )
    return n_echoes


def _partition_encode(axis, fov_z, matrix_z, owner, system):
    """The partition encode and its rewinder, or a refusal.

    Only a stack has an axis to encode on. A readout without one that is handed
    ``fov_z`` anyway has been mistaken for a stack, and quietly dropping the
    argument would leave a 2D acquisition wearing a 3D protocol.
    """
    if axis is None:
        if fov_z is not None or matrix_z is not None:
            raise ValueError(
                f"{owner} encodes no partition axis, so fov_z and matrix_z mean nothing to "
                "it; a stack variant is the class that takes them"
            )
        return None, None
    if not fov_z or not matrix_z:
        raise ValueError("a stack needs fov_z and matrix_z")
    gz_pre = pp.make_phase_encoding(axis, float(fov_z) / int(matrix_z), system=system)
    return gz_pre, pp.scale_grad(gz_pre, -1.0)


def _present(event):
    """``(event,)`` when there is one, so it can be splatted into a block."""
    return () if event is None else (event,)


def _armed(trigger):
    return () if trigger is None else (trigger,)


def _at(events, index):
    """The arm's entry of a per-arm list, or the shared event."""
    return events[index] if isinstance(events, list) else events


def _share_time_grid(system, events):
    """Re-emit a bracket's two halves on one set of vertex times.

    A prewinder and a rewinder are solved one axis at a time, so the x and y
    halves come back with different breakpoints and different lengths. Played
    as they are that is harmless -- each axis ends where it should. Under a
    rotation it is not: resolving the extension mixes the two axes, and a
    breakpoint one of them does not have becomes a step, which reads as a slew
    violation in a waveform that never violated anything.

    Both halves are therefore resampled onto the union of their times. The
    padding is exact rather than approximate: an aligned bracket has every
    half at zero on the side it does not reach, so a half that starts or ends
    early is being extended with the zero it already held.
    """
    present = [event for event in events if event is not None]
    if len(present) < 2:
        return events

    times = [float(event.delay) + np.asarray(event.tt, dtype=float) for event in present]
    grid = np.unique(np.concatenate(times))
    base = float(grid[0])

    shared = []
    for event in events:
        if event is None:
            shared.append(None)
            continue
        own = float(event.delay) + np.asarray(event.tt, dtype=float)
        amplitudes = np.interp(
            grid, own, np.asarray(event.waveform, dtype=float), left=0.0, right=0.0
        )
        rebuilt = pp.make_extended_trapezoid(
            channel=event.channel, amplitudes=amplitudes, times=grid - base, system=system
        )
        rebuilt.delay = base
        shared.append(rebuilt)
    return shared


def _bracket(arms, attribute, alignment, span, system):
    """The x and y halves of one bracket, per arm, padded to ``span``.

    Returns a bare event when every arm shares one -- the single-interleave
    case -- and a list of one per arm otherwise, which is what makes the module
    publish ``gx`` as a list exactly when the arms really differ.
    """
    per_arm: dict[str, list] = {"x": [], "y": []}
    for arm in arms:
        events = list(getattr(arm, attribute))
        if alignment and events:
            events = list(pp.align(**{alignment: [*events, pp.make_delay(span)]}))[:-1]
        halves = [next((e for e in events if e.channel == channel), None) for channel in ("x", "y")]
        if alignment:
            halves = _share_time_grid(system, halves)
        for channel, half in zip(("x", "y"), halves, strict=True):
            per_arm[channel].append(half)

    result = []
    for channel in ("x", "y"):
        entries = per_arm[channel]
        if all(entry is None for entry in entries):
            result.append(None)
        elif len(entries) == 1:
            result.append(entries[0])
        else:
            result.append(entries)
    return result


def _resolution(trajectory: NonCartesianGradient) -> float:
    """Nominal resolution of an interleave: half a period at its largest |k|."""
    path = np.asarray(trajectory.trajectory, dtype=float)
    kmax = float(np.max(np.linalg.norm(path[:, :2], axis=1)))
    return 1.0 / (2.0 * kmax)


def _echo_offset_of(trajectory: NonCartesianGradient, raster: float) -> float:
    """Time from the start of the acquisition block to its k = 0 crossing.

    Read off the integrated gradient rather than assumed: a spiral crosses at
    its first sample, an in-out spiral halfway through, and a rosette at every
    petal.

    Integrated against each event's own ``tt`` rather than against a raster
    count, because the two are not the same thing. A solved waveform stores one
    amplitude per raster, but an extended trapezoid stores only its vertices,
    whose midpoint is nowhere near halfway through the readout.
    """
    events = trajectory.gradients
    span = max(float(e.delay) + float(np.asarray(e.tt)[-1]) for e in events)
    grid = np.unique(
        np.concatenate(
            [float(e.delay) + np.asarray(e.tt, dtype=float) for e in events]
            + [np.arange(0.0, span + raster, raster)]
        )
    )
    grid = grid[grid <= span + 1e-12]

    path = np.zeros((grid.size, len(events)))
    for index, event in enumerate(events):
        times = float(event.delay) + np.asarray(event.tt, dtype=float)
        amplitudes = np.interp(grid, times, np.asarray(event.waveform, dtype=float))
        path[1:, index] = np.cumsum(0.5 * (amplitudes[1:] + amplitudes[:-1]) * np.diff(grid))

    for pre in trajectory.prewinders:
        path[:, trajectory.axes.index(pre.channel)] += float(np.trapezoid(pre.waveform, pre.tt))
    return float(grid[int(np.argmin(np.linalg.norm(path, axis=1)))])
