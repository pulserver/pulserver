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
from ._common import AXES, DEFAULT_BANDWIDTH_HZ, bridge, solve_delay
from ._trajectories import NonCartesianGradient, Radial, Rosette, Spiral


class NonCartesianReadout(SequenceModule):
    """One non-Cartesian interleave, from the RF that starts it to the end of the TR.

    The trajectory is designed by a :class:`NonCartesianGradient` -- a radial
    spoke, a spiral arm, a rosette -- and this places it: the pulse, the
    prewinder that reaches the start of the path, the acquisition, the
    rewinder that comes back, and whatever TE and TR were asked for.

    **No sampling pattern reaches this class.** A non-Cartesian scan is one
    interleave rotated many ways, and which rotations to play is the loop's
    business, not the readout's. The default lays out that one interleave and
    lets the loop orient it with a rotation extension::

        for angle in pp.calc_golden_angles(num_arms):
            rotation = pp.make_rotation(Rotation.from_euler("z", angle))
            seq.add_block(readout.gx_read, readout.gy_read, readout.adc, rotation)

    ``explicit=True`` is the way out for an interpreter that cannot apply a
    rotation at runtime: given ``angles``, the module builds one rotated
    waveform set per angle and lays them all out, so ``gx_read`` and
    ``gy_read`` come back as **lists**, one entry per arm. It costs one
    registered waveform per arm, which is exactly what the rotation extension
    exists to avoid -- so it is opt-in, and ``angles`` is required with it.

    Attributes
    ----------
    rf : RfEvent
        The pulse the module was given.
    gz_select : GradEvent
        Its selection gradient, if one was given.
    gx_read, gy_read : GradEvent
        The interleave. Lists of one entry per angle when ``explicit``.
    gx_pre, gy_pre : GradEvent
        Bridges reaching the start of the path, when it does not start at
        k = 0. Lists when ``explicit``.
    gx_rew, gy_rew : GradEvent
        Bridges returning to k = 0. Lists when ``explicit``.
    gz_phase, gz_rew : TrapEvent
        Partition encode and its rewinder. Stacks only.
    gspoil : GradEvent
        End-of-TR spoiler, when ``spoiling_cycles`` is nonzero.
    adc : AdcEvent
        The acquisition window.
    adc_labels : list of LabelSetEvent
        One per name in ``labels``, in order; empty when ``labels`` is.
    wait_te, wait_tr : DelayEvent
        Present only when a TE or TR longer than the minimum was asked for.
    trajectory : NonCartesianGradient
        The designed interleave, for its ``trajectory`` array and timings.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    rf : RfEvent
        The pulse that opens the repetition.
    gz_select : GradEvent, optional
        A selection gradient played in the same block as ``rf``.
    te, tr : float, optional
        Echo time (s) from the RF isodelay to the k = 0 crossing, and
        repetition time (s) over the whole module. ``None`` is as short as
        possible.
    spoiling_cycles : float, optional
        Dephasing left at the end of the TR, in cycles across
        ``voxel_size_m``. Zero leaves the shot rewound.
    voxel_size_m : float, optional
        Length the spoiling is counted over (m). Defaults to the resolution.
    spoiling_axis : {'z', 'x', 'y'}, optional
        Axis the spoiler is played on.
    num_echoes : int, optional
        Times the interleave is replayed per repetition.
    explicit : bool, optional
        Lay out one rotated copy per entry of ``angles`` instead of one base
        interleave.
    angles : array-like, optional
        In-plane rotations (rad). Required when ``explicit``, and refused
        otherwise -- a sampling pattern is not the readout's to hold.
    labels : sequence of str, optional
        Counters emitted on the acquisition block.
    trigger : event, optional
        A trigger or digital output armed on the block that opens the readout.

    Raises
    ------
    ValueError
        If ``num_echoes`` is below 1, ``angles`` and ``explicit`` disagree, or
        the requested TE or TR is shorter than the module can achieve.
    """

    #: Set by a subclass to the trajectory factory it wraps.
    _designer: Any = None
    #: Cartesian axis this family encodes on top of the interleave, if any.
    _phase_axis: str | None = None

    def init_module(
        self,
        system: pp.Opts,
        rf: Any,
        gz_select: Any = None,
        *,
        trajectory: NonCartesianGradient,
        phase_fov_m: float | None = None,
        phase_matrix: int | None = None,
        te: float | None = None,
        tr: float | None = None,
        spoiling_cycles: float = 0.0,
        voxel_size_m: float | None = None,
        spoiling_axis: str = "z",
        num_echoes: int = 1,
        explicit: bool = False,
        angles: Any = None,
        labels: tuple[str, ...] | None = None,
        trigger: Any = None,
    ) -> None:
        num_echoes = int(num_echoes)
        if num_echoes < 1:
            raise ValueError("num_echoes must be >= 1")
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

        arms = (
            [trajectory.rotated(float(angle)) for angle in np.atleast_1d(angles)]
            if explicit
            else [trajectory]
        )

        # --- partition encoding, for a stack ---------------------------------
        gz_phase = gz_rew = None
        if self._phase_axis is not None:
            if not phase_fov_m or not phase_matrix:
                raise ValueError("a stack needs phase_fov_m and phase_matrix")
            gz_phase = pp.make_phase_encoding(
                self._phase_axis, float(phase_fov_m) / int(phase_matrix), system=system
            )
            gz_rew = pp.scale_grad(gz_phase, -1.0)

        # --- the spoiler ------------------------------------------------------
        gspoil = None
        if spoiling_cycles:
            if voxel_size_m is None:
                voxel_size_m = _resolution(trajectory)
            if voxel_size_m <= 0:
                raise ValueError("voxel_size_m must be positive")
            gspoil, _, _ = pp.make_crusher(
                spoiling_cycles, voxel_size_m, spoiling_axis, system=system
            )

        # --- align every arm's brackets to one common length ------------------
        # Rotating re-solves the bridges, so arm to arm they differ by a raster
        # or two. A repetition whose length depended on which arm it played
        # would not have one TR, so they are all given the longest.
        # Rounded onto the block raster, because that is what `add_block` will
        # do to them anyway, and a TE computed from an unrounded span would be
        # a raster short of where the echo actually lands.
        _raster = system.block_duration_raster
        _pre_span = pp.ceil_to_raster(
            max(
                (
                    pp.calc_duration(*arm.prewinders, *([gz_phase] if gz_phase else []))
                    for arm in arms
                    if arm.prewinders or gz_phase
                ),
                default=0.0,
            ),
            _raster,
        )
        _rew_span = pp.ceil_to_raster(
            max(
                (
                    pp.calc_duration(
                        *arm.rewinders,
                        *([gz_rew] if gz_rew else []),
                        *([gspoil] if gspoil else []),
                    )
                    for arm in arms
                    if arm.rewinders or gz_rew or gspoil
                ),
                default=0.0,
            ),
            _raster,
        )
        _pre_floor = pp.make_delay(_pre_span) if _pre_span else None
        _rew_floor = pp.make_delay(_rew_span) if _rew_span else None

        gx_pre, gy_pre = _bracket(arms, "prewinders", "right", _pre_span, system)
        gx_read, gy_read = _bracket(arms, "gradients", None, 0.0, system)
        gx_rew, gy_rew = _bracket(arms, "rewinders", "left", _rew_span, system)

        adc = trajectory.adc
        adc_labels = [pp.make_label(type="SET", label=name, value=0) for name in labels or ()]
        self.register(adc_labels=adc_labels)

        # --- lay out the repetitions ------------------------------------------
        self.seq = pp.Sequence(system)
        _echo_offset = _echo_offset_of(trajectory, system.grad_raster_time)
        _rf_reference = float(rf.delay) + float(rf.center)
        _rf_block = pp.ceil_to_raster(
            pp.calc_duration(rf, gz_select) if gz_select is not None else pp.calc_duration(rf),
            _raster,
        )
        te_min = _rf_block - _rf_reference + _pre_span + _echo_offset
        _te_delay = solve_delay(te, te_min, "TE", system)
        wait_te = pp.make_delay(_te_delay) if _te_delay else None

        for _arm in range(len(arms)):
            if gz_select is not None:
                self.seq.add_block(rf, gz_select)
            else:
                self.seq.add_block(rf)
            if wait_te is not None:
                self.seq.add_block(wait_te)
            if _pre_floor is not None:
                self.seq.add_block(
                    *_at(gx_pre, _arm), *_at(gy_pre, _arm),
                    *([gz_phase] if gz_phase else []),
                    *([trigger] if trigger is not None else []),
                    _pre_floor,
                )
            for _echo in range(num_echoes):
                self.seq.add_block(*_at(gx_read, _arm), *_at(gy_read, _arm), adc, *adc_labels)
            if _rew_floor is not None:
                self.seq.add_block(
                    *_at(gx_rew, _arm), *_at(gy_rew, _arm),
                    *([gz_rew] if gz_rew else []),
                    *([gspoil] if gspoil else []),
                    _rew_floor,
                )

        _repetition = self.seq.duration()[0] / len(arms)
        _tr_delay = solve_delay(tr, _repetition, "TR", system)
        if _tr_delay:
            wait_tr = pp.make_delay(_tr_delay)
            # One per repetition, so every arm laid out is a whole TR.
            for _ in arms:
                self.seq.add_block(wait_tr)

        self.trajectory = trajectory
        self.echo_time = te_min + _te_delay
        self.center = self.echo_time + _rf_reference
        self.duration = _repetition + _tr_delay
        self.bandwidth_hz = 1.0 / float(adc.dwell)

        # The events are built here, but a subclass's `init_module` is the frame
        # automatic publication finds, and it has never heard of them.
        self.publish()


def _at(events, index):
    """The arm's entry of a per-arm list, or the shared event, as a block argument."""
    if events is None:
        return ()
    if isinstance(events, list):
        return (events[index],)
    return (events,)


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
    case -- and a list of one per arm otherwise, which is what makes the
    module publish ``gx_read`` as a list exactly when the arms really differ.
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
    its first sample, a radial spoke or an in-out spiral halfway through, and
    a rosette at every petal.

    Integrated against each event's own ``tt`` rather than against a raster
    count, because the two are not the same thing. A solved waveform stores
    one amplitude per raster, but an extended trapezoid stores only its
    vertices -- a flat radial lobe is *two* samples, whose midpoint is nowhere
    near halfway through the readout.
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
        path[1:, index] = np.cumsum(
            0.5 * (amplitudes[1:] + amplitudes[:-1]) * np.diff(grid)
        )

    for pre in trajectory.prewinders:
        path[:, trajectory.axes.index(pre.channel)] += float(
            np.trapezoid(pre.waveform, pre.tt)
        )
    return float(grid[int(np.argmin(np.linalg.norm(path, axis=1)))])


# ----------------------------------------------------------------------
# Concrete families
# ----------------------------------------------------------------------


class _Planar(NonCartesianReadout):
    """A readout whose interleave lies in one plane and is turned within it."""

    def init_module(self, system, rf, gz_select=None, *, fov_m, matrix, **kwargs):
        design = {name: kwargs.pop(name) for name in list(kwargs) if name in self._design_keys}
        trajectory = self._design(system, fov_m, matrix, **design)
        super().init_module(system, rf, gz_select, trajectory=trajectory, **kwargs)

    #: Arguments this family forwards to its trajectory designer.
    _design_keys: frozenset = frozenset()

    @staticmethod
    def _design(system, fov_m, matrix, **kwargs):
        raise NotImplementedError


class _Stack(_Planar):
    """A planar interleave promoted to 3D by a Cartesian partition encode."""

    _phase_axis = "z"

    def init_module(
        self, system, rf, gz_select=None, *, fov_m, matrix, fov_z_m, matrix_z, **kwargs
    ):
        super().init_module(
            system, rf, gz_select,
            fov_m=fov_m, matrix=matrix,
            phase_fov_m=fov_z_m, phase_matrix=matrix_z,
            **kwargs,
        )


class _Projection(_Planar):
    """A planar interleave the loop turns over a sphere rather than within a plane.

    Its blocks are a 2D readout's: what makes it a projection acquisition is
    the rotations applied to it, and those belong to the scan loop. The class
    exists so that intent is stated where the readout is built, and so that a
    partition encode -- which a projection acquisition has no place for -- is
    refused rather than silently accepted.
    """


# -- radial ---------------------------------------------------------------

_RADIAL_KEYS = frozenset({"bandwidth_hz_px", "oversamp", "ro_axis", "derate"})


def _design_radial(system, fov_m, matrix, **kwargs):
    return Radial(system, fov_m, matrix, **kwargs)


class RadialReadout2D(_Planar):
    """A full radial spoke through the centre of a plane.

    Examples
    --------
    >>> import pulserver.design as design
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts()
    >>> excitation = design.SpatialSelectiveExcitation(system, 15.0, 5e-3)
    >>> readout = design.RadialReadout2D(
    ...     system, excitation.rf, excitation.gz_select, fov_m=0.22, matrix=128
    ... )
    >>> readout.gx_read.channel
    'x'
    """

    _design_keys = _RADIAL_KEYS
    _design = staticmethod(_design_radial)


class RadialStackReadout(_Stack):
    """Radial spokes in-plane, Cartesian partitions along z: stack of stars."""

    _design_keys = _RADIAL_KEYS
    _design = staticmethod(_design_radial)


class RadialProjectionReadout(_Projection):
    """Radial spokes turned over a sphere: a koosh-ball acquisition."""

    _design_keys = _RADIAL_KEYS
    _design = staticmethod(_design_radial)


# -- spiral ---------------------------------------------------------------

_SPIRAL_KEYS = frozenset({
    "direction",
    "density",
    "inner_design_interleaves",
    "outer_design_interleaves",
    "variable_density_power",
    "transition_radius",
    "transition_speed",
    "num_points",
    "bandwidth_hz_px",
    "oversamp",
    "axes",
    "solver_oversampling",
    "derate",
})


def _design_spiral(system, fov_m, matrix, design_interleaves=16, **kwargs):
    return Spiral(system, fov_m, matrix, design_interleaves, **kwargs)


class SpiralReadout2D(_Planar):
    """One spiral arm in a plane.

    ``direction`` runs the arm centre-to-edge (``'outward'``), edge-to-centre
    (``'inward'``), or edge-to-centre-to-edge (``'in_out'``); ``density``
    chooses a constant, variable or dual pitch. ``design_interleaves`` sets
    that pitch and is **not** how many arms the loop plays.

    Examples
    --------
    >>> import pulserver.design as design
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts()
    >>> excitation = design.SpatialSelectiveExcitation(system, 15.0, 5e-3)
    >>> readout = design.SpiralReadout2D(
    ...     system, excitation.rf, excitation.gz_select,
    ...     fov_m=0.22, matrix=128, design_interleaves=16, direction="in_out",
    ... )
    >>> readout.trajectory.direction
    'in_out'
    """

    _design_keys = _SPIRAL_KEYS | {"design_interleaves"}
    _design = staticmethod(_design_spiral)


class SpiralStackReadout(_Stack):
    """Spiral arms in-plane, Cartesian partitions along z."""

    _design_keys = _SPIRAL_KEYS | {"design_interleaves"}
    _design = staticmethod(_design_spiral)


class SpiralProjectionReadout(_Projection):
    """Spiral arms turned over a sphere."""

    _design_keys = _SPIRAL_KEYS | {"design_interleaves"}
    _design = staticmethod(_design_spiral)


# -- rosette --------------------------------------------------------------

_ROSETTE_KEYS = frozenset({
    "petals",
    "angular_frequency_ratio",
    "echo_spacing_s",
    "bandwidth_hz_px",
    "oversamp",
    "axes",
    "solver_oversampling",
    "derate",
})


def _design_rosette(system, fov_m, matrix, **kwargs):
    return Rosette(system, fov_m, matrix, **kwargs)


class RosetteReadout2D(_Planar):
    """One multi-petal rosette interleave in a plane."""

    _design_keys = _ROSETTE_KEYS
    _design = staticmethod(_design_rosette)


class RosetteStackReadout(_Stack):
    """Rosette petals in-plane, Cartesian partitions along z."""

    _design_keys = _ROSETTE_KEYS
    _design = staticmethod(_design_rosette)


class RosetteProjectionReadout(_Projection):
    """Rosette petals turned over a sphere."""

    _design_keys = _ROSETTE_KEYS
    _design = staticmethod(_design_rosette)
