"""Zero-echo-time readouts: excite on the readout gradient, acquire from k = 0 out."""

from __future__ import annotations

__all__ = ["ZteReadout"]

import math
from typing import Any

import numpy as np

from ... import pypulseq as pp
from ..._core._module import SequenceModule
from ._common import AXES, solve_delay

#: Fraction of ``max_grad`` a spoke may reach.
_READOUT_GRAD_MARGIN = 0.95


class ZteReadout(SequenceModule):
    """One view of a continuous-gradient ZTE shell, and the step onto the next.

    The readout gradient is already at full amplitude when the pulse fires, so
    encoding begins at the pulse and the spoke runs from the centre of k-space
    outward. It never returns to zero: after the acquisition it slews straight
    to the next view's direction, so a whole shell costs one ramp up and one
    ramp down.

    Every view is the same two blocks -- pulse and hold, then acquire and turn
    -- under a rotation of its own, one designed transition serving the lot,
    which needs consecutive views to be a **constant angle** apart. Leaving
    ``directions`` unset deals a Nyquist-matched sphere into ``n_shots``
    congruent shells that satisfy this; supplying one has it measured, and
    refused if its step wanders::

        zte = design.ZteReadout(system, hard.rf, fov=0.24, matrix=192)

        for shot in zte.shot_rotations:
            turns = [pp.make_rotation(Rotation.from_matrix(shot @ turn))
                     for turn in zte.view_rotations]
            seq.add_block(*zte.g_ramp, turns[0])
            for view, turn in enumerate(turns):
                last = view == len(turns) - 1
                seq.add_block(zte.rf, *zte.g_hold, turn)
                seq.add_block(zte.adc, *(zte.g_end if last else zte.g_read), turn)

    The centre of k-space is not acquired. Transmit ringdown and receiver dead
    time run into the spoke, and the samples that fall inside them are dropped
    rather than squeezed: ``n_missing`` of them, reported so a reconstruction
    can fill the gap. This module does no filling of its own.

    Attributes
    ----------
    rf : RfEvent
        The pulse, delayed by the transmit dead time. Non-selective, and short:
        it plays on a gradient, so its bandwidth has to cover the whole spoke.
    g_ramp : list of GradEvent
        Zero to the first view's direction, played once before a shell.
    g_hold : list of GradEvent
        The plateau the pulse runs on, held through the dead-time gap.
    g_read : list of GradEvent
        The plateau under the acquisition, then the slew onto the next view.
        One event per axis, so a rotation has all three to turn.
    g_end : list of GradEvent
        The same, slewing to zero instead, for the last view of a shell.
    adc : AdcEvent
        The acquisition, delayed past the gap.
    adc_labels : LabelSetEvent or list of LabelSetEvent
        One per name in ``labels``; a bare event when there is one.
    view_rotations : numpy.ndarray
        ``(n_views, 3, 3)``, the rotation that carries the designed view -- a
        spoke along x, stepping into the x-y plane -- onto each view of
        ``directions``.
    shot_rotations : numpy.ndarray
        ``(n_shots, 3, 3)``, the turn about ``z`` that puts the shell where
        each shot samples. Published only when the module generated the shell;
        a supplied ordering brings its own.
    directions : numpy.ndarray
        ``(n_views, 3)``, the shell the rotations were solved for.
    step_rad : float
        Angle between consecutive views, and so the slew each transition asks
        for. A generated shell fixes it at ``arccos(1 - 2 / (n_views - 1))``,
        the polar gap at the pole; there is no separate knob for it.
    tr : float
        Pulse centre to pulse centre (s).
    view_duration : float
        Plateau held for one view, before the transition (s).
    n_samples, n_missing, n_nominal : int
        Samples acquired, lost to the gap, and the full half-spoke.
    gap : float
        Pulse centre to the first sample (s).
    bandwidth_hz : float
        Achieved receiver bandwidth.
    delta_k : float
        Radial k-space step (1/m).
    kmax : float
        Half-spoke extent (1/m).

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    rf : RfEvent
        A non-selective pulse, short enough that its bandwidth spans the spoke.
    fov : float
        Isotropic field of view (m).
    matrix : int
        Isotropic matrix size.
    directions : array_like, optional
        ``(n_views, 3)`` unit spoke directions, consecutive views a constant
        angle apart. Supplying one silences the four generator arguments
        below; the default asks
        :func:`~pulserver.pypulseq.calc_projection_shell` for a shell.
    n_views : int, optional
        Spokes in one shell. Defaults to a Nyquist-matched sphere,
        ``ceil(pi * matrix ** 2)``, split evenly between the shots -- so
        raising ``n_shots`` shortens the shell rather than adding spokes.
    n_shots : int, optional
        Shells the sphere is split into, each the same one turned about ``z``
        by ``2 * pi / n_shots``. It sets the segment length, and playing only
        some of the shots is angular undersampling. The default balances the
        two spacings against each other; see ``step_rad``.
    scheme : {'spiral', 'meridian'}, optional
        Shape of the shell. See
        :func:`~pulserver.pypulseq.calc_projection_shell`.
    oversampling : float, optional
        Radial oversampling: a finer ``delta_k`` along the same spoke.
    readout_bandwidth_hz : float, optional
        Requested receiver bandwidth. Read ``bandwidth_hz`` for what the two
        rasters allowed. It sets the gradient amplitude too, the spoke being
        traversed at one sample per ``delta_k``.
    tr : float, optional
        Pulse centre to pulse centre (s). ``None`` is as short as the slew
        between the two widest-apart views allows.
    dead_time_s : float, optional
        Receiver dead time after the pulse. Defaults to ``system.adc_dead_time``;
        transmit ringdown is added on top either way.
    labels : sequence of str, optional
        Counters emitted on the acquisition block.

    Raises
    ------
    ValueError
        If a count is out of range, the pulse is too long for the dwell, the
        gap swallows the whole spoke, the directions do not step by a constant
        angle, or the requested TR is shorter than the transition needs.

    Examples
    --------
    >>> import pulserver.design as design
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts()
    >>> hard = design.NonSelectiveExcitation(system, 4.0, duration_s=10e-6)
    >>> zte = design.ZteReadout(
    ...     system, hard.rf, fov=0.24, matrix=96, n_views=64, n_shots=256
    ... )
    >>> zte.view_rotations.shape, zte.shot_rotations.shape
    ((64, 3, 3), (256, 3, 3))

    The samples the dead time costs are dropped, not compressed:

    >>> zte.n_samples + zte.n_missing == zte.n_nominal
    True
    """

    def init_module(
        self,
        system: pp.Opts,
        rf: Any,
        *,
        fov: float,
        matrix: int,
        directions: Any = None,
        n_views: int | None = None,
        n_shots: int | None = None,
        scheme: str = "spiral",
        oversampling: float = 2.0,
        readout_bandwidth_hz: float = 62.5e3,
        tr: float | None = None,
        dead_time_s: float | None = None,
        labels: tuple[str, ...] | None = None,
    ) -> None:
        if fov <= 0 or int(matrix) < 2:
            raise ValueError("fov must be positive and matrix must be >= 2")
        if oversampling < 1.0:
            raise ValueError("oversampling must be >= 1")
        if readout_bandwidth_hz <= 0:
            raise ValueError("readout_bandwidth_hz must be positive")

        shot_rotations = None
        if directions is None:
            # Rings hold the shots and the shell walks across them, so the two
            # spacings are reciprocal: ring to ring goes as 1 / n_views, within
            # a ring as 1 / n_shots. They agree at the equator when
            # n_shots = pi * n_views, which with a Nyquist-matched sphere over
            # the whole set puts n_views at the matrix size.
            if n_shots is None:
                n_shots = max(1, math.ceil(np.pi * (int(matrix) - 1)))
            total = math.ceil(np.pi * int(matrix) ** 2)
            n_views = (
                max(3, -(-total // int(n_shots))) if n_views is None else int(n_views)
            )
            directions, shot_rotations = pp.calc_projection_shell(
                n_views, n_shots, scheme=scheme
            )
        directions = _unit(directions)
        view_rotations, step_rad = _view_rotations(directions)
        dead_time_s = (
            system.adc_dead_time if dead_time_s is None else float(dead_time_s)
        )
        if dead_time_s < system.adc_dead_time:
            raise ValueError(
                f"dead_time_s must be at least the receiver's own "
                f"{system.adc_dead_time * 1e6:.1f} us, got {dead_time_s * 1e6:.1f} us"
            )

        # A spoke is a half line, so the sample count is half a matrix, and the
        # dwell fixes the amplitude: one delta_k of k per sample.
        delta_k = 1.0 / (oversampling * fov)
        kmax = 0.5 * int(matrix) / fov
        n_nominal = max(1, round(kmax / delta_k))
        dwell, _ = pp.calc_adc_timing(
            n_nominal,
            1.0 / readout_bandwidth_hz,
            grad_raster_time=system.grad_raster_time,
            adc_raster_time=system.adc_raster_time,
            min_readout_duration=kmax / (_READOUT_GRAD_MARGIN * system.max_grad),
        )
        amplitude = delta_k / dwell

        if float(rf.shape_dur) >= 2.0 * dwell - 1e-12:
            raise ValueError(
                f"the pulse is {float(rf.shape_dur) * 1e6:.3f} us long, which is not short "
                f"beside a {dwell * 1e6:.3f} us dwell: a ZTE pulse plays on the readout "
                f"gradient, so it has to be brief enough to excite the whole spoke"
            )

        # Encoding starts at the pulse centre, and the acquisition cannot start
        # until the transmit has rung down and the receiver has woken. The
        # block boundary goes at the end of the ringdown and the receiver's own
        # dead time is the ADC's delay, so neither pays for the other; anything
        # further the caller asked for is held before the boundary.
        rf.delay = max(float(rf.delay), system.rf_dead_time)
        rf_center = float(rf.delay) + float(rf.center)
        adc_delay = system.adc_dead_time
        hold_span = pp.ceil_to_raster(
            rf_center
            + float(rf.shape_dur)
            - float(rf.center)
            + float(rf.ringdown_time)
            + dead_time_s
            - adc_delay,
            system.grad_raster_time,
        )
        # A sample reports the centre of its dwell, so the first one is half a
        # dwell into the window.
        gap = hold_span - rf_center + adc_delay + 0.5 * dwell
        n_samples = math.ceil(n_nominal - gap * amplitude / delta_k)
        if n_samples < 1:
            raise ValueError(
                f"the {gap * 1e6:.1f} us gap after the pulse already carries k past the "
                f"{kmax:.1f} /m edge of the spoke, so there is nothing left to acquire; "
                f"raise readout_bandwidth_hz or shorten the pulse"
            )
        n_missing = n_nominal - n_samples
        adc = pp.make_adc(
            num_samples=n_samples, dwell=dwell, delay=adc_delay, system=system
        )
        read_span = pp.ceil_to_raster(
            adc_delay + n_samples * dwell, system.grad_raster_time
        )

        # One transition serves every view, so it is designed for the slew they
        # all ask for -- the step being constant -- and the ramp to zero, which
        # is the longest of the lot, closes the shell.
        start = amplitude * np.array([1.0, 0.0, 0.0])
        end = amplitude * np.array([np.cos(step_rad), np.sin(step_rad), 0.0])
        step_span = _slew_span(system, end - start)
        ramp_span = _slew_span(system, start)

        tr_min = pp.ceil_to_raster(
            hold_span + read_span + max(step_span, ramp_span),
            system.block_duration_raster,
        )
        tr_delay = solve_delay(tr, tr_min, "TR", system)
        tail = tr_min + tr_delay - hold_span - read_span

        g_ramp = _gradients(system, [0.0, ramp_span], [np.zeros(3), start])
        g_hold = _gradients(system, [0.0, hold_span], [start, start])
        times = [0.0, read_span, read_span + step_span]
        amplitudes = [start, start, end]
        if tail > step_span + 1e-12:
            times.append(read_span + tail)
            amplitudes.append(end)
        g_read = _gradients(system, times, amplitudes)
        g_end = _gradients(
            system, [0.0, read_span, read_span + ramp_span], [start, start, np.zeros(3)]
        )

        adc_labels = [
            pp.make_label(type="SET", label=name, value=0) for name in labels or ()
        ]

        # Two views: the shortest thing that starts at rest, plays a transition
        # and comes back to rest, so the module traces a real pair of spokes.
        self.seq = pp.Sequence(system)
        self.seq.add_block(*g_ramp)
        self.seq.add_block(rf, *g_hold)
        self.seq.add_block(adc, *g_read, *adc_labels)
        self.seq.add_block(rf, *g_hold, _turn(step_rad))
        self.seq.add_block(adc, *g_end, _turn(step_rad), *adc_labels)

        self.register(view_rotations=view_rotations, directions=directions)
        if shot_rotations is not None:
            self.register(shot_rotations=shot_rotations)
        self.center = ramp_span + rf_center
        self.tr = hold_span + read_span + tail
        self.view_duration = hold_span + read_span
        self.step_rad = step_rad
        self.n_samples = n_samples
        self.n_missing = n_missing
        self.n_nominal = n_nominal
        self.gap = gap
        self.bandwidth_hz = 1.0 / dwell
        self.delta_k = delta_k
        self.kmax = kmax
        self.gradient_amplitude = amplitude


def _unit(directions: Any) -> np.ndarray:
    """The directions as unit rows, read-only."""
    values = np.asarray(directions, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 2:
        raise ValueError(
            "directions must have shape (views, 3) with at least two views"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("directions must be finite")
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms == 0.0):
        raise ValueError("directions must be non-zero")
    unit = np.array(values / norms[:, None])
    unit.setflags(write=False)
    return unit


def _view_rotations(directions: np.ndarray, tolerance: float = 1e-8):
    """One rotation per view, carrying the designed transition onto that view's.

    A view is played as view 0's waveform turned, so what is needed is a
    rotation taking the *pair* ``(d0, d1)`` onto ``(dk, dk+1)``. One exists
    exactly when the two pairs subtend the same angle -- when consecutive views
    are a constant angle apart. That is far weaker than a circle: any
    constant-step walk on the sphere qualifies. The last view has no successor
    and only slews to zero, so any rotation placing ``d0`` on it will do.
    """
    from scipy.spatial.transform import Rotation

    turning = np.array(directions, dtype=float)
    cosines = np.clip(np.sum(turning[:-1] * turning[1:], axis=1), -1.0, 1.0)
    steps = np.arccos(cosines)
    if float(np.ptp(steps)) > tolerance:
        raise ValueError(
            f"consecutive views step by {np.degrees(steps.min()):.3f} to "
            f"{np.degrees(steps.max()):.3f} degrees, so no single designed transition can "
            f"carry one view onto the next; use calc_projection_shell, whose orderings step "
            f"by a constant angle"
        )
    if np.any(np.abs(cosines) >= 1.0 - 1e-12):
        raise ValueError("consecutive views must be neither coincident nor antipodal")

    # The reference is the frame the module designs in -- a view along x and a
    # step of the same angle in the x-y plane -- not the first pair of the
    # shell, because it is that designed waveform the rotations have to carry.
    step = float(steps[0])
    designed = _pair_frame(
        np.array([1.0, 0.0, 0.0]), np.array([np.cos(step), np.sin(step), 0.0])
    ).T
    rotations = [
        _pair_frame(turning[index], turning[index + 1]) @ designed
        for index in range(len(turning) - 1)
    ]
    rotations.append(_minimal_turn(np.array([1.0, 0.0, 0.0]), turning[-1]).as_matrix())
    matrices = np.asarray(
        [Rotation.from_matrix(turn).as_matrix() for turn in rotations]
    )
    matrices.setflags(write=False)
    return matrices, step


def _pair_frame(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Orthonormal frame of an ordered pair of distinct unit directions."""
    normal = np.cross(first, second)
    normal = normal / np.linalg.norm(normal)
    return np.column_stack((first, np.cross(normal, first), normal))


def _minimal_turn(start: np.ndarray, end: np.ndarray):
    """Smallest rotation carrying unit vector ``start`` onto unit vector ``end``."""
    from scipy.spatial.transform import Rotation

    cross = np.cross(start, end)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.dot(start, end))
    if sine <= 1e-12:
        if cosine > 0.0:
            return Rotation.identity()
        basis = np.eye(3)[int(np.argmin(np.abs(start)))]
        axis = np.cross(start, basis)
        return Rotation.from_rotvec(np.pi * axis / np.linalg.norm(axis))
    return Rotation.from_rotvec(np.arctan2(sine, cosine) * cross / sine)


def _turn(step_rad: float):
    """The rotation event that carries the designed view onto the next one."""
    from scipy.spatial.transform import Rotation

    return pp.make_rotation(Rotation.from_euler("z", step_rad))


def _slew_span(system: pp.Opts, delta: np.ndarray) -> float:
    """Time to change the gradient vector by ``delta`` within the slew limit."""
    return max(
        system.grad_raster_time,
        pp.ceil_to_raster(
            float(np.linalg.norm(delta)) / system.max_slew, system.grad_raster_time
        ),
    )


def _gradients(system: pp.Opts, times, amplitudes) -> list:
    """One extended trapezoid per axis, every axis present so a rotation can turn it."""
    amplitudes = np.asarray(amplitudes, dtype=float)
    return [
        pp.make_extended_trapezoid(
            channel=axis,
            times=np.asarray(times, dtype=float),
            amplitudes=amplitudes[:, index],
            system=system,
        )
        for index, axis in enumerate(AXES)
    ]
