"""Zero-echo-time readouts: excite on the readout gradient, acquire from k = 0 out."""

from __future__ import annotations

__all__ = ["ZteReadout"]

import math
from itertools import pairwise
from typing import Any

import numpy as np

from ... import pypulseq as pp
from .._module import SequenceModule
from ._common import AXES, solve_delay

#: Fraction of ``max_grad`` a spoke may reach.
_READOUT_GRAD_MARGIN = 0.95


class ZteReadout(SequenceModule):
    """A whole shell of a continuous-gradient ZTE, ramp up to ramp down.

    Two things define the family, and the module is laid out to give both.
    The readout gradient is already at full amplitude when the pulse fires, so
    encoding begins at the pulse and every spoke runs from the centre of
    k-space outward -- the echo time is the dead time, not a design choice.
    And the gradient never returns to zero between spokes: after each
    acquisition it slews straight onto the next direction, so a shell costs one
    ramp up and one ramp down however many views it holds. A gradient that
    only ever *turns* is what makes a well-designed ZTE quiet.

    The shell is therefore one continuous waveform, written out view by view::

        zte = design.ZteReadout(system, hard.rf, fov=0.24, matrix=192)

        for shot in zte.shot_rotations:
            turn = pp.make_rotation(Rotation.from_matrix(shot))
            seq.add_block(*zte.g_ramp, turn)
            for view in range(len(zte.directions)):
                seq.add_block(zte.rf, *zte.g_hold[view], turn)
                seq.add_block(zte.adc, *zte.g_read[view], turn)

    A view is two blocks -- pulse and hold, then acquire and turn -- because a
    block carries at most one of an RF and an ADC. It costs nothing: gradients
    need not reach zero at a block boundary, so the plateau runs through and
    the whole shell is one segment.

    **The only rotation is the shot.** A generated shell runs pole to pole, so
    turning it about ``z`` by ``2 * pi / n_shots`` leaves its ends where they
    were and slides every intermediate spoke onto the azimuthal gaps the shell
    left behind. ``n_shots`` congruent shells then cover the sphere, one
    ``ROTATIONS`` extension each, and the waveform memory holds one shell
    rather than the whole sphere. It is also what keeps a shell short enough
    to fit: raising ``n_shots`` divides a fixed sphere into more, shorter
    segments rather than acquiring more spokes.

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
        Zero to the first view's direction, played once at the head of a shell.
        One event per axis.
    g_hold : list of list of GradEvent
        Per view, the plateau the pulse runs on, held through the dead-time gap.
    g_read : list of list of GradEvent
        Per view, the plateau under the acquisition and then the turn onto the
        next view. The last entry slews to zero instead, closing the shell.
    g_dummy : list of GradEvent
        The same span as a ``g_read``, held at the first view's direction. A
        view played on this one drives the magnetisation without moving the
        gradient, which is what a ZTE preparation wants: the pulse train has to
        settle, the orbit does not.
    adc : AdcEvent
        The acquisition, delayed past the gap.
    adc_labels : LabelSetEvent or list of LabelSetEvent
        One per name in ``labels``; a bare event when there is one.
    shot_rotations : numpy.ndarray
        ``(n_shots, 3, 3)``, the turn about ``z`` that puts the shell where
        each shot samples. Published only when the module generated the shell;
        a supplied ordering brings its own.
    directions : numpy.ndarray
        ``(n_views, 3)``, the spoke directions of one shell, in play order.
    step_rad : float
        The widest angle between consecutive views. The turn between views is
        budgeted for it, so a shell whose steps are equal wastes none of it.
    tr : float
        Pulse centre to pulse centre (s).
    view_duration : float
        Plateau held for one view, before the turn (s).
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
        ``(n_views, 3)`` unit spoke directions, in the order the shell walks
        them. Supplying one silences the four generator arguments below; the
        default asks :func:`~pulserver.pypulseq.calc_projection_shell` for a
        shell. An ordering whose steps vary is accepted, but every turn is
        given the widest one's slot, so the repetition pays for the worst step
        throughout.
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
        Pulse centre to pulse centre (s). ``None`` is as short as the widest
        turn allows. A longer one is spent slewing more gently, not waiting.
    dead_time_s : float, optional
        Receiver dead time after the pulse. Defaults to ``system.adc_dead_time``;
        transmit ringdown is added on top either way.
    labels : sequence of str, optional
        Counters emitted on the acquisition block.

    Raises
    ------
    ValueError
        If a count is out of range, the pulse is too long for the dwell, the
        gap swallows the whole spoke, two consecutive views coincide, or the
        requested TR is shorter than the turn needs.

    Examples
    --------
    >>> import pulserver.design as design
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts()
    >>> hard = design.NonSelectiveExcitation(system, 4.0, duration_s=10e-6)
    >>> zte = design.ZteReadout(
    ...     system, hard.rf, fov=0.24, matrix=96, n_views=64, n_shots=256
    ... )
    >>> len(zte.g_read), zte.shot_rotations.shape
    (64, (256, 3, 3))

    The samples the dead time costs are dropped, not compressed:

    >>> zte.n_samples + zte.n_missing == zte.n_nominal
    True

    The module is the shell, so what it plays is the koosh ball one shot
    acquires -- every spoke leaving the centre, their ends walking pole to
    pole:

    .. plot::

       import pulserver.design as design
       import pulserver.pypulseq as pp

       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       hard = design.NonSelectiveExcitation(system, 4.0, duration_s=10e-6)
       zte = design.ZteReadout(
           system, hard.rf, fov=0.24, matrix=48, n_views=24, n_shots=1
       )
       zte.plot_kspace(plot_now=False)
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

        # Where the gradient sits under each view, and the chord it crosses to
        # reach the next one. Every turn gets the same slot -- the repetition
        # is what the magnetisation sees, so it has to be one number -- and the
        # slot is the widest chord's. Closing the shell is the long one: from
        # full amplitude down to zero.
        vertices = amplitude * directions
        chords = np.linalg.norm(np.diff(vertices, axis=0), axis=1)
        if not np.all(chords > 0.0):
            raise ValueError("consecutive views must not coincide")
        widest_turn = _slew_span(system, float(chords.max()))
        ramp_span = _slew_span(system, amplitude)

        tr_min = pp.ceil_to_raster(
            hold_span + read_span + widest_turn, system.block_duration_raster
        )
        view_span = tr_min + solve_delay(tr, tr_min, "TR", system)
        # Whatever the repetition leaves over is spent turning more slowly, not
        # sitting still: a gentler slew is a quieter one, and nothing is
        # encoded between the acquisitions.
        turn_span = view_span - hold_span - read_span
        close_span = max(ramp_span, turn_span)

        g_ramp = _gradients(system, [0.0, ramp_span], [np.zeros(3), vertices[0]])
        g_hold = [
            _gradients(system, [0.0, hold_span], [vertex, vertex])
            for vertex in vertices
        ]
        g_read = [
            _gradients(
                system,
                [0.0, read_span, read_span + turn_span],
                [start, start, end],
            )
            for start, end in pairwise(vertices)
        ]
        g_read.append(
            _gradients(
                system,
                [0.0, read_span, read_span + close_span],
                [vertices[-1], vertices[-1], np.zeros(3)],
            )
        )
        g_dummy = _gradients(
            system, [0.0, read_span + turn_span], [vertices[0], vertices[0]]
        )

        adc_labels = [
            pp.make_label(type="SET", label=name, value=0) for name in labels or ()
        ]

        self.seq = pp.Sequence(system)
        self.seq.add_block(*g_ramp)
        for view in range(len(directions)):
            self.seq.add_block(rf, *g_hold[view])
            self.seq.add_block(adc, *g_read[view], *adc_labels)

        self.register(
            g_hold=g_hold, g_read=g_read, g_dummy=g_dummy, directions=directions
        )
        if shot_rotations is not None:
            self.register(shot_rotations=shot_rotations)
        self.center = ramp_span + rf_center
        self.tr = view_span
        self.view_duration = hold_span + read_span
        self.step_rad = float(
            np.arccos(
                np.clip(np.sum(directions[:-1] * directions[1:], axis=1), -1, 1)
            ).max()
        )
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


def _slew_span(system: pp.Opts, delta: float) -> float:
    """Time to change the gradient vector by ``delta`` within the slew limit."""
    return max(
        system.grad_raster_time,
        pp.ceil_to_raster(delta / system.max_slew, system.grad_raster_time),
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
