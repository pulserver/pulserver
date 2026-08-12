"""Non-Cartesian tilt support and acquisition ordering."""

from __future__ import annotations

import math

import numpy as np

from ...pypulseq._angles import (  # noqa: F401
    _accumulated,
    _generalized_fibonacci,
    calc_golden_angles,
    calc_raga_angles,
    calc_tiny_golden_angles,
    calc_uniform_angles,
)
from ._axes import EncodingAxis
from ._scanloop import ScanLoop

_PHI = (1.0 + math.sqrt(5.0)) / 2.0

# A non-Cartesian loop encodes with rotations, so its counter is the view
# number -- the position index, not the angle. ``LIN`` is where a stack or
# projection reconstruction expects to find it.
_ANGLE_AXES = (EncodingAxis("LIN", kind="angle"),)
_DIRECTION_AXES = (EncodingAxis("LIN", kind="direction"),)


def _segments(indices, length):
    return tuple(np.asarray(indices[i : i + length], dtype=np.intp) for i in range(0, len(indices), length))




def make_radial_tilt(
    n_spokes,
    *,
    scheme="linear",
    period=np.pi,
    increment=None,
    tiny_index=1,
    approximation_order=13,
    segment_length=1,
):
    """Generate 2D radial spoke angles and their acquisition segments.

    The returned pattern's ``positions`` hold one angle (radians) per spoke and
    its ``order`` groups spokes into contiguous segments — the natural
    continuous-gradient / inner-train boundary for a radial readout.

    ``"linear"`` sweeps ``period`` uniformly. ``"golden"`` and
    ``"tiny_golden"`` accumulate an irrational increment so any temporal window
    stays near-uniformly distributed. ``"raga"`` is the rational
    approximation of the golden increment: its angular support is *finite and
    equidistant* (Fibonacci-sized) while the temporal index order stays
    golden-like, which is what makes it exactly reproducible bin-to-bin.

    Parameters
    ----------
    n_spokes : int
        Number of spokes.
    scheme : {'linear', 'golden', 'tiny_golden', 'raga'}, optional
        Angular increment scheme.
    period : float, optional
        Angular period; ``np.pi`` (default) for a half-circle acquisition with
        symmetric spokes, ``2 * np.pi`` for full-circle / partial-echo spokes.
    increment : float or None, optional
        Explicit increment (rad) overriding the ``"linear"`` default of
        ``period / n_spokes``.
    tiny_index : int, optional
        Tiny-golden index ``N`` (increment ``period / (phi + N - 1)``); also
        the RAGA tiny index.
    approximation_order : int, optional
        RAGA Fibonacci order; the angular support has
        ``fib(approximation_order, tiny_index)`` distinct angles.
    segment_length : int, optional
        Spokes per acquisition segment.

    Returns
    -------
    ScanLoop
        ``positions`` of shape ``(n_spokes, 1)`` in radians; one entry in ``shots`` per
        segment.

    Examples
    --------
    >>> import numpy as np
    >>> from pulserver.design._lowlevel import make_radial_tilt
    >>> uniform = make_radial_tilt(8)
    >>> np.rad2deg(uniform.positions[:, 0]).round(1)
    array([  0. ,  22.5,  45. ,  67.5,  90. , 112.5, 135. , 157.5])
    >>> golden = make_radial_tilt(400, scheme="golden", segment_length=100)
    >>> golden.n_shots
    4

    .. plot::
       :include-source: false

       import numpy as np
       import matplotlib.pyplot as plt
       from pulserver.design._lowlevel import make_radial_tilt
       fig, axes = plt.subplots(1, 2, figsize=(8, 4), subplot_kw={"polar": True})
       for ax, scheme in zip(axes, ("linear", "golden")):
           angles = make_radial_tilt(21, scheme=scheme).positions[:, 0]
           for i, a in enumerate(angles):
               ax.plot([a, a], [-1, 1], color=plt.cm.viridis(i / 20))
           ax.set_yticks([]); ax.set_title(f"{scheme}, 21 spokes")
       fig.tight_layout()

    References
    ----------
    Scholand et al., RAGA sampling, DOI ``10.1002/mrm.30254``.

    See Also
    --------
    pulserver.ScanLoop.to_rotations : turn the locations into block rotations.
    """
    n_spokes, segment_length = int(n_spokes), int(segment_length)
    period = float(period)
    if n_spokes < 0 or segment_length <= 0 or not np.isfinite(period) or period <= 0:
        raise ValueError("n_spokes must be nonnegative and period/segment_length positive")
    if scheme == "raga":
        tiny_index, approximation_order = int(tiny_index), int(approximation_order)
        if tiny_index < 1 or approximation_order < 2:
            raise ValueError("tiny_index must be >= 1 and approximation_order >= 2")
        support_size = _generalized_fibonacci(approximation_order, tiny_index)
        step = _generalized_fibonacci(approximation_order - 1, 1)
        support = (np.arange(support_size) * period / support_size)[:, None]
        chronological = (np.arange(n_spokes, dtype=np.intp) * step) % support_size
    else:
        if scheme == "linear":
            step = (
                period / n_spokes
                if increment is None and n_spokes
                else (0.0 if increment is None else float(increment))
            )
        elif scheme == "golden":
            step = period / _PHI
        elif scheme == "tiny_golden":
            tiny_index = int(tiny_index)
            if tiny_index < 1:
                raise ValueError("tiny_index must be >= 1")
            step = period / (_PHI + tiny_index - 1)
        else:
            raise ValueError("scheme must be linear, golden, tiny_golden, or raga")
        if not np.isfinite(step):
            raise ValueError("increment must be finite")
        support = np.mod(np.arange(n_spokes) * step, period)[:, None]
        chronological = np.arange(n_spokes, dtype=np.intp)
    return ScanLoop(support, _segments(chronological, segment_length), None, _ANGLE_AXES)












def _directions(z, azimuth):
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack((radius * np.cos(azimuth), radius * np.sin(azimuth), z))


def make_golden_means_3d_tilt(n_spokes, *, segment_length=1):
    """Generate 3D centre-out spoke directions from the 2D golden means.

    Extends the golden-angle idea to the sphere: polar and azimuthal indices
    advance by two mutually irrational means, so *any* contiguous window of
    spokes covers the sphere near-uniformly. That makes it the default for
    self-navigated or retrospectively binned 3D radial acquisitions, where the
    number of spokes per bin is not known at design time.

    Parameters
    ----------
    n_spokes : int
        Number of spokes.
    segment_length : int, optional
        Spokes per acquisition segment (one entry in ``shots`` each).

    Returns
    -------
    ScanLoop
        ``positions`` of shape ``(n_spokes, 3)``: unit direction vectors.

    Examples
    --------
    >>> from pulserver.design._lowlevel import make_golden_means_3d_tilt
    >>> pattern = make_golden_means_3d_tilt(6, segment_length=3)
    >>> pattern.positions.shape, pattern.n_shots
    ((6, 3), 2)

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       from pulserver.design._lowlevel import make_golden_means_3d_tilt
       d = make_golden_means_3d_tilt(300).positions
       fig = plt.figure(figsize=(5, 5))
       ax = fig.add_subplot(projection="3d")
       ax.scatter(d[:, 0], d[:, 1], d[:, 2], s=6)
       ax.set_box_aspect((1, 1, 1))
       ax.set_title("make_golden_means_3d_tilt(300)")

    References
    ----------
    Chan et al., temporal stability of 3D golden-means radial, DOI
    ``10.1002/mrm.22732``.

    See Also
    --------
    make_spiral_phyllotaxis_tilt : interleaved alternative with smooth intra-shot paths.
    pulserver.ScanLoop.to_rotations : convert the locations to block rotations.
    """
    n_spokes, segment_length = int(n_spokes), int(segment_length)
    if n_spokes < 0 or segment_length <= 0:
        raise ValueError("n_spokes must be nonnegative and segment_length positive")
    m = np.arange(n_spokes, dtype=float)
    z = 2.0 * np.mod(m * 0.465571231876768, 1.0) - 1.0
    azimuth = 2.0 * np.pi * np.mod(m * 0.682327803828019, 1.0)
    support = _directions(z, azimuth)
    return ScanLoop(support, _segments(np.arange(n_spokes, dtype=np.intp), segment_length), None, _DIRECTION_AXES)


def _is_fibonacci(value):
    return any(
        math.isqrt(candidate) ** 2 == candidate for candidate in (5 * value * value + 4, 5 * value * value - 4)
    )


def make_spiral_phyllotaxis_tilt(n_spokes, n_interleaves, *, require_fibonacci=True):
    """Generate 3D spoke directions on a spiral phyllotaxis, interleaved.

    Directions follow a single pole-to-pole spiral (the sunflower-seed
    arrangement), then are dealt round-robin into ``n_interleaves`` shots.
    Each shot therefore traverses the sphere smoothly — minimising eddy-current
    and steady-state disruption within a shot — while the union stays uniform.

    ``n_interleaves`` must be a Fibonacci number for the interleaved subsets to
    remain uniform; pass ``require_fibonacci=False`` to bypass the check
    knowingly.

    Parameters
    ----------
    n_spokes : int
        Total number of spokes; must be divisible by ``n_interleaves``.
    n_interleaves : int
        Number of shots to deal the spiral into.
    require_fibonacci : bool, optional
        Enforce that ``n_interleaves`` is a Fibonacci number (default True).

    Returns
    -------
    ScanLoop
        ``positions`` of shape ``(n_spokes, 3)``: unit direction vectors; one
        entry in ``shots`` per interleaf.

    Examples
    --------
    >>> from pulserver.design._lowlevel import make_spiral_phyllotaxis_tilt
    >>> pattern = make_spiral_phyllotaxis_tilt(8, 2)
    >>> pattern.n_shots, pattern.shots[0]
    (2, array([0, 2, 4, 6]))

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       from pulserver.design._lowlevel import make_spiral_phyllotaxis_tilt
       pattern = make_spiral_phyllotaxis_tilt(377, 13)
       fig = plt.figure(figsize=(5, 5))
       ax = fig.add_subplot(projection="3d")
       for shot in range(3):
           d = pattern[shot]
           ax.plot(d[:, 0], d[:, 1], d[:, 2], marker="o", ms=3, lw=0.6)
       ax.set_box_aspect((1, 1, 1))
       ax.set_title("make_spiral_phyllotaxis_tilt(377, 13): first 3 shots")

    References
    ----------
    Piccini et al., spiral phyllotaxis, DOI ``10.1002/mrm.22898``.

    See Also
    --------
    make_golden_means_3d_tilt : uniform in any window, at the cost of shot smoothness.
    """
    n_spokes, n_interleaves = int(n_spokes), int(n_interleaves)
    if n_spokes <= 0 or n_interleaves <= 0 or n_spokes % n_interleaves:
        raise ValueError("positive n_spokes must be divisible by positive n_interleaves")
    if require_fibonacci and not _is_fibonacci(n_interleaves):
        raise ValueError("n_interleaves must be a Fibonacci number")
    m = np.arange(n_spokes, dtype=float)
    z = np.ones(1) if n_spokes == 1 else 1.0 - 2.0 * m / (n_spokes - 1)
    support = _directions(z, m * np.pi * (3.0 - math.sqrt(5.0)))
    order = tuple(np.arange(j, n_spokes, n_interleaves, dtype=np.intp) for j in range(n_interleaves))
    return ScanLoop(support, order, None, _DIRECTION_AXES)


def _constant_step_spiral(views: int, step_deg: float | None) -> np.ndarray:
    """A pole-to-pole spiral shell whose consecutive views are a constant angle apart.

    The polar ladder is fixed first, at equal-area ring centres, and each
    azimuth increment is then *solved* rather than chosen, from

    ``cos(step) = cos(t_k) cos(t_k+1) + sin(t_k) sin(t_k+1) cos(dphi)``

    so every consecutive pair subtends exactly ``step``. That is the condition
    :func:`~pulserver.design._readout.zte._view_rotations` needs, and it is
    what lets a whole shell be played from one designed transition.

    ``step`` is a genuine knob rather than a detail: it cannot go below the
    widest polar gap (near the poles, where equal-area rings are furthest apart
    in angle), and raising it above that trades a longer slew per view for more
    azimuthal winding. The default takes the smallest feasible value, which is
    the gentlest on the gradients.
    """
    views = int(views)
    if views == 1:
        return np.array([[0.0, 0.0, 1.0]])
    z = 1.0 - 2.0 * (np.arange(views) + 0.5) / views
    polar = np.arccos(z)
    widest_gap = float(np.max(np.diff(polar)))
    if step_deg is None:
        step = widest_gap
    else:
        step = math.radians(float(step_deg))
        if step < widest_gap - 1e-12:
            raise ValueError(
                f"step_deg={step_deg} is below the {math.degrees(widest_gap):.3f} deg polar gap this "
                f"{views}-view shell already spans; no azimuth increment can make up the difference"
            )
    if step >= np.pi:
        raise ValueError("step_deg must be below 180 degrees")

    numerator = math.cos(step) - np.cos(polar[:-1]) * np.cos(polar[1:])
    denominator = np.sin(polar[:-1]) * np.sin(polar[1:])
    azimuth = np.concatenate(([0.0], np.cumsum(np.arccos(np.clip(numerator / denominator, -1.0, 1.0)))))
    return _directions(z, azimuth)


def make_rotated_segment_tilt(
    views_per_shot, shots, *, scheme="spiral", step_deg=None, require_fibonacci=True
):
    """Generate one base spoke segment plus the rotations that sweep it over the sphere.

    The other 3D tilt factories hand back every spoke direction as its own
    position, which is the right answer when each spoke is encoded by its own
    gradient. A continuous-gradient readout cannot afford that: an interpreter
    stores one waveform per gradient *definition* and a bounded number of
    shapes per definition, so a scan of N arbitrary directions asks for N
    distinct waveforms and is rejected. Here the whole shot is designed once
    and replayed under a rigid rotation, so the hardware sees a single set of
    waveforms however many shots are played, and the rotation rides along as a
    block extension.

    That trade is only sound if the shots really are congruent — every shot
    must be the base segment moved rigidly, not merely a similar-looking one.
    Every scheme below rotates about ``z`` alone, so the base polar angles are
    shared by every shot: the sphere is covered by ``views_per_shot`` polar
    rings of ``shots`` spokes each, rather than by ``views_per_shot * shots``
    freely placed directions. Coverage within a ring is uniform whatever the
    base does, because the shot rotations are what spread it.

    A second condition applies *inside* the segment, and it is what separates
    the schemes: replaying one designed transition for every view needs
    consecutive views to be a **constant angle** apart. That admits far more
    than a circle — any constant-step walk on the sphere qualifies, a
    pole-to-pole spiral included — but it does rule out orderings whose step
    wanders, phyllotaxis among them.

    Parameters
    ----------
    views_per_shot : int
        Spokes in the base segment — the number of views one shot acquires.
    shots : int
        Number of rotated replays of that segment.
    scheme : {'spiral', 'disk', 'phyllotaxis'}, optional
        ``'spiral'`` (default) walks a pole-to-pole spiral at a constant
        angular step: it spans the full polar range like a phyllotaxis
        interleaf, and being constant-step it can be played from one designed
        transition however long the shell is.
        ``'disk'`` makes the base a full great circle in the ``x``-``z`` plane
        and rotates it about ``z`` over ``[0, pi)``. Also constant-step, and
        the simplest, at the cost of oversampling the poles.
        ``'phyllotaxis'`` deals a Piccini spiral into interleaves. Best
        uniformity of the three, but its angular step wanders, so it *cannot*
        be rotation-encoded and a continuous-gradient readout is limited to a
        short shell by the interpreter's per-definition shape budget.
    step_deg : float, optional
        Angular step for ``'spiral'``. Defaults to the smallest feasible value
        — the widest polar gap of the equal-area ladder — which is the gentlest
        on the gradients; larger values wind further in azimuth per view.
    require_fibonacci : bool, optional
        Enforce that ``shots`` is a Fibonacci number for ``'phyllotaxis'``
        (default True); pass ``False`` to bypass the check knowingly.

    Returns
    -------
    base : ScanLoop
        One shot holding the ``(views_per_shot, 3)`` unit base directions.
    rotations : numpy.ndarray
        ``(shots, 3, 3)`` rotation matrices, one per shot.

    Raises
    ------
    ValueError
        If either count is not positive, ``scheme`` is unknown, or
        ``require_fibonacci`` is set and ``shots`` is not Fibonacci.

    Examples
    --------
    >>> import numpy as np
    >>> from pulserver.design._lowlevel import make_rotated_segment_tilt
    >>> base, rotations = make_rotated_segment_tilt(8, 13)
    >>> base.positions.shape, rotations.shape
    ((8, 3), (13, 3, 3))

    Every shot really is the base moved rigidly, which is what lets one
    waveform set serve them all:

    >>> spokes = np.einsum("sij,vj->svi", rotations, base.positions)
    >>> bool(np.allclose(np.linalg.norm(spokes, axis=-1), 1.0))
    True

    ...and inside a shell the step is constant, which is what lets one designed
    transition serve every view:

    >>> steps = np.arccos(np.clip(np.sum(base.positions[:-1] * base.positions[1:], axis=1), -1, 1))
    >>> bool(np.ptp(steps) < 1e-9)
    True

    See Also
    --------
    make_spiral_phyllotaxis_tilt : freely placed interleaves, one waveform per spoke.
    pulserver.design.make_zte_readout : the readout that consumes the base segment.
    """
    views_per_shot, shots = int(views_per_shot), int(shots)
    if views_per_shot <= 0 or shots <= 0:
        raise ValueError("views_per_shot and shots must be positive")

    if scheme == "spiral":
        base = _constant_step_spiral(views_per_shot, step_deg)
        angles = np.arange(shots, dtype=float) * np.pi * (3.0 - math.sqrt(5.0))
    elif scheme == "phyllotaxis":
        if require_fibonacci and not _is_fibonacci(shots):
            raise ValueError(
                "shots must be a Fibonacci number so the base segment advances in small azimuth "
                "steps; pass require_fibonacci=False to bypass"
            )
        golden = np.pi * (3.0 - math.sqrt(5.0))
        index = np.arange(views_per_shot, dtype=float)
        # Equal-area ring centres: symmetric about the equator, and covering
        # both poles as closely as views_per_shot allows.
        z = 1.0 - 2.0 * (index + 0.5) / views_per_shot
        # Advancing by shots * golden keeps the union of the rotated segments
        # on the same phyllotaxis lattice a single-shot spiral would trace.
        base = _directions(z, index * shots * golden)
        angles = np.arange(shots, dtype=float) * golden
    elif scheme == "disk":
        theta = 2.0 * np.pi * np.arange(views_per_shot, dtype=float) / views_per_shot
        base = np.column_stack((np.cos(theta), np.zeros(views_per_shot), np.sin(theta)))
        angles = np.pi * np.arange(shots, dtype=float) / shots
    else:
        raise ValueError("scheme must be spiral, disk, or phyllotaxis")

    loop = ScanLoop(base, (np.arange(views_per_shot, dtype=np.intp),), None, _DIRECTION_AXES)
    return loop, angles_to_rotations(angles)


def angles_to_rotations(angles):
    """Return one rotation about ``z`` per in-plane spoke angle (radians)."""
    angles = np.asarray(angles, dtype=float).reshape(-1)
    if not np.all(np.isfinite(angles)):
        raise ValueError("angles must be finite")
    cosine, sine = np.cos(angles), np.sin(angles)
    result = np.zeros((len(angles), 3, 3), dtype=float)
    result[:, 0, 0] = cosine
    result[:, 0, 1] = -sine
    result[:, 1, 0] = sine
    result[:, 1, 1] = cosine
    result[:, 2, 2] = 1.0
    return result


def directions_to_rotations(directions, *, reference=(1.0, 0.0, 0.0)):
    """Return the minimal-angle rotation taking ``reference`` onto each direction."""
    directions = np.asarray(directions, dtype=float)
    if directions.ndim == 1:
        directions = directions[None, :]
    reference = np.asarray(reference, dtype=float)
    if directions.ndim != 2 or directions.shape[1] != 3 or reference.shape != (3,):
        raise ValueError("directions must have shape (3,) or (N, 3), and reference shape (3,)")
    if not np.all(np.isfinite(directions)) or not np.all(np.isfinite(reference)):
        raise ValueError("directions and reference must be finite")
    ref_norm = np.linalg.norm(reference)
    norms = np.linalg.norm(directions, axis=1)
    if ref_norm == 0 or np.any(norms == 0):
        raise ValueError("directions and reference must be nonzero")
    ref = reference / ref_norm
    result = np.empty((len(directions), 3, 3), dtype=float)
    identity = np.eye(3)
    for idx, direction in enumerate(directions / norms[:, None]):
        cross = np.cross(ref, direction)
        sine = np.linalg.norm(cross)
        cosine = float(np.dot(ref, direction))
        if sine > 1e-12:
            axis = cross / sine
            skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            result[idx] = identity + sine * skew + (1.0 - cosine) * (skew @ skew)
        elif cosine > 0:
            result[idx] = identity
        else:
            basis = identity[np.argmin(np.abs(identity @ ref))]
            axis = np.cross(ref, basis)
            axis /= np.linalg.norm(axis)
            result[idx] = 2.0 * np.outer(axis, axis) - identity
    return result
