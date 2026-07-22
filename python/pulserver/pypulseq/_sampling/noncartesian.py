"""Non-Cartesian tilt support and acquisition ordering."""

from __future__ import annotations

import math

import numpy as np

from ._pattern import SamplingPattern

_PHI = (1.0 + math.sqrt(5.0)) / 2.0


def _segments(indices, length):
    return tuple(np.asarray(indices[i : i + length], dtype=np.intp) for i in range(0, len(indices), length))


def _generalized_fibonacci(order, index):
    if order == 1:
        return 1
    previous, current = 1, int(index)
    for _ in range(2, order):
        previous, current = current, previous + current
    return current


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

    The returned pattern's ``support`` holds one angle (radians) per spoke and
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
    SamplingPattern
        ``support`` of shape ``(n_spokes, 1)`` in radians; one order entry per
        segment.

    Examples
    --------
    >>> import numpy as np
    >>> from pulserver.pypulseq import make_radial_tilt
    >>> uniform = make_radial_tilt(8)
    >>> np.rad2deg(uniform.support[:, 0]).round(1)
    array([  0. ,  22.5,  45. ,  67.5,  90. , 112.5, 135. , 157.5])
    >>> golden = make_radial_tilt(400, scheme="golden", segment_length=100)
    >>> golden.n_shots
    4

    .. plot::
       :include-source: false

       import numpy as np
       import matplotlib.pyplot as plt
       from pulserver.pypulseq import make_radial_tilt
       fig, axes = plt.subplots(1, 2, figsize=(8, 4), subplot_kw={"polar": True})
       for ax, scheme in zip(axes, ("linear", "golden")):
           angles = make_radial_tilt(21, scheme=scheme).support[:, 0]
           for i, a in enumerate(angles):
               ax.plot([a, a], [-1, 1], color=plt.cm.viridis(i / 20))
           ax.set_yticks([]); ax.set_title(f"{scheme}, 21 spokes")
       fig.tight_layout()

    References
    ----------
    Scholand et al., RAGA sampling, DOI ``10.1002/mrm.30254``.

    See Also
    --------
    pulserver.SamplingPattern.to_rotations : turn the support into block rotations.
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
    return SamplingPattern(support, _segments(chronological, segment_length))


def _accumulated(n: int, step: float) -> np.ndarray:
    n = int(n)
    if n < 0:
        raise ValueError("n must be nonnegative")
    return np.mod(np.arange(n, dtype=float) * float(step), 2.0 * np.pi)


def calc_golden_angles(n: int) -> np.ndarray:
    """Return ``n`` golden-angle spoke rotations, in radians.

    Consecutive spokes advance by the MRI golden angle ``pi / phi`` (111.246
    degrees), so any contiguous temporal window of spokes stays
    near-uniformly distributed over the circle — the property that makes
    golden-angle radial the default for retrospectively binned and
    free-breathing acquisitions.

    A flat angle array rather than a :class:`~pulserver.SamplingPattern`, in
    the full-circle (``2 * pi``) convention: pair it directly with a base
    waveform and :func:`pulserver.pypulseq.make_rotation`. Use
    :func:`make_radial_tilt` when the angular period or the segmentation
    matters.

    Parameters
    ----------
    n : int
        Number of angles.

    Returns
    -------
    numpy.ndarray
        Angles (rad), length ``n``, accumulated modulo ``2 pi``.

    Examples
    --------
    >>> import numpy as np
    >>> import pulserver.pypulseq as pp
    >>> np.rad2deg(pp.calc_golden_angles(4)).round(2)
    array([  0.  , 111.25, 222.49, 333.74])

    Where each increment puts the first 34 spokes — golden angle spreads them in any window, tiny golden angle does the same in smaller steps, RAGA snaps them to a fixed equidistant support:

    .. plot::
       :include-source: false

       import numpy as np
       import matplotlib.pyplot as plt
       import pulserver.pypulseq as pp
       schemes = [
           ("uniform", pp.calc_uniform_angles(34)),
           ("golden", pp.calc_golden_angles(34)),
           ("tiny golden, N=4", pp.calc_tiny_golden_angles(34, index=4)),
           ("RAGA", pp.calc_raga_angles(34, approximation_order=9)),
       ]
       fig, axes = plt.subplots(1, 4, figsize=(11, 3.1), subplot_kw={"polar": True})
       for ax, (name, angles) in zip(axes, schemes):
           for order, angle in enumerate(angles):
               ax.plot([angle, angle + np.pi], [1, 1], lw=1,
                       color=plt.cm.viridis(order / (len(angles) - 1)))
           ax.set_yticks([])
           ax.set_title(name, fontsize=8)
       fig.tight_layout()

    References
    ----------
    Winkelmann et al., golden-ratio profile order, DOI ``10.1109/TMI.2006.885337``.

    See Also
    --------
    calc_tiny_golden_angles : smaller increments with the same uniformity.
    calc_raga_angles : rational, exactly repeatable approximation.
    make_radial_tilt : full spoke tilt schedule with period and segmentation control.
    """
    return _accumulated(n, np.pi / _PHI)


def calc_raga_angles(n: int, *, tiny_index: int = 1, approximation_order: int = 13) -> np.ndarray:
    """Return ``n`` RAGA (rational approximate golden-angle) spoke rotations.

    RAGA replaces the irrational golden increment with the nearest Fibonacci
    ratio, so the angular *support* is finite and exactly equidistant while
    the temporal index order stays golden-like. Every bin of a binned
    reconstruction therefore draws from the same fixed angle set, which is
    what makes RAGA reproducible bin to bin where plain golden angle is not.

    The support holds ``fib(approximation_order, tiny_index)`` distinct
    angles; ``n`` may exceed that, in which case angles repeat.

    Parameters
    ----------
    n : int
        Number of angles.
    tiny_index : int, optional
        Tiny-golden index the rational approximation is built from.
    approximation_order : int, optional
        Fibonacci order; sets the size of the angular support.

    Returns
    -------
    numpy.ndarray
        Angles (rad), length ``n``, drawn from a finite equidistant support.

    Examples
    --------
    >>> import numpy as np
    >>> import pulserver.pypulseq as pp
    >>> angles = pp.calc_raga_angles(1000, approximation_order=8)
    >>> len(np.unique(angles.round(9)))
    21

    References
    ----------
    Scholand et al., RAGA sampling, DOI ``10.1002/mrm.30254``.

    See Also
    --------
    calc_golden_angles : the irrational increment RAGA approximates.
    """
    pattern = make_radial_tilt(
        n,
        scheme="raga",
        period=2.0 * np.pi,
        tiny_index=tiny_index,
        approximation_order=approximation_order,
    )
    return pattern.flatten()[:, 0]


def calc_tiny_golden_angles(n: int, *, index: int = 2) -> np.ndarray:
    """Return ``n`` tiny-golden-angle spoke rotations, in radians.

    The increment ``pi / (phi + index - 1)`` shrinks with ``index`` while
    keeping the golden distribution: ``index=1`` reproduces
    :func:`calc_golden_angles`, higher indices step less far between
    consecutive spokes. Smaller steps mean smaller eddy-current and
    steady-state disruption per view, which is why tiny golden angles are
    preferred for bSSFP and other steady-state radial acquisitions.

    Parameters
    ----------
    n : int
        Number of angles.
    index : int, optional
        Tiny-golden index ``N >= 1`` (default 2).

    Returns
    -------
    numpy.ndarray
        Angles (rad), length ``n``, accumulated modulo ``2 pi``.

    Raises
    ------
    ValueError
        If ``index`` is smaller than 1.

    Examples
    --------
    >>> import numpy as np
    >>> import pulserver.pypulseq as pp
    >>> np.rad2deg(pp.calc_tiny_golden_angles(3, index=2)).round(2)
    array([  0.  ,  68.75, 137.51])

    References
    ----------
    Wundrak et al., tiny golden angles, DOI ``10.1002/mrm.25831``.

    See Also
    --------
    calc_golden_angles : the ``index=1`` case.
    """
    index = int(index)
    if index < 1:
        raise ValueError("index must be >= 1")
    return _accumulated(n, np.pi / (_PHI + index - 1))


def calc_uniform_angles(n: int) -> np.ndarray:
    """Return ``n`` equally spaced full-circle spoke rotations, in radians.

    The uniform counterpart of :func:`calc_golden_angles`, in the same
    flat-array, full-circle convention. Optimal coverage for a *fixed*,
    known-in-advance spoke count — and only then, since any partial window of
    the acquisition leaves an angular gap.

    Parameters
    ----------
    n : int
        Number of angles.

    Returns
    -------
    numpy.ndarray
        Angles (rad), length ``n``, spaced by ``2 * pi / n``.

    Examples
    --------
    >>> import numpy as np
    >>> import pulserver.pypulseq as pp
    >>> np.rad2deg(pp.calc_uniform_angles(4))
    array([  0.,  90., 180., 270.])

    See Also
    --------
    calc_golden_angles : uniform in any temporal window instead.
    """
    n = int(n)
    if n < 0:
        raise ValueError("n must be nonnegative")
    return _accumulated(n, 0.0 if n == 0 else 2.0 * np.pi / n)


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
        Spokes per acquisition segment (one order entry each).

    Returns
    -------
    SamplingPattern
        ``support`` of shape ``(n_spokes, 3)``: unit direction vectors.

    Examples
    --------
    >>> from pulserver.pypulseq import make_golden_means_3d_tilt
    >>> pattern = make_golden_means_3d_tilt(6, segment_length=3)
    >>> pattern.support.shape, pattern.n_shots
    ((6, 3), 2)

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       from pulserver.pypulseq import make_golden_means_3d_tilt
       d = make_golden_means_3d_tilt(300).support
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
    pulserver.SamplingPattern.to_rotations : convert the support to block rotations.
    """
    n_spokes, segment_length = int(n_spokes), int(segment_length)
    if n_spokes < 0 or segment_length <= 0:
        raise ValueError("n_spokes must be nonnegative and segment_length positive")
    m = np.arange(n_spokes, dtype=float)
    z = 2.0 * np.mod(m * 0.465571231876768, 1.0) - 1.0
    azimuth = 2.0 * np.pi * np.mod(m * 0.682327803828019, 1.0)
    support = _directions(z, azimuth)
    return SamplingPattern(support, _segments(np.arange(n_spokes, dtype=np.intp), segment_length))


def _is_fibonacci(value):
    return any(
        int(math.isqrt(candidate)) ** 2 == candidate for candidate in (5 * value * value + 4, 5 * value * value - 4)
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
    SamplingPattern
        ``support`` of shape ``(n_spokes, 3)``: unit direction vectors; one
        order entry per interleaf.

    Examples
    --------
    >>> from pulserver.pypulseq import make_spiral_phyllotaxis_tilt
    >>> pattern = make_spiral_phyllotaxis_tilt(8, 2)
    >>> pattern.n_shots, pattern.order[0]
    (2, array([0, 2, 4, 6]))

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       from pulserver.pypulseq import make_spiral_phyllotaxis_tilt
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
    return SamplingPattern(support, order)


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
