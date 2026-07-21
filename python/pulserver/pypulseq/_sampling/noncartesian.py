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


def radial_2d(
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
    >>> from pulserver.pypulseq import radial_2d
    >>> uniform = radial_2d(8)
    >>> np.rad2deg(uniform.support[:, 0]).round(1)
    array([  0. ,  22.5,  45. ,  67.5,  90. , 112.5, 135. , 157.5])
    >>> golden = radial_2d(400, scheme="golden", segment_length=100)
    >>> golden.n_shots
    4

    .. plot::
       :include-source: false

       import numpy as np
       import matplotlib.pyplot as plt
       from pulserver.pypulseq import radial_2d

       fig, axes = plt.subplots(1, 2, figsize=(8, 4), subplot_kw={"polar": True})
       for ax, scheme in zip(axes, ("linear", "golden")):
           angles = radial_2d(21, scheme=scheme).support[:, 0]
           for i, a in enumerate(angles):
               ax.plot([a, a + np.pi], [1, 1], color=plt.cm.viridis(i / 20))
           ax.set_yticks([]); ax.set_title(f"{scheme}, 21 spokes")
       fig.tight_layout()

    References
    ----------
    Scholand et al., RAGA sampling, DOI ``10.1002/mrm.30254``.

    See Also
    --------
    directions_to_rotations : turn 3D directions into block rotations.
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


def golden_angles(n: int) -> np.ndarray:
    """Return ``n`` full-circle golden-angle spoke angles, in radians.

    A flat angle array rather than a :class:`SamplingPattern`, in the
    full-circle (``2 * pi``) convention used by
    :mod:`pulserver.pypulseq.arbgrad` — pair it directly with a base waveform
    and :func:`pulserver.pypulseq.make_rotation`. Use :func:`radial_2d` when
    the angular period or the segmentation matters.

    Parameters
    ----------
    n : int
        Number of angles.

    Returns
    -------
    numpy.ndarray
        Angles (rad), length ``n``, accumulated monotonically modulo ``2 pi``.

    Examples
    --------
    >>> import numpy as np
    >>> from pulserver.pypulseq import golden_angles
    >>> np.rad2deg(golden_angles(4)).round(2)
    array([  0.  , 111.25, 222.49, 333.74])

    See Also
    --------
    radial_2d : full spoke plan with period, scheme, and segmentation control.
    """
    # Legacy API uses the full-circle, monotonically accumulated arbgrad
    # convention.  ``radial_2d`` remains the explicit half/full-period API.
    from .cartesian import golden_angles as _legacy_golden_angles

    return _legacy_golden_angles(n)


def uniform_angles(n: int) -> np.ndarray:
    """Return ``n`` equally spaced full-circle spoke angles, in radians.

    The uniform counterpart of :func:`golden_angles`, in the same flat-array,
    full-circle convention.

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
    >>> from pulserver.pypulseq import uniform_angles
    >>> np.rad2deg(uniform_angles(4))
    array([  0.,  90., 180., 270.])
    """
    from .cartesian import uniform_angles as _legacy_uniform_angles

    return _legacy_uniform_angles(n)


def _directions(z, azimuth):
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack((radius * np.cos(azimuth), radius * np.sin(azimuth), z))


def golden_means_3d(n_spokes, *, segment_length=1):
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
    >>> from pulserver.pypulseq import golden_means_3d
    >>> pattern = golden_means_3d(6, segment_length=3)
    >>> pattern.support.shape, pattern.n_shots
    ((6, 3), 2)

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       from pulserver.pypulseq import golden_means_3d

       d = golden_means_3d(300).support
       fig = plt.figure(figsize=(5, 5))
       ax = fig.add_subplot(projection="3d")
       ax.scatter(d[:, 0], d[:, 1], d[:, 2], s=6)
       ax.set_box_aspect((1, 1, 1))
       ax.set_title("golden_means_3d(300)")

    References
    ----------
    Chan et al., temporal stability of 3D golden-means radial, DOI
    ``10.1002/mrm.22732``.

    See Also
    --------
    spiral_phyllotaxis : interleaved alternative with smooth intra-shot paths.
    directions_to_rotations : convert the directions to block rotations.
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


def spiral_phyllotaxis(n_spokes, n_interleaves, *, require_fibonacci=True):
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
    >>> from pulserver.pypulseq import spiral_phyllotaxis
    >>> pattern = spiral_phyllotaxis(8, 2)
    >>> pattern.n_shots, pattern.order[0]
    (2, array([0, 2, 4, 6]))

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       from pulserver.pypulseq import spiral_phyllotaxis

       pattern = spiral_phyllotaxis(377, 13)
       fig = plt.figure(figsize=(5, 5))
       ax = fig.add_subplot(projection="3d")
       for shot in range(3):
           d = pattern[shot]
           ax.plot(d[:, 0], d[:, 1], d[:, 2], marker="o", ms=3, lw=0.6)
       ax.set_box_aspect((1, 1, 1))
       ax.set_title("spiral_phyllotaxis(377, 13): first 3 shots")

    References
    ----------
    Piccini et al., spiral phyllotaxis, DOI ``10.1002/mrm.22898``.

    See Also
    --------
    golden_means_3d : uniform in any window, at the cost of shot smoothness.
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


def directions_to_rotations(directions, *, reference=(1.0, 0.0, 0.0)):
    """Convert spoke directions into minimal-angle rotation matrices.

    The bridge between a sampling plan and the sequence: a non-Cartesian
    readout designs **one** base waveform along ``reference``, and each shot
    replays it under the matrix returned here. Each rotation is the shortest
    one taking ``reference`` onto that direction (Rodrigues); antipodal
    directions get a well-defined 180-degree rotation rather than a singular
    matrix.

    Feed the result to :func:`pulserver.pypulseq.make_rotation`, or pass it to
    a readout module's ``rotation=`` argument.

    Parameters
    ----------
    directions : array_like
        Shape ``(3,)`` or ``(N, 3)``; need not be normalised.
    reference : array_like, optional
        Direction the base waveform was designed along (default ``+x``).

    Returns
    -------
    numpy.ndarray
        Shape ``(N, 3, 3)`` rotation matrices, one per direction.

    Examples
    --------
    >>> import numpy as np
    >>> from pulserver.pypulseq import directions_to_rotations, golden_means_3d
    >>> rotations = directions_to_rotations([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    >>> rotations.shape
    (2, 3, 3)
    >>> rotations[1] @ np.array([1.0, 0.0, 0.0])
    array([0., 1., 0.])

    Rotate a 3D radial base waveform shot by shot::

        pattern = golden_means_3d(1000)
        for rotation in directions_to_rotations(pattern.support):
            readout(seq, rotation=rotation)
    """
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
