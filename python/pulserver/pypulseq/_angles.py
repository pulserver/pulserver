"""Projection angles for radial and other rotated acquisitions.

Each returns one angle per spoke, in radians. Grouping spokes into shots, and
turning angles into rotation matrices, belongs to :mod:`pulserver.design`.
"""

from __future__ import annotations

__all__ = [
    "calc_golden_angles",
    "calc_projection_shell",
    "calc_raga_angles",
    "calc_tiny_golden_angles",
    "calc_uniform_angles",
]

import math

import numpy as np

#: The golden ratio, whose irrationality is what keeps any window of
#: consecutive spokes near-uniformly distributed.
_PHI = (1.0 + math.sqrt(5.0)) / 2.0


def _accumulated(n: int, step: float) -> np.ndarray:
    n = int(n)
    if n < 0:
        raise ValueError("n must be nonnegative")
    return np.mod(np.arange(n, dtype=float) * float(step), 2.0 * np.pi)

def _generalized_fibonacci(order, index):
    if order == 1:
        return 1
    previous, current = 1, int(index)
    for _ in range(2, order):
        previous, current = current, previous + current
    return current

def calc_golden_angles(n: int) -> np.ndarray:
    """Return ``n`` golden-angle spoke rotations, in radians.

    Consecutive spokes advance by the MRI golden angle ``pi / phi`` (111.246
    degrees), so any contiguous temporal window of spokes stays
    near-uniformly distributed over the circle — the property that makes
    golden-angle radial the default for retrospectively binned and
    free-breathing acquisitions.

    A flat angle array rather than a :class:`~pulserver.ScanLoop`, in
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
    tiny_index, approximation_order = int(tiny_index), int(approximation_order)
    if n < 0:
        raise ValueError("n must be nonnegative")
    if tiny_index < 1 or approximation_order < 2:
        raise ValueError("tiny_index must be >= 1 and approximation_order >= 2")

    # The support is a finite, equidistant set of Fibonacci-many angles; the
    # order they are visited in is what stays golden-like.
    support_size = _generalized_fibonacci(approximation_order, tiny_index)
    step = _generalized_fibonacci(approximation_order - 1, 1)
    support = np.arange(support_size) * (2.0 * np.pi) / support_size
    visited = (np.arange(n, dtype=np.intp) * step) % support_size
    return support[visited]

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

def calc_projection_shell(n_views: int, n_shots: int = 1, *, scheme: str = "spiral"):
    """Cover the sphere with one base shell of spokes and a rotation per shot.

    A continuous-gradient readout cannot afford a waveform per spoke, so the
    shell is designed once and replayed rigidly rotated. Two properties make
    that work. Consecutive views inside the shell subtend a **constant angle**,
    which lets one designed transition carry any view onto the next under a
    rotation of its own. And the shell runs from ``+z`` to ``-z``, visiting
    every polar ring once, so turning it about ``z`` by ``2 * pi / n_shots``
    puts ``n_shots`` evenly spaced spokes on each ring: full coverage, and
    every shot congruent with every other.

    The two poles are the exception -- they sit on the rotation axis, so all
    shots share them.

    Parameters
    ----------
    n_views : int
        Spokes in the base shell, at least three.
    n_shots : int, optional
        Rotated replays of that shell.
    scheme : {'spiral', 'meridian'}, optional
        ``'spiral'`` winds pole to pole across equal-area rings, so the shell
        alone is already near-uniform. ``'meridian'`` is a half great circle in
        the x-z plane at equal polar steps, which is simpler and oversamples
        the poles.

    Returns
    -------
    directions : numpy.ndarray
        Unit spoke directions of the base shell, shape ``(n_views, 3)``.
    rotations : numpy.ndarray
        Rotation matrices, shape ``(n_shots, 3, 3)``, each turning the whole
        shell to where that shot samples.

    Raises
    ------
    ValueError
        If a count is out of range or ``scheme`` is unknown.

    Examples
    --------
    >>> import numpy as np
    >>> import pulserver.pypulseq as pp
    >>> directions, rotations = pp.calc_projection_shell(32, n_shots=13)
    >>> directions.shape, rotations.shape
    ((32, 3), (13, 3, 3))

    The shell runs pole to pole, so the shot rotations leave its ends alone:

    >>> bool(np.allclose(directions[[0, -1]], [[0, 0, 1], [0, 0, -1]]))
    True

    Consecutive views are exactly one step apart, which is the condition a
    rotation-encoded readout depends on:

    >>> steps = np.arccos(np.clip(np.sum(directions[:-1] * directions[1:], axis=1), -1, 1))
    >>> bool(np.ptp(steps) < 1e-9)
    True

    See Also
    --------
    calc_golden_angles : in-plane spoke angles, one per shot, for 2D radial.
    """
    n_views, n_shots = int(n_views), int(n_shots)
    if n_views < 3:
        raise ValueError("a pole-to-pole shell needs at least three views")
    if n_shots < 1:
        raise ValueError("n_shots must be positive")

    if scheme == "spiral":
        directions = _constant_step_spiral(n_views)
    elif scheme == "meridian":
        polar = np.linspace(0.0, np.pi, n_views)
        directions = np.column_stack((np.sin(polar), np.zeros(n_views), np.cos(polar)))
    else:
        raise ValueError(f"scheme must be 'spiral' or 'meridian', got {scheme!r}")
    return directions, _turns_about_z(2.0 * np.pi * np.arange(n_shots, dtype=float) / n_shots)


def _constant_step_spiral(n_views: int) -> np.ndarray:
    """A pole-to-pole spiral whose consecutive views are a constant angle apart.

    The polar ladder is fixed first, at equal-area heights from ``+1`` to
    ``-1``, and each azimuth increment is then *solved* rather than chosen,
    from ``cos(step) = cos(t_k) cos(t_k+1) + sin(t_k) sin(t_k+1) cos(dphi)``.

    The step itself is not free: at the poles ``sin(t) = 0``, so no azimuth
    increment buys any angle there and the whole step has to be polar. That
    fixes it at the polar gap next to the pole, which is the widest of them;
    every other gap has slack, and the azimuth increments take it up.
    """
    height = np.linspace(1.0, -1.0, n_views)
    polar = np.arccos(np.clip(height, -1.0, 1.0))
    step = float(np.max(np.diff(polar)))

    numerator = math.cos(step) - np.cos(polar[:-1]) * np.cos(polar[1:])
    denominator = np.sin(polar[:-1]) * np.sin(polar[1:])
    increment = np.zeros(n_views - 1)
    turning = denominator > 1e-12
    increment[turning] = np.arccos(
        np.clip(numerator[turning] / denominator[turning], -1.0, 1.0)
    )
    azimuth = np.concatenate(([0.0], np.cumsum(increment)))
    radius = np.sqrt(np.maximum(0.0, 1.0 - height**2))
    return np.column_stack((radius * np.cos(azimuth), radius * np.sin(azimuth), height))


def _turns_about_z(angles: np.ndarray) -> np.ndarray:
    """One rotation matrix about ``z`` per angle."""
    cosine, sine = np.cos(angles), np.sin(angles)
    turns = np.zeros((len(angles), 3, 3), dtype=float)
    turns[:, 0, 0] = turns[:, 1, 1] = cosine
    turns[:, 0, 1] = -sine
    turns[:, 1, 0] = sine
    turns[:, 2, 2] = 1.0
    return turns
