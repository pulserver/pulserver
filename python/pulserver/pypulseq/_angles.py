"""Projection angles for radial and other rotated acquisitions.

Each returns one angle per spoke, in radians. Grouping spokes into shots, and
turning angles into rotation matrices, belongs to :mod:`pulserver.design`.
"""

from __future__ import annotations

__all__ = [
    "calc_golden_angles",
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
