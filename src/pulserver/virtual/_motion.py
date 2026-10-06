"""A subject moving as a rigid body during a scan."""

from __future__ import annotations

__all__ = ["RigidMotion"]

import argparse
from collections.abc import Callable, Sequence

import numpy as np


class RigidMotion:
    """A subject moving as a rigid body.

    At time ``t``, in s from the start of the scan, the point ``p`` of the
    subject at rest lies at ``R(t) (p - centre) + centre + offset(t)``, every
    vector along the physical axes, in m.

    Parameters
    ----------
    rotation : callable, optional
        ``R(t)``, a ``(3, 3)`` orthonormal matrix; the identity without one.
    offset : callable, optional
        ``offset(t)``, a translation of ``(3,)`` metres; none without one.
    centre : array_like, default=(0, 0, 0)
        The point ``rotation`` turns about, in m.

    Examples
    --------
    A nod of 2 degrees about x, once every 4 s, and a drift of 1 mm/min
    along z:

    >>> import numpy as np
    >>> from pulserver.virtual import RigidMotion
    >>> def nod(t):
    ...     angle = np.radians(2.0) * np.sin(2.0 * np.pi * t / 4.0)
    ...     c, s = np.cos(angle), np.sin(angle)
    ...     return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    >>> motion = RigidMotion(nod, lambda t: [0.0, 0.0, 1e-3 * t / 60.0])
    >>> motion(1.0, np.array([[0.0, 0.1, 0.0]])).round(5)
    array([[0.     , 0.09994, 0.00351]])
    """

    def __init__(
        self,
        rotation: Callable[[float], np.ndarray] | None = None,
        offset: Callable[[float], Sequence[float]] | None = None,
        centre: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.rotation = rotation
        self.offset = offset
        self.centre = np.asarray(centre, dtype=float)

    @classmethod
    def jumps(
        cls,
        rate_hz: float,
        translation_m: float,
        rotation_rad: float,
        duration_s: float,
        *,
        centre: Sequence[float] = (0.0, 0.0, 0.0),
        seed: int = 0,
    ) -> RigidMotion:
        """Return a subject that holds still between sudden moves.

        The moves come at the times of a Poisson process of ``rate_hz`` over
        ``duration_s``, so the pose is a Markov jump process. Each move adds
        to the pose a translation and a rotation vector whose components are
        normal, of standard deviations ``translation_m`` and ``rotation_rad``.
        """
        rng = np.random.default_rng(seed)
        times = np.cumsum(rng.exponential(1.0 / rate_hz, int(rate_hz * duration_s) + 8))
        times = times[times < duration_s]
        shifts = np.cumsum(rng.normal(0.0, translation_m, (times.size, 3)), axis=0)
        turns = [np.eye(3)]
        for vector in rng.normal(0.0, rotation_rad, (times.size, 3)):
            turns.append(_rotation(vector) @ turns[-1])
        shifts = np.concatenate([np.zeros((1, 3)), shifts])

        def pose(t: float) -> int:
            return int(np.searchsorted(times, t, side="right"))

        return cls(lambda t: turns[pose(t)], lambda t: shifts[pose(t)], centre)

    def transform(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(R, shift)``, with which the point ``p`` at rest lies at ``R @ p + shift`` at ``t``."""
        turn = np.eye(3)
        if self.rotation is not None:
            turn = np.asarray(self.rotation(t), dtype=float)
            if turn.shape != (3, 3) or not np.allclose(
                turn @ turn.T, np.eye(3), atol=1e-9
            ):
                raise ValueError(f"the rotation at {t} s is not a (3, 3) rotation")
        shift = self.centre - turn @ self.centre
        if self.offset is not None:
            shift = shift + np.asarray(self.offset(t), dtype=float)
        return turn, shift

    def __call__(self, t: float, positions: np.ndarray) -> np.ndarray:
        """Return where the ``(n, 3)`` points at rest lie at ``t``."""
        turn, shift = self.transform(t)
        return positions @ turn.T + shift


def _rotation(vector: np.ndarray) -> np.ndarray:
    """Return the rotation about ``vector`` through its length, in rad."""
    angle = float(np.linalg.norm(vector))
    if angle == 0.0:
        return np.eye(3)
    x, y, z = vector / angle
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * cross @ cross


#: How long a subject that jumps is drawn moving for, in s: longer than a scan.
_JUMPING_S = 4 * 3600.0


def motion_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the options :func:`subject_motion` reads to ``parser``."""
    parser.add_argument(
        "--nod",
        type=float,
        nargs=2,
        metavar=("DEGREES", "PERIOD"),
        help="the subject turns about the physical x axis through the isocentre "
        "by DEGREES times the sine of 2 pi t / PERIOD, PERIOD in s",
    )
    parser.add_argument(
        "--drift",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="the subject drifts along the physical axes, in mm/min",
    )
    parser.add_argument(
        "--jumps",
        type=float,
        nargs=3,
        metavar=("RATE", "MM", "DEGREES"),
        help="the subject moves suddenly RATE times a second on average, each "
        "move a translation of MM and a rotation of DEGREES standard deviation "
        "about each axis",
    )


def subject_motion(args: argparse.Namespace) -> RigidMotion | None:
    """Return the motion ``--nod``, ``--drift`` and ``--jumps`` describe, composed in that order; None without any.

    Raises
    ------
    ValueError
        If a nod's period or the jumps' rate is not above zero.
    """
    parts = []
    if args.nod is not None:
        amplitude, period = np.radians(args.nod[0]), args.nod[1]
        if not period > 0.0:
            raise ValueError(f"a nod's period is above zero, not {period} s")

        def nod(t: float) -> np.ndarray:
            return _rotation(
                np.array([amplitude * np.sin(2.0 * np.pi * t / period), 0.0, 0.0])
            )

        parts.append(RigidMotion(nod))
    if args.drift is not None:
        velocity = 1e-3 * np.asarray(args.drift, dtype=float) / 60.0
        parts.append(RigidMotion(offset=lambda t: velocity * t))
    if args.jumps is not None:
        rate, mm, degrees = args.jumps
        if not rate > 0.0:
            raise ValueError(f"jumps come at a rate above zero, not {rate} /s")
        parts.append(
            RigidMotion.jumps(rate, 1e-3 * mm, np.radians(degrees), _JUMPING_S)
        )
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]

    def composed(t: float) -> tuple[np.ndarray, np.ndarray]:
        turn, shift = np.eye(3), np.zeros(3)
        for part in parts:
            r, s = part.transform(t)
            turn, shift = r @ turn, r @ shift + s
        return turn, shift

    return RigidMotion(lambda t: composed(t)[0], lambda t: composed(t)[1])
