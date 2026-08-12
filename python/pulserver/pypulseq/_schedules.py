"""RF phase and refocusing-flip schedules."""

from __future__ import annotations

__all__ = ["make_phase_cycling_schedule", "make_rf_spoiling_schedule", "make_traps_schedule"]

from collections.abc import Sequence

import numpy as np


def make_rf_spoiling_schedule(
    length: int,
    *,
    increment: float = np.deg2rad(117.0),
    initial_phase: float = 0.0,
    initial_increment: float = 0.0,
) -> np.ndarray:
    """Return the quadratic RF-spoiling phase schedule, in radians.

    RF spoiling destroys residual transverse magnetization by advancing the
    excitation phase quadratically: the increment itself grows by
    ``increment`` every TR, so the phase never repeats over any short cycle
    and coherences average away. 117 degrees is the standard choice for
    spoiled GRE.

    Set the same value as the excitation ``phase_offset`` **and** the ADC
    ``phase_offset`` of each TR, so the receiver demodulates in step.

    Parameters
    ----------
    length : int
        Number of repetitions.
    increment : float, optional
        Quadratic phase increment (rad); 117 degrees by default.
    initial_phase : float, optional
        Phase of the first repetition (rad).
    initial_increment : float, optional
        Linear increment at the first repetition (rad).

    Returns
    -------
    numpy.ndarray
        Phases (rad) in ``[0, 2 pi)``, length ``length``.

    Examples
    --------
    >>> import numpy as np
    >>> from pulserver.pypulseq import make_rf_spoiling_schedule
    >>> np.rad2deg(make_rf_spoiling_schedule(4)).round(1)
    array([  0.,   0., 117., 351.])

    Apply it per TR, to the pulse and the receiver alike::

        phases = make_rf_spoiling_schedule(n_reps)
        for shot, phase in enumerate(phases):
            readout.rf.phase_offset = readout.adc.phase_offset = phase
            ...
    .. plot::
       :include-source: false

       import numpy as np
       import matplotlib.pyplot as plt
       from pulserver.pypulseq import make_rf_spoiling_schedule
       plt.figure(figsize=(6, 3))
       plt.plot(np.rad2deg(make_rf_spoiling_schedule(64)), marker="o", ms=3, lw=0.7)
       plt.xlabel("repetition"); plt.ylabel("RF phase [deg]")
       plt.title("quadratic RF spoiling, 117 deg increment")
       plt.tight_layout()

    See Also
    --------
    make_phase_cycling_schedule : the balanced-SSFP alternative.
    """
    if length < 0:
        raise ValueError("length must be >= 0")
    phases = np.empty(length, dtype=float)
    phase = float(initial_phase)
    phase_increment = float(initial_increment)
    for index in range(length):
        phases[index] = phase % (2.0 * np.pi)
        phase = (phase + phase_increment) % (2.0 * np.pi)
        phase_increment = (phase_increment + increment) % (2.0 * np.pi)
    return phases


def make_phase_cycling_schedule(
    length: int,
    phases: Sequence[float] = (0.0, np.pi),
) -> np.ndarray:
    """Tile a fixed phase cycle to ``length`` entries, in radians.

    Unlike RF spoiling, phase *cycling* deliberately repeats: the excitation
    phase steps through a short, fixed pattern so that the steady state is
    preserved and its off-resonance response is shifted in a controlled way.
    The default ``(0, pi)`` alternation is standard bSSFP; longer cycles
    (``0, pi/2, pi, 3pi/2``) are used to move or average the banding pattern.

    Parameters
    ----------
    length : int
        Number of repetitions to fill.
    phases : sequence of float, optional
        The cycle to repeat (rad); ``(0, pi)`` by default.

    Returns
    -------
    numpy.ndarray
        Phases (rad) in ``[0, 2 pi)``, length ``length``.

    Examples
    --------
    >>> import numpy as np
    >>> from pulserver.pypulseq import make_phase_cycling_schedule
    >>> np.rad2deg(make_phase_cycling_schedule(5))
    array([  0., 180.,   0., 180.,   0.])
    >>> np.rad2deg(make_phase_cycling_schedule(4, (0.0, np.pi / 2)))
    array([ 0., 90.,  0., 90.])

    .. plot::
       :include-source: false

       import numpy as np
       import matplotlib.pyplot as plt
       from pulserver.pypulseq import make_phase_cycling_schedule
       plt.figure(figsize=(6, 3))
       plt.plot(np.rad2deg(make_phase_cycling_schedule(16)), marker="o", ms=3, lw=0.7)
       plt.xlabel("repetition"); plt.ylabel("RF phase [deg]")
       plt.title("bSSFP (0, pi) phase cycling")
       plt.tight_layout()

    See Also
    --------
    make_rf_spoiling_schedule : for spoiled, non-steady-state sequences.
    """
    if length < 0:
        raise ValueError("length must be >= 0")
    cycle = np.asarray(phases, dtype=float)
    if cycle.ndim != 1 or cycle.size == 0 or not np.all(np.isfinite(cycle)):
        raise ValueError("phases must be a non-empty one-dimensional finite sequence")
    return np.resize(cycle, length) % (2.0 * np.pi)


def make_traps_schedule(
    length: int,
    target_flip_angle: float,
    *,
    variable: bool = True,
) -> np.ndarray:
    """Return a TRAPS variable refocusing flip-angle schedule, in radians.

    A long FSE train played at a constant 180 degrees both exceeds SAR limits
    and decays fast. TRAPS instead starts high and sweeps down to a low target
    flip angle, driving the magnetization into a pseudo-steady state whose
    signal decays far more slowly — so a much longer echo train stays usable.

    ``variable=False`` returns a constant ``target_flip_angle`` train, which is
    the useful comparison baseline.

    Parameters
    ----------
    length : int
        Echo train length (>= 1).
    target_flip_angle : float
        Asymptotic refocusing flip angle (rad), positive.
    variable : bool, optional
        Sweep down to the target (default) instead of holding it constant.

    Returns
    -------
    numpy.ndarray
        Refocusing flip angles (rad), length ``length``, monotonically
        decreasing towards ``target_flip_angle``.

    Examples
    --------
    >>> import numpy as np
    >>> from pulserver.pypulseq import make_traps_schedule
    >>> flips = make_traps_schedule(8, np.deg2rad(120))
    >>> np.rad2deg(flips)[[0, -1]].round(1)
    array([153. , 120.2])

    Use it as a per-echo envelope scale, ``flips / np.pi`` being the factor
    each refocusing pulse's amplitude is multiplied by::

        flips = make_traps_schedule(etl, np.deg2rad(120))

    .. plot::
       :include-source: false

       import numpy as np
       import matplotlib.pyplot as plt
       from pulserver.pypulseq import make_traps_schedule
       plt.figure(figsize=(6, 3))
       for target in (60, 90, 120):
           flips = make_traps_schedule(32, np.deg2rad(target))
           plt.plot(np.rad2deg(flips), marker="o", ms=3, label=f"target {target} deg")
       plt.xlabel("echo"); plt.ylabel("refocusing flip [deg]")
       plt.title("TRAPS schedules, ETL 32"); plt.legend(fontsize=8)
       plt.tight_layout()

    References
    ----------
    Alsop, TRAPS / variable-flip refocusing, DOI ``10.1002/mrm.1910370422``.

"""
    if length < 1:
        raise ValueError("length must be >= 1")
    if not np.isfinite(target_flip_angle) or target_flip_angle <= 0:
        raise ValueError("target_flip_angle must be a positive finite angle")
    if not variable or length == 1:
        return np.full(length, target_flip_angle, dtype=float)

    first = (
        np.pi / 2.0
        + target_flip_angle / 2.0
        + 0.4 * ((2.0 - 1.0) / 2.0) ** 2.0 * (np.pi / 2.0 - target_flip_angle / 2.0)
    )
    delta = first - target_flip_angle
    echo = np.arange(2, length + 1, dtype=float)
    return np.concatenate(([first], target_flip_angle + delta / (2.0 ** (echo - 0.5))))
