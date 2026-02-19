"""Phase Cycling schemes."""

__all__ = ["RfPhaseCycle"]


import numpy as np


class RfPhaseCycle:
    """
    Calculate phase cycling pattern for RF pulses used in acquisition and simulation.

    Parameters
    ----------
    num_pulses : int
        Number of RF pulses.
    phase_increment : float | str
        Phase increment in degrees (float) for quadratic spoiling, or a string (e.g., ``"balanced"``)
        to generate a balanced phase cycling pattern.

    Returns
    -------
    numpy.ndarray
        Phase cycling pattern in radians.

    Notes
    -----
    If ``phase_increment`` is a float, the function calculates a quadratic phase spoiling pattern
    with the given phase increment (converted from degrees to radians).
    If ``phase_increment`` is a string (``="balanced"``), a balanced phase cycling pattern is generated.

    Examples
    --------
    Quadratic phase spoiling with 90-degree increments for 4 RF pulses:

    >>> from pulseforge import RfPhaseCycle
    >>> RfPhaseCycle(4, 90.0)._phase
    array([ 0.        ,  1.57079633,  4.71238898,  9.42477796])

    Balanced phase cycling for 4 RF pulses:

    >>> RfPhaseCycle(4, "balanced")._phase
    array([0.        , 3.14159265, 0.        , 3.14159265])

    Ported from:
    Shaihan Malik, July 2017

    """

    def __init__(self, num_pulses: int, phase_increment: float | str):
        if isinstance(phase_increment, str) is False:
            is_quadratic_spoiling = True
            phase_increment = np.deg2rad(phase_increment)
        else:
            assert (
                phase_increment == "balanced"
            ), f"phase_increment must be either a float or 'balanced', got {phase_increment}"
            is_quadratic_spoiling = False

        if is_quadratic_spoiling:
            # Quadratic spoiling phase
            pulse_indices = np.arange(num_pulses)
            phase_pattern = pulse_indices * (pulse_indices + 1) / 2 * phase_increment
        else:
            # Balanced phase cycling case
            if num_pulses % 2 == 0:
                phase_pattern = np.tile(np.asarray([0, np.pi]), num_pulses // 2)
            else:
                phase_pattern = np.tile(np.asarray([0, np.pi]), num_pulses // 2)
                phase_pattern = np.concatenate((phase_pattern, [0]))

        _phase = np.deg2rad(phase_pattern)
        self._phase = np.mod(_phase + np.pi, 2 * np.pi) - np.pi
        self.count = 0

    def __call__(self):  # noqa
        self.count += 1
        return self._phase[self.count - 1]

    def reset(self):  # noqa
        self.count = 0


def _map_to_0_pi(angles):
    angles = np.mod(angles, 360.0)
    angles = np.where(angles > 180.0, 360.0 - angles, angles)
    return angles
