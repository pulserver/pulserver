"""Respiratory and cardiac fluctuation of the main field."""

from __future__ import annotations

__all__ = ["PhysiologicalField"]

from collections.abc import Sequence

import numpy as np


class PhysiologicalField:
    """A field offset common to the whole object, a sum of sinusoids of time.

    At time ``t``, in s from the start of the scan, the field is
    ``sum_k a_k sin(2 pi t / T_k + phi_k)`` Hz. Each readout precesses by the
    offset at its excitation from that excitation's time on, so the image of
    a shot moves along its phase encode as a field moves it; changes during
    an echo train are neglected. The phase is applied to the whole readout,
    so a pathway carried from an earlier excitation takes the latest
    excitation's phase only, which a spoiled scan barely sees.

    Parameters
    ----------
    amplitudes_hz : Sequence[float]
        Peak offset of each term (Hz).
    periods_s : Sequence[float]
        Period of each term (s).
    phases_rad : Sequence[float], default=None
        Phase of each term at the start of the scan (rad); drawn uniformly
        from ``seed`` when not given.
    seed : int, default=0
        Seed of the drawn phases.

    Raises
    ------
    ValueError
        If the terms are not given alike, or a period is not positive.

    Examples
    --------
    Respiration of 1.5 Hz every 4 s and a cardiac term of 0.5 Hz every 1 s,
    of the order a 7T brain sees:

    >>> from pulserver.virtual import PhysiologicalField
    >>> field = PhysiologicalField([1.5, 0.5], [4.0, 1.0], phases_rad=[0.0, 0.0])
    >>> round(float(field(1.0)), 3)
    1.5
    """

    def __init__(
        self,
        amplitudes_hz: Sequence[float],
        periods_s: Sequence[float],
        phases_rad: Sequence[float] | None = None,
        *,
        seed: int = 0,
    ) -> None:
        self.amplitudes_hz = np.asarray(amplitudes_hz, dtype=np.float64)
        self.periods_s = np.asarray(periods_s, dtype=np.float64)
        if phases_rad is None:
            rng = np.random.default_rng(seed)
            phases_rad = rng.uniform(0.0, 2.0 * np.pi, self.amplitudes_hz.size)
        self.phases_rad = np.asarray(phases_rad, dtype=np.float64)
        if (
            not self.amplitudes_hz.shape
            == self.periods_s.shape
            == self.phases_rad.shape
        ):
            raise ValueError("each term needs an amplitude, a period and a phase")
        if np.any(self.periods_s <= 0.0):
            raise ValueError("periods must be positive")

    def __call__(self, t_s: np.ndarray | float) -> np.ndarray:
        """Return the offset (Hz) at ``t_s``, in s from the start of the scan."""
        t = np.asarray(t_s, dtype=np.float64)[..., None]
        return np.sum(
            self.amplitudes_hz
            * np.sin(2.0 * np.pi * t / self.periods_s + self.phases_rad),
            axis=-1,
        )
