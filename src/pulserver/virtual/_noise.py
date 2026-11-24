"""Thermal receive noise, correlated between the channels of an array."""

from __future__ import annotations

__all__ = ["COIL_NOISE", "ReceiveNoise"]

from collections.abc import Sequence

import numpy as np

from ._tissue import Tissue

#: Variance of each loop's own coil and preamplifier noise, as a fraction of
#: the mean variance the sample induces in a channel. At 0.3 the median
#: channel of a close-fitting 32-loop 7 T head array takes about two thirds of
#: its noise from the sample, an unloaded-to-loaded Q ratio near 3, and the
#: noise standard deviation spans a few times between channels.
COIL_NOISE = 0.3


class ReceiveNoise:
    """Complex Gaussian noise added to every sample, correlated between channels by ``covariance``.

    Readout ``at`` of a scan always receives the same noise for a given
    ``seed``, so a readout acquired twice is acquired alike.

    Parameters
    ----------
    covariance : np.ndarray
        ``(coils, coils)`` Hermitian, positive definite channel covariance.
    std : float
        Standard deviation of the real and of the imaginary part of a sample
        of a channel whose covariance diagonal is one, in the units of the
        readouts.
    seed : int, default=0
        Seed of the noise.

    Raises
    ------
    ValueError
        If ``covariance`` is not square or not positive definite.
    """

    def __init__(self, covariance: np.ndarray, std: float, *, seed: int = 0) -> None:
        covariance = np.asarray(covariance, dtype=np.complex128)
        if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
            raise ValueError(f"a channel covariance is square, not {covariance.shape}")
        try:
            self._factor = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as error:
            raise ValueError("a channel covariance is positive definite") from error
        self.covariance = covariance
        self.std = float(std)
        self.seed = int(seed)

    @classmethod
    def of_tissue(
        cls,
        tissue: Tissue,
        std: float,
        *,
        coil_noise: float = COIL_NOISE,
        gains: Sequence[float] | None = None,
        seed: int = 0,
    ) -> ReceiveNoise:
        """Return the noise of the tissue's receive array loaded by the tissue.

        The sample term of channels ``i`` and ``j`` is the overlap of their
        sensitivities over the entries holding protons, ``sum conj(S_i) S_j``,
        each entry taken as conducting alike. Each channel adds its own coil
        noise, ``coil_noise`` times the mean sample variance, and is scaled by
        its preamplifier gain. The covariance is scaled to a mean diagonal of
        one.

        Raises
        ------
        ValueError
            If the tissue has a single coil of unit sensitivity.
        """
        sensitivity = tissue.receive(tissue.positions[tissue.density > 0])
        if sensitivity is None:
            raise ValueError("one coil of unit sensitivity has no channel covariance")
        sensitivity = np.asarray(sensitivity, dtype=np.complex128)
        sample = sensitivity.T @ sensitivity.conj() / sensitivity.shape[0]
        floor = coil_noise * float(np.real(np.trace(sample))) / tissue.coils
        covariance = sample + floor * np.eye(tissue.coils)
        if gains is not None:
            gain = np.asarray(gains, dtype=np.float64)
            covariance = gain[:, None] * covariance * gain[None, :]
        covariance /= float(np.real(np.trace(covariance))) / tissue.coils
        return cls(covariance, std, seed=seed)

    def __call__(self, at: int, samples: int) -> np.ndarray:
        """Return the noise of readout ``at``, ``(coils, samples)`` complex64."""
        return self._drawn(np.random.default_rng((self.seed, int(at))), samples)

    def readouts(self, count: int, samples: int) -> list[np.ndarray]:
        """Return ``count`` noise-only readouts of ``samples`` each, as a noise scan acquires them, apart from any readout of the scan."""
        rng = np.random.default_rng((self.seed, 0, 1))
        return [self._drawn(rng, samples) for _ in range(count)]

    def _drawn(self, rng: np.random.Generator, samples: int) -> np.ndarray:
        coils = self._factor.shape[0]
        white = rng.standard_normal((coils, samples)) + 1j * rng.standard_normal(
            (coils, samples)
        )
        return (self.std * self._factor @ white).astype(np.complex64)
