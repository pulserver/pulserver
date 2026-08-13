"""Reconstruction for :mod:`pulserver.seqzoo.gre_2d`.

One image per slice, handling the two things that sequence can undersample:

Parallel imaging
    Uniformly undersampled phase encoding with a fully sampled centre. The
    sequence flags that centre ``REF``/``IMA``, so the runtime hands it back
    separately in ``bucket.ref``; coil sensitivities come from it by NLINV and
    the image from a CG-SENSE solve against
    :class:`pulserver.recon.physics.Cartesian2D`.

Partial echo
    A readout truncated before the echo. Nothing has to declare it: the
    acquisition header records where the echo sits, so the acquired samples
    are placed against a full-width readout axis and the missing edge is
    filled by the phase-constrained :class:`pulserver.recon.preprocessing.POCS`.

The order is parallel imaging first, partial Fourier second — the phase
constraint POCS relies on is only meaningful once the image is unaliased.

The accelerated branch needs the numerical stack:
``pip install "pulserver[recon-cpu]"``. A fully sampled slice, with or without
a partial echo, reconstructs with NumPy alone.
"""

from __future__ import annotations

__all__ = ["PLUGIN", "Gre2DRecon"]

from typing import Any

import numpy as np

from pulserver import AcquisitionBucket, ReconApp, ReconContext, ReconResult
from pulserver.recon.postprocessing import coil_combine
from pulserver.recon.preprocessing import grid_cartesian


class Gre2DRecon(ReconApp):
    """Reconstruct one slice of a 2D Cartesian gradient echo.

    Parameters
    ----------
    regularization
        Tikhonov weight of the CG-SENSE solve.
    iterations
        Maximum CG iterations.
    calibration_width
        Centred region NLINV estimates the sensitivities over.
    pocs_iterations
        Partial-echo POCS iterations.
    device
        Torch device the accelerated branch runs on.
    """

    def __init__(
        self,
        *,
        regularization: float = 1e-3,
        iterations: int = 40,
        calibration_width: int = 24,
        pocs_iterations: int = 12,
        device: Any = None,
    ) -> None:
        super().__init__(
            split_on="ACQ_LAST_IN_SLICE",
            reject_flags=("ACQ_IS_NOISE_MEASUREMENT", "ACQ_IS_PHASECORR_DATA"),
        )
        self.regularization = float(regularization)
        self.iterations = int(iterations)
        self.calibration_width = int(calibration_width)
        self.pocs_iterations = int(pocs_iterations)
        self.device = device

    def reconstruct(
        self,
        bucket: AcquisitionBucket,
        context: ReconContext,
    ) -> ReconResult:
        """Reconstruct the acquisitions of one slice into one image."""
        n_y = _encoded_lines(context.header, bucket)
        kspace, mask = _place(bucket, n_y)

        if len(bucket.ref):
            calibration, _ = _place(bucket, n_y, reference=True)
            image = self._sense(kspace, mask, calibration)
        else:
            image = coil_combine(self._complete(kspace, mask[0]), coil_axis=0)

        return ReconResult(np.abs(image))

    def _sense(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        calibration: np.ndarray,
    ) -> np.ndarray:
        """Unalias the slice against NLINV sensitivities, then fill the echo."""
        import torch

        from pulserver.recon import pics
        from pulserver.recon.calibration import NLINV
        from pulserver.recon.physics import Cartesian2D

        device = "cpu" if self.device is None else self.device
        data = torch.as_tensor(kspace, dtype=torch.complex64, device=device)
        reference = torch.as_tensor(calibration, dtype=torch.complex64, device=device)
        sampling = torch.as_tensor(mask, dtype=torch.float32, device=device)

        coil_maps = NLINV(calibration_width=self.calibration_width)(reference[None])
        solution = pics(
            torch.view_as_real(data[None]),
            Cartesian2D(sampling[None], coil_maps),
            regularization=self.regularization,
            iterations=self.iterations,
        )
        image = torch.view_as_complex(solution.movedim(1, -1).contiguous())[0]
        return self._complete(_fft2c(image.cpu().numpy()), mask[0])

    def _complete(self, kspace: np.ndarray, samples: np.ndarray) -> np.ndarray:
        """Transform to image space, filling a truncated echo on the way."""
        if samples.all():
            return _ifft2c(kspace)

        from pulserver.recon.preprocessing import POCS

        return POCS(dimension=2, partial_axis=-1, iterations=self.pocs_iterations)(
            kspace, samples
        )


# %% private module subroutines


def _place(
    bucket: AcquisitionBucket,
    n_y: int,
    *,
    reference: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Zero-fill one bucket's acquisitions onto a full Cartesian grid."""
    kspace = np.asarray(bucket.kspace(reference=reference))
    if kspace.ndim != 3:
        raise ValueError("gre_2d expects every acquisition to have the same length")
    centers = np.asarray(bucket.labels("center_sample", reference=reference), dtype=int)
    return grid_cartesian(
        kspace,
        bucket.labels("kspace_encode_step_1", reference=reference),
        n_y,
        echo_position=int(centers[0]) if centers.size and centers[0] else None,
    )


def _encoded_lines(header: Any, bucket: AcquisitionBucket) -> int:
    """Phase-encode matrix size, from the MRD header or from what arrived."""
    try:
        return int(header.encoding[0].encodedSpace.matrixSize.y)
    except (AttributeError, IndexError, TypeError):
        return int(np.max(bucket.labels("kspace_encode_step_1"))) + 1


def _fft2c(image: np.ndarray) -> np.ndarray:
    """Centred orthonormal 2D forward transform over the trailing two axes."""
    axes = (-2, -1)
    return np.fft.fftshift(
        np.fft.fftn(np.fft.ifftshift(image, axes=axes), axes=axes, norm="ortho"),
        axes=axes,
    )


def _ifft2c(kspace: np.ndarray) -> np.ndarray:
    """Centred orthonormal 2D inverse transform over the trailing two axes."""
    axes = (-2, -1)
    return np.fft.fftshift(
        np.fft.ifftn(np.fft.ifftshift(kspace, axes=axes), axes=axes, norm="ortho"),
        axes=axes,
    )


PLUGIN = Gre2DRecon()
