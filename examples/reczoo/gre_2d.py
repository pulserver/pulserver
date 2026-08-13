"""Reconstruction for :mod:`pulserver.seqzoo.gre_2d`.

This is a Gadgetron/ISMRMRD-style *bucketed* reconstruction, not an offline
solver handed a finished k-space array. While the scan runs the runtime streams
``Acquisition`` objects, sorts them by their labels, and -- because this app
declares ``split_on="ACQ_LAST_IN_SLICE"`` -- calls :meth:`Gre2DRecon.recon`
once per slice with that slice's acquisitions as a :class:`AcquisitionBucket`.
Everything the reconstruction needs it reads back from the data the sequence
encoded: the phase encode of each acquisition, the calibration block it flagged,
and the echo position it recorded.

The method has two layers, kept visibly apart so each is legible on its own:

ISMRMRD adapter
    Turns one slice's bucket into the arrays the numerics speak -- the phase
    encodes (``kspace_encode_step_1``) gridded onto a full Cartesian grid, the
    echo position (``center_sample``) that places a partial echo, and the
    matrix size from the header. This is the layer a Gadgetron/ISMRMRD user
    recognises and an offline-recon user has to learn.

Reconstruction
    Pure arrays, expressed in the framework's DeepInverse-style operator
    vocabulary -- a :class:`~pulserver.recon.physics.Cartesian2D` operator, its
    coil-wise adjoint, :func:`~pulserver.recon.pics`, NLINV and POCS, with the
    coils combined by an explicit :func:`~pulserver.recon.postprocessing.coil_combine`.
    This is the layer a sigpy/mrpro/DeepInverse user recognises. It carries no
    hand-written FFT: a Cartesian operator with no sensitivities *is* the
    centered transform, one image per coil.

Which of three reconstructions runs is detected from the bucket, never
declared:

* fully sampled, full echo -- the coil-wise adjoint (a centered inverse
  transform per coil), the coils combined by root-sum-of-squares;
* a readout truncated before the echo -- phase-constrained POCS fills the
  missing edge against a full-width readout axis;
* uniformly undersampled with a fully sampled centre -- CG-SENSE against NLINV
  coil maps.

Parallel imaging comes first and partial Fourier second: the phase constraint
POCS relies on is only meaningful once the image is unaliased.

The result is one magnitude image per slice. The app does not build the image
header: it returns a :class:`ReconResult` naming the acquisition its geometry
comes from and asking for DICOM, and the runtime reads that acquisition's
header (slice position, orientation) and the MRD header's recon field of view
to make the ``ismrmrd.Image`` and the DICOM series.

Needs the numerical stack: ``pip install "pulserver[recon-cpu]"``.
"""

from __future__ import annotations

__all__ = ["PLUGIN", "Gre2DRecon"]

from typing import Any

import numpy as np

from pulserver import AcquisitionBucket, ReconApp, ReconContext, ReconResult
from pulserver.recon.postprocessing import coil_combine
from pulserver.recon.preprocessing import POCS, grid_cartesian


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
        Torch device the reconstruction runs on. ``None`` is the CPU.
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

    def recon(
        self,
        bucket: AcquisitionBucket,
        context: ReconContext,
    ) -> ReconResult:
        """Reconstruct the acquisitions of one slice into one image.

        The phase-encode matrix comes from the MRD header; the image geometry
        and the DICOM series come from the runtime, which reads the referenced
        acquisition's header (slice position and orientation) and the header's
        recon field of view when it turns the returned :class:`ReconResult` into
        an ``ismrmrd.Image`` and, because ``dicom=True``, a DICOM.
        """
        # ISMRMRD adapter: one slice's acquisitions onto a full Cartesian grid.
        n_y = _phase_encode_lines(context.header, bucket)
        kspace, samples = _grid(bucket, n_y)
        # The readout sampling profile, shared by every acquired line: full when
        # the whole echo was read, truncated for a partial echo.
        readout = samples.any(axis=0)
        calibration = _grid(bucket, n_y, reference=True)[0] if len(bucket.ref) else None

        # Reconstruction: one operator vocabulary, the branch chosen by the data.
        if calibration is not None:
            image = self._sense(kspace, samples, readout, calibration)
        else:
            # No calibration: root-sum-of-squares of the per-coil images. A
            # partial echo is filled first (POCS); a full echo is the adjoint of
            # a coil-wise operator. Either way the coil combination is one
            # explicit step, not folded into the transform.
            if readout.all():
                coils = _coil_images(kspace, samples, self.device)
            else:
                coils = POCS(
                    dimension=2, partial_axis=-1, iterations=self.pocs_iterations
                )(kspace, readout)
            image = coil_combine(coils, coil_axis=0)

        # One magnitude image for the slice, transposed to the column/row order
        # an image (and its DICOM) is read in. ``reference`` names the
        # acquisition whose header carries this slice's geometry, which the
        # runtime copies into the image; ``dicom=True`` asks it to emit one.
        return ReconResult(
            np.abs(_to_numpy(image)).transpose(),
            reference=0,
            image_type="magnitude",
            dicom=True,
        )

    def _sense(
        self,
        kspace: np.ndarray,
        samples: np.ndarray,
        readout: np.ndarray,
        calibration: np.ndarray,
    ) -> Any:
        """Unalias the slice against NLINV sensitivities, then fill the echo."""
        import torch

        from pulserver.recon import pics
        from pulserver.recon.calibration import NLINV
        from pulserver.recon.physics import Cartesian2D

        device = "cpu" if self.device is None else self.device
        data = torch.as_tensor(kspace, dtype=torch.complex64, device=device)
        reference = torch.as_tensor(calibration, dtype=torch.complex64, device=device)
        sampling = torch.as_tensor(samples, dtype=torch.float32, device=device)

        coil_maps = NLINV(calibration_width=self.calibration_width)(reference[None])
        physics = Cartesian2D(sampling[None], coil_maps)
        solution = pics(
            torch.view_as_real(data[None]),
            physics,
            regularization=self.regularization,
            iterations=self.iterations,
        )
        image = torch.view_as_complex(solution.movedim(1, -1).contiguous())[0]
        if readout.all():
            return image

        # Partial echo: the same operator supplies the centered k-space POCS
        # needs, and POCS fills the truncated readout under a phase constraint --
        # valid only now that parallel imaging has removed the aliasing.
        return POCS(dimension=2, partial_axis=-1, iterations=self.pocs_iterations)(
            physics.fft(image),
            torch.as_tensor(readout, device=device),
        )


# %% ISMRMRD adapter -- reads what the sequence encoded


def _grid(
    bucket: AcquisitionBucket,
    n_y: int,
    *,
    reference: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Zero-fill one bucket's acquisitions onto a full Cartesian grid.

    Every acquisition carries the phase encode it belongs to
    (``kspace_encode_step_1``) and where its echo sits (``center_sample``),
    which is all :func:`grid_cartesian` needs to place a partial echo
    right-aligned against a full-width readout axis.
    """
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


def _phase_encode_lines(header: Any, bucket: AcquisitionBucket) -> int:
    """Phase-encode matrix size, from the MRD header or from what arrived."""
    try:
        return int(header.encoding[0].encodedSpace.matrixSize.y)
    except (AttributeError, IndexError, TypeError):
        return int(np.max(bucket.labels("kspace_encode_step_1"))) + 1


# %% reconstruction -- operator vocabulary, no hand-written transform


def _coil_images(kspace: np.ndarray, samples: np.ndarray, device: Any) -> Any:
    """Per-coil images, from the adjoint of a coil-wise Cartesian operator.

    A Cartesian operator built with no sensitivity maps is coil-wise: its
    adjoint zero-fills, inverts and hands back one image per coil, without a
    hand-written FFT and without folding a coil combination into the transform.
    The caller root-sum-of-squares combines them.
    """
    import torch

    from pulserver.recon.physics import Cartesian2D

    device = "cpu" if device is None else device
    _, n_y, n_x = kspace.shape
    mask = torch.as_tensor(samples, dtype=torch.float32, device=device)[None, None]
    measurement = torch.view_as_real(
        torch.as_tensor(kspace, dtype=torch.complex64, device=device)[None]
    )
    coils = Cartesian2D(mask, img_size=(n_y, n_x)).A_adjoint(measurement)
    return torch.view_as_complex(coils.movedim(1, -1).contiguous())[0]


def _to_numpy(array: Any) -> np.ndarray:
    """Bring a reconstruction back to NumPy, whether it is Torch or already so."""
    return array.cpu().numpy() if hasattr(array, "cpu") else np.asarray(array)


PLUGIN = Gre2DRecon()
