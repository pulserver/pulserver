"""Reconstruction for :mod:`pulserver.seqzoo.gre_2d`.

A streamed reconstruction in three hooks: ``startup`` allocates the encoded
grid the MRD header describes, ``receive`` places each line where its counters
say, and ``recon`` runs at every boundary the sequence flags -- estimating coil
sensitivities when the calibration segment closes, reconstructing when the
slice does. The calibration lines are imaging data too, so there is one buffer
and no second grid.

Which of three reconstructions runs is read off the sampling mask rather than
declared: the coil-wise adjoint when everything is there, POCS when the readout
is truncated, CG-SENSE against NLINV maps when phase encodes are missing.
Parallel imaging comes first, because the phase constraint POCS relies on only
means something once the image is unaliased.

One magnitude image per slice, cropped to the prescribed matrix. Needs the
numerical stack: ``pip install "pulserver[recon-cpu]"``.
"""

from __future__ import annotations

__all__ = [
    "PLUGIN",
    "Gre2DRecon",
    "coil_images",
    "fill_partial_echo",
    "sense",
    "sensitivities",
]

from typing import Any

import numpy as np

from pulserver import AcquisitionBucket, ReconContext, ReconPlugin, ReconResult
from pulserver.recon import has_acquisition_flag
from pulserver.recon.postprocessing import center_crop, coil_combine
from pulserver.recon.preprocessing import (
    POCS,
    CartesianGridder,
    encoded_shape,
    receiver_channels,
    recon_shape,
)


class Gre2DRecon(ReconPlugin):
    """Reconstruct a 2D Cartesian gradient echo, one image per slice.

    Parameters
    ----------
    regularization
        Tikhonov weight of the CG-SENSE solve.
    iterations
        Maximum CG iterations.
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
        pocs_iterations: int = 12,
        device: Any = None,
    ) -> None:
        super().__init__(
            split_on=("ACQ_LAST_IN_SEGMENT", "ACQ_LAST_IN_SLICE"),
            reject_flags=("ACQ_IS_NOISE_MEASUREMENT", "ACQ_IS_PHASECORR_DATA"),
        )
        self.regularization = float(regularization)
        self.iterations = int(iterations)
        self.pocs_iterations = int(pocs_iterations)
        self.device = device

    def startup(self, context: ReconContext) -> None:
        """Allocate the scan's k-space from the grid the header describes."""
        self.buffer = CartesianGridder(
            encoded_shape(context.header),
            coils=receiver_channels(context.header),
        )
        self.image_shape = recon_shape(context.header)
        self.coil_maps: dict[int, Any] = {}

    def receive(self, acquisition: Any, context: ReconContext) -> None:
        """Place one arriving line at its slice and phase encode."""
        del context
        self.buffer.add(
            acquisition.data,
            acquisition.idx.slice,
            acquisition.idx.kspace_encode_step_1,
        )

    def recon(
        self, bucket: AcquisitionBucket, context: ReconContext
    ) -> ReconResult | None:
        """Calibrate or reconstruct, according to which boundary ended this bucket."""
        del context
        index = int(bucket.labels("slice")[-1])
        kspace, mask = self.buffer[index]

        # A segment that ends without ending its slice, or the scan, is the
        # calibration block: estimate the sensitivities now, and let the rest
        # of the slice keep arriving. There is no image to send yet. A
        # single-slice scan carries no slice counter and so no slice boundary,
        # which is why the end of the measurement has to be asked about too.
        last = bucket.acquisitions[-1]
        if has_acquisition_flag(last, "ACQ_LAST_IN_SEGMENT") and not (
            has_acquisition_flag(last, "ACQ_LAST_IN_SLICE")
            or has_acquisition_flag(last, "ACQ_LAST_IN_MEASUREMENT")
        ):
            self.coil_maps[index] = sensitivities(kspace, mask, device=self.device)
            return None

        # What the scan sampled is what selects the reconstruction: a phase
        # encode with no samples was skipped, and a readout sample missing from
        # every line is echo the sequence never acquired.
        lines = mask.any(axis=-1)
        readout = mask.any(axis=0)

        if lines.all():
            coils = (
                coil_images(kspace, mask, device=self.device)
                if readout.all()
                else fill_partial_echo(
                    kspace, readout, self.pocs_iterations, device=self.device
                )
            )
            image = coil_combine(coils, coil_axis=0)
        else:
            maps = self.coil_maps.get(index)
            if maps is None:
                maps = sensitivities(kspace, mask, device=self.device)
            image = sense(
                kspace,
                mask,
                maps,
                readout,
                regularization=self.regularization,
                iterations=self.iterations,
                pocs_iterations=self.pocs_iterations,
                device=self.device,
            )

        # One magnitude image, cropped from the encoded grid to the prescribed
        # matrix and transposed to the column/row order an image (and its
        # DICOM) is read in. ``reference`` names the acquisition that ended the
        # bucket, so the runtime copies this slice's geometry into the image;
        # ``dicom=True`` asks it to emit one.
        image = center_crop(np.abs(_asnumpy(image)), self.image_shape)
        return ReconResult(
            image.transpose(),
            reference=-1,
            image_type="magnitude",
            dicom=True,
        )


# %% reconstruction -- operator vocabulary, no hand-written transform


def sensitivities(kspace: Any, mask: Any, *, device: Any = None) -> Any:
    """Estimate coil sensitivities from the calibration block of one slice.

    The mask says which positions the scan took, so NLINV reads the calibration
    region off it: the fully sampled square around the centre of k-space. It
    solves the maps at that size and resamples them to the full matrix by
    zero-filling in Fourier space, which is the smooth interpolation a
    sensitivity map wants. Nothing outside the block enters the estimate, so it
    does not matter whether the slice around it has been filled in yet.

    Parameters
    ----------
    kspace
        One slice's k-space, ``(coil, phase encode, readout)``.
    mask
        Its sampling mask.
    device
        Torch device. ``None`` is the CPU.

    Returns
    -------
    torch.Tensor
        Sensitivities, ``(1, coil, phase encode, readout)``.

    Raises
    ------
    ValueError
        If no fully sampled block surrounds the centre of k-space.
    """
    from pulserver.recon.calibration import NLINV

    return NLINV()(
        _tensor(kspace, "complex64", device)[None],
        mask=_tensor(mask, "bool", device),
    )


def coil_images(kspace: Any, mask: Any, *, device: Any = None) -> Any:
    """Return one image per coil, from a coil-wise Cartesian adjoint.

    A Cartesian operator built with no sensitivity maps is coil-wise: its
    adjoint zero-fills, inverts and hands back one image per coil, without
    folding a coil combination into the transform.

    Parameters
    ----------
    kspace
        One slice's k-space, ``(coil, phase encode, readout)``.
    mask
        Its sampling mask.
    device
        Torch device. ``None`` is the CPU.

    Returns
    -------
    numpy.ndarray
        Complex coil images, ``(coil, phase encode, readout)``.
    """
    from pulserver.recon.physics import Cartesian2D

    _, n_y, n_x = kspace.shape
    physics = Cartesian2D(_device_mask(mask, device)[None, None], img_size=(n_y, n_x))
    # Native complex throughout: the coil-wise adjoint takes the measurement and
    # answers with one complex image per coil, and the NumPy in, NumPy out
    # boundary means no hand conversion around it.
    return physics.A_adjoint(kspace[None])[0]


def sense(
    kspace: Any,
    mask: Any,
    coil_maps: Any,
    readout: Any,
    *,
    regularization: float = 1e-3,
    iterations: int = 40,
    pocs_iterations: int = 12,
    device: Any = None,
) -> Any:
    """Unalias one slice against its sensitivities, then fill a partial echo.

    Parameters
    ----------
    kspace
        One slice's k-space, ``(coil, phase encode, readout)``.
    mask
        Its sampling mask.
    coil_maps
        Sensitivities, as :func:`sensitivities` returns them.
    readout
        Which readout samples were acquired, over the full width.
    regularization, iterations
        Tikhonov weight and iteration ceiling of the CG solve.
    pocs_iterations
        POCS iterations, used only when the echo is partial.
    device
        Torch device. ``None`` is the CPU.

    Returns
    -------
    numpy.ndarray
        The complex image, ``(phase encode, readout)``.
    """
    from pulserver.recon import pics
    from pulserver.recon.physics import Cartesian2D

    physics = Cartesian2D(_device_mask(mask, device)[None], coil_maps)
    # The measurement crosses the boundary as NumPy and the unaliased image
    # comes straight back the same way -- no real/complex view juggling.
    image = pics(
        kspace[None],
        physics,
        regularization=regularization,
        iterations=iterations,
    )[0]
    if readout.all():
        return image
    # The same operator supplies the centered k-space POCS needs, and the phase
    # constraint is meaningful only now that the aliasing is gone.
    centered = physics.fft(_tensor(image, "complex64", device))
    return _asnumpy(
        fill_partial_echo(centered, readout, pocs_iterations, device=device)
    )


def fill_partial_echo(
    kspace: Any, readout: Any, iterations: int = 12, *, device: Any = None
) -> Any:
    """Recover the readout edge a partial echo never acquired.

    Parameters
    ----------
    kspace
        K-space over the full readout width, coil-wise or combined.
    readout
        Which readout samples were acquired.
    iterations
        POCS iterations.
    device
        Torch device. ``None`` is the CPU.

    Returns
    -------
    array
        The filled image, one per input channel, in the namespace of ``kspace``.
    """
    return POCS(dimension=2, partial_axis=-1, iterations=iterations)(
        kspace, _tensor(readout, "bool", device)
    )


def _tensor(array: Any, dtype: str, device: Any) -> Any:
    """Put one array on the reconstruction's device, in the dtype named.

    Only the calibration and POCS helpers -- which are not MRI physics and so
    do not carry the automatic array boundary -- still build tensors by hand;
    ``dtype`` is a Torch dtype name so nothing here imports Torch itself.
    """
    import torch

    return torch.as_tensor(
        array, dtype=getattr(torch, dtype), device="cpu" if device is None else device
    )


def _device_mask(mask: Any, device: Any) -> Any:
    """The sampling mask as a float tensor the Cartesian physics reads its device from."""
    return _tensor(mask, "float32", device)


def _asnumpy(array: Any) -> np.ndarray:
    """Bring a reconstruction back to NumPy, whether it is Torch or already so."""
    return array.cpu().numpy() if hasattr(array, "cpu") else np.asarray(array)


PLUGIN = Gre2DRecon()
