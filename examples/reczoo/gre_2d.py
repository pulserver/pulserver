"""Reconstruction for :mod:`pulserver.seqzoo.gre_2d`.

This is a Gadgetron/ISMRMRD-style *streamed* reconstruction, not an offline
solver handed a finished k-space array. The runtime hands over ``Acquisition``
objects while the scan runs, and the plugin is three hooks over them:

Allocate, once
    :meth:`~Gre2DRecon.startup` reads the encoded grid out of the MRD header
    and allocates the whole scan's k-space --
    ``(slices, coils, phase encodes, readout)``. The grid is the encoded one,
    so the readout axis is the oversampled width the scanner digitises and the
    phase-encode axis spans the prescribed matrix rather than the lines this
    acceleration happens to sample.

Place, per acquisition
    :meth:`~Gre2DRecon.receive` puts each line where its counters say it
    belongs. A partial echo is right-aligned in the readout axis by the buffer
    itself, so nothing downstream has to know the echo was truncated.

Reconstruct, at each boundary
    The sequence acquires its autocalibration block first and flags the end of
    it, and flags the end of each slice. :meth:`~Gre2DRecon.recon` runs at both
    and decides which it is: the earlier one estimates the coil sensitivities
    and produces no image, so the calibration is done while the rest of the
    slice is still arriving; the later one reconstructs.

Those calibration lines are imaging data as well as calibration data, so there
is one buffer and no second grid -- the calibration region is the fully sampled
block at the centre of it, which the sampling mask already says where to find.

Which of three reconstructions runs is read off the sampling mask, never
declared:

* fully sampled, full echo -- the coil-wise adjoint (a centered inverse
  transform per coil), the coils combined by root-sum-of-squares;
* a readout truncated before the echo -- phase-constrained POCS fills the
  missing edge against a full-width readout axis;
* phase encodes missing -- CG-SENSE against the NLINV coil maps.

Parallel imaging comes first and partial Fourier second: the phase constraint
POCS relies on is only meaningful once the image is unaliased.

The numerics are standalone functions, expressed in the framework's
DeepInverse-style operator vocabulary -- a
:class:`~pulserver.recon.physics.Cartesian2D` operator, its coil-wise adjoint,
:func:`~pulserver.recon.pics`, NLINV and POCS, with the coils combined by an
explicit :func:`~pulserver.recon.postprocessing.coil_combine`. There is no
hand-written FFT anywhere: a Cartesian operator with no sensitivities *is* the
centered transform, one image per coil.

The result is one magnitude image per slice, cropped from the encoded grid to
the prescribed matrix. The plugin does not build the image header: it returns a
:class:`ReconResult` naming the acquisition its geometry comes from and asking
for DICOM, and the runtime reads that acquisition's header (slice position,
orientation) and the MRD header's recon field of view to make the
``ismrmrd.Image`` and the DICOM series.

Needs the numerical stack: ``pip install "pulserver[recon-cpu]"``.
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

        # A segment that ends without ending its slice is the calibration
        # block: estimate the sensitivities now, and let the rest of the slice
        # keep arriving. There is no image to send yet.
        last = bucket.acquisitions[-1]
        if has_acquisition_flag(last, "ACQ_LAST_IN_SEGMENT") and not (
            has_acquisition_flag(last, "ACQ_LAST_IN_SLICE")
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
        image = center_crop(np.abs(_to_numpy(image)), self.image_shape)
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
    import torch

    from pulserver.recon.calibration import NLINV

    return NLINV()(
        _tensor(kspace, torch.complex64, device)[None],
        mask=_tensor(mask, torch.bool, device),
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
    torch.Tensor
        Complex coil images, ``(coil, phase encode, readout)``.
    """
    import torch

    from pulserver.recon.physics import Cartesian2D

    _, n_y, n_x = kspace.shape
    physics = Cartesian2D(
        _tensor(mask, torch.float32, device)[None, None], img_size=(n_y, n_x)
    )
    coils = physics.A_adjoint(
        torch.view_as_real(_tensor(kspace, torch.complex64, device)[None])
    )
    return torch.view_as_complex(coils.movedim(1, -1).contiguous())[0]


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
    torch.Tensor
        The complex image, ``(phase encode, readout)``.
    """
    import torch

    from pulserver.recon import pics
    from pulserver.recon.physics import Cartesian2D

    physics = Cartesian2D(_tensor(mask, torch.float32, device)[None], coil_maps)
    solution = pics(
        torch.view_as_real(_tensor(kspace, torch.complex64, device)[None]),
        physics,
        regularization=regularization,
        iterations=iterations,
    )
    image = torch.view_as_complex(solution.movedim(1, -1).contiguous())[0]
    if readout.all():
        return image
    # The same operator supplies the centered k-space POCS needs, and the phase
    # constraint is meaningful only now that the aliasing is gone.
    return fill_partial_echo(
        physics.fft(image), readout, pocs_iterations, device=device
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
    torch.Tensor
        The filled image, one per input channel.
    """
    import torch

    return POCS(dimension=2, partial_axis=-1, iterations=iterations)(
        kspace, _tensor(readout, torch.bool, device)
    )


def _tensor(array: Any, dtype: Any, device: Any) -> Any:
    """Put one array on the reconstruction's device, in the dtype asked for."""
    import torch

    return torch.as_tensor(
        array, dtype=dtype, device="cpu" if device is None else device
    )


def _to_numpy(array: Any) -> np.ndarray:
    """Bring a reconstruction back to NumPy, whether it is Torch or already so."""
    return array.cpu().numpy() if hasattr(array, "cpu") else np.asarray(array)


PLUGIN = Gre2DRecon()
