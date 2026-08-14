"""Reconstruction for :mod:`pulserver.seqzoo.gre_3d`.

The 2D pipeline of :mod:`pulserver.reczoo.gre_2d`, lifted to a volume: one
Cartesian grid over ``(partition, line, readout)``, filled as acquisitions
arrive, calibrated when the autocalibration rectangle's segment closes, and
reconstructed once at the end of the measurement -- a 3D scan has one slab,
so there is no per-slice boundary to reconstruct at.

Which of three reconstructions runs is read off the sampling mask rather than
declared: the coil-wise adjoint when everything is there, POCS when the
readout is truncated, CG-SENSE against 3D NLINV maps when phase encodes are
missing on either axis. Parallel imaging comes first, because the phase
constraint POCS relies on only means something once the image is unaliased.

One magnitude volume, cropped to the prescribed matrix. Needs the numerical
stack: ``pip install "pulserver[recon-cpu]"``.
"""

from __future__ import annotations

__all__ = [
    "PLUGIN",
    "Gre3DRecon",
    "coil_volumes",
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
    receiver_channels,
)


def encoded_volume(header: Any, *, encoding: int = 0) -> tuple[int, int, int]:
    """Return the ``(n_z, n_y, n_x)`` grid an MRD header describes for a slab.

    The volume counterpart of
    :func:`pulserver.recon.preprocessing.encoded_shape`: the partition extent
    comes from the encoded matrix rather than the slice counter, because a 3D
    scan encodes z instead of stepping through it.

    Parameters
    ----------
    header
        Parsed MRD XML header.
    encoding
        Encoding space to read.

    Returns
    -------
    tuple of int
        Partitions, phase encodes, readout samples.

    Raises
    ------
    ValueError
        If the header carries no such encoded space.
    """
    return _matrix(header, encoding, "encodedSpace")


def recon_volume(header: Any, *, encoding: int = 0) -> tuple[int, int, int]:
    """Return the ``(n_z, n_y, n_x)`` image matrix an MRD header asks for.

    Parameters
    ----------
    header
        Parsed MRD XML header.
    encoding
        Encoding space to read.

    Returns
    -------
    tuple of int
        Partitions, phase encodes, readout samples.

    Raises
    ------
    ValueError
        If the header carries no such reconstructed space.
    """
    return _matrix(header, encoding, "reconSpace")


def _matrix(header: Any, encoding: int, name: str) -> tuple[int, int, int]:
    try:
        space = getattr(header.encoding[encoding], name)
        size = space.matrixSize
    except (AttributeError, IndexError, TypeError) as error:
        raise ValueError(
            f"MRD encoding {encoding} carries no {name} matrix size"
        ) from error
    return int(size.z), int(size.y), int(size.x)


class Gre3DRecon(ReconPlugin):
    """Reconstruct a 3D Cartesian gradient echo, one volume per measurement.

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
            split_on=("ACQ_LAST_IN_SEGMENT", "ACQ_LAST_IN_MEASUREMENT"),
            reject_flags=("ACQ_IS_NOISE_MEASUREMENT", "ACQ_IS_PHASECORR_DATA"),
        )
        self.regularization = float(regularization)
        self.iterations = int(iterations)
        self.pocs_iterations = int(pocs_iterations)
        self.device = device

    def startup(self, context: ReconContext) -> None:
        """Allocate the scan's k-space from the grid the header describes."""
        self.buffer = CartesianGridder(
            encoded_volume(context.header),
            coils=receiver_channels(context.header),
        )
        self.image_shape = recon_volume(context.header)
        self.coil_maps: Any = None

    def receive(self, acquisition: Any, context: ReconContext) -> None:
        """Place one arriving line at its partition and phase encode."""
        del context
        self.buffer.add(
            acquisition.data,
            acquisition.idx.kspace_encode_step_2,
            acquisition.idx.kspace_encode_step_1,
        )

    def recon(
        self, bucket: AcquisitionBucket, context: ReconContext
    ) -> ReconResult | None:
        """Calibrate or reconstruct, according to which boundary ended this bucket."""
        del context
        kspace, mask = self.buffer.kspace, self.buffer.mask

        # A segment that ends before the measurement does is the calibration
        # rectangle: estimate the sensitivities now, and let the rest of the
        # volume keep arriving. There is no image to send yet.
        last = bucket.acquisitions[-1]
        if not has_acquisition_flag(last, "ACQ_LAST_IN_MEASUREMENT"):
            self.coil_maps = sensitivities(kspace, mask, device=self.device)
            return None

        # What the scan sampled is what selects the reconstruction: a phase
        # encode with no samples was skipped, and a readout sample missing
        # from every line is echo the sequence never acquired.
        encodes = mask.any(axis=-1)
        readout = mask.reshape(-1, mask.shape[-1]).any(axis=0)

        if encodes.all():
            coils = (
                coil_volumes(kspace, mask, device=self.device)
                if readout.all()
                else fill_partial_echo(
                    kspace, readout, self.pocs_iterations, device=self.device
                )
            )
            image = coil_combine(coils, coil_axis=0)
        else:
            maps = self.coil_maps
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

        # One magnitude volume, cropped from the encoded grid to the
        # prescribed matrix, its in-plane axes transposed to the column/row
        # order an image is read in. ``reference`` names the acquisition that
        # ended the bucket, so the runtime copies the slab's geometry into the
        # image; ``dicom=True`` asks it to emit one.
        image = center_crop(np.abs(_to_numpy(image)), self.image_shape)
        return ReconResult(
            image.transpose(0, 2, 1),
            reference=-1,
            image_type="magnitude",
            dicom=True,
        )


# %% reconstruction -- operator vocabulary, no hand-written transform


def sensitivities(kspace: Any, mask: Any, *, device: Any = None) -> Any:
    """Estimate 3D coil sensitivities from the calibration rectangle.

    The mask says which positions the scan took, so NLINV reads the
    calibration region off it: the fully sampled block around the centre of
    k-space. It solves the maps at that size and resamples them to the full
    matrix by zero-filling in Fourier space. Nothing outside the block enters
    the estimate, so it does not matter whether the volume around it has been
    filled in yet.

    Parameters
    ----------
    kspace
        The volume's k-space, ``(coil, partition, phase encode, readout)``.
    mask
        Its sampling mask.
    device
        Torch device. ``None`` is the CPU.

    Returns
    -------
    torch.Tensor
        Sensitivities, ``(1, coil, partition, phase encode, readout)``.

    Raises
    ------
    ValueError
        If no fully sampled block surrounds the centre of k-space.
    """
    import torch

    from pulserver.recon.calibration import NLINV

    return NLINV(spatial_ndim=3)(
        _tensor(kspace, torch.complex64, device)[None],
        mask=_tensor(mask, torch.bool, device),
    )


def coil_volumes(kspace: Any, mask: Any, *, device: Any = None) -> Any:
    """Return one volume per coil, from a coil-wise Cartesian adjoint.

    Parameters
    ----------
    kspace
        The volume's k-space, ``(coil, partition, phase encode, readout)``.
    mask
        Its sampling mask.
    device
        Torch device. ``None`` is the CPU.

    Returns
    -------
    torch.Tensor
        Complex coil volumes, ``(coil, partition, phase encode, readout)``.
    """
    import torch

    from pulserver.recon.physics import Cartesian3D

    _, n_z, n_y, n_x = kspace.shape
    physics = Cartesian3D(
        _tensor(mask, torch.float32, device)[None, None], img_size=(n_z, n_y, n_x)
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
    """Unalias the volume against its sensitivities, then fill a partial echo.

    Parameters
    ----------
    kspace
        The volume's k-space, ``(coil, partition, phase encode, readout)``.
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
        The complex volume, ``(partition, phase encode, readout)``.
    """
    import torch

    from pulserver.recon import pics
    from pulserver.recon.physics import Cartesian3D

    physics = Cartesian3D(_tensor(mask, torch.float32, device)[None], coil_maps)
    solution = pics(
        torch.view_as_real(_tensor(kspace, torch.complex64, device)[None]),
        physics,
        regularization=regularization,
        iterations=iterations,
    )
    image = torch.view_as_complex(solution.movedim(1, -1).contiguous())[0]
    if readout.all():
        return image
    # The same operator supplies the centered k-space POCS needs, and the
    # phase constraint is meaningful only now that the aliasing is gone.
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
        The filled volume, one per input channel.
    """
    import torch

    return POCS(dimension=3, partial_axis=-1, iterations=iterations)(
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


PLUGIN = Gre3DRecon()
