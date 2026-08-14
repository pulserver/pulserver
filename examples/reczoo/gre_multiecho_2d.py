"""Reconstruction for :mod:`pulserver.seqzoo.gre_multiecho_2d`.

The 2D Cartesian pipeline of :mod:`pulserver.reczoo.gre_2d`, once per echo:
the grid gains an echo axis fed by the ``contrast`` counter the sequence's
``ECO`` label becomes, the coil sensitivities are estimated once -- from the
first echo, whose calibration block has the most signal -- and each echo is
then unaliased and filled against the same maps. One magnitude image per
``(slice, echo)``, the echo as the image series.

Needs the numerical stack: ``pip install "pulserver[recon-cpu]"``.
"""

from __future__ import annotations

__all__ = ["PLUGIN", "GreMultiecho2DRecon", "echo_count"]

from typing import Any

import numpy as np

from pulserver import AcquisitionBucket, ReconContext, ReconPlugin, ReconResult
from pulserver.recon import has_acquisition_flag
from pulserver.recon.postprocessing import center_crop, coil_combine
from pulserver.recon.preprocessing import (
    CartesianGridder,
    encoded_shape,
    receiver_channels,
    recon_shape,
)
from pulserver.reczoo.gre_2d import (
    coil_images,
    fill_partial_echo,
    sense,
    sensitivities,
)


def echo_count(header: Any, *, encoding: int = 0) -> int:
    """Return how many echoes the MRD header declares.

    The sequence's ``ECO`` label arrives as the ``contrast`` counter, so its
    encoding limit is the echo count. A header without one is a single-echo
    scan.

    Parameters
    ----------
    header
        Parsed MRD XML header.
    encoding
        Encoding space to read.

    Returns
    -------
    int
        Echoes per repetition.
    """
    try:
        limits = header.encoding[encoding].encodingLimits.contrast
    except (AttributeError, IndexError, TypeError):
        return 1
    return 1 if limits is None else int(limits.maximum) + 1


class GreMultiecho2DRecon(ReconPlugin):
    """Reconstruct a multi-echo 2D Cartesian gradient echo, per slice and echo.

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
        """Allocate one grid per slice and echo, from the header."""
        n_slices, n_y, n_x = encoded_shape(context.header)
        self.n_echoes = echo_count(context.header)
        self.buffer = CartesianGridder(
            (n_slices, self.n_echoes, n_y, n_x),
            coils=receiver_channels(context.header),
        )
        self.image_shape = recon_shape(context.header)
        self.coil_maps: dict[int, Any] = {}

    def receive(self, acquisition: Any, context: ReconContext) -> None:
        """Place one arriving line at its slice, echo and phase encode."""
        del context
        self.buffer.add(
            acquisition.data,
            acquisition.idx.slice,
            acquisition.idx.contrast,
            acquisition.idx.kspace_encode_step_1,
        )

    def recon(
        self, bucket: AcquisitionBucket, context: ReconContext
    ) -> list[ReconResult] | None:
        """Calibrate or reconstruct, according to which boundary ended this bucket."""
        del context
        index = int(bucket.labels("slice")[-1])
        kspace, mask = self.buffer[index]

        # The calibration boundary: maps from the first echo, which carries
        # the most signal, and no image yet.
        last = bucket.acquisitions[-1]
        if has_acquisition_flag(last, "ACQ_LAST_IN_SEGMENT") and not (
            has_acquisition_flag(last, "ACQ_LAST_IN_SLICE")
            or has_acquisition_flag(last, "ACQ_LAST_IN_MEASUREMENT")
        ):
            self.coil_maps[index] = sensitivities(
                kspace[:, 0], mask[0], device=self.device
            )
            return None

        results = []
        for echo in range(self.n_echoes):
            echo_kspace, echo_mask = kspace[:, echo], mask[echo]
            lines = echo_mask.any(axis=-1)
            readout = echo_mask.any(axis=0)

            if lines.all():
                coils = (
                    coil_images(echo_kspace, echo_mask, device=self.device)
                    if readout.all()
                    else fill_partial_echo(
                        echo_kspace, readout, self.pocs_iterations, device=self.device
                    )
                )
                image = coil_combine(coils, coil_axis=0)
            else:
                maps = self.coil_maps.get(index)
                if maps is None:
                    maps = sensitivities(echo_kspace, echo_mask, device=self.device)
                    self.coil_maps[index] = maps
                image = sense(
                    echo_kspace,
                    echo_mask,
                    maps,
                    readout,
                    regularization=self.regularization,
                    iterations=self.iterations,
                    pocs_iterations=self.pocs_iterations,
                    device=self.device,
                )

            image = center_crop(np.abs(_to_numpy(image)), self.image_shape)
            results.append(
                ReconResult(
                    image.transpose(),
                    reference=-1,
                    series_index=echo,
                    image_type="magnitude",
                    dicom=True,
                )
            )
        return results


def _to_numpy(array: Any) -> np.ndarray:
    """Bring a reconstruction back to NumPy, whether it is Torch or already so."""
    return array.cpu().numpy() if hasattr(array, "cpu") else np.asarray(array)


PLUGIN = GreMultiecho2DRecon()
