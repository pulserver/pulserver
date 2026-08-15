"""Minimal inline/offline Cartesian reconstruction plugin."""

from __future__ import annotations

__all__ = ["PLUGIN", "CartesianFftRecon"]

import numpy as np

from pulserver import AcquisitionBucket, ReconPlugin, ReconContext, ReconResult
from pulserver.recon.postprocessing import coil_combine
from pulserver.recon.preprocessing import ifftc


class CartesianFftRecon(ReconPlugin):
    """Reconstruct one Cartesian slice without handling an MRD connection."""

    def __init__(self) -> None:
        super().__init__(
            split_on="ACQ_LAST_IN_SLICE",
            reject_flags=(
                "ACQ_IS_NOISE_MEASUREMENT",
                "ACQ_IS_PHASECORR_DATA",
            ),
        )

    def recon(
        self,
        bucket: AcquisitionBucket,
        context: ReconContext,
    ) -> ReconResult:
        """Apply a centred IFFT and root-sum-of-squares coil combination."""
        del context
        # AcquisitionBucket.kspace() is [acquisition, coil, readout].
        kspace = np.moveaxis(bucket.kspace(), 0, -1)
        image = ifftc(kspace, axes=(-2, -1))
        image = coil_combine(image, coil_axis=0)
        return ReconResult(image.transpose())


PLUGIN = CartesianFftRecon()
