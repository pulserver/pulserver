"""Cartesian parallel-imaging compressed-sensing reconstruction with bartorch, as ``bart ecalib`` and ``bart pics``."""

from __future__ import annotations

__all__ = ["PLUGIN", "PicsRecon"]

import numpy as np

from ...mrd._images import center_crop
from .cartesian import CartesianRecon


class PicsRecon(CartesianRecon):
    """Images of :class:`~pulserver.recon.handlers.cartesian.CartesianRecon`, each a ``pics`` solve over ESPIRiT maps.

    The k-space of each image, unsampled lines zero, gives one set of ESPIRiT
    sensitivities from its centre (``bart ecalib -m 1``), and the image
    minimises ``|P F S x - y|^2 + lambda |W x|_1`` over them (``bart pics -R
    W``), ``W`` the wavelet transform over the encoded axes. The sampled
    centre must span the calibration region, 24 lines by default. bartorch is
    imported when the first image is made, which the ``coils`` extra installs.

    Parameters
    ----------
    wavelet
        ``lambda``, relative to the data scaling ``pics`` estimates.
    iterations
        Iterations of the solve.
    """

    def __init__(self, wavelet: float = 0.005, iterations: int = 30) -> None:
        super().__init__()
        self.wavelet = wavelet
        self.iterations = iterations

    def image(
        self, kspace: np.ndarray, shape: tuple[int, ...], device: str | None
    ) -> np.ndarray:
        import torch
        from bartorch import apps, priors, tools

        volume = torch.from_numpy(
            np.ascontiguousarray(kspace if kspace.ndim == 4 else kspace[:, None])
        ).to(device=device, dtype=torch.complex64)
        maps = tools.ecalib(volume, maps=1).reshape(volume.shape)
        encoded = (-1, -2) if volume.shape[1] == 1 else (-1, -2, -3)
        image = apps.pics(
            volume,
            maps,
            regularizers=priors.Wavelet(encoded, self.wavelet),
            maxiter=self.iterations,
        )
        image = image.abs().cpu().numpy().reshape(volume.shape[1:])
        if kspace.ndim == 3:
            image = image[0]
        return np.array(center_crop(image, shape[-image.ndim :]))


PLUGIN = PicsRecon()
