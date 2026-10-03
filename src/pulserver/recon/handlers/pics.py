"""Cartesian parallel-imaging compressed-sensing reconstruction with bartorch, as ``bart nlinv``, ``bart pics`` and ``bart homodyne``."""

from __future__ import annotations

__all__ = ["PLUGIN", "PicsRecon"]

import numpy as np

from ...mrd._acquisitions import AcquisitionFlag
from ...mrd._images import center_crop
from .._buffers import ReconData
from .._calibration import coil_maps
from ..gadgets import AsymmetricEcho, Prewhiten, RemoveReadoutOversampling
from ..plugin import ReconContext, ReconResult
from .cartesian import CartesianRecon


class PicsRecon(CartesianRecon):
    """Images of :class:`~pulserver.recon.handlers.cartesian.CartesianRecon`, each a ``pics`` solve over coil sensitivities.

    Each readout is whitened with the noise measurement of the stream
    (:class:`~pulserver.recon.Prewhiten`), completed to a full echo
    (:class:`~pulserver.recon.AsymmetricEcho`), has its readout oversampling
    removed (:class:`~pulserver.recon.RemoveReadoutOversampling`) and is placed
    by its echo along the readout and by the offset of its lines from the
    k-space centre, so the k-space is at the reconstruction matrix when its
    slice closes. Noise readouts are consumed by the whitening;
    phase-correction readouts are rejected.

    The coil sensitivities are those of :func:`~pulserver.recon.coil_maps`
    with :func:`bartorch.apps.nlinv_maps` as the estimate: from the unit's
    calibration readouts, else the maps stored for its slice, else the exam's.
    A unit for which none of these has maps takes them from the
    low-resolution centre of its own k-space (``bart nlinv -m 1``), with unit
    sensitivity for one coil. A unit that holds calibration readouts and no
    imaging readouts makes no image; its maps are stored for the later units
    of its slice. The image minimises ``|P F S x - y|^2 + lambda |W x|_1``
    over the maps (``bart pics -R W``), ``W`` the wavelet transform over the
    encoded axes. An encoded axis is partial Fourier where the lines missing
    at one end outnumber the widest gap between sampled lines, and
    :func:`bartorch.apps.partial_fourier` completes the image along it
    (``bart homodyne -I -C``). bartorch is imported when the first unit is
    reconstructed, which the ``coils`` extra installs.

    Parameters
    ----------
    wavelet
        ``lambda``, relative to the data scaling ``pics`` estimates.
    iterations
        Iterations of the solve.
    """

    def __init__(self, wavelet: float = 0.005, iterations: int = 30) -> None:
        super().__init__()
        self.gadgets = (Prewhiten(), AsymmetricEcho(), RemoveReadoutOversampling())
        self.reject_flags = (AcquisitionFlag.IS_PHASECORR_DATA,)
        self.wavelet = wavelet
        self.iterations = iterations

    def recon(
        self, context: ReconContext, branch: str, data: ReconData
    ) -> ReconResult | None:
        if data.data is None and data.ref is not None:
            from bartorch import apps

            coil_maps(context, data, estimate=apps.nlinv_maps)
        return super().recon(context, branch, data)

    def image(
        self,
        kspace: np.ndarray,
        shape: tuple[int, ...],
        context: ReconContext,
        data: ReconData,
    ) -> np.ndarray:
        import torch
        from bartorch import apps, priors

        volume = kspace if kspace.ndim == 4 else kspace[:, None]
        volume = torch.from_numpy(np.ascontiguousarray(volume)).to(
            device=context.device, dtype=torch.complex64
        )
        maps = coil_maps(context, data, estimate=apps.nlinv_maps, required=False)
        maps = apps.nlinv_maps(volume) if maps is None else maps.reshape(volume.shape)
        encoded = (-1, -2) if volume.shape[1] == 1 else (-1, -2, -3)
        image = apps.pics(
            volume,
            maps,
            regularizers=priors.Wavelet(encoded, self.wavelet),
            maxiter=self.iterations,
        ).reshape(volume.shape[1:])
        image = apps.partial_fourier(image, volume.abs().sum(dim=0) > 0)
        image = image.abs().cpu().numpy()
        if kspace.ndim == 3:
            image = image[0]
        return np.array(center_crop(image, shape[-image.ndim :]))


PLUGIN = PicsRecon()
