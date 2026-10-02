"""Cartesian parallel-imaging compressed-sensing reconstruction with bartorch, as ``bart nlinv``, ``bart pics`` and ``bart homodyne``."""

from __future__ import annotations

__all__ = ["PLUGIN", "PicsRecon", "RemoveReadoutOversampling"]

import numpy as np

from ...mrd._acquisitions import AcquisitionFlag
from ...mrd._header import EncodingSpace
from ...mrd._images import center_crop
from ...mrd._metadata import acquisition_label
from .._buffers import ReconData
from .._units import carries
from ..plugin import Gadget
from .cartesian import CartesianRecon

#: The flags either of which marks a navigator readout: pypulseqpp's ``NAV``
#: label, or ``RTFEEDBACK``.
NAVIGATOR = AcquisitionFlag.IS_NAVIGATION_DATA | AcquisitionFlag.IS_RTFEEDBACK_DATA


class RemoveReadoutOversampling(Gadget):
    """Crop each readout to the readout extent of its space's reconstruction matrix.

    The readout is Fourier transformed, cropped to the centre of the readout
    field of view and transformed back, as Gadgetron's
    ``RemoveROOversamplingGadget``. A partial echo is zero-filled to the full
    echo for the transform, ``center_sample`` locating the echo, and only its
    acquired part is returned, so the buffer still sees a partial echo.
    Readouts not longer than the matrix pass unchanged, and so do navigator
    readouts (:data:`NAVIGATOR`), which are not sampled on the imaging grid.
    """

    def startup(self, context):
        super().startup(context)
        encodings = getattr(context.header, "encoding", None) or ()
        self.columns = [
            EncodingSpace.from_header(context.header, index).recon_matrix[-1]
            for index in range(len(encodings))
        ]

    def __call__(self, acquisition, data):
        space = int(acquisition_label(acquisition, "encoding_space_ref", 0) or 0)
        if space >= len(self.columns):
            return data
        columns = self.columns[space]
        samples = data.shape[-1]
        centre = int(acquisition_label(acquisition, "center_sample", 0) or 0)
        full = max(samples, 2 * (samples - centre))
        if full <= columns or carries(acquisition, NAVIGATOR):
            return data
        echo = np.zeros((data.shape[0], full), dtype=np.complex64)
        echo[:, full - samples :] = data
        profile = _centred(np.fft.ifft, echo)
        start = (full - columns) // 2
        cropped = _centred(np.fft.fft, profile[:, start : start + columns])
        acquired = columns - round((full - samples) * columns / full)
        return cropped[:, columns - acquired :]


class PicsRecon(CartesianRecon):
    """Images of :class:`~pulserver.recon.handlers.cartesian.CartesianRecon`, each a ``pics`` solve over nlinv sensitivities.

    Each readout has its oversampling removed on arrival
    (:class:`RemoveReadoutOversampling`) and is placed by its counters, so the
    k-space is complete and at the reconstruction matrix when its slice
    closes. It then gives one set of coil sensitivities by nonlinear inversion
    of its low-resolution centre (:func:`sensitivities`; unit sensitivity for
    one coil), and the image minimises ``|P F S x - y|^2 + lambda
    |W x|_1`` over them (``bart pics -R W``), ``W`` the wavelet transform over
    the encoded axes. An encoded axis sampled on one side only, by more lines
    than the widest gap between sampled lines, is partial Fourier: the image
    is completed along it by ``bart homodyne -I -C``. bartorch is imported
    when the first image is made, which the ``coils`` extra installs.

    Parameters
    ----------
    wavelet
        ``lambda``, relative to the data scaling ``pics`` estimates.
    iterations
        Iterations of the solve.
    """

    def __init__(self, wavelet: float = 0.005, iterations: int = 30) -> None:
        super().__init__()
        self.gadgets = (RemoveReadoutOversampling(),)
        self.wavelet = wavelet
        self.iterations = iterations

    def image(
        self,
        kspace: np.ndarray,
        shape: tuple[int, ...],
        device: str | None,
        data: ReconData | None = None,
    ) -> np.ndarray:
        del data
        import torch
        from bartorch import apps, priors, tools

        volume = kspace if kspace.ndim == 4 else kspace[:, None]
        sampled = np.abs(volume).sum(axis=0) > 0
        volume = torch.from_numpy(np.ascontiguousarray(volume)).to(
            device=device, dtype=torch.complex64
        )
        maps = sensitivities(volume) if volume.shape[0] > 1 else torch.ones_like(volume)
        encoded = (-1, -2) if volume.shape[1] == 1 else (-1, -2, -3)
        image = apps.pics(
            volume,
            maps,
            regularizers=priors.Wavelet(encoded, self.wavelet),
            maxiter=self.iterations,
        ).reshape(volume.shape[1:])
        for axis in range(image.ndim):
            acquired, high = partial_fourier(sampled, axis)
            if acquired == 1.0:
                continue
            # bart homodyne takes the acquired side at the low indices.
            if high:
                image = torch.flip(image, (axis,))
            image = tools.homodyne(
                axis, acquired, image.contiguous(), I=True, C=True
            ).reshape(image.shape)
            if high:
                image = torch.flip(image, (axis,))
        image = image.abs().cpu().numpy()
        if kspace.ndim == 3:
            image = image[0]
        return np.array(center_crop(image, shape[-image.ndim :]))


PLUGIN = PicsRecon()


def partial_fourier(sampled: np.ndarray, axis: int) -> tuple[float, bool]:
    """Return the acquired fraction of ``axis`` and whether its high indices are the acquired side.

    ``(1.0, False)`` unless the lines missing at one end outnumber the widest
    gap between sampled lines, which undersampling alone leaves.
    """
    profile = sampled.any(axis=tuple(a for a in range(sampled.ndim) if a != axis))
    lines = np.flatnonzero(profile)
    n = profile.size
    if lines.size < 2:
        return 1.0, False
    gap = int(np.diff(lines).max()) - 1
    low, high = int(lines[0]), n - 1 - int(lines[-1])
    missing = max(low, high)
    if missing <= gap:
        return 1.0, False
    return (n - missing) / n, low > high


def sensitivities(volume, size: int = 24):
    """Return one set of coil sensitivities of ``(coils, z, y, x)`` k-space, as ``bart nlinv -m 1``.

    The fit is to the central ``size`` lines of every encoded axis, the rest
    zero-filled, so the sensitivities are smooth at the full matrix.
    """
    import torch
    from bartorch import tools

    centre = tuple(
        slice((n - min(size, n)) // 2, (n + min(size, n)) // 2)
        for n in volume.shape[1:]
    )
    low = torch.zeros_like(volume)
    low[(slice(None), *centre)] = volume[(slice(None), *centre)]
    _, maps = tools.nlinv(low, maps=1, return_sensitivities=True)
    maps = maps.reshape(volume.shape)
    return maps / maps.abs().square().sum(dim=0, keepdim=True).sqrt().clamp_min(1e-12)


def _centred(transform, data):
    return np.fft.fftshift(transform(np.fft.ifftshift(data, axes=-1)), axes=-1)
