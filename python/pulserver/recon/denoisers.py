"""DeepInverse denoiser factories used by :func:`pulserver.recon.pics`."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "TGV",
    "TV",
    "Wavelet",
    "denoiser",
    "tgv",
    "tv",
    "wavelet",
]


def _models() -> Any:
    try:
        return import_module("deepinv.models")
    except ImportError as error:
        raise ImportError(
            "Reconstruction denoisers require DeepInverse; install "
            "pulserver[recon-cpu] or pulserver[recon-cuda]."
        ) from error


def wavelet(
    *,
    dimension: int = 2,
    wavelet: str = "db8",
    level: int = 3,
    complex_data: bool = False,
    **kwargs: Any,
) -> Any:
    """Create DeepInverse's 2D or 3D orthogonal-wavelet denoiser.

    Batch entries are independent, which makes the same object suitable for
    slices, contrasts, and dynamic frames. ``dimension`` selects only the
    spatial transform dimensionality.
    """
    if dimension not in (2, 3):
        raise ValueError("dimension must be 2 or 3")
    return _models().WaveletDenoiser(
        wvdim=dimension,
        wv=wavelet,
        level=level,
        is_complex=complex_data,
        **kwargs,
    )


Wavelet = wavelet


def tv(**kwargs: Any) -> Any:
    """Create DeepInverse's spatially 2D/3D-agnostic TV denoiser."""
    return _models().TVDenoiser(**kwargs)


TV = tv


def tgv(**kwargs: Any) -> Any:
    """Create DeepInverse's spatially 2D/3D-agnostic TGV denoiser."""
    return _models().TGVDenoiser(**kwargs)


TGV = tgv


def denoiser(name: str, **kwargs: Any) -> Any:
    """Create one of the public ``wavelet``, ``tv``, or ``tgv`` denoisers."""
    factories = {"wavelet": wavelet, "tv": tv, "tgv": tgv}
    try:
        factory = factories[name.lower()]
    except KeyError as error:
        choices = ", ".join(factories)
        raise ValueError(
            f"Unknown denoiser {name!r}; choose one of {choices}"
        ) from error
    return factory(**kwargs)
