"""Density-compensation estimators backed by MRI-NUFFT."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["pipe_menon_dcf"]


def pipe_menon_dcf(
    trajectory: Any,
    image_shape: tuple[int, ...],
    *,
    backend: str = "finufft",
    **kwargs: Any,
) -> Any:
    """Estimate Pipe--Menon density-compensation weights with MRI-NUFFT.

    ``kwargs`` are passed unchanged to the selected backend's ``pipe``
    implementation, including options such as ``max_iter`` and
    normalization. The returned array remains in the array/device ecosystem
    selected by MRI-NUFFT.
    """
    if len(image_shape) not in (2, 3) or any(int(item) < 1 for item in image_shape):
        raise ValueError("image_shape must contain two or three positive entries")
    try:
        density = import_module("mrinufft.density")
    except ImportError as error:
        raise ImportError(
            "Pipe-Menon DCF estimation requires mri-nufft; install "
            "pulserver[recon-cpu] or pulserver[recon-cuda]."
        ) from error
    return density.pipe(
        trajectory,
        tuple(int(item) for item in image_shape),
        backend=backend,
        **kwargs,
    )


pipe = pipe_menon_dcf
PipeMenonDCF = pipe_menon_dcf
