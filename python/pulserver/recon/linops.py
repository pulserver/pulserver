"""Access maintained MRI linear operators without copying their implementations.

MRPro owns Cartesian SENSE, Fourier, sampling, and NUFFT operators.  mri-nufft
owns its FINUFFT/CUFINUFFT operator backends.  This module deliberately only
selects and returns those upstream objects.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "available_nufft_backends",
    "deepinverse_multicoil_mri",
    "mrinufft_operator",
    "mrpro_operator",
]

_MRPRO_OPERATORS = {
    "cartesian_mask": "CartesianMaskingOp",
    "cartesian_sampling": "CartesianSamplingOp",
    "fast_fourier": "FastFourierOp",
    "fourier": "FourierOp",
    "nufft": "NonUniformFastFourierOp",
    "sensitivity": "SensitivityOp",
}


def mrpro_operator(name: str) -> type[Any]:
    """Return a maintained MRPro operator class by role.

    ``name`` is one of ``cartesian_mask``, ``cartesian_sampling``,
    ``fast_fourier``, ``fourier``, ``nufft``, or ``sensitivity``.  Compose the
    returned MRPro classes directly (for example ``FourierOp @ SensitivityOp``)
    and use MRPro's reconstruction algorithms with MRPro ``KData`` objects.
    """
    try:
        class_name = _MRPRO_OPERATORS[name]
    except KeyError as error:
        choices = ", ".join(sorted(_MRPRO_OPERATORS))
        raise ValueError(f"Unknown MRPro operator {name!r}; choose one of {choices}") from error
    try:
        module = import_module("mrpro.operators")
    except ImportError as error:
        raise ImportError("MRPro operators require pulserver[recon-cpu].") from error
    return getattr(module, class_name)


def deepinverse_multicoil_mri(*, mask: Any, coil_maps: Any, three_d: bool = False, **kwargs: Any) -> Any:
    """Construct DeepInverse's Torch-native Cartesian parallel-MRI physics.

    ``mask`` and ``coil_maps`` must be ``torch.Tensor`` objects.  The physics
    module remains on their device (CPU or CUDA) and therefore preserves
    DeepInverse's automatic Torch dispatch.  Additional keyword arguments are
    forwarded unchanged to :class:`deepinv.physics.MultiCoilMRI`.
    """
    try:
        physics = import_module("deepinv.physics")
    except ImportError as error:
        raise ImportError("DeepInverse physics requires pulserver[recon-cpu].") from error
    device = getattr(coil_maps, "device", getattr(mask, "device", "cpu"))
    return physics.MultiCoilMRI(
        mask=mask,
        coil_maps=coil_maps,
        three_d=three_d,
        device=device,
        **kwargs,
    )


def mrinufft_operator(backend: str = "finufft", **kwargs: Any) -> Any:
    """Create and return an mri-nufft operator with the requested backend.

    Parameters are passed unchanged to ``mrinufft.get_operator(backend)``.
    Use ``backend="finufft"`` with ``pulserver[recon-cpu]`` or
    ``backend="cufinufft"`` with ``pulserver[recon-cuda]``.
    """
    try:
        get_operator = import_module("mrinufft").get_operator
    except ImportError as error:
        raise ImportError("mri-nufft requires pulserver[recon-cpu] or pulserver[recon-cuda].") from error
    return get_operator(backend)(**kwargs)


def available_nufft_backends() -> list[str]:
    """Return available mri-nufft backends, or an empty list if it is absent."""
    try:
        list_backends = import_module("mrinufft.operators").list_backends
    except ImportError:
        return []
    return list_backends(available_only=True)
