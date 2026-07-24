"""Access DeepInverse priors and proximal operators.

DeepInverse owns the regularisers, proximal maps, and plug-and-play priors.
This module only resolves its public classes by a stable reconstruction role.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["deepinverse_prior"]

_DEEPINVERSE_PRIORS = {
    "l1": "L1Prior",
    "l12": "L12Prior",
    "pnp": "PnP",
    "tikhonov": "Tikhonov",
    "tv": "TVPrior",
    "wavelet": "WaveletPrior",
}


def deepinverse_prior(name: str, **kwargs: Any) -> Any:
    """Construct a DeepInverse prior by role.

    ``name`` is one of ``l1``, ``l12``, ``pnp``, ``tikhonov``, ``tv``, or
    ``wavelet``.  Keyword arguments are passed unchanged to the selected
    DeepInverse class, whose ``prox`` implementation remains authoritative.
    """
    try:
        class_name = _DEEPINVERSE_PRIORS[name]
    except KeyError as error:
        choices = ", ".join(sorted(_DEEPINVERSE_PRIORS))
        raise ValueError(f"Unknown DeepInverse prior {name!r}; choose one of {choices}") from error
    try:
        module = import_module("deepinv.optim")
    except ImportError as error:
        raise ImportError("DeepInverse priors require pulserver[recon-cpu].") from error
    return getattr(module, class_name)(**kwargs)
