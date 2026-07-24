"""Access MRPro and DeepInverse optimisation implementations.

No iterative algorithm lives here.  The functions return the maintained
upstream implementation selected by its reconstruction role.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any

__all__ = ["deepinverse_optimizer", "mrpro_optimizer"]

_MRPRO_OPTIMIZERS = {
    "adam": "adam",
    "cg": "cg",
    "lbfgs": "lbfgs",
    "pdhg": "pdhg",
    "pgd": "pgd",
}

_DEEPINVERSE_OPTIMIZERS = {
    "admm": "ADMM",
    "fista": "FISTA",
    "hqs": "HQS",
    "pdhg": "PDCP",
    "pgd": "PGD",
}


def mrpro_optimizer(name: str) -> Callable[..., Any]:
    """Return an MRPro optimizer function by role.

    ``name`` is one of ``adam``, ``cg``, ``lbfgs``, ``pdhg``, or ``pgd``.
    The returned function uses the exact MRPro tensor/operator interface.
    """
    try:
        function_name = _MRPRO_OPTIMIZERS[name]
    except KeyError as error:
        choices = ", ".join(sorted(_MRPRO_OPTIMIZERS))
        raise ValueError(f"Unknown MRPro optimizer {name!r}; choose one of {choices}") from error
    try:
        module = import_module("mrpro.algorithms.optimizers")
    except ImportError as error:
        raise ImportError("MRPro optimizers require pulserver[recon-cpu].") from error
    return getattr(module, function_name)


def deepinverse_optimizer(name: str, **kwargs: Any) -> Any:
    """Construct a DeepInverse optimiser by role.

    ``name`` is one of ``admm``, ``fista``, ``hqs``, ``pdhg``, or ``pgd``.
    Parameters are passed unchanged to the selected DeepInverse optimiser.
    """
    try:
        class_name = _DEEPINVERSE_OPTIMIZERS[name]
    except KeyError as error:
        choices = ", ".join(sorted(_DEEPINVERSE_OPTIMIZERS))
        raise ValueError(f"Unknown DeepInverse optimizer {name!r}; choose one of {choices}") from error
    try:
        module = import_module("deepinv.optim")
    except ImportError as error:
        raise ImportError("DeepInverse optimizers require pulserver[recon-cpu].") from error
    return getattr(module, class_name)(**kwargs)
