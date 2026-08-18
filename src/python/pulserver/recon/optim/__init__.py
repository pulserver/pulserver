"""DeepInverse-style optimization algorithms for MRI reconstruction.

Members are resolved on first use so that importing this namespace, and the
:func:`pics` entry point in particular, does not pull DeepInverse and Torch
into a process that only wants to read the API. Each solver raises a clear
ImportError if the numerical stack is genuinely absent when it runs.
"""

from __future__ import annotations

__all__ = [
    "ADMM",
    "CGInfo",
    "ConjugateGradient",
    "FISTA",
    "IRGNM",
    "OptimResult",
    "OptimState",
    "PDHG",
    "PolynomialPreconditioner",
    "StackedPrior",
    "pics",
]

import importlib
from typing import TYPE_CHECKING, Any

#: Public name to the private module defining it.
_MEMBERS = {
    "ADMM": "admm",
    "CGInfo": "cg",
    "ConjugateGradient": "cg",
    "FISTA": "fista",
    "IRGNM": "irgnm",
    "OptimResult": "state",
    "OptimState": "state",
    "PDHG": "pdhg",
    "PolynomialPreconditioner": "_algorithms",
    "StackedPrior": "prior",
    "pics": "_algorithms",
}


def __getattr__(name: str) -> Any:
    """Resolve one optimizer or entry point on first use."""
    module = _MEMBERS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f"{__name__}.{module}"), name)


def __dir__() -> list[str]:
    """Return the public optimization namespace."""
    return sorted(__all__)


if TYPE_CHECKING:
    from ._algorithms import PolynomialPreconditioner as PolynomialPreconditioner
    from ._algorithms import pics as pics
    from .admm import ADMM as ADMM
    from .cg import CGInfo as CGInfo
    from .cg import ConjugateGradient as ConjugateGradient
    from .fista import FISTA as FISTA
    from .irgnm import IRGNM as IRGNM
    from .pdhg import PDHG as PDHG
    from .prior import StackedPrior as StackedPrior
    from .state import OptimResult as OptimResult
    from .state import OptimState as OptimState
