"""Optimisation selectors and reconstruction-specific accelerators."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from math import isfinite
from typing import Any

__all__ = [
    "PolynomialPreconditioner",
    "deepinverse_optimizer",
    "mrpro_optimizer",
]

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


def _l2_optimal_coefficients(
    degree: int,
    lower: float,
    upper: float,
) -> tuple[float, ...]:
    """Return coefficients minimizing the inverse residual over an interval."""
    numpy = import_module("numpy")
    indices = numpy.arange(degree + 1, dtype=numpy.float64)
    powers = indices[:, None] + indices[None, :]
    matrix_order = powers + 3.0
    matrix = (upper**matrix_order - lower**matrix_order) / matrix_order
    vector_order = indices + 2.0
    vector = (upper**vector_order - lower**vector_order) / vector_order
    coefficients = numpy.linalg.pinv(matrix) @ vector
    return tuple(float(item) for item in coefficients)


class PolynomialPreconditioner:
    """L2-optimal polynomial approximation to a normal-operator inverse.

    Given a self-adjoint positive-semidefinite operator ``T`` and a spectral
    interval, this evaluates ``p(scale * T)`` with Horner's method. A degree
    ``d`` application costs exactly ``d`` calls to ``normal``. The
    coefficients minimize

    ``integral(lower, upper, (1 - x p(x)) ** 2 dx)``.

    This is the polynomial construction used by the MRF reference
    reconstruction and PyGROG.
    """

    def __init__(
        self,
        normal: Callable[[Any], Any],
        *,
        degree: int,
        spectrum: tuple[float, float] = (0.0, 1.0),
        scale: float = 1.0,
    ) -> None:
        if not isinstance(degree, int) or isinstance(degree, bool) or degree < 0:
            raise ValueError("degree must be a non-negative integer")
        lower, upper = (float(item) for item in spectrum)
        if not isfinite(lower) or not isfinite(upper) or lower < 0 or upper <= lower:
            raise ValueError(
                "spectrum must be a finite increasing non-negative interval"
            )
        if not isfinite(scale) or scale <= 0:
            raise ValueError("scale must be positive and finite")

        self.normal = normal
        self.degree = degree
        self.spectrum = (lower, upper)
        self.scale = float(scale)
        self.coefficients = _l2_optimal_coefficients(
            degree,
            lower,
            upper,
        )

    def apply(self, value: Any) -> Any:
        """Apply the fixed polynomial using ``degree`` normal operations."""
        result = self.coefficients[-1] * value
        for coefficient in reversed(self.coefficients[:-1]):
            result = self.scale * self.normal(result) + coefficient * value
        return result

    def __call__(self, value: Any) -> Any:
        return self.apply(value)

    def __repr__(self) -> str:
        return (
            f"PolynomialPreconditioner(degree={self.degree}, "
            f"spectrum={self.spectrum}, scale={self.scale})"
        )


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
