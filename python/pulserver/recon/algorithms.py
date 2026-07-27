"""High-level parallel-imaging compressed-sensing reconstruction."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from math import isfinite
from typing import Any

from .denoisers import average
from .optimizers import PolynomialPreconditioner
from .physics import MRIPhysics

__all__ = ["pics"]


def _linear_physics(physics: Any) -> Any:
    if isinstance(physics, MRIPhysics):
        return physics
    required = ("A", "A_adjoint", "A_adjoint_A")
    if not all(hasattr(physics, name) for name in required):
        raise TypeError(
            "physics must be an MRIPhysics or DeepInverse linear physics object"
        )
    return physics


def _initial_fista_iterates(
    init: Any | None,
    data: Any,
    physics: Any,
    default: Any,
) -> tuple[Any, Any]:
    if init is None:
        return default, default
    if callable(init):
        init = init(data, physics)
    if isinstance(init, dict):
        init = init["est"]
    if isinstance(init, tuple):
        if not init:
            raise ValueError("init tuple cannot be empty")
        return init[0], init[1] if len(init) > 1 else init[0]
    return init, init


def _polynomial_fista(
    data: Any,
    physics: Any,
    denoiser: Any,
    *,
    regularization: float,
    iterations: int,
    stepsize: float,
    degree: int,
    init: Any | None,
) -> Any:
    rhs = physics.A_adjoint(data)
    x, z = _initial_fista_iterates(init, data, physics, rhs)
    preconditioner = PolynomialPreconditioner(
        physics.A_adjoint_A,
        degree=degree,
        spectrum=(0.0, 1.0),
        scale=stepsize,
    )
    for iteration in range(iterations):
        gradient = physics.A_adjoint_A(z) - rhs
        next_x = denoiser(
            z - preconditioner(stepsize * gradient),
            regularization,
        )
        momentum = (iteration + 2.0) / (iteration + 3.0)
        z = next_x + momentum * (next_x - x)
        x = next_x
    return x


def pics(
    data: Any,
    physics: Any,
    denoiser: Any | None = None,
    *,
    regularization: float = 0.0,
    iterations: int = 30,
    tolerance: float = 1e-5,
    stepsize: float | None = None,
    polynomial_degree: int = 0,
    init: Any | None = None,
    verbose: bool = False,
) -> Any:
    """Reconstruct parallel MRI data with CG or plug-and-play FISTA.

    With no denoiser, this solves

    ``(AᴴA + regularization I)x = Aᴴy``

    by conjugate gradients. With one denoiser, DeepInverse FISTA is used and
    ``regularization`` becomes its threshold/noise-level parameter. Passing a
    sequence of denoisers applies their equal-weight average at every proximal
    step.

    A positive ``polynomial_degree`` selects polynomial-preconditioned FISTA.
    Its degree-``d`` gradient step applies ``AᴴA`` ``d + 1`` times, using the
    fast normal implementation exposed by the physics (for example Toeplitz),
    before the denoising step. Degrees 2--5 are generally the useful range.
    """
    if iterations < 1:
        raise ValueError("iterations must be at least one")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if regularization < 0:
        raise ValueError("regularization must be non-negative")
    if (
        not isinstance(polynomial_degree, int)
        or isinstance(polynomial_degree, bool)
        or polynomial_degree < 0
    ):
        raise ValueError("polynomial_degree must be a non-negative integer")
    linear = _linear_physics(physics)

    if denoiser is None:
        if polynomial_degree:
            raise ValueError(
                "polynomial_degree is only available for denoiser-based FISTA"
            )
        try:
            conjugate_gradient = import_module(
                "deepinv.optim.linear"
            ).conjugate_gradient
        except ImportError as error:
            raise ImportError(
                "PICS reconstruction requires DeepInverse; install "
                "pulserver[recon-cpu] or pulserver[recon-cuda]."
            ) from error
        rhs = linear.A_adjoint(data)

        def normal(x: Any) -> Any:
            result = linear.A_adjoint_A(x)
            return result + regularization * x if regularization else result

        return conjugate_gradient(
            normal,
            rhs,
            max_iter=iterations,
            tol=tolerance,
            init=init,
            parallel_dim=0,
            verbose=verbose,
        )

    if isinstance(denoiser, Sequence):
        denoiser = average(denoiser)
    if regularization <= 0:
        raise ValueError("regularization must be positive when a denoiser is provided")

    if stepsize is None:
        x0 = linear.A_adjoint(data)
        compute_sqnorm = getattr(linear, "compute_sqnorm", None)
        if compute_sqnorm is None:
            operator = getattr(linear, "operator", None)
            compute_sqnorm = getattr(operator, "compute_sqnorm", None)
        if compute_sqnorm is None:
            raise TypeError(
                "physics cannot estimate its norm; provide an explicit stepsize"
            )
        lipschitz = compute_sqnorm(
            x0,
            max_iter=min(20, iterations),
            tol=1e-3,
            verbose=False,
        )
        lipschitz = float(lipschitz)
        if not isfinite(lipschitz) or lipschitz <= 0:
            raise ValueError("physics returned a non-positive norm estimate")
        stepsize = 0.95 / lipschitz
    if not isfinite(stepsize) or stepsize <= 0:
        raise ValueError("stepsize must be positive and finite")

    if polynomial_degree:
        return _polynomial_fista(
            data,
            linear,
            denoiser,
            regularization=regularization,
            iterations=iterations,
            stepsize=stepsize,
            degree=polynomial_degree,
            init=init,
        )

    try:
        optim = import_module("deepinv.optim")
    except ImportError as error:
        raise ImportError(
            "PICS reconstruction requires DeepInverse; install "
            "pulserver[recon-cpu] or pulserver[recon-cuda]."
        ) from error

    model = optim.FISTA(
        data_fidelity=optim.L2(),
        prior=optim.PnP(denoiser),
        # PnP ignores the explicit lambda/gamma product and consumes g_param
        # as the denoiser strength. Keeping lambda at one avoids two competing
        # public regularization parameters.
        lambda_reg=1.0,
        g_param=regularization,
        stepsize=stepsize,
        max_iter=iterations,
        early_stop=False,
    )
    call_kwargs = {"init": init} if init is not None else {}
    return model(data, linear, **call_kwargs)
