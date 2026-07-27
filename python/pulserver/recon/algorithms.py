"""High-level parallel-imaging compressed-sensing reconstruction."""

from __future__ import annotations

from importlib import import_module
from typing import Any

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


def pics(
    data: Any,
    physics: Any,
    denoiser: Any | None = None,
    *,
    regularization: float = 0.0,
    iterations: int = 30,
    tolerance: float = 1e-5,
    stepsize: float | None = None,
    init: Any | None = None,
    verbose: bool = False,
) -> Any:
    """Reconstruct parallel MRI data with CG or plug-and-play FISTA.

    With no denoiser, this solves

    ``(AᴴA + regularization I)x = Aᴴy``

    by conjugate gradients. With a denoiser, DeepInverse FISTA is used and
    ``regularization`` becomes the denoiser threshold/noise-level parameter.
    The latter is the natural strength parameter for the exposed wavelet, TV,
    and TGV proximal denoisers.
    """
    if iterations < 1:
        raise ValueError("iterations must be at least one")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if regularization < 0:
        raise ValueError("regularization must be non-negative")
    linear = _linear_physics(physics)

    if denoiser is None:
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

    if regularization <= 0:
        raise ValueError("regularization must be positive when a denoiser is provided")
    try:
        optim = import_module("deepinv.optim")
    except ImportError as error:
        raise ImportError(
            "PICS reconstruction requires DeepInverse; install "
            "pulserver[recon-cpu] or pulserver[recon-cuda]."
        ) from error

    if stepsize is None:
        x0 = linear.A_adjoint(data)
        compute_sqnorm = getattr(linear, "compute_sqnorm", None)
        if compute_sqnorm is None:
            compute_sqnorm = getattr(linear.operator, "compute_sqnorm", None)
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
        stepsize = 0.95 / float(lipschitz)
    if stepsize <= 0:
        raise ValueError("stepsize must be positive")

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
