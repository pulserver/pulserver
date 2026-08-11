"""High-level parallel-imaging compressed-sensing reconstruction."""

from __future__ import annotations

__all__ = ["PolynomialPreconditioner", "pics"]

from collections.abc import Sequence
from importlib import import_module
from math import isfinite
from typing import Any

from .optimizers import PolynomialPreconditioner


def _linear_physics(physics: Any) -> Any:
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


def _host_fista(
    data: Any,
    physics: Any,
    denoiser: Any,
    *,
    regularization: float,
    iterations: int,
    stepsize: float,
    init: Any | None,
) -> Any:
    """FISTA with CPU-resident iterates and streamed operator/denoiser calls."""
    rhs = physics.A_adjoint(data)
    x, z = _initial_fista_iterates(init, data, physics, rhs)
    if hasattr(x, "device") and x.device.type != "cpu":
        x = x.to("cpu")
    if hasattr(z, "device") and z.device.type != "cpu":
        z = z.to("cpu")
    for iteration in range(iterations):
        gradient = physics.A_adjoint_A(z) - rhs
        next_x = denoiser(z - stepsize * gradient, regularization)
        momentum = (iteration + 2.0) / (iteration + 3.0)
        z = next_x + momentum * (next_x - x)
        x = next_x
    return x


def _host_sqnorm(
    physics: Any,
    initial: Any,
    *,
    iterations: int,
    tolerance: float = 1e-3,
) -> float:
    """Estimate the normal-operator norm without bypassing streamed staging."""
    torch = import_module("torch")
    vector = initial.to("cpu")
    scale = torch.linalg.vector_norm(vector)
    if not bool(torch.isfinite(scale)) or float(scale) == 0.0:
        vector = torch.ones_like(vector)
        scale = torch.linalg.vector_norm(vector)
    vector = vector / scale
    previous = 0.0
    estimate = 0.0
    for _ in range(iterations):
        transformed = physics.A_adjoint_A(vector)
        norm = torch.linalg.vector_norm(transformed)
        estimate = float(norm)
        if not isfinite(estimate) or estimate <= 0.0:
            break
        vector = transformed / norm
        if abs(estimate - previous) <= tolerance * estimate:
            break
        previous = estimate
    return estimate


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
    streaming: Any | None = None,
    preconditioner: Any | None = None,
) -> Any:
    """Reconstruct parallel MRI data with CG or plug-and-play FISTA.

    With no denoiser, this solves

    ``(AᴴA + regularization I)x = Aᴴy``

    by implicitly differentiated conjugate gradients. With one denoiser,
    Pulserver's DeepInverse FISTA subclass is used and ``regularization``
    becomes its threshold/noise-level parameter. Use ``recon.optim.PDHG`` or
    ``recon.optim.ADMM`` with ``StackedPrior`` for simultaneous regularizers.

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
    if streaming is not None:
        torch = import_module("torch")
        if not isinstance(data, torch.Tensor):
            data = torch.as_tensor(data)
        elif data.device.type != "cpu":
            data = data.to("cpu")
        if isinstance(init, torch.Tensor) and init.device.type != "cpu":
            init = init.to("cpu")
        streaming.configure_physics(linear)

    if denoiser is None:
        if polynomial_degree:
            raise ValueError(
                "polynomial_degree is only available for denoiser-based FISTA"
            )
        rhs = linear.A_adjoint(data)
        if (
            streaming is not None
            and hasattr(init, "device")
            and init.device.type != "cpu"
        ):
            init = init.to("cpu")

        def normal(x: Any) -> Any:
            result = linear.A_adjoint_A(x)
            return result + regularization * x if regularization else result

        if init is None:
            init = import_module("torch").zeros_like(rhs)
        solver = _optim_class("ConjugateGradient")(
            max_iter=iterations,
            rtol=tolerance,
            batch_dim=0 if rhs.ndim > 1 else None,
            preconditioner=preconditioner,
        )
        result = solver(
            normal,
            rhs,
            init=init,
        )
        if streaming is not None and streaming.result_device == "cuda":
            return result.to(streaming.torch_device, non_blocking=True)
        return result

    if isinstance(denoiser, Sequence):
        raise TypeError(
            "pics accepts one denoiser; use optim.StackedPrior with PDHG or "
            "ADMM for simultaneous regularizers"
        )
    if preconditioner is not None:
        raise ValueError("preconditioner is only available for CG reconstruction")
    if streaming is not None:
        denoiser = streaming.wrap_denoiser(denoiser)
    if regularization <= 0:
        raise ValueError("regularization must be positive when a denoiser is provided")

    if stepsize is None:
        x0 = linear.A_adjoint(data)
        if streaming is not None:
            lipschitz = _host_sqnorm(
                linear,
                x0,
                iterations=min(20, iterations),
            )
        else:
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
        result = _polynomial_fista(
            data,
            linear,
            denoiser,
            regularization=regularization,
            iterations=iterations,
            stepsize=stepsize,
            degree=polynomial_degree,
            init=init,
        )
        if streaming is not None and streaming.result_device == "cuda":
            return result.to(streaming.torch_device, non_blocking=True)
        return result

    if streaming is not None:
        result = _host_fista(
            data,
            linear,
            denoiser,
            regularization=regularization,
            iterations=iterations,
            stepsize=stepsize,
            init=init,
        )
        if streaming.result_device == "cuda":
            return result.to(streaming.torch_device, non_blocking=True)
        return result

    try:
        optim = import_module("deepinv.optim")
    except ImportError as error:
        raise ImportError(
            "PICS reconstruction requires DeepInverse; install "
            "pulserver[recon-cpu] or pulserver[recon-cuda]."
        ) from error

    model = _optim_class("FISTA")(
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
        verbose=verbose,
    )
    call_kwargs = {"init": init} if init is not None else {}
    return model(data, linear, **call_kwargs)


# %% private module subroutines


def _optim_class(name: str) -> type[Any]:
    try:
        module = import_module(f"{__package__}.optim")
    except ImportError as error:
        raise ImportError(
            "PICS reconstruction requires DeepInverse and Torch; install "
            "pulserver[recon-cpu] or pulserver[recon-cuda]."
        ) from error
    return getattr(module, name)
