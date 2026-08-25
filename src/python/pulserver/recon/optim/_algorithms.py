"""High-level parallel-imaging compressed-sensing reconstruction."""

from __future__ import annotations

__all__ = ["PolynomialPreconditioner", "pics"]

from collections.abc import Callable, Sequence
from importlib import import_module
from math import isfinite
from typing import Any


class PolynomialPreconditioner:
    """L2-optimal polynomial approximation to a normal-operator inverse.

    Given a self-adjoint positive-semidefinite operator ``T`` and a spectral
    interval, this evaluates ``p(scale * T)`` with Horner's method. A degree
    ``d`` application costs exactly ``d`` calls to ``normal``. The
    coefficients minimize

    ``integral(lower, upper, (1 - x p(x)) ** 2 dx)``.

    This is the polynomial construction used by the MRF reference
    reconstruction and PyGROG.

    Parameters
    ----------
    normal
        Normal operator to precondition.
    degree
        Degree of the inverse-approximating polynomial.
    spectrum
        Lower and upper bounds of the scaled normal-operator spectrum.
    scale
        Positive scaling applied to each normal-operator evaluation.

    Examples
    --------
    A polynomial in the normal operator, applied where the operator's spectrum is
    known well enough to flatten it -- fewer iterations for the same answer.

    >>> import torch
    >>> import pulserver.recon as recon
    >>> preconditioner = recon.PolynomialPreconditioner(
    ...     lambda x: x, degree=2, spectrum=(0.0, 1.0)
    ... )
    >>> tuple(preconditioner(torch.ones(4)).shape)
    (4,)
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


def _plain_fista(
    data: Any,
    physics: Any,
    denoiser: Any,
    *,
    regularization: float,
    iterations: int,
    stepsize: float,
    init: Any | None,
    gradient_transform: Any | None = None,
    host: bool = False,
) -> Any:
    """One FISTA loop for the fast paths: optionally preconditioned, optionally
    with CPU-resident iterates so streamed operator and denoiser calls stage
    their own device transfers."""
    rhs = physics.A_adjoint(data)
    x, z = _initial_fista_iterates(init, data, physics, rhs)
    if host:
        if hasattr(x, "device") and x.device.type != "cpu":
            x = x.to("cpu")
        if hasattr(z, "device") and z.device.type != "cpu":
            z = z.to("cpu")
    for iteration in range(iterations):
        gradient = physics.A_adjoint_A(z) - rhs
        step = stepsize * gradient
        if gradient_transform is not None:
            step = gradient_transform(step)
        next_x = denoiser(z - step, regularization)
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


def _pics(
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
        result = _plain_fista(
            data,
            linear,
            denoiser,
            regularization=regularization,
            iterations=iterations,
            stepsize=stepsize,
            init=init,
            gradient_transform=PolynomialPreconditioner(
                linear.A_adjoint_A,
                degree=polynomial_degree,
                spectrum=(0.0, 1.0),
                scale=stepsize,
            ),
        )
        if streaming is not None and streaming.result_device == "cuda":
            return result.to(streaming.torch_device, non_blocking=True)
        return result

    if streaming is not None:
        result = _plain_fista(
            data,
            linear,
            denoiser,
            regularization=regularization,
            iterations=iterations,
            stepsize=stepsize,
            init=init,
            host=True,
        )
        if streaming.result_device == "cuda":
            return result.to(streaming.torch_device, non_blocking=True)
        return result

    try:
        optim = import_module("deepinv.optim")
    except ImportError as error:
        raise ImportError(
            "PICS reconstruction requires DeepInverse, which ships with "
            "pulserver; reinstall the package to restore it."
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


def pics(
    data: Any,
    physics: Any,
    denoiser: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Reconstruct parallel MRI data, accepting NumPy or Torch transparently.

    A thin native-complex boundary over the solver: measurements given as a
    NumPy array are moved to Torch in the complex working dtype -- zero-copy
    when they already are complex64 on the host -- and the reconstruction is
    handed back as NumPy, so a pipeline never converts by hand. A DeepInverse
    denoiser that works in two real channels is adapted internally so it can
    regularize the complex image directly; Pulserver's own complex-aware
    denoisers pass through untouched. See :func:`_pics` for the algorithm and
    its parameters.

    Examples
    --------
    >>> import torch
    >>> import pulserver.recon as recon
    >>> physics = recon.Cartesian2D(
    ...     torch.ones(1, 1, 16, 16),
    ...     torch.ones(1, 2, 16, 16, dtype=torch.complex64) / 2 ** 0.5,
    ... )
    >>> measurement = physics.A(torch.zeros(1, 16, 16, dtype=torch.complex64))
    >>> recon.pics(measurement, physics, iterations=4).shape
    torch.Size([1, 16, 16])

    Three-fold undersampled with a calibration block: the adjoint aliases,
    and the solve unfolds it against the sensitivities.

    .. plot::

       import torch
       import pulserver.recon as recon
       from _figures import images, phantom

       truth, coil_maps = phantom(64, coils=4)
       mask = torch.zeros(64, 64)
       mask[:, ::3] = 1.0
       mask[:, 24:40] = 1.0
       physics = recon.Cartesian2D(mask[None, None], coil_maps)
       measured = physics.A(truth)
       images(
           [
               ("object", truth),
               ("zero-filled, R = 3", physics.A_adjoint(measured)),
               ("CG-SENSE", recon.pics(measured, physics, iterations=20)),
           ],
           title="pics, Cartesian, three-fold undersampled",
       )
    """
    denoiser = _complex_denoiser(denoiser)
    # Direct imports rather than the module's ``import_module`` so the boundary
    # keeps working when a test stubs the latter for solver selection.
    import numpy

    if not isinstance(data, numpy.ndarray):
        return _pics(data, physics, denoiser, **kwargs)
    import torch

    from ..physics import _operator_device

    device = _operator_device(physics)
    tensor = torch.as_tensor(data)
    if tensor.is_complex():
        tensor = tensor.to(torch.complex64)
    tensor = tensor.to(device)
    init = kwargs.get("init")
    if isinstance(init, numpy.ndarray):
        kwargs["init"] = torch.as_tensor(init).to(tensor.dtype).to(device)
    result = _pics(tensor, physics, denoiser, **kwargs)
    return result.detach().to("cpu").numpy() if hasattr(result, "detach") else result


def _complex_denoiser(denoiser: Any | None) -> Any | None:
    """Wrap a real-valued DeepInverse denoiser so it acts on complex images.

    A plain DeepInverse model expects two real channels, so it is wrapped in
    the one adapter that packs complex to real and back -- which reproduces
    the two-channel view these denoisers always saw. A denoiser marked
    ``handles_complex`` is left alone, and a sequence is passed through
    untouched so :func:`_pics` can reject it.
    """
    if denoiser is None or getattr(denoiser, "handles_complex", False):
        return denoiser
    if isinstance(denoiser, Sequence):
        return denoiser
    from ..learned import _ComplexAdapter

    return _ComplexAdapter(denoiser)


# %% private module subroutines


def _optim_class(name: str) -> type[Any]:
    try:
        module = import_module(__package__)
    except ImportError as error:
        raise ImportError(
            "PICS reconstruction requires DeepInverse and Torch, which "
            "ship with pulserver; reinstall the package to restore them."
        ) from error
    return getattr(module, name)
