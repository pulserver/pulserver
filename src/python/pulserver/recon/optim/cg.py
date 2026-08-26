"""Memory-efficient conjugate gradients with implicit differentiation."""

from __future__ import annotations

__all__ = ["CGInfo", "ConjugateGradient"]

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from math import isfinite
from typing import Any

import warnings

import torch
from torch.autograd.function import once_differentiable


@dataclass(frozen=True)
class CGInfo:
    """Diagnostics from a conjugate-gradient solve.

    Parameters
    ----------
    iterations
        Number of Krylov iterations performed.
    converged
        Whether every independent system met the stopping criterion.
    residual_norm
        Final residual norm for each independent system.

    Examples
    --------
    >>> import pulserver.recon as recon
    >>> info = recon.CGInfo(iterations=12, converged=True, residual_norm=1e-6)
    >>> info.iterations, info.converged
    (12, True)
    """

    iterations: int
    converged: bool
    residual_norm: torch.Tensor


class ConjugateGradient(torch.nn.Module):
    """Solve Hermitian positive-definite systems with constant-memory backward.

    The forward and backward passes each run conjugate gradients. The Krylov
    history is never retained. Gradients with respect to explicit operator
    parameters are computed with one implicit vector-Jacobian product, while
    the preconditioner is deliberately treated as fixed.

    Parameters
    ----------
    max_iter
        Maximum number of forward iterations.
    rtol, atol
        Stopping criterion ``||r|| <= atol + rtol * ||b||``. Setting both to
        zero runs a fixed number of iterations without per-iteration host
        synchronization.
    backward_max_iter, backward_rtol, backward_atol
        Optional independent accuracy settings for the adjoint solve.
    batch_dim
        Dimension indexing independent systems. Use ``None`` for one system.
    preconditioner
        Optional fixed positive-definite preconditioner.

    Examples
    --------
    The solver takes an operator and a right-hand side, so a SENSE reconstruction
    is ``A^H A x = A^H y`` -- which is what :func:`~pulserver.recon.pics` builds
    when no prior is given.

    .. plot::

       import pulserver.recon as recon
       from _figures import images, phantom, radial_spokes

       truth, coil_maps = phantom(64, coils=4)
       physics = recon.NonCartesian2D(
           radial_spokes(64, 16), (64, 64), coil_maps=coil_maps[0]
       )
       measured = physics.A(truth)
       rhs = physics.A_adjoint(measured)

       solver = recon.ConjugateGradient(max_iter=10)
       solved, info = solver(physics.A_adjoint_A, rhs, return_info=True)

       images(
           [
               ("truth", truth[0]),
               ("right-hand side", rhs[0, 0]),
               (f"after {info.iterations} iterations", solved[0, 0]),
           ],
           title="conjugate gradients on the normal equations",
       )
    """

    def __init__(
        self,
        *,
        max_iter: int = 30,
        rtol: float = 1e-5,
        atol: float = 0.0,
        backward_max_iter: int | None = None,
        backward_rtol: float | None = None,
        backward_atol: float | None = None,
        batch_dim: int | None = 0,
        preconditioner: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        _validate_iterations(max_iter, "max_iter")
        _validate_tolerance(rtol, "rtol")
        _validate_tolerance(atol, "atol")
        if backward_max_iter is not None:
            _validate_iterations(backward_max_iter, "backward_max_iter")
        if backward_rtol is not None:
            _validate_tolerance(backward_rtol, "backward_rtol")
        if backward_atol is not None:
            _validate_tolerance(backward_atol, "backward_atol")

        self.max_iter = max_iter
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.backward_max_iter = backward_max_iter or max_iter
        self.backward_rtol = (
            float(backward_rtol) if backward_rtol is not None else float(rtol)
        )
        self.backward_atol = (
            float(backward_atol) if backward_atol is not None else float(atol)
        )
        self.batch_dim = batch_dim
        self.preconditioner = preconditioner

    def forward(
        self,
        operator: Callable[[torch.Tensor], torch.Tensor],
        rhs: torch.Tensor,
        *,
        init: torch.Tensor | None = None,
        parameters: Iterable[torch.Tensor] = (),
        preconditioner: Callable[[torch.Tensor], torch.Tensor] | None = None,
        return_info: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, CGInfo]:
        """Solve ``operator(x) = rhs``.

        Parameters
        ----------
        operator
            Hermitian positive-definite linear operator.
        rhs
            Right-hand side. Systems along ``batch_dim`` are solved in
            parallel using vectorized tensor operations.
        init
            Optional initial estimate. Its derivative is intentionally zero,
            as implicit differentiation describes the converged solution.
        parameters
            Explicit trainable tensors used by ``operator``. Listing them is
            what allows the custom backward to return parameter gradients
            without mutating ``.grad`` as a side effect.
        preconditioner
            Per-call override of the fixed preconditioner.
        return_info
            Return :class:`CGInfo` in addition to the solution.
        """
        if not isinstance(rhs, torch.Tensor):
            raise TypeError("rhs must be a torch.Tensor")
        if init is None:
            init = torch.zeros_like(rhs)
        elif init.shape != rhs.shape:
            raise ValueError("init and rhs must have the same shape")
        if self.batch_dim is not None and not -rhs.ndim <= self.batch_dim < rhs.ndim:
            raise ValueError("batch_dim is outside rhs dimensions")

        selected_parameters = _unique_tensors(parameters)
        config = _CGConfig(
            operator=operator,
            preconditioner=(
                self.preconditioner if preconditioner is None else preconditioner
            ),
            max_iter=self.max_iter,
            rtol=self.rtol,
            atol=self.atol,
            backward_max_iter=self.backward_max_iter,
            backward_rtol=self.backward_rtol,
            backward_atol=self.backward_atol,
            batch_dim=self.batch_dim,
        )
        solution, iterations, residual_norm, converged = _ImplicitCG.apply(
            rhs,
            init,
            config,
            *selected_parameters,
        )
        if not return_info:
            return solution
        info = CGInfo(
            iterations=int(iterations.item()),
            converged=bool(converged.item()),
            residual_norm=residual_norm,
        )
        return solution, info


# %% private module subroutines


@dataclass(frozen=True)
class _CGConfig:
    operator: Callable[[torch.Tensor], torch.Tensor]
    preconditioner: Callable[[torch.Tensor], torch.Tensor] | None
    max_iter: int
    rtol: float
    atol: float
    backward_max_iter: int
    backward_rtol: float
    backward_atol: float
    batch_dim: int | None


class _ImplicitCG(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        rhs: torch.Tensor,
        init: torch.Tensor,
        config: _CGConfig,
        *parameters: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            solution, iterations, residual_norm, converged = _solve(
                config.operator,
                rhs,
                init,
                preconditioner=config.preconditioner,
                max_iter=config.max_iter,
                rtol=config.rtol,
                atol=config.atol,
                batch_dim=config.batch_dim,
            )
        ctx.config = config
        ctx.save_for_backward(solution, *parameters)
        iterations_tensor = torch.tensor(iterations, device=rhs.device)
        converged_tensor = torch.tensor(converged, device=rhs.device)
        ctx.mark_non_differentiable(
            iterations_tensor,
            residual_norm,
            converged_tensor,
        )
        return solution, iterations_tensor, residual_norm, converged_tensor

    @staticmethod
    @once_differentiable
    def backward(
        ctx: Any,
        grad_solution: torch.Tensor,
        _grad_iterations: torch.Tensor,
        _grad_residual: torch.Tensor,
        _grad_converged: torch.Tensor,
    ) -> tuple[Any, ...]:
        solution, *parameters = ctx.saved_tensors
        config = ctx.config
        with torch.no_grad():
            adjoint, _, _, _ = _solve(
                config.operator,
                grad_solution,
                torch.zeros_like(grad_solution),
                preconditioner=config.preconditioner,
                max_iter=config.backward_max_iter,
                rtol=config.backward_rtol,
                atol=config.backward_atol,
                batch_dim=config.batch_dim,
            )

        parameter_gradients: list[torch.Tensor | None] = [None] * len(parameters)
        required = [
            (index, parameter)
            for index, parameter in enumerate(parameters)
            if parameter.requires_grad
        ]
        if required:
            with torch.enable_grad():
                applied = config.operator(solution.detach())
                pseudo_loss = -_real_inner(adjoint.detach(), applied, batch_dim=None)
            if pseudo_loss.requires_grad:
                gradients = torch.autograd.grad(
                    pseudo_loss,
                    [parameter for _, parameter in required],
                    allow_unused=True,
                )
                for (index, _), gradient in zip(required, gradients, strict=True):
                    parameter_gradients[index] = gradient
        return adjoint, None, None, *parameter_gradients


def _solve(
    operator: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    init: torch.Tensor,
    *,
    preconditioner: Callable[[torch.Tensor], torch.Tensor] | None,
    max_iter: int,
    rtol: float,
    atol: float,
    batch_dim: int | None,
) -> tuple[torch.Tensor, int, torch.Tensor, bool]:
    solution = init.clone()
    residual = rhs - operator(solution)
    preconditioned = residual if preconditioner is None else preconditioner(residual)
    direction = preconditioned.clone()
    rho = _real_inner(residual, preconditioned, batch_dim)
    rhs_norm = _norm(rhs, batch_dim)
    threshold = atol + rtol * rhs_norm
    check_stopping = rtol > 0.0 or atol > 0.0
    iterations = 0
    valid = torch.ones_like(rho, dtype=torch.bool)

    for iteration in range(max_iter):
        residual_norm = _norm(residual, batch_dim)
        if check_stopping and bool(torch.all(residual_norm <= threshold)):
            break
        applied = operator(direction)
        denominator = _real_inner(direction, applied, batch_dim)
        active = residual_norm > threshold if check_stopping else residual_norm > 0.0
        # Negative curvature is not a reason to stop stepping. The operator a
        # compressed Toeplitz normal stands for carries eigenvalues just below
        # zero, and the iteration keeps reducing the residual through them --
        # refusing the step freezes it instead, well short of the answer it
        # would have reached. What the sign is recorded for is the diagnosis
        # and the choice of which iterate to answer with.
        recurrence_valid = (~active) | (
            torch.isfinite(rho)
            & torch.isfinite(denominator)
            & (rho > 0.0)
            & (denominator > 0.0)
        )
        valid = valid & recurrence_valid
        # Only an exactly zero curvature has no step to take.
        usable = active & (denominator != 0.0)
        safe_denominator = torch.where(
            usable,
            denominator,
            torch.ones_like(denominator),
        )
        step = torch.where(usable, rho / safe_denominator, 0.0)
        solution = solution + step * direction
        residual = residual - step * applied
        next_preconditioned = (
            residual if preconditioner is None else preconditioner(residual)
        )
        next_rho = _real_inner(residual, next_preconditioned, batch_dim)
        safe_rho = torch.where(rho > 0.0, rho, torch.ones_like(rho))
        momentum = torch.where(
            usable & torch.isfinite(next_rho) & (next_rho >= 0.0),
            next_rho / safe_rho,
            0.0,
        )
        direction = next_preconditioned + momentum * direction
        rho = next_rho
        iterations = iteration + 1

    residual_norm = _reported_norm(residual, batch_dim)
    converged = bool(torch.all(_norm(residual, batch_dim) <= threshold))
    if not bool(torch.all(valid)):
        # The operator carries a negative eigenvalue -- a compressed Toeplitz
        # normal can dip below zero by the largest transfer value its support
        # left out -- so the quadratic the iteration descends is unbounded in
        # that direction and the answer is whatever the last step left. The
        # iteration is not stopped for it: the residual keeps falling through
        # negative curvature, and refusing the step only freezes it short.
        warnings.warn(
            "CG met a non-positive recurrence; the operator is not positive "
            "definite and the answer is not a minimizer. Raise the "
            "regularization, stop earlier, or keep the whole Toeplitz "
            "transfer (toeplitz={'compress': False}) if it is compressed.",
            stacklevel=2,
        )
    return solution, iterations, residual_norm.detach(), converged


def _real_inner(
    left: torch.Tensor,
    right: torch.Tensor,
    batch_dim: int | None,
) -> torch.Tensor:
    product = (left.conj() * right).real
    if batch_dim is None:
        return product.sum()
    normalized = batch_dim % product.ndim
    dimensions = tuple(index for index in range(product.ndim) if index != normalized)
    return product.sum(dim=dimensions, keepdim=True) if dimensions else product


def _norm(value: torch.Tensor, batch_dim: int | None) -> torch.Tensor:
    return _real_inner(value, value, batch_dim).clamp_min(0.0).sqrt()


def _reported_norm(value: torch.Tensor, batch_dim: int | None) -> torch.Tensor:
    norm = _norm(value, batch_dim)
    if batch_dim is None:
        return norm
    return norm.reshape(value.shape[batch_dim])


def _unique_tensors(parameters: Iterable[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    selected = []
    identities = set()
    for parameter in parameters:
        if not isinstance(parameter, torch.Tensor):
            raise TypeError("operator parameters must be torch tensors")
        if id(parameter) not in identities:
            identities.add(id(parameter))
            selected.append(parameter)
    return tuple(selected)


def _validate_iterations(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _validate_tolerance(value: float, name: str) -> None:
    if not isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
