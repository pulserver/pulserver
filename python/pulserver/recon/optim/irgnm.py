"""Iteratively regularized Gauss--Newton composition for nonlinear MRI."""

from __future__ import annotations

__all__ = ["IRGNM"]

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import torch

import deepinv

from ._base import _AlgorithmSchedule, _IterativeOptimizer, _unique_parameters
from .cg import ConjugateGradient
from .state import OptimState


class IRGNM(_IterativeOptimizer):
    r"""Linearize nonlinear physics around any Pulserver linear solver.

    At outer iteration ``k``, this constructs the tangent data

    ``y - physics.A(x_k) + J_k.A(x_k)``

    and calls ``inner(tangent_data, J_k, init=x_k)``. Consequently the same
    :class:`FISTA`, :class:`PDHG`, or :class:`ADMM` object can become an
    IRGNM-FISTA, IRGNM-PDHG, or IRGNM-ADMM reconstruction without copying its
    implementation.

    Passing :class:`ConjugateGradient` selects IRGNM-CG. That path solves the
    damped normal equation for the *increment*,

    ``(J_k^H J_k + a I) h = J_k^H (y - F(x_k)) + a (x_ref - x_k)``,

    and takes ``x_{k+1} = x_k + h``, regularizing towards the initial estimate
    (or ``reference``). Solving for the increment is what makes the inner
    solver's relative tolerance mean what it says: its right-hand side is the
    Gauss--Newton gradient, so it shrinks as the outer iteration converges.

    Nonlinear physics may implement ``linearize(x)`` and return either a
    Jacobian physics object or ``(prediction, jacobian)`` for a fast analytic
    path. Ordinary :class:`deepinv.physics.Physics` objects work directly:
    Pulserver constructs their Jacobian with ``A_jvp``/``A_vjp`` when supplied
    and otherwise uses :mod:`torch.func` without materializing a Jacobian.

    Parameters
    ----------
    inner
        Linear Pulserver optimizer or :class:`ConjugateGradient`.
    damping
        Non-negative scalar or outer-iteration schedule used by IRGNM-CG.
    max_iter
        Number of Gauss--Newton linearizations.
    reference
        Tikhonov reference for IRGNM-CG. The initial estimate is used by
        default.
    linearize
        Optional ``linearize(physics, x)`` override.
    inner_kwargs
        Fixed keyword arguments forwarded to non-CG inner solvers.
    iteration_callback
        Optional ``callback(estimate, physics, completed_iterations)`` hook.
        It may return a gauge-equivalent or otherwise projected estimate.
    unfold
        Enable differentiation through outer iterations.
    """

    def __init__(
        self,
        inner: torch.nn.Module,
        *,
        damping: float | Sequence[float] | torch.Tensor = 1e-3,
        max_iter: int = 6,
        crit_conv: str = "residual",
        thres_conv: float = 1e-5,
        early_stop: bool = False,
        custom_init: Callable[[torch.Tensor, Any], Any] | None = None,
        reference: torch.Tensor
        | Callable[[torch.Tensor, Any], torch.Tensor]
        | None = None,
        linearize: Callable[[Any, torch.Tensor], Any] | None = None,
        inner_kwargs: dict[str, Any] | None = None,
        iteration_callback: Callable[[torch.Tensor, Any, int], torch.Tensor]
        | None = None,
        unfold: bool = False,
        trainable_params: list[str] | None = None,
    ) -> None:
        if not isinstance(inner, torch.nn.Module):
            raise TypeError("inner must be a torch.nn.Module")
        if crit_conv != "residual":
            raise ValueError("IRGNM currently supports crit_conv='residual' only")
        if bool(torch.any(torch.as_tensor(damping) < 0.0)):
            raise ValueError("damping must be non-negative")
        if (
            unfold
            and not isinstance(inner, ConjugateGradient)
            and hasattr(inner, "unfold")
            and not inner.unfold
        ):
            raise ValueError("an unfolded IRGNM requires an unfolded inner optimizer")
        super().__init__(
            max_iter=max_iter,
            early_stop=early_stop,
            thres_conv=thres_conv,
        )
        self.inner = inner
        self.params_algo = _AlgorithmSchedule(
            {"damping": damping},
            max_iter=max_iter,
            unfold=unfold,
            trainable_params=trainable_params,
        )
        self.custom_init = custom_init
        self.__dict__["reference"] = reference
        self.__dict__["linearize"] = linearize
        self.inner_kwargs = dict(inner_kwargs or {})
        self.__dict__["iteration_callback"] = iteration_callback
        self.unfold = bool(unfold)

    def init_state(
        self,
        y: torch.Tensor,
        physics: Any,
        init: Any | None = None,
    ) -> OptimState:
        """Initialize the nonlinear estimate and Tikhonov reference."""
        selected = self.custom_init if init is None else init
        if callable(selected):
            selected = selected(y, physics)
        if isinstance(selected, OptimState):
            return selected
        if isinstance(selected, dict):
            variables = dict(selected)
            estimate = variables["est"][0]
        else:
            if isinstance(selected, tuple):
                if not selected:
                    raise ValueError("IRGNM tuple initialization cannot be empty")
                selected = selected[0]
            estimate = physics.A_adjoint(y) if selected is None else selected
            variables = {"est": (estimate,), "cost": None}
        if not isinstance(estimate, torch.Tensor):
            raise TypeError("IRGNM initialization must resolve to a torch.Tensor")

        reference = self.reference
        if callable(reference):
            reference = reference(y, physics)
        if reference is None:
            reference = estimate.clone()
        elif not isinstance(reference, torch.Tensor):
            raise TypeError("IRGNM reference must resolve to a torch.Tensor")
        else:
            reference = reference.to(device=estimate.device, dtype=estimate.dtype)
        if reference.shape != estimate.shape:
            raise ValueError("IRGNM reference and estimate must have the same shape")
        variables["reference"] = reference
        return OptimState(variables=variables)

    def step(
        self,
        state: OptimState,
        y: torch.Tensor,
        physics: Any,
        iteration: int | None = None,
        *,
        operator_parameters: Iterable[torch.Tensor] | None = None,
        inner_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> OptimState:
        """Run one nonlinear linearize-and-solve iteration."""
        index = state.iteration if iteration is None else iteration
        if index >= self.max_iter:
            raise IndexError("iteration exceeds max_iter")
        estimate = state.estimate
        prediction, jacobian = _linearization(
            physics,
            estimate,
            self.linearize,
        )
        if isinstance(self.inner, ConjugateGradient):
            damping = self.params_algo.at("damping", index)
            reference = state.variables["reference"]
            # Solve for the increment. The right-hand side is then the
            # Gauss-Newton gradient, which is what the inner solver's relative
            # tolerance has to be measured against: an absolute solve warm
            # started at the estimate puts ``J^H J x`` into that norm and makes
            # the tolerance progressively meaningless as the estimate grows.
            rhs = jacobian.A_adjoint(y - prediction) + damping * (
                reference - estimate
            )

            def normal(value: torch.Tensor) -> torch.Tensor:
                method = getattr(jacobian, "A_adjoint_A", None)
                applied = (
                    method(value)
                    if method is not None
                    else jacobian.A_adjoint(jacobian.A(value))
                )
                return applied + damping * value

            parameters = list(self.physics_parameters(physics))
            parameters.extend(self.physics_parameters(jacobian))
            parameters.extend(self.algorithm_parameters())
            if estimate.requires_grad:
                parameters.append(estimate)
            if operator_parameters is not None:
                parameters.extend(operator_parameters)
            updated = estimate + self.inner(
                normal,
                rhs,
                init=torch.zeros_like(estimate),
                parameters=parameters,
            )
        else:
            tangent_data = y - prediction + jacobian.A(estimate)
            call_kwargs = {**self.inner_kwargs, **(inner_kwargs or {}), **kwargs}
            updated = self.inner(
                tangent_data,
                jacobian,
                init=estimate,
                **call_kwargs,
            )
        if self.iteration_callback is not None:
            projected = self.iteration_callback(updated, physics, index + 1)
            if not isinstance(projected, torch.Tensor):
                raise TypeError("iteration_callback must return a torch.Tensor")
            if projected.shape != updated.shape:
                raise ValueError("iteration_callback must preserve the estimate shape")
            updated = projected
        variables = {
            "est": (updated,),
            "reference": state.variables["reference"],
            "cost": None,
        }
        return OptimState(variables=variables, iteration=index + 1)

    def prior_parameters(
        self,
        stages: Sequence[int] | None = None,
    ) -> list[torch.nn.Parameter]:
        """Return prior parameters owned by the inner optimizer."""
        method = getattr(self.inner, "prior_parameters", None)
        return [] if method is None else method(stages)

    def transform_parameters(
        self,
        stages: Sequence[int] | None = None,
    ) -> list[torch.nn.Parameter]:
        """Return transform parameters owned by the inner optimizer."""
        method = getattr(self.inner, "transform_parameters", None)
        return [] if method is None else method(stages)

    def algorithm_parameters(self) -> list[torch.nn.Parameter]:
        """Return outer damping and inner algorithm parameters."""
        parameters = list(self.params_algo.parameters())
        method = getattr(self.inner, "algorithm_parameters", None)
        if method is not None:
            parameters.extend(method())
        return _unique_parameters(parameters)


# %% private module subroutines


def _linearization(
    physics: Any,
    estimate: torch.Tensor,
    override: Callable[[Any, torch.Tensor], Any] | None,
) -> tuple[torch.Tensor, Any]:
    if override is not None:
        result = override(physics, estimate)
    else:
        method = getattr(physics, "linearize", None)
        result = (
            method(estimate)
            if method is not None
            else (physics.A(estimate), _AutomaticJacobian(physics, estimate))
        )
    if isinstance(result, tuple):
        if len(result) != 2:
            raise ValueError(
                "linearize must return a Jacobian or (prediction, Jacobian)"
            )
        prediction, jacobian = result
    else:
        prediction, jacobian = physics.A(estimate), result
    for name in ("A", "A_adjoint"):
        if not hasattr(jacobian, name):
            raise TypeError(f"linearized physics must expose {name}")
    return prediction, jacobian


class _AutomaticJacobian(deepinv.physics.LinearPhysics):
    """Matrix-free Jacobian of an ordinary DeepInverse physics object."""

    def __init__(self, physics: Any, estimate: torch.Tensor) -> None:
        super().__init__()
        self.__dict__["physics"] = physics
        self.__dict__["estimate"] = estimate

    def A(self, vector: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Apply the Jacobian-vector product at the fixed estimate."""
        del kwargs
        method = getattr(self.physics, "A_jvp", None)
        if method is not None:
            return method(self.estimate, vector)
        return torch.func.jvp(
            self.physics.A,
            (self.estimate,),
            (vector,),
        )[1]

    def A_adjoint(self, vector: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Apply the vector-Jacobian product at the fixed estimate."""
        del kwargs
        method = getattr(self.physics, "A_vjp", None)
        if method is not None:
            return method(self.estimate, vector)
        _, pullback = torch.func.vjp(self.physics.A, self.estimate)
        return pullback(vector)[0]

    def A_adjoint_A(self, vector: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Apply the Jacobian normal operator without storing intermediates."""
        del kwargs
        return self.A_adjoint(self.A(vector))
