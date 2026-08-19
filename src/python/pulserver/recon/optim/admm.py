"""Transformed-consensus ADMM for MRI reconstruction."""

from __future__ import annotations

__all__ = ["ADMM"]

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import torch


from ._base import (
    _AlgorithmSchedule,
    _IterativeOptimizer,
    _data_fidelity_schedule,
    _prior_schedule,
    _validate_algorithm_values,
    _validate_prior_width,
    _validate_schedule_length,
)
from .cg import ConjugateGradient
from .prior import StackedPrior
from .state import OptimState


class ADMM(_IterativeOptimizer):
    r"""Consensus ADMM for one or more transformed MRI priors.

    The optimizer minimizes

    ``0.5 * ||A(x) - y||^2 / sigma^2 + lambda_reg * sum_i w_i g_i(T_i x)``.

    The image update uses :class:`ConjugateGradient` on the Toeplitz-accelerated
    normal operator when supplied by the physics. ``stepsize`` is the inverse
    augmented-Lagrangian penalty, matching DeepInverse's proximal-step naming.

    Parameters
    ----------
    data_fidelity
        A DeepInverse L2 fidelity, or one per iteration.
    prior
        A prior, a :class:`StackedPrior`, or an iteration-wise list of either.
    lambda_reg
        Global regularization weight.
    stepsize
        Inverse ADMM penalty and prior proximal step size.
    beta
        Relaxation parameter.
    g_param
        Prior parameter when ``prior`` is a single DeepInverse prior.
    preconditioner
        Fixed CG preconditioner, for example DCF-derived diagonal weights.
    unfold
        Enable implicit gradients and selected trainable algorithm parameters.
    """

    def __init__(
        self,
        data_fidelity: Any | list[Any] | None = None,
        prior: Any | list[Any] | None = None,
        lambda_reg: float | Sequence[float] | torch.Tensor = 1.0,
        stepsize: float | Sequence[float] | torch.Tensor = 1.0,
        beta: float | Sequence[float] | torch.Tensor = 1.0,
        g_param: Any | None = None,
        sigma_denoiser: Any | None = None,
        max_iter: int = 100,
        crit_conv: str = "residual",
        thres_conv: float = 1e-5,
        early_stop: bool = False,
        custom_init: Callable[[torch.Tensor, Any], Any] | None = None,
        unfold: bool = False,
        trainable_params: list[str] | None = None,
        params_algo: dict[str, Any] | None = None,
        cg_max_iter: int = 20,
        cg_rtol: float = 1e-5,
        cg_atol: float = 0.0,
        cg_backward_max_iter: int | None = None,
        cg_backward_rtol: float | None = None,
        cg_backward_atol: float | None = None,
        cg_batch_dim: int | None = 0,
        preconditioner: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        if crit_conv != "residual":
            raise ValueError("ADMM currently supports crit_conv='residual' only")
        super().__init__(
            max_iter=max_iter,
            early_stop=early_stop,
            thres_conv=thres_conv,
        )
        if g_param is None:
            g_param = sigma_denoiser
        self.data_fidelity = torch.nn.ModuleList(_data_fidelity_schedule(data_fidelity))
        self.prior = torch.nn.ModuleList(_prior_schedule(prior, g_param))
        _validate_schedule_length(self.data_fidelity, max_iter, "data_fidelity")
        _validate_schedule_length(self.prior, max_iter, "prior")
        _validate_prior_width(self.prior)
        values = {
            "lambda": lambda_reg,
            "stepsize": stepsize,
            "beta": beta,
        }
        if params_algo is not None:
            unknown = set(params_algo).difference(values)
            if unknown:
                raise ValueError(f"unknown ADMM parameters: {sorted(unknown)}")
            values.update(params_algo)
        _validate_algorithm_values(values)
        self.params_algo = _AlgorithmSchedule(
            values,
            max_iter=max_iter,
            unfold=unfold,
            trainable_params=trainable_params,
        )
        self.cg = ConjugateGradient(
            max_iter=cg_max_iter,
            rtol=cg_rtol,
            atol=cg_atol,
            backward_max_iter=cg_backward_max_iter,
            backward_rtol=cg_backward_rtol,
            backward_atol=cg_backward_atol,
            batch_dim=cg_batch_dim,
            preconditioner=preconditioner,
        )
        self.custom_init = custom_init
        self.unfold = bool(unfold)

    def init_state(
        self,
        y: torch.Tensor,
        physics: Any,
        init: Any | None = None,
    ) -> OptimState:
        """Initialize image, split, and scaled-dual variables."""
        selected = self.custom_init if init is None else init
        if callable(selected):
            selected = selected(y, physics)
        if isinstance(selected, OptimState):
            return selected
        if isinstance(selected, dict):
            return OptimState(variables=selected)
        if isinstance(selected, tuple):
            if len(selected) != 3:
                raise ValueError("ADMM tuple initialization must contain x, z, and u")
            return OptimState(variables={"est": selected, "cost": None})

        x = physics.A_adjoint(y) if selected is None else selected
        if not isinstance(x, torch.Tensor):
            raise TypeError("ADMM initialization must resolve to a torch.Tensor")
        stack = self._prior_at(0)
        split = tuple(stack.transform(index, x) for index in range(len(stack)))
        dual = tuple(torch.zeros_like(item) for item in split)
        return OptimState(variables={"est": (x, split, dual), "cost": None})

    def step(
        self,
        state: OptimState,
        y: torch.Tensor,
        physics: Any,
        iteration: int | None = None,
        *,
        operator_parameters: Iterable[torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> OptimState:
        """Run one transformed-consensus ADMM iteration."""
        del kwargs
        index = state.iteration if iteration is None else iteration
        if index >= self.max_iter:
            raise IndexError("iteration exceeds max_iter")
        x_previous, split_previous, dual_previous = state.variables["est"]
        stack = self._prior_at(index)
        fidelity = self._data_fidelity_at(index)
        if not hasattr(fidelity, "norm"):
            raise TypeError("ADMM image updates require a DeepInverse L2 fidelity")
        params = self.params_algo.current(index)
        gamma = params["stepsize"]
        lambda_reg = params["lambda"]
        beta = params["beta"]
        fidelity_norm = fidelity.norm
        adjoint_data = physics.A_adjoint(y)

        rhs = gamma * fidelity_norm * adjoint_data
        for block_index, (block, dual) in enumerate(
            zip(split_previous, dual_previous, strict=True)
        ):
            rhs = rhs + stack.adjoint(block_index, block - dual)

        def normal(value: torch.Tensor) -> torch.Tensor:
            return gamma * fidelity_norm * physics.A_adjoint_A(value) + stack.normal(
                value
            )

        explicit_parameters = list(self.physics_parameters(physics))
        explicit_parameters.extend(stack.transform_parameters())
        explicit_parameters.extend(self.algorithm_parameters())
        if operator_parameters is not None:
            explicit_parameters.extend(operator_parameters)
        x = self.cg(
            normal,
            rhs,
            init=x_previous,
            parameters=explicit_parameters,
        )
        transformed = tuple(
            stack.transform(block_index, x) for block_index in range(len(stack))
        )
        relaxed = tuple(
            beta * current + (1.0 - beta) * previous
            for current, previous in zip(transformed, split_previous, strict=True)
        )
        split = tuple(
            stack.prox(
                block_index,
                current + dual_previous[block_index],
                gamma=gamma,
                weight_scale=lambda_reg,
            )
            for block_index, current in enumerate(relaxed)
        )
        dual = tuple(
            old + current - updated
            for old, current, updated in zip(
                dual_previous,
                relaxed,
                split,
                strict=True,
            )
        )
        variables = {"est": (x, split, dual), "cost": None}
        return OptimState(variables=variables, iteration=index + 1)

    def _prior_at(self, iteration: int) -> StackedPrior:
        return self.prior[iteration] if len(self.prior) > 1 else self.prior[0]

    def _data_fidelity_at(self, iteration: int) -> torch.nn.Module:
        return (
            self.data_fidelity[iteration]
            if len(self.data_fidelity) > 1
            else self.data_fidelity[0]
        )
