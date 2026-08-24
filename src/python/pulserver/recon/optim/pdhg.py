"""Product-space primal-dual reconstruction built on DeepInverse."""

from __future__ import annotations

__all__ = ["PDHG"]

from collections.abc import Callable, Sequence
from typing import Any

import torch

import deepinv

from ._base import (
    _AlgorithmSchedule,
    _IterativeOptimizer,
    _data_fidelity_schedule,
    _prior_schedule,
    _validate_algorithm_values,
    _validate_prior_width,
    _validate_schedule_length,
)
from .prior import StackedPrior
from .state import OptimState


class PDHG(_IterativeOptimizer):
    r"""Primal-dual hybrid gradient for data and multiple transformed priors.

    The optimizer minimizes

    ``data_fidelity(A(x), y) + lambda_reg * sum_i w_i g_i(T_i x)``.

    Its single iteration is DeepInverse's Chambolle--Pock iterator applied to
    a product space. Passing a list directly as ``prior`` keeps DeepInverse's
    iteration-schedule meaning. Use :class:`StackedPrior` for simultaneous
    BART-style regularizers.

    Parameters
    ----------
    data_fidelity
        DeepInverse data-fidelity term, or one per iteration.
    prior
        A prior, a :class:`StackedPrior`, or an iteration-wise list of either.
    lambda_reg
        Global regularization weight.
    stepsize, stepsize_dual
        Primal and dual step sizes.
    beta
        Primal extrapolation parameter.
    g_param
        Prior parameter when ``prior`` is a single DeepInverse prior.
    max_iter
        Number of primal-dual iterations.
    unfold
        Register selected algorithm parameters as trainable tensors.

    Examples
    --------
    The primal-dual pair, for the same simultaneous-regularizer case ADMM covers,
    without the inner linear solve::

        import deepinv
        import pulserver.recon as recon

        solver = recon.PDHG(
            data_fidelity=deepinv.optim.L2(),
            prior=recon.StackedPrior([wavelet_prior, tv_prior]),
            lambda_reg=0.01,
            stepsize=0.4,
            stepsize_dual=0.4,
            max_iter=100,
        )
        image = solver(measured, physics)
    """

    def __init__(
        self,
        data_fidelity: Any | list[Any] | None = None,
        prior: Any | list[Any] | None = None,
        lambda_reg: float | Sequence[float] | torch.Tensor = 1.0,
        stepsize: float | Sequence[float] | torch.Tensor = 1.0,
        stepsize_dual: float | Sequence[float] | torch.Tensor = 1.0,
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
    ) -> None:
        if crit_conv != "residual":
            raise ValueError("PDHG currently supports crit_conv='residual' only")
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
            "stepsize_dual": stepsize_dual,
            "beta": beta,
        }
        if params_algo is not None:
            unknown = set(params_algo).difference(values)
            if unknown:
                raise ValueError(f"unknown PDHG parameters: {sorted(unknown)}")
            values.update(params_algo)
        _validate_algorithm_values(values, positive=("stepsize", "stepsize_dual"))
        self.params_algo = _AlgorithmSchedule(
            values,
            max_iter=max_iter,
            unfold=unfold,
            trainable_params=trainable_params,
        )
        self.iterator = deepinv.optim.optim_iterators.CPIteration(
            g_first=False,
            has_cost=False,
        )
        self.zero_prior = deepinv.optim.ZeroPrior()
        self.custom_init = custom_init
        self.unfold = bool(unfold)

    def init_state(
        self,
        y: torch.Tensor,
        physics: Any,
        init: Any | None = None,
    ) -> OptimState:
        """Initialize primal, extrapolated-primal, and product-dual variables."""
        selected = self.custom_init if init is None else init
        if callable(selected):
            selected = selected(y, physics)
        if isinstance(selected, OptimState):
            return selected
        if isinstance(selected, dict):
            return OptimState(variables=selected)
        if isinstance(selected, tuple):
            if len(selected) != 3:
                raise ValueError("PDHG tuple initialization must contain x, z, and u")
            return OptimState(variables={"est": selected, "cost": None})

        x = physics.A_adjoint(y) if selected is None else selected
        if not isinstance(x, torch.Tensor):
            raise TypeError("PDHG initialization must resolve to a torch.Tensor")
        stack = self._prior_at(0)
        product = _product_forward(x, physics, stack)
        dual = deepinv.utils.TensorList([torch.zeros_like(item) for item in product])
        return OptimState(variables={"est": (x, x, dual), "cost": None})

    def step(
        self,
        state: OptimState,
        y: torch.Tensor,
        physics: Any,
        iteration: int | None = None,
        **kwargs: Any,
    ) -> OptimState:
        """Run one product-space Chambolle--Pock iteration."""
        index = state.iteration if iteration is None else iteration
        if index >= self.max_iter:
            raise IndexError("iteration exceeds max_iter")
        stack = self._prior_at(index)
        fidelity = self._data_fidelity_at(index)
        scheduled = self.params_algo.current(index)
        product_fidelity = _ProductFunctional(
            fidelity,
            stack,
            scheduled["lambda"],
        )
        params = {
            **scheduled,
            "g_param": None,
            "K": lambda value: _product_forward(value, physics, stack),
            "K_adjoint": lambda value: _product_adjoint(value, physics, stack),
        }
        variables = self.iterator(
            state.variables,
            product_fidelity,
            self.zero_prior,
            params,
            y,
            physics,
            **kwargs,
        )
        return OptimState(variables=variables, iteration=index + 1)

    def _prior_at(self, iteration: int) -> StackedPrior:
        return self.prior[iteration] if len(self.prior) > 1 else self.prior[0]

    def _data_fidelity_at(self, iteration: int) -> torch.nn.Module:
        return (
            self.data_fidelity[iteration]
            if len(self.data_fidelity) > 1
            else self.data_fidelity[0]
        )


# %% private module subroutines


class _ProductFunctional:
    def __init__(
        self,
        data_fidelity: Any,
        prior: StackedPrior,
        lambda_reg: torch.Tensor,
    ) -> None:
        self.data_fidelity = data_fidelity
        self.prior = prior
        self.lambda_reg = lambda_reg

    def prox_conjugate(
        self,
        value: Any,
        y: torch.Tensor,
        physics: Any,
        *,
        gamma: float | torch.Tensor,
    ) -> Any:
        del physics
        blocks = [
            self.data_fidelity.prox_d_conjugate(
                value[0],
                y,
                gamma=gamma,
            )
        ]
        blocks.extend(
            self.prior.prox_conjugate(
                index,
                value[index + 1],
                gamma=gamma,
                weight_scale=self.lambda_reg,
            )
            for index in range(len(self.prior))
        )
        return deepinv.utils.TensorList(blocks)


def _product_forward(
    value: torch.Tensor,
    physics: Any,
    prior: StackedPrior,
) -> Any:
    return deepinv.utils.TensorList(
        [physics.A(value)]
        + [prior.transform(index, value) for index in range(len(prior))]
    )


def _product_adjoint(
    value: Any,
    physics: Any,
    prior: StackedPrior,
) -> torch.Tensor:
    result = physics.A_adjoint(value[0])
    for index in range(len(prior)):
        result = result + prior.adjoint(index, value[index + 1])
    return result
