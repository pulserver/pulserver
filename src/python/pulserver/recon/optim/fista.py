"""DeepInverse FISTA with a public state-wise training interface."""

from __future__ import annotations

__all__ = ["FISTA"]

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import torch

import deepinv

from ._base import _physics_parameters, _run_state_aware, _unique_parameters
from .prior import StackedPrior
from .state import OptimResult, OptimState


class FISTA(deepinv.optim.FISTA):
    """DeepInverse FISTA with selective state exposure for advanced training.

    Constructor arguments follow :class:`deepinv.optim.FISTA`. A raw list of
    priors retains DeepInverse's meaning of one prior per iteration. Multiple
    simultaneous priors require PDHG or ADMM; FISTA rejects them because an
    average of proximity operators is not the proximity operator of their sum.

    Parameters
    ----------
    data_fidelity, prior, lambda_reg, stepsize, a, g_param
        See :class:`deepinv.optim.FISTA`.
    max_iter
        Number of FISTA iterations.
    unfold
        Enable end-to-end differentiation through the iterations.
    """

    def __init__(
        self,
        data_fidelity: Any | list[Any] | None = None,
        prior: Any | list[Any] | None = None,
        lambda_reg: float | torch.Tensor = 1.0,
        stepsize: float | torch.Tensor = 1.0,
        a: int = 3,
        g_param: Any | None = None,
        sigma_denoiser: Any | None = None,
        max_iter: int = 100,
        crit_conv: str = "residual",
        thres_conv: float = 1e-5,
        early_stop: bool = False,
        custom_metrics: dict[str, Any] | None = None,
        custom_init: Callable[[torch.Tensor, Any], dict[str, Any]] | None = None,
        g_first: bool = False,
        unfold: bool = False,
        trainable_params: list[str] | None = None,
        cost_fn: Callable[..., torch.Tensor] | None = None,
        params_algo: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        prior, lambda_reg, g_param = _prepare_prior(
            prior,
            lambda_reg,
            g_param,
        )
        if unfold and trainable_params is None and params_algo is None:
            trainable_params = ["lambda", "stepsize", "a"]
            if g_param is not None or sigma_denoiser is not None:
                trainable_params.append("g_param")
        super().__init__(
            data_fidelity=data_fidelity,
            prior=prior,
            lambda_reg=lambda_reg,
            stepsize=stepsize,
            a=a,
            g_param=g_param,
            sigma_denoiser=sigma_denoiser,
            max_iter=max_iter,
            crit_conv=crit_conv,
            thres_conv=thres_conv,
            early_stop=early_stop,
            custom_metrics=custom_metrics,
            custom_init=custom_init,
            g_first=g_first,
            unfold=unfold,
            trainable_params=trainable_params,
            cost_fn=cost_fn,
            params_algo=params_algo,
            **kwargs,
        )
        self.__dict__["_deepinv_get_output"] = self.get_output
        del self.__dict__["get_output"]

    @property
    def iterations(self) -> int:
        """Return the configured number of iterations."""
        return self.max_iter

    def init_state(
        self,
        y: torch.Tensor,
        physics: Any,
        init: Any | None = None,
    ) -> OptimState:
        """Initialize a FISTA state using DeepInverse's initializer."""
        variables = self.init_iterate_fn(
            y,
            physics,
            init=init,
            cost_fn=self.fixed_point.iterator.cost_fn,
        )
        return OptimState(variables=variables)

    def step(
        self,
        state: OptimState,
        y: torch.Tensor,
        physics: Any,
        iteration: int | None = None,
        **kwargs: Any,
    ) -> OptimState:
        """Run one DeepInverse FISTA iteration."""
        index = state.iteration if iteration is None else iteration
        if index >= self.max_iter:
            raise IndexError("iteration exceeds max_iter")
        variables = self.fixed_point.iterator(
            state.variables,
            self.update_data_fidelity_fn(index),
            self.update_prior_fn(index),
            self.update_params_fn(index),
            y,
            physics,
            **kwargs,
        )
        return OptimState(variables=variables, iteration=index + 1)

    def get_output(self, state: OptimState | dict[str, Any]) -> torch.Tensor:
        """Extract the reconstruction from a state or DeepInverse iterate."""
        variables = state.variables if isinstance(state, OptimState) else state
        return self._deepinv_get_output(variables)

    def forward(
        self,
        y: torch.Tensor,
        physics: Any,
        init: Any | None = None,
        x_gt: torch.Tensor | None = None,
        compute_metrics: bool = False,
        *,
        iterations: int | None = None,
        return_info: bool = False,
        record_iterations: Iterable[int] = (),
        detach_every: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | OptimResult | tuple[torch.Tensor, dict[str, Any]]:
        """Run standard DeepInverse FISTA or its state-aware training path."""
        selected = set(record_iterations)
        advanced = (
            iterations is not None
            or return_info
            or bool(selected)
            or detach_every is not None
        )
        if not advanced:
            return super().forward(
                y,
                physics,
                init=init,
                x_gt=x_gt,
                compute_metrics=compute_metrics,
                **kwargs,
            )
        if compute_metrics:
            raise ValueError(
                "compute_metrics cannot be combined with state-aware output"
            )
        return _run_state_aware(
            self,
            y,
            physics,
            init,
            iterations=iterations,
            return_info=return_info,
            record_iterations=record_iterations,
            detach_every=detach_every,
            **kwargs,
        )

    def prior_parameters(
        self,
        stages: Sequence[int] | None = None,
    ) -> list[torch.nn.Parameter]:
        """Return parameters of selected iteration-wise priors."""
        selected = range(len(self.prior)) if stages is None else stages
        return _unique_parameters(
            parameter
            for index in selected
            for parameter in self.prior[index].parameters()
        )

    def transform_parameters(
        self,
        stages: Sequence[int] | None = None,
    ) -> list[torch.nn.Parameter]:
        """Return an empty list because plain FISTA has no transformed blocks."""
        del stages
        return []

    def algorithm_parameters(self) -> list[torch.nn.Parameter]:
        """Return trainable DeepInverse algorithm parameters."""
        return list(self.params_algo.parameters()) if self.unfold else []

    def physics_parameters(self, physics: Any) -> list[torch.nn.Parameter]:
        """Return explicitly trainable physics parameters."""
        return _physics_parameters(physics)

    def parameter_groups(
        self,
        *,
        physics: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Return semantic PyTorch optimizer parameter groups."""
        groups = []
        for name, parameters in (
            ("prior", self.prior_parameters()),
            ("algorithm", self.algorithm_parameters()),
            (
                "physics",
                [] if physics is None else self.physics_parameters(physics),
            ),
        ):
            selected = _unique_parameters(parameters)
            if selected:
                groups.append({"name": name, "params": selected})
        return groups

    def set_trainable(
        self,
        role: str,
        *,
        enabled: bool,
        stages: Sequence[int] | None = None,
        physics: Any | None = None,
    ) -> None:
        """Freeze or unfreeze priors, algorithm parameters, or physics."""
        if role == "prior":
            parameters = self.prior_parameters(stages)
        elif role == "algorithm":
            if stages is not None:
                raise ValueError("stages are only defined for priors")
            parameters = self.algorithm_parameters()
        elif role == "physics":
            if stages is not None:
                raise ValueError("stages are only defined for priors")
            if physics is None:
                raise ValueError("physics must be provided for role='physics'")
            parameters = self.physics_parameters(physics)
        else:
            raise ValueError("role must be 'prior', 'algorithm', or 'physics'")
        for parameter in parameters:
            parameter.requires_grad_(enabled)


# %% private module subroutines


def _prepare_prior(
    prior: Any | list[Any] | None,
    lambda_reg: float | torch.Tensor,
    g_param: Any | None,
) -> tuple[Any | list[Any] | None, float | torch.Tensor, Any | None]:
    if isinstance(prior, list):
        if any(isinstance(item, StackedPrior) for item in prior):
            raise ValueError(
                "StackedPrior cannot be used as an iteration schedule in FISTA"
            )
        return prior, lambda_reg, g_param
    if not isinstance(prior, StackedPrior):
        return prior, lambda_reg, g_param
    if not prior.is_single_identity:
        raise ValueError(
            "FISTA requires one untransformed proximal prior; use PDHG or ADMM "
            "for simultaneous or transformed priors"
        )
    if prior.algorithm_parameters():
        raise ValueError(
            "FISTA trainable weights and g_params must use lambda_reg and g_param"
        )
    weight = prior.weights[0]
    selected_g_param = prior.g_params[0] if g_param is None else g_param
    return prior.priors[0], lambda_reg * weight, selected_g_param
