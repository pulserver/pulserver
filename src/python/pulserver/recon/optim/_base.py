"""Shared implementation for Pulserver iterative optimizers."""

from __future__ import annotations

__all__: list[str] = []

from collections.abc import Iterable, Sequence
from contextlib import nullcontext
from math import isfinite
from typing import Any

import torch

from .prior import StackedPrior
from .state import OptimResult, OptimState


class _IterativeOptimizer(torch.nn.Module):
    def __init__(
        self,
        *,
        max_iter: int,
        early_stop: bool,
        thres_conv: float,
    ) -> None:
        super().__init__()
        if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if not isfinite(thres_conv) or thres_conv <= 0.0:
            raise ValueError("thres_conv must be positive and finite")
        self.max_iter = max_iter
        self.early_stop = bool(early_stop)
        self.thres_conv = float(thres_conv)

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
        """Initialize the optimizer state."""
        raise NotImplementedError

    def step(
        self,
        state: OptimState,
        y: torch.Tensor,
        physics: Any,
        iteration: int | None = None,
        **kwargs: Any,
    ) -> OptimState:
        """Run one optimizer iteration."""
        raise NotImplementedError

    def get_output(self, state: OptimState) -> torch.Tensor:
        """Extract the reconstruction from an optimizer state."""
        return state.estimate

    def forward(
        self,
        y: torch.Tensor,
        physics: Any,
        init: Any | None = None,
        *,
        iterations: int | None = None,
        return_info: bool = False,
        record_iterations: Iterable[int] = (),
        detach_every: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | OptimResult:
        """Reconstruct data using the standard or state-aware interface."""
        count = self.max_iter if iterations is None else iterations
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError("iterations must be a positive integer")
        if count > self.max_iter:
            raise ValueError("iterations cannot exceed the configured max_iter")
        if detach_every is not None and (
            not isinstance(detach_every, int)
            or isinstance(detach_every, bool)
            or detach_every < 1
        ):
            raise ValueError("detach_every must be a positive integer")
        selected = set(record_iterations)
        if any(index < 0 or index > count for index in selected):
            raise ValueError(
                "record_iterations entries must lie between 0 and iterations"
            )

        context = nullcontext() if getattr(self, "unfold", False) else torch.no_grad()
        with context:
            state = self.init_state(y, physics, init)
            history = {0: state.estimate} if 0 in selected else {}
            for iteration in range(count):
                previous = state.estimate
                state = self.step(
                    state,
                    y,
                    physics,
                    iteration=iteration,
                    **kwargs,
                )
                completed = iteration + 1
                if completed in selected:
                    history[completed] = state.estimate
                if (
                    self.early_stop
                    and _relative_residual(previous, state.estimate) <= self.thres_conv
                ):
                    break
                if detach_every is not None and completed % detach_every == 0:
                    state = state.detach()
        reconstruction = self.get_output(state)
        if return_info:
            return OptimResult(reconstruction, state, history)
        return reconstruction

    def prior_parameters(
        self,
        stages: Sequence[int] | None = None,
    ) -> list[torch.nn.Parameter]:
        """Return parameters belonging to learned prior blocks."""
        return _parameters_from_stacks(self, "prior_parameters", stages)

    def transform_parameters(
        self,
        stages: Sequence[int] | None = None,
    ) -> list[torch.nn.Parameter]:
        """Return parameters belonging to regularizer transforms."""
        return _parameters_from_stacks(self, "transform_parameters", stages)

    def algorithm_parameters(self) -> list[torch.nn.Parameter]:
        """Return trainable step sizes, penalties, and regularization weights."""
        schedule = getattr(self, "params_algo", None)
        parameters = (
            list(schedule.parameters()) if isinstance(schedule, torch.nn.Module) else []
        )
        prior = getattr(self, "prior", None)
        stacks = [prior] if isinstance(prior, StackedPrior) else prior or []
        for stack in stacks:
            if isinstance(stack, StackedPrior):
                parameters.extend(stack.algorithm_parameters())
        return _unique_parameters(parameters)

    def physics_parameters(self, physics: Any) -> list[torch.nn.Parameter]:
        """Return explicitly trainable parameters owned by a physics object."""
        return _physics_parameters(physics)

    def parameter_groups(
        self,
        *,
        physics: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Return named PyTorch optimizer groups for advanced training policies."""
        groups = []
        for name, parameters in (
            ("prior", self.prior_parameters()),
            ("transform", self.transform_parameters()),
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
        """Freeze or unfreeze a semantic parameter group."""
        if role == "prior":
            parameters = self.prior_parameters(stages)
        elif role == "transform":
            parameters = self.transform_parameters(stages)
        elif role == "algorithm":
            if stages is not None:
                raise ValueError("stages are only defined for priors and transforms")
            parameters = self.algorithm_parameters()
        elif role == "physics":
            if stages is not None:
                raise ValueError("stages are only defined for priors and transforms")
            if physics is None:
                raise ValueError("physics must be provided for role='physics'")
            parameters = self.physics_parameters(physics)
        else:
            raise ValueError(
                "role must be 'prior', 'transform', 'algorithm', or 'physics'"
            )
        for parameter in parameters:
            parameter.requires_grad_(enabled)


class _AlgorithmSchedule(torch.nn.Module):
    def __init__(
        self,
        values: dict[str, float | Sequence[float] | torch.Tensor],
        *,
        max_iter: int,
        unfold: bool,
        trainable_params: Sequence[str] | None,
    ) -> None:
        super().__init__()
        selected = set(values) if trainable_params is None else set(trainable_params)
        unknown = selected.difference(values)
        if unknown:
            raise ValueError(f"unknown trainable parameters: {sorted(unknown)}")
        self._names = tuple(values)
        self._trainable = torch.nn.ParameterDict()
        self._buffers: dict[str, str] = {}
        for name, value in values.items():
            tensor = _schedule_tensor(value, max_iter, name)
            if unfold and name in selected:
                self._trainable[name] = torch.nn.Parameter(tensor)
            else:
                buffer_name = f"_value_{name}"
                self.register_buffer(buffer_name, tensor)
                self._buffers[name] = buffer_name

    def at(self, name: str, iteration: int) -> torch.Tensor:
        values = (
            self._trainable[name]
            if name in self._trainable
            else getattr(self, self._buffers[name])
        )
        return values[iteration] if values.numel() > 1 else values[0]

    def current(self, iteration: int) -> dict[str, torch.Tensor]:
        return {name: self.at(name, iteration) for name in self._names}


# %% private module subroutines


def _schedule_tensor(
    value: float | Sequence[float] | torch.Tensor,
    max_iter: int,
    name: str,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim == 0:
        tensor = tensor.reshape(1)
    if tensor.ndim != 1 or tensor.numel() not in {1, max_iter}:
        raise ValueError(f"{name} must be scalar or contain max_iter values")
    if not bool(torch.all(torch.isfinite(tensor))):
        raise ValueError(f"{name} must be finite")
    return tensor


def _relative_residual(previous: torch.Tensor, current: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(current - previous)
    denominator = torch.linalg.vector_norm(current).clamp_min(1e-12)
    return float(numerator / denominator)


def _parameters_from_stacks(
    module: torch.nn.Module,
    method: str,
    stages: Sequence[int] | None,
) -> list[torch.nn.Parameter]:
    prior = getattr(module, "prior", None)
    if isinstance(prior, StackedPrior):
        return getattr(prior, method)(stages)
    if isinstance(prior, torch.nn.ModuleList):
        selected = range(len(prior)) if stages is None else stages
        parameters = []
        for index in selected:
            block = prior[index]
            if isinstance(block, StackedPrior):
                parameters.extend(getattr(block, method)())
            elif method == "prior_parameters":
                parameters.extend(block.parameters())
        return _unique_parameters(parameters)
    return [] if prior is None else list(prior.parameters())


def _physics_parameters(physics: Any) -> list[torch.nn.Parameter]:
    candidates = [physics, getattr(physics, "operator", None)]
    parameters = []
    for candidate in candidates:
        method = getattr(candidate, "parameters", None)
        if method is not None:
            parameters.extend(method())
        buffers = getattr(candidate, "buffers", None)
        if buffers is not None:
            parameters.extend(buffer for buffer in buffers() if buffer.requires_grad)
    return _unique_parameters(parameters)


def _unique_parameters(
    parameters: Iterable[torch.nn.Parameter],
) -> list[torch.nn.Parameter]:
    result = []
    identities = set()
    for parameter in parameters:
        if id(parameter) not in identities:
            identities.add(id(parameter))
            result.append(parameter)
    return result
