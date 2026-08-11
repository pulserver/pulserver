"""Product-space priors for simultaneous MRI regularization."""

from __future__ import annotations

__all__ = ["StackedPrior"]

from collections.abc import Sequence
from typing import Any

import torch


class StackedPrior(torch.nn.Module):
    r"""A simultaneous sum of transformed DeepInverse priors.

    This represents ``sum_i weights[i] * priors[i](transforms[i](x))``.
    Unlike passing a list directly to a DeepInverse optimizer, the entries are
    active at the same iteration rather than scheduled across iterations.

    Parameters
    ----------
    priors
        DeepInverse-compatible prior modules.
    transforms
        Linear transforms exposing ``A`` and ``A_adjoint``, ``(A, A_adjoint)``
        callable pairs, or ``None`` for identity.
    weights
        Non-negative regularization weights, one per prior.
    g_params
        Prior parameters such as denoiser noise levels, one per prior.
    trainable_weights
        Register ``weights`` as a trainable parameter.
    trainable_g_params
        Register numeric prior parameters as trainable parameters.
    """

    def __init__(
        self,
        priors: Sequence[torch.nn.Module],
        *,
        transforms: Sequence[Any | None] | None = None,
        weights: float | Sequence[float] | torch.Tensor = 1.0,
        g_params: Any | Sequence[Any] | None = None,
        trainable_weights: bool = False,
        trainable_g_params: bool = False,
    ) -> None:
        super().__init__()
        selected_priors = list(priors)
        if transforms is None:
            transforms = [None] * len(selected_priors)
        if len(transforms) != len(selected_priors):
            raise ValueError("transforms and priors must have the same length")
        if not all(isinstance(prior, torch.nn.Module) for prior in selected_priors):
            raise TypeError("every prior must be a torch.nn.Module")

        self.priors = torch.nn.ModuleList(selected_priors)
        self.transforms = torch.nn.ModuleList(
            [_TransformAdapter(transform) for transform in transforms]
        )
        weight_tensor = _weights_tensor(weights, len(selected_priors))
        if trainable_weights:
            self.weights = torch.nn.Parameter(weight_tensor)
        else:
            self.register_buffer("weights", weight_tensor)
        selected_g_params = _expand_values(g_params, len(selected_priors))
        if trainable_g_params:
            if any(value is None for value in selected_g_params):
                raise ValueError("trainable g_params cannot contain None")
            self.g_params = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(torch.as_tensor(value, dtype=torch.float32))
                    for value in selected_g_params
                ]
            )
        else:
            self.g_params = selected_g_params

    def __len__(self) -> int:
        return len(self.priors)

    @property
    def is_single_identity(self) -> bool:
        """Whether this is one untransformed prior usable by plain FISTA."""
        return len(self) == 1 and self.transforms[0].is_identity

    def transform(self, index: int, value: torch.Tensor) -> torch.Tensor:
        """Apply one regularizer transform."""
        return self.transforms[index].A(value)

    def adjoint(self, index: int, value: torch.Tensor) -> torch.Tensor:
        """Apply one regularizer transform adjoint."""
        return self.transforms[index].A_adjoint(value)

    def normal(self, value: torch.Tensor) -> torch.Tensor:
        """Apply the sum of transform normal operators."""
        result = torch.zeros_like(value)
        for transform in self.transforms:
            result = result + transform.A_adjoint(transform.A(value))
        return result

    def prox(
        self,
        index: int,
        value: torch.Tensor,
        *,
        gamma: float | torch.Tensor,
        weight_scale: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        """Apply one weighted prior proximity operator."""
        weight = self.weights[index].to(device=value.device, dtype=value.real.dtype)
        return self.priors[index].prox(
            value,
            self.g_params[index],
            gamma=gamma * weight_scale * weight,
        )

    def prox_conjugate(
        self,
        index: int,
        value: torch.Tensor,
        *,
        gamma: float | torch.Tensor,
        weight_scale: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        """Apply one weighted convex-conjugate proximity operator."""
        weight = self.weights[index].to(device=value.device, dtype=value.real.dtype)
        return self.priors[index].prox_conjugate(
            value,
            self.g_params[index],
            gamma=gamma,
            lamb=weight_scale * weight,
        )

    def cost(self, value: torch.Tensor) -> torch.Tensor:
        """Evaluate the explicit weighted prior sum."""
        result = None
        for index, prior in enumerate(self.priors):
            transformed = self.transform(index, value)
            current = prior(transformed, self.g_params[index])
            weight = self.weights[index].to(
                device=current.device,
                dtype=current.dtype,
            )
            result = weight * current if result is None else result + weight * current
        if result is None:
            return torch.zeros(
                value.shape[0], device=value.device, dtype=value.real.dtype
            )
        return result

    def prior_parameters(
        self,
        stages: Sequence[int] | None = None,
    ) -> list[torch.nn.Parameter]:
        """Return parameters of selected prior blocks."""
        selected = range(len(self)) if stages is None else stages
        return _unique_parameters(
            parameter
            for index in selected
            for parameter in self.priors[index].parameters()
        )

    def transform_parameters(
        self,
        stages: Sequence[int] | None = None,
    ) -> list[torch.nn.Parameter]:
        """Return parameters of selected transform blocks."""
        selected = range(len(self)) if stages is None else stages
        return _unique_parameters(
            parameter
            for index in selected
            for parameter in self.transforms[index].parameters()
        )

    def algorithm_parameters(self) -> list[torch.nn.Parameter]:
        """Return trainable regularizer weights and prior parameters."""
        parameters = []
        if isinstance(self.weights, torch.nn.Parameter):
            parameters.append(self.weights)
        if isinstance(self.g_params, torch.nn.ParameterList):
            parameters.extend(self.g_params.parameters())
        return parameters


# %% private module subroutines


class _TransformAdapter(torch.nn.Module):
    def __init__(self, transform: Any | None) -> None:
        super().__init__()
        self.is_identity = transform is None
        if isinstance(transform, torch.nn.Module):
            self.operator = transform
        else:
            self.__dict__["operator"] = transform

    def A(self, value: torch.Tensor) -> torch.Tensor:
        if self.operator is None:
            return value
        if isinstance(self.operator, tuple):
            return self.operator[0](value)
        method = getattr(self.operator, "A", None)
        return method(value) if method is not None else self.operator(value)

    def A_adjoint(self, value: torch.Tensor) -> torch.Tensor:
        if self.operator is None:
            return value
        if isinstance(self.operator, tuple):
            return self.operator[1](value)
        method = getattr(self.operator, "A_adjoint", None)
        if method is None:
            method = getattr(self.operator, "adjoint", None)
        if method is None:
            raise TypeError("a transform must expose A_adjoint or adjoint")
        return method(value)


def _weights_tensor(
    weights: float | Sequence[float] | torch.Tensor,
    count: int,
) -> torch.Tensor:
    values = torch.as_tensor(weights, dtype=torch.float32)
    if values.ndim == 0:
        values = values.repeat(count)
    if values.ndim != 1 or values.numel() != count:
        raise ValueError("weights must be scalar or contain one value per prior")
    if not bool(torch.all(torch.isfinite(values))) or bool(torch.any(values < 0.0)):
        raise ValueError("weights must be finite and non-negative")
    return values


def _expand_values(value: Any, count: int) -> list[Any]:
    if count == 0:
        return []
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, torch.Tensor)
    ):
        values = list(value)
        if len(values) != count:
            raise ValueError("g_params must contain one value per prior")
        return values
    return [value] * count


def _unique_parameters(
    parameters: Sequence[torch.nn.Parameter] | Any,
) -> list[torch.nn.Parameter]:
    result = []
    identities = set()
    for parameter in parameters:
        if id(parameter) not in identities:
            identities.add(id(parameter))
            result.append(parameter)
    return result
