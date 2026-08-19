"""Stateful learned-reconstruction primitives for memory-aware training."""

from __future__ import annotations

__all__ = [
    "GradientDataConsistency",
    "ScaledAdjoint",
    "StatefulReconstructor",
    "UnrollResult",
    "UnrollState",
    "UnrolledReconstructor",
]

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from math import isfinite
from typing import Any, Protocol, runtime_checkable

import torch


def _as_real_channels(
    value: torch.Tensor,
    *,
    channel_dim: int = 1,
) -> torch.Tensor:
    """Pack complex channels into adjacent real and imaginary channels.

    Parameters
    ----------
    value
        Complex tensor.
    channel_dim
        Channel dimension in ``value``.

    Returns
    -------
    torch.Tensor
        Real tensor with twice as many channels.
    """
    if not isinstance(value, torch.Tensor) or not value.is_complex():
        raise TypeError("value must be a complex Torch tensor")
    axis = _channel_axis(value.ndim, channel_dim)
    paired = torch.view_as_real(value).movedim(-1, axis + 1)
    shape = list(value.shape)
    shape[axis] *= 2
    return paired.reshape(shape)


def _as_complex_channels(
    value: torch.Tensor,
    *,
    channel_dim: int = 1,
) -> torch.Tensor:
    """Restore complex channels from adjacent real and imaginary pairs.

    Parameters
    ----------
    value
        Real tensor with adjacent real and imaginary channel pairs.
    channel_dim
        Channel dimension in ``value``.

    Returns
    -------
    torch.Tensor
        Complex tensor with half as many channels.
    """
    if not isinstance(value, torch.Tensor) or value.is_complex():
        raise TypeError("value must be a real Torch tensor")
    axis = _channel_axis(value.ndim, channel_dim)
    channels = value.shape[axis]
    if channels % 2:
        raise ValueError("paired-real tensors require an even channel count")
    shape = list(value.shape)
    shape[axis : axis + 1] = [channels // 2, 2]
    paired = value.reshape(shape).movedim(axis + 1, -1)
    return torch.view_as_complex(paired.contiguous())


class _ComplexAdapter(torch.nn.Module):
    """Apply a paired-real model to native complex MRI tensors.

    Parameters
    ----------
    model
        Module accepting and returning paired-real channels.
    channel_dim
        Channel dimension of input and output tensors.
    """

    def __init__(self, model: torch.nn.Module, *, channel_dim: int = 1) -> None:
        super().__init__()
        self.model = model
        self.channel_dim = int(channel_dim)

    def forward(
        self,
        value: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply the model while preserving a complex input representation."""
        restore_complex = value.is_complex()
        model_input = (
            _as_real_channels(value, channel_dim=self.channel_dim)
            if restore_complex
            else value
        )
        result = self.model(model_input, *args, **kwargs)
        if not isinstance(result, torch.Tensor):
            raise TypeError("a complex-adapted model must return a Torch tensor")
        if not restore_complex or result.is_complex():
            return result
        return _as_complex_channels(result, channel_dim=self.channel_dim)


@runtime_checkable
class StatefulReconstructor(Protocol):
    """Interface implemented by step-wise reconstruction models."""

    @property
    def iterations(self) -> int:
        """Return the configured number of reconstruction steps."""
        ...

    def init_state(
        self,
        y: torch.Tensor,
        physics: Any,
        init: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Create the state consumed by the first reconstruction step."""
        ...

    def step(
        self,
        state: Any,
        y: torch.Tensor,
        physics: Any,
        iteration: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """Apply one reconstruction step."""
        ...

    def get_output(self, state: Any) -> torch.Tensor:
        """Extract the image estimate from a reconstruction state."""
        ...


@dataclass(frozen=True)
class UnrollState:
    """State carried between learned reconstruction steps.

    Parameters
    ----------
    estimate
        Current image or coefficient estimate.
    rhs
        Adjoint data term used by data-consistency updates.
    iteration
        Number of completed reconstruction steps.
    auxiliary
        Method-specific values carried between steps.
    """

    estimate: torch.Tensor
    rhs: torch.Tensor
    iteration: int = 0
    auxiliary: Mapping[str, Any] = field(default_factory=dict)

    def detach(self) -> UnrollState:
        """Detach all tensor state at a greedy or truncated-BPTT boundary."""
        return replace(
            self,
            estimate=self.estimate.detach(),
            rhs=self.rhs.detach(),
            auxiliary=_detach(self.auxiliary),
        )


@dataclass(frozen=True)
class UnrollResult:
    """Learned reconstruction output with state and selected iterates.

    Parameters
    ----------
    reconstruction
        Final image or coefficient estimate.
    state
        Final reconstruction state.
    history
        Requested estimates indexed by completed step count.
    """

    reconstruction: torch.Tensor
    state: UnrollState
    history: Mapping[int, torch.Tensor]


class ScaledAdjoint(torch.nn.Module):
    """Initialize an unroll with an optionally norm-aligned adjoint.

    Parameters
    ----------
    normalize
        Scale the adjoint so its re-encoding norm matches the measurements.
    epsilon
        Minimum denominator used in the per-batch norm ratio.
    """

    def __init__(self, *, normalize: bool = True, epsilon: float = 1e-8) -> None:
        super().__init__()
        if not isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("epsilon must be positive and finite")
        self.normalize = bool(normalize)
        self.epsilon = float(epsilon)

    def forward(
        self,
        y: torch.Tensor,
        physics: Any,
        **_context: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the initialized estimate and adjoint right-hand side."""
        rhs = physics.A_adjoint(y)
        if not self.normalize:
            return rhs, rhs
        encoded = physics.A(rhs)
        numerator = _batch_norm(y)
        denominator = _batch_norm(encoded).clamp_min(self.epsilon)
        scale = numerator / denominator
        scale = scale.reshape(scale.shape + (1,) * (rhs.ndim - scale.ndim))
        return rhs * scale, rhs


class GradientDataConsistency(torch.nn.Module):
    """Apply trainable gradient data-consistency updates.

    Parameters
    ----------
    iterations
        Number of unrolled updates.
    stepsize
        Scalar or one initial step size per update.
    trainable
        Register step sizes as trainable parameters.
    """

    def __init__(
        self,
        iterations: int,
        *,
        stepsize: float | Sequence[float] | torch.Tensor = 1.0,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        _iterations(iterations)
        values = _schedule(stepsize, iterations, name="stepsize", positive=True)
        if trainable:
            self.stepsizes = torch.nn.Parameter(values)
        else:
            self.register_buffer("stepsizes", values)

    @property
    def iterations(self) -> int:
        """Return the number of configured data-consistency updates."""
        return int(self.stepsizes.numel())

    def forward(
        self,
        estimate: torch.Tensor,
        y: torch.Tensor,
        physics: Any,
        *,
        iteration: int,
        rhs: torch.Tensor | None = None,
        **_context: Any,
    ) -> torch.Tensor:
        """Apply one gradient step using the physics normal operator."""
        if iteration < 0 or iteration >= self.iterations:
            raise IndexError("iteration exceeds configured data-consistency steps")
        if rhs is None:
            rhs = physics.A_adjoint(y)
        normal = getattr(physics, "A_adjoint_A", None)
        transformed = (
            normal(estimate)
            if normal is not None
            else physics.A_adjoint(physics.A(estimate))
        )
        stepsize = self.stepsizes[iteration].to(
            device=estimate.device,
            dtype=estimate.real.dtype,
        )
        return estimate - stepsize * (transformed - rhs)


class UnrolledReconstructor(torch.nn.Module):
    """Compose data consistency and learned denoisers with explicit state.

    Parameters
    ----------
    denoiser
        Shared denoiser or one denoiser per reconstruction step. Repeating the
        same module in a sequence expresses grouped weight sharing.
    iterations
        Number of unrolled reconstruction steps.
    data_consistency
        Step module with the ``GradientDataConsistency`` call contract.
    initializer
        Callable returning ``(estimate, rhs)`` or an initial estimate.
    denoiser_strength
        Scalar, one value per step, or ``None``.
    trainable_strength
        Register denoiser strengths as trainable parameters.
    conditioning
        ``"none"``, ``"iteration"``, ``"context"``, ``"all"``, or a
        callable producing denoiser keyword arguments.
    """

    def __init__(
        self,
        denoiser: Any | Sequence[Any],
        *,
        iterations: int,
        data_consistency: Any | None = None,
        initializer: Any | None = None,
        denoiser_strength: float | Sequence[float] | torch.Tensor | None = None,
        trainable_strength: bool = False,
        conditioning: str
        | Callable[[int, Mapping[str, Any]], Mapping[str, Any]] = "none",
    ) -> None:
        super().__init__()
        _iterations(iterations)
        self._iterations = iterations
        if isinstance(denoiser, Sequence) and not isinstance(
            denoiser, (str, bytes, torch.nn.Module)
        ):
            selected = tuple(denoiser)
            if len(selected) != iterations:
                raise ValueError("denoiser sequences require one entry per iteration")
            if not all(isinstance(item, torch.nn.Module) for item in selected):
                raise TypeError("every denoiser must be a Torch module")
            self.denoisers = torch.nn.ModuleList(selected)
            self.denoiser = None
        else:
            if not isinstance(denoiser, torch.nn.Module):
                raise TypeError("denoiser must be a Torch module")
            self.denoiser = denoiser
            self.denoisers = None

        self.data_consistency = (
            GradientDataConsistency(iterations)
            if data_consistency is None
            else data_consistency
        )
        if not callable(self.data_consistency):
            raise TypeError("data_consistency must be callable")
        self.initializer = ScaledAdjoint() if initializer is None else initializer
        if not callable(self.initializer):
            raise TypeError("initializer must be callable")

        if denoiser_strength is None:
            self.register_buffer("denoiser_strengths", None)
        else:
            strengths = _schedule(
                denoiser_strength,
                iterations,
                name="denoiser_strength",
                positive=False,
            )
            if trainable_strength:
                self.denoiser_strengths = torch.nn.Parameter(strengths)
            else:
                self.register_buffer("denoiser_strengths", strengths)

        if isinstance(conditioning, str) and conditioning not in {
            "none",
            "iteration",
            "context",
            "all",
        }:
            raise ValueError(
                "conditioning must be 'none', 'iteration', 'context', 'all', "
                "or a callable"
            )
        if not isinstance(conditioning, str) and not callable(conditioning):
            raise TypeError("conditioning must be a supported string or callable")
        self.conditioning = conditioning

    @property
    def iterations(self) -> int:
        """Return the configured number of unrolled steps."""
        return self._iterations

    def init_state(
        self,
        y: torch.Tensor,
        physics: Any,
        init: Any | None = None,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> UnrollState:
        """Create an initialized reconstruction state."""
        selected_context = {} if context is None else dict(context)
        if init is None:
            initialized = self.initializer(y, physics, **selected_context)
        elif callable(init):
            initialized = init(y, physics, **selected_context)
        else:
            initialized = init
        if isinstance(initialized, tuple):
            if len(initialized) != 2:
                raise ValueError("initializer tuples must contain estimate and rhs")
            estimate, rhs = initialized
        else:
            estimate = initialized
            rhs = physics.A_adjoint(y)
        return UnrollState(estimate=estimate, rhs=rhs)

    def data_consistent(
        self,
        state: UnrollState,
        y: torch.Tensor,
        physics: Any,
        *,
        iteration: int | None = None,
        context: Mapping[str, Any] | None = None,
        checkpoint: bool = False,
    ) -> torch.Tensor:
        """Apply one optionally checkpointed data-consistency block."""
        index = state.iteration if iteration is None else iteration
        selected_context = {} if context is None else dict(context)

        def apply(estimate: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
            return self.data_consistency(
                estimate,
                y,
                physics,
                iteration=index,
                rhs=rhs,
                **selected_context,
            )

        return (
            _checkpoint(apply, state.estimate, state.rhs)
            if checkpoint
            else apply(
                state.estimate,
                state.rhs,
            )
        )

    def denoise(
        self,
        value: torch.Tensor,
        *,
        iteration: int,
        context: Mapping[str, Any] | None = None,
        checkpoint: bool = False,
    ) -> torch.Tensor:
        """Apply one optionally checkpointed denoiser block."""
        if iteration < 0 or iteration >= self.iterations:
            raise IndexError("iteration exceeds configured denoiser steps")
        model = self.denoiser if self.denoisers is None else self.denoisers[iteration]
        selected_context = {} if context is None else dict(context)
        kwargs = self._conditioning_kwargs(iteration, selected_context)
        strength = (
            None
            if self.denoiser_strengths is None
            else self.denoiser_strengths[iteration].to(
                device=value.device,
                dtype=value.real.dtype,
            )
        )

        def apply(tensor: torch.Tensor) -> torch.Tensor:
            if strength is None:
                return model(tensor, **kwargs)
            return model(tensor, strength, **kwargs)

        return _checkpoint(apply, value) if checkpoint else apply(value)

    def step(
        self,
        state: UnrollState,
        y: torch.Tensor,
        physics: Any,
        iteration: int | None = None,
        *,
        context: Mapping[str, Any] | None = None,
        checkpoint: bool | str | Iterable[str] = False,
    ) -> UnrollState:
        """Apply one data-consistency and denoising step."""
        index = state.iteration if iteration is None else iteration
        if index < 0 or index >= self.iterations:
            raise IndexError("iteration exceeds configured unroll steps")
        blocks = _checkpoint_blocks(checkpoint)
        intermediate = self.data_consistent(
            state,
            y,
            physics,
            iteration=index,
            context=context,
            checkpoint="data_consistency" in blocks,
        )
        estimate = self.denoise(
            intermediate,
            iteration=index,
            context=context,
            checkpoint="denoiser" in blocks,
        )
        return UnrollState(
            estimate=estimate,
            rhs=state.rhs,
            iteration=index + 1,
            auxiliary=state.auxiliary,
        )

    def iterate(
        self,
        y: torch.Tensor,
        physics: Any,
        init: Any | None = None,
        *,
        iterations: int | None = None,
        context: Mapping[str, Any] | None = None,
        checkpoint: bool | str | Iterable[str] = False,
        detach_between: bool = False,
    ) -> Iterator[UnrollState]:
        """Yield each state for intermediate supervision or greedy training.

        When ``detach_between`` is true, the yielded state's graph remains valid
        until the generator advances. A caller can backpropagate its local loss
        before requesting the next state.
        """
        count = self.iterations if iterations is None else iterations
        _run_iterations(count, self.iterations)
        state = self.init_state(y, physics, init, context=context)
        for index in range(count):
            state = self.step(
                state,
                y,
                physics,
                index,
                context=context,
                checkpoint=checkpoint,
            )
            yield state
            if detach_between:
                state = state.detach()

    def get_output(self, state: UnrollState) -> torch.Tensor:
        """Return the current image estimate."""
        return state.estimate

    def forward(
        self,
        y: torch.Tensor,
        physics: Any,
        init: Any | None = None,
        *,
        iterations: int | None = None,
        context: Mapping[str, Any] | None = None,
        checkpoint: bool | str | Iterable[str] = False,
        detach_every: int | None = None,
        return_info: bool = False,
        record_iterations: Iterable[int] = (),
    ) -> torch.Tensor | UnrollResult:
        """Run a full, checkpointed, or truncated unroll."""
        count = self.iterations if iterations is None else iterations
        _run_iterations(count, self.iterations)
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
        state = self.init_state(y, physics, init, context=context)
        history = {0: state.estimate} if 0 in selected else {}
        for index in range(count):
            state = self.step(
                state,
                y,
                physics,
                index,
                context=context,
                checkpoint=checkpoint,
            )
            completed = index + 1
            if completed in selected:
                history[completed] = state.estimate
            if (
                detach_every is not None
                and completed < count
                and completed % detach_every == 0
            ):
                state = state.detach()
        if return_info:
            return UnrollResult(state.estimate, state, history)
        return state.estimate

    def _conditioning_kwargs(
        self,
        iteration: int,
        context: Mapping[str, Any],
    ) -> dict[str, Any]:
        if callable(self.conditioning):
            result = self.conditioning(iteration, context)
            if not isinstance(result, Mapping):
                raise TypeError("conditioning callables must return a mapping")
            return dict(result)
        result = dict(context) if self.conditioning in {"context", "all"} else {}
        if self.conditioning in {"iteration", "all"}:
            result["iteration"] = iteration
        return result


def _batch_norm(value: torch.Tensor) -> torch.Tensor:
    if value.ndim < 1:
        raise ValueError("measurement tensors require a batch dimension")
    if value.ndim == 1:
        return value.abs()
    return torch.linalg.vector_norm(value, dim=tuple(range(1, value.ndim)))


def _channel_axis(ndim: int, channel_dim: int) -> int:
    if ndim < 2:
        raise ValueError("MRI model tensors require at least two dimensions")
    axis = channel_dim if channel_dim >= 0 else ndim + channel_dim
    if axis < 0 or axis >= ndim:
        raise ValueError(f"channel_dim {channel_dim} is invalid for {ndim} dimensions")
    return axis


def _schedule(
    value: float | Sequence[float] | torch.Tensor,
    iterations: int,
    *,
    name: str,
    positive: bool,
) -> torch.Tensor:
    values = torch.as_tensor(value, dtype=torch.float32)
    if values.ndim == 0:
        values = values.repeat(iterations)
    if values.ndim != 1 or values.numel() != iterations:
        raise ValueError(f"{name} must be scalar or contain one value per iteration")
    if not bool(torch.all(torch.isfinite(values))):
        raise ValueError(f"{name} must be finite")
    if positive and not bool(torch.all(values > 0.0)):
        raise ValueError(f"{name} must be positive")
    if not positive and bool(torch.any(values < 0.0)):
        raise ValueError(f"{name} must be non-negative")
    return values


def _iterations(iterations: int) -> None:
    if (
        not isinstance(iterations, int)
        or isinstance(iterations, bool)
        or iterations < 1
    ):
        raise ValueError("iterations must be a positive integer")


def _run_iterations(iterations: int, maximum: int) -> None:
    _iterations(iterations)
    if iterations > maximum:
        raise ValueError("iterations cannot exceed the configured unroll")


def _checkpoint_blocks(value: bool | str | Iterable[str]) -> frozenset[str]:
    if value is True:
        return frozenset({"data_consistency", "denoiser"})
    if value is False:
        return frozenset()
    selected = {value} if isinstance(value, str) else set(value)
    unknown = selected - {"data_consistency", "denoiser"}
    if unknown:
        raise ValueError(
            "checkpoint blocks must be 'data_consistency' and/or 'denoiser'"
        )
    return frozenset(selected)


def _checkpoint(
    function: Callable[..., torch.Tensor], *args: torch.Tensor
) -> torch.Tensor:
    return torch.utils.checkpoint.checkpoint(function, *args, use_reentrant=False)


def _detach(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, tuple):
        return tuple(_detach(item) for item in value)
    if isinstance(value, list):
        return [_detach(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _detach(item) for key, item in value.items()}
    detach = getattr(value, "detach", None)
    return detach() if callable(detach) else value
