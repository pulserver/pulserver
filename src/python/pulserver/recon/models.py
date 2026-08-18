"""Selected DeepInverse MRI models and MRI-specific model adapters.

``RAM`` and ``MoDL`` are their native DeepInverse classes: RAM is a pretrained
single-pass option, MoDL a general physics-aware unroll. Both reach the data
only through the physics adjoint, so they accept a Pulserver measurement
unchanged.

``VarNet`` refines the measurement itself rather than only pulling it through
the adjoint, and it is DeepInverse's own class, so it reads measurements in
DeepInverse's channel-first layout. Convert a Pulserver measurement with
:func:`pulserver.recon.physics.measurement_to_channels` before handing it over.
Import other backbones and research models directly from :mod:`deepinv.models`.
"""

from __future__ import annotations

__all__ = ["RAM", "ContextAgnosticDenoiser", "MoDL", "VarNet"]

from math import prod
from typing import Any

import torch

try:
    from deepinv.models import Denoiser as _Denoiser
    from deepinv.models import MoDL as MoDL
    from deepinv.models import RAM as RAM
    from deepinv.models import VarNet as VarNet
except ImportError as error:
    raise ImportError(
        "Learned reconstruction models require DeepInverse, which ships "
        "with pulserver; reinstall the package to restore it."
    ) from error


class ContextAgnosticDenoiser(_Denoiser):
    """Apply a native DeepInverse denoiser independently over context axes.

    Parameters
    ----------
    denoiser
        DeepInverse denoiser accepting ``(x, sigma=..., **kwargs)`` with a
        batch-first, channel-second input.
    spatial_ndim
        Number of trailing spatial axes consumed jointly by the denoiser.
        Axes between channel and space are treated as independent context.
    channels_per_group
        Consecutive channels consumed jointly by the denoiser. ``None`` keeps
        all channels together. For paired-real MRI, ``2`` applies a
        two-channel model independently to each complex coefficient.
    max_batch_size
        Maximum context-and-group batch passed to the denoiser at once.
        ``None`` processes every element together.
    streaming
        Optional :class:`pulserver.recon.execution.CudaStreaming` policy.
        CPU-resident context batches are distributed exactly over its CUDA
        streams and devices during evaluation. Training uses the ordinary
        differentiable path.

    Raises
    ------
    ValueError
        If ``spatial_ndim``, ``channels_per_group``, or ``max_batch_size`` is
        invalid.
    """

    def __init__(
        self,
        denoiser: _Denoiser,
        *,
        spatial_ndim: int = 2,
        channels_per_group: int | None = None,
        max_batch_size: int | None = None,
        streaming: Any | None = None,
    ) -> None:
        super().__init__()
        if (
            not isinstance(spatial_ndim, int)
            or isinstance(spatial_ndim, bool)
            or spatial_ndim < 1
        ):
            raise ValueError("spatial_ndim must be a positive integer")
        if channels_per_group is not None and (
            not isinstance(channels_per_group, int)
            or isinstance(channels_per_group, bool)
            or channels_per_group < 1
        ):
            raise ValueError("channels_per_group must be a positive integer or None")
        if max_batch_size is not None and (
            not isinstance(max_batch_size, int)
            or isinstance(max_batch_size, bool)
            or max_batch_size < 1
        ):
            raise ValueError("max_batch_size must be a positive integer or None")
        self.denoiser = denoiser
        self.spatial_ndim = spatial_ndim
        self.channels_per_group = channels_per_group
        self.max_batch_size = max_batch_size
        if streaming is not None and not callable(
            getattr(streaming, "wrap_batch_denoiser", None)
        ):
            raise TypeError("streaming must provide wrap_batch_denoiser")
        self.streaming = streaming
        self._batch_executor = None

    def train(self, mode: bool = True) -> ContextAgnosticDenoiser:
        """Set training mode and invalidate cached inference replicas."""
        result = super().train(mode)
        self._batch_executor = None
        return result

    def forward(
        self,
        x: torch.Tensor,
        sigma: float | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Denoise a tensor while preserving its batch, context, and layout.

        Parameters
        ----------
        x
            Tensor shaped ``[B, C, *context, *spatial]``. ``C`` must be
            divisible by ``channels_per_group`` when grouping is enabled.
        sigma
            Shared, batch-wise, context-wise, or image-shaped noise level.
        **kwargs
            Conditions forwarded to the native denoiser. Tensor conditions
            with a leading batch axis are aligned with flattened context.

        Returns
        -------
        torch.Tensor
            Denoised tensor shaped ``[B, C_out, *context, *spatial_out]``.

        Raises
        ------
        ValueError
            If the input or denoiser output cannot preserve context layout.
        TypeError
            If the denoiser does not return a tensor.
        """
        flat, batch_size, input_channels, context_shape, group_count = _flatten_context(
            x,
            spatial_ndim=self.spatial_ndim,
            channels_per_group=self.channels_per_group,
        )
        context_count = prod(context_shape)
        total = flat.shape[0]
        adapted_sigma = _adapt_condition(
            sigma,
            batch_size=batch_size,
            input_channels=input_channels,
            context_shape=context_shape,
            group_count=group_count,
        )
        adapted_kwargs = {
            name: _adapt_condition(
                value,
                batch_size=batch_size,
                input_channels=input_channels,
                context_shape=context_shape,
                group_count=group_count,
            )
            for name, value in kwargs.items()
        }
        if total < 1:
            raise ValueError("x must contain at least one batch and context element")
        if (
            self.streaming is not None
            and flat.device.type == "cpu"
            and not self.training
        ):
            if self._batch_executor is None:
                self._batch_executor = self.streaming.wrap_batch_denoiser(
                    self.denoiser,
                    max_batch_size=self.max_batch_size,
                )
            combined = self._batch_executor(
                flat,
                sigma=adapted_sigma,
                **adapted_kwargs,
            )
        else:
            chunk_size = self.max_batch_size or total
            outputs = []
            for start in range(0, total, chunk_size):
                stop = min(start + chunk_size, total)
                chunk_sigma = _slice_condition(adapted_sigma, start, stop, total)
                chunk_kwargs = {
                    name: _slice_condition(value, start, stop, total)
                    for name, value in adapted_kwargs.items()
                }
                output = self.denoiser(
                    flat[start:stop],
                    sigma=chunk_sigma,
                    **chunk_kwargs,
                )
                if not isinstance(output, torch.Tensor):
                    raise TypeError("denoiser must return a Torch tensor")
                if output.ndim < 2 or output.shape[0] != stop - start:
                    raise ValueError(
                        "denoiser output must preserve flattened batch size"
                    )
                outputs.append(output)
            combined = torch.cat(outputs, dim=0)
        if combined.shape[0] != batch_size * context_count * group_count:
            raise ValueError("denoiser output cannot be restored to the input context")
        return _restore_context(
            combined,
            batch_size=batch_size,
            context_shape=context_shape,
            group_count=group_count,
        )


def _flatten_context(
    value: torch.Tensor,
    *,
    spatial_ndim: int,
    channels_per_group: int | None,
) -> tuple[torch.Tensor, int, int, tuple[int, ...], int]:
    if not isinstance(value, torch.Tensor):
        raise TypeError("x must be a Torch tensor")
    if value.ndim < spatial_ndim + 2:
        raise ValueError("x must contain batch, channel, and spatial axes")
    batch_size = value.shape[0]
    input_channels = value.shape[1]
    group_size = input_channels if channels_per_group is None else channels_per_group
    if input_channels % group_size:
        raise ValueError("input channels must be divisible by channels_per_group")
    group_count = input_channels // group_size
    context_shape = tuple(value.shape[2:-spatial_ndim])
    context_axes = tuple(range(2, 2 + len(context_shape)))
    spatial_axes = tuple(range(2 + len(context_shape), value.ndim))
    flat = value.permute(0, *context_axes, 1, *spatial_axes).reshape(
        batch_size * prod(context_shape) * group_count,
        group_size,
        *value.shape[-spatial_ndim:],
    )
    return flat, batch_size, input_channels, context_shape, group_count


def _restore_context(
    value: torch.Tensor,
    *,
    batch_size: int,
    context_shape: tuple[int, ...],
    group_count: int,
) -> torch.Tensor:
    if not context_shape and group_count == 1:
        return value
    context_ndim = len(context_shape)
    arranged = value.reshape(
        batch_size,
        *context_shape,
        group_count,
        value.shape[1],
        *value.shape[2:],
    )
    restored = arranged.permute(
        0,
        context_ndim + 1,
        context_ndim + 2,
        *range(1, context_ndim + 1),
        *range(context_ndim + 3, arranged.ndim),
    )
    spatial_ndim = value.ndim - 2
    return restored.reshape(
        batch_size,
        group_count * value.shape[1],
        *context_shape,
        *value.shape[-spatial_ndim:],
    )


def _adapt_condition(
    value: Any,
    *,
    batch_size: int,
    input_channels: int,
    context_shape: tuple[int, ...],
    group_count: int,
) -> Any:
    if not isinstance(value, torch.Tensor) or value.ndim == 0:
        return value
    context_count = prod(context_shape)
    context_batch = batch_size * context_count
    flat_batch = context_batch * group_count
    if value.shape[0] == flat_batch:
        return value
    if value.shape[0] != batch_size:
        return value
    if value.ndim == 1:
        return value.repeat_interleave(context_count * group_count, dim=0)

    context_ndim = len(context_shape)
    if (
        value.ndim >= context_ndim + 2
        and tuple(value.shape[2 : 2 + context_ndim]) == context_shape
    ):
        context_axes = tuple(range(2, 2 + context_ndim))
        trailing_axes = tuple(range(2 + context_ndim, value.ndim))
        arranged = value.permute(0, *context_axes, 1, *trailing_axes)
        if value.shape[1] == input_channels:
            return arranged.reshape(
                flat_batch,
                input_channels // group_count,
                *value.shape[2 + context_ndim :],
            )
        return arranged.reshape(
            context_batch,
            value.shape[1],
            *value.shape[2 + context_ndim :],
        ).repeat_interleave(group_count, dim=0)
    if tuple(value.shape[1 : 1 + context_ndim]) == context_shape:
        return value.reshape(
            context_batch, *value.shape[1 + context_ndim :]
        ).repeat_interleave(group_count, dim=0)
    if value.shape[1] == input_channels:
        grouped = value.reshape(
            batch_size,
            group_count,
            input_channels // group_count,
            *value.shape[2:],
        )
        return (
            grouped.unsqueeze(1)
            .expand(
                batch_size,
                context_count,
                group_count,
                input_channels // group_count,
                *value.shape[2:],
            )
            .reshape(
                flat_batch,
                input_channels // group_count,
                *value.shape[2:],
            )
        )
    return value.repeat_interleave(context_count * group_count, dim=0)


def _slice_condition(value: Any, start: int, stop: int, total: int) -> Any:
    if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == total:
        return value[start:stop]
    return value
