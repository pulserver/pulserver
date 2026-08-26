"""MRI-specific adapters over DeepInverse.

The learned reconstructions are DeepInverse's. Any of its optimizers becomes
an unrolled network by taking ``unfold=True``, and its model zoo -- DnCNN,
DRUNet, MoDL, VarNet, RAM and the rest -- is imported directly; nothing here
reimplements one. What lives here is the adaptation each needs to reach MRI
data, and every name plugs into an extension point DeepInverse already
defines.

A DeepInverse denoiser expects one real image: ``(batch, channels, height,
width)``, two or three channels, everything else in the batch. MRI carries
complex volumes with a contrast, echo, or frame axis on top of the spatial
ones, and asks for them to be denoised together. Each model adapter closes
one of those gaps and returns a denoiser, so they compose:

.. code-block:: text

    ContextAgnosticDenoiser   an extra axis (slices, frames, coefficients)
    ComplexDenoiser           complex data through a real-valued network
    NoiseConditioned          a blind network told what noise level to remove
    Checkpointed              recompute in the backward pass instead of storing

A network that works in k-space is not one of these. Keeping the iterate in
k-space changes the algorithm, not the model -- the measured lines have to be
held fixed and the unmeasured ones filled -- and the network in such a cascade
still denoises an image. :class:`deepinv.models.VarNet` with
``mode="e2e-varnet"`` is that algorithm.

Three more adapters serve the algorithm rather than the model:

.. code-block:: text

    ScaledAdjoint             the initializer, passed as ``custom_init``
    NormalEquationL2          data fidelity through the normal operator
    StepwiseUnroll            the optimizer's loop, one step at a time
"""

from __future__ import annotations

__all__ = [
    "Checkpointed",
    "ComplexDenoiser",
    "ContextAgnosticDenoiser",
    "NoiseConditioned",
    "NormalEquationL2",
    "ScaledAdjoint",
    "StepwiseUnroll",
]

import weakref
from collections.abc import Iterator
from math import isfinite, prod
from typing import Any

import torch
import torch.utils.checkpoint

try:
    from deepinv.models import Denoiser as _Denoiser
    from deepinv.optim.data_fidelity import L2 as _L2
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

    Examples
    --------
    Wraps a denoiser so it can be applied to whatever a reconstruction is
    carrying -- coil images, subspace coefficients, a stack of slices -- by
    folding the extra axes into the channel dimension it expects.

    >>> import pulserver.recon as recon
    >>> wrapped = recon.ContextAgnosticDenoiser(recon.TV(), spatial_ndim=2)
    >>> isinstance(wrapped, recon.ContextAgnosticDenoiser)
    True
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


def _as_real_channels(value: torch.Tensor, *, channel_dim: int = 1) -> torch.Tensor:
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


def _as_complex_channels(value: torch.Tensor, *, channel_dim: int = 1) -> torch.Tensor:
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


def _channel_axis(ndim: int, channel_dim: int) -> int:
    if not isinstance(channel_dim, int) or isinstance(channel_dim, bool):
        raise TypeError("channel_dim must be an integer")
    axis = channel_dim + ndim if channel_dim < 0 else channel_dim
    if axis < 0 or axis >= ndim:
        raise ValueError("channel_dim is out of range for the given tensor")
    return axis


def _denoise(
    model: torch.nn.Module,
    value: torch.Tensor,
    sigma: float | torch.Tensor | None,
    kwargs: dict[str, Any],
) -> torch.Tensor:
    """Call a model, omitting a noise level it was not given."""
    if sigma is None and not kwargs:
        return model(value)
    return model(value, sigma, **kwargs)


class ComplexDenoiser(_Denoiser):
    """Put a complex image through a real-valued denoiser.

    The real and imaginary parts become two adjacent channels of a single
    call, so the network sees them together. That is the convention the
    learned MRI reconstructions are built on, and it is not the same model as
    denoising each part on its own: a phase-sensitive network needs both
    halves of a voxel at once.

    Real input is passed through untouched, which is what lets the same
    wrapper sit in front of a magnitude reconstruction.

    Parameters
    ----------
    model
        Denoiser accepting a real tensor with a channel axis.
    channel_dim
        Channel dimension of the input and output tensors.

    Examples
    --------
    A two-channel network denoises a complex image, and the result comes back
    complex.

    >>> import torch
    >>> import pulserver.recon as recon
    >>> network = torch.nn.Conv2d(2, 2, 3, padding=1)
    >>> image = torch.randn(1, 1, 16, 16, dtype=torch.complex64)
    >>> denoised = recon.ComplexDenoiser(network)(image)
    >>> denoised.shape, denoised.is_complex()
    (torch.Size([1, 1, 16, 16]), True)
    """

    def __init__(self, model: torch.nn.Module, *, channel_dim: int = 1) -> None:
        super().__init__()
        self.model = model
        self.channel_dim = int(channel_dim)

    def forward(self, value: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
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


#: The name the reconstruction algorithms reach for when they wrap a denoiser
#: themselves.
_ComplexAdapter = ComplexDenoiser


class NoiseConditioned(_Denoiser):
    """Give a network the noise level to remove as an input channel.

    A plug-and-play reconstruction calls its denoiser at a schedule of noise
    levels, and the regularization parameter is that level. A network that
    ignores it -- :class:`deepinv.models.DnCNN` says so in as many words --
    leaves the parameter inert, and the reconstruction returns the same image
    whatever it is set to. This wrapper is the arrangement
    :class:`~deepinv.models.DRUNet` and FFDNet use: the level is broadcast to
    a constant channel, concatenated to the input, and the network is trained
    across the range it will be called over.

    The wrapped network must accept the extra channel. A residual network
    must also return as many channels as it takes, and the trailing ones are
    then discarded.

    Parameters
    ----------
    model
        Network taking the data channels plus one, along ``channel_dim``.
    channel_dim
        Channel dimension of the input and output tensors.

    Examples
    --------
    The wrapped network takes one channel more than the data carries, and the
    level reaches it.

    >>> import torch
    >>> import pulserver.recon as recon
    >>> conditioned = recon.NoiseConditioned(torch.nn.Conv2d(3, 2, 3, padding=1))
    >>> image = torch.randn(1, 2, 16, 16)
    >>> conditioned(image, 0.05).shape
    torch.Size([1, 2, 16, 16])
    >>> torch.allclose(conditioned(image, 0.01), conditioned(image, 0.20))
    False
    """

    def __init__(self, model: torch.nn.Module, *, channel_dim: int = 1) -> None:
        super().__init__()
        self.model = model
        self.channel_dim = int(channel_dim)

    def forward(
        self,
        value: torch.Tensor,
        sigma: float | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Denoise ``value`` at noise level ``sigma``."""
        if not isinstance(value, torch.Tensor):
            raise TypeError("x must be a Torch tensor")
        axis = _channel_axis(value.ndim, self.channel_dim)
        level = torch.as_tensor(
            0.0 if sigma is None else sigma,
            dtype=value.real.dtype,
            device=value.device,
        )
        shape = list(value.shape)
        shape[axis] = 1
        level = level.reshape(-1, *([1] * (value.ndim - 1))).expand(shape)
        result = self.model(torch.cat([value, level], dim=axis), **kwargs)
        if not isinstance(result, torch.Tensor):
            raise TypeError("a noise-conditioned model must return a Torch tensor")
        if result.shape[axis] == value.shape[axis]:
            return result
        return result.narrow(axis, 0, value.shape[axis])


class Checkpointed(_Denoiser):
    """Recompute a denoiser in the backward pass instead of storing it.

    An unrolled reconstruction holds every block's activations until the
    backward pass reaches them, and the denoiser is nearly all of that. Under
    this wrapper the block runs again during backpropagation and only its
    input is kept, which trades one extra forward pass for the memory. It is
    what lets a three-dimensional unroll train at a depth its activations
    could not otherwise reach.

    Outside a gradient-tracking context the wrapper is a passthrough.

    Parameters
    ----------
    model
        Denoiser to recompute.

    Examples
    --------
    The wrapper is transparent: the value is the model's own.

    >>> import torch
    >>> import pulserver.recon as recon
    >>> network = torch.nn.Conv2d(2, 2, 3, padding=1)
    >>> image = torch.randn(1, 2, 16, 16)
    >>> checkpointed = recon.Checkpointed(network)
    >>> torch.allclose(checkpointed(image), network(image))
    True

    What it costs and what it buys, on a ten-step unroll. Autograd's
    saved-tensor hooks count exactly what the backward pass is holding, so the
    measurement is the quantity that decides whether a training step fits --
    not a proxy for it, and not tied to a particular device.

    >>> import deepinv
    >>> from contextlib import contextmanager
    >>> @contextmanager
    ... def retained():
    ...     "Bytes autograd keeps alive for the backward pass."
    ...     total = [0]
    ...     def pack(tensor):
    ...         total[0] += tensor.numel() * tensor.element_size()
    ...         return tensor
    ...     with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
    ...         yield total
    >>> def unroll(recompute):
    ...     _ = torch.manual_seed(0)
    ...     prior = recon.NoiseConditioned(deepinv.models.DnCNN(
    ...         in_channels=3, out_channels=3, depth=8, nf=32, pretrained=None))
    ...     return deepinv.optim.PGD(
    ...         data_fidelity=recon.NormalEquationL2(),
    ...         prior=deepinv.optim.PnP(
    ...             recon.Checkpointed(prior) if recompute else prior),
    ...         params_algo={"stepsize": 1.0, "g_param": 0.02},
    ...         max_iter=10, unfold=True, custom_init=recon.ScaledAdjoint())
    >>> maps = torch.ones(1, 4, 64, 64, dtype=torch.complex64) / 2.0
    >>> mask = torch.zeros(1, 1, 64, 64)
    >>> mask[..., ::3, :] = 1.0
    >>> physics = recon.Cartesian2D(mask, maps, viewed_as_real=True)
    >>> measured = physics.A(torch.randn(1, 2, 64, 64))
    >>> held, gradients = {}, {}
    >>> for recompute in (False, True):
    ...     model = unroll(recompute)
    ...     with retained() as total:
    ...         model(measured, physics).pow(2).sum().backward()
    ...     held[recompute] = total[0]
    ...     gradients[recompute] = torch.cat(
    ...         [p.grad.flatten() for p in model.parameters() if p.grad is not None])

    An order of magnitude less is held, and the gradient is the same one.

    >>> held[False] > 10 * held[True]
    True
    >>> torch.allclose(gradients[False], gradients[True], atol=1e-5)
    True
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        value: torch.Tensor,
        sigma: float | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run the model, recomputing it in the backward pass when training."""
        if not torch.is_grad_enabled():
            return _denoise(self.model, value, sigma, kwargs)
        return torch.utils.checkpoint.checkpoint(
            lambda tensor: _denoise(self.model, tensor, sigma, kwargs),
            value,
            use_reentrant=False,
        )


def _batch_norm(value: torch.Tensor) -> torch.Tensor:
    flattened = value.reshape(value.shape[0], -1)
    return torch.linalg.vector_norm(flattened, dim=1)


class ScaledAdjoint(torch.nn.Module):
    """Initialize an unroll with an optionally norm-aligned adjoint.

    Parameters
    ----------
    normalize
        Scale the adjoint so its re-encoding norm matches the measurements.
    epsilon
        Minimum denominator used in the per-batch norm ratio.

    Examples
    --------
    The adjoint, rescaled so its output is on the order of the image -- the
    usual first step of an unrolled network. The call signature is
    DeepInverse's ``custom_init``, so it is passed straight to an optimizer.

    >>> import torch
    >>> import pulserver.recon as recon
    >>> physics = recon.Cartesian2D(
    ...     torch.ones(1, 1, 16, 16),
    ...     torch.ones(1, 2, 16, 16, dtype=torch.complex64) / 2 ** 0.5,
    ... )
    >>> measured = physics.A(torch.randn(1, 16, 16, dtype=torch.complex64))
    >>> recon.ScaledAdjoint()(measured, physics).shape
    torch.Size([1, 16, 16])
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
    ) -> torch.Tensor:
        """Return the initial image estimate."""
        rhs = physics.A_adjoint(y)
        if not self.normalize:
            return rhs
        encoded = physics.A(rhs)
        numerator = _batch_norm(y)
        denominator = _batch_norm(encoded).clamp_min(self.epsilon)
        scale = numerator / denominator
        scale = scale.reshape(scale.shape + (1,) * (rhs.ndim - scale.ndim))
        return rhs * scale


class NormalEquationL2(_L2):
    r"""Least-squares data fidelity evaluated through the normal operator.

    The gradient of :math:`\tfrac{1}{2\sigma^2}\|Ax - y\|^2` is
    :math:`A^{H}Ax - A^{H}y`. DeepInverse forms it as written, one encode
    followed by one adjoint at every step. This term asks the physics for
    :math:`A^{H}A` instead and reuses :math:`A^{H}y`, which it computes once
    per measurement.

    What that is worth depends on the scan. A non-Cartesian physics answers
    :meth:`A_adjoint_A` with its Toeplitz kernel when one is built, and a
    densely sampled readout is then cheaper than the transform pair it
    replaces; a short, heavily undersampled readout is not, and the physics
    is what decides -- ``normal_mode`` reports which form it is using.

    Parameters
    ----------
    sigma
        Noise standard deviation, as in :class:`deepinv.optim.data_fidelity.L2`.

    Examples
    --------
    The gradient is the one DeepInverse computes, by a different route.

    >>> import torch
    >>> import deepinv
    >>> import pulserver.recon as recon
    >>> physics = recon.Cartesian2D(
    ...     torch.ones(1, 1, 16, 16),
    ...     torch.ones(1, 2, 16, 16, dtype=torch.complex64) / 2 ** 0.5,
    ... )
    >>> truth = torch.randn(1, 16, 16, dtype=torch.complex64)
    >>> measured = physics.A(truth)
    >>> estimate = truth + 0.1 * torch.randn_like(truth)
    >>> ours = recon.NormalEquationL2().grad(estimate, measured, physics)
    >>> theirs = deepinv.optim.data_fidelity.L2().grad(estimate, measured, physics)
    >>> torch.allclose(ours, theirs, atol=1e-5)
    True
    """

    def __init__(self, sigma: float = 1.0) -> None:
        super().__init__(sigma=sigma)
        self._cached: tuple[Any, Any, torch.Tensor] | None = None

    def grad(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        physics: Any,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Return the least-squares gradient at ``x``."""
        normal = getattr(physics, "A_adjoint_A", None)
        if normal is None:
            return super().grad(x, y, physics, *args, **kwargs)
        return (normal(x) - self.rhs(y, physics)) * self.norm

    def rhs(self, y: torch.Tensor, physics: Any) -> torch.Tensor:
        """Return :math:`A^{H}y`, recomputing it only for a new measurement.

        The measurement and the operator are held weakly, so a training loop
        that discards a batch cannot be served that batch's right-hand side.
        """
        if self._cached is not None:
            measurement, operator, value = self._cached
            if measurement() is y and operator() is physics:
                return value
        value = physics.A_adjoint(y)
        try:
            self._cached = (weakref.ref(y), weakref.ref(physics), value)
        except TypeError:
            self._cached = None
        return value


class StepwiseUnroll:
    r"""Drive an unfolded DeepInverse optimizer one step at a time.

    A deep unroll does not always fit in one backward pass, and its loss does
    not always belong only at the end. This runs the same algorithm the
    optimizer would run itself -- the same iterates, to the bit -- while
    handing back each step, so the caller can supervise a step directly or cut
    the graph between them.

    Two schedules it exists for. *Greedy* training attaches a loss to every
    step and backpropagates it there, which trains a deep unroll one block at
    a time. *Truncated backpropagation* keeps the forward pass whole but
    limits how far back credit is assigned, by detaching the state every few
    steps.

    Neither is a way to save memory on its own: :class:`~pulserver.recon.Checkpointed`
    around the denoiser costs less than either and changes no gradient. Reach
    for these when the training signal is what needs to change.

    Only an optimizer built with ``unfold=True`` has parameters to train.
    Stepping one built without it still works and is how an iteration is
    looked at rather than trained.

    Parameters
    ----------
    model
        A DeepInverse optimizer built with ``unfold=True``.

    Examples
    --------
    Stepping reproduces the optimizer's own run exactly.

    >>> import torch
    >>> import deepinv
    >>> import pulserver.recon as recon
    >>> _ = torch.manual_seed(0)
    >>> physics = recon.Cartesian2D(
    ...     torch.ones(1, 1, 16, 16),
    ...     torch.ones(1, 2, 16, 16, dtype=torch.complex64) / 2 ** 0.5,
    ...     viewed_as_real=True,
    ... )
    >>> measured = physics.A(torch.randn(1, 2, 16, 16))
    >>> model = deepinv.optim.PGD(
    ...     data_fidelity=recon.NormalEquationL2(),
    ...     prior=deepinv.optim.PnP(deepinv.models.MedianFilter()),
    ...     params_algo={"stepsize": 1.0, "g_param": 0.01},
    ...     max_iter=4,
    ...     unfold=True,
    ... )
    >>> unroll = recon.StepwiseUnroll(model)
    >>> steps = [image for _, image in unroll.steps(measured, physics)]
    >>> len(steps), torch.allclose(steps[-1], model(measured, physics))
    (4, True)

    An unrolled reconstruction is a sequence of images, and a plain call
    returns only the last of them. Stepping is how the ones in between are
    reached -- to look at, to supervise, or to cut the graph after.

    The figure reconstructs a fastMRI brain slice sampled four-fold with a
    fully sampled centre. Both runs are DeepInverse's proximal gradient
    descent on the same Pulserver physics and differ only in the prior: total
    variation, and a denoiser trained on fastMRI slices and deployed the way a
    scanner deploys one -- :func:`~pulserver.recon.load_model` resolves it by
    name and the manifest says what to build.

    .. plot::

       import torch
       import deepinv
       import pulserver.recon as recon
       from _figures import brain, MODELS
       import matplotlib.pyplot as plt

       truth, coil_maps = brain(160, coils=4)
       axis = torch.linspace(-1, 1, 160)
       rows, columns = torch.meshgrid(axis, axis, indexing="ij")
       truth = truth * torch.exp(1j * 1.5 * (columns ** 2 - 0.5 * rows))
       image = torch.stack([truth.real, truth.imag], 1)

       mask = torch.zeros(160, 160)
       mask[::4] = 1.0
       mask[72:88] = 1.0
       physics = recon.Cartesian2D(mask[None, None], coil_maps, viewed_as_real=True)
       measured = physics.A(image)

       learned = recon.NoiseConditioned(
           recon.load_model("fastmri-denoiser", paths=[MODELS])).eval()
       priors = {
           "total variation": recon.ComplexDenoiser(recon.TV()),
           "learned": learned,
       }

       def unroll(prior):
           return recon.StepwiseUnroll(deepinv.optim.PGD(
               data_fidelity=recon.NormalEquationL2(),
               prior=deepinv.optim.PnP(prior),
               params_algo={"stepsize": 1.0, "g_param": 0.01},
               max_iter=16,
               custom_init=recon.ScaledAdjoint(),
           ))

       def decibels(value):
           error = torch.nn.functional.mse_loss(value, image)
           return float(10 * torch.log10(image.abs().max() ** 2 / error))

       def magnitude(value):
           return torch.complex(value[:, 0], value[:, 1])[0].abs()

       with torch.no_grad():
           start = recon.ScaledAdjoint()(measured, physics)
           stepped = {label: list(unroll(prior).steps(measured, physics))
                      for label, prior in priors.items()}

       figure = plt.figure(figsize=(11.5, 2.8), constrained_layout=True)
       panels = figure.subplots(1, 5)
       shown = [("object", truth[0].abs(), None),
                ("zero filled", magnitude(start), decibels(start))]
       shown += [(label, magnitude(steps[-1][1]), decibels(steps[-1][1]))
                 for label, steps in stepped.items()]
       for axes, (label, value, score) in zip(panels[:4], shown):
           axes.imshow(value, cmap="gray", vmin=0, vmax=float(truth.abs().max()))
           axes.set_title(label if score is None else f"{label}\n{score:.1f} dB",
                          fontsize=9)
           axes.set_axis_off()
       for label, steps in stepped.items():
           panels[4].plot(range(1, len(steps) + 1),
                          [decibels(value) for _, value in steps],
                          marker="o", markersize=3, label=label)
       panels[4].axhline(decibels(start), color="0.6", linestyle=":",
                         label="zero filled")
       panels[4].set_xlabel("unrolled step")
       panels[4].set_ylabel("PSNR (dB)")
       panels[4].set_xticks([1, 4, 8, 12, 16])
       panels[4].legend(fontsize=7)
       figure.suptitle("the same sixteen steps, two priors")
    """

    def __init__(self, model: Any) -> None:
        if not hasattr(model, "fixed_point") or not hasattr(model, "get_output"):
            raise TypeError("model must be a DeepInverse optimizer")
        self.model = model

    @property
    def iterations(self) -> int:
        """Return the number of unrolled steps the optimizer runs."""
        return int(self.model.fixed_point.max_iter)

    def steps(
        self,
        y: torch.Tensor,
        physics: Any,
        init: Any = None,
        *,
        iterations: int | None = None,
        detach_every: int | None = None,
        **kwargs: Any,
    ) -> Iterator[tuple[int, torch.Tensor]]:
        """Yield ``(step, image)`` for each unrolled step.

        Parameters
        ----------
        y
            Measurement.
        physics
            Forward operator.
        init
            Initial estimate, as accepted by the optimizer.
        iterations
            Number of steps to run. ``None`` runs the optimizer's own count.
        detach_every
            Cut the graph after every this many steps. ``None`` keeps the
            whole unroll differentiable end to end.
        **kwargs
            Forwarded to the optimizer's iterator.

        Yields
        ------
        tuple[int, torch.Tensor]
            The one-based step number and the image estimate after it. The
            graph is cumulative unless ``detach_every`` cuts it, so greedy
            training -- a backward pass per step -- pairs with
            ``detach_every=1``; without it only the last estimate can be
            backpropagated.
        """
        count = self.iterations if iterations is None else int(iterations)
        if count < 1 or count > self.iterations:
            raise ValueError("iterations must lie between 1 and the unrolled depth")
        if detach_every is not None and (
            not isinstance(detach_every, int)
            or isinstance(detach_every, bool)
            or detach_every < 1
        ):
            raise ValueError("detach_every must be a positive integer or None")
        loop = self.model.fixed_point
        state = loop.init_iterate_fn(y, physics, init, cost_fn=loop.iterator.cost_fn)
        for step in range(count):
            state = loop.single_iteration(state, step, y, physics, **kwargs)
            yield step + 1, self.model.get_output(state)
            if detach_every is not None and (step + 1) % detach_every == 0:
                state = _detached_state(state)


def _detached_state(state: Any) -> Any:
    if isinstance(state, torch.Tensor):
        return state.detach()
    if isinstance(state, tuple):
        return tuple(_detached_state(item) for item in state)
    if isinstance(state, list):
        return [_detached_state(item) for item in state]
    if isinstance(state, dict):
        return {key: _detached_state(value) for key, value in state.items()}
    return state
