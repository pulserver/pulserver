"""The normal operator as a precomputed kernel.

``A^H A`` is a convolution, so it is a multiplication on a doubled grid: pad,
transform, multiply by the transfer the scan grids to, transform back, crop.
The transfer is built by an adjoint transform of the sampling weights, and
cut to the locations the samples reached."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from contextlib import ExitStack, contextmanager, suppress
from importlib import import_module
from math import prod
from types import MethodType, SimpleNamespace
from typing import Any


from .._toeplitz import (
    CompactToeplitzKernel,
    _device_is_full,
    as_torch,
)
from .._views import image_as_cpx as _image_as_cpx
from .._views import image_as_real as _image_as_real

from ._base import MRIPhysics, _init_from
from ._common import (
    _base_fourier_operator,
    _require_mrinufft,
    _support_locations,
    _toeplitz_options,
)
from ._frames import _LazyFramePhysics, _mrinufft_norm_factor


_PSF_OPERATOR_SLOT: dict[tuple[Any, ...], Any] = {}

# The transfer is held as complex64 and then cut to the support the scan
# reached, so gridding it tighter than this buys nothing that survives either
# step and costs several times the build.
_PSF_TOLERANCE = 1e-4

# The tolerance to ask the narrow spreading grid for. Interpolation width is
# set by the pair, not by either alone: the narrow grid reaches the same
# five-wide kernel here that the wide grid reaches at _PSF_TOLERANCE, so the
# fallback costs a factor on the transfer's accuracy rather than on the
# spreading, which is what it would cost at the tighter tolerance.
_NARROW_PSF_TOLERANCE = 1e-3
_NARROW_PSF_UPSAMPLING = 1.25


def _psf_operator(
    samples: Any,
    backend: str,
    spatial_shape: tuple[int, ...],
) -> Any:
    """A NUFFT on the doubled grid, for gridding the transfer onto it.

    One plan is kept per (backend, grid, sample count) and retargeted at each
    trajectory it is asked for. Planning a NUFFT is the expensive part of
    building a kernel, and holding a second plan on the doubled grid is what
    makes a build run out of device memory.
    """
    mrinufft = _require_mrinufft()
    shape = tuple(int(size) for size in spatial_shape)
    key = (backend, shape, int(samples.shape[0]))
    operator = _PSF_OPERATOR_SLOT.get(key)
    if operator is not None:
        operator.update_samples(samples)
        return operator
    build = mrinufft.get_operator(backend)
    _yield_cached_device_memory(getattr(samples, "device", None))
    settings: dict[str, Any] = _psf_settings(shape, samples)
    try:
        operator = build(
            samples=samples,
            shape=shape,
            density=None,
            n_coils=1,
            squeeze_dims=False,
            **settings,
        )
    except TypeError:
        operator = build(
            samples=samples,
            shape=shape,
            density=None,
            n_coils=1,
            squeeze_dims=False,
        )
    # One slot: a plan on the doubled grid is the largest device allocation a
    # build makes, and holding a second one is what makes a build run out.
    _PSF_OPERATOR_SLOT.clear()
    _PSF_OPERATOR_SLOT[key] = operator
    return operator


def _yield_cached_device_memory(device: Any) -> None:
    """Hand the allocator's spare blocks back to the driver.

    A NUFFT plan is allocated outside Torch, so blocks Torch is holding for
    reuse are neither available to it nor counted as free -- and Torch does
    not release them when another library runs out. What a build measures and
    what it can take are both only true once these are returned.
    """
    torch = import_module("torch")
    if "cuda" not in str(device):
        return
    with suppress(RuntimeError):
        torch.cuda.empty_cache()


def _psf_settings(shape: tuple[int, ...], samples: Any) -> dict[str, Any]:
    """What to plan the gridding NUFFT with, given what the device has room for.

    A NUFFT spreads onto a grid of its own on the way to the one it answers
    on; that grid is internal and does not touch the transfer, so it is chosen
    for what it costs. The wide one is the default. On the doubled grid a
    kernel is built on it is eight times the transfer, so at these sizes it
    stops fitting, and the narrow one is asked for a looser tolerance -- which
    keeps its interpolation kernel the width the wide one has, and spends the
    difference on the transfer rather than on every point spread onto it.
    """
    torch = import_module("torch")
    narrow = {"eps": _NARROW_PSF_TOLERANCE, "upsampfac": _NARROW_PSF_UPSAMPLING}
    # NumPy answers `device` with a plain string, Torch with an object.
    device = getattr(samples, "device", None)
    if "cuda" not in str(device):
        return {"eps": _PSF_TOLERANCE}
    free, _ = torch.cuda.mem_get_info(device)
    spreading = 8 * (2 ** len(shape)) * prod(shape)
    wide = spreading + 8 * prod(shape) + 8 * int(samples.shape[0])
    return {"eps": _PSF_TOLERANCE} if wide < 0.6 * free else narrow


def _within_psf_plans(build: Any) -> Any:
    """Release the gridding plan a builder makes when its build ends."""

    @wraps(build)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with _psf_plans():
            return build(*args, **kwargs)

    return wrapper


@contextmanager
def _psf_plans() -> Any:
    """Hold one gridding plan for the length of a build, then release it.

    A plan on the doubled grid is the largest device allocation a build makes
    -- larger than the kernel it produces -- and the solve that follows needs
    that memory for its own transforms.
    """
    try:
        yield
    finally:
        _PSF_OPERATOR_SLOT.clear()
        with suppress(ImportError, AttributeError):
            torch = import_module("torch")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _compute_toeplitz_transfer(
    native_operator: Any,
    weights: Any | None = None,
    *,
    complex_weights: bool = False,
) -> Any:
    """The transfer a Toeplitz normal operator multiplies by.

    The point-spread function is the adjoint of the sample weights taken on a
    grid twice the image in every dimension -- ones for a plain normal, the
    density for a compensated one, a basis product for a subspace frame or an
    off-resonance segment -- and the transfer is its transform.

    Gridding is what puts the weight where the trajectory is. The adjoint
    interpolates each sample onto the doubled grid with the backend's own
    kernel, so the transfer holds weight where the scan reached and in the rim
    that interpolation spreads into, and nowhere else. That is the same
    operator the forward NUFFT applies, so the normal is the Gram of the
    transform actually being inverted.
    """
    del complex_weights
    torch = import_module("torch")
    base = _base_fourier_operator(native_operator)
    image_shape = tuple(int(size) for size in base.shape)
    spatial_shape = tuple(2 * size for size in image_shape)
    operator = _psf_operator(
        base.samples,
        getattr(base, "backend", "finufft"),
        spatial_shape,
    )

    if weights is None:
        # The plain normal is weighted by whatever the operator itself carries:
        # its adjoint applies the density once, so the Gram does too.
        weights = getattr(base, "density", None)
    if weights is None:
        values = torch.ones(
            operator.n_samples,
            dtype=torch.complex64,
            device=as_torch(base.samples).device,
        )
    else:
        values = as_torch(weights).reshape(-1).to(torch.complex64)

    # Backends differ on whether they take a bare sample vector, so the
    # batch and coil axes are stated and dropped again.
    psf = as_torch(operator.adj_op(values.reshape(1, 1, -1))).reshape(spatial_shape)
    axes = tuple(range(len(spatial_shape)))
    # ``adj_op`` answers a centred image and divides by the doubled grid's own
    # normalization, while the normal operator this stands in for carries the
    # image grid's twice -- once in the forward and once in the adjoint.
    scale = float(operator.norm_factor) / float(base.norm_factor) ** 2
    return torch.fft.fftn(torch.fft.ifftshift(psf, dim=axes), dim=axes) * scale


def _sense_maps(native_operator: Any, reference: Any) -> Any:
    """Return sensitivity maps as a Torch tensor, on whatever device holds them.

    A normal application reads one coil at a time, so maps the caller left on
    the host are staged coil by coil rather than moved whole -- the difference
    is the whole bank against one map of it.
    """
    torch = import_module("torch")
    base = _base_fourier_operator(native_operator)
    maps = getattr(base, "smaps", None)
    if maps is None:
        return torch.ones(
            (1, *base.shape),
            dtype=reference.dtype,
            device=reference.device,
        )
    maps = as_torch(maps).to(reference.dtype)
    spatial_ndim = len(base.shape)
    if maps.ndim == spatial_ndim:
        return maps[None]
    if maps.ndim in {spatial_ndim + 1, spatial_ndim + 2}:
        return maps
    raise ValueError(
        "sensitivity maps must have shape (coils, *image_shape) or "
        "(batch, coils, *image_shape)"
    )


def _frame_coil_view(frame: MRIPhysics | _LazyFramePhysics) -> Any:
    """What a kernel needs of a frame: its coils and its grid, never a plan."""
    view = getattr(frame, "coil_view", None)
    return frame.native_operator if view is None else view


def _coils_split_across_devices(
    kernel: CompactToeplitzKernel,
    image: Any,
    maps: Any,
    streaming: Any,
    *,
    batched_maps: bool,
    n_coils: int,
) -> Any:
    """Sum a normal application over coils divided between CUDA devices.

    Coils are independent until the sum that ends them, so each device is given
    a share of them, its own copy of the transfer and its own copy of the
    image, and returns the part of the sum it computed.

    This has not been run on a machine with more than one GPU. What it assumes
    of a second device is what ``for_device`` and ``_apply_sense_toeplitz``
    already assume of the first.
    """
    devices = streaming.torch_devices[: min(streaming.device_count, n_coils)]
    edges = [(index * n_coils) // len(devices) for index in range(len(devices) + 1)]

    def share(position: int) -> Any:
        device = devices[position]
        start, stop = edges[position], edges[position + 1]
        coils = (maps[:, start:stop] if batched_maps else maps[start:stop]).to(device)
        held = SimpleNamespace(
            shape=kernel.image_shape,
            smaps=coils,
            uses_sense=True,
        )
        return _apply_sense_toeplitz(
            kernel.for_device(device),
            image.to(device),
            held,
            coil_batch_size=1,
        )

    with ThreadPoolExecutor(max_workers=len(devices)) as workers:
        parts = list(workers.map(share, range(len(devices))))
    total = parts[0].to(image.device)
    for part in parts[1:]:
        total = total + part.to(image.device)
    return total


def _apply_sense_toeplitz(
    kernel: CompactToeplitzKernel,
    image: Any,
    native_operator: Any,
    *,
    right_factors: Any | None = None,
    left_factors: Any | None = None,
    coil_batch_size: int = 1,
    streaming: Any | None = None,
) -> Any:
    """Apply a compact transfer between optional spatial factor banks."""
    torch = import_module("torch")
    if streaming is not None and streaming.device_count > 1:
        # Coils are independent until their final SENSE reduction.  Group at
        # least one coil per device so even a single-image reconstruction can
        # fan its Toeplitz work across a multi-GPU recon host.
        coil_batch_size = max(coil_batch_size, streaming.device_count)
    maps = _sense_maps(native_operator, image)
    # An image is (batch, *spatial) and unbatched maps are (coils, *spatial),
    # so the two carry the same rank; only the maps' own rank separates them.
    batched_maps = maps.ndim == len(kernel.image_shape) + 2
    if batched_maps:
        if maps.shape[0] == 1:
            maps = maps.expand(image.shape[0], *maps.shape[1:])
        elif maps.shape[0] != image.shape[0]:
            raise ValueError(
                "batched sensitivity maps must have one entry per image batch"
            )
        n_coils = maps.shape[1]
    else:
        n_coils = maps.shape[0]
    if (
        streaming is not None
        and streaming.device_count > 1
        and n_coils > 1
        and left_factors is None
        and right_factors is None
    ):
        return _coils_split_across_devices(
            kernel,
            image,
            maps,
            streaming,
            batched_maps=batched_maps,
            n_coils=n_coils,
        )
    result_rank = 1 if left_factors is not None else kernel.rank
    result = torch.zeros(
        (image.shape[0], result_rank, *kernel.image_shape),
        dtype=image.dtype,
        device=image.device,
    )
    if right_factors is not None:
        right_factors = as_torch(right_factors, device=image.device).to(image.dtype)
        right_factors = right_factors.reshape(kernel.rank, *kernel.image_shape)
    if left_factors is not None:
        left_factors = as_torch(left_factors, device=image.device).to(image.dtype)
        left_factors = left_factors.reshape(kernel.rank, *kernel.image_shape)

    staged_coils = None
    if (
        streaming is not None
        and image.device.type == "cpu"
        and streaming.pin_memory
        and coil_batch_size > 1
    ):
        staged_coils = torch.empty(
            (
                image.shape[0],
                min(coil_batch_size, n_coils),
                image.shape[1],
                *kernel.image_shape,
            ),
            dtype=image.dtype,
            device="cpu",
            pin_memory=True,
        )

    for start in range(0, n_coils, coil_batch_size):
        if batched_maps:
            coil_maps = maps[:, start : start + coil_batch_size].to(image.device)
            left = image[:, None]
            right = coil_maps[:, :, None]
        else:
            coil_maps = maps[start : start + coil_batch_size].to(image.device)
            left = image[:, None]
            right = coil_maps[None, :, None]
        coil_count = coil_maps.shape[1] if batched_maps else coil_maps.shape[0]
        resident_sense = (
            streaming is None
            and image.device.type == "cuda"
            and coil_count == 1
            and right_factors is None
            and left_factors is None
            and kernel._select_cuda_mode(image) == "resident"
        )
        if resident_sense:
            maps_batch = (
                coil_maps[:, 0]
                if batched_maps
                else coil_maps[0][None].expand(
                    image.shape[0],
                    *kernel.image_shape,
                )
            )
            factor = maps_batch[:, None].expand_as(image)
            try:
                kernel._apply_cuda_resident(
                    image,
                    right_factor=factor,
                    left_factor=factor.conj(),
                    output=result,
                )
            except RuntimeError as error:
                if kernel.cuda_mode == "resident" or not _device_is_full(error):
                    raise
                kernel._resident_refused()
            else:
                kernel._last_cuda_mode = "resident"
                continue
        fused_streaming = (
            streaming is not None and image.device.type == "cpu" and coil_count == 1
        )
        if fused_streaming:
            maps_batch = (
                coil_maps[:, 0]
                if batched_maps
                else coil_maps[0][None].expand(
                    image.shape[0],
                    *kernel.image_shape,
                )
            )
            fused_right = maps_batch[:, None].expand(
                image.shape[0],
                kernel.rank,
                *kernel.image_shape,
            )
            if right_factors is not None:
                fused_right = fused_right * right_factors[None]
            fused_left = maps_batch.conj()[:, None].expand(
                image.shape[0],
                kernel.rank,
                *kernel.image_shape,
            )
            if left_factors is not None:
                fused_left = fused_left * left_factors.conj()[None]
            transformed = kernel.apply_streamed(
                image,
                streaming,
                right_factor=fused_right,
                left_factor=fused_left,
            )
        elif staged_coils is None:
            coil_images = left * right
        else:
            coil_images = staged_coils[:, :coil_count]
            torch.mul(left, right, out=coil_images)
        if not fused_streaming:
            coil_images = coil_images.flatten(0, 1)
            if right_factors is not None:
                coil_images = coil_images * right_factors[None]
            transformed = (
                kernel.apply_streamed(coil_images, streaming)
                if streaming is not None and coil_images.device.type == "cpu"
                else kernel.apply(coil_images)
            )
        if left_factors is not None:
            transformed = (
                transformed.sum(dim=1, keepdim=True)
                if fused_streaming
                else (left_factors.conj()[None] * transformed).sum(
                    dim=1,
                    keepdim=True,
                )
            )
        transformed = transformed.unflatten(0, (image.shape[0], coil_count))
        if fused_streaming:
            result += transformed.sum(dim=1)
        else:
            result += (
                (transformed * coil_maps.conj()[None, :, None]).sum(dim=1)
                if not batched_maps
                else (transformed * coil_maps.conj()[:, :, None]).sum(dim=1)
            )
    kernel.settle_allocator()
    return result


def _selected_transfer(
    transfer: Any,
    indices: Any,
    *,
    streaming: Any | None,
) -> Any:
    """Select retained locations and optionally move them to host storage."""
    torch = import_module("torch")
    transfer = as_torch(transfer).flatten()
    selected = torch.index_select(
        transfer,
        0,
        indices.to(transfer.device, dtype=torch.int64),
    )
    return selected.to("cpu") if streaming is not None else selected


@_within_psf_plans
def _build_scalar_toeplitz(
    native_operator: Any,
    options: dict[str, Any],
    streaming: Any | None = None,
) -> CompactToeplitzKernel:
    """Build Pulserver's compact rank-one NUFFT transfer."""
    base = _base_fourier_operator(native_operator)
    image_shape = tuple(int(size) for size in base.shape)
    spatial_shape = tuple(2 * size for size in image_shape)
    transfer = as_torch(_compute_toeplitz_transfer(base)).flatten()
    indices = _support_locations(
        getattr(base, "samples", None),
        spatial_shape,
        "cpu" if streaming is not None else transfer.device,
        options["compress"],
    )
    values = _selected_transfer(transfer, indices, streaming=streaming).real[None]
    kernel = CompactToeplitzKernel(
        values,
        indices,
        spatial_shape,
        1,
        image_shape=image_shape,
        chunk_size=options["chunk_size"],
        cuda_mode=options["cuda_mode"],
        cuda_max_device_fraction=options["cuda_max_device_fraction"],
        cuda_transfer_precision=options["cuda_transfer_precision"],
    )
    return kernel


def _configure_base_toeplitz(
    operator: Any,
    native_operator: Any,
    *,
    enabled: bool,
    best_effort: bool = False,
    options: dict[str, Any],
) -> Any:
    """Install Pulserver's lazy scalar normal on a native NUFFT adapter.

    The kernel is built on the first normal-operator call, so an operator that
    only ever encodes or decodes pays nothing for carrying one. ``best_effort``
    is what ``toeplitz="auto"`` asks for: a shape the backend cannot embed
    circulantly reverts to the exact normal instead of raising.
    """
    operator.use_toeplitz = enabled
    operator.toeplitz_best_effort = best_effort
    operator.toeplitz_kernel = None
    operator._toeplitz_options = dict(options)
    operator.streaming_policy = None
    operator.streaming_methods = {"A_adjoint_A"}

    def enable_toeplitz(self: Any, new_options: dict[str, Any]) -> None:
        self.use_toeplitz = True
        self.toeplitz_best_effort = False
        self._toeplitz_options = dict(new_options)
        self.toeplitz_kernel = None

    def enable_streaming(self: Any, policy: Any) -> None:
        self.streaming_policy = policy
        self.toeplitz_kernel = None

    def scalar_normal(self: Any, x: Any, **kwargs: Any) -> Any:
        del kwargs
        if not self.use_toeplitz:
            return self.A_adjoint(self.A(x))
        image = _image_as_cpx(x) if self.viewed_as_real else x
        if self.toeplitz_kernel is None:
            try:
                self.toeplitz_kernel = _build_scalar_toeplitz(
                    native_operator,
                    self._toeplitz_options,
                    self.streaming_policy,
                )
            except (ValueError, NotImplementedError):
                if not self.toeplitz_best_effort:
                    raise
                self.use_toeplitz = False
                return self.A_adjoint(self.A(x))
        base = _base_fourier_operator(native_operator)
        if getattr(base, "uses_sense", False):
            result = _apply_sense_toeplitz(
                self.toeplitz_kernel,
                image,
                native_operator,
                coil_batch_size=self._toeplitz_options["coil_batch_size"],
                streaming=self.streaming_policy,
            )
        else:
            batch, channels, *spatial = image.shape
            flattened = image.reshape(batch * channels, 1, *spatial)
            result = (
                self.toeplitz_kernel.apply_streamed(
                    flattened,
                    self.streaming_policy,
                )
                if self.streaming_policy is not None and flattened.device.type == "cpu"
                else self.toeplitz_kernel.apply(flattened)
            ).reshape(batch, channels, *spatial)
        self.toeplitz_kernel.settle_allocator()
        return _image_as_real(result) if self.viewed_as_real else result

    operator.enable_toeplitz = MethodType(enable_toeplitz, operator)
    operator.enable_streaming = MethodType(enable_streaming, operator)
    operator.A_adjoint_A = MethodType(scalar_normal, operator)
    return operator


def _subspace_frame_blocks(
    frame_physics: Sequence[MRIPhysics | _LazyFramePhysics],
    basis: Any,
    rows: Any,
    columns: Any,
) -> tuple[list[tuple[Any, Any, Any]], str, tuple[int, ...]]:
    """Group the frames onto the distinct trajectories they were acquired on.

    Returns one entry per distinct trajectory -- its samples, its sample
    weights and the coefficient every upper-triangular basis pair enters it
    with, summed over the frames that share it -- alongside the backend and
    the image grid they all agree on.
    """
    torch = import_module("torch")
    order: list[Any] = []
    blocks: dict[Any, tuple[Any, Any, Any]] = {}
    backend = None
    image_shape = None
    for frame, item in enumerate(frame_physics):
        coefficients = basis[rows, frame] * basis[columns, frame].conj()
        if isinstance(item, _LazyFramePhysics):
            key: Any = ("lazy", id(item.provider), item.index)
            samples, weights = item.samples, item.density
            frame_backend, frame_shape = item.backend, item.image_shape
        else:
            native = item.native_operator
            if native is None or hasattr(native, "B"):
                raise RuntimeError(
                    "A combined subspace kernel requires undecorated frame NUFFTs."
                )
            base = _base_fourier_operator(native)
            key = id(base)
            samples, weights = base.samples, getattr(base, "density", None)
            frame_backend = getattr(base, "backend", "finufft")
            frame_shape = tuple(int(size) for size in base.shape)
        if image_shape is None:
            backend, image_shape = frame_backend, frame_shape
        elif frame_shape != image_shape:
            raise ValueError("all subspace frames must share one image shape")
        if key in blocks:
            held = blocks[key]
            blocks[key] = (held[0], held[1], held[2] + coefficients.to(held[2].device))
        else:
            order.append(key)
            blocks[key] = (samples, weights, coefficients)
    if image_shape is None:
        raise ValueError("a subspace kernel needs at least one frame")
    assert backend is not None
    del torch
    return [blocks[key] for key in order], backend, image_shape


def _centring_signs(indices: Any, spatial_shape: tuple[int, ...]) -> Any:
    """The sign that centres a transfer, at the locations it is kept over.

    Shifting a point-spread function by half the grid before transforming it
    multiplies every output by ``(-1)`` raised to the sum of its coordinates,
    so the shift never has to be performed and no copy of the doubled grid is
    made to hold it.
    """
    torch = import_module("torch")
    flat = as_torch(indices).to(torch.int64)
    parity = torch.zeros_like(flat)
    stride = 1
    for size in reversed(spatial_shape):
        parity = parity + (flat // stride) % size
        stride *= size
    return torch.where(parity % 2 == 0, 1.0, -1.0).to(torch.complex64)


def _subspace_pair_transfers(
    blocks: Sequence[tuple[Any, Any, Any]],
    backend: str,
    image_shape: tuple[int, ...],
    samples: Any,
    counts: Sequence[int],
    indices: Any,
    *,
    streaming: Any | None = None,
    keep_complex: bool = True,
) -> Any:
    """Grid one transfer per upper-triangular basis pair, over every sample.

    A pair's transfer is the adjoint of one weight per sample -- the frame's
    basis product, times whatever density the acquisition carries -- so the
    whole dynamic acquisition grids in a single pass and the count of NUFFTs
    is the size of the basis, not the length of the scan.

    Each is cut to ``indices`` as it is gridded and put down on the host in the
    form it is kept in, so a build holds one row of the device rather than the
    whole packed set twice over -- once complex and once made real.
    """
    torch = import_module("torch")
    spatial_shape = tuple(2 * size for size in image_shape)
    # One weight per sample, assembled in a single pass: a dynamic acquisition
    # has as many blocks as it has frames, and touching each of them per basis
    # pair is thousands of launches for one vector.
    weights = None
    if any(block[1] is not None for block in blocks):
        pieces = []
        for (_, density, _), count in zip(blocks, counts, strict=True):
            if density is None:
                pieces.append(torch.ones(count, device=samples.device))
                continue
            piece = as_torch(density).reshape(-1).to(samples.device)
            if piece.numel() != count:
                raise ValueError("density and samples must have the same length")
            pieces.append(piece)
        weights = torch.cat(pieces).to(torch.complex64)
    repeats = torch.tensor(counts, device=samples.device)
    coefficients = torch.stack([block[2] for block in blocks], dim=1)

    operator = _psf_operator(samples, backend, spatial_shape)
    axes = tuple(range(len(spatial_shape)))
    signs = _centring_signs(indices, spatial_shape)
    scale = float(operator.norm_factor) / _mrinufft_norm_factor(image_shape) ** 2
    n_pairs = int(blocks[0][2].numel())

    packed = None
    coefficients = coefficients.to(device=samples.device, dtype=torch.complex64)
    for pair in range(n_pairs):
        values = torch.repeat_interleave(coefficients[pair], repeats)
        if weights is not None:
            values = values * weights
        values_view = values.reshape(1, 1, -1)
        del values
        psf = as_torch(operator.adj_op(values_view)).reshape(spatial_shape)
        # Transformed in place, and the centring folded into a sign on the
        # locations kept: shifting the point-spread function by half the grid
        # is the same as alternating the sign of what comes out, and a copy of
        # the doubled grid is the largest thing a build holds after the plan.
        torch.fft.fftn(psf, dim=axes, out=psf)
        selected = _selected_transfer(psf.reshape(-1), indices, streaming=streaming)
        del psf
        row = selected * signs * scale
        del selected
        if not keep_complex:
            row = row.real
        if packed is None:
            packed = torch.empty(
                (n_pairs, row.numel()),
                dtype=row.dtype,
                device="cpu",
            )
        packed[pair].copy_(row)
        del row
    assert packed is not None
    return packed


_PLAN_SETTINGS = "_pulserver_plan_settings"
_LAZY_PLANS = "_pulserver_lazy_plans"


def _remember_plan_settings(native_operator: Any, settings: dict[str, Any]) -> None:
    """Keep what an operator was planned with, so it can be planned again."""
    base = _base_fourier_operator(native_operator)
    with suppress(AttributeError):
        setattr(base, _PLAN_SETTINGS, dict(settings))


def _plans_made_when_asked(raw: Any, settings: dict[str, Any], samples: Any) -> None:
    """Let each transform of ``raw`` plan itself the first time it runs.

    Points are held rather than set while a plan is absent, so an operator can
    be aimed at new samples without a plan to aim, and arrives at the transform
    pointed where its last caller asked.
    """
    state = getattr(raw, _LAZY_PLANS, None)
    if state is not None:
        state["settings"] = dict(settings)
        state["samples"] = {1: samples, 2: samples}
        return

    state = {"settings": dict(settings), "samples": {1: samples, 2: samples}}
    set_pts = raw._set_pts

    def points(typ: Any, new_samples: Any) -> None:
        if typ in state["samples"] and raw.plans[typ] is None:
            state["samples"][typ] = new_samples
            return
        set_pts(typ, new_samples)

    def planned(typ: int) -> Any:
        if raw.plans[typ] is None:
            raw._make_plan(typ, **state["settings"])
            set_pts(typ, state["samples"][typ])
        return raw.plans[typ]

    def type1(coefficients: Any, grid: Any) -> Any:
        return planned(1).execute(coefficients, grid)

    def type2(grid: Any, coefficients: Any) -> Any:
        return planned(2).execute(grid, coefficients)

    raw._set_pts = points
    raw.type1 = type1
    raw.type2 = type2
    setattr(raw, _LAZY_PLANS, state)


@contextmanager
def _plans_given_up(native_operator: Any) -> Any:
    """Release an operator's NUFFT plans, and plan again when one is asked for.

    A plan is bound to the grid it answers on, so an encoding plan and the
    gridding plan of a kernel built from the same points cannot be one object.
    They can, however, take turns: the samples outlive the plan, so the
    encoding side gives its plan up for the length of a build. It takes the
    plan back on its next transform rather than at the end of the build,
    because what a kernel is for is standing in for that transform -- a solve
    that has one applies it many times over and encodes no further.
    """
    base = _base_fourier_operator(native_operator)
    raw = getattr(base, "raw_op", None)
    settings = getattr(base, _PLAN_SETTINGS, None)
    samples = getattr(base, "_samples", None)
    held = getattr(raw, "plans", None)
    reusable = (
        settings is not None
        and samples is not None
        and held is not None
        and getattr(raw, "grad_plan", None) is None
        and "cuda" in str(getattr(samples, "device", ""))
    )
    if not reusable:
        yield
        return
    # The plans go only when nothing refers to them, this frame included.
    width = len(held)
    del held
    raw.plans = [None] * width
    _yield_cached_device_memory(getattr(samples, "device", None))
    try:
        yield
    finally:
        _plans_made_when_asked(raw, settings, samples)


def _maps_parked_on_host(native_operator: Any) -> None:
    """Send an operator's sensitivities to the host for a kernel's lifetime.

    Sensitivities have the same lifetime on a device as the plans beside them:
    a normal operator that is a kernel reads its own copy a coil at a time, and
    the operator they belong to encodes once, if at all. It stages them back a
    coil at a time when it is next asked to.
    """
    base = _base_fourier_operator(native_operator)
    maps = getattr(base, "_smaps", None)
    device = getattr(maps, "device", None)
    # A host array carries a device too, spelled as a bare string.
    if maps is None or getattr(device, "type", "cpu") == "cpu":
        return
    with suppress(AttributeError, RuntimeError):
        base._smaps = maps.to("cpu")


@contextmanager
def _frames_release_their_plans(
    frame_physics: Sequence[MRIPhysics | _LazyFramePhysics],
) -> Any:
    """Give a build the device to itself.

    Building a kernel needs the samples and the basis, not the operator that
    encodes with them -- and that operator holds a plan the size of the one the
    gridding is about to ask for. Two of them on a card sized for one is what
    turns a transform into several. Frames plan again the next time they are
    asked to encode, which costs one plan against the several a build spends
    starved of memory.
    """
    providers = {
        id(item.provider): item.provider
        for item in frame_physics
        if isinstance(item, _LazyFramePhysics)
    }
    for provider in providers.values():
        provider.shared = None
        provider.target = None
        provider.cache.clear()
        _maps_parked_on_host(provider.physics.native_operator)
    if providers:
        with suppress(ImportError, AttributeError):
            torch = import_module("torch")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    with ExitStack() as encoding:
        for provider in providers.values():
            encoding.enter_context(_plans_given_up(provider.physics.native_operator))
        yield


@_within_psf_plans
def _build_subspace_toeplitz(
    frame_physics: Sequence[MRIPhysics | _LazyFramePhysics],
    basis: Any,
    options: dict[str, Any],
    streaming: Any | None = None,
) -> CompactToeplitzKernel:
    """Grid one transfer per basis pair and pack them as coefficient matrices."""
    torch = import_module("torch")
    basis = torch.as_tensor(basis)
    rank, _ = basis.shape
    rows, columns = torch.triu_indices(rank, rank, device=basis.device)

    blocks, backend, image_shape = _subspace_frame_blocks(
        frame_physics,
        basis,
        rows,
        columns,
    )
    stack = ExitStack()
    stack.enter_context(_frames_release_their_plans(frame_physics))
    spatial_shape = tuple(2 * size for size in image_shape)
    # The support is the union of what the frames reached, read off their
    # trajectories and needing none of their transfers -- so it is known before
    # the first one is gridded, and each can be cut as it comes.
    ndim = len(image_shape)
    counts = [as_torch(block[0]).reshape(-1, ndim).shape[0] for block in blocks]
    samples = torch.cat([as_torch(block[0]).reshape(-1, ndim) for block in blocks])
    blocks = [(None, block[1], block[2]) for block in blocks]
    indices = _support_locations(
        samples,
        spatial_shape,
        "cpu" if streaming is not None else samples.device,
        options["compress"],
    )
    packed = _subspace_pair_transfers(
        blocks,
        backend,
        image_shape,
        samples,
        counts,
        indices,
        streaming=streaming,
        keep_complex=bool(basis.is_complex()),
    )

    stack.close()
    values = (
        packed.to(basis.dtype) if basis.is_complex() else packed.real.to(basis.dtype)
    )
    kernel = CompactToeplitzKernel(
        values,
        indices,
        spatial_shape,
        rank,
        image_shape=image_shape,
        chunk_size=options["chunk_size"],
        cuda_mode=options["cuda_mode"],
        cuda_max_device_fraction=options["cuda_max_device_fraction"],
        cuda_transfer_precision=options["cuda_transfer_precision"],
    )
    return kernel


@_within_psf_plans
def _build_cartesian_subspace_toeplitz(
    frame_physics: Sequence[MRIPhysics],
    basis: Any,
    options: dict[str, Any],
    streaming: Any | None = None,
) -> tuple[CompactToeplitzKernel, Any]:
    """Build an exact packed Cartesian subspace transfer without 2x padding."""
    torch = import_module("torch")
    basis = torch.as_tensor(basis)
    rank, n_frames = basis.shape
    operator = frame_physics[0].operator
    mask = as_torch(operator.mask)
    maps = as_torch(operator.coil_maps)
    if streaming is not None:
        mask = mask.to("cpu")
        maps = maps.to("cpu")
    spatial_ndim = getattr(frame_physics[0], "spatial_ndim", maps.ndim - 1)
    image_shape = tuple(int(size) for size in maps.shape[-spatial_ndim:])
    # DeepInverse represents Cartesian masks as (batch, real/imag, H, W).
    # The two channels are identical; retain only one before interpreting the
    # leading dimension as shared/per-frame masks.
    channel_axis = -(spatial_ndim + 1)
    if mask.ndim >= spatial_ndim + 2 and mask.shape[channel_axis] == 2:
        mask = mask.select(channel_axis, 0)
    masks = mask.reshape(-1, *image_shape)
    if masks.shape[0] == 1:
        masks = masks.expand(n_frames, *image_shape)
    elif masks.shape[0] != n_frames:
        raise ValueError(
            "Cartesian subspace mask must be shared or have one mask per frame"
        )
    masks = torch.fft.ifftshift(masks, dim=(-2, -1)).abs().square()
    # A Cartesian mask fills its own grid, so there is nothing to leave out.
    indices = _support_locations(None, image_shape, masks.device, options["compress"])
    rows, columns = torch.triu_indices(rank, rank, device=basis.device)
    packed = torch.zeros(
        (rows.numel(), indices.numel()),
        dtype=torch.promote_types(basis.dtype, masks.dtype),
        device=masks.device,
    )
    basis = basis.to(masks.device)
    for frame in range(n_frames):
        mixing = (
            basis[rows.to(masks.device), frame]
            * basis[columns.to(masks.device), frame].conj()
        )
        sampled_mask = torch.index_select(masks[frame].flatten(), 0, indices)
        packed += mixing[:, None] * sampled_mask[None]
    packed = (
        packed.to(basis.dtype) if basis.is_complex() else packed.real.to(basis.dtype)
    )
    kernel = CompactToeplitzKernel(
        packed,
        indices,
        image_shape,
        rank,
        image_shape=image_shape,
        chunk_size=options["chunk_size"],
        cuda_mode=options["cuda_mode"],
        cuda_max_device_fraction=options["cuda_max_device_fraction"],
        cuda_transfer_precision=options["cuda_transfer_precision"],
    )
    proxy = SimpleNamespace(
        shape=image_shape,
        smaps=maps,
    )
    return kernel, proxy


def _off_resonance_scalar_transfers(
    corrected_operator: Any,
    options: dict[str, Any],
    indices: Any | None = None,
    streaming: Any | None = None,
) -> tuple[Any, Any]:
    """Return upper-triangular segment transfers at their retained locations."""
    torch = import_module("torch")
    base = _base_fourier_operator(corrected_operator)
    temporal = corrected_operator.B
    rank = int(temporal.shape[1])
    image_shape = tuple(int(size) for size in base.shape)
    spatial_shape = tuple(2 * size for size in image_shape)
    rows, columns = torch.triu_indices(rank, rank)
    temporal_is_complex = as_torch(temporal).is_complex()
    density = getattr(base, "density", None)
    packed = []
    kernel_device = indices.device if indices is not None else None
    for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
        weights = temporal[:, row].conj() * temporal[:, column]
        if corrected_operator.n_shots > 1:
            try:
                weights = weights.reshape(1, -1).repeat(
                    corrected_operator.n_shots, axis=0
                )
            except TypeError:
                weights = weights.reshape(1, -1).repeat(corrected_operator.n_shots, 1)
        weights = weights.reshape(-1)
        if density is not None:
            weights = weights * density
        complex_weights = temporal_is_complex and row != column
        scalar = as_torch(
            _compute_toeplitz_transfer(
                base,
                weights,
                complex_weights=complex_weights,
            )
        ).flatten()
        if indices is None:
            kernel_device = "cpu" if streaming is not None else scalar.device
            indices = _support_locations(
                getattr(base, "samples", None),
                spatial_shape,
                kernel_device,
                options["compress"],
            )
        packed.append(
            _selected_transfer(
                scalar,
                indices,
                streaming=streaming,
            )
        )
    assert indices is not None
    values = torch.stack(packed)
    temporal_dtype = as_torch(temporal).dtype
    values = (
        values.to(temporal_dtype)
        if temporal_is_complex
        else values.real.to(temporal_dtype)
    )
    return values, indices


@_within_psf_plans
def _build_off_resonance_toeplitz(
    corrected_operator: Any,
    options: dict[str, Any],
    streaming: Any | None = None,
) -> tuple[CompactToeplitzKernel, Any]:
    """Build packed interpolation-segment cross-transfer kernels."""
    base = _base_fourier_operator(corrected_operator)
    spatial = corrected_operator.C
    rank = int(corrected_operator.B.shape[1])
    image_shape = tuple(int(size) for size in base.shape)
    spatial_shape = tuple(2 * size for size in image_shape)
    values, indices = _off_resonance_scalar_transfers(
        corrected_operator,
        options,
        streaming=streaming,
    )
    kernel = CompactToeplitzKernel(
        values,
        indices,
        spatial_shape,
        rank,
        image_shape=image_shape,
        chunk_size=options["chunk_size"],
        cuda_mode=options["cuda_mode"],
        cuda_max_device_fraction=options["cuda_max_device_fraction"],
        cuda_transfer_precision=options["cuda_transfer_precision"],
    )
    return kernel, spatial


def _build_subspace_off_resonance_toeplitz(
    frame_physics: Sequence[MRIPhysics],
    basis: Any,
    options: dict[str, Any],
    streaming: Any | None = None,
) -> tuple[CompactToeplitzKernel, Any]:
    """Combine shared spatial off-resonance factors with a temporal subspace."""
    torch = import_module("torch")
    basis = torch.as_tensor(basis)
    coefficient_rank, n_frames = basis.shape
    if len(frame_physics) != n_frames:
        raise ValueError("basis frame count does not match frame physics")
    first = frame_physics[0].native_operator
    if first is None or not hasattr(first, "B"):
        raise RuntimeError("expected off-resonance-corrected frame operators")
    spatial_factors = first.C
    if any(
        item.native_operator is None
        or not hasattr(item.native_operator, "B")
        or item.native_operator.C is not spatial_factors
        for item in frame_physics
    ):
        raise RuntimeError(
            "combined subspace/off-resonance Toeplitz requires shared "
            "spatial interpolation factors"
        )

    segment_rank = int(first.B.shape[1])
    combined_rank = coefficient_rank * segment_rank
    rows, columns = torch.triu_indices(combined_rank, combined_rank)
    out_coefficients = rows // segment_rank
    in_coefficients = columns // segment_rank
    out_segments = rows % segment_rank
    in_segments = columns % segment_rank
    segment_rows, segment_columns = torch.triu_indices(
        segment_rank,
        segment_rank,
    )
    segment_lookup = torch.empty(
        (segment_rank, segment_rank),
        dtype=torch.int64,
    )
    packed_segment = torch.arange(segment_rows.numel())
    segment_lookup[segment_rows, segment_columns] = packed_segment
    segment_lookup[segment_columns, segment_rows] = packed_segment
    lookup = segment_lookup[out_segments, in_segments]
    conjugate = out_segments > in_segments

    image_shape = tuple(int(size) for size in first.shape)
    spatial_shape = tuple(2 * size for size in image_shape)
    packed = None
    indices = None
    for frame, item in enumerate(frame_physics):
        native = item.native_operator
        assert native is not None
        if tuple(native.shape) != image_shape:
            raise ValueError("all frames must share one image shape")
        segment_values, indices = _off_resonance_scalar_transfers(
            native,
            options,
            indices,
            streaming,
        )
        device = segment_values.device
        basis_device = basis.to(device)
        mixing = (
            basis_device[out_coefficients.to(device), frame]
            * basis_device[in_coefficients.to(device), frame].conj()
        )
        dtype = torch.promote_types(mixing.dtype, segment_values.dtype)
        if streaming is not None:
            if packed is None:
                packed = torch.zeros(
                    (mixing.numel(), segment_values.shape[1]),
                    dtype=dtype,
                    device="cpu",
                )
            for packed_index, coefficient in enumerate(mixing):
                selected = segment_values[lookup[packed_index]]
                if conjugate[packed_index]:
                    selected = selected.conj()
                packed[packed_index].add_(
                    selected.to(dtype),
                    alpha=coefficient.item(),
                )
        else:
            selected = segment_values[lookup.to(device)]
            mask = conjugate.to(device)
            selected[mask] = selected[mask].conj()
            contribution = mixing[:, None].to(dtype) * selected.to(dtype)
            packed = contribution if packed is None else packed + contribution
    assert packed is not None and indices is not None
    packed = packed.to(torch.promote_types(basis.dtype, as_torch(first.B).dtype))
    kernel = CompactToeplitzKernel(
        packed,
        indices,
        spatial_shape,
        combined_rank,
        image_shape=image_shape,
        chunk_size=options["chunk_size"],
        cuda_mode=options["cuda_mode"],
        cuda_max_device_fraction=options["cuda_max_device_fraction"],
        cuda_transfer_precision=options["cuda_transfer_precision"],
    )
    return kernel, spatial_factors


def _apply_subspace_off_resonance_toeplitz(
    kernel: CompactToeplitzKernel,
    image: Any,
    native_operator: Any,
    spatial_factors: Any,
    *,
    coefficient_rank: int,
    coil_batch_size: int,
    streaming: Any | None = None,
) -> Any:
    """Apply a combined coefficient/segment transfer with SENSE maps."""
    torch = import_module("torch")
    if streaming is not None and streaming.device_count > 1:
        coil_batch_size = max(coil_batch_size, streaming.device_count)
    maps = _sense_maps(native_operator, image)
    segment_rank = kernel.rank // coefficient_rank
    spatial_factors = as_torch(
        spatial_factors,
        device=image.device,
    ).to(image.dtype)
    spatial_factors = spatial_factors.reshape(
        segment_rank,
        *kernel.image_shape,
    )
    result = torch.zeros_like(image)
    for start in range(0, maps.shape[0], coil_batch_size):
        coil_maps = maps[start : start + coil_batch_size].to(image.device)
        coil_images = image[:, None] * coil_maps[None, :, None]
        expanded = coil_images[:, :, :, None] * spatial_factors[None, None, None]
        expanded = expanded.flatten(0, 1).flatten(1, 2)
        transformed = (
            kernel.apply_streamed(expanded, streaming)
            if streaming is not None and expanded.device.type == "cpu"
            else kernel.apply(expanded)
        )
        transformed = transformed.unflatten(
            1,
            (coefficient_rank, segment_rank),
        )
        transformed = (transformed * spatial_factors.conj()[None, None]).sum(dim=2)
        transformed = transformed.unflatten(
            0,
            (image.shape[0], coil_maps.shape[0]),
        )
        result += (transformed * coil_maps.conj()[None, :, None]).sum(dim=1)
    kernel.settle_allocator()
    return result


def _enable_toeplitz(
    physics: MRIPhysics,
    *,
    best_effort: bool = False,
    compress: bool = True,
    chunk_size: int = 65536,
    coil_batch_size: int = 1,
    cuda_mode: str = "auto",
    cuda_max_device_fraction: float = 0.85,
    cuda_transfer_precision: str = "auto",
) -> None:
    """Give a physics object a Toeplitz normal operator.

    The kernel is the trajectory gridded onto a grid twice the image in every
    dimension, stored over the locations it reached. Subspace and off-resonance
    decorators carry a matrix-valued transfer built the same way, whose
    Hermitian upper triangle is packed and whose real bases keep real storage.
    An even transfer -- what a trajectory closed under ``k -> -k`` leaves -- is
    stored over half its locations and mirrored as it is applied.

    None of that is a choice. What the arguments settle is execution: how much
    is unpacked at a time, how many coils share a pass, and what a CUDA device
    holds.
    """
    options = _toeplitz_options(
        compress=compress,
        chunk_size=chunk_size,
        coil_batch_size=coil_batch_size,
        cuda_mode=cuda_mode,
        cuda_max_device_fraction=cuda_max_device_fraction,
        cuda_transfer_precision=cuda_transfer_precision,
    )
    if "toeplitz" not in physics.modifiers:
        physics.modifiers = (*physics.modifiers, "toeplitz")
    physics.toeplitz_options = options
    enable = getattr(physics.operator, "enable_toeplitz", None)
    if enable is not None:
        enable(options)
    elif physics.native_operator is not None:
        physics.operator.use_toeplitz = True
    if best_effort and hasattr(physics.operator, "toeplitz_best_effort"):
        physics.operator.toeplitz_best_effort = True


class Toeplitz(MRIPhysics):
    """A physics object whose normal operator uses a precomputed kernel.

    The non-Cartesian operators already build one, so this is the spelling for
    building it with different options, and for accelerating a physics that
    was made without one.

    Parameters
    ----------
    physics
        Base physics to accelerate.
    **options
        Toeplitz options: ``compress``, ``chunk_size``, ``coil_batch_size``
        and the CUDA transfer settings.

    Examples
    --------
    The kernel is the normal operator, not an approximation of it: a scan's
    transfer gridded onto a doubled grid, so ``A^H A`` becomes pad, transform,
    multiply, transform back, crop -- and gives the same answer the two transforms
    would have.

    >>> import torch
    >>> import pulserver.recon as recon
    >>> import sys; sys.path.insert(0, "docs")
    >>> from _figures import phantom, radial_spokes
    >>> truth, coil_maps = phantom(64, coils=4)
    >>> plain = recon.NonCartesian2D(
    ...     radial_spokes(64, 24), (64, 64), coil_maps=coil_maps[0], toeplitz=False
    ... )
    >>> kernelled = recon.Toeplitz(plain)
    >>> exact = plain.A_adjoint_A(truth)
    >>> through_kernel = kernelled.A_adjoint_A(truth)
    >>> error = torch.linalg.vector_norm(exact - through_kernel)
    >>> bool(error / torch.linalg.vector_norm(exact) < 1e-5)
    True
    """

    def __init__(self, physics: MRIPhysics, **kwargs: Any) -> None:
        _enable_toeplitz(physics, **kwargs)
        _init_from(self, physics)
