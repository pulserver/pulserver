"""Backends, array views, and the options a Toeplitz request carries.

What every physics model needs before it can describe a scan: which NUFFT
backend answers, how a measurement is laid out against the coil axis, and how
an image is presented to a solver that wants real numbers."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from importlib import import_module
from itertools import chain
from typing import Any

import deepinv

from .._toeplitz import (
    occupancy_indices,
    support_indices,
)
from .._views import kspace_as_cpx as _kspace_as_cpx
from .._views import kspace_as_real as _kspace_as_real


def _require_deepinv() -> Any:
    try:
        return import_module("deepinv.physics")
    except ImportError as error:
        raise ImportError(
            "MRI physics operators require DeepInverse, which ships with "
            "pulserver; reinstall the package to restore it."
        ) from error


def _require_mrinufft() -> Any:
    try:
        module = import_module("mrinufft")
    except ImportError as error:
        raise ImportError(
            "Non-Cartesian MRI physics requires mri-nufft, which ships "
            "with pulserver; reinstall the package to restore it."
        ) from error
    from .._torch_cufinufft import register_torch_cufinufft

    register_torch_cufinufft()
    return module


def available_nufft_backends() -> list[str]:
    """Return available MRI-NUFFT backends, including Pulserver's Torch CUDA adapter.

    Examples
    --------
    Which non-Cartesian backends this installation can actually use -- the CUDA
    ones appear only where a card and its libraries are present.

    >>> import pulserver.recon as recon
    >>> "finufft" in recon.available_nufft_backends()
    True
    """
    try:
        _require_mrinufft()
    except ImportError:
        return []
    list_backends = import_module("mrinufft.operators").list_backends
    public_names = {
        "cufinufft" if name == "cufinufft-torch" else name
        for name in list_backends(available_only=True)
    }
    return sorted(public_names)


def _resolve_nufft_backend(
    backend: str,
) -> str:
    """Select FINUFFT on CPU and the private Torch CUFINUFFT adapter on CUDA."""
    selected = backend.lower()
    if selected == "cufinufft":
        return "cufinufft-torch"
    if selected != "auto":
        return selected
    try:
        torch = import_module("torch")
    except ImportError:
        return "finufft"
    return "cufinufft-torch" if torch.cuda.is_available() else "finufft"


def _cartesian_image_as_real(value: Any) -> Any:
    """Pack a complex Cartesian image into DeepInverse's channel-one real layout.

    The Cartesian operators are DeepInverse ``MultiCoilMRI`` objects, which
    carry the real and imaginary parts of an image in a dedicated axis at
    position one -- ``(batch, 2, ...)`` -- rather than interleaved into the
    channel axis the way :func:`_image_as_real` does for the mri-nufft path.
    This is the single conversion the complex-native Cartesian boundary uses.
    """
    torch = import_module("torch")
    return torch.view_as_real(value).movedim(-1, 1)


def _cartesian_image_as_cpx(value: Any) -> Any:
    """Restore a complex Cartesian image from the channel-one real layout."""
    torch = import_module("torch")
    return torch.view_as_complex(value.movedim(1, -1).contiguous())


def _operator_device(physics: Any) -> Any:
    """Where a physics object computes, defaulting to the CPU.

    An operator carries tensors of several kinds -- a mask, sensitivities, a
    trajectory, a stack of per-slice operators -- and placing it moves the ones
    the arithmetic runs on while small host-side ones stay behind. So any
    tensor on an accelerator is the answer, and only an operator with none of
    them computes on the host.
    """
    torch = import_module("torch")
    buffers = getattr(physics, "buffers", None)
    parameters = getattr(physics, "parameters", None)
    if callable(buffers) and callable(parameters):
        for tensor in chain(buffers(), parameters()):
            if tensor.device.type != "cpu":
                return tensor.device
    return torch.device("cpu")


def _mirror_array_namespace(method: Callable) -> Callable:
    """Let a physics method accept NumPy and answer in the caller's namespace.

    Torch tensors pass straight through, so the operator's internal recursion
    is untouched and there is no per-call overhead once inside the stack. A
    NumPy array is moved onto the operator's device in its complex or real
    working dtype -- a no-op, and hence zero-copy, when it already is that
    dtype on the host -- and the Torch result is handed back as NumPy. This is
    what lets a reconstruction pass raw arrays straight to ``A``/``A_adjoint``
    without shuttling them across the boundary by hand.
    """

    @wraps(method)
    def wrapper(self: Any, value: Any, *args: Any, **kwargs: Any) -> Any:
        torch = import_module("torch")
        if isinstance(value, torch.Tensor):
            return method(self, value, *args, **kwargs)
        numpy = import_module("numpy")
        if not isinstance(value, numpy.ndarray):
            return method(self, value, *args, **kwargs)
        tensor = torch.as_tensor(value)
        if tensor.is_complex():
            tensor = tensor.to(torch.complex64)
        elif tensor.dtype.is_floating_point:
            tensor = tensor.to(torch.float32)
        tensor = tensor.to(_operator_device(self))
        result = method(self, tensor, *args, **kwargs)
        if isinstance(result, torch.Tensor):
            return result.detach().to("cpu").numpy()
        return result

    return wrapper


def _measurement_to_trailing(value: Any) -> Any:
    """Move a measurement's real/imaginary axis from channel one to the end.

    Pulserver carries the real and imaginary parts of a *measurement* in a
    trailing axis, ``(batch, coils, ..., 2)``; DeepInverse carries them in
    channel position one, ``(batch, 2, coils, ...)``. Images use the packed
    channel form on both sides, so only measurements cross this boundary.

    Parameters
    ----------
    value
        Measurement in the DeepInverse channel-first layout.

    Returns
    -------
    torch.Tensor
        The same measurement in Pulserver's trailing layout.
    """
    return value.movedim(1, -1)


def _measurement_to_channels(value: Any) -> Any:
    """Move a measurement's real/imaginary axis from the end to channel one.

    The inverse of :func:`_measurement_to_trailing`.
    """
    return value.movedim(-1, 1)


class _TrailingRealView(deepinv.physics.LinearPhysics):
    """Present a DeepInverse MRI operator in Pulserver's measurement layout.

    Wraps the operator rather than converting at each call site so that every
    physics object in the package answers with the same measurement shape,
    whatever library implements it underneath. It stays a Torch module so that
    the wrapped operator keeps taking part in ``.to()`` and in the attribute
    forwarding :class:`MRIPhysics` performs.
    """

    def __init__(self, operator: Any) -> None:
        super().__init__()
        self.operator = operator

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("operator"), name)

    def A(self, x: Any, **kwargs: Any) -> Any:
        """Encode an image, answering in the trailing layout."""
        return _measurement_to_trailing(self.operator.A(x, **kwargs))

    def A_adjoint(self, y: Any, **kwargs: Any) -> Any:
        """Decode a measurement given in the trailing layout."""
        return self.operator.A_adjoint(_measurement_to_channels(y), **kwargs)

    def A_adjoint_A(self, x: Any, **kwargs: Any) -> Any:
        """Apply the normal operator, which never leaves image space."""
        return self.operator.A_adjoint_A(x, **kwargs)


class _CartesianComplexView(deepinv.physics.LinearPhysics):
    """Present a trailing-real Cartesian operator as complex-native.

    Wraps a :class:`_TrailingRealView` so images and measurements cross the
    boundary as native complex tensors -- image ``(batch, [coils,] *spatial)``,
    measurement ``(batch, coils, *kspace)`` -- packing to the two-channel real
    layout DeepInverse's ``MultiCoilMRI`` works in only for the duration of one
    call. It is the complex-facing dual of :class:`_TrailingRealView`, and the
    only place the Cartesian real/complex conversion happens once the whole
    stack is complex by default.
    """

    def __init__(self, operator: Any) -> None:
        super().__init__()
        self.operator = operator

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("operator"), name)

    def A(self, x: Any, **kwargs: Any) -> Any:
        """Encode a complex image, answering with complex k-space."""
        real = self.operator.A(_cartesian_image_as_real(x), **kwargs)
        return _kspace_as_cpx(real)

    def A_adjoint(self, y: Any, **kwargs: Any) -> Any:
        """Decode complex k-space, answering with a complex image."""
        image = self.operator.A_adjoint(_kspace_as_real(y), **kwargs)
        return _cartesian_image_as_cpx(image)

    def A_adjoint_A(self, x: Any, **kwargs: Any) -> Any:
        """Apply the exact FFT normal operator, staying in complex image space."""
        result = self.operator.A_adjoint_A(_cartesian_image_as_real(x), **kwargs)
        return _cartesian_image_as_cpx(result)


def _toeplitz_options(
    *,
    compress: bool = True,
    polyphase: bool = True,
    chunk_size: int = 65536,
    coil_batch_size: int = 1,
    cuda_mode: str = "auto",
    cuda_max_device_fraction: float = 0.85,
    cuda_transfer_precision: str = "auto",
) -> dict[str, Any]:
    """Validate how a Toeplitz kernel is applied.

    Nothing here says how the kernel is built. It is gridded onto the doubled
    grid the way BART and MRISubspaceRecon.jl build theirs, and that is not a
    choice. What is left is what it is stored and executed on: whether the
    locations the trajectory never reached are dropped, how much is unpacked
    at a time, how many coils share a pass, and what a CUDA device holds.

    ``compress`` is BART's ``--compress-psf``: the transfer is kept where the
    gridded trajectory is non-zero -- the sampled region plus the rim the
    interpolation spreads into -- and dropped outside, which is what makes a
    large three-dimensional kernel fit. The transfer multiplies the spectrum
    pointwise, so what compression leaves out perturbs the normal operator by
    at most the largest value dropped, which the kernel records as its
    truncation bound. A conjugate-gradient solve that meets the resulting
    indefiniteness stops on its last valid iterate rather than diverging. A
    calibration solved over a small window keeps the whole transfer instead.

    ``polyphase`` files the transfer as one component per parity of the
    doubled grid's coordinates, so the convolution runs on the image grid and
    the doubled one is never materialised. It is the same operator either way.
    """
    if chunk_size <= 0:
        raise ValueError("Toeplitz chunk_size must be positive")
    if coil_batch_size <= 0:
        raise ValueError("Toeplitz coil_batch_size must be positive")
    if cuda_mode not in {"auto", "resident", "compact"}:
        raise ValueError("Toeplitz cuda_mode must be 'auto', 'resident', or 'compact'")
    if not 0.0 < cuda_max_device_fraction <= 1.0:
        raise ValueError("Toeplitz cuda_max_device_fraction must be in (0, 1]")
    if cuda_transfer_precision not in {"auto", "float32", "bfloat16"}:
        raise ValueError(
            "Toeplitz cuda_transfer_precision must be 'auto', 'float32', or 'bfloat16'"
        )
    return {
        "compress": bool(compress),
        "polyphase": bool(polyphase),
        "chunk_size": int(chunk_size),
        "coil_batch_size": int(coil_batch_size),
        "cuda_mode": cuda_mode,
        "cuda_max_device_fraction": float(cuda_max_device_fraction),
        "cuda_transfer_precision": cuda_transfer_precision,
    }


#: Cells either side of a sample that the backend's interpolation spreads
#: into. The gridded transfer is non-zero within this reach of the trajectory
#: and nowhere else, which is the support the kernel is stored over.
_SPREAD_HALF_WIDTH = 4


def _support_locations(
    samples: Any,
    spatial_shape: tuple[int, ...],
    device: Any,
    compress: bool = True,
) -> Any:
    """The locations a gridded transfer holds weight at.

    Where the trajectory landed on the doubled grid, plus the neighbourhood the
    interpolation spread into. An encoding with no trajectory to read, which is
    what a Cartesian one leaves, keeps every location, and so does one that
    asks not to be compressed.
    """
    if samples is None or not compress:
        return support_indices(spatial_shape, support="full", radius=1.0, device=device)
    return occupancy_indices(samples, spatial_shape, width=_SPREAD_HALF_WIDTH).to(
        device
    )


def _toeplitz_request(
    value: bool | str | dict[str, Any],
) -> tuple[bool, bool, dict[str, Any]]:
    """Decode a ``toeplitz=`` argument into enabled, best-effort and options.

    ``"auto"`` asks for the acceleration wherever it is available, so a shape
    or a backend that cannot carry a transfer kernel falls back to the exact
    normal operator rather than raising. ``True`` and an options mapping ask
    for it outright, and say so if it cannot be built.
    """
    if isinstance(value, dict):
        return True, False, _toeplitz_options(**value)
    if value == "auto":
        return True, True, _toeplitz_options()
    return bool(value), False, _toeplitz_options()


def _base_fourier_operator(native_operator: Any) -> Any:
    """Return the undecorated Fourier operator beneath mri-nufft wrappers."""
    return getattr(native_operator, "_fourier_op", native_operator)
