"""Physics for a scan that samples a trajectory.

Radial, spiral and koosh acquisitions, and stacks of them, where the encoding
is a NUFFT and the normal operator is a Toeplitz kernel by default."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from math import prod
from types import SimpleNamespace
from typing import Any


from .._stacked import _StackedNUFFTLinearPhysics

from ._base import MRIPhysics, _init_from
from ._cartesian import _single_precision
from ._common import _require_mrinufft, _resolve_nufft_backend, _toeplitz_request
from ._frames import _frame_density, _native_linear_physics
from ._kernel import (
    _configure_base_toeplitz,
    _enable_toeplitz,
    _remember_plan_settings,
)


def _stacked_trajectory_bank(
    trajectory: Any,
    z_index: Any,
    stack_size: int,
) -> tuple[list[Any], Any, list[Any] | None]:
    """Resolve shared or plane-specific 2D trajectories and stack indices."""
    numpy = import_module("numpy")

    def host(value: Any) -> Any:
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        return numpy.asarray(value)

    shape = getattr(trajectory, "shape", ())
    if not shape and isinstance(trajectory, (list, tuple)) and trajectory:
        shape = getattr(trajectory[0], "shape", ())
    coordinate_dim = shape[-1] if shape else None
    if coordinate_dim == 3:
        samples = host(trajectory).reshape(-1, 3)
        z_coordinates = numpy.asarray(
            import_module("mrinufft._utils").proper_trajectory(
                samples[:, 2],
                normalize="unit",
            )
        ).reshape(-1)
        _, first = numpy.unique(z_coordinates, return_index=True)
        ordered_z = z_coordinates[numpy.sort(first)]
        groups = [
            numpy.flatnonzero(numpy.isclose(z_coordinates, value))
            for value in ordered_z
        ]
        planes = [samples[group, :2] for group in groups]
        indices = numpy.rint(ordered_z * stack_size + stack_size // 2).astype(
            numpy.int64
        )
        if numpy.any(indices < 0) or numpy.any(indices >= stack_size):
            raise ValueError("stacked trajectory contains an out-of-grid z coordinate")
        if numpy.unique(indices).size != indices.size:
            raise ValueError("stacked trajectory maps multiple planes to one z index")
        return planes, indices, groups
    if coordinate_dim != 2:
        raise ValueError("stacked trajectories must end in two or three coordinates")

    if z_index is None or (isinstance(z_index, str) and z_index == "auto"):
        indices = numpy.arange(stack_size, dtype=numpy.int64)
    else:
        try:
            indices = numpy.arange(stack_size, dtype=numpy.int64)[host(z_index)]
        except IndexError as error:
            raise ValueError("z_index must select valid stack entries") from error
        indices = numpy.asarray(indices, dtype=numpy.int64).reshape(-1)
    if indices.size == 0:
        raise ValueError("z_index must select at least one stack entry")
    if numpy.unique(indices).size != indices.size:
        raise ValueError("z_index must not contain duplicate stack entries")

    explicit_sequence = isinstance(trajectory, (list, tuple))
    array = None if explicit_sequence else host(trajectory)
    banked_array = (
        array is not None and array.ndim >= 4 and array.shape[0] == indices.size
    )
    if explicit_sequence or banked_array:
        entries = list(trajectory) if explicit_sequence else list(array)
        if len(entries) != indices.size:
            raise ValueError("one 2D trajectory is required per selected stack plane")
        planes = [host(entry).reshape(-1, 2) for entry in entries]
    else:
        shared = host(trajectory).reshape(-1, 2)
        planes = [shared] * indices.size
    return planes, indices, None


def _stacked_density_bank(
    density: Any | None,
    trajectories: Sequence[Any],
    sample_groups: Sequence[Any] | None,
) -> list[Any | None]:
    """Resolve shared, banked, or flattened stack density weights."""
    if density is None:
        return [None] * len(trajectories)
    numpy = import_module("numpy")

    def host(value: Any) -> Any:
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        return numpy.asarray(value)

    counts = [int(numpy.asarray(item).reshape(-1, 2).shape[0]) for item in trajectories]
    if isinstance(density, (list, tuple)):
        if len(density) != len(trajectories):
            raise ValueError("one density array is required per stack trajectory")
        result = [host(item).reshape(-1) for item in density]
    else:
        weights = host(density)
        if weights.size == counts[0] and all(count == counts[0] for count in counts):
            result = [weights.reshape(-1)] * len(trajectories)
        elif (
            weights.ndim >= 2
            and weights.shape[0] == len(trajectories)
            and all(weights[index].size == count for index, count in enumerate(counts))
        ):
            result = [weights[index].reshape(-1) for index in range(len(counts))]
        elif weights.size == sum(counts):
            flattened = weights.reshape(-1)
            if sample_groups is not None:
                result = [flattened[group] for group in sample_groups]
            else:
                result = []
                start = 0
                for count in counts:
                    result.append(flattened[start : start + count])
                    start += count
        else:
            raise ValueError("stacked density does not match the trajectory bank")
    if any(item.size != count for item, count in zip(result, counts, strict=True)):
        raise ValueError("stacked density must have one weight per sample")
    return result


def _stacked_linear_physics(
    mrinufft: Any,
    trajectory: Any,
    image_shape: tuple[int, int, int],
    *,
    coil_maps: Any | None,
    density: Any | None,
    backend: str,
    n_coils: int,
    n_batchs: int,
    z_index: Any,
    viewed_as_real: bool,
    toeplitz_enabled: bool,
    toeplitz_options: dict[str, Any],
    operator_kwargs: dict[str, Any],
) -> tuple[Any, Any]:
    """Build shared-batch or independent plane NUFFTs for a stack."""
    numpy = import_module("numpy")
    trajectories, indices, groups = _stacked_trajectory_bank(
        trajectory,
        z_index,
        image_shape[-1],
    )
    densities = _stacked_density_bank(density, trajectories, groups)
    shared = all(
        numpy.array_equal(trajectories[0], item) for item in trajectories[1:]
    ) and all(
        (densities[0] is None and item is None)
        or (
            densities[0] is not None
            and item is not None
            and numpy.array_equal(densities[0], item)
        )
        for item in densities[1:]
    )

    common = {
        "shape": image_shape[:2],
        "smaps": None,
        "n_batchs": n_batchs,
        "squeeze_dims": False,
        **operator_kwargs,
    }
    if shared:
        native_operators = [
            mrinufft.get_operator(backend)(
                samples=trajectories[0],
                density=densities[0],
                n_coils=n_coils * len(indices),
                **common,
            )
        ]
    else:
        native_operators = [
            mrinufft.get_operator(backend)(
                samples=samples,
                density=weights,
                n_coils=n_coils,
                **common,
            )
            for samples, weights in zip(trajectories, densities, strict=True)
        ]
    plane_physics = [
        _native_linear_physics(native, viewed_as_real=False)
        for native in native_operators
    ]
    operator = _StackedNUFFTLinearPhysics(
        plane_physics,
        native_operators,
        indices,
        image_shape,
        coil_maps=coil_maps,
        n_coils=n_coils,
        viewed_as_real=viewed_as_real,
        toeplitz=toeplitz_enabled,
        toeplitz_options=toeplitz_options,
        shared_operator=shared,
    )
    native_proxy = SimpleNamespace(
        shape=image_shape,
        smaps=coil_maps,
        plane_operators=tuple(native_operators),
        z_index=indices,
        shared_trajectory=shared,
        stacked=True,
    )
    return operator, native_proxy


def _noncartesian(
    trajectory: Any,
    image_shape: tuple[int, ...],
    *,
    spatial_ndim: int,
    coil_maps: Any | None,
    density: Any | None,
    backend: str,
    n_coils: int,
    n_batchs: int,
    stacked: bool,
    z_index: Any,
    toeplitz: bool | str | dict[str, Any],
    viewed_as_real: bool,
    streaming: Any | None,
    operator_kwargs: dict[str, Any],
) -> MRIPhysics:
    toeplitz_enabled, best_effort, toeplitz_config = _toeplitz_request(toeplitz)
    if len(image_shape) != spatial_ndim:
        raise ValueError(
            f"image_shape must have {spatial_ndim} entries, got {image_shape!r}"
        )
    trajectory = _single_precision(trajectory)
    density = _single_precision(density)
    coil_maps = _single_precision(coil_maps)
    trajectory_shape = getattr(trajectory, "shape", ())
    if not trajectory_shape and isinstance(trajectory, (list, tuple)) and trajectory:
        trajectory_shape = getattr(trajectory[0], "shape", ())
    trajectory_dim = trajectory_shape[-1] if trajectory_shape else None
    valid_dimensions = {2, 3} if stacked else {spatial_ndim}
    if trajectory_dim is not None and trajectory_dim not in valid_dimensions:
        raise ValueError(
            f"trajectory must end in {sorted(valid_dimensions)} coordinates, "
            f"got {trajectory_dim}"
        )
    if stacked and spatial_ndim != 3:
        raise ValueError("stacked trajectories are only supported by NonCartesian3D")
    map_shape = getattr(coil_maps, "shape", ())
    if coil_maps is not None:
        if len(map_shape) == spatial_ndim + 1:
            inferred_coils = int(map_shape[0])
        elif len(map_shape) == spatial_ndim + 2:
            inferred_coils = int(map_shape[1])
        else:
            raise ValueError(
                "coil_maps must have shape (coils, *image_shape) or "
                "(batch, coils, *image_shape)"
            )
        if n_coils not in {1, inferred_coils}:
            raise ValueError("n_coils conflicts with the sensitivity-map bank")
        n_coils = inferred_coils

    mrinufft = _require_mrinufft()
    backend = _resolve_nufft_backend(backend)
    operator_kwargs = dict(operator_kwargs)
    if backend == "cufinufft-torch" and "gpu_device_id" not in operator_kwargs:
        selected_device = (
            streaming.torch_device
            if streaming is not None and hasattr(streaming, "torch_device")
            else getattr(trajectory, "device", None)
        )
        if getattr(selected_device, "type", None) == "cuda":
            operator_kwargs["gpu_device_id"] = selected_device.index or 0
    native_coil_maps = coil_maps
    if (
        backend == "finufft"
        and hasattr(coil_maps, "detach")
        and getattr(coil_maps.device, "type", None) == "cpu"
    ):
        native_coil_maps = coil_maps.detach().numpy()
    frame_stacked = not stacked and streaming is not None and len(trajectory_shape) >= 4
    native_trajectory = trajectory[0] if frame_stacked else trajectory
    native_density = density
    density_shape = getattr(density, "shape", ())
    if (
        frame_stacked
        and density is not None
        and density_shape
        and density_shape[0] == trajectory_shape[0]
    ):
        native_density = density[0]
    elif (
        density is not None
        and len(trajectory_shape) >= 3
        and len(density_shape) > 1
        and prod(density_shape) == prod(trajectory_shape[:-1])
    ):
        # The operator plans on every frame's samples at once, so a density
        # given one row per frame is flat to it.
        native_density = density.reshape(-1)
    if stacked:
        operator, native = _stacked_linear_physics(
            mrinufft,
            native_trajectory,
            image_shape,
            coil_maps=coil_maps,
            density=native_density,
            backend=backend,
            n_coils=n_coils,
            n_batchs=n_batchs,
            z_index=z_index,
            viewed_as_real=viewed_as_real,
            toeplitz_enabled=toeplitz_enabled,
            toeplitz_options=toeplitz_config,
            operator_kwargs=operator_kwargs,
        )
    else:
        native = mrinufft.get_operator(backend)(
            samples=native_trajectory,
            shape=image_shape,
            smaps=native_coil_maps,
            density=native_density,
            n_coils=n_coils,
            n_batchs=n_batchs,
            squeeze_dims=False,
            **operator_kwargs,
        )
        _remember_plan_settings(native, operator_kwargs)
        operator = _native_linear_physics(native, viewed_as_real=viewed_as_real)
        operator = _configure_base_toeplitz(
            operator,
            native,
            enabled=toeplitz_enabled,
            best_effort=best_effort,
            options=toeplitz_config,
        )

    def rebuild(
        new_trajectory: Any,
        frame_index: int | None = None,
    ) -> MRIPhysics:
        frame_density = _frame_density(
            density,
            trajectory,
            frame_index,
            prod(getattr(new_trajectory, "shape", (0,))[:-1]),
        )
        return _noncartesian(
            new_trajectory,
            image_shape,
            spatial_ndim=spatial_ndim,
            coil_maps=coil_maps,
            density=frame_density,
            backend=backend,
            n_coils=n_coils,
            n_batchs=n_batchs,
            stacked=stacked,
            z_index=z_index,
            toeplitz=toeplitz,
            viewed_as_real=viewed_as_real,
            streaming=streaming,
            operator_kwargs=operator_kwargs,
        )

    result = MRIPhysics(
        operator,
        native_operator=native,
        kind=f"noncartesian{spatial_ndim}d",
        spatial_ndim=spatial_ndim,
        viewed_as_real=viewed_as_real,
        modifiers=(("stacked",) if stacked else ())
        + (("toeplitz",) if toeplitz_enabled else ()),
        trajectory=trajectory,
        rebuild=rebuild,
        toeplitz_options=toeplitz_config if toeplitz_enabled else None,
    )

    def replicate(device: Any, device_policy: Any) -> MRIPhysics:
        replica_kwargs = dict(operator_kwargs)
        replica_kwargs["gpu_device_id"] = device.index
        return _noncartesian(
            trajectory,
            image_shape,
            spatial_ndim=spatial_ndim,
            coil_maps=coil_maps,
            density=density,
            backend=backend,
            n_coils=n_coils,
            n_batchs=n_batchs,
            stacked=stacked,
            z_index=z_index,
            toeplitz=toeplitz,
            viewed_as_real=viewed_as_real,
            streaming=device_policy,
            operator_kwargs=replica_kwargs,
        )

    if backend == "cufinufft-torch":
        result._replicate = replicate
    if streaming is not None:
        result.enable_streaming(streaming)
    return result


class NonCartesian2D(MRIPhysics):
    """Two-dimensional non-Cartesian MRI physics, over MRI-NUFFT.

    Parameters
    ----------
    trajectory
        K-space coordinates shaped ``(samples, 2)`` or ``(shots, samples, 2)``,
        in MRI-NUFFT's ``[-0.5, 0.5)`` units.
    image_shape
        Reconstructed matrix, ``(h, w)``.
    coil_maps
        Complex sensitivities shaped ``(coils, h, w)``. ``None`` encodes a
        single channel.
    density
        Density-compensation weights, or a name MRI-NUFFT recognises. See
        :func:`pulserver.mrd.pipe_menon_dcf`.
    backend
        MRI-NUFFT backend. ``"auto"`` picks FINUFFT on CPU and Pulserver's
        Torch-native CUFINUFFT adapter on a CUDA host.
    n_coils, n_batchs
        Coil and batch counts the backend plans for.
    toeplitz
        How the normal operator is computed. ``"auto"``, the default, builds a
        transfer kernel on the first normal-operator call -- exact, and what
        makes an iterative solve worth running -- and falls back to the plain
        adjoint-of-forward for a shape no kernel can embed. ``False`` is the
        plain one outright, ``True`` insists on the kernel, and a dict is the
        kernel with these options.
    viewed_as_real
        Exchange images and measurements through real views.
    streaming
        Optional :class:`pulserver.recon.execution.CudaStreaming` policy.
    **kwargs
        Forwarded to the MRI-NUFFT operator.

    Notes
    -----
    Images are ``(batch, 2, h, w)``, measurements ``(batch, coils, k, 2)``.

    Examples
    --------
    >>> import numpy as np
    >>> import torch
    >>> import pulserver.recon as recon
    >>> angles = np.linspace(0, np.pi, 8, endpoint=False)
    >>> radius = np.linspace(-0.5, 0.5, 32)
    >>> trajectory = np.stack(
    ...     [np.outer(np.cos(angles), radius), np.outer(np.sin(angles), radius)], -1
    ... ).reshape(-1, 2)

    With no maps the adjoint grids each coil onto the image matrix, which is
    what a density-compensated first estimate is made of:

    >>> physics = recon.NonCartesian2D(trajectory, (16, 16))
    >>> physics.A_adjoint(torch.ones(1, 1, 256, dtype=torch.complex64)).shape
    torch.Size([1, 1, 16, 16])

    Golden-angle spokes, gridded and then solved. The adjoint needs the
    density compensation because the samples crowd the centre; the solve does
    not, because the operator's normal equations already account for it:

    .. plot::

       import numpy as np
       import torch
       import pulserver.recon as recon
       import pulserver.mrd as mrd
       from _figures import images, phantom

       truth, coil_maps = phantom(64, coils=4)
       angles = np.pi * (np.arange(48) * 0.618034 % 1.0)
       radius = np.linspace(-0.5, 0.5, 128)
       trajectory = np.stack(
           [np.outer(np.cos(angles), radius), np.outer(np.sin(angles), radius)], -1
       ).reshape(-1, 2)

       physics = recon.NonCartesian2D(trajectory, (64, 64), coil_maps=coil_maps)
       measured = physics.A(truth)
       weights = torch.as_tensor(
           np.asarray(mrd.pipe_menon_dcf(trajectory, (64, 64))), dtype=torch.complex64
       )
       images(
           [
               ("object", truth),
               ("density-compensated adjoint", physics.A_adjoint(measured * weights)),
               ("CG-SENSE", recon.pics(measured, physics, iterations=15)),
           ],
           title="NonCartesian2D, 48 golden-angle spokes",
       )
    """

    def __init__(
        self,
        trajectory: Any,
        image_shape: tuple[int, int],
        *,
        coil_maps: Any | None = None,
        density: Any | None = None,
        backend: str = "auto",
        n_coils: int = 1,
        n_batchs: int = 1,
        toeplitz: bool | str | dict[str, Any] = "auto",
        viewed_as_real: bool = False,
        streaming: Any | None = None,
        **kwargs: Any,
    ) -> None:
        base = _noncartesian(
            trajectory,
            image_shape,
            spatial_ndim=2,
            coil_maps=coil_maps,
            density=density,
            backend=backend,
            n_coils=n_coils,
            n_batchs=n_batchs,
            stacked=False,
            z_index=None,
            toeplitz=toeplitz,
            viewed_as_real=viewed_as_real,
            streaming=streaming,
            operator_kwargs=kwargs,
        )
        enabled, best_effort, options = _toeplitz_request(toeplitz)
        if enabled:
            _enable_toeplitz(base, best_effort=best_effort, **options)
        _init_from(self, base)


class NonCartesian3D(MRIPhysics):
    """Three-dimensional or stack-of-NUFFTs MRI physics.

    With ``stacked=True``, one 2D trajectory array is batched across selected
    stack-frequency planes. A Python sequence supplies independent plane
    trajectories; a 3D-coordinate trajectory is grouped by its Cartesian
    stack coordinate. Shared and plane-specific density layouts follow the
    same convention.

    Parameters
    ----------
    trajectory
        K-space coordinates ending in three components, or -- under
        ``stacked`` -- one 2D trajectory or a sequence of per-plane ones.
    image_shape
        Reconstructed matrix, ``(d, h, w)``.
    coil_maps
        Complex sensitivities shaped ``(coils, d, h, w)``.
    density
        Density-compensation weights, shared or per plane.
    backend
        MRI-NUFFT backend.
    n_coils, n_batchs
        Coil and batch counts the backend plans for.
    stacked
        Encode a stack of 2D trajectories rather than a full 3D one.
    z_index
        Stack-frequency planes to encode. ``"auto"`` takes them from the
        trajectory.
    toeplitz
        How the normal operator is computed. ``"auto"``, the default, builds a
        transfer kernel on the first normal-operator call -- exact, and what
        makes an iterative solve worth running -- and falls back to the plain
        adjoint-of-forward for a shape no kernel can embed. ``False`` is the
        plain one outright, ``True`` insists on the kernel, and a dict is the
        kernel with these options.
    viewed_as_real
        Exchange images and measurements through real views.
    streaming
        Optional :class:`pulserver.recon.execution.CudaStreaming` policy.
    **kwargs
        Forwarded to the MRI-NUFFT operator.

    Notes
    -----
    Images are ``(batch, 2, d, h, w)``, measurements ``(batch, coils, k, 2)``.

    Examples
    --------
    Spiral projection imaging is a centre-out scan: every interleave starts at
    the middle of k-space and winds outward in a plane through it, so the
    adjoint alone is heavily weighted there and needs the density compensation
    a solve applies for it.

    .. plot::

       import pulserver.recon as recon
       from _figures import images, spiral_projections, volume

       truth, coil_maps = volume(24, coils=4, depth=24)
       trajectory = spiral_projections(24, 48)

       physics = recon.NonCartesian3D(trajectory, (24, 24, 24), coil_maps=coil_maps[0])
       measured = physics.A(truth)
       adjoint = physics.A_adjoint(measured)
       solved = recon.pics(measured, physics, iterations=10)

       middle = 12
       images(
           [
               ("truth", truth[0, middle]),
               ("adjoint", adjoint[0, 0, middle]),
               ("CG-SENSE, 10 iterations", solved[0, 0, middle]),
           ],
           title="NonCartesian3D over 48 spiral projections",
       )
    """

    def __init__(
        self,
        trajectory: Any,
        image_shape: tuple[int, int, int],
        *,
        coil_maps: Any | None = None,
        density: Any | None = None,
        backend: str = "auto",
        n_coils: int = 1,
        n_batchs: int = 1,
        stacked: bool = False,
        z_index: Any = "auto",
        toeplitz: bool | str | dict[str, Any] = "auto",
        viewed_as_real: bool = False,
        streaming: Any | None = None,
        **kwargs: Any,
    ) -> None:
        base = _noncartesian(
            trajectory,
            image_shape,
            spatial_ndim=3,
            coil_maps=coil_maps,
            density=density,
            backend=backend,
            n_coils=n_coils,
            n_batchs=n_batchs,
            stacked=stacked,
            z_index=z_index,
            toeplitz=toeplitz,
            viewed_as_real=viewed_as_real,
            streaming=streaming,
            operator_kwargs=kwargs,
        )
        enabled, best_effort, options = _toeplitz_request(toeplitz)
        if enabled:
            _enable_toeplitz(base, best_effort=best_effort, **options)
        _init_from(self, base)
