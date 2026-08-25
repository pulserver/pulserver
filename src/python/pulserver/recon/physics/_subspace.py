"""Physics for a scan whose contrast varies along a temporal basis.

The image is a small set of coefficient maps rather than one image per frame,
and the encoding carries each frame's basis weight. One kernel serves every
frame: it is built once per basis pair, not once per frame."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from importlib import import_module
from math import prod
from typing import Any


from ._base import MRIPhysics, _init_from
from ._common import _base_fourier_operator, _require_deepinv, _toeplitz_options
from ._frames import _FramePhysicsProvider, _LazyFramePhysics
from ._kernel import (
    _apply_sense_toeplitz,
    _apply_subspace_off_resonance_toeplitz,
    _build_cartesian_subspace_toeplitz,
    _build_subspace_off_resonance_toeplitz,
    _build_subspace_toeplitz,
    _enable_toeplitz,
    _frame_coil_view,
)


class _FlatSubspaceEncoding:
    """Encode a dynamic acquisition through one plan over every sample.

    A subspace acquisition is a NUFFT per frame only if you insist on frames.
    The transform is linear in the data, so weighting a frame's samples by its
    basis coefficient and gridding the whole trajectory at once gives the same
    answer with one plan instead of one per frame -- and without accumulating a
    volume per frame, which is what a frame-at-a-time adjoint really spends.

    Sample sets that large are held one group of coils at a time: the data for
    every coil at once is the largest array in a reconstruction, and there is
    no reason for a second one beside it.
    """

    def __init__(self, physics: MRIPhysics, trajectory: Any) -> None:
        self.physics = physics
        shape = tuple(int(size) for size in trajectory.shape)
        self.n_frames = shape[0]
        self.per_frame = prod(shape[1:-1])
        native = _base_fourier_operator(physics.native_operator)
        self.n_coils = int(getattr(native, "n_coils", 1))
        self.uses_sense = getattr(native, "smaps", None) is not None

    def _chunks(self, reference: Any) -> list[range]:
        """Coil groups sized to what the device has room for."""
        if not self.uses_sense:
            return [range(self.n_coils)]
        torch = import_module("torch")
        per_coil = 8 * self.n_frames * self.per_frame * max(reference.shape[0], 1)
        if reference.device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(reference.device)
            budget = int(0.2 * free)
        else:
            budget = 4 * 1024**3
        width = max(1, min(self.n_coils, budget // max(2 * per_coil, 1)))
        return [
            range(start, min(start + width, self.n_coils))
            for start in range(0, self.n_coils, width)
        ]

    @contextmanager
    def _restricted(self, coils: range) -> Any:
        """The encoding operator seen as carrying only these coils."""
        native = _base_fourier_operator(self.physics.native_operator)
        if not self.uses_sense or len(coils) == self.n_coils:
            yield self.physics
            return
        held = native.smaps
        native.smaps = held[coils.start : coils.stop]
        try:
            yield self.physics
        finally:
            native.smaps = held

    def encode(self, coefficients: Any, basis: Any) -> Any:
        """Measurements ``(batch, frames, coils, samples)`` from coefficients."""
        torch = import_module("torch")
        batch, rank = coefficients.shape[0], coefficients.shape[1]
        weights = basis.to(device=coefficients.device, dtype=coefficients.dtype).conj()
        measurements = None
        for coils in self._chunks(coefficients):
            part = None
            for index in range(rank):
                with self._restricted(coils) as operator:
                    full = operator.A(coefficients[:, index])
                full = full.reshape(batch, len(coils), self.n_frames, self.per_frame)
                scaled = full * weights[index].reshape(1, 1, -1, 1)
                part = scaled if part is None else part + scaled
                del full, scaled
            part = part.transpose(1, 2)
            if measurements is None:
                measurements = torch.empty(
                    (batch, self.n_frames, self.n_coils, self.per_frame),
                    dtype=part.dtype,
                    device=part.device,
                )
            measurements[:, :, coils.start : coils.stop] = part
            del part
        assert measurements is not None
        return measurements

    def decode(self, measurements: Any, basis: Any, rank: int) -> Any:
        """Coefficients ``(batch, rank, *image)`` from measurements."""
        torch = import_module("torch")
        batch = measurements.shape[0]
        weights = basis.to(device=measurements.device, dtype=measurements.dtype)
        coefficients = None
        for coils in self._chunks(measurements):
            block = measurements[:, :, coils.start : coils.stop]
            for index in range(rank):
                weighted = block * weights[index].reshape(1, -1, 1, 1)
                weighted = weighted.transpose(1, 2).reshape(batch, len(coils), -1)
                with self._restricted(coils) as operator:
                    image = operator.A_adjoint(weighted)
                del weighted
                image = image.reshape(batch, *image.shape[-self.physics.spatial_ndim :])
                if coefficients is None:
                    coefficients = torch.zeros(
                        (batch, rank, *image.shape[1:]),
                        dtype=image.dtype,
                        device=image.device,
                    )
                coefficients[:, index] += image
                del image
            del block
        assert coefficients is not None
        return coefficients


def _subspace_linear_physics(
    frame_physics: Sequence[MRIPhysics | _LazyFramePhysics],
    basis: Any,
    *,
    viewed_as_real: bool,
    toeplitz_config: dict[str, Any] | None,
    flat_encoding: _FlatSubspaceEncoding | None = None,
) -> Any:
    physics_module = _require_deepinv()
    try:
        torch = import_module("torch")
    except ImportError as error:
        raise ImportError("Subspace physics requires Torch.") from error

    class _SubspaceLinearPhysics(physics_module.LinearPhysics):
        def __init__(self) -> None:
            super().__init__()
            self.__dict__["frame_physics"] = tuple(frame_physics)
            self.__dict__["flat_encoding"] = flat_encoding
            self.__dict__["spatial_rank"] = int(
                getattr(frame_physics[0], "spatial_ndim", 2)
            )
            self.__dict__["basis"] = torch.as_tensor(basis)
            self.viewed_as_real = viewed_as_real
            self.use_toeplitz = bool(frame_physics) and all(
                item.normal_mode in {"toeplitz", "exact-fft"} for item in frame_physics
            )
            self._toeplitz_options = (
                dict(toeplitz_config)
                if toeplitz_config is not None
                else _toeplitz_options()
            )
            self.toeplitz_kernel = None
            if all(item.kind.startswith("cartesian") for item in frame_physics):
                self._compact_toeplitz = "cartesian-subspace"
            elif any("stacked" in item.modifiers for item in frame_physics):
                # Each frame already owns a compact plane-kernel bank. The
                # general subspace loop composes those exact normals without
                # materializing a dense rank-by-rank 3D transfer.
                self._compact_toeplitz = None
            elif any(isinstance(item, _LazyFramePhysics) for item in frame_physics):
                self._compact_toeplitz = (
                    "subspace-off-resonance"
                    if "off_resonance" in frame_physics[0].modifiers
                    else "subspace"
                )
            elif all(
                (native := item.native_operator) is not None
                and not hasattr(native, "B")
                for item in frame_physics
            ):
                self._compact_toeplitz = "subspace"
            elif (
                frame_physics
                and all(
                    (native := item.native_operator) is not None
                    and hasattr(native, "B")
                    for item in frame_physics
                )
                and all(
                    item.native_operator.C is frame_physics[0].native_operator.C
                    for item in frame_physics
                )
            ):
                self._compact_toeplitz = "subspace-off-resonance"
            else:
                self._compact_toeplitz = None
            self._toeplitz_spatial_factors = None
            self._toeplitz_native_proxy = None
            self.streaming_policy = None
            self.streaming_methods = {"A", "A_adjoint", "A_adjoint_A"}

        def enable_toeplitz(self, options: dict[str, Any]) -> None:
            self._toeplitz_options = dict(options)
            self.toeplitz_kernel = None
            self._toeplitz_spatial_factors = None
            self._toeplitz_native_proxy = None
            for item in self.frame_physics:
                if isinstance(item, _LazyFramePhysics):
                    item.enable_toeplitz(options)
                else:
                    _enable_toeplitz(item, **options)
            self.use_toeplitz = bool(self.frame_physics) and all(
                item.normal_mode in {"toeplitz", "exact-fft"}
                for item in self.frame_physics
            )

        def enable_streaming(self, policy: Any) -> None:
            self.streaming_policy = policy
            unique_frames = {id(frame): frame for frame in self.frame_physics}.values()
            for frame in unique_frames:
                frame.enable_streaming(policy)

        @staticmethod
        def _image_as_cpx(x: Any) -> Any:
            batch, channels, *spatial = x.shape
            if channels % 2:
                raise ValueError(
                    "real-view subspace images need 2 channels per coefficient"
                )
            x = x.reshape(batch, channels // 2, 2, *spatial).movedim(2, -1)
            return torch.view_as_complex(x.contiguous())

        @staticmethod
        def _image_as_real(x: Any) -> Any:
            batch, channels, *spatial = x.shape
            x = torch.view_as_real(x).movedim(-1, 2)
            return x.reshape(batch, channels * 2, *spatial)

        def _expand(self, coefficients: Any) -> Any:
            basis_t = self.basis.to(
                device=coefficients.device, dtype=coefficients.dtype
            )
            return torch.einsum("kt,bk...->bt...", basis_t.conj(), coefficients)

        def _project(self, frames: Any) -> Any:
            basis_t = self.basis.to(device=frames.device, dtype=frames.dtype)
            return torch.einsum("kt,bt...->bk...", basis_t, frames)

        def A(self, x: Any, **kwargs: Any) -> Any:
            del kwargs
            coefficients = self._image_as_cpx(x) if self.viewed_as_real else x
            if (
                self.streaming_policy is not None
                and isinstance(coefficients, torch.Tensor)
                and coefficients.device.type == "cpu"
            ):
                basis_cpu = self.basis.to(
                    device="cpu",
                    dtype=coefficients.dtype,
                )
                measurements = None
                for index, frame_physics_item in enumerate(self.frame_physics):
                    frame = (
                        basis_cpu[:, index]
                        .conj()
                        .reshape(1, -1, *([1] * (coefficients.ndim - 2)))
                        * coefficients
                    ).sum(dim=1, keepdim=True)
                    if frame_physics_item.viewed_as_real:
                        frame = self._image_as_real(frame)
                    else:
                        frame = frame[:, 0]
                    measurement = frame_physics_item.A(frame)
                    if measurements is None:
                        measurements = torch.empty(
                            (
                                measurement.shape[0],
                                len(self.frame_physics),
                                *measurement.shape[1:],
                            ),
                            dtype=measurement.dtype,
                            device="cpu",
                            pin_memory=self.streaming_policy.pin_memory,
                        )
                    measurements[:, index].copy_(measurement)
                assert measurements is not None
                return measurements
            if self.flat_encoding is not None:
                return self.flat_encoding.encode(coefficients, self.basis)
            frames = self._expand(coefficients)
            measurements = []
            for index, physics in enumerate(self.frame_physics):
                frame = frames[:, index : index + 1]
                if physics.viewed_as_real:
                    frame = self._image_as_real(frame)
                else:
                    # A complex frame physics takes the image without the
                    # subspace coefficient axis, ``(batch, *spatial)``.
                    frame = frame[:, 0]
                measurements.append(physics.A(frame))
            return torch.stack(measurements, dim=1)

        def A_adjoint(self, y: Any, **kwargs: Any) -> Any:
            del kwargs
            if (
                self.streaming_policy is not None
                and isinstance(y, torch.Tensor)
                and y.device.type == "cpu"
            ):
                policy = self.streaming_policy
                streams = [
                    torch.cuda.Stream(device=policy.torch_device)
                    for _ in range(policy.streams)
                ]
                staged: list[Any | None] = [None] * policy.streams
                events: list[Any | None] = [None] * policy.streams

                def prefetch(frame_index: int) -> None:
                    slot = frame_index % policy.streams
                    streams[slot].synchronize()
                    source = y[:, frame_index]
                    if policy.pin_memory and not source.is_pinned():
                        host = torch.empty_like(source, pin_memory=True)
                        host.copy_(source)
                    else:
                        host = source
                    with torch.cuda.stream(streams[slot]):
                        staged[slot] = host.to(
                            policy.torch_device,
                            non_blocking=policy.pin_memory,
                        )
                        events[slot] = torch.cuda.Event()
                        events[slot].record(streams[slot])

                prefetch(0)
                coefficients = None
                basis_cpu = self.basis.to("cpu")
                for index, frame_physics_item in enumerate(self.frame_physics):
                    if index + 1 < len(self.frame_physics):
                        prefetch(index + 1)
                    slot = index % policy.streams
                    events[slot].synchronize()
                    frame = frame_physics_item.A_adjoint(staged[slot])
                    if frame_physics_item.viewed_as_real:
                        frame = self._image_as_cpx(frame)
                    else:
                        frame = frame.reshape(
                            frame.shape[0], 1, *frame.shape[-self.spatial_rank :]
                        )
                    frame = frame.to("cpu")
                    if coefficients is None:
                        coefficients = torch.zeros(
                            (
                                frame.shape[0],
                                self.basis.shape[0],
                                *frame.shape[2:],
                            ),
                            dtype=frame.dtype,
                            device="cpu",
                            pin_memory=policy.pin_memory,
                        )
                    coefficients += (
                        basis_cpu[:, index]
                        .to(frame.dtype)
                        .reshape(1, -1, *([1] * (frame.ndim - 2)))
                        * frame
                    )
                assert coefficients is not None
                return (
                    self._image_as_real(coefficients)
                    if self.viewed_as_real
                    else coefficients
                )
            if self.flat_encoding is not None:
                coefficients = self.flat_encoding.decode(
                    y,
                    self.basis,
                    int(self.basis.shape[0]),
                )
                return (
                    self._image_as_real(coefficients)
                    if self.viewed_as_real
                    else coefficients
                )
            frames = []
            for index, physics in enumerate(self.frame_physics):
                frame = physics.A_adjoint(y[:, index])
                if physics.viewed_as_real:
                    frame = self._image_as_cpx(frame)
                else:
                    # One coefficient axis, whether or not the frame physics
                    # answered with a coil axis of its own.
                    frame = frame.reshape(
                        frame.shape[0], 1, *frame.shape[-self.spatial_rank :]
                    )
                frames.append(frame)
            coefficients = self._project(torch.cat(frames, dim=1))
            return (
                self._image_as_real(coefficients)
                if self.viewed_as_real
                else coefficients
            )

        def A_adjoint_A(self, x: Any, **kwargs: Any) -> Any:
            del kwargs
            coefficients = self._image_as_cpx(x) if self.viewed_as_real else x
            if self.use_toeplitz and self._compact_toeplitz is not None:
                if self.toeplitz_kernel is None:
                    if self._compact_toeplitz == "cartesian-subspace":
                        (
                            self.toeplitz_kernel,
                            self._toeplitz_native_proxy,
                        ) = _build_cartesian_subspace_toeplitz(
                            self.frame_physics,
                            self.basis,
                            self._toeplitz_options,
                            self.streaming_policy,
                        )
                    elif self._compact_toeplitz == "subspace":
                        self.toeplitz_kernel = _build_subspace_toeplitz(
                            self.frame_physics,
                            self.basis,
                            self._toeplitz_options,
                            self.streaming_policy,
                        )
                    else:
                        (
                            self.toeplitz_kernel,
                            self._toeplitz_spatial_factors,
                        ) = _build_subspace_off_resonance_toeplitz(
                            self.frame_physics,
                            self.basis,
                            self._toeplitz_options,
                            self.streaming_policy,
                        )
                if self._compact_toeplitz in {
                    "cartesian-subspace",
                    "subspace",
                }:
                    selected_native = (
                        self._toeplitz_native_proxy
                        if self._compact_toeplitz == "cartesian-subspace"
                        else _frame_coil_view(self.frame_physics[0])
                    )
                    result = _apply_sense_toeplitz(
                        self.toeplitz_kernel,
                        coefficients,
                        selected_native,
                        coil_batch_size=self._toeplitz_options["coil_batch_size"],
                        streaming=self.streaming_policy,
                    )
                else:
                    result = _apply_subspace_off_resonance_toeplitz(
                        self.toeplitz_kernel,
                        coefficients,
                        _frame_coil_view(self.frame_physics[0]),
                        self._toeplitz_spatial_factors,
                        coefficient_rank=self.basis.shape[0],
                        coil_batch_size=self._toeplitz_options["coil_batch_size"],
                        streaming=self.streaming_policy,
                    )
                return self._image_as_real(result) if self.viewed_as_real else result
            if (
                self.streaming_policy is not None
                and isinstance(coefficients, torch.Tensor)
                and coefficients.device.type == "cpu"
            ):
                basis_cpu = self.basis.to(
                    device="cpu",
                    dtype=coefficients.dtype,
                )
                result = torch.zeros_like(
                    coefficients,
                    device="cpu",
                    pin_memory=self.streaming_policy.pin_memory,
                )
                for index, frame_physics_item in enumerate(self.frame_physics):
                    frame = (
                        basis_cpu[:, index]
                        .conj()
                        .reshape(1, -1, *([1] * (coefficients.ndim - 2)))
                        * coefficients
                    ).sum(dim=1, keepdim=True)
                    if frame_physics_item.viewed_as_real:
                        frame = self._image_as_real(frame)
                        normal = frame_physics_item.A_adjoint_A(frame)
                        normal = self._image_as_cpx(normal)
                    else:
                        normal = frame_physics_item.A_adjoint_A(frame[:, 0])[:, None]
                    result += basis_cpu[:, index].reshape(
                        1, -1, *([1] * (normal.ndim - 2))
                    ) * normal.to("cpu")
                return self._image_as_real(result) if self.viewed_as_real else result
            frames = self._expand(coefficients)
            normal_frames = []
            for index, physics in enumerate(self.frame_physics):
                frame = frames[:, index : index + 1]
                if physics.viewed_as_real:
                    frame = self._image_as_real(frame)
                    normal = physics.A_adjoint_A(frame)
                    normal = self._image_as_cpx(normal)
                else:
                    normal = physics.A_adjoint_A(frame[:, 0])[:, None]
                normal_frames.append(normal)
            result = self._project(torch.cat(normal_frames, dim=1))
            return self._image_as_real(result) if self.viewed_as_real else result

    return _SubspaceLinearPhysics()


def _subspace(
    physics: MRIPhysics,
    basis: Any,
    *,
    streaming: Any | None = None,
) -> MRIPhysics:
    """Decorate Cartesian or non-Cartesian physics with a low-rank subspace.

    ``basis`` has shape ``(rank, frames)``. Coefficient images use shape
    ``(batch, 2 * rank, *spatial)`` in the default real view, and measurements
    gain a frame dimension at axis 1. If a non-Cartesian trajectory has a
    leading frame dimension, one mri-nufft operator is built per frame.
    """
    if "subspace" in physics.modifiers:
        raise ValueError("physics already has a subspace decorator")
    shape = getattr(basis, "shape", ())
    if len(shape) != 2:
        raise ValueError("basis must have shape (rank, frames)")
    n_frames = int(shape[1])

    if streaming is None:
        streaming = physics.streaming_policy

    frame_physics: list[MRIPhysics | _LazyFramePhysics]
    flat_encoding: _FlatSubspaceEncoding | None = None
    trajectory = physics.trajectory
    trajectory_shape = getattr(trajectory, "shape", ())
    if (
        trajectory is not None
        and len(trajectory_shape) >= 3
        and trajectory_shape[0] == n_frames
    ):
        provider = _FramePhysicsProvider(physics, trajectory, streaming)
        frame_physics = [
            _LazyFramePhysics(provider, index) for index in range(n_frames)
        ]
        native = _base_fourier_operator(physics.native_operator)
        if (
            physics.kind.startswith("noncartesian")
            and "stacked" not in physics.modifiers
            and not physics.viewed_as_real
            and native is not None
            and not hasattr(native, "B")
        ):
            flat_encoding = _FlatSubspaceEncoding(physics, trajectory)
    else:
        frame_physics = [physics] * n_frames

    operator = _subspace_linear_physics(
        frame_physics,
        basis,
        viewed_as_real=physics.viewed_as_real,
        toeplitz_config=physics.toeplitz_options,
        flat_encoding=flat_encoding,
    )
    result = MRIPhysics(
        operator,
        native_operator=None,
        kind=physics.kind,
        spatial_ndim=physics.spatial_ndim,
        viewed_as_real=physics.viewed_as_real,
        modifiers=(*physics.modifiers, "subspace"),
        trajectory=trajectory,
        toeplitz_options=physics.toeplitz_options,
    )
    if streaming is not None:
        result.enable_streaming(streaming)
    return result


class Subspace(MRIPhysics):
    """Subspace encoding composed with frame-wise MRI physics.

    Solves for a small number of temporal coefficients instead of one image
    per frame, with the base physics applied to each expanded frame.

    Parameters
    ----------
    physics
        Base MRI physics, applied per frame.
    basis
        Temporal basis shaped ``(rank, frames)`` -- rank first. Rows are the
        retained singular vectors of a signal dictionary.
    **kwargs
        Forwarded to the base physics wrapper.

    Notes
    -----
    Coefficient images are native complex, ``(batch, rank, *image_shape)`` --
    one complex channel per retained coefficient.

    Examples
    --------
    >>> import torch
    >>> from pulserver.recon.physics import Cartesian2D, Subspace
    >>> base = Cartesian2D(
    ...     torch.ones(1, 1, 8, 8),
    ...     torch.ones(1, 2, 8, 8, dtype=torch.complex64) / 2 ** 0.5,
    ... )
    >>> physics = Subspace(base, torch.randn(3, 5, dtype=torch.complex64))
    >>> physics.A(torch.randn(1, 3, 8, 8, dtype=torch.complex64)).shape
    torch.Size([1, 5, 2, 8, 8])
    """

    def __init__(self, physics: MRIPhysics, basis: Any, **kwargs: Any) -> None:
        _init_from(self, _subspace(physics, basis, **kwargs))
