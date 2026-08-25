"""One operator per frame, or one operator retargeted to all of them.

A dynamic scan acquires each frame on its own trajectory. Frames that share a
grid and a sample count share a plan, which is retargeted frame by frame; a
ragged acquisition cannot, and falls back to one operator per frame."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from importlib import import_module
from math import prod, sqrt
from types import SimpleNamespace
from typing import Any


from .._views import image_as_cpx as _image_as_cpx
from .._views import image_as_real as _image_as_real
from .._views import kspace_as_cpx as _kspace_as_cpx
from .._views import kspace_as_real as _kspace_as_real

from ._base import MRIPhysics
from ._common import _base_fourier_operator, _require_deepinv, _require_mrinufft


class _FramePhysicsProvider:
    """Small LRU of frame-specific native operators for streamed dynamics."""

    def __init__(
        self,
        physics: MRIPhysics,
        trajectory: Any,
        policy: Any,
    ) -> None:
        self.physics = physics
        self.trajectory = trajectory
        self.policy = policy
        self.cache: OrderedDict[int, MRIPhysics] = OrderedDict()
        self.toeplitz_options = physics.toeplitz_options
        self.shared: MRIPhysics | None = None
        self.target: int | None = None
        native = _base_fourier_operator(physics.native_operator)
        self.has_density = getattr(native, "density", None) is not None
        # A ragged acquisition has to keep a plan per frame: a NUFFT is planned
        # for a fixed number of points.
        self.shareable = (
            hasattr(native, "update_samples")
            and len(
                {int(prod(getattr(frame, "shape", (0,))[:-1])) for frame in trajectory}
            )
            == 1
        )

    def samples(self, index: int) -> Any:
        """One frame's sample set, without building the plan that reads it."""
        _require_mrinufft()
        return import_module("mrinufft._utils").proper_trajectory(
            self.trajectory[index],
            normalize="pi",
        )

    def density(self, index: int) -> Any:
        """One frame's sample weights, without building its plan."""
        native = _base_fourier_operator(self.physics.native_operator)
        return _frame_density(
            getattr(native, "density", None),
            self.trajectory,
            index,
            prod(getattr(self.trajectory, "shape", (0, 0))[1:-1]),
        )

    def _build(self, index: int) -> MRIPhysics:
        # The kernel is built over the frames this provider serves, so it
        # imports from here; the import lives in the call.
        from ._kernel import _enable_toeplitz

        result = self.physics.rebuild(self.trajectory[index], index)
        if self.policy is not None:
            result.enable_streaming(self.policy)
        if self.toeplitz_options is not None:
            _enable_toeplitz(result, **self.toeplitz_options)
        return result

    def get(self, index: int) -> MRIPhysics:
        """The physics for one frame, planned once and retargeted after that.

        Frames of a dynamic acquisition differ only in where their samples
        fall, so they share the plan -- by far the most expensive part of a
        frame, and on CUDA the part that holds device memory for as long as
        the physics lives. Callers use one frame at a time.
        """
        if self.shared is not None:
            if self.target != index:
                native = _base_fourier_operator(self.shared.native_operator)
                native.update_samples(self.samples(index))
                if self.has_density:
                    native.density = self.density(index)
                self.target = index
            return self.shared
        if index in self.cache:
            result = self.cache.pop(index)
            self.cache[index] = result
            return result
        result = self._build(index)
        if self.shareable:
            self.shared, self.target = result, index
            return result
        self.cache[index] = result
        limit = 1 if self.policy is None else self.policy.frame_cache_size
        while len(self.cache) > limit:
            self.cache.popitem(last=False)
        return result


class _LazyFramePhysics:
    """Frame facade resolving through a shared bounded LRU."""

    def __init__(self, provider: _FramePhysicsProvider, index: int) -> None:
        self.provider = provider
        self.index = index
        self.kind = provider.physics.kind
        self.viewed_as_real = provider.physics.viewed_as_real
        self.modifiers = provider.physics.modifiers
        self.spatial_ndim = provider.physics.spatial_ndim

    @property
    def normal_mode(self) -> str:
        if "toeplitz" in self.modifiers:
            return "exact-fft" if self.kind.startswith("cartesian") else "toeplitz"
        return self.provider.physics.normal_mode

    @property
    def native_operator(self) -> Any:
        return self.provider.get(self.index).native_operator

    def A(self, value: Any) -> Any:
        return self.provider.get(self.index).A(value)

    def A_adjoint(self, value: Any) -> Any:
        return self.provider.get(self.index).A_adjoint(value)

    def A_adjoint_A(self, value: Any) -> Any:
        return self.provider.get(self.index).A_adjoint_A(value)

    def enable_streaming(self, policy: Any) -> None:
        self.provider.policy = policy

    def enable_toeplitz(self, options: dict[str, Any]) -> None:
        self.provider.toeplitz_options = options
        self.modifiers = tuple(dict.fromkeys((*self.modifiers, "toeplitz")))

    @property
    def samples(self) -> Any:
        """This frame's trajectory, in the units a NUFFT plans on."""
        return self.provider.samples(self.index)

    @property
    def density(self) -> Any:
        """This frame's sample weights, if the acquisition carries any."""
        return self.provider.density(self.index)

    @property
    def backend(self) -> str:
        """The NUFFT backend the frames are planned on."""
        reference = _base_fourier_operator(self.provider.physics.native_operator)
        return getattr(reference, "backend", "finufft")

    @property
    def image_shape(self) -> tuple[int, ...]:
        """The image grid every frame shares."""
        reference = _base_fourier_operator(self.provider.physics.native_operator)
        return tuple(int(size) for size in reference.shape)

    @property
    def coil_view(self) -> Any:
        """The coils and the grid, without a plan built to read them off.

        Applying a kernel needs the sensitivities and the image shape, and the
        frames share both with the acquisition they came from.
        """
        reference = _base_fourier_operator(self.provider.physics.native_operator)
        return SimpleNamespace(
            shape=tuple(int(size) for size in reference.shape),
            smaps=getattr(reference, "smaps", None),
            uses_sense=getattr(reference, "uses_sense", False),
            n_coils=int(getattr(reference, "n_coils", 1) or 1),
        )


def _plan_batch_width(native_operator: Any, batch: int) -> int:
    """How many images an mri-nufft operator can be pointed at in one call.

    ``n_batchs`` only tells the operator how to fold its input; the plan is
    sized by ``n_trans``, which has to divide the transforms a call asks for.
    """
    trans = int(getattr(native_operator, "n_trans", 1) or 1)
    coils = int(getattr(native_operator, "n_coils", 1) or 1)
    if trans == 1 or (batch * coils) % trans == 0:
        return batch
    return int(getattr(native_operator, "n_batchs", 1) or 1)


@contextmanager
def _batches_of(native_operator: Any, width: int) -> Any:
    """The operator seen as folding its input into ``width`` images."""
    held = getattr(native_operator, "n_batchs", None)
    if held is None or int(held) == width:
        yield
        return
    native_operator.n_batchs = width
    try:
        yield
    finally:
        native_operator.n_batchs = held


def _over_batches(apply: Any, native_operator: Any, value: Any) -> Any:
    """Apply an operator to any number of images, whatever it plans for.

    The leading axis is the batch on both sides of a NUFFT, so a call the plan
    cannot take in one go is served in groups of the width it can, the last of
    them padded up and cut back.
    """
    torch = import_module("torch")
    batch = int(value.shape[0])
    width = _plan_batch_width(native_operator, batch)
    if width == batch:
        with _batches_of(native_operator, batch):
            return apply(value)
    results = []
    for start in range(0, batch, width):
        group = value[start : start + width]
        short = width - int(group.shape[0])
        if short:
            group = torch.cat((group, group[-1:].expand(short, *group.shape[1:])), 0)
        with _batches_of(native_operator, width):
            outcome = apply(group)
        results.append(outcome[: width - short] if short else outcome)
    return torch.cat(results, 0)


def _native_linear_physics(
    native_operator: Any,
    *,
    viewed_as_real: bool,
) -> Any:
    """Adapt an mri-nufft operator to DeepInverse without exposing glue."""
    physics_module = _require_deepinv()
    try:
        import_module("mrinufft.operators.autodiff")
    except ImportError as error:
        raise ImportError(
            "The mri-nufft DeepInverse adapter requires Torch and DeepInverse."
        ) from error

    # Prefer mri-nufft's maintained DeepInverse/autograd interface. The custom
    # adapter below is only needed for third-party operators that correctly
    # implement forward/adjoint operations but do not expose an autograd
    # wrapper. PICS itself uses explicit adjoints, so both paths have identical
    # reconstruction semantics.
    if getattr(native_operator, "autograd_available", False):
        inner = native_operator.make_deepinv_phy()

        class _MRIViewPhysics(physics_module.LinearPhysics):
            def __init__(self) -> None:
                super().__init__()
                self.__dict__["inner"] = inner
                self.viewed_as_real = viewed_as_real
                self.use_toeplitz = False

            def A(self, x: Any, **kwargs: Any) -> Any:
                if self.viewed_as_real:
                    x = _image_as_cpx(x)
                result = _over_batches(
                    lambda value: self.inner.A(value, **kwargs),
                    native_operator,
                    x,
                )
                return _kspace_as_real(result) if self.viewed_as_real else result

            def A_adjoint(self, y: Any, **kwargs: Any) -> Any:
                if self.viewed_as_real:
                    y = _kspace_as_cpx(y)
                result = _over_batches(
                    lambda value: self.inner.A_adjoint(value, **kwargs),
                    native_operator,
                    y,
                )
                return _image_as_real(result) if self.viewed_as_real else result

            def A_dagger(self, y: Any, **kwargs: Any) -> Any:
                if self.viewed_as_real:
                    y = _kspace_as_cpx(y)
                result = _over_batches(
                    lambda value: self.inner.A_dagger(value, **kwargs),
                    native_operator,
                    y,
                )
                return _image_as_real(result) if self.viewed_as_real else result

            def A_adjoint_A(self, x: Any, **kwargs: Any) -> Any:
                del kwargs
                return self.A_adjoint(self.A(x))

        return _MRIViewPhysics()

    class _MRIThinPhysics(physics_module.LinearPhysics):
        def __init__(self) -> None:
            super().__init__()
            self.__dict__["native_operator"] = native_operator
            self.viewed_as_real = viewed_as_real
            self.use_toeplitz = False

        def A(self, x: Any, **kwargs: Any) -> Any:
            del kwargs
            if self.viewed_as_real:
                x = _image_as_cpx(x)
            result = _over_batches(self.native_operator.op, self.native_operator, x)
            return _kspace_as_real(result) if self.viewed_as_real else result

        def A_adjoint(self, y: Any, **kwargs: Any) -> Any:
            del kwargs
            if self.viewed_as_real:
                y = _kspace_as_cpx(y)
            result = _over_batches(self.native_operator.adj_op, self.native_operator, y)
            return _image_as_real(result) if self.viewed_as_real else result

        def A_adjoint_A(self, x: Any, **kwargs: Any) -> Any:
            del kwargs
            return self.A_adjoint(self.A(x))

    # Keep the explicit reference so static analyzers and mocked test modules
    # both validate that this is a DeepInverse physics implementation.
    assert issubclass(_MRIThinPhysics, physics_module.LinearPhysics)
    return _MRIThinPhysics()


def _mrinufft_norm_factor(shape: tuple[int, ...]) -> float:
    """The normalization an mri-nufft operator on ``shape`` divides by."""
    return sqrt(prod(shape) * 2 ** len(shape))


def _frame_density(
    density: Any,
    trajectory: Any,
    frame_index: int | None,
    frame_samples: int,
) -> Any:
    """One frame's share of a density given for a whole dynamic trajectory.

    A density may be given per frame, flat over every sample, or once for a
    trajectory every frame shares; only the first two are split.
    """
    if frame_index is None or density is None:
        return density
    density_shape = getattr(density, "shape", ())
    trajectory_shape = getattr(trajectory, "shape", ())
    if not trajectory_shape:
        return density
    if len(density_shape) > 1 and density_shape[0] == trajectory_shape[0]:
        return density[frame_index]
    if prod(density_shape) == trajectory_shape[0] * frame_samples:
        start = frame_index * frame_samples
        return density.reshape(-1)[start : start + frame_samples]
    return density
