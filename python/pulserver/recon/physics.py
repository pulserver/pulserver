"""MRI physics factories with a uniform DeepInverse-facing interface.

The public factories own the mri-nufft/DeepInverse integration boundary.
Callers never need to construct an mri-nufft autodiff wrapper themselves.
Subspace, off-resonance, and Toeplitz behavior are composed as decorators so
that the API does not grow one class for every possible combination.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from importlib import import_module
from types import MethodType
from typing import Any

__all__ = [
    "Cartesian2D",
    "MRIPhysics",
    "NonCartesian2D",
    "NonCartesian3D",
    "OffResonance",
    "Subspace",
    "Toeplitz",
    "cartesian_2d",
    "noncartesian_2d",
    "noncartesian_3d",
    "off_resonance",
    "subspace",
    "toeplitz",
]


def _require_deepinv() -> Any:
    try:
        return import_module("deepinv.physics")
    except ImportError as error:
        raise ImportError(
            "MRI physics operators require DeepInverse; install "
            "pulserver[recon-cpu] or pulserver[recon-cuda]."
        ) from error


def _require_mrinufft() -> Any:
    try:
        return import_module("mrinufft")
    except ImportError as error:
        raise ImportError(
            "Non-Cartesian MRI physics requires mri-nufft; install "
            "pulserver[recon-cpu] or pulserver[recon-cuda]."
        ) from error


class MRIPhysics:
    """A small, transparent facade over a DeepInverse linear physics object.

    The facade records reconstruction-specific metadata and forwards the
    numerical contract (``A``, ``A_adjoint``, ``A_adjoint_A``, ``A_dagger``)
    to DeepInverse. ``native_operator`` is the underlying mri-nufft operator
    for non-Cartesian acquisitions and is intentionally read-only.
    """

    def __init__(
        self,
        operator: Any,
        *,
        native_operator: Any | None,
        kind: str,
        spatial_ndim: int,
        viewed_as_real: bool = True,
        modifiers: tuple[str, ...] = (),
        trajectory: Any | None = None,
        rebuild: Callable[[Any, int | None], MRIPhysics] | None = None,
    ) -> None:
        self.operator = operator
        self.native_operator = native_operator
        self.kind = kind
        self.spatial_ndim = spatial_ndim
        self.viewed_as_real = viewed_as_real
        self.modifiers = modifiers
        self.trajectory = trajectory
        self._rebuild = rebuild

    @property
    def normal_mode(self) -> str:
        """Return ``"toeplitz"``, ``"exact-fft"``, or ``"exact"``."""
        if "toeplitz" in self.modifiers:
            if self.kind == "cartesian2d":
                return "exact-fft"
            if getattr(self.operator, "use_toeplitz", False):
                return "toeplitz"
        return "exact"

    def A(self, x: Any, **kwargs: Any) -> Any:
        """Apply the forward encoding operator."""
        return self.operator.A(x, **kwargs)

    def A_adjoint(self, y: Any, **kwargs: Any) -> Any:
        """Apply the adjoint encoding operator."""
        return self.operator.A_adjoint(y, **kwargs)

    def A_adjoint_A(self, x: Any, **kwargs: Any) -> Any:
        """Apply the normal operator, using Toeplitz acceleration when valid."""
        return self.operator.A_adjoint_A(x, **kwargs)

    def A_vjp(self, x: Any, v: Any, **kwargs: Any) -> Any:
        """Return the vector-Jacobian product required by DeepInverse."""
        method = getattr(self.operator, "A_vjp", None)
        if method is not None:
            return method(x, v, **kwargs)
        return self.A_adjoint(v, **kwargs)

    def A_dagger(self, y: Any, **kwargs: Any) -> Any:
        """Apply the least-squares pseudo-inverse supplied by DeepInverse."""
        return self.operator.A_dagger(y, **kwargs)

    def __call__(self, x: Any, **kwargs: Any) -> Any:
        return self.A(x, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.operator, name)

    def to(self, *args: Any, **kwargs: Any) -> MRIPhysics:
        """Move the underlying Torch physics object and return ``self``."""
        self.operator.to(*args, **kwargs)
        return self

    def rebuild(
        self,
        trajectory: Any,
        frame_index: int | None = None,
    ) -> MRIPhysics:
        """Rebuild a non-Cartesian factory for a frame-specific trajectory."""
        if self._rebuild is None:
            return self
        return self._rebuild(trajectory, frame_index)


def _native_linear_physics(
    native_operator: Any,
    *,
    viewed_as_real: bool,
    use_toeplitz: bool = False,
    native_toeplitz_valid: bool = True,
) -> Any:
    """Adapt an mri-nufft operator to DeepInverse without exposing glue."""
    physics_module = _require_deepinv()
    try:
        conversions = import_module("mrinufft.operators.autodiff")
    except ImportError as error:
        raise ImportError(
            "The mri-nufft DeepInverse adapter requires Torch and DeepInverse."
        ) from error

    # Prefer mri-nufft's maintained DeepInverse/autograd interface. The custom
    # adapter below is only needed for compositions (notably stacked NUFFT)
    # that correctly implement forward/adjoint operations but do not expose
    # an autograd wrapper. PICS itself uses explicit adjoints, so both paths
    # have identical reconstruction semantics.
    if getattr(native_operator, "autograd_available", False):
        result = native_operator.make_deepinv_phy(viewed_as_real=viewed_as_real)
        result.use_toeplitz = use_toeplitz and native_toeplitz_valid
        if result.use_toeplitz:

            def toeplitz_normal(self: Any, x: Any, **kwargs: Any) -> Any:
                del kwargs
                if self.viewed_as_real:
                    x = conversions.image_as_cpx(x)
                normal = native_operator.gram_op(x, toeplitz=True)
                return (
                    conversions.image_as_real(normal) if self.viewed_as_real else normal
                )

            result.A_adjoint_A = MethodType(toeplitz_normal, result)
        return result

    class _MRIThinPhysics(physics_module.LinearPhysics):
        def __init__(self) -> None:
            super().__init__()
            self.__dict__["native_operator"] = native_operator
            self.viewed_as_real = viewed_as_real
            self.use_toeplitz = use_toeplitz and native_toeplitz_valid

        def A(self, x: Any, **kwargs: Any) -> Any:
            del kwargs
            if self.viewed_as_real:
                x = conversions.image_as_cpx(x)
            result = self.native_operator.op(x)
            return conversions.kspace_as_real(result) if self.viewed_as_real else result

        def A_adjoint(self, y: Any, **kwargs: Any) -> Any:
            del kwargs
            if self.viewed_as_real:
                y = conversions.kspace_as_cpx(y)
            result = self.native_operator.adj_op(y)
            return conversions.image_as_real(result) if self.viewed_as_real else result

        def A_adjoint_A(self, x: Any, **kwargs: Any) -> Any:
            del kwargs
            if not self.use_toeplitz:
                return self.A_adjoint(self.A(x))
            if self.viewed_as_real:
                x = conversions.image_as_cpx(x)
            result = self.native_operator.gram_op(x, toeplitz=True)
            return conversions.image_as_real(result) if self.viewed_as_real else result

    # Keep the explicit reference so static analyzers and mocked test modules
    # both validate that this is a DeepInverse physics implementation.
    assert issubclass(_MRIThinPhysics, physics_module.LinearPhysics)
    return _MRIThinPhysics()


def cartesian_2d(
    mask: Any,
    coil_maps: Any,
    *,
    toeplitz: bool = False,
    **kwargs: Any,
) -> MRIPhysics:
    """Create 2D Cartesian SENSE physics.

    Leading dimensions are handled by DeepInverse as batch dimensions, so
    slices, contrasts, and dynamic frames are reconstructed independently.
    Cartesian normal operations already use exact FFTs; ``toeplitz=True`` is
    accepted for API symmetry and reports ``normal_mode == "exact-fft"``.
    """
    physics_module = _require_deepinv()
    device = getattr(coil_maps, "device", getattr(mask, "device", "cpu"))
    operator = physics_module.MultiCoilMRI(
        mask=mask,
        coil_maps=coil_maps,
        three_d=False,
        device=device,
        **kwargs,
    )
    result = MRIPhysics(
        operator,
        native_operator=None,
        kind="cartesian2d",
        spatial_ndim=2,
        modifiers=("toeplitz",) if toeplitz else (),
    )
    return result


Cartesian2D = cartesian_2d


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
    toeplitz: bool,
    viewed_as_real: bool,
    operator_kwargs: dict[str, Any],
) -> MRIPhysics:
    if len(image_shape) != spatial_ndim:
        raise ValueError(
            f"image_shape must have {spatial_ndim} entries, got {image_shape!r}"
        )
    trajectory_dim = getattr(trajectory, "shape", (None,))[-1]
    if trajectory_dim is not None and trajectory_dim != spatial_ndim:
        raise ValueError(
            f"trajectory must end in {spatial_ndim} coordinates, got {trajectory_dim}"
        )
    if stacked and spatial_ndim != 3:
        raise ValueError("stacked trajectories are only supported by NonCartesian3D")

    mrinufft = _require_mrinufft()
    common = {
        "samples": trajectory,
        "shape": image_shape,
        "smaps": coil_maps,
        "n_coils": n_coils,
        "n_batchs": n_batchs,
        "squeeze_dims": False,
        **operator_kwargs,
    }
    if stacked:
        native = mrinufft.get_operator("stacked")(
            backend=backend,
            z_index=z_index,
            **common,
        )
    else:
        native = mrinufft.get_operator(backend)(density=density, **common)

    # mri-nufft's generic Toeplitz Gram implementation is valid for the base
    # NUFFT. Stacked FFT/NUFFT composition currently has no equivalent native
    # kernel, so it retains the exact normal operation.
    native_toeplitz_valid = not stacked
    operator = _native_linear_physics(
        native,
        viewed_as_real=viewed_as_real,
        use_toeplitz=toeplitz,
        native_toeplitz_valid=native_toeplitz_valid,
    )

    def rebuild(
        new_trajectory: Any,
        frame_index: int | None = None,
    ) -> MRIPhysics:
        frame_density = density
        density_shape = getattr(density, "shape", ())
        trajectory_shape = getattr(trajectory, "shape", ())
        if (
            frame_index is not None
            and density is not None
            and len(density_shape) > 1
            and trajectory_shape
            and density_shape[0] == trajectory_shape[0]
        ):
            frame_density = density[frame_index]
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
            operator_kwargs=operator_kwargs,
        )

    return MRIPhysics(
        operator,
        native_operator=native,
        kind=f"noncartesian{spatial_ndim}d",
        spatial_ndim=spatial_ndim,
        viewed_as_real=viewed_as_real,
        modifiers=(("stacked",) if stacked else ())
        + (("toeplitz",) if toeplitz else ()),
        trajectory=trajectory,
        rebuild=rebuild,
    )


def noncartesian_2d(
    trajectory: Any,
    image_shape: tuple[int, int],
    *,
    coil_maps: Any | None = None,
    density: Any | None = None,
    backend: str = "finufft",
    n_coils: int = 1,
    n_batchs: int = 1,
    toeplitz: bool = False,
    viewed_as_real: bool = True,
    **kwargs: Any,
) -> MRIPhysics:
    """Create 2D non-Cartesian mri-nufft physics."""
    result = _noncartesian(
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
        operator_kwargs=kwargs,
    )
    return Toeplitz(result) if toeplitz else result


NonCartesian2D = noncartesian_2d


def noncartesian_3d(
    trajectory: Any,
    image_shape: tuple[int, int, int],
    *,
    coil_maps: Any | None = None,
    density: Any | None = None,
    backend: str = "finufft",
    n_coils: int = 1,
    n_batchs: int = 1,
    stacked: bool = False,
    z_index: Any = "auto",
    toeplitz: bool = False,
    viewed_as_real: bool = True,
    **kwargs: Any,
) -> MRIPhysics:
    """Create full-3D or stack-of-trajectories mri-nufft physics."""
    result = _noncartesian(
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
        operator_kwargs=kwargs,
    )
    return Toeplitz(result) if toeplitz else result


NonCartesian3D = noncartesian_3d


def _subspace_linear_physics(
    frame_physics: Sequence[MRIPhysics],
    basis: Any,
    *,
    viewed_as_real: bool,
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
            self.__dict__["basis"] = torch.as_tensor(basis)
            self.viewed_as_real = viewed_as_real
            self.use_toeplitz = any(
                item.normal_mode == "toeplitz" for item in frame_physics
            )

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
            frames = self._expand(coefficients)
            measurements = []
            for index, physics in enumerate(self.frame_physics):
                frame = frames[:, index : index + 1]
                if physics.viewed_as_real:
                    frame = self._image_as_real(frame)
                measurements.append(physics.A(frame))
            return torch.stack(measurements, dim=1)

        def A_adjoint(self, y: Any, **kwargs: Any) -> Any:
            del kwargs
            frames = []
            for index, physics in enumerate(self.frame_physics):
                frame = physics.A_adjoint(y[:, index])
                if physics.viewed_as_real:
                    frame = self._image_as_cpx(frame)
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
            frames = self._expand(coefficients)
            normal_frames = []
            for index, physics in enumerate(self.frame_physics):
                frame = frames[:, index : index + 1]
                if physics.viewed_as_real:
                    frame = self._image_as_real(frame)
                normal = physics.A_adjoint_A(frame)
                if physics.viewed_as_real:
                    normal = self._image_as_cpx(normal)
                normal_frames.append(normal)
            result = self._project(torch.cat(normal_frames, dim=1))
            return self._image_as_real(result) if self.viewed_as_real else result

    return _SubspaceLinearPhysics()


def subspace(physics: MRIPhysics, basis: Any) -> MRIPhysics:
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

    frame_physics: list[MRIPhysics]
    trajectory = physics.trajectory
    trajectory_shape = getattr(trajectory, "shape", ())
    if (
        trajectory is not None
        and len(trajectory_shape) >= 3
        and trajectory_shape[0] == n_frames
    ):
        frame_physics = [
            physics.rebuild(trajectory[index], index) for index in range(n_frames)
        ]
    else:
        frame_physics = [physics] * n_frames

    operator = _subspace_linear_physics(
        frame_physics,
        basis,
        viewed_as_real=physics.viewed_as_real,
    )
    return MRIPhysics(
        operator,
        native_operator=None,
        kind=physics.kind,
        spatial_ndim=physics.spatial_ndim,
        viewed_as_real=physics.viewed_as_real,
        modifiers=(*physics.modifiers, "subspace"),
        trajectory=trajectory,
    )


Subspace = subspace


def off_resonance(
    physics: MRIPhysics,
    field_map: Any,
    readout_time: Any,
    *,
    r2star_map: Any | None = None,
    mask: Any | None = None,
    interpolator: str | dict[str, Any] | tuple[Any, Any] = "svd",
) -> MRIPhysics:
    """Decorate non-Cartesian physics with mri-nufft off-resonance correction."""
    if physics.native_operator is None:
        raise TypeError("OffResonance requires base non-Cartesian physics")
    if "subspace" in physics.modifiers:
        raise ValueError(
            "Apply OffResonance before Subspace so field correction occurs "
            "in every acquired frame."
        )
    if "off_resonance" in physics.modifiers:
        raise ValueError("physics already has an off-resonance decorator")
    try:
        corrected_class = import_module("mrinufft.operators").MRIFourierCorrected
    except ImportError as error:
        raise ImportError("Off-resonance physics requires mri-nufft.") from error

    native = corrected_class(
        physics.native_operator,
        b0_map=field_map,
        readout_time=readout_time,
        r2star_map=r2star_map,
        mask=mask,
        interpolator=interpolator,
    )
    operator = _native_linear_physics(
        native,
        viewed_as_real=physics.viewed_as_real,
        use_toeplitz=False,
        native_toeplitz_valid=False,
    )

    def rebuild(
        new_trajectory: Any,
        frame_index: int | None = None,
    ) -> MRIPhysics:
        frame_readout_time = readout_time
        time_shape = getattr(readout_time, "shape", ())
        trajectory_shape = getattr(physics.trajectory, "shape", ())
        if (
            frame_index is not None
            and len(time_shape) > 1
            and trajectory_shape
            and time_shape[0] == trajectory_shape[0]
        ):
            frame_readout_time = readout_time[frame_index]
        return off_resonance(
            physics.rebuild(new_trajectory, frame_index),
            field_map,
            frame_readout_time,
            r2star_map=r2star_map,
            mask=mask,
            interpolator=interpolator,
        )

    return MRIPhysics(
        operator,
        native_operator=native,
        kind=physics.kind,
        spatial_ndim=physics.spatial_ndim,
        viewed_as_real=physics.viewed_as_real,
        modifiers=(*physics.modifiers, "off_resonance"),
        trajectory=physics.trajectory,
        rebuild=rebuild,
    )


OffResonance = off_resonance


def toeplitz(physics: MRIPhysics) -> MRIPhysics:
    """Enable a Toeplitz normal operator wherever the backend supports it.

    Base 2D/3D mri-nufft operators use their lazily computed Toeplitz kernel.
    Cartesian FFTs are already exact and fast. Stacked and off-resonance
    compositions retain an exact ``AᴴA`` because mri-nufft does not currently
    expose a valid combined Toeplitz kernel for those operators.
    """
    if "toeplitz" not in physics.modifiers:
        physics.modifiers = (*physics.modifiers, "toeplitz")
    if (
        physics.native_operator is not None
        and "stacked" not in physics.modifiers
        and "off_resonance" not in physics.modifiers
    ):
        physics.operator.use_toeplitz = True
    return physics


Toeplitz = toeplitz
