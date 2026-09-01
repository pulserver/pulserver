"""Physics that carries the field the spins actually precess in.

A field map turns a readout into a sum of segments, each a plain encoding at
its own frequency, weighted by how much of the readout it explains."""

from __future__ import annotations

from functools import cache
from importlib import import_module
from math import prod
from types import MethodType
from typing import Any


from .._views import image_as_cpx as _image_as_cpx
from .._views import image_as_real as _image_as_real

from ._base import MRIPhysics, _init_from
from ._common import _toeplitz_options
from ._frames import _native_linear_physics
from ._kernel import _apply_sense_toeplitz, _build_off_resonance_toeplitz


def _configure_off_resonance_toeplitz(
    operator: Any,
    corrected_operator: Any,
    *,
    enabled: bool,
    options: dict[str, Any],
) -> Any:
    """Install a lazy segmented Toeplitz normal on a DeepInverse adapter."""
    operator.use_toeplitz = enabled
    operator.toeplitz_kernel = None
    operator._toeplitz_options = dict(options)
    operator.streaming_policy = None
    operator.streaming_methods = {"A_adjoint_A"}

    def enable_toeplitz(self: Any, new_options: dict[str, Any]) -> None:
        self.use_toeplitz = True
        self._toeplitz_options = dict(new_options)
        self.toeplitz_kernel = None

    def enable_streaming(self: Any, policy: Any) -> None:
        self.streaming_policy = policy

    def segmented_normal(self: Any, x: Any, **kwargs: Any) -> Any:
        del kwargs
        if not self.use_toeplitz:
            return self.A_adjoint(self.A(x))
        if self.viewed_as_real:
            x = _image_as_cpx(x)
        if self.toeplitz_kernel is None:
            self.toeplitz_kernel, spatial_factors = _build_off_resonance_toeplitz(
                corrected_operator,
                self._toeplitz_options,
                self.streaming_policy,
            )
            self._toeplitz_spatial_factors = spatial_factors
        result = _apply_sense_toeplitz(
            self.toeplitz_kernel,
            x,
            corrected_operator,
            right_factors=self._toeplitz_spatial_factors,
            left_factors=self._toeplitz_spatial_factors,
            coil_batch_size=self._toeplitz_options["coil_batch_size"],
            streaming=self.streaming_policy,
        )
        return _image_as_real(result) if self.viewed_as_real else result

    operator.enable_toeplitz = MethodType(enable_toeplitz, operator)
    operator.enable_streaming = MethodType(enable_streaming, operator)
    operator.A_adjoint_A = MethodType(segmented_normal, operator)
    return operator


def _host_array(value: Any) -> Any:
    """A field map or a readout clock as contiguous NumPy, for the fit to run on."""
    if value is None:
        return None
    numpy = import_module("numpy")
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach().cpu().numpy()
    return numpy.ascontiguousarray(value)


def _contiguous_basis(native: Any) -> None:
    """Lay the fitted temporal basis out so a Torch tensor can wrap it.

    The partial decomposition answers descending singular order as a reversed
    view, and a tensor cannot be made from an array with a negative stride.
    """
    numpy = import_module("numpy")
    basis = getattr(native, "B", None)
    if basis is not None:
        native.B = numpy.ascontiguousarray(basis)


@cache
def _torch_corrected_class() -> Any:
    """mri-nufft's off-resonance operator, summed in Torch.

    The correction itself is upstream's -- the field model, the interpolators
    fitted from it, the segment count. What is replaced is the two loops that
    add the segments up, because upstream writes them for NumPy and CuPy and
    reaches for CuPy to put them on a GPU. Ours is one Torch operator over
    Torch factors, so the sum runs wherever the samples already are and the
    package needs no second array library to use a card.
    """
    corrected_class = import_module("mrinufft.operators").MRIFourierCorrected
    numpy = import_module("numpy")
    torch = import_module("torch")

    class TorchFourierCorrected(corrected_class):  # type: ignore[misc,valid-type]
        """Off-resonance correction whose segment sum is Torch throughout."""

        def to_torch(self, device: Any) -> None:
            """Hold the interpolators as Torch tensors on ``device``."""

            def held(factor: Any) -> Any:
                if not isinstance(factor, torch.Tensor):
                    factor = torch.as_tensor(numpy.asarray(factor))
                factor = factor.to(torch.complex64, copy=False)
                return factor if device is None else factor.to(device)

            self.B, self.C = held(self.B), held(self.C)

        def _staged(self, factor: Any, like: Any) -> Any:
            return factor.to(device=like.device, dtype=like.dtype)

        def op(self, data: Any, *args: Any) -> Any:
            """Distorted k-space: the segments, weighted and summed."""
            data = torch.as_tensor(data)
            batches, coils = self.n_batchs, self.n_coils
            shots, per_shot = self.n_shots, self.n_samples_per_shot
            kspace = None
            for segment in range(self.n_interpolators):
                spatial = self._staged(self.C[segment], data)
                partial = self._fourier_op.op(spatial * data, *args)
                partial = partial.reshape(batches, coils, shots, per_shot)
                temporal = self._staged(self.B[:, segment], partial)
                weighted = (temporal * partial).reshape(batches, coils, self.n_samples)
                kspace = weighted if kspace is None else kspace + weighted
            return self._safe_squeeze(kspace)

        def adj_op(self, coeffs: Any, *args: Any) -> Any:
            """The adjoint of that sum, segment by conjugated segment."""
            coeffs = torch.as_tensor(coeffs)
            batches, coils = self.n_batchs, self.n_coils
            shots, per_shot = self.n_shots, self.n_samples_per_shot
            folded = coeffs.reshape(batches, coils, shots, per_shot)
            image = None
            for segment in range(self.n_interpolators):
                temporal = self._staged(self.B[:, segment], folded).conj()
                weighted = (temporal * folded).reshape(batches, coils, self.n_samples)
                partial = self._fourier_op.adj_op(weighted, *args)
                spatial = self._staged(self.C[segment], partial).conj()
                contribution = spatial * partial
                image = contribution if image is None else image + contribution
            return self._safe_squeeze(image)

    return TorchFourierCorrected


def _off_resonance(
    physics: MRIPhysics,
    field_map: Any,
    readout_time: Any,
    *,
    r2star_map: Any | None = None,
    mask: Any | None = None,
    interpolator: str | dict[str, Any] | tuple[Any, Any] = "svd",
) -> MRIPhysics:
    """Decorate non-Cartesian physics with mri-nufft off-resonance correction.

    The field model keeps only the handful of components the correction uses,
    so it is fitted by the partial decomposition: the matrix it factors has one
    row per readout sample and one column per field bin, and asking for all of
    its singular vectors when eight are wanted costs thirty times as much. The
    partial routine answers descending order as a reversed view, which a Torch
    tensor cannot wrap, so the basis is made contiguous before it is handed
    over.
    """
    if physics.native_operator is None:
        raise TypeError("OffResonance requires base non-Cartesian physics")
    if "stacked" in physics.modifiers:
        raise ValueError(
            "OffResonance for stack-of-NUFFTs needs a stack-frequency field "
            "model and is not yet a valid composition."
        )
    if "subspace" in physics.modifiers:
        raise ValueError(
            "Apply OffResonance before Subspace so field correction occurs "
            "in every acquired frame."
        )
    if "off_resonance" in physics.modifiers:
        raise ValueError("physics already has an off-resonance decorator")
    try:
        _torch_corrected_class()
    except ImportError as error:
        raise ImportError("Off-resonance physics requires mri-nufft.") from error

    corrected_interpolator = interpolator
    if isinstance(interpolator, str):
        corrected_interpolator = {"name": interpolator, "partial_svd": True}
    elif isinstance(interpolator, dict):
        corrected_interpolator = {"partial_svd": True, **interpolator}
    corrected_readout_time = readout_time
    trajectory_shape = getattr(physics.trajectory, "shape", ())
    time_shape = getattr(readout_time, "shape", ())
    frame_samples = prod(trajectory_shape[1:-1]) if len(trajectory_shape) >= 4 else 0
    dynamic_readout = bool(
        frame_samples
        and time_shape
        and prod(time_shape) != frame_samples
        and (
            time_shape[0] == trajectory_shape[0]
            or prod(time_shape) == trajectory_shape[0] * frame_samples
        )
    )
    if physics.streaming_policy is not None and dynamic_readout:
        corrected_readout_time = (
            readout_time[0]
            if time_shape[0] == trajectory_shape[0]
            else readout_time.reshape(-1)[:frame_samples]
        )
    if isinstance(interpolator, tuple):
        # mri-nufft's array-interface decorator currently turns nested tuples
        # into lists before MRIFourierCorrected can recognize its documented
        # ``(B, C)`` input. A callable preserves the supplied arrays and also
        # lets frame rebuilds reuse one spatial factor bank by identity.
        temporal_factors, spatial_factors = interpolator
        temporal_shape = getattr(temporal_factors, "shape", ())
        if (
            physics.streaming_policy is not None
            and len(trajectory_shape) >= 4
            and temporal_shape
            and temporal_shape[0] == trajectory_shape[0]
        ):
            temporal_factors = temporal_factors[0]
        elif (
            physics.streaming_policy is not None
            and frame_samples
            and temporal_shape
            and temporal_shape[0] == trajectory_shape[0] * frame_samples
        ):
            temporal_factors = temporal_factors[:frame_samples]

        def supplied_interpolator(**_kwargs: Any) -> tuple[Any, Any, None]:
            return temporal_factors, spatial_factors, None

        corrected_interpolator = supplied_interpolator

    native = _torch_corrected_class()(
        physics.native_operator,
        b0_map=_host_array(field_map),
        readout_time=_host_array(corrected_readout_time),
        r2star_map=_host_array(r2star_map),
        mask=_host_array(mask),
        interpolator=corrected_interpolator,
    )
    _contiguous_basis(native)
    native.to_torch(getattr(physics.native_operator, "device", None))
    toeplitz_enabled = "toeplitz" in physics.modifiers
    options = physics.toeplitz_options or _toeplitz_options()
    operator = _native_linear_physics(
        native,
        viewed_as_real=physics.viewed_as_real,
    )
    operator = _configure_off_resonance_toeplitz(
        operator,
        native,
        enabled=toeplitz_enabled,
        options=options,
    )

    def rebuild(
        new_trajectory: Any,
        frame_index: int | None = None,
    ) -> MRIPhysics:
        frame_readout_time = readout_time
        frame_interpolator = interpolator
        time_shape = getattr(readout_time, "shape", ())
        trajectory_shape = getattr(physics.trajectory, "shape", ())
        frame_samples = prod(getattr(new_trajectory, "shape", (0,))[:-1])
        if (
            frame_index is not None
            and len(time_shape) > 1
            and trajectory_shape
            and time_shape[0] == trajectory_shape[0]
        ):
            frame_readout_time = readout_time[frame_index]
        elif (
            frame_index is not None
            and time_shape
            and trajectory_shape
            and prod(time_shape) == trajectory_shape[0] * frame_samples
        ):
            start = frame_index * frame_samples
            frame_readout_time = readout_time.reshape(-1)[start : start + frame_samples]
        if frame_index is not None and not dynamic_readout:
            temporal_factors = native.B
            if temporal_factors.shape[0] == frame_samples:
                frame_interpolator = (temporal_factors, native.C)
            elif (
                trajectory_shape
                and temporal_factors.shape[0] == trajectory_shape[0] * frame_samples
            ):
                start = frame_index * frame_samples
                frame_interpolator = (
                    temporal_factors[start : start + frame_samples],
                    native.C,
                )
        return OffResonance(
            physics.rebuild(new_trajectory, frame_index),
            field_map,
            frame_readout_time,
            r2star_map=r2star_map,
            mask=mask,
            interpolator=frame_interpolator,
        )

    result = MRIPhysics(
        operator,
        native_operator=native,
        kind=physics.kind,
        spatial_ndim=physics.spatial_ndim,
        viewed_as_real=physics.viewed_as_real,
        modifiers=(*physics.modifiers, "off_resonance"),
        trajectory=physics.trajectory,
        rebuild=rebuild,
        toeplitz_options=options if toeplitz_enabled else None,
    )
    if physics.streaming_policy is not None:
        result.enable_streaming(physics.streaming_policy)
    return result


class OffResonance(MRIPhysics):
    """Multi-frequency-interpolation off-resonance correction.

    Parameters
    ----------
    physics
        Base **non-Cartesian** physics. Off-resonance evolves along the
        readout, which a Cartesian encode samples too briefly for this model
        to be the right correction; reverse-polarity distortion correction
        (:func:`pulserver.recon.postprocessing.run_pyhysco`) is the Cartesian
        EPI route.
    field_map
        Off-resonance in Hz over the image grid.
    readout_time
        Time of every sample relative to the echo, in seconds.
    **kwargs
        Segmentation options forwarded to the interpolation.

    Raises
    ------
    TypeError
        If ``physics`` is Cartesian.

    Examples
    --------
    A long readout accumulates phase wherever the field is off resonance, so a
    spiral or a radial train blurs where the field is wrong. The correction is
    a time-segmented model of that phase: the field map, and the time each
    sample was taken.

    .. plot::

       import numpy as np
       import pulserver.recon as recon
       from _figures import images, phantom, radial_spokes

       truth, coil_maps = phantom(64, coils=4)
       spokes = radial_spokes(64, 48)
       trajectory = np.asarray(spokes).reshape(-1, 2).astype(np.float32)

       # Half the field offset by 500 Hz, over a 16 ms readout.
       field_map = np.zeros((64, 64), dtype=np.float32)
       field_map[:, 32:] = 500.0
       readout_time = np.tile(
           np.linspace(0, 16e-3, np.asarray(spokes).shape[1], dtype=np.float32),
           np.asarray(spokes).shape[0],
       )

       encoding = recon.NonCartesian2D(
           trajectory, (64, 64), coil_maps=coil_maps[0], n_coils=4
       )
       corrected = recon.OffResonance(encoding, field_map, readout_time)
       measured = corrected.A(truth)

       ignored = recon.pics(measured, encoding, iterations=10)[0]
       modelled = recon.pics(measured, corrected, iterations=10)[0]

       images(
           [
               ("truth", truth[0]),
               ("field ignored", ignored),
               ("field modelled", modelled),
           ],
           title="A 16 ms readout through a 500 Hz offset",
       )

    The field model is fitted once, when the operator is built, so a solve pays
    for it only at the start.
    """

    def __init__(
        self,
        physics: MRIPhysics,
        field_map: Any,
        readout_time: Any,
        **kwargs: Any,
    ) -> None:
        _init_from(
            self,
            _off_resonance(physics, field_map, readout_time, **kwargs),
        )
