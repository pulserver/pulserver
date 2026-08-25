"""Physics for a scan that samples a grid.

The encoding is a mask on an FFT, so the normal operator is the mask itself.
Simultaneous multislice rides the same path with a CAIPI phase per slice."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import deepinv

from ..execution import _resolve_device
from .._sms import _SMSLinearPhysics

from ._base import MRIPhysics
from ._common import (
    _CartesianComplexView,
    _TrailingRealView,
    _require_deepinv,
    _toeplitz_request,
)


class _CoilwiseCartesianMRI(deepinv.physics.MultiCoilMRI):
    """A Cartesian MRI operator with no sensitivity maps: the coil axis passes
    through untouched.

    DeepInverse's :class:`~deepinv.physics.MultiCoilMRI` collapses the coils in
    its adjoint -- a sensitivity combination when maps are given, and a plain
    sum when they are not. A coil sum is not a meaningful image, so with no maps
    this variant keeps the coils instead: the adjoint returns one image per coil
    and the forward encodes each coil independently. That matches the
    convention the non-Cartesian (mri-nufft) operators already follow, and
    leaves the coil combination to the caller as an explicit step.
    """

    def _spatial_dims(self) -> tuple[int, ...]:
        return (-3, -2, -1) if self.three_d else (-2, -1)

    def A(self, x: Any, mask: Any = None, **kwargs: Any) -> Any:
        """Encode each coil independently: a masked FFT, coils untouched."""
        self.update_parameters(mask=mask, check_coil_maps=False, **kwargs)
        spectrum = self.fft(self.to_torch_complex(x), dim=self._spatial_dims())
        return self.mask[:, :, None] * self.from_torch_complex(spectrum)

    def A_adjoint(
        self, y: Any, mask: Any = None, crop: bool = False, **kwargs: Any
    ) -> Any:
        """Return one image per coil, without combining them."""
        self.update_parameters(mask=mask, check_coil_maps=False, **kwargs)
        masked = self.to_torch_complex(self.mask[:, :, None] * y)
        coil_images = self.ifft(masked, dim=self._spatial_dims())
        return self.crop(self.from_torch_complex(coil_images), crop=crop)

    def A_adjoint_A(self, x: Any, **kwargs: Any) -> Any:
        """Per-coil normal operator: an exact FFT round trip on each coil."""
        return self.A_adjoint(self.A(x, **kwargs))


def _single_precision(value: Any) -> Any:
    """``value`` in the precision every operator here works in.

    A trajectory is what a NUFFT plans on and sensitivities are what it
    applies, so a double-precision one plans a double-precision transform
    and then meets single-precision data -- which the backend reports as a
    dtype mismatch, from inside a plan, far from the call that caused it.
    Whatever arrives, NumPy or Torch, a sequence of either, leaves single.
    """
    import numpy
    import torch

    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return type(value)(_single_precision(item) for item in value)
    if isinstance(value, torch.Tensor):
        if value.dtype == torch.complex128:
            return value.to(torch.complex64)
        if value.dtype == torch.float64:
            return value.to(torch.float32)
        return value
    if isinstance(value, numpy.ndarray):
        if value.dtype == numpy.complex128:
            return value.astype(numpy.complex64, copy=False)
        if value.dtype == numpy.float64:
            return value.astype(numpy.float32, copy=False)
        return value
    return value


def _init_cartesian(
    physics: MRIPhysics,
    mask: Any,
    coil_maps: Any,
    *,
    spatial_ndim: int,
    toeplitz: bool | dict[str, Any] = False,
    viewed_as_real: bool = False,
    **kwargs: Any,
) -> None:
    """Initialize Cartesian physics in place.

    With ``coil_maps`` this is SENSE: the adjoint combines the coils through the
    sensitivities. With ``coil_maps=None`` it is a coil-wise operator whose
    adjoint returns one image per coil, for an explicit coil combination
    afterwards (see :class:`_CoilwiseCartesianMRI`).

    Leading dimensions are handled by DeepInverse as batch dimensions, so
    slices, contrasts, and dynamic frames are reconstructed independently.
    Cartesian normal operations already use exact FFTs; ``toeplitz=True`` is
    accepted for API symmetry and reports ``normal_mode == "exact-fft"``.
    """
    toeplitz_enabled, _, options = _toeplitz_request(toeplitz)
    physics_module = _require_deepinv()
    # Direct imports rather than the module's ``import_module`` so the array
    # boundary keeps working when a test stubs the latter for operator
    # selection.
    import numpy
    import torch

    requested_device = _resolve_device(kwargs.pop("device", None))
    if isinstance(mask, numpy.ndarray):
        mask = torch.as_tensor(mask).to(torch.float32)
    if isinstance(coil_maps, numpy.ndarray):
        coil_maps = torch.as_tensor(coil_maps).to(torch.complex64)
    mask, coil_maps = _single_precision(mask), _single_precision(coil_maps)
    if requested_device is not None:
        if hasattr(mask, "to"):
            mask = mask.to(requested_device)
        if coil_maps is not None and hasattr(coil_maps, "to"):
            coil_maps = coil_maps.to(requested_device)
    device = getattr(coil_maps, "device", getattr(mask, "device", "cpu"))
    operator_class = (
        _CoilwiseCartesianMRI if coil_maps is None else physics_module.MultiCoilMRI
    )
    operator = operator_class(
        mask=mask,
        coil_maps=coil_maps,
        three_d=spatial_ndim == 3,
        device=device,
        **kwargs,
    )
    boundary = _TrailingRealView(operator)
    if not viewed_as_real:
        boundary = _CartesianComplexView(boundary)
    MRIPhysics.__init__(
        physics,
        boundary,
        native_operator=None,
        kind=f"cartesian{spatial_ndim}d",
        spatial_ndim=spatial_ndim,
        viewed_as_real=viewed_as_real,
        modifiers=("toeplitz",) if toeplitz_enabled else (),
        toeplitz_options=options if toeplitz_enabled else None,
    )
    physics._streaming_parameters = {
        "mask": getattr(operator, "mask", mask),
        "coil_maps": getattr(operator, "coil_maps", coil_maps),
    }


class Cartesian2D(MRIPhysics):
    """Two-dimensional Cartesian physics.

    Parameters
    ----------
    mask
        Sampling mask over the encoded grid, shaped ``(h, w)``, ``(c, h, w)``
        or ``(batch, c, h, w)``. Non-zero marks an acquired position.
    coil_maps
        Complex sensitivities shaped ``(coils, h, w)`` or
        ``(batch, coils, h, w)``. ``None`` (the default) is a coil-wise
        operator with no sensitivities: the adjoint returns one image per coil
        rather than a combined image, ``img_size=(h, w)`` is then required, and
        the coil combination is the caller's own explicit step.
    toeplitz
        Accepted for symmetry with the non-Cartesian operators. A Cartesian
        normal operator is already an exact FFT, so this only changes what
        ``normal_mode`` reports.
    **kwargs
        Forwarded to :class:`deepinv.physics.MultiCoilMRI`.

    Notes
    -----
    Everything is native complex: a SENSE image is ``(batch, h, w)``, a
    coil-wise (no-maps) image keeps its coils, ``(batch, coils, h, w)``, and a
    measurement is ``(batch, coils, h, w)`` -- the complex layout every physics
    in this package answers in. Leading dimensions beyond the batch are
    independent problems, so slices, contrasts and frames reconstruct together.

    Examples
    --------
    >>> import torch
    >>> from pulserver.recon.physics import Cartesian2D
    >>> physics = Cartesian2D(
    ...     torch.ones(1, 1, 8, 8),
    ...     torch.ones(1, 3, 8, 8, dtype=torch.complex64) / 3 ** 0.5,
    ... )
    >>> physics.A(torch.randn(1, 8, 8, dtype=torch.complex64)).shape
    torch.Size([1, 3, 8, 8])

    With no maps the adjoint keeps one image per coil, for an explicit
    combination afterwards:

    >>> coil_wise = Cartesian2D(torch.ones(1, 1, 8, 8), img_size=(8, 8))
    >>> coil_wise.A_adjoint(torch.randn(1, 4, 8, 8, dtype=torch.complex64)).shape
    torch.Size([1, 4, 8, 8])

    What the operator does, on DeepInverse's phantom: measure the object
    through each element of the array, and bring it back. Without maps the
    adjoint keeps the coils apart, which is what a calibration wants to see:

    .. plot::

       import torch
       import pulserver.recon as recon
       from _figures import images, phantom

       truth, coil_maps = phantom(64, coils=4)
       mask = torch.ones(1, 1, 64, 64)
       coil_wise = recon.Cartesian2D(mask, img_size=(64, 64))
       measured = recon.Cartesian2D(mask, coil_maps).A(truth)
       coils = coil_wise.A_adjoint(measured)
       images(
           [("object", truth), ("coil 0", coils[0, 0]), ("coil 2", coils[0, 2])],
           title="Cartesian2D, fully sampled, four elements",
       )
    """

    def __init__(
        self,
        mask: Any,
        coil_maps: Any = None,
        *,
        toeplitz: bool | dict[str, Any] = False,
        **kwargs: Any,
    ) -> None:
        _init_cartesian(
            self,
            mask,
            coil_maps,
            spatial_ndim=2,
            toeplitz=toeplitz,
            **kwargs,
        )


class Cartesian3D(MRIPhysics):
    """Three-dimensional Cartesian physics.

    Parameters
    ----------
    mask
        Sampling mask over the encoded volume, trailing ``(d, h, w)``.
    coil_maps
        Complex sensitivities shaped ``(coils, d, h, w)`` or with a leading
        batch. ``None`` (the default) is a coil-wise operator; see
        :class:`Cartesian2D`.
    toeplitz
        Accepted for symmetry; see :class:`Cartesian2D`.
    **kwargs
        Forwarded to :class:`deepinv.physics.MultiCoilMRI`.

    Notes
    -----
    Native complex throughout: a SENSE image is ``(batch, d, h, w)``, a
    coil-wise image ``(batch, coils, d, h, w)``, and a measurement
    ``(batch, coils, d, h, w)``. See :class:`Cartesian2D` for the layout
    convention.

    Examples
    --------
    >>> import torch
    >>> import pulserver.recon as recon
    >>> physics = recon.Cartesian3D(
    ...     torch.ones(1, 1, 8, 8, 8),
    ...     torch.ones(1, 2, 8, 8, 8, dtype=torch.complex64) / 2 ** 0.5,
    ... )
    >>> physics.A(torch.zeros(1, 8, 8, 8, dtype=torch.complex64)).shape
    torch.Size([1, 2, 8, 8, 8])
    """

    def __init__(
        self,
        mask: Any,
        coil_maps: Any = None,
        *,
        toeplitz: bool | dict[str, Any] = False,
        **kwargs: Any,
    ) -> None:
        _init_cartesian(
            self,
            mask,
            coil_maps,
            spatial_ndim=3,
            toeplitz=toeplitz,
            **kwargs,
        )


class SMS(MRIPhysics):
    """Model-based simultaneous-multislice MRI physics.

    A shared base physics is vectorized over the product of batch and slice
    axes, allowing its existing dual-GPU streaming policy to distribute all
    slices together. A sequence of physics objects represents slices with
    distinct trajectories or sampling operators and is composed exactly.

    Parameters
    ----------
    physics
        One shared MRI physics object or one object per simultaneously excited
        slice.
    caipi_encoding
        Complex CAIPI modulation or phase in radians. Its first axis is slice;
        remaining axes broadcast over the trailing measurement dimensions.
    n_slices
        Slice count when a shared physics and no encoding tensor are supplied.
    streaming
        Optional Pulserver CUDA streaming policy forwarded to the base physics.

    Examples
    --------
    Simultaneous multi-slice: the slices are excited together and arrive summed,
    with a CAIPI phase telling them apart. The operator carries that phase, so a
    solve unfolds the slices rather than a separate unaliasing step doing it.

    >>> import torch
    >>> import pulserver.recon as recon
    >>> base = recon.Cartesian2D(torch.ones(16, 16))
    >>> physics = recon.SMS(base, n_slices=2)
    >>> isinstance(physics, recon.MRIPhysics)
    True
    """

    def __init__(
        self,
        physics: MRIPhysics | Sequence[MRIPhysics],
        caipi_encoding: Any | None = None,
        *,
        n_slices: int | None = None,
        streaming: Any | None = None,
    ) -> None:
        selected = list(physics) if isinstance(physics, Sequence) else physics
        operator = _SMSLinearPhysics(selected, caipi_encoding, n_slices)
        base = selected[0] if isinstance(selected, list) else selected
        super().__init__(
            operator,
            native_operator=None,
            kind="sms",
            spatial_ndim=int(getattr(base, "spatial_ndim", 2)),
            viewed_as_real=operator.viewed_as_real,
            modifiers=tuple(dict.fromkeys((*getattr(base, "modifiers", ()), "sms"))),
        )
        if streaming is not None:
            self.enable_streaming(streaming)
