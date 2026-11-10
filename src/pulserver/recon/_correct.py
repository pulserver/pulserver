"""Corrections of a reconstructed image: Fermi k-space filter, gradient nonlinearity and receive-field intensity."""

from __future__ import annotations

__all__ = [
    "GRADIENT_UNWARP",
    "INTENSITY_CORRECTION",
    "KSPACE_FILTER",
    "Corrections",
    "IntensityCorrection",
    "fermi_filtered",
    "gradient_unwarped",
    "states_gradient_coefficients",
]

import dataclasses
import logging
import time
from typing import Any

import numpy as np

#: User parameters a gradient coil's spherical-harmonic table is stated in:
#: ``gradient_coefficients`` is the name a scanner-side interpreter writes; the
#: others are names bartorch also reads.
COEFFICIENT_PARAMETERS = (
    "gradient_coefficients",
    "gradientcoefficients",
    "gradunwarp_coefficients",
    "coeff_dat",
)

#: Recon parameters turning each of :class:`Corrections` off where 0 and on
#: where any other number, whatever the plugin's argument.
KSPACE_FILTER = "KSpaceFilter"
GRADIENT_UNWARP = "GradientUnwarp"
INTENSITY_CORRECTION = "IntensityCorrection"

_log = logging.getLogger("pulserver.recon")


def states_gradient_coefficients(header: Any) -> bool:
    """Whether an MRD header carries a gradient coil's coefficient table."""
    return _coefficient_parameter(header) is not None


def _coefficient_parameter(header: Any) -> str | None:
    parameters = getattr(
        getattr(header, "userParameters", None), "userParameterString", None
    )
    for parameter in parameters or ():
        name = str(getattr(parameter, "name", "")).strip()
        if name.lower() in COEFFICIENT_PARAMETERS and getattr(parameter, "value", None):
            return name
    return None


def gradient_unwarped(image: Any, context: Any, data: Any) -> Any:
    """Correct an image for gradient nonlinearity.

    The coil's spherical-harmonic field is evaluated at the scanner position of
    every voxel of the reconstruction matrix, and the image is resampled at the
    displaced positions; a complex image has its real and imaginary parts
    resampled separately. Geometry comes from the header's reconstruction space
    and the unit's reference acquisition, so the correction is in the frame the
    series was acquired in, oblique prescriptions included. An encoding of one
    partition, a 2D slice, is corrected in its plane: the displacement across
    the slice moves the slice, which resampling cannot undo.

    Parameters
    ----------
    image : array_like
        ``(..., [partitions,] lines, columns)``, NumPy array or Torch tensor.
    context : ReconContext
        The plugin's :class:`ReconContext`, whose header states the table.
    data : ReconData
        The plugin's :class:`ReconData`, whose buffer states the geometry.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        The corrected image, of the type and on the device of ``image``.
        ``image`` itself where the header states no table, where the unit has
        no reference acquisition, or where the correction fails; the last two
        are logged as warnings and an uncorrected image is sent on.
    """
    header = getattr(context, "header", None)
    if header is None or not states_gradient_coefficients(header):
        return image

    buffer = getattr(data, "data", None)
    reference = getattr(buffer, "reference", None)
    if reference is None:
        _log.warning(
            "the header states a gradient coil table and the unit has no "
            "reference acquisition; the image is not corrected"
        )
        return image

    try:
        import torch
        from bartorch.tools import Gradunwarp

        correction = Gradunwarp.from_mrd(header, reference)
        if correction.shape[0] == 1:
            correction = _in_plane(correction, header, reference)
        corrected = correction(torch.as_tensor(image))
    except Exception as error:
        _log.warning("the image was not corrected for gradient nonlinearity: %s", error)
        return image

    # NumPy arrays carry a ``device`` of their own under the array API, so the
    # type is asked for rather than duck-typed.
    return corrected if isinstance(image, torch.Tensor) else corrected.cpu().numpy()


def _in_plane(correction: Any, header: Any, reference: Any) -> Any:
    """Return ``correction``, of one partition, on the slice's ``(lines, columns)`` grid.

    Same table, matrix, field of view and centre, with ``position`` the centre
    of voxel ``n // 2`` as :meth:`bartorch.tools.Gradunwarp.from_mrd` reads it.
    """
    from bartorch.tools import Gradunwarp

    space = header.encoding[int(getattr(reference, "encoding_space_ref", 0) or 0)]
    matrix, extent = space.reconSpace.matrixSize, space.reconSpace.fieldOfView_mm
    shape = np.array([int(matrix.y), int(matrix.x)])
    fov_mm = np.array([float(extent.y), float(extent.x)])
    orientation = np.stack(
        [np.asarray(reference.phase_dir, float), np.asarray(reference.read_dir, float)],
        axis=1,
    )
    offset = (shape - 1) / 2 - shape // 2
    centre = np.asarray(reference.position, float) + orientation @ (
        offset * fov_mm / shape
    )
    return Gradunwarp(
        correction.coefficients,
        tuple(shape),
        tuple(fov_mm),
        orientation,
        tuple(centre),
        jacobian=correction.jacobian,
    )


def fermi_filtered(image: Any, axes: tuple[int, ...] | None = None) -> np.ndarray:
    """Return a complex image with its k-space multiplied by the Fermi window.

    The k-space is the centred unitary Fourier transform of ``image`` over
    ``axes`` (:func:`bartorch.fft`), the zero frequency at index ``n // 2``,
    and the window :func:`bartorch.fermi_window` at its defaults over those
    axes: radial, half height at the Nyquist edge of each axis, its transition
    ten samples of the longest (Bernstein et al., JMRI 2001;14:270). For an
    image a linear reconstruction makes from fully sampled k-space this is the
    apodization of the acquired k-space.

    Parameters
    ----------
    image : array_like
        Complex image, NumPy array or Torch tensor, on the host.
    axes : tuple of int, default=None
        Axes Fourier encoded together; every axis of more than one point when
        ``None``.

    Returns
    -------
    numpy.ndarray
        ``complex64``, shaped like ``image``.
    """
    import bartorch
    import torch

    values = torch.as_tensor(np.ascontiguousarray(image)).to(torch.complex64)
    if axes is None:
        axes = tuple(axis for axis, n in enumerate(values.shape) if n > 1)
    axes = tuple(axis % values.ndim for axis in axes)
    kspace = bartorch.fft(values, axes, unitary=True).reshape(values.shape)
    kspace = bartorch.apodize(kspace, kind="fermi", axes=axes)
    filtered = bartorch.fft(kspace, axes, inverse=True, unitary=True)
    return filtered.reshape(values.shape).numpy()


def fermi_profile(extent: tuple[int, ...], axis: int) -> np.ndarray:
    """Return the Fermi window of :func:`fermi_filtered` over ``extent`` along ``axis``, through the centre of the other axes."""
    import bartorch

    window = bartorch.fermi_window(tuple(int(n) for n in extent))
    centre: list[Any] = [n // 2 for n in extent]
    centre[axis] = slice(None)
    return window[tuple(centre)].numpy()


def _stated(header: Any, name: str, default: bool) -> bool:
    """Whether the recon parameter ``name`` of ``header`` turns its correction on; ``default`` where it is absent or not a number."""
    from ..mrd._metadata import user_parameter

    value = user_parameter(header, name, None)
    if value is None:
        return default
    try:
        return float(value) != 0
    except (TypeError, ValueError):
        _log.warning(
            "recon parameter %s %r is not a number; it is taken as %d",
            name,
            value,
            int(default),
        )
        return default


def _off(header: Any, name: str, argument: str) -> str:
    """Why a step is off: the header's recon parameter ``name``, else the plugin's ``argument``."""
    from ..mrd._metadata import user_parameter

    value = user_parameter(header, name, None)
    if value is None:
        return f"off: the plugin's {argument} is false"
    return f"off: recon parameter {name} is {value}"


class Corrections:
    """The corrections a shipped reconstruction makes to the images of one stream.

    In order:

    1. the Fermi filter of the k-space of each complex image the solve returns
       (:func:`fermi_filtered`), history ``FERMI``;
    2. gradient nonlinearity correction (:func:`gradient_unwarped`), history
       ``GRADUNWARP``, where the header states the coil's table;
    3. the division of the magnitude images by the receive field
       (:class:`IntensityCorrection`), history ``N4``; real, imaginary and
       phase images are sent as reconstructed.

    The header's recon parameter ``KSpaceFilter``, ``GradientUnwarp`` or
    ``IntensityCorrection`` turns its step off where it is 0 and on where it is
    any other number; where the header states none, the argument decides.
    Which steps are on, and why any is off, is logged when the stream starts;
    each step is logged when it runs, at ``INFO`` for the first image of the
    stream and at ``DEBUG`` after. A step that fails leaves the image as it
    is, and the log says why. Make one per stream.

    Parameters
    ----------
    header : ismrmrd.xsd.ismrmrdHeader
        The stream's header.
    kspace_filter, gradient_unwarp, intensity_correction : bool, default=True
        Each step where the header states no recon parameter for it.
    """

    def __init__(
        self,
        header: Any,
        *,
        kspace_filter: bool = True,
        gradient_unwarp: bool = True,
        intensity_correction: bool = True,
    ) -> None:
        self.kspace_filter = _stated(header, KSPACE_FILTER, kspace_filter)
        table = _coefficient_parameter(header)
        unwarp = _stated(header, GRADIENT_UNWARP, gradient_unwarp)
        self.gradient_unwarp = unwarp and table is not None
        self.intensity = (
            IntensityCorrection()
            if _stated(header, INTENSITY_CORRECTION, intensity_correction)
            else None
        )
        self._applied: list[str] = []
        self._logged: set[str] = set()
        if not unwarp:
            unwarp_state = _off(header, GRADIENT_UNWARP, "gradient_unwarp")
        elif table is None:
            unwarp_state = (
                "off: the header states no gradient coil table (user parameter "
                + " or ".join(COEFFICIENT_PARAMETERS)
                + ")"
            )
        else:
            unwarp_state = f"on, coil table from user parameter {table}"
        _log.info(
            "image corrections: Fermi k-space filter %s; gradient nonlinearity "
            "correction %s; N4 intensity correction of the magnitude %s",
            "on"
            if self.kspace_filter
            else _off(header, KSPACE_FILTER, "kspace_filter"),
            unwarp_state,
            "on"
            if self.intensity is not None
            else _off(header, INTENSITY_CORRECTION, "intensity_correction"),
        )

    def filtered(self, image: Any, axes: tuple[int, ...] | None = None) -> Any:
        """Return ``image`` Fermi filtered (:func:`fermi_filtered`) where the filter is on; ``image`` itself otherwise or where it fails."""
        if not self.kspace_filter:
            return image
        start = time.perf_counter()
        try:
            filtered = fermi_filtered(image, axes)
        except Exception:
            _log.exception("the Fermi k-space filter failed; the image is not filtered")
            return image
        self._ran(
            "FERMI",
            "Fermi k-space filter of a %s image in %.2f s",
            np.shape(image),
            time.perf_counter() - start,
        )
        return filtered

    def filtered_along(
        self, kspace: Any, axis: int, extent: tuple[int, ...], along: int = 0
    ) -> Any:
        """Return ``kspace``, centred along ``axis``, multiplied by :func:`fermi_profile` of ``extent`` along its axis ``along``, where the filter is on.

        For an axis Fourier transformed before the solve, such as the
        partitions of a stack of non-Cartesian planes, whose planes are solved
        apart and filtered in their own plane.
        """
        if not self.kspace_filter:
            return kspace
        try:
            profile = fermi_profile(extent, along)
        except Exception:
            _log.exception("the Fermi k-space filter failed; the image is not filtered")
            return kspace
        shape = [1] * np.ndim(kspace)
        shape[axis] = profile.size
        return kspace * profile.reshape(shape)

    def unwarped(self, image: Any, context: Any, data: Any) -> Any:
        """Return ``image`` corrected for gradient nonlinearity (:func:`gradient_unwarped`) where the correction is on."""
        if not self.gradient_unwarp:
            return image
        start = time.perf_counter()
        corrected = gradient_unwarped(image, context, data)
        if corrected is not image:
            self._ran(
                "GRADUNWARP",
                "gradient nonlinearity corrected for a %s image in %.2f s",
                np.shape(image),
                time.perf_counter() - start,
            )
        return corrected

    def intensity_corrected(self, results: Any, data: Any) -> Any:
        """Return ``results`` with their magnitude images divided by the receive field (:class:`IntensityCorrection`) where the correction is on."""
        if self.intensity is None:
            return results
        return self.intensity(results, data)

    def applied(self) -> list[str]:
        """Return the history tags of the steps applied since the last call, in order, and forget them."""
        applied, self._applied = list(dict.fromkeys(self._applied)), []
        return applied

    def _ran(self, tag: str, message: str, *args: Any) -> None:
        self._applied.append(tag)
        level = logging.DEBUG if tag in self._logged else logging.INFO
        self._logged.add(tag)
        _log.log(level, message, *args)


class IntensityCorrection:
    """Division of the magnitude images of a unit by the receive field N4 estimates from them.

    The field is :func:`bartorch.tools.bias_field_correct`'s (SimpleITK's
    N4) over Otsu's mask of the magnitude image, divided by its mean inside
    the mask and clipped to the range it takes there, so the image keeps its
    mean over the object and the background is not amplified beyond the
    object's worst voxel. Real, imaginary and phase images are not divided,
    so the phase of a complex image is sent as reconstructed.

    One field is estimated for each slice and set, from the first unit that
    closes with a magnitude image, and divides every later contrast,
    repetition and cardiac phase of it: echoes and time points keep their
    intensity ratios, so a T2* fit or a time series is not changed by the
    correction. Make one instance per stream.

    A magnitude image that is not 2D or 3D once its axes of one point are
    dropped, or whose mask holds fewer than 64 pixels or 128 voxels, is sent
    uncorrected, as are all images when SimpleITK is not installed or
    the estimate fails; the reason is logged.
    """

    def __init__(self) -> None:
        self.fields: dict[tuple[Any, ...], np.ndarray | None] = {}

    def __call__(self, results: Any, data: Any) -> Any:
        """Return ``results``, a :class:`~pulserver.recon.ReconResult` or a list of them, with their magnitude images divided by the field.

        Each corrected result is a copy whose ``ImageProcessingHistory``
        attribute ends in ``"N4"``. ``data`` is the unit's
        :class:`~pulserver.recon.ReconData`, whose ``slice`` and ``set``
        counters select the field.
        """
        from .plugin import ReconResult

        listed = results if isinstance(results, list) else [results]
        magnitude = next(
            (
                r
                for r in listed
                if isinstance(r, ReconResult) and r.image_type == "magnitude"
            ),
            None,
        )
        if magnitude is None:
            return results
        try:
            field = self._field(_host(magnitude.data), data)
            if field is None:
                return results
            corrected = [
                _divided(r, field)
                if isinstance(r, ReconResult)
                and r.image_type == "magnitude"
                and np.shape(r.data) == field.shape
                else r
                for r in listed
            ]
        except ImportError as error:
            _log.warning("the images were not intensity corrected: %s", error)
            return results
        except Exception:
            _log.exception("the images were not intensity corrected")
            return results
        return corrected if isinstance(results, list) else corrected[0]

    def _field(self, magnitude: np.ndarray, data: Any) -> np.ndarray | None:
        """Return the field of the slice and set of ``data``, estimating it from ``magnitude`` the first time."""
        counters = getattr(data, "counters", None) or {}
        key = (
            int(counters.get("slice", 0)),
            int(counters.get("set", 0)),
            magnitude.shape,
        )
        if key in self.fields:
            return self.fields[key]
        squeezed = tuple(n for n in magnitude.shape if n > 1)
        if len(squeezed) not in (2, 3):
            _log.info(
                "a %s image is not 2D or 3D; it is not intensity corrected",
                magnitude.shape,
            )
            self.fields[key] = None
            return None
        start = time.perf_counter()
        field = receive_field(np.abs(magnitude).reshape(squeezed))
        self.fields[key] = None if field is None else field.reshape(magnitude.shape)
        _log.info(
            "N4 receive field of slice %d, set %d, %s, estimated in %.1f s%s",
            key[0],
            key[1],
            squeezed,
            time.perf_counter() - start,
            "" if field is not None else "; its mask is too small to correct",
        )
        return self.fields[key]


def receive_field(magnitude: np.ndarray) -> np.ndarray | None:
    """Return the N4 field of a 2D or 3D magnitude image, mean 1 inside its Otsu mask and clipped to its range there.

    ``None`` where the mask holds fewer than 64 pixels or 128 voxels.
    """
    import SimpleITK as sitk
    import torch
    from bartorch.tools import bias_field_correct

    image = np.ascontiguousarray(magnitude, dtype=np.float32)
    mask = sitk.GetArrayFromImage(
        sitk.OtsuThreshold(sitk.GetImageFromArray(image), 0, 1, 200)
    ).astype(bool)
    if mask.sum() < 16 * 2**image.ndim:
        return None
    _, field = bias_field_correct(
        torch.from_numpy(image),
        mask=torch.from_numpy(mask.astype(np.uint8)),
        return_field=True,
    )
    field = field.numpy().astype(np.float64)
    inside = field[mask]
    return (np.clip(field, inside.min(), inside.max()) / inside.mean()).astype(
        np.float32
    )


def _host(array: Any) -> np.ndarray:
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    return np.asarray(array)


def _divided(result: Any, field: np.ndarray) -> Any:
    values = _host(result.data)
    attributes = dict(result.attributes)
    history = list(attributes.get("ImageProcessingHistory", ()))
    attributes["ImageProcessingHistory"] = [*history, "N4"]
    return dataclasses.replace(
        result, data=(values / field).astype(values.dtype), attributes=attributes
    )
