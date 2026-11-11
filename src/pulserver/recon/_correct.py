"""Corrections of a reconstructed image: Fermi k-space filter, gradient nonlinearity and receive-field intensity."""

from __future__ import annotations

__all__ = [
    "GRADIENT_UNWARP",
    "INTENSITY_CORRECTION",
    "INTENSITY_CORRECTION_LEVELS",
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

#: Recon parameter stating the fitting levels of the N4 field, an integer
#: clamped to 1 to :data:`N4_MAX_LEVELS`; :data:`N4_FITTING_LEVELS` where
#: absent or not an integer.
INTENSITY_CORRECTION_LEVELS = "IntensityCorrectionLevels"

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


def gradient_unwarped(
    image: Any, context: Any, data: Any, cache: dict | None = None
) -> Any:
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
    cache : dict, default=None
        Corrections already built for the stream, by the reference
        acquisition's encoding space, position and orientation; the one this
        unit needs is taken from it or added to it. A stream of several units
        of one geometry, the echoes of a multi-echo scan or the volumes of a
        time series, so builds its correction once.

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

        correction = _correction(header, reference, cache)
        corrected = correction(torch.as_tensor(image))
    except Exception as error:
        _log.warning("the image was not corrected for gradient nonlinearity: %s", error)
        return image

    # NumPy arrays carry a ``device`` of their own under the array API, so the
    # type is asked for rather than duck-typed.
    return corrected if isinstance(image, torch.Tensor) else corrected.cpu().numpy()


#: Voxels the corrections a stream keeps (:func:`gradient_unwarped`'s
#: ``cache``) may span together, about two whole-head volumes; the oldest
#: are dropped beyond it. A correction holds about 60 bytes a voxel, its
#: displacement field and sample positions: some 0.7 GB for a 176 x 256 x 256
#: volume, 2 GB at the cap.
UNWARP_CACHE_VOXELS = 2 * 256**3


def _correction(header: Any, reference: Any, cache: dict | None) -> Any:
    """Return the gradient nonlinearity correction of ``reference``'s geometry, from ``cache`` where it holds it."""
    from bartorch.tools import Gradunwarp

    key = (
        int(getattr(reference, "encoding_space_ref", 0) or 0),
        *(
            tuple(float(v) for v in getattr(reference, name))
            for name in ("position", "read_dir", "phase_dir", "slice_dir")
        ),
    )
    if cache is not None and key in cache:
        cache[key] = cache.pop(key)
        return cache[key]
    correction = Gradunwarp.from_mrd(header, reference)
    if correction.shape[0] == 1:
        correction = _in_plane(correction, header, reference)
    if cache is not None:
        cache[key] = correction
        while (
            len(cache) > 1
            and sum(np.prod(c.shape) for c in cache.values()) > UNWARP_CACHE_VOXELS
        ):
            cache.pop(next(iter(cache)))
    return correction


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


#: Words a recon parameter turns its correction on or off with, besides a
#: number (0 off, any other on), in any case.
SWITCH_WORDS = {
    "on": True,
    "true": True,
    "yes": True,
    "off": False,
    "false": False,
    "no": False,
}


def _stated(header: Any, name: str, default: bool) -> bool:
    """Whether the recon parameter ``name`` of ``header`` turns its correction on; ``default`` where it is absent or neither a number nor one of :data:`SWITCH_WORDS`."""
    from ..mrd._metadata import user_parameter

    value = user_parameter(header, name, None)
    if value is None:
        return default
    word = str(value).strip().lower()
    if word in SWITCH_WORDS:
        return SWITCH_WORDS[word]
    try:
        return float(word) != 0
    except ValueError:
        _log.warning(
            "recon parameter %s %r is neither a number nor on/off; it is taken as %s",
            name,
            value,
            "on" if default else "off",
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
    ``IntensityCorrection`` turns its step off where it is 0 or one of the
    off words of :data:`SWITCH_WORDS`, and on where it is any other number or
    an on word; where the header states none, or a value of neither kind, the
    argument decides.
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
            IntensityCorrection(_levels(header))
            if _stated(header, INTENSITY_CORRECTION, intensity_correction)
            else None
        )
        self._applied: list[str] = []
        self._logged: set[str] = set()
        self._unwarps: dict = {}
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
            f"on, {self.intensity.levels} fitting levels"
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
        """Return ``image`` corrected for gradient nonlinearity (:func:`gradient_unwarped`) where the correction is on, each geometry's correction built once for the stream."""
        if not self.gradient_unwarp:
            return image
        start = time.perf_counter()
        corrected = gradient_unwarped(image, context, data, self._unwarps)
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

    def ran(self, tag: str, message: str, *args: Any) -> None:
        """Record a step a plugin made to the image itself, such as the EPI susceptibility correction.

        ``tag`` joins the history :meth:`applied` returns, and ``message``, a
        logging format with ``args``, is logged as each correction's own line
        is: at ``INFO`` the first time the tag runs in the stream, at
        ``DEBUG`` after.
        """
        self._ran(tag, message, *args)

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
    correction. Each slice of a 2D multi-slice series has a field of its own,
    of mean 1 inside its own mask, so ratios between slices are not kept.
    Make one instance per stream.

    A magnitude image that is not 2D or 3D once its axes of one point are
    dropped, or whose mask holds fewer than 64 pixels or 128 voxels or less
    than :data:`N4_MIN_MASK_FRACTION` of the image, or that falls apart into
    separate compartments (:data:`N4_MIN_CONNECTED_FRACTION`), is sent
    uncorrected, as are all images when SimpleITK is not installed or
    the estimate fails; the reason is logged.

    Parameters
    ----------
    levels : int, default=None
        Fitting levels of the field; :data:`N4_FITTING_LEVELS` when ``None``.
    """

    def __init__(self, levels: int | None = None) -> None:
        self.levels = N4_FITTING_LEVELS if levels is None else int(levels)
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
        field = receive_field(np.abs(magnitude).reshape(squeezed), self.levels)
        self.fields[key] = None if field is None else field.reshape(magnitude.shape)
        _log.info(
            "N4 receive field of slice %d, set %d, %s, estimated in %.1f s%s",
            key[0],
            key[1],
            squeezed,
            time.perf_counter() - start,
            ""
            if field is not None
            else "; its mask is too small to tell shading from contrast, so it is not corrected",
        )
        return self.fields[key]


#: Fitting levels of the N4 field by default. The B-spline mesh of
#: SimpleITK's N4, one element across the image at the first level, doubles
#: at each. At four the field removes a head coil's centre-bright shading at
#: 7 T, and compresses the intensity ratios between separate compartments,
#: such as vials in a bath, towards one; at two it keeps those ratios and
#: leaves that shading.
N4_FITTING_LEVELS = 4

#: Least fraction of N4's Otsu mask its largest connected part must hold. A
#: mask of a few separate compartments, such as vials in air, gives N4 nothing
#: to tell shading from their contrast by, and its field scrambles their
#: intensities; a head or a phantom in a bath is one part.
N4_MIN_CONNECTED_FRACTION = 0.5

#: Least fraction of the image N4's Otsu mask must hold.
N4_MIN_MASK_FRACTION = 0.01


#: Most fitting levels :data:`INTENSITY_CORRECTION_LEVELS` may state.
N4_MAX_LEVELS = 4


def _largest_part(mask: np.ndarray) -> int:
    """Return the voxels of the largest connected part of a boolean mask."""
    import SimpleITK as sitk

    parts = sitk.RelabelComponent(
        sitk.ConnectedComponent(sitk.GetImageFromArray(mask.astype(np.uint8))),
        sortByObjectSize=True,
    )
    return int((sitk.GetArrayViewFromImage(parts) == 1).sum())


def _levels(header: Any) -> int:
    """Return the N4 fitting levels the header's :data:`INTENSITY_CORRECTION_LEVELS` states, else :data:`N4_FITTING_LEVELS`."""
    from ..mrd._metadata import user_parameter

    value = user_parameter(header, INTENSITY_CORRECTION_LEVELS, None)
    if value is None:
        return N4_FITTING_LEVELS
    try:
        number = float(str(value).strip())
        levels = int(number)
    except (ValueError, OverflowError):
        number = levels = None
    if levels is None or levels != number:
        _log.warning(
            "recon parameter %s %r is not an integer; the N4 field is fitted at "
            "%d levels",
            INTENSITY_CORRECTION_LEVELS,
            value,
            N4_FITTING_LEVELS,
        )
        return N4_FITTING_LEVELS
    clamped = min(max(levels, 1), N4_MAX_LEVELS)
    if clamped != levels:
        _log.warning(
            "recon parameter %s %r is outside 1 to %d; the N4 field is fitted at "
            "%d levels",
            INTENSITY_CORRECTION_LEVELS,
            value,
            N4_MAX_LEVELS,
            clamped,
        )
    return clamped


def receive_field(
    magnitude: np.ndarray, levels: int = N4_FITTING_LEVELS
) -> np.ndarray | None:
    """Return the N4 field of a 2D or 3D magnitude image, mean 1 inside its Otsu mask and clipped to its range there.

    The field is fitted at ``levels`` B-spline levels. ``None`` where the mask
    holds fewer than 64 pixels or 128 voxels, or less than
    :data:`N4_MIN_MASK_FRACTION` of the image, or whose largest connected part
    holds less than :data:`N4_MIN_CONNECTED_FRACTION` of it.
    """
    import SimpleITK as sitk
    import torch
    from bartorch.tools import bias_field_correct

    image = np.ascontiguousarray(magnitude, dtype=np.float32)
    mask = sitk.GetArrayFromImage(
        sitk.OtsuThreshold(sitk.GetImageFromArray(image), 0, 1, 200)
    ).astype(bool)
    if mask.sum() < max(16 * 2**image.ndim, N4_MIN_MASK_FRACTION * mask.size):
        return None
    if _largest_part(mask) < N4_MIN_CONNECTED_FRACTION * mask.sum():
        return None
    _, field = bias_field_correct(
        torch.from_numpy(image),
        mask=torch.from_numpy(mask.astype(np.uint8)),
        iterations=(50,) * int(levels),
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
