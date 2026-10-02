"""Drives a reconstruction plugin over an MRD stream and emits its outputs."""

from __future__ import annotations

__all__ = ["run_application"]

import ctypes
import dataclasses
import numbers
from collections.abc import Iterable, Mapping
from typing import Any

import ismrmrd
import numpy as np

from ...mrd._acquisitions import AcquisitionBucket, AcquisitionBucketStats
from ...mrd._header import EncodingSpace
from ...mrd._images import as_numpy
from ...mrd._metadata import acquisition_label
from .._buffers import ReconData, nearest_to_centre, readout_roles
from ..plugin import ReconContext, ReconPlugin, ReconResult
from .mrd2dicom import MrdDicomBuilder


def run_application(
    plugin: ReconPlugin,
    connection: Any,
    context: ReconContext,
) -> None:
    """Run ``plugin`` over one MRD stream and send what it emits on ``connection``.

    The stream runs on ``plugin.spawn()``. Acquisitions go to
    :meth:`~pulserver.recon.ReconPlugin.receive` as they arrive, and each unit
    it closes is emitted at once; waveforms are held for the next unit that
    closes; any other item is sent back unchanged. When the stream ends the
    units still open are reconstructed and
    :meth:`~pulserver.recon.ReconPlugin.finish` runs.
    :class:`~pulserver.recon.ReconResult` outputs become MRD images, or DICOM
    when requested; a ``(z, y, x)`` volume becomes one image per partition
    along ``slice_dir``, partition ``z // 2`` at the volume's centre, as an FFT
    places it, each stating the range of the volume in its meta attributes
    ``ArrayMinimum`` and ``ArrayMaximum``. One DICOM builder serves the whole
    stream, so a series has one rescale mapping across its images.
    """
    app = plugin.spawn()
    image_index = 1
    dicom_builder: MrdDicomBuilder | None = None

    def emit(emitted: list[tuple[ReconData | None, Any]]) -> None:
        nonlocal image_index, dicom_builder
        for data, output in emitted:
            bucket = _make_bucket(data)
            for item in _outputs(output):
                if not isinstance(item, ReconResult):
                    connection.send(item)
                    continue
                images, image_index = _make_images(
                    item, bucket, context, image_index, type(app).__name__
                )
                for image in images:
                    if item.dicom:
                        if dicom_builder is None:
                            dicom_builder = MrdDicomBuilder(context.header)
                        image = dicom_builder(image)
                    connection.send(image)

    app.startup(context)
    for item in connection:
        if isinstance(item, ismrmrd.Acquisition):
            emit(app.receive(item, context))
        elif isinstance(item, ismrmrd.Waveform):
            app.receive_waveform(item)
        else:
            connection.send(item)
    emit(app.flush(context))


# %% private module subroutines


def _make_bucket(data: ReconData | None) -> AcquisitionBucket:
    """Return a unit's acquisitions split as Gadgetron's bucket splits them."""
    acquisitions = () if data is None else tuple(data.acquisitions)
    imaging: list[Any] = []
    reference: list[Any] = []
    for acquisition in acquisitions:
        in_data, in_ref = readout_roles(acquisition)
        if in_data:
            imaging.append(acquisition)
        if in_ref:
            reference.append(acquisition)
    return AcquisitionBucket(
        data=tuple(imaging),
        datastats=_bucket_stats(imaging),
        ref=tuple(reference),
        refstats=_bucket_stats(reference),
        waveforms=() if data is None else data.waveforms,
        acquisitions=acquisitions,
    )


def _bucket_stats(acquisitions: list[Any]) -> tuple[AcquisitionBucketStats, ...]:
    if not acquisitions:
        return ()
    fields = tuple(AcquisitionBucketStats.__dataclass_fields__)
    encoding_spaces = max(
        int(acquisition_label(acquisition, "encoding_space_ref", 0) or 0)
        for acquisition in acquisitions
    )
    values = [{name: set() for name in fields} for _ in range(encoding_spaces + 1)]
    for acquisition in acquisitions:
        encoding = int(acquisition_label(acquisition, "encoding_space_ref", 0) or 0)
        for name in fields:
            values[encoding][name].add(
                int(acquisition_label(acquisition, name, 0) or 0)
            )
    return tuple(
        AcquisitionBucketStats(
            **{name: frozenset(value) for name, value in encoding.items()}
        )
        for encoding in values
    )


def _outputs(output: Any) -> Iterable[Any]:
    if output is None:
        return ()
    if isinstance(output, ReconResult) or _is_native_output(output):
        return (output,)
    if _is_array(output):
        return (ReconResult(output),)
    if isinstance(output, Iterable) and not isinstance(output, (str, bytes, Mapping)):
        return output
    return (output,)


def _is_native_output(value: Any) -> bool:
    return isinstance(value, (ismrmrd.Acquisition, ismrmrd.Image, ismrmrd.Waveform))


def _is_array(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return True
    try:
        import torch
    except ImportError:
        return False
    return isinstance(value, torch.Tensor)


def _make_images(
    result: ReconResult,
    bucket: AcquisitionBucket,
    context: ReconContext,
    next_image_index: int,
    app_name: str,
) -> tuple[list[ismrmrd.Image], int]:
    data = as_numpy(result.data)
    unit = _unit_header(result, bucket, context.header)
    if data.ndim != 3 or data.shape[0] == 1:
        image, next_image_index = _make_image(result, unit, next_image_index, app_name)
        return [image], next_image_index
    images = []
    attributes = {**_array_range(data), **result.attributes}
    for partition, plane in enumerate(data):
        image, next_image_index = _make_image(
            dataclasses.replace(
                result,
                data=plane,
                reference=None,
                attributes=attributes,
                image_index=None
                if result.image_index is None
                else result.image_index + partition,
            ),
            unit,
            next_image_index,
            app_name,
        )
        thickness = float(image.field_of_view[2]) / len(data)
        shift = (partition - len(data) // 2) * thickness
        image.position = tuple(
            float(image.position[axis]) + shift * float(image.slice_dir[axis])
            for axis in range(3)
        )
        image.field_of_view = (*image.field_of_view[:2], thickness)
        image.slice = partition
        images.append(image)
    return images, next_image_index


def _array_range(array: np.ndarray) -> dict[str, float]:
    """Return the smallest and largest finite value of a floating-point ``array``, of its magnitude when it is complex, as meta attributes."""
    if not np.issubdtype(array.dtype, np.inexact):
        return {}
    values = np.abs(array) if np.iscomplexobj(array) else array
    finite = values[np.isfinite(values)]
    if not finite.size:
        return {}
    return {
        "ArrayMinimum": float(finite.min()),
        "ArrayMaximum": float(finite.max()),
    }


def _unit_header(
    result: ReconResult, bucket: AcquisitionBucket, header: Any
) -> dict[str, Any]:
    """Return the ``Image.from_array`` arguments shared by the images of one result.

    The reference acquisition is the result's, or the unit's imaging
    acquisition nearest the k-space centre. The field of view is that of the
    reconstruction space of the unit's encoding space, ``(x, y, z)`` in mm, and
    the time stamp the earliest of the unit's imaging acquisitions.
    """
    # A result must name a real acquisition for its geometry.
    if not bucket.data:
        raise ValueError("ReconResult requires at least one imaging acquisition")
    space = int(acquisition_label(bucket.data[0], "encoding_space_ref", 0) or 0)
    reference = result.reference
    if reference is None:
        reference = _reference_acquisition(bucket.data, header, space)
    elif isinstance(reference, numbers.Integral):
        if not -len(bucket.data) <= reference < len(bucket.data):
            raise IndexError(
                f"ReconResult reference {reference} is outside a bucket of "
                f"{len(bucket.data)} acquisitions"
            )
        reference = bucket.data[reference]
    return {
        "acquisition": reference,
        "field_of_view": _field_of_view(header, space),
        "acquisition_time_stamp": min(
            int(acquisition.acquisition_time_stamp) for acquisition in bucket.data
        ),
    }


def _reference_acquisition(acquisitions: Any, header: Any, space: int) -> Any:
    """Return the acquisition nearest the k-space centre of encoding space ``space``, the first when the header does not describe it."""
    try:
        encoding = EncodingSpace.from_header(header, space)
    except (AttributeError, IndexError, TypeError):
        return acquisitions[0]
    return nearest_to_centre(encoding, acquisitions)


def _make_image(
    result: ReconResult,
    unit: Mapping[str, Any],
    next_image_index: int,
    app_name: str,
) -> tuple[ismrmrd.Image, int]:
    data = _image_values(as_numpy(result.data))
    image_index = (
        next_image_index if result.image_index is None else int(result.image_index)
    )
    image = ismrmrd.Image.from_array(
        data,
        image_index=image_index,
        image_type=_image_type(result.image_type),
        transpose=False,
        **unit,
    )
    image.image_series_index = int(result.series_index)

    attributes = {
        "DataRole": "Image",
        "ImageProcessingHistory": ["PULSERVER", app_name],
        **dict(result.attributes),
    }
    head = image.getHead()
    attributes.setdefault(
        "ImageRowDir", [f"{float(head.read_dir[index]):.18f}" for index in range(3)]
    )
    attributes.setdefault(
        "ImageColumnDir",
        [f"{float(head.phase_dir[index]):.18f}" for index in range(3)],
    )
    image.attribute_string = ismrmrd.Meta(attributes).serialize()
    return image, max(next_image_index + 1, image_index + 1)


def _image_values(data: np.ndarray) -> np.ndarray:
    """Return ``data`` as an MRD image stores it: floating point as ``float32``, complex as ``complex64``, integers as they are."""
    if np.issubdtype(data.dtype, np.complexfloating):
        return data.astype(np.complex64, copy=False)
    if np.issubdtype(data.dtype, np.floating):
        return data.astype(np.float32, copy=False)
    return data


def _image_type(name: str) -> int:
    types = {
        "magnitude": ismrmrd.IMTYPE_MAGNITUDE,
        "phase": ismrmrd.IMTYPE_PHASE,
        "real": ismrmrd.IMTYPE_REAL,
        "imaginary": ismrmrd.IMTYPE_IMAG,
        "complex": ismrmrd.IMTYPE_COMPLEX,
    }
    try:
        return types[name.lower()]
    except KeyError as error:
        raise ValueError(f"unknown MRD image type {name!r}") from error


def _field_of_view(header: Any, space: int) -> tuple[ctypes.c_float, ...] | None:
    try:
        encoding = header.encoding[space]
        region = getattr(encoding, "reconSpace", None) or encoding.encodedSpace
        fov = region.fieldOfView_mm
        return tuple(ctypes.c_float(float(value)) for value in (fov.x, fov.y, fov.z))
    except (AttributeError, IndexError, TypeError):
        return None
