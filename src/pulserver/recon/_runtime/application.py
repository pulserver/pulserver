"""Drives a reconstruction plugin over an MRD stream and emits its outputs."""

from __future__ import annotations

__all__ = ["run_application"]

import ctypes
import dataclasses
from collections.abc import Iterable, Mapping
from typing import Any

import ismrmrd
import numpy as np

from ...mrd._acquisitions import AcquisitionBucket, AcquisitionBucketStats
from ...mrd._images import as_numpy
from ...mrd._metadata import acquisition_label, has_acquisition_flag
from ..plugin import ReconContext, ReconPlugin, ReconResult
from .mrd2dicom import MrdDicomBuilder


def run_application(
    plugin: ReconPlugin,
    connection: Any,
    context: ReconContext,
) -> None:
    """Run ``plugin`` over one MRD stream and send what it emits on ``connection``.

    The stream runs on ``plugin.spawn()``. Acquisitions the plugin's flags accept
    go to :meth:`~pulserver.recon.ReconPlugin.receive` as they arrive; waveforms
    are collected, and each emitted unit carries all received so far; any other
    item is sent back unchanged. :class:`~pulserver.recon.ReconResult` outputs
    become MRD images, or DICOM when requested; a ``(z, y, x)`` volume becomes
    one image per partition along ``slice_dir``, partition ``z // 2`` at
    the volume's centre, as an FFT places it. When the stream ends on
    acquisitions that closed no branch, ``"imaging"`` is reconstructed once more.
    """
    app = plugin.spawn()
    acquisitions: list[Any] = []
    waveforms: list[Any] = []
    image_index = 1
    dicom_builder: MrdDicomBuilder | None = None

    def emit(output: Any, bucket: AcquisitionBucket) -> None:
        nonlocal image_index, dicom_builder
        for item in _outputs(output):
            if isinstance(item, ReconResult):
                images, image_index = _make_images(
                    item,
                    bucket,
                    context,
                    image_index,
                    type(app).__name__,
                )
                for emitted in images:
                    if item.dicom:
                        if dicom_builder is None:
                            dicom_builder = MrdDicomBuilder(context.header)
                        emitted = dicom_builder(emitted)
                    connection.send(emitted)
            else:
                connection.send(item)

    def deliver(output: Any) -> None:
        nonlocal acquisitions
        if output is None:
            return
        # Waveforms belong to the measurement, not the unit, so every unit
        # carries all of them.
        bucket = _make_bucket(acquisitions, waveforms)
        acquisitions = []
        emit(output, bucket)

    app.startup(context)
    last: Any = None
    for item in connection:
        if isinstance(item, ismrmrd.Acquisition):
            if _accept(item, app):
                acquisitions.append(item)
                last = item
                deliver(app.receive(item, context))
        elif isinstance(item, ismrmrd.Waveform):
            waveforms.append(item)
        else:
            connection.send(item)

    if acquisitions and app.branch_for(last) is None:
        deliver(app.recon("imaging", context))


# %% private module subroutines


def _accept(acquisition: Any, app: ReconPlugin) -> bool:
    return all(
        has_acquisition_flag(acquisition, flag) for flag in app.require_flags
    ) and not any(has_acquisition_flag(acquisition, flag) for flag in app.reject_flags)


def _make_bucket(
    acquisitions: list[Any],
    waveforms: list[Any],
) -> AcquisitionBucket:
    data: list[Any] = []
    reference: list[Any] = []
    for acquisition in acquisitions:
        calibration = has_acquisition_flag(acquisition, "ACQ_IS_PARALLEL_CALIBRATION")
        calibration_and_imaging = has_acquisition_flag(
            acquisition, "ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING"
        )
        phase_correction = has_acquisition_flag(acquisition, "ACQ_IS_PHASECORR_DATA")
        if calibration or calibration_and_imaging:
            reference.append(acquisition)
        if not calibration and not phase_correction:
            data.append(acquisition)

    return AcquisitionBucket(
        data=tuple(data),
        datastats=_bucket_stats(data),
        ref=tuple(reference),
        refstats=_bucket_stats(reference),
        waveforms=tuple(waveforms),
        acquisitions=tuple(acquisitions),
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
    if data.ndim != 3 or data.shape[0] == 1:
        image, next_image_index = _make_image(
            result, bucket, context, next_image_index, app_name
        )
        return [image], next_image_index
    images = []
    for partition, plane in enumerate(data):
        image, next_image_index = _make_image(
            dataclasses.replace(
                result,
                data=plane,
                image_index=None
                if result.image_index is None
                else result.image_index + partition,
            ),
            bucket,
            context,
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


def _make_image(
    result: ReconResult,
    bucket: AcquisitionBucket,
    context: ReconContext,
    next_image_index: int,
    app_name: str,
) -> tuple[ismrmrd.Image, int]:
    # A result must name a real acquisition for its geometry.
    if not bucket.data:
        raise ValueError("ReconResult requires at least one imaging acquisition")
    # Negative indices count from the end, so -1 names the acquisition that
    # triggered the bucket -- which is the one belonging to the unit just
    # completed when the bucket also carries earlier units.
    if not -len(bucket.data) <= result.reference < len(bucket.data):
        raise IndexError(
            f"ReconResult reference {result.reference} is outside a bucket of "
            f"{len(bucket.data)} acquisitions"
        )
    acquisition = bucket.data[result.reference]
    data = as_numpy(result.data)
    image_index = (
        next_image_index if result.image_index is None else int(result.image_index)
    )
    field_of_view = _field_of_view(context.header)
    image = ismrmrd.Image.from_array(
        data,
        acquisition=acquisition,
        image_index=image_index,
        image_type=_image_type(result.image_type),
        field_of_view=field_of_view,
        transpose=False,
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


def _field_of_view(header: Any) -> tuple[ctypes.c_float, ...] | None:
    try:
        fov = header.encoding[0].reconSpace.fieldOfView_mm
        return tuple(ctypes.c_float(float(value)) for value in (fov.x, fov.y, fov.z))
    except (AttributeError, IndexError, TypeError):
        return None
