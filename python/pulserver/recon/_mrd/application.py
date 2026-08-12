"""Private adapter between MRD streams and public reconstruction apps."""

from __future__ import annotations

__all__ = ["run_application"]

import ctypes
from collections.abc import Iterable, Mapping
from typing import Any

import ismrmrd
import numpy as np

from ..app import (
    AcquisitionBucket,
    AcquisitionBucketStats,
    ReconApp,
    ReconContext,
    ReconResult,
)
from .metadata import acquisition_label, has_acquisition_flag
from .mrd2dicom import MrdDicomBuilder


def run_application(
    app: ReconApp,
    connection: Any,
    context: ReconContext,
) -> None:
    """Consume one MRD stream through ``app`` and emit its outputs."""
    acquisitions: list[Any] = []
    waveforms: list[Any] = []
    image_index = 1
    dicom_builder: MrdDicomBuilder | None = None

    def reconstruct_bucket() -> None:
        nonlocal acquisitions, waveforms, image_index, dicom_builder
        if not acquisitions:
            return
        bucket = _make_bucket(acquisitions, waveforms)
        acquisitions = []
        waveforms = []
        output = app.reconstruct(bucket, context)
        for item in _outputs(output):
            if isinstance(item, ReconResult):
                emitted, image_index = _make_image(
                    item,
                    bucket,
                    context,
                    image_index,
                    type(app).__name__,
                )
                if item.dicom:
                    if dicom_builder is None:
                        dicom_builder = MrdDicomBuilder(context.header)
                    emitted = dicom_builder(emitted)
                connection.send(emitted)
            else:
                connection.send(item)

    for item in connection:
        if isinstance(item, ismrmrd.Acquisition):
            finish = app.split_on is not None and has_acquisition_flag(
                item, app.split_on
            )
            if _accept(item, app):
                acquisitions.append(item)
            if finish:
                reconstruct_bucket()
        elif isinstance(item, ismrmrd.Waveform):
            waveforms.append(item)
        else:
            connection.send(item)

    reconstruct_bucket()


# %% private module subroutines


def _accept(acquisition: Any, app: ReconApp) -> bool:
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


def _make_image(
    result: ReconResult,
    bucket: AcquisitionBucket,
    context: ReconContext,
    next_image_index: int,
    app_name: str,
) -> tuple[ismrmrd.Image, int]:
    if not bucket.data:
        raise ValueError("ReconResult requires at least one imaging acquisition")
    if not 0 <= result.reference < len(bucket.data):
        raise IndexError(
            f"ReconResult reference {result.reference} is outside a bucket of "
            f"{len(bucket.data)} acquisitions"
        )
    data = _as_numpy(result.data)
    image_index = (
        next_image_index if result.image_index is None else int(result.image_index)
    )
    field_of_view = _field_of_view(context.header)
    image = ismrmrd.Image.from_array(
        data,
        acquisition=bucket.data[result.reference],
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


def _as_numpy(value: Any) -> np.ndarray:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
        numpy = getattr(value, "numpy", None)
        if callable(numpy):
            value = numpy()
    return np.asarray(value)


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
