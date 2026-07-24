"""DICOM serialisation convenience functions for reconstructed MRD images.

The implementation delegates image/header conversion to the established
:class:`~pulserver.recon.mrd2dicom.MrdDicomBuilder`; this module only makes a
complete Gadgetron-style series operation convenient for handlers.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .mrd2dicom import DicomWithName, MrdDicomBuilder

__all__ = ["images_to_dicom", "write_dicom_series"]


def images_to_dicom(images: Iterable[Any], metadata: Any) -> list[DicomWithName]:
    """Convert reconstructed MRD images to named DICOM datasets.

    Parameters
    ----------
    images : iterable
        ``ismrmrd.Image`` objects to serialise.
    metadata : object
        Parsed MRD XML header passed to :class:`MrdDicomBuilder`.
    """
    builder = MrdDicomBuilder(metadata)
    return [builder(image) for image in images]


def write_dicom_series(images: Iterable[Any], metadata: Any, output_dir: str | Path) -> list[Path]:
    """Convert images to DICOM and write a complete series to ``output_dir``.

    Returns the written paths.  A failed image conversion raises ``ValueError``
    rather than silently producing an incomplete series.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for converted in images_to_dicom(images, metadata):
        if converted.dset is None:
            raise ValueError(f"DICOM conversion failed for {converted.filename}")
        path = output_dir / converted.filename
        converted.dset.save_as(path, enforce_file_format=True)
        paths.append(path)
    return paths
