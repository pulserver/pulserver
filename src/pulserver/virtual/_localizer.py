"""Three-plane localizer images drawn from a phantom's ground truth."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import ismrmrd
import numpy as np
import pydicom
import pypulseqpp as pp

from ..recon._runtime.mrd2dicom import MrdDicomBuilder

#: Read and phase directions of each plane along the physical axes, which are
#: the patient's LPS axes for a subject lying head first and supine: the
#: radiological views, anterior and superior at the top of the image.
PLANES = {
    "axial": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "coronal": ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
    "sagittal": ((0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),
}


def localizer(
    phantom: Any,
    *,
    field_t: float,
    subject: str = "",
    fov: float = 0.256,
    matrix: int = 128,
    thickness: float = 5e-3,
    centre: Sequence[float] = (0.0, 0.0, 0.0),
) -> list[pydicom.Dataset]:
    """Return the axial, coronal and sagittal images of a phantom's proton density, as DICOM.

    Each image is the phantom's proton density in a slab ``thickness`` thick
    through ``centre``, sampled at the centres of ``matrix`` by ``matrix``
    pixels over ``fov``, as :meth:`proton_density` of the phantom gives it: the
    ground truth, with no sequence played. The images carry the geometry of
    the reconstructed ones, so a console plans on them as it plans on a
    scanned localizer.

    Parameters
    ----------
    phantom
        A :class:`~pulserver.virtual.Phantom` or :class:`~pulserver.virtual.BrainWeb`.
    field_t
        The magnet's field, in T, recorded as the imaging frequency.
    subject
        Patient name of the series.
    fov
        Field of view of each image, in metres.
    matrix
        Pixels along each side.
    thickness
        Slab thickness, in metres.
    centre
        Physical point the three planes cross at, in metres.
    """
    header = ismrmrd.xsd.ismrmrdHeader(
        experimentalConditions=ismrmrd.xsd.experimentalConditionsType(
            H1resonanceFrequency_Hz=round(pp.Opts().gamma * field_t)
        ),
        subjectInformation=ismrmrd.xsd.subjectInformationType(patientName=subject),
        measurementInformation=ismrmrd.xsd.measurementInformationType(
            patientPosition=ismrmrd.xsd.patientPositionType.HFS,
            seriesDescription="Localizer",
        ),
    )
    convert = MrdDicomBuilder(header)
    centre = np.asarray(centre, dtype=float)
    offsets = (np.arange(matrix) - 0.5 * (matrix - 1)) * fov / matrix
    rows, columns = np.meshgrid(offsets, offsets, indexing="ij")
    images = []
    for index, (read, phase) in enumerate(PLANES.values()):
        read, phase = np.asarray(read), np.asarray(phase)
        normal = np.cross(read, phase)
        points = centre + columns[..., None] * read + rows[..., None] * phase
        density = phantom.proton_density(
            points.reshape(-1, 3), normal=normal, thickness=thickness
        )
        image = ismrmrd.Image.from_array(
            density.reshape(matrix, matrix).astype(np.float32), transpose=False
        )
        head = image.getHead()
        head.position = tuple(1e3 * centre)
        head.read_dir = tuple(read)
        head.phase_dir = tuple(phase)
        head.slice_dir = tuple(normal)
        head.field_of_view = (1e3 * fov, 1e3 * fov, 1e3 * thickness)
        head.slice = index
        image.setHead(head)
        images.append(convert(image).dset)
    return images
