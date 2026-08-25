"""The MRD data model, and the array toolbox both sides of a scan speak.

One flat namespace. What a scanner sends, how the scan is described, the
array operations a reconstruction reads and writes with, what the sequence
told the reconstruction about itself, and the files a study is stored as::

    import pulserver.mrd as mrd

    clean = mrd.noise_prewhiten(data, noise)
    image = mrd.center_crop(mrd.coil_combine(coil_images), shape)

The names are grouped by what they are for in the API documentation, which is
where a grouping belongs: it is a way of reading the library, not a constraint
on how to import from it.

Names resolve on first use, so importing this module needs neither an MRD
server environment nor any optional numerical backend; only the names actually
touched pull their dependencies in.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

#: Public name to the module defining it. Every name is reachable flat; this
#: is the only place the file layout is written down.
_MEMBERS = {
    "AcquisitionBucket": "_acquisitions",
    "AcquisitionBucketStats": "_acquisitions",
    "AcquisitionFlag": "_acquisitions",
    "AdcRole": "_seqdesc",
    "EncodingSpace": "_header",
    "EventType": "_seqdesc",
    "Homodyne": "_arrays",
    "POCS": "_arrays",
    "RfDefinition": "_seqdesc",
    "RfShape": "_seqdesc",
    "RfUse": "_seqdesc",
    "SequenceDescription": "_seqdesc",
    "SequenceDescriptionCollection": "_seqdesc",
    "SequenceEvent": "_seqdesc",
    "SequenceParameters": "_seqdesc",
    "ShimDefinition": "_seqdesc",
    "acquisition_label": "_acquisitions",
    "acquisition_labels": "_acquisitions",
    "as_numpy": "_images",
    "center_crop": "_images",
    "coil_combine": "_images",
    "coil_compress": "_arrays",
    "correct_lines": "_arrays",
    "decode_sequence_description": "_seqdesc",
    "decompress_shape": "_seqdesc",
    "dicom_folder_to_mrd": "_files",
    "diffusion_table": "_header",
    "epi_ramp_operator": "_arrays",
    "estimate_epi_phase": "_arrays",
    "fftc": "_arrays",
    "fill_partial_echo": "_arrays",
    "has_acquisition_flag": "_acquisitions",
    "ifftc": "_arrays",
    "images_to_dicom": "_files",
    "noise_prewhiten": "_arrays",
    "pipe_menon_dcf": "_arrays",
    "reconstruct_file": "_files",
    "remove_readout_oversampling": "_arrays",
    "user_parameter": "_header",
}

#: Reachable on the namespace but not part of its vocabulary. The encoding
#: space is the layout a header states; the stream contract builds it and a
#: plugin reads it off ``buffer.space``. Stating one directly is how a test
#: asks for a buffer of a given shape without a scan file, so the name
#: resolves -- it is not something a plugin writes.
_INTERNAL = frozenset({"EncodingSpace"})

__all__ = sorted(_MEMBERS.keys() - _INTERNAL)


def __getattr__(name: str) -> Any:
    """Resolve one public name on first use."""
    module = _MEMBERS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f"{__name__}.{module}"), name)


def __dir__() -> list[str]:
    """Return the flat public MRD namespace."""
    return sorted(__all__)


if TYPE_CHECKING:
    from ._acquisitions import AcquisitionBucket as AcquisitionBucket
    from ._acquisitions import AcquisitionBucketStats as AcquisitionBucketStats
    from ._acquisitions import AcquisitionFlag as AcquisitionFlag
    from ._acquisitions import acquisition_label as acquisition_label
    from ._acquisitions import acquisition_labels as acquisition_labels
    from ._acquisitions import has_acquisition_flag as has_acquisition_flag
    from ._arrays import Homodyne as Homodyne
    from ._arrays import POCS as POCS
    from ._arrays import coil_compress as coil_compress
    from ._arrays import correct_lines as correct_lines
    from ._arrays import epi_ramp_operator as epi_ramp_operator
    from ._arrays import estimate_epi_phase as estimate_epi_phase
    from ._arrays import fftc as fftc
    from ._arrays import fill_partial_echo as fill_partial_echo
    from ._arrays import ifftc as ifftc
    from ._arrays import noise_prewhiten as noise_prewhiten
    from ._arrays import pipe_menon_dcf as pipe_menon_dcf
    from ._arrays import remove_readout_oversampling as remove_readout_oversampling
    from ._files import dicom_folder_to_mrd as dicom_folder_to_mrd
    from ._files import images_to_dicom as images_to_dicom
    from ._files import reconstruct_file as reconstruct_file
    from ._header import EncodingSpace as EncodingSpace
    from ._header import diffusion_table as diffusion_table
    from ._header import user_parameter as user_parameter
    from ._images import as_numpy as as_numpy
    from ._images import center_crop as center_crop
    from ._images import coil_combine as coil_combine
    from ._seqdesc import AdcRole as AdcRole
    from ._seqdesc import EventType as EventType
    from ._seqdesc import RfDefinition as RfDefinition
    from ._seqdesc import RfShape as RfShape
    from ._seqdesc import RfUse as RfUse
    from ._seqdesc import SequenceDescription as SequenceDescription
    from ._seqdesc import SequenceDescriptionCollection as SequenceDescriptionCollection
    from ._seqdesc import SequenceEvent as SequenceEvent
    from ._seqdesc import SequenceParameters as SequenceParameters
    from ._seqdesc import ShimDefinition as ShimDefinition
    from ._seqdesc import decode_sequence_description as decode_sequence_description
    from ._seqdesc import decompress_shape as decompress_shape
