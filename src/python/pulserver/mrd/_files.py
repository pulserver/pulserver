"""Reading a scan from a file and writing images back out.

A stored MRD file is replayed through the same contract a live scanner drives,
so a reconstruction written for the scanner runs unchanged over a recording.
DICOM is what leaves at the end, and what a converted study arrives as.

These are the names here that reach into the reconstruction side, and they do
so because of what they are: replaying a file means driving a reconstruction
plugin over it, and a study leaves as DICOM through the same writers the
scanner is served by. The names resolve on first use, so reaching for one of
them is what pulls that machinery in -- importing this package does not.
"""

from __future__ import annotations

__all__ = ["dicom_folder_to_mrd", "images_to_dicom", "reconstruct_file"]

from ..recon._server.dicom2mrd import dicom_folder_to_mrd
from ..recon._server.offline import reconstruct_file
from ..recon._server.serialization import images_to_dicom
