"""Replay of an ISMRMRD HDF5 file through a reconstruction plugin, in process."""

from __future__ import annotations

__all__ = ["reconstruct_file"]

import contextlib
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import ismrmrd
import ismrmrd.xsd

from ..plugin import ExamCache, ReconContext
from .application import run_application
from .mrd2dicom import MrdDicomBuilder, name_dictionary

_log = logging.getLogger(__name__)


class _FileConnection:
    """Connection reading a dataset and keeping what is sent."""

    def __init__(
        self, dataset: Any, design: Any = None, fov_offset_m: Any = None
    ) -> None:
        self._dataset = dataset
        self._design = design
        self._fov_offset_m = fov_offset_m
        self.sent: list[Any] = []

    def __iter__(self) -> Any:
        if self._design is None:
            return self._items()
        from ...proxy._proxy import _enriched

        return _enriched(self._items(), self._design, self._fov_offset_m)

    def _items(self) -> Any:
        # Waveforms first, so the first unit to close carries all of them.
        for index in range(_count(self._dataset.number_of_waveforms)):
            yield self._dataset.read_waveform(index)
        for index in range(_count(self._dataset.number_of_acquisitions)):
            yield self._dataset.read_acquisition(index)

    def send(self, item: Any) -> None:
        self.sent.append(item)


def reconstruct_file(
    plugin: Any,
    path: str,
    *,
    group: str = "dataset",
    exam_id: Any = None,
    config: Any = None,
    store: Any = None,
    design: Any = None,
    dicom: bool = False,
    dicom_dictionary: Any = None,
) -> list[Any]:
    """Drive ``plugin`` over one ISMRMRD HDF5 file and return what it emitted.

    Implements :meth:`pulserver.recon.ReconPlugin.run`, which documents the
    parameters and exceptions.
    """
    dataset = ismrmrd.Dataset(path, group, create_if_needed=False)
    try:
        document = dataset.read_xml_header()
        if not document:
            raise ValueError(
                f"{path} carries no MRD XML header; a reconstruction needs the "
                "header the scanner sends ahead of the data"
            )
        header = ismrmrd.xsd.CreateFromDocument(document)
        if dicom_dictionary is not None:
            name_dictionary(header, dicom_dictionary)
        played = _played(header, store, design)
        fov_offset_m = None
        if played is not None:
            from ...proxy._enrich import enrich_header, header_fov_offset_m, is_enriched

            if is_enriched(header):
                # Kept by a proxy as it reconstructed it: enriching it again
                # would apply a shifted field of view's phase twice.
                _log.info("%s is enriched already; replayed as it is", path)
                played = None
            else:
                enrich_header(header, played.table)
                fov_offset_m = header_fov_offset_m(header)
        context = ReconContext(
            header=header,
            exam=ExamCache(path if exam_id is None else exam_id),
            config=config,
        )
        connection = _FileConnection(dataset, played, fov_offset_m)
        run_application(plugin, connection, context)
        if not dicom:
            return connection.sent
        convert = MrdDicomBuilder(header)
        return [
            convert(item) if isinstance(item, ismrmrd.Image) else item
            for item in connection.sent
        ]
    finally:
        with contextlib.suppress(Exception):
            dataset.close()


def _played(header: Any, store: Any, design: Any) -> Any:
    """Return the design a series was played from, or None when neither source is given."""
    if store is None and design is None:
        return None
    from ...proxy._designs import DesignCache

    if design is None:
        return DesignCache(store).resolve(header)
    if store is not None:
        raise ValueError("give the design store or the sequence file, not both")
    return _design_of(header, Path(design))


def _design_of(header: Any, entry: Path) -> Any:
    """Return the design whose chain starts at ``entry``, checked against its manifest and the header."""
    from ...host._store import MANIFEST
    from ...mrd._metadata import user_parameter
    from ...proxy._designs import (
        DESIGN_PARAMETER,
        Design,
        SequenceTable,
        _asks_for_motion_correction,
        _fov_offset_m,
    )

    directory = entry.parent
    if not entry.is_file():
        raise FileNotFoundError(f"no sequence file {entry}")
    if not (directory / MANIFEST).is_file():
        raise FileNotFoundError(f"no design manifest beside {entry}")
    manifest = json.loads((directory / MANIFEST).read_text())
    # An imported chain's entry links to its first file, which the manifest
    # records under its own name.
    played = entry.resolve()
    digest = hashlib.sha256(played.read_bytes()).hexdigest()
    if (
        played.parent != directory.resolve()
        or manifest.get("files", {}).get(played.name) != digest
    ):
        raise ValueError(f"{entry} is not a file of design {manifest.get('id')}")
    named = user_parameter(header, DESIGN_PARAMETER)
    if named not in (None, "") and str(named).strip().lower() != manifest.get("id"):
        raise ValueError(
            f"the series was played from design {str(named).strip()}, "
            f"not from design {manifest.get('id')}"
        )
    return Design(
        directory=directory,
        table=SequenceTable.read(entry, fov_offset_m=_fov_offset_m(directory)),
        prospective_motion=_asks_for_motion_correction(entry),
    )


def _count(number_of: Any) -> int:
    """Return a dataset count; ISMRMRD raises instead of answering 0 for an absent group."""
    try:
        return int(number_of())
    except LookupError:
        return 0
