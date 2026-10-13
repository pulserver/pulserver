"""Replay of an ISMRMRD HDF5 file through a reconstruction plugin, in process."""

from __future__ import annotations

__all__ = ["reconstruct_file"]

import contextlib
import hashlib
import json
from pathlib import Path
from typing import Any

import ismrmrd
import ismrmrd.xsd

from ..plugin import ExamCache, ReconContext
from .application import run_application


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
        played = _played(header, store, design)
        fov_offset_m = None
        if played is not None:
            from ...proxy._enrich import enrich_header, header_fov_offset_m

            enrich_header(header, played.table)
            fov_offset_m = header_fov_offset_m(header)
        context = ReconContext(
            header=header,
            exam=ExamCache(path if exam_id is None else exam_id),
            config=config,
        )
        connection = _FileConnection(dataset, played, fov_offset_m)
        run_application(plugin, connection, context)
        return connection.sent
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
    digest = hashlib.sha256(entry.read_bytes()).hexdigest()
    if manifest.get("files", {}).get(entry.name) != digest:
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
