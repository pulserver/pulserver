"""Replay of an ISMRMRD HDF5 file through a reconstruction plugin, in process."""

from __future__ import annotations

__all__ = ["reconstruct_file"]

import contextlib
from typing import Any

import ismrmrd
import ismrmrd.xsd

from ..plugin import ExamCache, ReconContext
from .application import run_application


class _FileConnection:
    """Connection reading a dataset and keeping what is sent."""

    def __init__(self, dataset: Any, design: Any = None) -> None:
        self._dataset = dataset
        self._design = design
        self.sent: list[Any] = []

    def __iter__(self) -> Any:
        if self._design is None:
            return self._items()
        from ...proxy._proxy import _enriched

        return _enriched(self._items(), self._design)

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
        design = None
        if store is not None:
            from ...proxy._designs import DesignCache
            from ...proxy._enrich import enrich_header

            design = DesignCache(store).resolve(header)
            enrich_header(header, design.table)
        context = ReconContext(
            header=header,
            exam=ExamCache(path if exam_id is None else exam_id),
            config=config,
        )
        connection = _FileConnection(dataset, design)
        run_application(plugin, connection, context)
        return connection.sent
    finally:
        with contextlib.suppress(Exception):
            dataset.close()


def _count(number_of: Any) -> int:
    """Return a dataset count; ISMRMRD raises instead of answering 0 for an absent group."""
    try:
        return int(number_of())
    except LookupError:
        return 0
