"""Reconstructing a file the way the scanner's stream is reconstructed.

An MRD file holds what a scan sent: the header first, then the acquisitions in
the order they were acquired. That is the same thing a scanner connection
delivers, so a file needs no second reconstruction path -- only something to
read it and something to catch what comes back.

Both are here, and both are in-process: no socket, no port, no server to start.
:func:`reconstruct_file` opens the file, hands the header to the plugin as its
context, and drives the ordinary lifecycle over the acquisitions, so a plugin
reconstructs a file through exactly the hooks it reconstructs a scan through.
"""

from __future__ import annotations

__all__ = ["reconstruct_file"]

import contextlib
from typing import Any

import ismrmrd
import ismrmrd.xsd

from ..plugin import ReconContext
from .application import run_application


class _FileConnection:
    """A connection whose stream is a file and whose peer is a list.

    Satisfies what :func:`run_application` asks of a connection -- iterate the
    incoming items, and accept the outgoing ones -- without a socket.
    """

    def __init__(self, dataset: Any) -> None:
        self._dataset = dataset
        self.sent: list[Any] = []

    def __iter__(self) -> Any:
        # Waveforms first: a scanner sends them ahead of the acquisitions, and
        # the runtime hands every bucket the same set.
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
) -> list[Any]:
    """Drive one plugin over one MRD file and return what it produced.

    Parameters
    ----------
    plugin
        The :class:`~pulserver.ReconPlugin` to reconstruct through.
    path
        An ISMRMRD HDF5 file.
    group
        The HDF5 group the scan was written under.
    exam_id
        Identifier for the exam-scoped artifact cache. Defaults to ``path``,
        so reconstructing the same file twice shares nothing it should not.
    config
        Configuration payload, as an inline connection would carry.

    Returns
    -------
    list
        Everything the plugin emitted, in order -- ``ismrmrd.Image`` objects,
        or DICOM datasets from a plugin that asked for them.

    Raises
    ------
    FileNotFoundError
        If there is no such file.
    ValueError
        If the file carries no XML header, and so does not say what was
        acquired.
    """
    dataset = ismrmrd.Dataset(path, group, create_if_needed=False)
    try:
        document = dataset.read_xml_header()
        if not document:
            raise ValueError(
                f"{path} carries no MRD XML header; a reconstruction needs the "
                "header the scanner sends ahead of the data"
            )
        context = ReconContext(
            header=ismrmrd.xsd.CreateFromDocument(document),
            exam=_exam_cache(path if exam_id is None else exam_id),
            config=config,
        )
        connection = _FileConnection(dataset)
        run_application(plugin, connection, context)
        return connection.sent
    finally:
        with contextlib.suppress(Exception):
            dataset.close()


def _count(number_of: Any) -> int:
    """How many of something the file holds, none being a normal answer.

    ISMRMRD raises rather than answering zero when a file carries no group at
    all, which is what an acquisition-only file looks like.
    """
    try:
        return int(number_of())
    except LookupError:
        return 0


def _exam_cache(exam_id: Any) -> Any:
    from ..plugin import ExamCache

    return ExamCache(exam_id)
