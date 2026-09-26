"""The enriched stream of a series held on disk until a slot frees."""

from __future__ import annotations

__all__ = ["QueueFile"]

import contextlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import ismrmrd

from ..recon._runtime.writers import header_document

_GROUP = "dataset"


class QueueFile:
    """An ISMRMRD file holding one series as the worker will receive it.

    The header and every acquisition are the enriched ones, so replaying the
    file gives a worker the stream the live path would have given it.
    Acquisitions and waveforms are stored in separate tables, so replay yields
    the waveforms first, which is how every emitted unit carries all of them.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._dataset: Any = ismrmrd.Dataset(
            str(self.path), _GROUP, create_if_needed=True
        )

    def write_header(self, header: Any) -> None:
        self._dataset.write_xml_header(header_document(header))

    def append(self, item: Any) -> None:
        """Store an acquisition or a waveform; anything else is not queued."""
        if isinstance(item, ismrmrd.Acquisition):
            self._dataset.append_acquisition(item)
        elif isinstance(item, ismrmrd.Waveform):
            self._dataset.append_waveform(item)

    def close(self) -> None:
        dataset, self._dataset = self._dataset, None
        if dataset is not None:
            dataset.close()

    def unlink(self) -> None:
        """Close the file and remove it."""
        with contextlib.suppress(Exception):
            self.close()
        self.path.unlink(missing_ok=True)

    def __iter__(self) -> Iterator[Any]:
        dataset = ismrmrd.Dataset(str(self.path), _GROUP, create_if_needed=False)
        try:
            for index in range(_count(dataset.number_of_waveforms)):
                yield dataset.read_waveform(index)
            for index in range(_count(dataset.number_of_acquisitions)):
                yield dataset.read_acquisition(index)
        finally:
            with contextlib.suppress(Exception):
                dataset.close()


def _count(number_of: Any) -> int:
    """Return a dataset count; ISMRMRD raises instead of answering 0 for an absent group."""
    try:
        return int(number_of())
    except LookupError:
        return 0
