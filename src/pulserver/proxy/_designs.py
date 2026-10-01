"""Designs of a design store, found from the header of an incoming stream."""

from __future__ import annotations

__all__ = ["DESIGN_PARAMETER", "Design", "DesignCache"]

import json
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

from ..host._store import ID_DIGITS, MANIFEST
from ..mrd._metadata import user_parameter
from ..mrd._sequence import read_chain
from ._enrich import SequenceTable
from ._seqdesc import describe, message

#: Header user parameter naming the design a stream was played from: its
#: identifier, as ``GENERATED`` or ``IMPORTED`` replied it.
DESIGN_PARAMETER = "pulserver_design"

_ENTRY = "sequence.seq"
_IDENTIFIER = re.compile(rf"[0-9a-f]{{{ID_DIGITS}}}")

#: Designs a cache keeps tabulated.
_KEPT = 8


@dataclass(frozen=True)
class Design:
    """A stored design, as the reconstruction side reads it.

    Attributes
    ----------
    directory
        ``<store>/<id>/``.
    recon
        Reconstruction plugin the design names; empty when it names none, which
        leaves the choice to the client's config.
    table
        The readouts of the design's sequence chain, in play order.
    prospective_motion
        Whether the scan asks to be corrected for motion while it plays, which
        is what sends its series to the one reconstruction that may publish a
        pose.
    """

    directory: Path
    recon: str
    table: SequenceTable
    prospective_motion: bool = False

    @cached_property
    def description(self) -> str:
        """The MRD text message describing the chain to a simulation.

        See :func:`~pulserver.proxy._seqdesc.message`. Built on first use and
        kept with the design.
        """
        return message(
            [
                describe(seq, subsequence_index=index)
                for index, (_, seq) in enumerate(read_chain(self.directory / _ENTRY))
            ]
        )


class DesignCache:
    """The designs of a store, the most recently read kept tabulated.

    Reading a design tabulates its sequence chain, which is the expensive
    part of accepting a series; concurrent streams of one design wait for the
    first to finish rather than tabulating it again. A design that is no
    longer among the most recently read is read again when next named. The
    store is read and never written.
    """

    def __init__(self, store: Path | str) -> None:
        self.store = Path(store)
        self._lock = threading.Lock()
        self._building: dict[Path, threading.Lock] = {}
        self._designs: OrderedDict[Path, Design] = OrderedDict()

    def locate(self, header: Any) -> Path:
        """Return the directory of the design a header names.

        The design is named by :data:`DESIGN_PARAMETER`, a
        ``userParameterString`` holding the identifier.

        Raises
        ------
        ValueError
            If the header names no design, or names it by something that is
            not an identifier.
        FileNotFoundError
            If the store holds no such design.
        """
        named = user_parameter(header, DESIGN_PARAMETER)
        if named in (None, ""):
            raise ValueError(f"the header carries no {DESIGN_PARAMETER}")
        design = str(named).strip().lower()
        if not _IDENTIFIER.fullmatch(design):
            raise ValueError(
                f"{DESIGN_PARAMETER}={named!r} is not a design identifier: "
                f"{ID_DIGITS} hexadecimal digits"
            )
        directory = self.store / design
        if not (directory / MANIFEST).is_file():
            raise FileNotFoundError(f"no design {design} in {self.store}")
        return directory

    def read(self, directory: Path) -> Design:
        """Return a design directory read, tabulating its chain unless kept."""
        directory = Path(directory)
        with self._lock:
            building = self._building.setdefault(directory, threading.Lock())
        with building:
            with self._lock:
                if directory in self._designs:
                    self._designs.move_to_end(directory)
                    return self._designs[directory]
            manifest = json.loads((directory / MANIFEST).read_text())
            design = Design(
                directory=directory,
                recon=str(manifest.get("recon", "")),
                table=SequenceTable.read(directory / _ENTRY),
                prospective_motion=_asks_for_motion_correction(directory / _ENTRY),
            )
            with self._lock:
                self._designs[directory] = design
                while len(self._designs) > _KEPT:
                    forgotten, _ = self._designs.popitem(last=False)
                    self._building.pop(forgotten, None)
            return design

    def resolve(self, header: Any) -> Design:
        """Return the design a header names; see :meth:`locate` and :meth:`read`."""
        return self.read(self.locate(header))


def _asks_for_motion_correction(entry: Path) -> bool:
    """Whether a sequence file asks to be corrected for motion while it plays.

    Read from the preamble rather than the cache: a design is resolved before
    anything has been converted.
    """
    try:
        with entry.open("r", errors="replace") as f:
            inside = False
            for line in f:
                stripped = line.strip()
                if stripped.startswith("["):
                    if inside:
                        return False
                    inside = stripped.startswith("[DEFINITIONS]")
                    continue
                if inside and stripped.startswith("EnablePmc"):
                    parts = stripped.split()
                    return len(parts) > 1 and parts[1] not in ("0", "0.0")
    except OSError:
        return False
    return False
