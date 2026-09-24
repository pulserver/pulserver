"""Content-addressed store of generated and imported designs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

#: Hexadecimal digits of a design identifier: 72 bits, three 24-bit integers,
#: each held exactly by a float32.
ID_DIGITS = 18
#: The file of a design directory that records what the design depends on.
MANIFEST = "manifest.json"
# A stage older than this is left by a process that ended mid-design.
_STALE_STAGE = 3600.0


def design_identity(
    plugin: str,
    limits: Mapping[str, Any],
    values: Mapping[str, Any],
    source: str = "",
) -> str:
    """Return the SHA-256, in hexadecimal, of everything a design depends on.

    ``plugin`` is the scanner-sequence plugin, empty for an imported chain;
    ``limits`` the scanner limits, conversion options and check limits;
    ``values`` the resolved protocol, prescription included, or what
    identifies an imported chain; ``source`` a digest of the code that
    designs. Independent of mapping order.
    """
    canonical = json.dumps(
        {
            "plugin": plugin,
            "limits": dict(limits),
            "values": dict(values),
            "source": source,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def design_id(identity: str) -> str:
    """Return the identifier of a design: the first ``ID_DIGITS`` digits of its identity."""
    return identity[:ID_DIGITS]


class DesignStore:
    """The designs under one directory, each in ``<id>/`` and immutable once written.

    A design directory holds the Pulseq files of its chain, its IR cache and
    :data:`MANIFEST`. It is written in a stage and renamed into place, so a
    reader sees a design whole or not at all, and two processes writing one
    design leave one directory.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def directory(self, design: str) -> Path:
        """Return the directory of a design, whether or not it exists."""
        return self.root / design

    def find(self, identity: str) -> str | None:
        """Return the identifier of the stored design with this identity, or ``None``.

        A design found is marked as used, for :meth:`prune`.

        Raises
        ------
        ValueError
            If the directory of the identifier holds a design of another identity.
        """
        design = design_id(identity)
        manifest = self.root / design / MANIFEST
        if not manifest.is_file():
            return None
        if json.loads(manifest.read_text()).get("identity") != identity:
            raise ValueError(
                f"design {design} holds another design with the same identifier"
            )
        os.utime(manifest)
        return design

    def manifest(self, design: str) -> dict[str, Any]:
        """Return a stored design's manifest.

        Raises
        ------
        FileNotFoundError
            If the store holds no design with this identifier.
        """
        path = self.root / design / MANIFEST
        if not path.is_file():
            raise FileNotFoundError(f"no design {design} in {self.root}")
        return json.loads(path.read_text())

    def stage(self) -> Path:
        """Create an empty directory to write a design into."""
        return Path(tempfile.mkdtemp(prefix=".stage-", dir=self.root))

    def commit(self, identity: str, staged: Path, manifest: Mapping[str, Any]) -> str:
        """Publish a staged directory as the design of ``identity``; return its identifier.

        The manifest is written with the identity, the identifier and the
        SHA-256 of every file of the stage. A design another process committed
        first is kept and the stage discarded.
        """
        design = design_id(identity)
        files = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(staged.iterdir())
            if path.is_file() and not path.is_symlink()
        }
        record = {**manifest, "id": design, "identity": identity, "files": files}
        (staged / MANIFEST).write_text(json.dumps(record, indent=2))
        try:
            staged.replace(self.root / design)
        except OSError:
            self.discard(staged)
            if self.find(identity) is None:
                raise
        return design

    @staticmethod
    def discard(staged: Path) -> None:
        shutil.rmtree(staged, ignore_errors=True)

    def __iter__(self) -> Iterator[str]:
        """Yield the identifier of every stored design."""
        for manifest in sorted(self.root.glob(f"*/{MANIFEST}")):
            if not manifest.parent.name.startswith("."):
                yield manifest.parent.name

    def prune(
        self,
        *,
        max_age: float | None = None,
        max_bytes: int | None = None,
        now: float | None = None,
    ) -> list[str]:
        """Remove designs, least recently used first; return their identifiers.

        A design is used when it is committed or found again. Designs unused
        for longer than ``max_age`` seconds are removed, then the least
        recently used until the store holds at most ``max_bytes``. Stages
        left by a process that ended mid-design are removed as well. A
        design a series still needs must not be pruned: the reconstruction
        side reads it by identifier.
        """
        now = time.time() if now is None else now
        for stage in self.root.glob(".stage-*"):
            if now - stage.stat().st_mtime > _STALE_STAGE:
                self.discard(stage)
        used = sorted(((self.root / d / MANIFEST).stat().st_mtime, d) for d in self)
        sizes = {d: _size(self.root / d) for _, d in used}
        removed = []
        total = sum(sizes.values())
        for last, design in used:
            too_old = max_age is not None and now - last > max_age
            too_big = max_bytes is not None and total > max_bytes
            if not (too_old or too_big):
                continue
            shutil.rmtree(self.root / design, ignore_errors=True)
            total -= sizes[design]
            removed.append(design)
        return removed


def _size(directory: Path) -> int:
    return sum(
        path.stat().st_size
        for path in directory.iterdir()
        if path.is_file() and not path.is_symlink()
    )
