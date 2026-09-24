"""Content-addressed store of generated and imported designs."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tarfile
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
#: Largest bundle a store receives, in bytes, compressed or not.
BUNDLE_LIMIT = 1 << 31
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
        return self._publish(identity, staged)

    def pack(self, design: str) -> bytes:
        """Return a stored design as a bundle: a gzip-compressed tar of its files.

        The bundle holds each file of the design directory under its name, the
        manifest among them, and each symbolic link as a link.

        Raises
        ------
        FileNotFoundError
            If the store holds no design with this identifier.
        """
        self.manifest(design)
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz", compresslevel=1) as bundle:
            for path in sorted((self.root / design).iterdir()):
                bundle.add(path, arcname=path.name, recursive=False)
        return buffer.getvalue()

    def receive(self, bundle: bytes) -> str:
        """Store a bundle :meth:`pack` made, after checking it; return its identifier.

        The bundle's manifest must name the identifier of its identity, every
        file must be one the manifest records, with the SHA-256 it records, and
        a symbolic link must name a file of the bundle. A design already stored
        is kept.

        Raises
        ------
        ValueError
            If the bundle is not a design :meth:`pack` makes, or a file
            differs from its manifest.
        """
        files, links = _read_bundle(bundle)
        try:
            manifest = json.loads(files[MANIFEST])
        except (KeyError, ValueError):
            raise ValueError("the bundle holds no manifest") from None
        identity = manifest.get("identity")
        if not isinstance(identity, str) or manifest.get("id") != design_id(identity):
            raise ValueError("the bundle's manifest names no design")
        recorded = manifest.get("files")
        if not isinstance(recorded, dict) or set(recorded) != set(files) - {MANIFEST}:
            raise ValueError("the bundle's files are not those its manifest records")
        for name, content in files.items():
            if (
                name != MANIFEST
                and hashlib.sha256(content).hexdigest() != recorded[name]
            ):
                raise ValueError(f"{name} differs from the manifest of its bundle")
        for name, target in links.items():
            if target not in files:
                raise ValueError(f"{name} links to {target!r}, which the bundle lacks")
        staged = self.stage()
        try:
            for name, content in files.items():
                (staged / name).write_bytes(content)
            for name, target in links.items():
                (staged / name).symlink_to(target)
            return self._publish(identity, staged)
        except BaseException:
            self.discard(staged)
            raise

    def _publish(self, identity: str, staged: Path) -> str:
        design = design_id(identity)
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


def _read_bundle(bundle: bytes) -> tuple[dict[str, bytes], dict[str, str]]:
    """Return a bundle's files and links, by name; refuse anything else a tar can hold."""
    if len(bundle) > BUNDLE_LIMIT:
        raise ValueError(f"the bundle exceeds {BUNDLE_LIMIT} bytes")
    files: dict[str, bytes] = {}
    links: dict[str, str] = {}
    total = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:*") as archive:
            for member in archive:
                name = member.name
                if not _flat(name) or name in files or name in links:
                    raise ValueError(f"the bundle holds an entry named {name!r}")
                if member.issym():
                    if not _flat(member.linkname):
                        raise ValueError(f"{name} links outside its bundle")
                    links[name] = member.linkname
                    continue
                if not member.isfile():
                    raise ValueError(f"{name} is neither a file nor a link")
                total += member.size
                if total > BUNDLE_LIMIT:
                    raise ValueError(f"the bundle exceeds {BUNDLE_LIMIT} bytes")
                files[name] = archive.extractfile(member).read()
    except (tarfile.TarError, EOFError, OSError) as error:
        raise ValueError(f"the bundle cannot be read: {error}") from None
    return files, links


def _flat(name: str) -> bool:
    return (
        bool(name) and "/" not in name and "\\" not in name and not name.startswith(".")
    )


def _size(directory: Path) -> int:
    return sum(
        path.stat().st_size
        for path in directory.iterdir()
        if path.is_file() and not path.is_symlink()
    )
