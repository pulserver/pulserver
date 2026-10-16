"""Store of generated and imported designs: one directory per scan, each design in it addressed by its content."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import secrets
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
# What design_identity returns; nothing else names a design directory.
_IDENTITY = re.compile(r"[0-9a-f]{64}")
# A stage older than this is left by a process that ended mid-design.
_STALE_STAGE = 3600.0
# A scan used more recently than this may be playing or reconstructing.
_IN_USE = 86400.0


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


def _umask() -> int:
    """Return the process umask, without leaving it changed."""
    mask = os.umask(0o022)
    os.umask(mask)
    return mask


class DesignStore:
    """The designs of one scan, each in ``<root>/<scan>/<id>/`` and immutable once written.

    A store root holds one directory per scan, named by whatever plays it: a
    scanner names it after the scan, and the reconstruction finds a series'
    design from the scan and the identifier its header carries. A design
    directory holds the Pulseq files of its chain, its IR cache and
    :data:`MANIFEST`. It is written in a stage and renamed into place, so a
    reader sees a design whole or not at all, and two processes writing one
    design leave one directory. A scan plays one design, so committing a
    design removes the scan's others. A design whose sequence sets
    ``EnablePmc`` is published with its scan's empty pose file,
    ``<root>/<scan>/motion.buf``.

    Parameters
    ----------
    root : Path | str
        Directory of scans; created when missing.
    scan : str
        Name of the scan's directory: one path component, not starting with
        ``.``.

    Raises
    ------
    ValueError
        If ``scan`` is not a directory name.
    """

    def __init__(self, root: Path | str, scan: str) -> None:
        if not _flat(str(scan)):
            raise ValueError(f"a scan is named by one directory name, not {scan!r}")
        self.root = Path(root)
        self.scan = str(scan)
        self.path = self.root / self.scan
        self.path.mkdir(parents=True, exist_ok=True)

    def directory(self, design: str) -> Path:
        """Return the directory of a design, whether or not it exists."""
        return self.path / design

    def find(self, identity: str) -> str | None:
        """Return the identifier of the stored design with this identity, or ``None``.

        A design found is marked as used, for :meth:`prune`.

        Raises
        ------
        ValueError
            If the directory of the identifier holds a design of another identity.
        """
        design = design_id(identity)
        manifest = self.path / design / MANIFEST
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
        path = self.path / design / MANIFEST
        if not path.is_file():
            raise FileNotFoundError(f"no design {design} in {self.path}")
        return json.loads(path.read_text())

    def stage(self) -> Path:
        """Create an empty directory to write a design into."""
        return Path(tempfile.mkdtemp(prefix=".stage-", dir=self.path))

    def commit(self, identity: str, staged: Path, manifest: Mapping[str, Any]) -> str:
        """Publish a staged directory as the design of ``identity``; return its identifier.

        The manifest is written with the identity, the identifier and the
        SHA-256 of every file of the stage. A design another process committed
        first is kept and the stage discarded. The scan's other designs are
        removed.
        """
        design = design_id(identity)
        files = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(staged.iterdir())
            if path.is_file() and not path.is_symlink()
        }
        record = {**manifest, "id": design, "identity": identity, "files": files}
        (staged / MANIFEST).write_text(json.dumps(record, indent=2))
        published = self._publish(identity, staged)
        for other in self:
            if other != published:
                shutil.rmtree(self.path / other, ignore_errors=True)
        return published

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
            for path in sorted((self.path / design).iterdir()):
                bundle.add(path, arcname=path.name, recursive=False)
        return buffer.getvalue()

    def receive(self, bundle: bytes, design: str | None = None) -> str:
        """Store a bundle :meth:`pack` made, after checking it; return its identifier.

        The bundle's manifest must name a SHA-256 identity and the identifier
        of that identity, which must be ``design`` when one is given; every
        file must be one the manifest records, with the SHA-256 it records, and
        a symbolic link must name a file of the bundle. Nothing is written
        before these checks pass. A design already stored is kept.

        Raises
        ------
        ValueError
            If the bundle is not a design :meth:`pack` makes, is not
            ``design``, or a file differs from its manifest.
        """
        files, links = _read_bundle(bundle)
        try:
            manifest = json.loads(files[MANIFEST])
        except (KeyError, ValueError):
            raise ValueError("the bundle holds no manifest") from None
        identity = manifest.get("identity")
        if (
            not isinstance(identity, str)
            or not _IDENTITY.fullmatch(identity)
            or manifest.get("id") != design_id(identity)
        ):
            raise ValueError("the bundle's manifest names no design")
        if design is not None and design_id(identity) != design:
            raise ValueError(
                f"the bundle holds design {design_id(identity)}, not {design}"
            )
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
        # A design exists to be read by something other than what wrote it --
        # on a scanner the reconstruction runs as another user than the design
        # service -- and tempfile.mkdtemp() makes the stage private. Publish it
        # as an ordinary mkdir would, and let a site that wants the store
        # private say so with its umask.
        staged.chmod(0o777 & ~_umask())
        try:
            staged.replace(self.path / design)
        except OSError:
            self.discard(staged)
            if self.find(identity) is None:
                raise
        # The scan opens its pose file once, as it starts, before any series
        # is reconstructed: it has to be there with the design.
        from ..proxy._motion import prepare

        prepare(self.path, self.path / design / "sequence.seq")
        return design

    @staticmethod
    def discard(staged: Path) -> None:
        shutil.rmtree(staged, ignore_errors=True)

    def __iter__(self) -> Iterator[str]:
        """Yield the identifier of every design of the scan."""
        for manifest in sorted(self.path.glob(f"*/{MANIFEST}")):
            if not manifest.parent.name.startswith("."):
                yield manifest.parent.name


def new_scan(now: float | None = None) -> str:
    """Return a name for a new scan directory: its time, then a random suffix."""
    stamp = time.strftime("%Y%m%dT%H%M%S", time.localtime(now))
    return f"{stamp}-{secrets.token_hex(3)}"


def find_scan(root: Path | str, design: str) -> str:
    """Return the newest scan of a store root that holds a design.

    Raises
    ------
    FileNotFoundError
        If no scan of the root holds the design.
    """
    holding = [
        scan
        for scan in scans(root)
        if (Path(root) / scan / design / MANIFEST).is_file()
    ]
    if not holding:
        raise FileNotFoundError(f"no scan of {root} holds design {design}")
    return max(holding, key=lambda scan: _last_use(Path(root) / scan))


def scans(root: Path | str) -> list[str]:
    """Return the name of every scan directory of a store root."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(
        entry.name for entry in root.iterdir() if entry.is_dir() and _flat(entry.name)
    )


def prune(
    root: Path | str,
    *,
    max_age: float | None = None,
    max_bytes: int | None = None,
    now: float | None = None,
) -> list[str]:
    """Remove scan directories of a store root, least recently used first; return their names.

    A scan is used when anything in its directory, or in a design of it, is
    written, or a design of it is found again. Scans unused for longer than
    ``max_age`` seconds are removed, then the least recently used until the
    root holds at most ``max_bytes``. A scan used within the last day is kept
    whatever the limits, since it may still be playing or reconstructing.
    Stages older than an hour, left by a process that ended mid-design, are
    removed as well.
    """
    root = Path(root)
    now = time.time() if now is None else now
    for stage in root.glob("*/.stage-*"):
        if now - stage.stat().st_mtime > _STALE_STAGE:
            shutil.rmtree(stage, ignore_errors=True)
    used = sorted((_last_use(root / scan), scan) for scan in scans(root))
    sizes = {scan: _size(root / scan) for _, scan in used}
    removed = []
    total = sum(sizes.values())
    for last, scan in used:
        too_old = max_age is not None and now - last > max_age
        too_big = max_bytes is not None and total > max_bytes
        if now - last < _IN_USE or not (too_old or too_big):
            continue
        shutil.rmtree(root / scan, ignore_errors=True)
        total -= sizes[scan]
        removed.append(scan)
    return removed


def _last_use(directory: Path) -> float:
    """Return when anything in a scan directory, or in a design of it, was last written."""
    times = [directory.stat().st_mtime]
    for entry in directory.iterdir():
        times.append(entry.lstat().st_mtime)
        if entry.is_dir() and not entry.is_symlink():
            times.extend(path.lstat().st_mtime for path in entry.iterdir())
    return max(times)


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
        for path in directory.rglob("*")
        if path.is_file() and not path.is_symlink()
    )
