"""Unix-socket server answering PSD host processes."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import multiprocessing
import re
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any

from ..protocol import (
    PROTOCOL_END,
    Parameter,
    format_listing,
    format_validation,
    format_values,
    parse_values,
)
from . import _worker
from ._sessions import Session, SessionKey, SessionStore, revision_hash

LIMITS_BEGIN = "[Limits]"
LIMITS_END = "[Limits End]"
IMPORT_BEGIN = "[Import]"
IMPORT_END = "[Import End]"

# The file name the target loads in a revision.
_ENTRY = "sequence.seq"

_PLUGIN_NAME = re.compile(r"[A-Za-z0-9_\-]+")
_log = logging.getLogger("pulserver.host")


class CommandError(Exception):
    """A command that fails with an ``ERROR`` reply."""


def _limit(text: str) -> Any:
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            continue
    return text


def parse_limits(block: str) -> dict[str, Any]:
    """Read a limits block: ``pypulseqpp.Opts`` keyword arguments, one per line."""
    limits = {}
    for line in block.splitlines():
        if ": " in line and LIMITS_BEGIN not in line:
            name, value = line.split(": ", 1)
            limits[name.strip()] = _limit(value.strip())
    return limits


def format_limits(limits: dict[str, Any]) -> str:
    lines = [f"{name}: {value}" for name, value in limits.items()]
    return "\n".join([LIMITS_BEGIN, *lines, LIMITS_END]) + "\n"


def parse_import(block: str) -> Path:
    """Read an import block: the ``file`` line naming the first file of a chain."""
    for line in block.splitlines():
        if line.startswith("file: "):
            return Path(line.removeprefix("file: ").strip())
    raise CommandError("IMPORT needs a file line")


def format_import(path: Path | str) -> str:
    return f"{IMPORT_BEGIN}\nfile: {path}\n{IMPORT_END}\n"


class HostDaemon:
    """Design sessions for every PSD host process on this host.

    Each request is a command line, ``COMMAND <session> [plugin]``, followed by
    a block for ``OPEN`` (limits and ``ir_`` conversion options), for
    ``VALIDATE`` and ``GENERATE`` (protocol values) and for ``IMPORT`` (the
    file). A session opened without a plugin only imports sequence files.
    Every revision holds ``sequence.seq`` and the IR cache the target loads.
    Replies:

    - ``OPEN``, ``CLOSE``: ``OK``.
    - ``LIST_PROTOCOL``: ``PROTOCOL`` and a listing block.
    - ``VALIDATE``: ``VALID <seconds>`` or ``INVALID``, an ``INFO`` line and a
      value block.
    - ``GENERATE``: ``GENERATED <revision>``. A request that resolves to an
      already generated protocol returns that revision and makes it current.
    - ``IMPORT``: ``IMPORTED <revision>``. The file's ``NextSequence`` chain is
      copied into the revision; identical files return the revision holding
      them.

    Any command can instead reply with a single ``ERROR <message>`` line.
    Commands of one session run one at a time; sessions run concurrently, with
    plugin code in a pool of spawned worker processes.

    Parameters
    ----------
    base
        Directory holding ``bucket/``.
    plugins
        Directory of plugin files, ``<plugin>.py``.
    workers
        Worker process count.
    """

    def __init__(self, base: Path, plugins: Path, *, workers: int = 2) -> None:
        self.store = SessionStore(base)
        self.plugins = Path(plugins)
        self._workers = workers
        self._pool = self._new_pool()
        self._locks: defaultdict[SessionKey, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._listings: dict[tuple[Path, int], dict[str, Parameter]] = {}
        self._writers: set[asyncio.StreamWriter] = set()

    def _new_pool(self) -> ProcessPoolExecutor:
        context = multiprocessing.get_context("spawn")
        return ProcessPoolExecutor(self._workers, mp_context=context)

    def shutdown(self) -> None:
        """Stop the worker pool, terminating the workers and any running design call.

        A worker left running would keep the interpreter from exiting, and once
        the daemon is gone it waits on its call queue indefinitely.
        """
        workers = list((self._pool._processes or {}).values())
        self._pool.shutdown(wait=False, cancel_futures=True)
        for worker in workers:
            worker.terminate()

    async def serve(self, socket_path: Path) -> None:
        """Answer commands on a Unix socket until cancelled.

        Cancellation closes the socket and every client connection.
        """
        server = await asyncio.start_unix_server(
            self._connection, path=str(socket_path)
        )
        _log.info("serving %s on %s", self.store.bucket, socket_path)
        try:
            # Not serve_forever: cancelled, it waits for the clients to disconnect.
            await asyncio.get_running_loop().create_future()
        finally:
            server.close()
            for writer in list(self._writers):
                writer.close()

    async def _connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self._writers.add(writer)
        try:
            while line := await reader.readline():
                command, *args = line.decode().split()
                end = {
                    "OPEN": LIMITS_END,
                    "VALIDATE": PROTOCOL_END,
                    "GENERATE": PROTOCOL_END,
                    "IMPORT": IMPORT_END,
                }
                block = (
                    await self._block(reader, end[command]) if command in end else ""
                )
                writer.write((await self._reply(command, args, block)).encode())
                await writer.drain()
        except (ConnectionError, ValueError) as error:
            _log.warning("dropping connection: %s", error)
        finally:
            self._writers.discard(writer)
            writer.close()

    @staticmethod
    async def _block(reader: asyncio.StreamReader, end: str) -> str:
        lines = []
        while line := (await reader.readline()).decode():
            lines.append(line)
            if end in line:
                break
        return "".join(lines)

    async def _reply(self, command: str, args: list[str], block: str) -> str:
        handlers = {
            "OPEN": self._open,
            "LIST_PROTOCOL": self._list_protocol,
            "VALIDATE": self._validate,
            "GENERATE": self._generate,
            "IMPORT": self._import,
            "CLOSE": self._close,
        }
        try:
            if command not in handlers:
                raise CommandError(f"unknown command {command}")
            if not args:
                raise CommandError(f"{command} needs a session")
            return await handlers[command](SessionKey.parse(args[0]), args[1:], block)
        except CommandError as error:
            return f"ERROR {error}\n"
        except KeyError as error:
            return f"ERROR session {error.args[0]} is not open\n"
        except Exception as error:  # a plugin bug must not stop the daemon
            _log.exception("%s failed", command)
            return f"ERROR {' '.join(str(error).split()) or type(error).__name__}\n"

    def _plugin_path(self, plugin: str) -> Path:
        if not plugin:
            raise CommandError(
                "the session has no plugin; it only imports sequence files"
            )
        if not _PLUGIN_NAME.fullmatch(plugin):
            raise CommandError(f"invalid plugin name {plugin!r}")
        path = self.plugins / f"{plugin}.py"
        if not path.is_file():
            raise CommandError(f"no plugin {plugin!r} in {self.plugins}")
        return path

    async def _run(self, function: Any, *args: Any) -> Any:
        """Run a design call in the worker pool.

        A worker that dies fails this command and the pool is replaced.
        """
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(self._pool, function, *args)
        except BrokenProcessPool:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = self._new_pool()
            raise CommandError("the design worker exited during this command") from None

    async def _listing(self, session: Session) -> dict[str, Parameter]:
        """Return the listing of the session's plugin, cached until its file changes."""
        path = self._plugin_path(session.plugin)
        key = (path, path.stat().st_mtime_ns)
        if key not in self._listings:
            self._listings[key] = await self._run(_worker.listing, str(path))
        return self._listings[key]

    async def _open(self, key: SessionKey, args: list[str], block: str) -> str:
        if len(args) > 1:
            raise CommandError("OPEN takes a session and at most one plugin")
        plugin = args[0] if args else ""
        if plugin:
            self._plugin_path(plugin)
        try:
            self.store.open(key, plugin, parse_limits(block))
        except ValueError as error:
            raise CommandError(str(error)) from None
        return "OK\n"

    async def _list_protocol(
        self, key: SessionKey, _args: list[str], _block: str
    ) -> str:
        session = self.store.get(key)
        return "PROTOCOL\n" + format_listing(await self._listing(session))

    async def _validate(self, key: SessionKey, _args: list[str], block: str) -> str:
        session = self.store.get(key)
        async with self._locks[key]:
            listing = await self._listing(session)
            request = parse_values(block, listing)
            session.save_protocol(block)
            validation = await self._run(
                _worker.validate,
                str(self._plugin_path(session.plugin)),
                session.limits,
                request,
            )
        return format_validation(validation, listing)

    async def _generate(self, key: SessionKey, _args: list[str], block: str) -> str:
        session = self.store.get(key)
        path = str(self._plugin_path(session.plugin))
        async with self._locks[key]:
            listing = await self._listing(session)
            request = parse_values(block, listing)
            validation = await self._run(
                _worker.validate, path, session.limits, request
            )
            if not validation.valid:
                raise CommandError(validation.info)
            digest = revision_hash(session.plugin, session.limits, validation.values)
            revision = session.find(digest)
            if revision is not None:
                session.select(revision)
                return f"GENERATED {revision}\n"
            staged = session.stage()
            try:
                validation, paths, cache, recon = await self._run(
                    _worker.generate,
                    path,
                    session.limits,
                    validation.values,
                    str(staged),
                )
                if not paths:
                    raise CommandError(validation.info)
                (staged / "resolved.protocol").write_text(
                    format_values(validation.values, listing)
                )
                meta = {
                    "plugin": session.plugin,
                    "recon": recon,
                    "limits": session.limits,
                    "hash": digest,
                    "files": [*(Path(p).name for p in paths), cache],
                }
                (staged / "meta.json").write_text(json.dumps(meta, indent=2))
            except BaseException:
                session.discard(staged)
                raise
            return f"GENERATED {session.commit(digest, staged)}\n"

    async def _import(self, key: SessionKey, _args: list[str], block: str) -> str:
        session = self.store.get(key)
        source = parse_import(block)
        async with self._locks[key]:
            files = [Path(p) for p in await self._run(_worker.chain, str(source))]
            contents = [
                [f.name, hashlib.sha256(f.read_bytes()).hexdigest()] for f in files
            ]
            digest = revision_hash(session.plugin, session.limits, {"import": contents})
            revision = session.find(digest)
            if revision is not None:
                session.select(revision)
                return f"IMPORTED {revision}\n"
            staged = session.stage()
            try:
                for file in files:
                    shutil.copyfile(file, staged / file.name)
                entry = staged / _ENTRY
                if files[0].name != _ENTRY:
                    entry.symlink_to(files[0].name)
                await self._run(_worker.convert, session.limits, str(entry))
                meta = {
                    "plugin": session.plugin,
                    "limits": session.limits,
                    "hash": digest,
                    "source": str(source),
                    "files": sorted(p.name for p in staged.iterdir()),
                }
                (staged / "meta.json").write_text(json.dumps(meta, indent=2))
            except BaseException:
                session.discard(staged)
                raise
            return f"IMPORTED {session.commit(digest, staged)}\n"

    async def _close(self, key: SessionKey, _args: list[str], _block: str) -> str:
        self.store.get(key).close()
        return "OK\n"
