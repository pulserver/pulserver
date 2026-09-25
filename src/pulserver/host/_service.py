"""The design calls of a PSD host process, each a function of its inputs.

A call returns the reply the interpreter parses: the text blocks of
:mod:`pulserver.protocol` and one status line. Nothing outlives a call but
the designs it adds to a :class:`~pulserver.host.DesignStore`.
"""

from __future__ import annotations

import datetime
import hashlib
import inspect
import re
import shutil
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import pypulseqpp as pp

from .. import __version__, ir
from ..design import ScannerSequence, load_plugin
from ..protocol import (
    Parameter,
    Validation,
    format_listing,
    format_validation,
    format_values,
    parse_values,
    prescribed_offset,
    prescribed_rotation,
)
from ._blocks import parse_import
from ._limits import design_system, split_limits
from ._push import push as push_design
from ._store import DesignStore, design_identity

_PLUGIN_NAME = re.compile(r"[A-Za-z0-9_\-]+")
# The file name the interpreter loads in a design.
_ENTRY = "sequence.seq"


class CallError(Exception):
    """A design call that fails with an ``ERROR`` reply."""


def plugin_path(plugins: Path | str, plugin: str) -> Path:
    """Return the file of a scanner-sequence plugin, ``<plugins>/<plugin>.py``.

    Raises
    ------
    CallError
        If the name is not a plugin name or no such file exists.
    """
    if not _PLUGIN_NAME.fullmatch(plugin):
        raise CallError(f"invalid plugin name {plugin!r}")
    path = Path(plugins) / f"{plugin}.py"
    if not path.is_file():
        raise CallError(f"no plugin {plugin!r} in {plugins}")
    return path


def list_protocol(plugins: Path | str, plugin: str) -> str:
    """Reply ``PROTOCOL`` and the plugin's listing block.

    The listing depends on the plugin file and the installed packages only.
    """
    listing = _listing(str(plugin_path(plugins, plugin)))
    return "PROTOCOL\n" + format_listing(listing)


def validate(
    plugins: Path | str, plugin: str, limits: Mapping[str, Any], block: str
) -> str:
    """Reply ``VALID <seconds>`` or ``INVALID``, an ``INFO`` line and the value block.

    The application is constructed under the design limits and its scan
    time computed; the sequence is not designed.
    """
    path = str(plugin_path(plugins, plugin))
    listing = _listing(path)
    request = _request(block, listing)
    return format_validation(_validated(path, _read(limits), request), listing)


def generate(
    plugins: Path | str,
    plugin: str,
    limits: Mapping[str, Any],
    block: str,
    store: DesignStore,
    push: str | None = None,
) -> str:
    """Reply ``GENERATED <id>`` for the design a request resolves to.

    A design already stored for the resolved protocol, the plugin source, the
    package versions and the limits is returned without designing again.
    Otherwise the application is designed once, under the scanner limits
    capped by the design limits ``design_max_grad`` and ``design_max_slew``,
    written in the logical frame, checked in the physical frame of the
    prescription rotation against the scanner limits with
    :func:`pulserver.ir.check`, converted to the IR cache at the prescribed
    field-of-view offset, and stored. A request that resolves to other values
    is designed from the resolved values, so a design is a function of its
    identifier. With ``push``, the URL of a design intake, the design is sent
    there unless the intake holds it already.

    Raises
    ------
    CallError
        If the request is invalid or the design fails a check, when nothing
        is stored; or if the design cannot be pushed, when it stays stored.
    """
    path = str(plugin_path(plugins, plugin))
    listing = _listing(path)
    request = _request(block, listing)
    limits = _read(limits)
    system, options, checked = split_limits(limits)
    design = design_system(limits)
    scanner = _plugin(path)
    requested = {name: p.value for name, p in listing.items() if p.editable}
    requested.update(request)
    app, validation = scanner.resolve(design, requested)
    if app is not None and validation.values != requested:
        app, validation = scanner.resolve(design, validation.values)
    if app is None:
        raise CallError(validation.info)
    source = _source(path)
    identity = design_identity(
        plugin, identified_limits(limits), validation.values, source
    )
    found = store.find(identity)
    if found is not None:
        return f"GENERATED {_pushed(store, found, push)}\n"
    staged = store.stage()
    try:
        paths = scanner.write(app, staged)
        rotation = prescribed_rotation(validation.values)
        problems = ir.check(paths[0], system, rotation=rotation, limits=checked)
        if problems:
            raise CallError("; ".join(problems))
        offset = prescribed_offset(validation.values)
        _converted(paths[0], system, checked, offset, options)
        (staged / "resolved.protocol").write_text(
            format_values(validation.values, listing)
        )
        manifest = {
            **_record(limits),
            "plugin": plugin,
            "recon": scanner.recon,
            "source": source,
            "scan_time": validation.duration,
        }
        design = store.commit(identity, staged, manifest)
    except BaseException:
        store.discard(staged)
        raise
    return f"GENERATED {_pushed(store, design, push)}\n"


def import_chain(
    limits: Mapping[str, Any],
    block: str,
    store: DesignStore,
    push: str | None = None,
) -> str:
    """Reply ``IMPORTED <id>`` for a sequence file, its chain and their IR cache.

    The files are identified by their names and contents and the
    prescription of the import block. A chain already stored is returned;
    otherwise it is copied, checked and converted as :func:`generate` does,
    and pushed as :func:`generate` pushes. The first file is also reachable as
    ``sequence.seq``.

    Raises
    ------
    CallError
        If the block is malformed, a file cannot be read, or the chain fails
        a check, when nothing is stored; or if the design cannot be pushed.
    """
    try:
        first, offset_mm, rotation = parse_import(block)
    except ValueError as error:
        raise CallError(str(error)) from None
    limits = _read(limits)
    files = [Path(p) for p in _chain(str(first))]
    contents = [[f.name, hashlib.sha256(f.read_bytes()).hexdigest()] for f in files]
    values = {
        "import": contents,
        "fov_offset_mm": list(offset_mm),
        "fov_rotation": rotation.ravel().tolist(),
    }
    identity = design_identity("", identified_limits(limits), values)
    found = store.find(identity)
    if found is not None:
        return f"IMPORTED {_pushed(store, found, push)}\n"
    staged = store.stage()
    try:
        for file in files:
            shutil.copyfile(file, staged / file.name)
        entry = staged / _ENTRY
        if files[0].name != _ENTRY:
            entry.symlink_to(files[0].name)
        problems = _check(limits, str(entry), rotation)
        if problems:
            raise CallError("; ".join(problems))
        offset = tuple(value * 1e-3 for value in offset_mm)
        _convert(limits, str(entry), offset)
        manifest = {
            **_record(limits),
            "plugin": "",
            "recon": "",
            "source": str(first),
            "fov_offset_mm": list(offset_mm),
            "fov_rotation": rotation.ravel().tolist(),
        }
        design = store.commit(identity, staged, manifest)
    except BaseException:
        store.discard(staged)
        raise
    return f"IMPORTED {_pushed(store, design, push)}\n"


def identified_limits(limits: Mapping[str, Any]) -> dict[str, Any]:
    """Return limits with the digest of the VOP file they name.

    A design whose SAR ratios came from one VOP file is not the design of
    another file written to the same path.

    Raises
    ------
    CallError
        If the VOP file cannot be read.
    """
    if "vop_file" not in limits:
        return dict(limits)
    try:
        digest = hashlib.sha256(Path(str(limits["vop_file"])).read_bytes()).hexdigest()
    except OSError as error:
        raise CallError(f"cannot read the VOP file: {error}") from None
    return {**limits, "vop_file_sha256": digest}


def call(name: str, **inputs: Any) -> tuple[int, str]:
    """Answer one design call: exit status 0 and its reply, or 1 and an ``ERROR`` line.

    ``name`` is ``list``, ``validate``, ``generate`` or ``import``, and
    ``inputs`` the keyword arguments of :func:`list_protocol`,
    :func:`validate`, :func:`generate` or :func:`import_chain`. An exception a
    plugin raises is an ``ERROR`` reply like any other.
    """
    calls = {
        "list": list_protocol,
        "validate": validate,
        "generate": generate,
        "import": import_chain,
    }
    try:
        return 0, calls[name](**inputs)
    except Exception as error:
        text = " ".join(str(error).split()) or type(error).__name__
        return 1, f"ERROR {text}\n"


def _pushed(store: DesignStore, design: str, push: str | None) -> str:
    if push:
        try:
            push_design(store, design, push)
        except (OSError, ValueError) as error:
            raise CallError(
                f"design {design} is stored but not pushed: {error}"
            ) from None
    return design


def _request(block: str, listing: Mapping[str, Parameter]) -> dict[str, Any]:
    try:
        return parse_values(block, listing)
    except ValueError as error:
        raise CallError(str(error)) from None


def _read(limits: Mapping[str, Any]) -> dict[str, Any]:
    """Return limits after checking that they are scanner limits, options and check limits."""
    try:
        split_limits(limits)
    except (TypeError, ValueError) as error:
        raise CallError(str(error)) from None
    return dict(limits)


def _record(limits: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "limits": dict(limits),
        "versions": {"pulserver": __version__, "pypulseqpp": pp.__version__},
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
    }


@lru_cache(maxsize=32)
def _cached(path: str, mtime_ns: int) -> ScannerSequence:  # noqa: ARG001 -- part of the key
    return load_plugin(Path(path))


def _plugin(path: str) -> ScannerSequence:
    """Return the plugin at ``path``, imported again when the file changed."""
    return _cached(path, Path(path).stat().st_mtime_ns)


def _listing(path: str) -> dict[str, Parameter]:
    return _plugin(path).listing()


def _source(path: str) -> str:
    """Return a digest of the code that designs with the plugin at ``path``.

    Covers the plugin file, the source file of the application it binds, and
    the installed versions of pypulseqpp and pulserver. Modules the
    application imports from elsewhere are covered only through those
    versions.
    """
    digest = hashlib.sha256(Path(path).read_bytes())
    try:
        module = inspect.getsourcefile(_plugin(path).app)
    except TypeError:
        module = None
    if module:
        digest.update(Path(module).read_bytes())
    digest.update(f"pypulseqpp {pp.__version__} pulserver {__version__}".encode())
    return digest.hexdigest()


def _validated(
    path: str, limits: Mapping[str, Any], request: Mapping[str, Any]
) -> Validation:
    return _plugin(path).validate(design_system(limits), request)


def _chain(first: str) -> list[str]:
    return [str(path) for path in ir.chain(first)]


def _check(limits: Mapping[str, Any], seq_path: str, rotation: Any = None) -> list[str]:
    system, _, checked = split_limits(limits)
    return ir.check(seq_path, system, rotation=rotation, limits=checked)


def _convert(
    limits: Mapping[str, Any],
    seq_path: str,
    fov_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> str:
    system, options, checked = split_limits(limits)
    return _converted(seq_path, system, checked, fov_offset, options).name


def _converted(
    seq_path: str,
    system: pp.Opts,
    checked: ir.CheckLimits,
    fov_offset: tuple[float, float, float],
    options: dict[str, Any],
) -> Path:
    ratios = None if checked.vops is None else ir.sar_ratios(seq_path, system, checked)
    return ir.convert(
        seq_path, system, fov_offset=fov_offset, sar_ratios=ratios, **options
    )
