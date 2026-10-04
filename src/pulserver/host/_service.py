"""The design calls of an interpreter host process, each a function of its inputs.

A call returns the reply the interpreter parses: the text blocks of
:mod:`pulserver.protocol` and one status line. Nothing outlives a call but
the designs it adds to a :class:`~pulserver.host.DesignStore`.
"""

from __future__ import annotations

import datetime
import functools
import hashlib
import inspect
import logging
import shutil
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import pypulseqpp as pp

from .. import __version__, _plugins, ir
from .._plugins import PluginPath
from ..design import RfLayout, SequencePlugin, load_plugin
from ..protocol import (
    Parameter,
    ProtocolKey,
    Validation,
    format_listing,
    format_rf_definitions,
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

_log = logging.getLogger("pulserver.host")

# The file name the interpreter loads in a design.
_ENTRY = "sequence.seq"


class CallError(Exception):
    """A design call that fails with an ``ERROR`` reply."""


def plugin_path(plugins: PluginPath, plugin: str) -> Path:
    """Return the file of a scanner-sequence plugin, ``<plugin>.py`` of the first of ``plugins`` holding it.

    Raises
    ------
    CallError
        If the name is not a plugin name or no directory holds it.
    """
    try:
        return _plugins.find(plugins, plugin)
    except (ValueError, FileNotFoundError) as error:
        raise CallError(str(error)) from None


def list_protocol(
    plugins: PluginPath,
    plugin: str,
    limits: Mapping[str, Any] | None = None,
    rf_definitions: bool = False,
) -> str:
    """Reply ``PROTOCOL`` and the plugin's listing block, and its RF definitions where asked.

    The listing depends on the plugin file and the installed packages only.
    With ``rf_definitions``, the ``[RfDefinitions]`` block of the plugin's
    evaluation at its default protocol under ``limits`` follows it, naming the
    definitions, and the peak amplitudes, the RF layouts of later validations
    refer to. No block follows where the evaluation states no RF layout. Where
    it is invalid, a warning is logged and the listing is replied without the
    block. ``limits`` is not read unless the definitions are asked for.

    Raises
    ------
    CallError
        If ``rf_definitions`` is asked without ``limits``.
    """
    path = str(plugin_path(plugins, plugin))
    reply = "PROTOCOL\n" + format_listing(_listing(path))
    if not rf_definitions:
        return reply
    if limits is None:
        raise CallError(
            "the RF definitions are evaluated under scanner limits, and the call "
            "carries none"
        )
    layout = _default_rf_layout(
        path, _read(limits), f"the RF definitions of {plugin} are not listed"
    )
    if layout is not None:
        reply += format_rf_definitions(layout.instances)
    return reply


def validate(
    plugins: PluginPath,
    plugin: str,
    limits: Mapping[str, Any],
    block: str,
    rf_layout: bool = False,
) -> str:
    """Reply ``VALID <seconds>`` or ``INVALID``, an ``INFO`` line and the value block.

    The plugin evaluates the request under the design limits; the sequence is
    not generated. The duration is ``?`` where the plugin states no scan time.
    With ``rf_layout``, a valid reply ends with the ``[RfLayout]`` block of the
    evaluation where the plugin states one, with amplitudes over the peaks the
    listing states, read by evaluating the plugin again at its default
    protocol. Where that evaluation is invalid, a warning is logged and the
    amplitudes are the layout's own.
    """
    path = str(plugin_path(plugins, plugin))
    listing = _listing(path)
    request = _request(block, listing)
    limits = _read(limits)
    validation = _validated(path, limits, request)
    listed: list[float] = []
    if rf_layout and validation.valid and validation.rf_layout is not None:
        default = _default_rf_layout(
            path,
            limits,
            f"the RF layout of {plugin} is not stated over the listed peaks",
        )
        if default is not None:
            listed = [
                definition.peak_hz for definition in default.instances.definitions
            ]
    return format_validation(
        validation, listing, rf_layout=rf_layout, listed_peak_hz=listed
    )


def generate(
    plugins: PluginPath,
    plugin: str,
    limits: Mapping[str, Any],
    block: str,
    store: DesignStore,
    push: str | None = None,
) -> str:
    """Reply ``GENERATED <id>`` for the design a request resolves to.

    A design already stored for the resolved protocol, the plugin source, the
    package versions and the limits is returned without designing again.
    Otherwise the sequence is designed once, under the scanner limits
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
    requested = {key: p.value for key, p in listing.items() if p.editable}
    requested.update(request)
    validation = scanner.validate(design, requested)
    if validation.valid and validation.values != requested:
        validation = scanner.validate(design, validation.values)
    if not validation.valid:
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
        accepted, paths = scanner.design(design, validation.values, staged)
        if not accepted.valid:
            raise CallError(accepted.info)
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
            "source": source,
            "scan_time": validation.duration,
            "fov_offset_mm": [1e3 * value for value in offset],
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
    """Return limits with the digest of each file they name: the VOP and vendor files.

    A design converted under one file is not the design of another file
    written to the same path.

    Raises
    ------
    CallError
        If a named file cannot be read.
    """
    identified = dict(limits)
    for key in ("vop_file", "ir_vendor_file"):
        if key not in limits:
            continue
        try:
            content = Path(str(limits[key])).read_bytes()
        except OSError as error:
            raise CallError(f"cannot read the {key}: {error}") from None
        identified[f"{key}_sha256"] = hashlib.sha256(content).hexdigest()
    return identified


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


def _request(
    block: str, listing: Mapping[ProtocolKey, Parameter]
) -> dict[ProtocolKey, Any]:
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
def _cached(path: str, mtime_ns: int) -> SequencePlugin:  # noqa: ARG001 -- part of the key
    return load_plugin(Path(path))


def _plugin(path: str) -> SequencePlugin:
    """Return the plugin at ``path``, imported again when the file changed."""
    return _cached(path, Path(path).stat().st_mtime_ns)


def _listing(path: str) -> dict[ProtocolKey, Parameter]:
    return _plugin(path).listing()


def _source(path: str) -> str:
    """Return a digest of the code that designs with the plugin at ``path``.

    Covers the plugin file, the source file of the app it binds, and the
    installed versions of pypulseqpp and pulserver. A :func:`functools.partial`
    app is covered through the function it wraps. Modules the app imports from
    elsewhere are covered only through those versions.
    """
    digest = hashlib.sha256(Path(path).read_bytes())
    app = _plugin(path).app
    while isinstance(app, functools.partial):
        app = app.func
    try:
        module = inspect.getsourcefile(app)
    except TypeError:
        module = None
    if module:
        digest.update(Path(module).read_bytes())
    digest.update(f"pypulseqpp {pp.__version__} pulserver {__version__}".encode())
    return digest.hexdigest()


def _validated(
    path: str, limits: Mapping[str, Any], request: Mapping[ProtocolKey, Any]
) -> Validation:
    return _plugin(path).validate(design_system(limits), request)


def _default_rf_layout(
    path: str, limits: Mapping[str, Any], unavailable: str
) -> RfLayout | None:
    """Return the RF layout of the plugin's evaluation at its default protocol, the RF the listing states.

    ``None`` where the evaluation states none. Where it is invalid, a warning
    that begins with ``unavailable`` is logged and the result is ``None``. The
    evaluation is made once per plugin file, as last modified, and limits.
    """
    evaluated = _default_validation(
        path, Path(path).stat().st_mtime_ns, tuple(sorted(limits.items()))
    )
    if not evaluated.valid:
        _log.warning(
            "%s: its default protocol is invalid: %s", unavailable, evaluated.info
        )
    return evaluated.rf_layout


@lru_cache(maxsize=32)
def _default_validation(path: str, mtime_ns: int, limits: tuple) -> Validation:  # noqa: ARG001 -- part of the key
    return _validated(path, dict(limits), {})


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
