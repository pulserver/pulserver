"""Design and conversion calls, run in worker processes."""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import pypulseqpp as pp

from .. import __version__, ir
from ..design import ScannerSequence, load_plugin
from ..protocol import Parameter, Validation, prescribed_offset

_IR_OPTIONS = ("ir_vendor", "ir_label_column_map", "ir_cache_ext")


@lru_cache(maxsize=32)
def _cached(path: str, mtime_ns: int) -> ScannerSequence:  # noqa: ARG001 -- part of the key
    return load_plugin(Path(path))


def _plugin(path: str) -> ScannerSequence:
    """Return the plugin at ``path``, imported again when the file changed."""
    return _cached(path, Path(path).stat().st_mtime_ns)


def split_limits(limits: Mapping[str, Any]) -> tuple[pp.Opts, dict[str, Any]]:
    """Separate a session's limits into scanner limits and IR conversion options.

    Keys starting with ``ir_`` are conversion options: ``ir_vendor``,
    ``ir_label_column_map`` (three integers separated by spaces) and
    ``ir_cache_ext``. The other keys are ``pypulseqpp.Opts`` keyword arguments.

    Raises
    ------
    ValueError
        If an ``ir_`` key is not a conversion option.
    """
    unknown = [k for k in limits if k.startswith("ir_") and k not in _IR_OPTIONS]
    if unknown:
        raise ValueError(f"not IR conversion options: {unknown}")
    system = pp.Opts(**{k: v for k, v in limits.items() if not k.startswith("ir_")})
    options: dict[str, Any] = {}
    if "ir_vendor" in limits:
        options["vendor"] = int(limits["ir_vendor"])
    if "ir_label_column_map" in limits:
        values = str(limits["ir_label_column_map"]).split()
        options["label_column_map"] = tuple(int(v) for v in values)
    if "ir_cache_ext" in limits:
        options["cache_ext"] = str(limits["ir_cache_ext"])
    return system, options


def listing(path: str) -> dict[str, Parameter]:
    return _plugin(path).listing()


def source(path: str) -> str:
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


def validate(
    path: str, limits: Mapping[str, Any], request: Mapping[str, Any]
) -> Validation:
    return _plugin(path).validate(split_limits(limits)[0], request)


def generate(
    path: str, limits: Mapping[str, Any], request: Mapping[str, Any], directory: str
) -> tuple[Validation, list[str], str | None, str]:
    """Design into ``directory``, convert the result, and name its reconstruction.

    The design is written in the logical frame; the conversion shifts it to
    the field-of-view offset the resolved protocol carries. The cache file
    name is ``None`` for an invalid request, which writes nothing.
    """
    system, options = split_limits(limits)
    plugin = _plugin(path)
    validation, paths = plugin.generate(system, request, Path(directory))
    if not paths:
        return validation, paths, None, plugin.recon
    offset = prescribed_offset(validation.values)
    cache = ir.convert(paths[0], system, fov_offset=offset, **options)
    return validation, paths, cache.name, plugin.recon


def chain(first: str) -> list[str]:
    return [str(path) for path in ir.chain(first)]


def convert(
    limits: Mapping[str, Any],
    seq_path: str,
    fov_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> str:
    system, options = split_limits(limits)
    return ir.convert(seq_path, system, fov_offset=fov_offset, **options).name
