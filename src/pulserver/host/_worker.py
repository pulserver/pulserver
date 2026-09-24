"""Design and conversion calls, run in worker processes."""

from __future__ import annotations

import cmath
import hashlib
import inspect
import re
from collections.abc import Mapping
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pypulseqpp as pp
from pypulseqpp import safety

from .. import __version__, ir
from ..design import ScannerSequence, load_plugin
from ..protocol import Parameter, Validation, prescribed_offset, prescribed_rotation

_IR_OPTIONS = ("ir_vendor", "ir_label_column_map", "ir_cache_ext")
_CHECK_PREFIXES = ("pns_", "forbidden_band_", "vop_")
_CHRONAXIE = {
    "pns_chronaxie": "chronaxie",
    "pns_rheobase": "rheobase",
    "pns_alpha": "alpha",
}
_SAFE = re.compile(r"pns_([xyz])_(a[123]|tau[123]|stim_limit|g_scale)")
_BAND = re.compile(r"forbidden_band_\d+")
_VOP = ("vop_file", "vop_drive_per_hz", "vop_default_shim")


@lru_cache(maxsize=32)
def _cached(path: str, mtime_ns: int) -> ScannerSequence:  # noqa: ARG001 -- part of the key
    return load_plugin(Path(path))


def _plugin(path: str) -> ScannerSequence:
    """Return the plugin at ``path``, imported again when the file changed."""
    return _cached(path, Path(path).stat().st_mtime_ns)


def split_limits(
    limits: Mapping[str, Any],
) -> tuple[pp.Opts, dict[str, Any], ir.CheckLimits]:
    """Separate a session's limits into scanner limits, IR conversion options and check limits.

    Keys starting with ``ir_`` are conversion options: ``ir_vendor``,
    ``ir_label_column_map`` (three integers separated by spaces) and
    ``ir_cache_ext``. Keys starting with ``pns_``, ``forbidden_band_`` and
    ``vop_`` are the check limits of :func:`check_limits`. The other keys are
    ``pypulseqpp.Opts`` keyword arguments.

    Raises
    ------
    ValueError
        If an ``ir_`` key is not a conversion option, or the check limits
        are malformed.
    """
    unknown = [k for k in limits if k.startswith("ir_") and k not in _IR_OPTIONS]
    if unknown:
        raise ValueError(f"not IR conversion options: {unknown}")
    system = pp.Opts(
        **{
            k: v
            for k, v in limits.items()
            if not k.startswith(("ir_", *_CHECK_PREFIXES))
        }
    )
    options: dict[str, Any] = {}
    if "ir_vendor" in limits:
        options["vendor"] = int(limits["ir_vendor"])
    if "ir_label_column_map" in limits:
        values = str(limits["ir_label_column_map"]).split()
        options["label_column_map"] = tuple(int(v) for v in values)
    if "ir_cache_ext" in limits:
        options["cache_ext"] = str(limits["ir_cache_ext"])
    return system, options, check_limits(limits)


def check_limits(limits: Mapping[str, Any]) -> ir.CheckLimits:
    """Read the nerve and resonance limits and the VOP entries among a session's limits.

    - ``pns_chronaxie`` (s), ``pns_rheobase`` (T/m/s) and optionally
      ``pns_alpha`` give a chronaxie nerve model; ``pns_<axis>_<field>``, for
      the axes ``x``, ``y`` and ``z`` and the fields of a SAFE description
      (``a1``-``a3``, ``tau1``-``tau3`` in ms, ``stim_limit`` in T/m/s,
      ``g_scale``), give a SAFE one. ``pns_limit`` is the largest response
      allowed, as a fraction of the model's threshold, 1 by default.
    - ``forbidden_band_<n>`` is one band: its physical axis (``x``, ``y``,
      ``z``, or ``all``), its lowest and highest frequency in Hz and,
      optionally, the largest amplitude allowed in it in mT/m, separated by
      spaces.
    - ``vop_file`` is a ``.mat`` or ``.npz`` file of VOPs the host can read,
      whose SAR ratios :func:`pulserver.ir.sar_ratios` writes into the cache;
      ``vop_drive_per_hz`` the relative channel drive per Hz of RF amplitude,
      one value or one per channel, 1 by default; and ``vop_default_shim`` the
      magnitude and phase in radians of each channel's weight for a pulse
      played without an RF shim, equal weights by default. Values are
      separated by spaces.

    Raises
    ------
    ValueError
        If a key of those families is not one of them, a model mixes the two
        kinds or misses a field, a band or a shim is malformed, or VOP limits
        are given without a file.
    """
    keys = [k for k in limits if k.startswith(_CHECK_PREFIXES)]
    unknown = [
        k
        for k in keys
        if k not in (*_CHRONAXIE, "pns_limit", *_VOP)
        and not _SAFE.fullmatch(k)
        and not _BAND.fullmatch(k)
    ]
    if unknown:
        raise ValueError(f"not check limits: {unknown}")
    arguments: dict[str, Any] = {"pns": _nerve_model(limits)}
    if "pns_limit" in limits:
        arguments["pns_limit"] = float(limits["pns_limit"])
    numbered = sorted(
        (k for k in keys if _BAND.fullmatch(k)), key=lambda k: int(k.rsplit("_", 1)[1])
    )
    arguments["bands"] = tuple(_band(str(limits[k])) for k in numbered)
    if "vop_file" in limits:
        arguments["vops"] = Path(str(limits["vop_file"]))
        drive = [float(v) for v in str(limits.get("vop_drive_per_hz", 1.0)).split()]
        arguments["drive_per_hz"] = drive[0] if len(drive) == 1 else tuple(drive)
        if "vop_default_shim" in limits:
            arguments["default_shim"] = _shim(str(limits["vop_default_shim"]))
    elif any(k.startswith("vop_") for k in keys):
        raise ValueError("the vop_ limits need a vop_file")
    return ir.CheckLimits(**arguments)


def _nerve_model(limits: Mapping[str, Any]) -> Any:
    chronaxie = {
        name: float(limits[k]) for k, name in _CHRONAXIE.items() if k in limits
    }
    safe = [_SAFE.fullmatch(k) for k in limits]
    safe = {m.groups(): float(limits[m.group(0)]) for m in safe if m}
    if chronaxie and safe:
        raise ValueError("a PNS model is a chronaxie model or a SAFE one, not both")
    if chronaxie:
        return safety.ChronaxieModel(**chronaxie)
    if safe:
        fields = ("a1", "a2", "a3", "tau1", "tau2", "tau3", "stim_limit", "g_scale")
        missing = [
            f"pns_{axis}_{field}"
            for axis in "xyz"
            for field in fields
            if (axis, field) not in safe
        ]
        if missing:
            raise ValueError(f"the SAFE model misses {missing}")
        return SimpleNamespace(
            **{
                axis: SimpleNamespace(**{f: safe[axis, f] for f in fields})
                for axis in "xyz"
            }
        )
    return None


def _shim(text: str) -> tuple[complex, ...]:
    try:
        values = [float(v) for v in text.split()]
    except ValueError:
        values = []
    if not values or len(values) % 2:
        raise ValueError(
            f"a default shim is a magnitude and a phase per channel: {text!r}"
        )
    return tuple(
        cmath.rect(magnitude, phase)
        for magnitude, phase in zip(values[::2], values[1::2], strict=True)
    )


def _band(text: str) -> safety.ForbiddenBand:
    words = text.split()
    try:
        if len(words) not in (3, 4) or words[0] not in ("x", "y", "z", "all"):
            raise ValueError
        low, high, *tolerance = (float(w) for w in words[1:])
    except ValueError:
        raise ValueError(
            f"a forbidden band is an axis, two frequencies and a tolerance: {text!r}"
        ) from None
    axis = None if words[0] == "all" else words[0]
    return safety.ForbiddenBand(axis, low, high, tolerance[0] if tolerance else 0.0)


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
    """Design into ``directory``, check and convert the result, and name its reconstruction.

    The design is written in the logical frame and checked in the physical
    frame of the rotation the resolved protocol carries; the conversion shifts
    it to the protocol's field-of-view offset and, when the limits name VOPs,
    writes each subsequence's :func:`pulserver.ir.sar_ratios` into the cache. A
    design that fails a check of :func:`pulserver.ir.check` is returned as an
    invalid request carrying the problems. The file list is empty and the cache
    file name ``None`` for an invalid request; what was written is left for the
    caller to discard.
    """
    system, options, checked = split_limits(limits)
    plugin = _plugin(path)
    validation, paths = plugin.generate(system, request, Path(directory))
    if not paths:
        return validation, paths, None, plugin.recon
    rotation = prescribed_rotation(validation.values)
    problems = ir.check(paths[0], system, rotation=rotation, limits=checked)
    if problems:
        refused = replace(validation, valid=False, info="; ".join(problems))
        return refused, [], None, plugin.recon
    offset = prescribed_offset(validation.values)
    cache = _converted(paths[0], system, checked, offset, options)
    return validation, paths, cache.name, plugin.recon


def chain(first: str) -> list[str]:
    return [str(path) for path in ir.chain(first)]


def check(limits: Mapping[str, Any], seq_path: str, rotation: Any = None) -> list[str]:
    system, _, checked = split_limits(limits)
    return ir.check(seq_path, system, rotation=rotation, limits=checked)


def convert(
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
