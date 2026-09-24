"""The limits a design call carries: scanner limits, IR conversion options and check limits."""

from __future__ import annotations

import cmath
import re
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pypulseqpp as pp
from pypulseqpp import safety

from .. import ir

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


def split_limits(
    limits: Mapping[str, Any],
) -> tuple[pp.Opts, dict[str, Any], ir.CheckLimits]:
    """Separate a call's limits into scanner limits, IR conversion options and check limits.

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
    """Read the nerve and resonance limits and the VOP entries among a call's limits.

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
