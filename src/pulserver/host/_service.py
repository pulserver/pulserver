"""The design calls of an interpreter host process, each a function of its inputs.

A call returns the reply the interpreter parses: the text blocks of
:mod:`pulserver.protocol` and one status line. Nothing outlives a call but
the designs it adds to a :class:`~pulserver.host.DesignStore` and, under a
warm server, what the server keeps for the calls after it
(:mod:`pulserver.host._memo`): the replies of ``list`` and ``validate``, the
RF a plugin lists and the slew rate each request was lowered to.
"""

from __future__ import annotations

import copy
import datetime
import functools
import hashlib
import inspect
import json
import logging
import math
import re
import shutil
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pypulseqpp as pp
from pypulseqpp import safety

from .. import __version__, _plugins, ir
from .._plugins import PluginPath
from ..design import (
    OptionError,
    OptionParam,
    SequencePlugin,
    load_plugin,
)
from ..design._plugin import requested_options
from ..ir._checks import _RASTERS, enables_pmc
from ..ir._convert import _payload, _played_time, _read_chain, _write_cache
from ..mrd._sequence import designed_chain
from ..protocol import (
    Parameter,
    ProtocolKey,
    TRPreset,
    UIParam,
    Validation,
    format_listing,
    format_rf_definitions,
    format_validation,
    format_values,
    parse_values,
    prescribed_offset,
    prescribed_rotation,
)
from . import _memo, _slew
from ._blocks import import_averages, parse_import
from ._limits import design_system, follows_poses, split_limits
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


class _Variant(NamedTuple):
    """A plugin file and the options it is loaded with, by name."""

    path: str
    options: tuple[tuple[str, Any], ...] = ()


def _variant(plugins: PluginPath, plugin: str, block: str = "") -> _Variant:
    """Return the plugin file of ``plugin`` and the options the option entries of ``block`` ask for.

    The option entries are those of the plugin loaded at its defaults; the
    other lines of ``block`` are not read here.

    Raises
    ------
    CallError
        If an option entry does not hold the index of one of its choices.
    """
    path = str(plugin_path(plugins, plugin))
    if not block:
        return _Variant(path)
    default = _plugin(_Variant(path))
    entries = {
        key: parameter
        for key, parameter in default.listing().items()
        if isinstance(default.protocol.get(key), OptionParam)
    }
    try:
        values = parse_values(block, entries, unlisted="skip")
        options = requested_options(default.protocol, values)
    except ValueError as error:
        raise CallError(str(error)) from None
    changed = {n: v for n, v in options.items() if v != default.options[n]}
    return _Variant(path, tuple(sorted(changed.items())))


def list_protocol(
    plugins: PluginPath,
    plugin: str,
    limits: Mapping[str, Any] | None = None,
    rf_definitions: bool = False,
    block: str = "",
) -> str:
    """Reply ``PROTOCOL`` and the plugin's listing block, and its RF definitions where asked.

    The plugin is listed at the options the option entries of the value block
    ``block`` hold, its defaults for those it does not hold; the other lines
    of ``block`` are not read. The listing depends on the plugin file, those
    options and the installed packages only.
    With ``rf_definitions``, the ``[RfDefinitions]`` block of the plugin's
    evaluation at its default protocol under ``limits`` follows it, naming the
    definitions, and the peak amplitudes, the RF layouts of later validations
    refer to. The evaluation places every explicit saturation band the plugin
    offers, so the listing holds the band pulse a console plays without
    listing again. No block follows where the evaluation states no RF layout.
    Where it is invalid, a warning is logged and the listing is replied
    without the block. ``limits`` is not read unless the definitions are asked for.
    Under a warm server the reply is kept, and replied again to the same call
    while the plugin file and the files the limits name are unchanged.

    Raises
    ------
    CallError
        If ``rf_definitions`` is asked without ``limits``.
    """
    if rf_definitions and limits is None:
        raise CallError(
            "the RF definitions are evaluated under scanner limits, and the call "
            "carries none"
        )
    limits = _read(limits) if rf_definitions else None
    key = _call_key("list", plugins, plugin, limits, block, rf_definitions)
    found, kept = _memo.read(_REPLIES, key)
    if found:
        if kept["warning"]:
            _log.warning("%s", kept["warning"])
        return kept["reply"]
    variant = _variant(plugins, plugin, block)
    reply = "PROTOCOL\n" + format_listing(_listing(variant))
    warning = None
    if rf_definitions:
        listed = _listed_rf(variant, limits)
        warning = _unlisted(listed, f"the RF definitions of {plugin} are not listed")
        if listed.block is not None:
            reply += listed.block
    _memo.write(_REPLIES, key, {"reply": reply, "warning": warning})
    return reply


def validate(
    plugins: PluginPath,
    plugin: str,
    limits: Mapping[str, Any],
    block: str,
    rf_layout: bool = False,
    exam: str | None = None,
) -> str:
    """Reply ``VALID <seconds>`` or ``INVALID``, an ``INFO`` line and the value block.

    The plugin evaluates the request under the design limits; the sequence is
    not generated. The duration is ``?`` where the plugin states no scan time.
    With ``rf_layout``, a valid reply ends with the ``[RfLayout]`` block of the
    evaluation where the plugin states one, with amplitudes over the peaks the
    listing states, read by evaluating the plugin again at its default
    protocol. Where that evaluation is invalid, a warning is logged and the
    amplitudes are the layout's own. ``exam`` is the directory of the current
    exam's cache, which the plugin's hooks read where they take it. The
    plugin is loaded at the options of ``block``, as :func:`list_protocol`
    loads it, and the block is read against that listing. Where the
    evaluation states a representative sequence and the plugin is not strict,
    the request is evaluated again at a lower slew rate while that sequence
    exceeds the nerve-stimulation or slew-rate limit, as :func:`generate`
    designs it, and the slew rate it ends at is where :func:`generate` starts.

    Under a warm server the reply is kept, and replied again to the same call
    while the plugin file and the files the limits name are unchanged, the
    request is still remembered at the slew rate the reply was evaluated
    under, which a :func:`generate` that lowers it further changes, and, for
    a plugin whose hooks take ``exam``, the files of the exam's cache are.
    """
    limits = _read(limits)
    key = _call_key("validate", plugins, plugin, limits, block, rf_layout)
    kept = _kept_validation(key, exam)
    if kept is not None:
        return kept
    variant = _variant(plugins, plugin, block)
    listing = _listing(variant)
    request = _request(block, listing)
    validation = _validated(variant, limits, request, exam, lower=True)
    listed: list[float] = []
    if rf_layout and validation.valid and validation.rf_layout is not None:
        default = _listed_rf(variant, limits)
        _unlisted(
            default, f"the RF layout of {plugin} is not stated over the listed peaks"
        )
        listed = default.peaks
    reply = format_validation(
        validation, listing, rf_layout=rf_layout, listed_peak_hz=listed
    )
    _keep_validation(key, reply, variant, limits, request, exam)
    return reply


def generate(
    plugins: PluginPath,
    plugin: str,
    limits: Mapping[str, Any],
    block: str,
    store: DesignStore,
    push: str | None = None,
    exam: str | None = None,
) -> str:
    """Reply ``GENERATED <id>`` for the design a request resolves to.

    A design the scan of ``store`` already holds for the resolved protocol,
    the plugin's options, the plugin source, the package versions and the
    limits is returned without designing again.
    Otherwise the sequence is designed once, under the scanner limits
    capped by the design limits ``design_max_grad`` and ``design_max_slew``,
    or by their ``_any_rotation`` counterparts where the site follows
    motion-correction poses (``pmc_enable``) and the plugin's sequence sets
    ``EnablePmc`` (the manifest's ``design_rotation`` says which), and laid
    out with the design dead times, written in the logical frame,
    checked in the physical frame of the
    prescription rotation against the scanner limits with
    :func:`pulserver.ir.check`, converted to the IR cache at the prescribed
    field-of-view offset, and stored in place of the scan's previous design.
    A request that resolves to other values
    is designed from the resolved values, so a design is a function of its
    identifier. With ``push``, the URL of a design intake, the design is sent
    there unless the intake holds it already. ``exam`` is passed as
    :func:`validate` passes it; for a plugin whose hooks take it, the files of
    the exam's cache are part of the design's identity.

    A design whose only problems a lower slew rate eases -- nerve stimulation,
    slew rate on a physical axis or, for a scan following poses, summed over
    the axes -- is made again at a lower slew rate, at most
    ``_slew.PASSES`` times, starting where :func:`validate` left the request,
    in this process or, under a warm server, in the call that validated it; a
    request no validation searched is searched here first as :func:`validate`
    searches it, on the representative sequence of its evaluation, so the
    scan is designed once wherever that sequence bounds it. Its other
    problems are checked again on each new design. Each file then
    sets ``DesignMaxSlew``, the slew rate per logical axis it was designed
    under in T/m/s, and the manifest records it and the passes. A strict
    plugin (:attr:`~pulserver.design.SequencePlugin.strict_limits`), whose
    files set ``StrictLimits``, or a design that sets ``StrictLimits`` itself,
    is refused as designed.

    Raises
    ------
    CallError
        If the request is invalid or the design fails a check, when nothing
        is stored; or if the design cannot be pushed, when it stays stored.
    """
    variant = _variant(plugins, plugin, block)
    listing = _listing(variant)
    request = _request(block, listing)
    limits = _read(limits)
    scanner = _plugin(variant)
    if not scanner.reads_exam:
        exam = None
    strict = _declares_strict(scanner)
    remembered = _slew.key(variant, limits, request)
    ceiling = None if strict else _searched(variant, limits, request, exam)
    tried: list[tuple[float, dict[str, float]]] = []
    stepped = 0
    while True:
        try:
            reply = _generate_under(
                plugin,
                variant,
                listing,
                request,
                limits,
                scanner,
                store,
                push,
                exam,
                ceiling,
                len(tried) + stepped,
                strict,
            )
        except _Infeasible as error:
            # A layout the plugin cannot make at the lowered slew rate: lower
            # it again, without a reading to model the step on.
            if ceiling is None or stepped == _slew.PASSES:
                raise CallError(str(error)) from None
            _log.info("%s: %s; designing again at a lower slew rate", plugin, error)
            stepped += 1
            ceiling *= _slew.TARGET
            continue
        except _SlewExcess as excess:
            if strict or excess.strict or len(tried) == _slew.PASSES:
                raise CallError(str(excess)) from None
            tried.append((excess.slew, excess.readings))
            ceiling = _slew.lowered(tried)
            _log.info(
                "%s: designed at %.1f T/m/s, %s; designing again at %.1f T/m/s",
                plugin,
                excess.slew / excess.gamma,
                _readings_text(excess.readings),
                ceiling / excess.gamma,
            )
            continue
        if not strict:
            _LOWERED.put(remembered, ceiling)
            if ceiling is not None and (tried or stepped):
                # The scan asked for less than its evaluation did: the
                # plugin's next request is tried there first.
                _LOWERED.put(_slew.key(variant, limits, {}), ceiling)
        return reply


#: The slew rate each request was last lowered to.
_LOWERED = _slew.Remembered()


def _searched(
    variant: _Variant,
    limits: Mapping[str, Any],
    request: Mapping[ProtocolKey, Any],
    exam: str | None,
) -> float | None:
    """Return the slew rate the design of a request starts at: where its validation left it, searched here where none did.

    ``None`` is the design limits. A search that fails leaves the request
    there, and the design then meets what it failed on.
    """
    remembered = _slew.key(variant, limits, request)
    found, ceiling = _LOWERED.entry(remembered)
    if found:
        return ceiling
    try:
        _validated(variant, limits, request, exam, lower=True)
    except Exception as error:
        _log.info("the slew rate is not searched before the design: %s", error)
        return None
    return _LOWERED.get(remembered)


class _Infeasible(CallError):
    """A request the plugin's evaluation refuses."""


class _SlewExcess(CallError):
    """A design whose problems include one a lower slew rate eases."""

    def __init__(
        self,
        problems: str,
        slew: float,
        gamma: float,
        readings: dict[str, float],
        strict: bool,
    ) -> None:
        super().__init__(problems)
        self.slew = slew
        self.gamma = gamma
        self.readings = readings
        self.strict = strict


def _readings_text(readings: Mapping[str, float]) -> str:
    names = {"pns": "PNS", "slew": "slew rate", "vector_slew": "summed slew rate"}
    return ", ".join(
        f"{names[k]} at {100 * v:.0f} % of its limit"
        for k, v in readings.items()
        if k in names
    )


def _declares_strict(scanner: SequencePlugin) -> bool:
    """Whether a plugin asks for its designs to be refused as designed, :attr:`~pulserver.design.SequencePlugin.strict_limits`."""
    return bool(getattr(type(scanner), "strict_limits", False))


def _sets_strict(chain: list[tuple[Path, pp.Sequence]]) -> bool:
    """Whether a file of a chain sets ``StrictLimits``."""
    for _, sequence in chain:
        value = sequence.definitions.get("StrictLimits")
        if value is not None and any(float(v) != 0.0 for v in np.ravel(value)):
            return True
    return False


def _generate_under(
    plugin,
    variant,
    listing,
    request,
    limits,
    scanner,
    store,
    push,
    exam,
    ceiling,
    passes,
    strict,
):
    """Resolve, design, check and store a request with the slew rate capped at ``ceiling``; see :func:`generate`."""
    _, options, _ = split_limits(limits)
    design = design_system(
        limits, any_rotation=_any_rotation(limits, scanner), max_slew=ceiling
    )
    requested = {key: p.value for key, p in listing.items() if p.editable}
    requested.update(request)
    validation = scanner.validate(design, requested, exam)
    if validation.valid and validation.values != requested:
        validation = scanner.validate(design, validation.values, exam)
    if not validation.valid:
        raise _Infeasible(validation.info)
    shortest = requested.get(UIParam.TR) == TRPreset.MINIMUM
    for _ in range(_WAVE_RETRIES):
        try:
            return _generate(
                plugin,
                limits,
                listing,
                scanner,
                variant,
                validation,
                store,
                push,
                exam,
                design,
                passes,
                strict,
            )
        except _SlewExcess:
            raise
        except (CallError, ValueError) as error:
            # The conversion raises a waveform memory it cannot load in time.
            longer = _tr_for_wave_loading(str(error), validation.values, options)
            if longer is None:
                raise
            if not shortest:
                raise CallError(
                    f"{error}; the waves stream in at a TR of about "
                    f"{1e-3 * longer:.1f} ms or more"
                ) from None
            retried = scanner.validate(
                design, {**validation.values, UIParam.TR: longer}, exam
            )
            if not retried.valid:
                raise
            _log.info(
                "%s: the shortest TR leaves no time to stream the waves in; "
                "designing at TR %.3f ms",
                plugin,
                1e-3 * longer,
            )
            validation = retried
    raise CallError(f"the waves of {plugin} cannot be streamed in at any TR tried")


#: Designs generate tries, lengthening the shortest TR, before it gives up on
#: streaming the waves in.
_WAVE_RETRIES = 6

_WAVE_LATE = re.compile(r"loading its waves ends ([0-9.]+) us after it starts")


def _tr_for_wave_loading(
    problem: str, values: Mapping[Any, Any], options: Mapping[str, Any]
) -> int | None:
    """Return the TR (us) at which the waves a problem says load late would load in time, or None.

    The loading may take ``headroom`` of the playout's time, so each microsecond
    late takes ``1 / headroom`` more of TR, and a tenth more as margin.
    """
    found = _WAVE_LATE.search(problem)
    budget = options.get("wave_budget")
    tr = values.get(UIParam.TR)
    if found is None or budget is None or tr is None or float(tr) <= 0:
        return None
    late_us = float(found.group(1))
    return math.ceil(float(tr) + 1.1 * late_us / float(budget.headroom))


def _generate(
    plugin,
    limits,
    listing,
    scanner,
    variant,
    validation,
    store,
    push,
    exam,
    design,
    passes=0,
    strict=False,
):
    """Design under ``design``, check, convert and store ``validation``'s values; see :func:`generate`.

    Raises :class:`_SlewExcess` for a design whose problems include one a
    lower slew rate eases.
    """
    system, options, checked = split_limits(limits)
    any_rotation = _any_rotation(limits, scanner)
    slew = round(design.max_slew / design.gamma, 4)
    definitions = {"DesignMaxSlew": slew}
    if strict:
        definitions["StrictLimits"] = 1
    source = _source(variant)
    identified = dict(validation.values)
    identified["design_max_slew_t_per_m_per_s"] = slew
    if scanner.options:
        identified["options"] = dict(scanner.options)
    if exam is not None:
        identified["exam"] = _exam_digest(exam)
    identity = design_identity(plugin, identified_limits(limits), identified, source)
    found = store.find(identity)
    if found is not None:
        return f"GENERATED {_pushed(store, found, push)}\n"
    staged = store.stage()
    try:
        accepted, paths, written = scanner._design(
            design, validation.values, staged, exam, definitions
        )
        if not accepted.valid:
            raise CallError(accepted.info)
        # The sequences as written stand for the files. The conversion moves
        # them to the offset in place, which changes only RF and ADC offsets
        # and phases; the check then rotates them in place and reads only
        # gradients and timing.
        chain = designed_chain(written)
        offset = prescribed_offset(validation.values)
        rotation = prescribed_rotation(validation.values)
        readings: dict[str, float] = {}
        played: dict[str, Any] = {}
        problems = _check_and_convert(
            paths[0],
            system,
            checked,
            offset,
            options,
            rotation,
            chain,
            readings,
            played,
        )
        if problems:
            if _slew.exceeds(readings):
                raise _SlewExcess(
                    "; ".join(problems),
                    design.max_slew,
                    design.gamma,
                    readings,
                    strict or _sets_strict(chain),
                )
            raise CallError("; ".join(problems))
        (staged / "resolved.protocol").write_text(
            format_values(validation.values, listing)
        )
        manifest = {
            **_record(limits),
            "plugin": plugin,
            "options": dict(scanner.options),
            "source": source,
            "dependencies": _dependencies(variant),
            "scan_time": validation.duration,
            "fov_offset_mm": [1e3 * value for value in offset],
            "design_rotation": "any" if any_rotation else "prescribed",
            "design_max_slew_t_per_m_per_s": slew,
            "slew_passes": passes,
            "strict_limits": strict or _sets_strict(chain),
            **played,
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
    prescription and averages of the import block. A chain the scan of
    ``store`` already holds is returned; otherwise it is copied, its main sequence, the last file, is
    played as many times as the block's ``nex`` line asks, written into its
    block table as binary Pulseq, and the chain is checked and converted as
    :func:`generate` does, stored and pushed as :func:`generate` stores and
    pushes it. The first
    file is also reachable as ``sequence.seq``.

    Raises
    ------
    CallError
        If the block is malformed, a file cannot be read, or the chain fails
        a check, when nothing is stored; or if the design cannot be pushed.
    """
    try:
        first, offset_mm, rotation = parse_import(block)
        averages = import_averages(block)
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
    if averages > 1:
        values["nex"] = averages
    identity = design_identity("", identified_limits(limits), values)
    found = store.find(identity)
    if found is not None:
        return f"IMPORTED {_pushed(store, found, push)}\n"
    staged = store.stage()
    try:
        for file in files:
            shutil.copyfile(file, staged / file.name)
        if averages > 1:
            _play_averages(staged / files[-1].name, averages)
        entry = staged / _ENTRY
        if files[0].name != _ENTRY:
            entry.symlink_to(files[0].name)
        system, options, checked = split_limits(limits)
        offset = tuple(value * 1e-3 for value in offset_mm)
        problems = _check_and_convert(
            str(entry), system, checked, offset, options, rotation
        )
        if problems:
            raise CallError("; ".join(problems))
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


#: The limits that name a file.
_LIMIT_FILES = ("vop_file", "ir_vendor_file", "acoustic_file")


def identified_limits(limits: Mapping[str, Any]) -> dict[str, Any]:
    """Return limits with the digest of each file they name: the VOP, vendor and acoustic files.

    A design converted under one file is not the design of another file
    written to the same path.

    Raises
    ------
    CallError
        If a named file cannot be read.
    """
    identified = dict(limits)
    for key in _LIMIT_FILES:
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
def _cached(variant: _Variant, mtimes_ns: tuple[int, ...]) -> SequencePlugin:  # noqa: ARG001 -- part of the key
    return load_plugin(Path(variant.path), dict(variant.options))


def _mtimes(variant: _Variant) -> tuple[int, ...]:
    """Return the modification time of the plugin file of a variant, which changes when it is edited."""
    return (Path(variant.path).stat().st_mtime_ns,)


def _plugin(variant: _Variant) -> SequencePlugin:
    """Return the plugin of a variant, imported again when its file changed.

    Raises
    ------
    CallError
        If the plugin does not take the variant's options.
    """
    try:
        return _cached(variant, _mtimes(variant))
    except OptionError as error:
        raise CallError(f"{variant.path}: {error}") from None


def _listing(variant: _Variant) -> dict[ProtocolKey, Parameter]:
    return _plugin(variant).listing()


def _dependencies(variant):
    from .._site_packages import manifest

    return manifest(variant.path)


def _source(variant: _Variant) -> str:
    """Return a digest of the code that designs with the plugin of a variant.

    Covers the plugin file, the source file of the app it binds, and the
    installed versions of pypulseqpp and pulserver, declared requirements and
    site-library installation metadata. A :func:`functools.partial` app is
    covered through the function it wraps. Other imports need a dependency
    declaration to identify their installed distribution.
    """
    digest = hashlib.sha256(Path(variant.path).read_bytes())
    app = _plugin(variant).app
    while isinstance(app, functools.partial):
        app = app.func
    try:
        module = inspect.getsourcefile(app)
    except TypeError:
        module = None
    if module:
        digest.update(Path(module).read_bytes())
    dependencies = _dependencies(variant)
    if dependencies:
        digest.update(json.dumps(dependencies, sort_keys=True).encode())
    digest.update(f"pypulseqpp {pp.__version__} pulserver {__version__}".encode())
    return digest.hexdigest()


def _validated(
    variant: _Variant,
    limits: Mapping[str, Any],
    request: Mapping[ProtocolKey, Any],
    exam: str | None = None,
    lower: bool = False,
) -> Validation:
    """Return the validation of a request under the design limits.

    With ``lower``, a plugin that is not strict and whose evaluation states a
    representative sequence is evaluated again at a lower slew rate while
    that sequence, rotated to the prescription, exceeds the nerve-stimulation
    or slew-rate limit, starting from the slew rate the request was last
    lowered to; the slew rate a valid request ends at is remembered for
    :func:`generate`, and for the plugin's next request under these limits,
    which is evaluated there first. A request searched before is evaluated at
    the slew rate it ended at. A request still exceeding them after
    ``_slew.PASSES`` passes is invalid.
    """
    scanner = _plugin(variant)
    any_rotation = _any_rotation(limits, scanner)
    if not lower or _declares_strict(scanner):
        design = design_system(limits, any_rotation=any_rotation)
        return scanner.validate(design, request, exam)
    remembered = _slew.key(variant, limits, request)
    family = _slew.key(variant, limits, {})
    system, _, checked = split_limits(limits)
    found, ceiling = _LOWERED.entry(remembered)
    if not found:
        guessed = _guessed(
            scanner,
            limits,
            request,
            exam,
            family,
            system,
            checked,
            any_rotation,
        )
        if guessed is not None:
            _LOWERED.put(remembered, guessed[0])
            return guessed[1]
    tried: list[tuple[float, dict[str, float]]] = []
    stepped = 0
    while True:
        design = design_system(limits, any_rotation=any_rotation, max_slew=ceiling)
        validation, _, sequence = scanner._evaluated(design, request, exam)
        if not validation.valid and ceiling is not None and stepped < _slew.PASSES:
            # A layout the plugin cannot make at the lowered slew rate.
            stepped += 1
            ceiling *= _slew.TARGET
            continue
        if (
            not validation.valid
            or sequence is None
            or _sets_strict([(Path(), sequence)])
        ):
            break
        readings = _representative_readings(
            sequence,
            system,
            checked,
            prescribed_rotation(validation.values),
            any_rotation,
        )
        if not _slew.exceeds(readings):
            break
        if len(tried) == _slew.PASSES:
            return Validation(
                False,
                None,
                f"no slew rate tried keeps the design within its limits: "
                f"{_readings_text(readings)} at {design.max_slew / design.gamma:.1f} T/m/s",
                validation.values,
            )
        tried.append((design.max_slew, readings))
        ceiling = _slew.lowered(tried)
        _log.debug(
            "evaluated at %.1f T/m/s, %s; evaluating again at %.1f T/m/s",
            design.max_slew / design.gamma,
            _readings_text(readings),
            ceiling / design.gamma,
        )
    if validation.valid:
        _LOWERED.put(remembered, ceiling)
        _LOWERED.put(family, ceiling)
    return validation


#: A slew rate a plugin's last request was lowered to is kept for the next
#: request only when its readings reach this fraction of their limits: below
#: it the gradients would be slower than the limits ask.
_NEAR = 0.85


def _guessed(scanner, limits, request, exam, family, system, checked, any_rotation):
    """Return the slew rate and validation of a request evaluated at the slew rate the plugin's last request under these limits was lowered to.

    ``None`` where none was, or where at that slew rate the request is
    invalid, exceeds a limit, or keeps every reading below :data:`_NEAR` of
    its limit: the request is then searched for from the design limits.
    """
    ceiling = _LOWERED.get(family)
    if ceiling is None:
        return None
    design = design_system(limits, any_rotation=any_rotation, max_slew=ceiling)
    validation, _, sequence = scanner._evaluated(design, request, exam)
    if not validation.valid or sequence is None:
        return None
    readings = _representative_readings(
        sequence, system, checked, prescribed_rotation(validation.values), any_rotation
    )
    eased = [readings.get(key, 0.0) for key in _slew.READINGS]
    if _slew.exceeds(readings) or max(eased) < _NEAR:
        return None
    return ceiling, validation


def _representative_readings(
    sequence: pp.Sequence,
    system: pp.Opts,
    checked: ir.CheckLimits,
    rotation: Any,
    any_rotation: bool,
) -> dict[str, float]:
    """Return the gradient readings of a representative sequence rotated to the prescription, as :func:`pulserver.ir.check` fills its ``readings``."""
    opts = copy.copy(system)
    for name in _RASTERS:
        setattr(opts, name, getattr(sequence, name))
    turned = sequence
    if rotation is not None:
        turned = pp.TransformFOV(
            rotation=np.asarray(rotation, dtype=float)
        ).apply_to_sequence(sequence, in_place=False)
    readings: dict[str, float] = {}
    turns = any_rotation and enables_pmc(sequence)
    for check, name in (
        (safety.check_max_grad, "grad"),
        (safety.check_max_slew, "slew"),
    ):
        _, found = check(turned, opts)
        if found.limit > 0:
            readings[name] = found.per_axis.value / found.limit
            if turns:
                readings[f"vector_{name}"] = found.vector.value / found.limit
    if checked.pns is not None:
        _, nerve = safety.check_pns(turned, checked.pns, system=opts)
        readings["pns"] = nerve.peak.value / checked.pns_limit
    return readings


def _any_rotation(limits: Mapping[str, Any], scanner: SequencePlugin) -> bool:
    """Whether ``scanner`` is designed for every rotation: the site follows poses and its sequence asks for them.

    A sequence function states that it sets ``EnablePmc`` by an
    ``enable_pmc`` attribute, as :func:`pulserver._zoo._pmc.navigated` sets
    it, on itself or on any function it wraps.
    """
    if not follows_poses(limits):
        return False
    app: Any = type(scanner).app
    seen: set[int] = set()
    while app is not None and id(app) not in seen:
        seen.add(id(app))
        if getattr(app, "enable_pmc", False):
            return True
        app = getattr(app, "__wrapped__", None) or getattr(app, "func", None)
    return False


def _exam_digest(exam: str) -> str:
    """Return a digest of the name, size and modification time of each file of an exam's cache."""
    digest = hashlib.sha256()
    for entry in sorted(Path(exam).glob("*")):
        if entry.is_file():
            stat = entry.stat()
            digest.update(f"{entry.name} {stat.st_size} {stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


class _ListedRf(NamedTuple):
    """The RF the listing of a plugin states: the ``[RfDefinitions]`` block, ``None`` where its default protocol states no RF layout, the peak of each definition in Hz, and why its default protocol is invalid, ``None`` where it is valid."""

    block: str | None
    peaks: list[float]
    problem: str | None


def _listed_rf(variant: _Variant, limits: Mapping[str, Any]) -> _ListedRf:
    """Return the RF of the plugin's evaluation at its default protocol, the RF the listing states.

    Every explicit saturation band the plugin offers is placed: which bands
    play is the console's choice, made without listing again, so the pulse
    they play must be listed whichever it places. The evaluation is made
    once per variant, as last modified, and limits: in this process and,
    under a warm server, by the first call that asks.
    """
    key = json.dumps(
        [variant.path, repr(variant.options), _mtimes(variant), _limits_key(limits)]
    )
    found, kept = _memo.read(_LISTED_RF, key)
    if found:
        return _ListedRf(*kept)
    evaluated = _default_validation(
        variant, _mtimes(variant), tuple(sorted(limits.items()))
    )
    layout = evaluated.rf_layout
    listed = _ListedRf(
        None if layout is None else format_rf_definitions(layout.instances),
        []
        if layout is None
        else [float(definition.peak_hz) for definition in layout.instances.definitions],
        None if evaluated.valid else evaluated.info,
    )
    _memo.write(_LISTED_RF, key, list(listed))
    return listed


def _unlisted(listed: _ListedRf, unavailable: str) -> str | None:
    """Log and return a warning that begins with ``unavailable`` where the default protocol of a plugin is invalid; ``None`` where it is valid."""
    if listed.problem is None:
        return None
    warning = f"{unavailable}: its default protocol is invalid: {listed.problem}"
    _log.warning("%s", warning)
    return warning


# The kinds of entries a warm server keeps (:mod:`pulserver.host._memo`).
_REPLIES = "reply"
_LISTED_RF = "listed-rf"


def _limits_key(limits: Mapping[str, Any]) -> list[Any]:
    """Return limits as a kept entry is keyed on them: their values, and when each file they name was last modified."""
    named = []
    for key in _LIMIT_FILES:
        if key in limits:
            try:
                stat = Path(str(limits[key])).stat()
            except OSError:
                named.append([key, None, None])
            else:
                named.append([key, stat.st_mtime_ns, stat.st_size])
    return [sorted((str(k), str(v)) for k, v in limits.items()), named]


def _call_key(
    name: str,
    plugins: PluginPath,
    plugin: str,
    limits: Mapping[str, Any] | None,
    block: str,
    flag: bool,
) -> str:
    """Return the key the reply of a ``list`` or ``validate`` call is kept under: the plugin file as last modified, the limits, the block and whether the RF is asked for.

    The block holds the options the plugin is loaded with, so the key covers
    them; the packages are those the server imported before it kept anything.
    """
    if not _memo.kept():
        return ""
    path = plugin_path(plugins, plugin)
    return json.dumps(
        [
            name,
            str(path),
            path.stat().st_mtime_ns,
            None if limits is None else _limits_key(limits),
            block,
            bool(flag),
        ]
    )


def _kept_validation(key: str, exam: str | None) -> str | None:
    """Return the reply kept for a ``validate`` call, ``None`` where none is or it no longer holds.

    It holds while the request is remembered at the slew rate the reply was
    evaluated under and, for a plugin that reads the exam's cache, the files
    of the cache are unchanged.
    """
    found, kept = _memo.read(_REPLIES, key)
    if not found:
        return None
    if list(_LOWERED.entry(kept["request"])) != kept["slew"]:
        return None
    if kept["exam"] is not False and kept["exam"] != _exam_state(exam):
        return None
    return kept["reply"]


def _keep_validation(
    key: str,
    reply: str,
    variant: _Variant,
    limits: Mapping[str, Any],
    request: Mapping[ProtocolKey, Any],
    exam: str | None,
) -> None:
    """Keep the reply of a ``validate`` call with what it holds under; see :func:`_kept_validation`."""
    if not _memo.kept():
        return
    remembered = _slew.key(variant, limits, request)
    kept = {
        "reply": reply,
        "request": remembered,
        "slew": list(_LOWERED.entry(remembered)),
        "exam": _exam_state(exam) if _plugin(variant).reads_exam else False,
    }
    _memo.write(_REPLIES, key, kept)


def _exam_state(exam: str | None) -> str | None:
    return None if exam is None else _exam_digest(exam)


@lru_cache(maxsize=32)
def _default_validation(
    variant: _Variant,
    mtimes_ns: tuple[int, ...],  # noqa: ARG001 -- part of the key
    limits: tuple,
) -> Validation:
    mask = _listing(variant).get(UIParam.EXSAT_MASK)
    placed = {} if mask is None else {UIParam.EXSAT_MASK: int(mask.range_max)}
    return _validated(variant, dict(limits), placed)


def _play_averages(path: Path, averages: int) -> None:
    """Rewrite the sequence file at ``path`` as binary Pulseq, played ``averages`` times in its block table."""
    seq = pp.io.read(str(path))
    seq.expand_repeats(averages)
    pp.io.write(seq, str(path), binary=True, remove_duplicates=False)


def _chain(first: str) -> list[str]:
    return [str(path) for path in ir.chain(first)]


def _check_and_convert(
    seq_path: str,
    system: pp.Opts,
    checked: ir.CheckLimits,
    fov_offset: tuple[float, float, float],
    options: dict[str, Any],
    rotation: Any,
    designed: list[tuple[Path, pp.Sequence]] | None = None,
    readings: dict[str, float] | None = None,
    record: dict[str, Any] | None = None,
) -> list[str]:
    """Check the chain in the frame of ``rotation`` while its cache is written; return the problems.

    ``readings`` is filled as :func:`pulserver.ir.check` fills it.

    The chain is ``designed`` when given, else read once from ``seq_path`` and
    verified. A receive calibration generated from it
    (:func:`~pulserver.ir.receive_calibration_sequence`) joins it as its last
    subsequence, so it is checked, costed and written like any other. Its
    sound pressure levels are taken first, held to the limits by the check
    and written into the cache. The cache is segmented from copies of the
    chain moved to ``fov_offset`` while the check rotates the chain in place.
    Where the limits state the sequencer's segment
    timing, the time the chain plays (:func:`pulserver.ir.played_time`) is
    put into ``record`` as ``played_time_s``, beside ``blocks_time_s``.
    """
    if designed is None:
        designed = _read_chain(Path(seq_path), True)
    calibration = ir.receive_calibration_sequence([s for _, s in designed], system)
    if calibration is not None:
        designed = [
            *designed,
            (Path(seq_path).with_name("prescan_calibration.seq"), calibration),
        ]
    levels = (
        None
        if checked.acoustic is None
        else ir.spl_levels(seq_path, system, checked, designed, rotation)
    )
    writer = _converting(
        seq_path, system, checked, fov_offset, options, designed, levels
    )
    timing = checked.sequencer_timing
    with ThreadPoolExecutor(2) as pool:
        cache = pool.submit(writer)
        # The libraries are copied here, before the check rotates the chain.
        played = (
            None
            if timing is None or record is None
            else pool.submit(
                _played_time,
                _payload(Path(seq_path), system, False, None, designed),
                system,
                timing,
                **options,
            )
        )
        problems = ir.check(
            seq_path,
            system,
            rotation=rotation,
            limits=checked,
            designed=designed,
            spl_levels=levels,
            readings=readings,
        )
        cache.result()
        if played is not None:
            found = played.result()
            record["played_time_s"] = found.played
            record["blocks_time_s"] = found.blocks
    return problems


def _converting(
    seq_path: str,
    system: pp.Opts,
    checked: ir.CheckLimits,
    fov_offset: tuple[float, float, float],
    options: dict[str, Any],
    designed: list[tuple[Path, pp.Sequence]],
    spl_levels: list[ir.SplLevels] | None,
) -> Callable[[], Path]:
    """Move the chain to ``fov_offset`` and return the call that writes its cache.

    The cache carries the chain's SAR ratios, taken here before the move, and
    ``spl_levels``.
    """
    ratios = (
        None
        if checked.vops is None
        else ir.sar_ratios(seq_path, system, checked, designed)
    )
    payload = _payload(Path(seq_path), system, True, fov_offset, designed)
    return partial(
        _write_cache,
        seq_path,
        system,
        payload,
        sar_ratios=ratios,
        spl_levels=spl_levels,
        **options,
    )
