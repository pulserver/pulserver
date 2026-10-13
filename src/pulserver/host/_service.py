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
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, NamedTuple

import pypulseqpp as pp

from .. import __version__, _plugins, ir
from .._plugins import PluginPath
from ..design import (
    OptionError,
    RfLayout,
    SequencePlugin,
    load_plugin,
    read_options,
)
from ..ir._convert import _payload, _read_chain, _write_cache
from ..mrd._sequence import designed_chain
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
from ._blocks import import_averages, parse_import
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


def parameter_file(
    path: Path, parameter_set: int, parameters: Path | str | None = None
) -> Path | None:
    """Return the parameter file of set ``parameter_set`` for the plugin file ``path``; ``None`` for set 0.

    The file is ``<parameters>/<script>/<NN>.toml``, the set number in two
    digits at least. ``<script>`` is the stem of the plugin file once its
    symbolic links are followed, so a slot ``sequence3.py`` linked to
    ``src/fse3d.py`` reads ``fse3d/``. ``parameters`` defaults to the
    ``parameters`` directory beside the plugin directory.

    Raises
    ------
    CallError
        If the set is negative, or positive and its file does not exist.
    """
    if parameter_set < 0:
        raise CallError(f"parameter set {parameter_set} is negative")
    if parameter_set == 0:
        return None
    directory = Path(parameters) if parameters else path.parent.parent / "parameters"
    script = path.resolve().stem
    file = directory / script / f"{parameter_set:02d}.toml"
    if not file.is_file():
        raise CallError(f"parameter set {parameter_set} of {script}: no file {file}")
    return file


class _Variant(NamedTuple):
    """A plugin file and the parameter file it is loaded with, if any."""

    path: str
    parameters: str | None


def _variant(
    plugins: PluginPath,
    plugin: str,
    parameter_set: int,
    parameters: Path | str | None,
) -> _Variant:
    path = plugin_path(plugins, plugin)
    file = parameter_file(path, parameter_set, parameters)
    return _Variant(str(path), None if file is None else str(file))


def list_protocol(
    plugins: PluginPath,
    plugin: str,
    limits: Mapping[str, Any] | None = None,
    rf_definitions: bool = False,
    parameter_set: int = 0,
    parameters: Path | str | None = None,
) -> str:
    """Reply ``PROTOCOL`` and the plugin's listing block, and its RF definitions where asked.

    The listing depends on the plugin file, the file of ``parameter_set``
    (see :func:`parameter_file`) and the installed packages only.
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
    variant = _variant(plugins, plugin, parameter_set, parameters)
    reply = "PROTOCOL\n" + format_listing(_listing(variant))
    if not rf_definitions:
        return reply
    if limits is None:
        raise CallError(
            "the RF definitions are evaluated under scanner limits, and the call "
            "carries none"
        )
    layout = _default_rf_layout(
        variant, _read(limits), f"the RF definitions of {plugin} are not listed"
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
    exam: str | None = None,
    parameter_set: int = 0,
    parameters: Path | str | None = None,
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
    plugin is loaded with the options of ``parameter_set``, as
    :func:`list_protocol` loads it.
    """
    variant = _variant(plugins, plugin, parameter_set, parameters)
    listing = _listing(variant)
    request = _request(block, listing)
    limits = _read(limits)
    validation = _validated(variant, limits, request, exam)
    listed: list[float] = []
    if rf_layout and validation.valid and validation.rf_layout is not None:
        default = _default_rf_layout(
            variant,
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
    exam: str | None = None,
    parameter_set: int = 0,
    parameters: Path | str | None = None,
) -> str:
    """Reply ``GENERATED <id>`` for the design a request resolves to.

    A design the scan of ``store`` already holds for the resolved protocol,
    the plugin's options, the plugin source, the package versions and the
    limits is returned without designing again.
    Otherwise the sequence is designed once, under the scanner limits
    capped by the design limits ``design_max_grad`` and ``design_max_slew``,
    written in the logical frame, checked in the physical frame of the
    prescription rotation against the scanner limits with
    :func:`pulserver.ir.check`, converted to the IR cache at the prescribed
    field-of-view offset, and stored in place of the scan's previous design.
    A request that resolves to other values
    is designed from the resolved values, so a design is a function of its
    identifier. With ``push``, the URL of a design intake, the design is sent
    there unless the intake holds it already. ``exam`` is passed as
    :func:`validate` passes it; for a plugin whose hooks take it, the files of
    the exam's cache are part of the design's identity.

    Raises
    ------
    CallError
        If the request is invalid or the design fails a check, when nothing
        is stored; or if the design cannot be pushed, when it stays stored.
    """
    variant = _variant(plugins, plugin, parameter_set, parameters)
    listing = _listing(variant)
    request = _request(block, listing)
    limits = _read(limits)
    system, options, checked = split_limits(limits)
    design = design_system(limits)
    scanner = _plugin(variant)
    requested = {key: p.value for key, p in listing.items() if p.editable}
    requested.update(request)
    if not scanner.reads_exam:
        exam = None
    validation = scanner.validate(design, requested, exam)
    if validation.valid and validation.values != requested:
        validation = scanner.validate(design, validation.values, exam)
    if not validation.valid:
        raise CallError(validation.info)
    source = _source(variant)
    identified = dict(validation.values)
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
            design, validation.values, staged, exam
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
        problems = _check_and_convert(
            paths[0], system, checked, offset, options, rotation, chain
        )
        if problems:
            raise CallError("; ".join(problems))
        (staged / "resolved.protocol").write_text(
            format_values(validation.values, listing)
        )
        manifest = {
            **_record(limits),
            "plugin": plugin,
            "options": dict(scanner.options),
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
    for key in ("vop_file", "ir_vendor_file", "acoustic_file"):
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
    options = {} if variant.parameters is None else read_options(variant.parameters)
    return load_plugin(Path(variant.path), options)


def _mtimes(variant: _Variant) -> tuple[int, ...]:
    """Return the modification times of the files of a variant, which change when either is edited."""
    return tuple(Path(f).stat().st_mtime_ns for f in variant if f is not None)


def _plugin(variant: _Variant) -> SequencePlugin:
    """Return the plugin of a variant, imported again when either of its files changed.

    Raises
    ------
    CallError
        If the parameter file is not TOML, or sets options the plugin does
        not declare or values it does not take.
    """
    try:
        return _cached(variant, _mtimes(variant))
    except OptionError as error:
        raise CallError(f"{variant.parameters}: {error}") from None


def _listing(variant: _Variant) -> dict[ProtocolKey, Parameter]:
    return _plugin(variant).listing()


def _source(variant: _Variant) -> str:
    """Return a digest of the code that designs with the plugin of a variant.

    Covers the plugin file, the source file of the app it binds, and the
    installed versions of pypulseqpp and pulserver. A :func:`functools.partial`
    app is covered through the function it wraps. Modules the app imports from
    elsewhere are covered only through those versions.
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
    digest.update(f"pypulseqpp {pp.__version__} pulserver {__version__}".encode())
    return digest.hexdigest()


def _validated(
    variant: _Variant,
    limits: Mapping[str, Any],
    request: Mapping[ProtocolKey, Any],
    exam: str | None = None,
) -> Validation:
    return _plugin(variant).validate(design_system(limits), request, exam)


def _exam_digest(exam: str) -> str:
    """Return a digest of the name, size and modification time of each file of an exam's cache."""
    digest = hashlib.sha256()
    for entry in sorted(Path(exam).glob("*")):
        if entry.is_file():
            stat = entry.stat()
            digest.update(f"{entry.name} {stat.st_size} {stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def _default_rf_layout(
    variant: _Variant, limits: Mapping[str, Any], unavailable: str
) -> RfLayout | None:
    """Return the RF layout of the plugin's evaluation at its default protocol, the RF the listing states.

    ``None`` where the evaluation states none. Where it is invalid, a warning
    that begins with ``unavailable`` is logged and the result is ``None``. The
    evaluation is made once per variant, as last modified, and limits.
    """
    evaluated = _default_validation(
        variant, _mtimes(variant), tuple(sorted(limits.items()))
    )
    if not evaluated.valid:
        _log.warning(
            "%s: its default protocol is invalid: %s", unavailable, evaluated.info
        )
    return evaluated.rf_layout


@lru_cache(maxsize=32)
def _default_validation(
    variant: _Variant,
    mtimes_ns: tuple[int, ...],  # noqa: ARG001 -- part of the key
    limits: tuple,
) -> Validation:
    return _validated(variant, dict(limits), {})


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
) -> list[str]:
    """Check the chain in the frame of ``rotation`` while its cache is written; return the problems.

    The chain is ``designed`` when given, else read once from ``seq_path`` and
    verified. Its sound pressure levels are taken first, held to the limits
    by the check and written into the cache. The cache is segmented from
    copies of the chain moved to ``fov_offset`` while the check rotates the
    chain in place.
    """
    if designed is None:
        designed = _read_chain(Path(seq_path), True)
    levels = (
        None
        if checked.acoustic is None
        else ir.spl_levels(seq_path, system, checked, designed, rotation)
    )
    writer = _converting(
        seq_path, system, checked, fov_offset, options, designed, levels
    )
    with ThreadPoolExecutor(1) as pool:
        cache = pool.submit(writer)
        problems = ir.check(
            seq_path,
            system,
            rotation=rotation,
            limits=checked,
            designed=designed,
            spl_levels=levels,
        )
        cache.result()
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
