"""Scanner-sequence plugins: a sequence app bound to the scanner protocol."""

from __future__ import annotations

import difflib
import importlib.util
import inspect
import logging
import sys
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pypulseqpp as pp
from pypulseqpp import sequences

from .. import ir
from ..protocol import (
    FOV_OFFSET,
    FOV_ROTATION,
    PRESCRIPTION,
    InputMode,
    Kind,
    Parameter,
    ProtocolKey,
    RfPulse,
    Validation,
    prescribed_rotation,
)
from ..protocol._keys import WIRE_NAMES
from ._entries import Entry, Protocol, _parameter

_log = logging.getLogger("pulserver.design")

#: Range of each prescription entry, in mm either side of the isocentre.
OFFSET_LIMIT_MM = 1000.0

# What pypulseqpp and PyPulseq raise for an event or a timing they cannot
# realize: ValueError, and AssertionError from PyPulseq's feasibility checks,
# such as an area that does not fit its duration.
_INFEASIBLE = (ValueError, AssertionError)

_FIRST_FILE = "sequence.seq"


@dataclass(frozen=True)
class Evaluation:
    """The outcome of :meth:`SequencePlugin.evaluate` for a valid protocol.

    Attributes
    ----------
    protocol
        The protocol the console shows back, holding the values the design
        achieves, such as the shortest echo time.
    duration
        Scan time in seconds. ``0.0`` states no estimate and is not invalid;
        the reply then reports an unknown scan time.
    info
        A note shown with the valid protocol. Whitespace, newlines included, is
        folded to single spaces on the wire.
    """

    protocol: Protocol
    duration: float = 0.0
    info: str = ""


class SequencePlugin:
    """A sequence app bound to the entries of the scanner protocol.

    A subclass sets :attr:`app` and, for the arguments the operator edits,
    :attr:`protocol`; without ``protocol`` the app plays its defaults. Entries
    a request omits keep their initial values, the entry's ``default`` or else
    the app's. Times travel as integer microseconds and other floats to six
    significant digits, the precision of a float32 parameter, so a reply stored
    in scanner parameters and sent back evaluates to itself.

    :meth:`validate` evaluates a request with :meth:`evaluate`, and
    :meth:`design` generates and writes it with :meth:`generate`. A subclass
    overrides the hooks :meth:`evaluate` and :meth:`generate`, and does not
    override :meth:`validate` or :meth:`design`.

    Attributes
    ----------
    app : callable
        Designs the sequence. Either a function ``app(system, **arguments)``,
        or a :class:`~pypulseqpp.sequences.SequenceApp` subclass.

        A function takes the scanner limits, a :class:`pypulseqpp.Opts`, and
        the keyword arguments the entries of :attr:`protocol` bind, and
        returns a :class:`pypulseqpp.Sequence` or a list of them, which is a
        chain: the prescans first and the main sequence last. The defaults
        the listing shows are those of its signature, so a
        :func:`functools.partial` is an app. It is called by :meth:`generate`
        only.

        A sequence application is constructed under the scanner limits, which
        checks the prescription, and designed by :meth:`design`.
    protocol : mapping
        The entries the operator edits, by key: the interpreter's parameter
        names, which are the members of
        :data:`~pulserver.protocol.ProtocolKey` that
        :class:`~pulserver.protocol.UIParam` collects. A plain string naming an
        entry is stored as its member. Empty by default.
    follows : tuple of str
        Which protocol parameter drives the angle of each pulse the sequence
        plays: one entry per distinct pulse, in the order the sequence first
        plays them, ``""`` for one the operator does not move. A sequence that
        excites and then refocuses under the one control declares
        ``(UIParam.FLIP, UIParam.FLIP)``.

        It is declared rather than worked out: which control drives a pulse is
        not a property of the pulse. Empty leaves a scanner to cost the RF when
        it has the design, which is what it does with any sequence.

        The angles are not declared. They are read from the sequence a design
        wrote, and a pulse's share of its parameter is the angle it was
        designed at over the value that parameter was designed with -- so a
        refocusing train whose angles vary gets one share each without any of
        them being written down.

    Raises
    ------
    ValueError
        When a subclass is defined with a ``protocol`` key the interpreter does
        not know, which its parser would drop, or with one of the prescription
        entries, which pulserver applies itself; or with both ``protocol`` and
        ``ui``.

    Warns
    -----
    DeprecationWarning
        When a subclass sets ``recon``, which has no effect: the reconstruction
        is named by the reconstruction client or, on a console, by the scan.
        When a subclass sets ``ui``, the deprecated name of ``protocol``, or
        subclasses :class:`ScannerSequence`, the deprecated name of this class.
    """

    app: ClassVar[Callable[..., Any]]
    protocol: ClassVar[Mapping[ProtocolKey, Entry]] = {}
    follows: ClassVar[tuple[ProtocolKey | str, ...]] = ()

    _deprecated_alias: ClassVar[bool] = False
    _built: tuple[pp.Opts, dict[str, Any], sequences.SequenceApp] | None = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        declared = cls.__dict__
        if any(base.__dict__.get("_deprecated_alias") for base in cls.__bases__):
            warnings.warn(
                f"{cls.__name__} subclasses ScannerSequence, which is deprecated; "
                "subclass SequencePlugin",
                DeprecationWarning,
                stacklevel=2,
            )
        if "recon" in declared:
            warnings.warn(
                f"{cls.__name__} sets recon, which has no effect: the reconstruction "
                "is named by the reconstruction client or by the scan",
                DeprecationWarning,
                stacklevel=2,
            )
        if inspect.isfunction(declared.get("app")):
            cls.app = staticmethod(declared["app"])
        if "ui" in declared:
            if "protocol" in declared:
                raise ValueError(
                    f"{cls.__name__} declares both protocol and ui, the deprecated "
                    "name of protocol"
                )
            warnings.warn(
                f"{cls.__name__} sets ui, which is deprecated; declare protocol",
                DeprecationWarning,
                stacklevel=2,
            )
            entries = declared["ui"]
        elif "protocol" in declared:
            entries = declared["protocol"]
        else:
            return
        unknown = sorted(name for name in entries if name not in WIRE_NAMES)
        if unknown:
            names = {name.lower(): name for name in WIRE_NAMES}
            close = sorted(
                {
                    names[m]
                    for n in unknown
                    for m in difflib.get_close_matches(n.lower(), names)
                }
            )
            hint = f"; did you mean {close}?" if close else ""
            raise ValueError(
                f"{cls.__name__} declares entries the interpreter does not know: "
                f"{unknown}{hint}"
            )
        keyed = {WIRE_NAMES[name]: entry for name, entry in entries.items()}
        reserved = sorted(key.value for key in keyed if key in PRESCRIPTION)
        if reserved:
            raise ValueError(
                f"{cls.__name__} binds the prescription entries {reserved}, which "
                "pulserver applies when it builds the IR"
            )
        cls.protocol = keyed
        if "ui" in declared:
            cls.ui = keyed

    def listing(self) -> dict[ProtocolKey, Parameter]:
        """Return the protocol with its schema, valued at the app's defaults.

        An argument defaulting to ``None`` shows the preset that requests ``None``.
        The ``PRESCRIPTION`` entries of :mod:`pulserver.protocol` follow the
        declared ones, not editable in the UI: the offset at zero, the rotation
        at the identity.
        """
        defaults = self._defaults()
        listing = {
            key: _parameter(key, entry, defaults)
            for key, entry in self.protocol.items()
        }
        for key in FOV_OFFSET:
            listing[key] = Parameter(
                Kind.FLOAT,
                0.0,
                InputMode.OFF,
                -OFFSET_LIMIT_MM,
                OFFSET_LIMIT_MM,
                0.1,
                "mm",
            )
        for key, value in zip(FOV_ROTATION, np.eye(3).ravel(), strict=True):
            listing[key] = Parameter(
                Kind.FLOAT, float(value), InputMode.OFF, -1.0, 1.0, 1e-6, ""
            )
        return listing

    def evaluate(self, system: pp.Opts, protocol: Protocol) -> Evaluation | None:
        """Evaluate a requested protocol, or refuse it by raising.

        The hook a subclass overrides to check a protocol and to state what the
        console shows with it. Returning ``None`` is returning
        ``Evaluation(protocol)``. An exception makes the protocol invalid, as
        :meth:`validate` describes.

        The default for a function app accepts the protocol unchanged with no
        scan time estimate, and does not call the app. The default for a
        sequence application constructs it, which checks the prescription, and
        returns its ``scan_time()`` and the protocol with each entry holding
        the value the design took, as the application records it with
        ``SequenceApp.resolve``; an argument it does not record keeps its
        requested value.

        Parameters
        ----------
        system
            The scanner limits the sequence is designed under.
        protocol
            The requested protocol, in the units of the app's arguments.

        Returns
        -------
        Evaluation or None
            The protocol to show back, the scan time and a note.
        """
        if not self._is_application():
            return Evaluation(protocol)
        app = self._application(system, protocol)
        duration = app.scan_time()
        readback = app.resolved
        recorded = {}
        for key, entry in self.protocol.items():
            value = readback.get(getattr(entry, "argument", None))
            if value is not None:
                recorded[key] = value
        return Evaluation(protocol.replace(recorded), duration)

    def generate(
        self, system: pp.Opts, protocol: Protocol
    ) -> pp.Sequence | list[pp.Sequence] | sequences.SequenceApp:
        """Return what :meth:`design` writes for a protocol.

        The hook a subclass overrides to build the sequence. The default for a
        function app returns ``app(system, **protocol.arguments)``. The default
        for a sequence application returns the application constructed from the
        arguments; :meth:`design` designs it, with its prescans.

        Parameters
        ----------
        system
            The scanner limits the sequence is designed under.
        protocol
            The requested protocol, in the units of the app's arguments.

        Returns
        -------
        pypulseqpp.Sequence or list of pypulseqpp.Sequence or SequenceApp
            A sequence, or a chain of them, prescans first and the main
            sequence last, or an application that designs its own chain.
        """
        if self._is_application():
            return self._application(system, protocol)
        return self.app(system, **protocol.arguments)

    def validate(
        self, system: pp.Opts, request: Mapping[ProtocolKey, Any]
    ) -> Validation:
        """Return the validation of a request, as the wire carries it.

        The request is in wire values, completed with the initial values of the
        entries it omits. It is converted to a :class:`Protocol`, its
        prescription rotation is checked to be orthonormal, and :meth:`evaluate`
        runs; nothing is generated. This is the one boundary between a request
        and the code of a plugin.

        A valid reply carries the evaluated protocol, the duration of the
        evaluation, ``None`` where it is ``0.0``, and its note. An exception
        raised by any of these steps makes the reply invalid, carrying the
        request, and is logged with its traceback. ``ValueError`` and
        ``AssertionError``, which pypulseqpp and PyPulseq raise for a protocol
        they cannot realize, are logged at WARNING and the message is the
        reply's ``info``. Any other exception is logged at ERROR, and the
        ``info`` is ``"<ExceptionType> in <Plugin>.evaluate"``.

        Raises
        ------
        ValueError
            If the declared entries are inconsistent with the app, as
            :meth:`listing` reports; this is a defect of the plugin and not of
            the request.
        """
        return self._validated(system, request)[0]

    def design(
        self,
        system: pp.Opts,
        request: Mapping[ProtocolKey, Any],
        directory: Path | str,
    ) -> tuple[Validation, list[str]]:
        """Write the sequence of a request into ``directory`` as signed binary Pulseq.

        The request is evaluated as in :meth:`validate`, :meth:`generate`
        builds the sequence of the requested protocol, and each sequence is
        written into ``directory``, which must exist. The request is designed
        as it stands: a request that evaluates to other values is designed
        from the values it was given, so a design that is a function of the
        evaluated protocol is made from ``validate(...).values``.

        The first file is ``sequence.seq``. Each later file of a chain is
        ``sequence_<name>.seq``, ``main`` for the last, and each file names the
        next as its ``NextSequence`` definition, so the chain is one scan.
        ``name`` is the name of the prescan for a sequence application, and
        ``prescan<n>`` for a list of sequences, ``n`` counting from 2.

        Returns
        -------
        Validation
            As :meth:`validate` returns it.
        list of str
            Written paths in play order, prescans first; empty for an invalid
            request, for which nothing is written.

        Raises
        ------
        TypeError
            If a function app returns neither a sequence nor a list of them.
        """
        try:
            validation, protocol = self._validated(system, request)
            if protocol is None:
                return validation, []
            built = self.generate(system, protocol)
            return validation, _write(built, Path(directory) / _FIRST_FILE)
        finally:
            self._built = None

    def rf_pulses(
        self, seq_path: Path | str, designed: Mapping[str, Any]
    ) -> list[RfPulse]:
        """Return the RF a design plays, as a scanner costs it while prescribing.

        The pulses and their angles are read off the sequence a design wrote,
        so nothing is designed to find them; which control drives each is
        :attr:`follows`, declared. A pulse takes a share of its parameter: the
        angle it was designed at over the value the parameter was designed
        with, so a refocusing train whose angles vary gets one share each.

        Parameters
        ----------
        seq_path
            The design's first sequence file.
        designed
            The prescription as designed, by the name of the app argument each
            entry binds.

        Returns
        -------
        list of RfPulse
            One entry per distinct pulse and angle. Empty where
            :attr:`follows` declares nothing, which leaves a scanner to cost
            the RF only once it has the design itself.

        Raises
        ------
        ValueError
            When the sequence does not play as many distinct pulses as
            :attr:`follows` declares, which would silently cost pulses against
            the wrong controls.
        """
        if not self.follows:
            return []
        played = ir.played_rf(pp.io.read(Path(seq_path)))
        order: list[int] = []
        for definition, _ in played:
            if definition not in order:
                order.append(definition)
        if len(order) != len(self.follows):
            raise ValueError(
                f"{type(self).__name__} declares {len(self.follows)} pulses but its "
                f"sequence plays {len(order)} distinct ones"
            )
        driving = {}
        for parameter in self.follows:
            entry = self.protocol.get(parameter or "")
            driving[parameter or ""] = (
                float(designed.get(entry.argument, 0.0) or 0.0) if entry else 0.0
            )

        found: dict[tuple[int, int], RfPulse] = {}
        for definition, angle in played:
            # One entry per angle: a definition played at several angles is a
            # train, and each of its pulses is costed at its own.
            key = (definition, round(angle * 1e3))
            if key in found:
                continue
            parameter = self.follows[order.index(definition)] or ""
            drives = driving[parameter]
            found[key] = RfPulse(
                definition=definition,
                flip_deg=angle,
                follows=parameter if drives else "",
                factor=angle / drives if drives else 1.0,
            )
        return list(found.values())

    @classmethod
    def _is_application(cls) -> bool:
        return inspect.isclass(cls.app) and issubclass(cls.app, sequences.SequenceApp)

    def _defaults(self) -> dict[str, Any]:
        """Return the default of each argument of the app, ``inspect.Parameter.empty`` for none."""
        if self._is_application():
            return self.app.protocol()
        # The first parameter is the scanner limits.
        arguments = list(inspect.signature(self.app).parameters.values())[1:]
        return {
            argument.name: argument.default
            for argument in arguments
            if argument.kind not in (argument.VAR_POSITIONAL, argument.VAR_KEYWORD)
        }

    def _application(
        self, system: pp.Opts, protocol: Protocol
    ) -> sequences.SequenceApp:
        """Return the application of a protocol, constructed once for one system and arguments."""
        arguments = protocol.arguments
        held = self._built
        if held is not None and held[0] is system and held[1] == arguments:
            return held[2]
        app = self.app(system, **arguments)
        self._built = (system, arguments, app)
        return app

    def _validated(
        self, system: pp.Opts, request: Mapping[ProtocolKey, Any]
    ) -> tuple[Validation, Protocol | None]:
        """Return the validation of a request and the protocol it requests, ``None`` where invalid."""
        listing = self.listing()
        values = {key: p.value for key, p in listing.items() if p.editable}
        values.update(request)
        name = type(self).__name__
        try:
            prescribed_rotation(values)
            protocol = Protocol.from_wire(self.protocol, values, system)
            evaluation = self.evaluate(system, protocol)
            if evaluation is None:
                evaluation = Evaluation(protocol)
            wire = evaluation.protocol.to_wire()
            duration = float(evaluation.duration)
            info = evaluation.info
        except _INFEASIBLE as error:
            message = str(error) or type(error).__name__
            _log.warning(
                "infeasible protocol in %s.evaluate: %s", name, message, exc_info=True
            )
        except Exception as error:
            message = f"{type(error).__name__} in {name}.evaluate"
            _log.error("%s", message, exc_info=True)
        else:
            return Validation(True, duration or None, info, wire), protocol
        asked = {key: value for key, value in values.items() if key in listing}
        return Validation(False, None, message, asked), None


class ScannerSequence(SequencePlugin):
    """Deprecated name of :class:`SequencePlugin`.

    Warns
    -----
    DeprecationWarning
        When a class subclasses it.
    """

    _deprecated_alias = True


def _write(
    built: pp.Sequence | list[pp.Sequence] | sequences.SequenceApp, first: Path
) -> list[str]:
    """Write a sequence application, a sequence or a chain of them as signed binary Pulseq; return the paths in play order."""
    if isinstance(built, sequences.SequenceApp):
        return built.write(first, offline=False)
    chain = [built] if isinstance(built, pp.Sequence) else built
    if (
        not isinstance(chain, list | tuple)
        or not chain
        or not all(isinstance(seq, pp.Sequence) for seq in chain)
    ):
        raise TypeError(
            "the app returned neither a Sequence nor a list of Sequences: "
            f"{type(built).__name__}"
        )
    later = [f"prescan{n}" for n in range(2, len(chain))] + ["main"]
    paths = [first] + [
        first.with_name(f"{first.stem}_{name}.seq") for name in later[: len(chain) - 1]
    ]
    for seq, path, following in zip(chain, paths, [*paths[1:], None], strict=True):
        if following is not None:
            seq.set_definition(key="NextSequence", value=following.name)
        pp.io.write(seq, str(path), binary=True)
    return [str(path) for path in paths]


def load_plugin(path: Path) -> SequencePlugin:
    """Import a plugin file and instantiate the one SequencePlugin it defines.

    The module is registered as ``pulserver_plugin_<file stem>``; loading
    another file with the same stem replaces it. A subclass of
    :class:`ScannerSequence` is a plugin too.

    Raises
    ------
    ValueError
        If the file defines no SequencePlugin subclass, or more than one.
    """
    path = Path(path)
    name = f"pulserver_plugin_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    found = [
        obj
        for obj in vars(module).values()
        if inspect.isclass(obj)
        and issubclass(obj, SequencePlugin)
        and obj.__module__ == name
    ]
    if len(found) != 1:
        raise ValueError(
            f"{path} defines {len(found)} SequencePlugin subclasses; a plugin defines one"
        )
    return found[0]()
