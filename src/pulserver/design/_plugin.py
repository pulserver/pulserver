"""Scanner-sequence plugins: a sequence app bound to the scanner protocol."""

from __future__ import annotations

import difflib
import functools
import importlib.util
import inspect
import logging
import sys
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pypulseqpp as pp

from ..protocol import (
    FOV_OFFSET,
    FOV_ROTATION,
    PRESCRIPTION,
    InputMode,
    Kind,
    Parameter,
    ProtocolKey,
    Validation,
    prescribed_rotation,
)
from ..protocol._keys import WIRE_NAMES
from ._entries import Entry, Protocol, _parameter
from ._rf import RfLayout

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
    rf_layout
        The RF one TR of the protocol plays, from which a scanner estimates the
        RF of a protocol before it is designed. ``()`` or ``[]``, the default,
        states no estimate and is not invalid. Every control of the layout must
        be an entry of :attr:`SequencePlugin.protocol` whose value in
        ``protocol`` is positive, or the protocol is invalid.
    """

    protocol: Protocol
    duration: float = 0.0
    info: str = ""
    rf_layout: RfLayout | tuple[()] = ()


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
        A function ``app(system, **arguments)`` that designs the sequence. It
        takes the scanner limits, a :class:`pypulseqpp.Opts`, and the keyword
        arguments the entries of :attr:`protocol` bind, and returns a
        :class:`pypulseqpp.Sequence` or a list of them, which is a chain: the
        prescans first and the main sequence last. The defaults the listing
        shows are those of its signature, so a :func:`functools.partial` is an
        app. :meth:`generate` calls it, and so does an :meth:`evaluate` that
        designs the sequence; the default :meth:`evaluate` does not.
    protocol : mapping
        The entries the operator edits, by key: the interpreter's parameter
        names, which are the members of
        :data:`~pulserver.protocol.ProtocolKey` that
        :class:`~pulserver.protocol.UIParam` collects. A plain string naming an
        entry is stored as its member. Empty by default.

    Raises
    ------
    ValueError
        When a subclass is defined with a ``protocol`` key the interpreter does
        not know, which its parser would drop, or with one of the prescription
        entries, which pulserver applies itself; or with both ``protocol`` and
        ``ui``.
    TypeError
        When a subclass is defined with a class as ``app``.

    Warns
    -----
    DeprecationWarning
        When a subclass sets ``recon``, which has no effect: the reconstruction
        is named by the reconstruction client or, on a console, by the scan.
        When a subclass sets ``follows``, which has no effect: the RF a scanner
        scales is the :class:`RfLayout` an evaluation returns.
        When a subclass sets ``ui``, the deprecated name of ``protocol``, or
        subclasses :class:`ScannerSequence`, the deprecated name of this class.
    """

    app: ClassVar[Callable[..., Any]]
    protocol: ClassVar[Mapping[ProtocolKey, Entry]] = {}

    _deprecated_alias: ClassVar[bool] = False

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
        if "follows" in declared:
            warnings.warn(
                f"{cls.__name__} sets follows, which has no effect: return an "
                "RfLayout from evaluate",
                DeprecationWarning,
                stacklevel=2,
            )
        app = declared.get("app")
        if inspect.isclass(app):
            raise TypeError(
                f"{cls.__name__} binds the class {app.__name__} as app; app is a "
                "function returning a sequence or a list of them"
            )
        if inspect.isfunction(app) or isinstance(app, functools.partial):
            cls.app = staticmethod(app)
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

    def evaluate(self, system: pp.Opts, protocol: Protocol) -> Evaluation | None:  # noqa: ARG002 -- the signature of the hook a subclass overrides
        """Evaluate a requested protocol, or refuse it by raising.

        The hook a subclass overrides to check a protocol and to state what the
        console shows with it. Returning ``None`` is returning
        ``Evaluation(protocol)``. An exception makes the protocol invalid, as
        :meth:`validate` describes.

        The default accepts the protocol unchanged with no scan time estimate,
        and does not call the app.

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
        return Evaluation(protocol)

    def generate(
        self, system: pp.Opts, protocol: Protocol
    ) -> pp.Sequence | list[pp.Sequence]:
        """Return what :meth:`design` writes for a protocol.

        The hook a subclass overrides to build the sequence. The default
        returns ``app(system, **protocol.arguments)``.

        Parameters
        ----------
        system
            The scanner limits the sequence is designed under.
        protocol
            The requested protocol, in the units of the app's arguments.

        Returns
        -------
        pypulseqpp.Sequence or list of pypulseqpp.Sequence
            A sequence, or a chain of them, prescans first and the main
            sequence last.
        """
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
        evaluation, ``None`` where it is ``0.0``, its note and its RF layout,
        ``None`` where it states none or plays no RF. An exception raised by
        any of these steps makes the reply invalid, carrying the request, and
        is logged with its traceback. A control of the RF layout that is not an
        entry of :attr:`protocol`, or whose value in the evaluated protocol is
        not positive, is a ``ValueError`` naming it. ``ValueError`` and
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
        ``sequence_prescan<n>.seq``, ``n`` counting from 2, or
        ``sequence_main.seq`` for the last, and each file names the next as its
        ``NextSequence`` definition, so the chain is one scan.

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
            If ``app`` returns neither a sequence nor a list of them.
        """
        validation, protocol = self._validated(system, request)
        if protocol is None:
            return validation, []
        built = self.generate(system, protocol)
        return validation, _write(built, Path(directory) / _FIRST_FILE)

    def _defaults(self) -> dict[str, Any]:
        """Return the default of each argument of the app, ``inspect.Parameter.empty`` for none."""
        # The first parameter is the scanner limits.
        arguments = list(inspect.signature(self.app).parameters.values())[1:]
        return {
            argument.name: argument.default
            for argument in arguments
            if argument.kind not in (argument.VAR_POSITIONAL, argument.VAR_KEYWORD)
        }

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
            layout = _stated_layout(self.protocol, evaluation)
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
            return Validation(True, duration or None, info, wire, layout), protocol
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


def _stated_layout(
    declared: Mapping[ProtocolKey, Entry], evaluation: Evaluation
) -> RfLayout | None:
    """Return the RF layout of an evaluation, ``None`` where it states none or plays no RF.

    Raises
    ------
    TypeError
        If ``rf_layout`` is neither an :class:`RfLayout` nor empty.
    ValueError
        If a control is not an entry of ``declared``, or its value in the
        evaluated protocol is not positive.
    """
    layout = evaluation.rf_layout
    if not isinstance(layout, RfLayout):
        if isinstance(layout, tuple | list) and not layout:
            return None
        raise TypeError(
            f"rf_layout is an RfLayout or empty, not {type(layout).__name__}"
        )
    for control in dict.fromkeys(c for c in layout.control if c is not None):
        if control not in declared:
            raise ValueError(
                f"the RF layout is scaled by {control}, which is not an entry of "
                "the protocol"
            )
        value = evaluation.protocol.get(control)
        if isinstance(value, bool) or not isinstance(value, Real) or not value > 0:
            raise ValueError(
                f"the RF layout is scaled by {control}, whose evaluated value "
                f"{value!r} is not positive"
            )
    return layout if len(layout.instances.definition) else None


def _write(built: pp.Sequence | list[pp.Sequence], first: Path) -> list[str]:
    """Write a sequence or a chain of them as signed binary Pulseq; return the paths in play order."""
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
