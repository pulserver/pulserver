"""The base class every reusable sequence module is built on."""

from __future__ import annotations

__all__ = ["SequenceModule"]

import sys
import warnings
from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import Any

#: Analyses a module answers by handing the question to the sequence it built
#: itself in. All of them exist on :class:`pulserver.pypulseq.Sequence`.
SEQUENCE_VIEWS = (
    "calculate_gradient_spectrum",
    "calculate_kspace",
    "calculate_pns",
    "check_timing",
    "paper_plot",
    "plot",
    "plot_kspace",
    "sound",
    "test_report",
    "waveforms_and_times",
)


def _unique(events: tuple) -> list:
    """The distinct objects in ``events``, by identity, in first-seen order."""
    seen: dict[int, Any] = {}
    for event in events:
        seen.setdefault(id(event), event)
    return list(seen.values())


class SequenceModule(ABC):
    """A reusable group of Pulseq blocks, designed once and named.

    Pulseq gives you *events* and *blocks*; a sequence is the loop you write
    over them. A module is the missing middle: the handful of blocks that
    always travel together — an excitation with its slice-select and rephaser,
    a preparation with its spoiler, one whole readout TR. It solves its
    gradients, budgets its TE and TR and lands its ADC on both rasters **once**,
    at construction, and then hands the results back under the names its
    constructor gave them.

    A subclass writes :meth:`init_module`, and it reads like an ordinary
    PyPulseq script::

        class Readout(SequenceModule):
            def init_module(self, system, num_arms=1):
                rf = pp.make_block_pulse(flip_angle=0.2, duration=1e-3, system=system)
                gread = [pp.make_trapezoid("x", area=a, system=system) for a in areas]
                adc = pp.make_adc(num_samples=256, dwell=4e-6, system=system)
                self.seq = pp.Sequence(system)
                for n in range(num_arms):
                    self.seq.add_block(rf)
                    self.seq.add_block(gread[n], adc)

    Everything that reached a block is published under the local variable that
    held it, so the caller writes ``readout.rf``, ``readout.adc``,
    ``readout.gread[n]`` — or ``readout.events.rf``, which is the same object
    and is how to reach an event whose name collides with one of the module's
    own attributes.

    **What a name is published as depends on how many distinct objects wore
    it.** Every event is recorded each time it reaches a block, and the
    recording is then deduplicated by identity: one survivor is published as
    that object, several as a list in play order. So ``rf`` above — one pulse
    added ``num_arms`` times — is a single event, while ``gread`` — one
    gradient per arm — stays a list. A ``gread`` built as ``num_arms``
    references to *one* gradient collapses back to that gradient, which is the
    right answer: a trajectory whose arms are a rotation of a base arm has one
    waveform, however many times it is played.

    :meth:`register` publishes the structure it is handed instead of deducing
    one, for the cases where a one-entry list has to stay a list.

    Using a module is never required, and a module never writes a scan loop.
    It owns the design; the plugin owns the loop::

        for shot, angle in enumerate(angles):
            readout.rf.phase_offset = phases[shot]
            seq.add_block(readout.rf)
            seq.add_block(readout.gread[shot], readout.adc)

    Parameters
    ----------
    *args, **kwargs
        Passed straight to :meth:`init_module`.

    Attributes
    ----------
    events : types.SimpleNamespace
        Every published event, by name.
    seq : pulserver.pypulseq.Sequence
        The sequence the module built itself in, holding one pass of its blocks.
    center : float
        Time from the start of the module to the point it is timed against —
        an RF pulse's isodelay, a readout's echo. The subclass sets it.
    duration : float
        Length of the module (s). Summed from the blocks unless the subclass
        assigns one.

    See Also
    --------
    pulserver.design : the shipped excitation, preparation and readout modules.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._seq = None
        self._blocks: list[tuple] = []
        self._played: dict[int, Any] = {}
        self._init_frame = None
        self._named: set[str] = set()
        self._mirrored: set[str] = set()
        self._duration: float | None = None
        self.events = SimpleNamespace()
        self.center = 0.0

        self.init_module(*args, **kwargs)

        self._finalize()

    @abstractmethod
    def init_module(self, *args: Any, **kwargs: Any) -> None:
        """Build the module: the part a subclass writes.

        Assign ``self.seq``, add blocks to it, and set :attr:`center` if the
        module is timed against something other than its own start. Events are
        published automatically; nothing has to be returned.
        """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @property
    def seq(self):
        """The module's own sequence. Assign one to start building."""
        return self._seq

    @seq.setter
    def seq(self, sequence) -> None:
        from ..pypulseq import Sequence

        if not isinstance(sequence, Sequence):
            raise TypeError(
                f"a module's seq must be a pulserver.pypulseq.Sequence, not {type(sequence).__name__}"
            )
        sequence.__class__ = _recording_sequence_class()
        sequence._module = self
        self._seq = sequence

    def _find_init_frame(self) -> None:
        """Keep hold of the running ``init_module`` frames, found from ``add_block``.

        A frame something still refers to keeps its locals after it returns --
        the same guarantee that lets a traceback be inspected once the stack
        has unwound -- so references taken at the first ``add_block`` are
        enough to read the constructor's *final* namespace in
        :meth:`_finalize`. That costs one stack walk per module rather than a
        dictionary copy per block.

        **Every** ``init_module`` on the stack is kept, not only the innermost.
        A subclass that narrows a family -- a spiral readout over the general
        non-Cartesian one -- builds some of its events itself and delegates the
        rest, so they live in two frames and both are the constructor. Reading
        one would leave half the module unpublished, and asking the author to
        say so would be asking them to know which frame the walk stopped at.
        """
        targets = {
            getattr(init, "__code__", None)
            for init in (getattr(klass, "init_module", None) for klass in type(self).__mro__)
            if init is not None
        }
        frames = []
        frame = sys._getframe(2)
        while frame is not None:
            if frame.f_code in targets and frame.f_locals.get("self") is self:
                frames.append(frame)
            frame = frame.f_back
        self._init_frame = frames

    def _finalize(self) -> None:
        """Check what ``init_module`` owed, and publish what it built."""
        name = type(self).__name__
        if self._seq is None:
            raise TypeError(f"{name}.init_module never assigned self.seq")
        if not self._blocks:
            raise TypeError(f"{name}.init_module added no blocks to self.seq")

        # Innermost frame first, so that where a subclass and its base both
        # bind a name, the subclass's meaning is the one published.
        for frame in self._init_frame or ():
            self._publish_locals(frame.f_locals)
        self._init_frame = None  # the constructor and its locals are free to go

        if not vars(self.events):
            warnings.warn(
                f"{name} published no events: nothing it added to self.seq was named by a local "
                f"variable of {name}.init_module. Call self.publish() from wherever the events "
                "were built, or name them with self.register(name=event).",
                stacklevel=3,
            )

    def _publish_locals(self, namespace) -> None:
        """Publish the played events found in one constructor namespace."""
        for name, value in namespace.items():
            if name.startswith("_") or value is self:
                continue
            items = tuple(value) if isinstance(value, list | tuple) else (value,)
            played = tuple(item for item in items if id(item) in self._played)
            if not played:
                continue
            if name in self._named:
                continue
            self._set_event(name, played)

    def publish(self, **named: Any) -> None:
        """Publish this module's events under the names the caller gave them.

        This happens on its own when ``init_module`` returns. Call it from a
        helper function whose locals ``init_module`` never sees, or pass
        keyword aliases, which win over the automatic names.
        """
        self._publish_locals(sys._getframe(1).f_locals)
        self.register(**named)

    def register(self, **events: Any) -> None:
        """Publish events by name, for whatever :meth:`publish` cannot see.

        Two things separate this from the automatic path. It does not require
        the event to have reached a block, so a bank of waveforms the loop will
        choose from can be published alongside the one that was laid out. And
        it publishes the structure it is **given** rather than deducing one: a
        list stays a list, however many distinct objects are in it. The
        automatic path has to deduce, because it cannot tell a bank of one from
        a single event repeated; a caller naming something has already said.
        """
        for name, event in events.items():
            self._publish(name, event)
            self._named.add(name)

    def _set_event(self, name: str, played: tuple) -> None:
        """Publish ``played`` under ``name``, collapsed to one object if it is one."""
        unique = _unique(played)
        self._publish(name, unique[0] if len(unique) == 1 else unique)

    def _publish(self, name: str, event: Any) -> None:
        """Put ``event`` on ``self.events``, and beside it on the module.

        The copy in the instance dictionary is what makes ``readout.gx_pre`` an
        ordinary attribute read. A design loop takes a dozen of them per shot,
        and reaching every one through :meth:`__getattr__` -- which Python
        calls only after the normal lookup has already failed -- costs more
        than the block it is building.
        """
        self._warn_if_shadowed(name)
        setattr(self.events, name, event)
        vars(self)[name] = event
        self._mirrored.add(name)

    def _warn_if_shadowed(self, name: str) -> None:
        """Say when a published name is one the module itself already answers to."""
        if name in self._mirrored:
            return  # published before: this is a re-publication, not a clash
        if name in vars(self) or hasattr(type(self), name) or name in SEQUENCE_VIEWS:
            warnings.warn(
                f"{type(self).__name__} publishes an event called {name!r}, which is also a "
                f"module attribute; reach it as .events.{name}",
                stacklevel=4,
            )

    # ------------------------------------------------------------------
    # Reading the result
    # ------------------------------------------------------------------

    def __getattr__(self, name: str):
        """Answer for a published event, or for an analysis of the inner sequence.

        Python reaches this only when ordinary lookup has already failed, so a
        module's own attributes always win and nothing an event is called can
        shadow ``seq``, ``center`` or ``duration``; :meth:`_set_event` warns
        when a constructor picks such a name.
        """
        if name.startswith("_"):
            # Never route dunder or private lookups: copy, pickle and inspect
            # probe for those, and answering would answer for the module.
            raise AttributeError(name)
        if name in SEQUENCE_VIEWS:
            sequence = object.__getattribute__(self, "_seq")
            if sequence is not None:
                return getattr(sequence, name)
        try:
            events = object.__getattribute__(self, "events")
        except AttributeError:
            raise AttributeError(name) from None
        try:
            return getattr(events, name)
        except AttributeError:
            raise AttributeError(
                f"{type(self).__name__!r} module has no attribute or event {name!r}"
            ) from None

    def __dir__(self):
        """Attributes, published events and the sequence views, so completion offers all three."""
        return sorted(set(super().__dir__()) | set(vars(self.events)) | set(SEQUENCE_VIEWS))

    @property
    def blocks(self) -> list[tuple]:
        """One tuple of events per block, in the order they were added.

        The structural view: what the module plays, for inspection, timing and
        replay. These are the *same* event objects the module published, so a
        scaled or re-phased event is reflected here without rebuilding
        anything.
        """
        return list(self._blocks)

    @property
    def duration(self) -> float:
        """Length of the module (s).

        Summed from the blocks, unless the module derived its timing
        analytically and assigned the answer.
        """
        if self._duration is not None:
            return self._duration
        return float(self._seq.duration()[0])

    @duration.setter
    def duration(self, value: float) -> None:
        self._duration = float(value)


#: Built on first use, because :mod:`pulserver.pypulseq` needs the optional
#: ``pypulseq`` dependency and this module must import without it --
#: ``pulserver.recon`` runs in the scanner's recon environment, which has no
#: PyPulseq, and reaches ``_core`` for the protocol types.
_RECORDING_SEQUENCE = None


def _recording_sequence_class():
    """The ``Sequence`` subclass a module's own sequence is rebranded into.

    Recording is not something a module should have to ask for. An event that
    reaches a block is an event the module plays, and playing it is what makes
    it worth publishing -- so the sequence notes it on the way past. That is
    also what lets :meth:`SequenceModule._finalize` tell what a constructor
    *uses* from what it merely built along the way.
    """
    global _RECORDING_SEQUENCE
    if _RECORDING_SEQUENCE is None:
        from ..pypulseq import Sequence

        class _ModuleSequence(Sequence):
            def add_block(self, *events):
                module = self._module
                if module._init_frame is None:
                    module._find_init_frame()
                module._blocks.append(events)
                for event in events:
                    module._played.setdefault(id(event), event)
                return super().add_block(*events)

        _RECORDING_SEQUENCE = _ModuleSequence
    return _RECORDING_SEQUENCE
