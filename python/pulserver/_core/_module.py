"""Common protocol for reusable sequence modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from typing import Any, TypeVar, overload

Block = tuple[Any, ...]
_ModuleT = TypeVar("_ModuleT", bound="SequenceModule")


class SequenceModule(Sequence[Block], ABC):
    """A reusable group of Pulseq blocks that re-renders itself per shot.

    Pulseq gives you *events* and *blocks*; a sequence is the loop you write
    over them. A module is the missing middle: the handful of blocks that
    always travel together — an excitation with its slice-select and rephaser,
    a preparation with its delay and spoiler, one readout train — designed
    **once** and replayed thousands of times with only a few numbers changing.

    Waveforms are built at construction. Per-shot variation (phase-encode
    index, RF phase, slice frequency, rotation) goes through ``set_state`` and
    is re-rendered lazily, so a loop over thousands of shots costs no redesign.

    A module *is* an immutable sequence of blocks for its current state, which
    makes the sequence loop plain Pulseq::

        readout.set_state(lin_idx=ky)
        for block in readout:
            seq.add_block(*block)

    ``len(module)`` is the block count, ``module[i]`` is one block, and each
    block is the tuple of events ``add_block`` expects. Nothing here is
    callable: ``set_state`` then iterate is the only path, so the argument
    names you read in ``set_state`` are the argument names you use.

    Subclass this only to implement a *new* reusable RF, preparation,
    encoding or readout module — the shipped factories cover the standard
    families. A subclass provides ``set_state`` and ``_current_blocks``; the
    collection protocol, labels, flags, triggers, duration, plotting and
    ``add_to`` come free.

    State keywords are shared vocabulary across every shipped module, so the
    same quantity is always spelled the same way:

    ============================ =========================================
    ``lin_idx``, ``par_idx``     absolute ``LIN``/``PAR`` index — both the
                                 encoding step and the emitted label
    ``phase_offset_rad``         transmit/receive phase (RF spoiling,
                                 phase cycling)
    ``freq_offset_hz``           transmit/receive frequency offset (slice
                                 selection, fat saturation, FOV shift)
    ``amplitude_scale``          RF amplitude scaling (variable flip angle)
    ``rotation``                 rotation applied to every gradient block
    ============================ =========================================

    Four setters, four separate concerns, each replacing its own state and
    none of them touching the others:

    ==================== ====================================================
    :meth:`set_state`    the numbers that re-render the waveforms
    :meth:`set_labels`   per-shot **counters** — where this data belongs
                         (``SLC``, ``REP``, ``SET``, ``PHS``, ``AVG``)
    :meth:`set_flags`    sticky **flags** — how these blocks are played or
                         classified (``NOROT``, ``NOPOS``, ``OFF``,
                         ``ONCE``, ``MODULE``, ``NAV``, ``PMC``)
    :meth:`set_triggers` trigger and digital-output events, per block
    ==================== ====================================================

    Parameters
    ----------
    system : pypulseq.Opts
        System limits the module was designed against.

    Attributes
    ----------
    system : pypulseq.Opts
        The limits passed at construction.

    Examples
    --------
    >>> import numpy as np
    >>> import pulserver.design as design
    >>> import pulserver.pypulseq as pp
    >>> from pulserver import SequenceModule
    >>> system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    >>> readout = design.make_line_readout(system, (0.22, 0.22), (128, 128))
    >>> isinstance(readout, SequenceModule)
    True
    >>> readout.set_state(lin_idx=0).num_blocks
    3

    Design once, re-index per shot — the whole GRE loop::

        excitation = design.make_slice_selective_pulse(np.deg2rad(15), 5e-3, system=system)
        readout = design.make_line_readout(system, fov, matrix)

        for ky in range(matrix[1]):
            excitation.set_state(phase_offset_rad=phases[ky])
            for block in excitation:
                seq.add_block(*block)
            readout.set_state(lin_idx=ky, phase_offset_rad=phases[ky])
            for block in readout:
                seq.add_block(*block)

    ``add_to`` collapses each of those two-line loops into one when the module
    needs no per-block handling::

        excitation.set_state(phase_offset_rad=phases[ky]).add_to(seq)
        readout.set_state(lin_idx=ky, phase_offset_rad=phases[ky]).add_to(seq)

    See Also
    --------
    SequenceModule.set_labels : attach per-shot counters to the first block.
    SequenceModule.set_flags : attach sticky flags, scoped to this module.
    SequenceModule.set_triggers : attach triggers and digital outputs.
    pulserver.ScanLoop : the encoding indices to feed ``set_state``.
    """

    #: Set through the ``duration`` setter by modules that know their timing.
    _duration: float | None = None
    _labels: tuple[Any, ...] = ()
    _flags: tuple[tuple[str, int, bool], ...] = ()
    _triggers: tuple[tuple[int, tuple[Any, ...]], ...] = ()

    def __init__(self, system: Any) -> None:
        self.system = system

    @abstractmethod
    def set_state(self: _ModuleT, *args: Any, **kwargs: Any) -> _ModuleT:
        """Replace the complete dynamic state and return ``self``."""

    @abstractmethod
    def _current_blocks(self) -> tuple[Block, ...]:
        """Return the immutable block snapshot for the current state."""

    def set_labels(self: _ModuleT, *labels: Any, **counters: int) -> _ModuleT:
        """Replace the counters merged into the first block, and return ``self``.

        Modules emit the counters they own — a readout knows its own ``LIN``,
        ``PAR`` and ``ECO``. Everything an outer *loop* owns (``SLC``,
        ``REP``, ``SET``, ``PHS``, ``AVG``, ``SEG``) belongs to the caller,
        and this is where it goes, so the loop never has to unpack the
        snapshot by hand to reach block 0.

        Counters may be passed as ready-made label events — normally straight
        from :meth:`pulserver.ScanLoop.labels`, which is what makes the
        ``FIRST_IN_*``/``LAST_IN_*`` MRD flags come out right — or as keyword
        arguments, which build the equivalent ``SET`` events for you.

        Like :meth:`set_state`, this replaces the complete counter state; call
        it with no arguments to clear. Flags set through :meth:`set_flags`
        are *not* affected, because they describe the module rather than the
        shot.

        Parameters
        ----------
        *labels
            Label events from :func:`pulserver.pypulseq.make_label`, or
            whole tuples from :meth:`pulserver.ScanLoop.labels`.
        **counters
            ``LABEL=value`` pairs, each turned into a ``SET`` label event.

        Returns
        -------
        SequenceModule
            ``self``, so it chains with :meth:`set_state` and :meth:`add_to`.

        Examples
        --------
        Tag every shot of a multi-slice, multi-frame loop::

            for f in range(len(frames)):
                for s in range(len(slices)):
                    excitation.set_state(freq_offset_hz=offsets[s])
                    excitation.set_labels(*frames.labels(f), *slices.labels(s))
                    excitation.add_to(seq)

        The keyword form spells the same thing without a loop object::

            excitation.set_labels(SLC=3, REP=frame)

        See Also
        --------
        set_flags : the sticky half of the label set.
        pulserver.ScanLoop.labels : where the events normally come from.
        """
        from pulserver.pypulseq import make_label

        events = list(labels)
        events += [make_label(label=name, type="SET", value=value) for name, value in counters.items()]
        self._labels = tuple(events)
        return self

    def set_flags(self: _ModuleT, *, scope: str | None = None, **flags: int) -> _ModuleT:
        """Replace the sticky flags this module carries, and return ``self``.

        Flags are the other half of the Pulseq label set: not *where an
        acquisition belongs* (that is :meth:`set_labels`) but *how these
        blocks are played or classified*. ``NOROT``/``NOPOS``/``NOSCL``
        exempt a preparation from the FOV transform, ``NAV`` routes a
        navigator to its own encoding space, ``OFF`` plays an ADC but drops
        its data, ``ONCE`` marks a preparation or cooldown TR, ``MODULE``
        groups blocks under one safety id, ``PMC`` requests motion
        correction. See :func:`pulserver.pypulseq.get_supported_labels` for
        the full table.

        Pulseq labels are sticky — a value set at one block persists until
        some later block sets it again — so a flag has to be *scoped* or it
        leaks into whatever the sequence plays next. Two scopes, chosen per
        flag by :data:`pulserver.pypulseq.STICKY_FLAGS`:

        - ``'module'`` (the default for everything else) emits the value on
          the module's first block and ``0`` on its last, so a fat-sat's
          ``NOPOS`` cannot escape into the readout that follows. This needs
          at least two blocks; a single-block module emits the opening value
          only, and the caller owns the reset.
        - ``'sticky'`` (the default for ``ONCE``, ``MODULE`` and ``TRID``)
          emits the value once and never resets it, because those three
          deliberately span more than one module: ``ONCE`` delimits a whole
          preparation or cooldown *section*, ``MODULE`` groups consecutive
          modules, ``TRID`` names a repeating TR.

        Like :meth:`set_state`, this replaces the complete flag state; call
        it with no arguments to clear.

        Parameters
        ----------
        scope : {'module', 'sticky'}, optional
            Override the per-flag default for every flag in this call.
        **flags
            ``LABEL=value`` pairs; values are coerced to ``int``, so
            ``NAV=True`` works.

        Returns
        -------
        SequenceModule
            ``self``.

        Raises
        ------
        ValueError
            If ``scope`` is neither ``'module'`` nor ``'sticky'``.

        Examples
        --------
        Play the readout but discard its data, as a prescan TR does:

        >>> import pulserver.design as design
        >>> import pulserver.pypulseq as pp
        >>> readout = design.make_line_readout(pp.Opts(), (0.22, 0.22), (64, 64))
        >>> blocks = readout.set_state(lin_idx=0).set_flags(OFF=1).blocks
        >>> [(e.label, e.value) for e in blocks[0] if getattr(e, "type", "") == "labelset"]
        [('OFF', 1)]
        >>> [(e.label, e.value) for e in blocks[-1] if getattr(e, "type", "") == "labelset"]
        [('OFF', 0)]

        Mark the leading dummy TRs of a steady-state sequence as preparation,
        and group a whole preparation train under one safety module id::

            for _ in range(n_dummies):
                excitation.set_flags(ONCE=1).add_to(seq)
                readout.set_state(lin_idx=0).set_flags(ONCE=1, OFF=1).add_to(seq)

        ``ONCE`` stays set across both modules — that is the point, since the
        interpreter reads it as one contiguous prep *section* — while ``OFF``
        clears at the end of the readout it belongs to.

        See Also
        --------
        set_labels : the per-shot counters.
        pulserver.pypulseq.get_supported_labels : every flag and what it means.
        """
        from pulserver.pypulseq import STICKY_FLAGS

        if scope not in (None, "module", "sticky"):
            raise ValueError("scope must be 'module' or 'sticky'")
        self._flags = tuple(
            (name, int(value), (name in STICKY_FLAGS) if scope is None else scope == "sticky")
            for name, value in flags.items()
        )
        return self

    def set_triggers(self: _ModuleT, *events: Any, block: int = 0) -> _ModuleT:
        """Replace the trigger / digital-output events on one block.

        Physiological triggers (:func:`pypulseq.make_trigger`) and digital
        output pulses (:func:`pypulseq.make_digital_output_pulse`) are
        ordinary block events, but which block they belong on is a property
        of the *module*: a cardiac trigger gates the excitation that opens a
        shot, a scope sync pulse marks the readout that must be captured.

        Call it repeatedly to arm several blocks — each call replaces only
        the events on the block it names. Call it with no events to clear
        that block, or :meth:`set_triggers` with ``block`` omitted and no
        events to clear block 0.

        Parameters
        ----------
        *events
            Trigger or digital-output events.
        block : int, optional
            Block index within the module; negative indices count from the
            end, so ``block=-1`` is its last block.

        Returns
        -------
        SequenceModule
            ``self``.

        Examples
        --------
        Gate every shot on the ECG and pulse the scope on the readout::

            excitation.set_triggers(pp.make_trigger("physio1", duration=100e-6))
            readout.set_triggers(pp.make_digital_output_pulse("osc0", duration=100e-6))

        See Also
        --------
        set_flags : ``TRID``, the label that names a repeating TR block.
        """
        remaining = tuple(item for item in self._triggers if item[0] != block)
        self._triggers = (*remaining, (block, tuple(events))) if events else remaining
        return self

    @property
    def flags(self) -> dict[str, int]:
        """Sticky flags currently set on this module, as ``{label: value}``."""
        return {name: value for name, value, _ in self._flags}

    def _rendered_blocks(self) -> tuple[Block, ...]:
        blocks = self._current_blocks()
        if not blocks or not (self._labels or self._flags or self._triggers):
            return blocks

        from pulserver.pypulseq import make_label

        opening = [make_label(label=name, type="SET", value=value) for name, value, _ in self._flags]
        closing = [make_label(label=name, type="SET", value=0) for name, _, sticky in self._flags if not sticky]
        rendered = [list(block) for block in blocks]
        rendered[0] += [*opening, *self._labels]
        if closing and len(rendered) > 1:
            rendered[-1] += closing
        for index, events in self._triggers:
            rendered[index] += list(events)
        return tuple(tuple(block) for block in rendered)

    def __len__(self) -> int:
        return len(self._rendered_blocks())

    @overload
    def __getitem__(self, index: int) -> Block: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[Block, ...]: ...

    def __getitem__(self, index: int | slice) -> Block | tuple[Block, ...]:
        return self._rendered_blocks()[index]

    def __iter__(self) -> Iterator[Block]:
        return iter(self._rendered_blocks())

    @property
    def blocks(self) -> tuple[Block, ...]:
        """Current immutable block snapshot, labels included."""
        return self._rendered_blocks()

    @property
    def num_blocks(self) -> int:
        """Number of blocks in the current state snapshot."""
        return len(self)

    @property
    def duration(self) -> float:
        """Duration (s) of the current block snapshot.

        Modules that derive their timing analytically publish it here
        directly; otherwise it is summed over the rendered blocks. Either way
        this is the number a TE/TR budget needs, so a plugin never has to
        write ``sum(pp.calc_duration(*block) for block in module)``.
        """
        if self._duration is not None:
            return self._duration
        from pypulseq import calc_duration

        return float(sum(calc_duration(*block) for block in self))

    @duration.setter
    def duration(self, value: float) -> None:
        self._duration = float(value)

    def add_to(self, sequence):
        """Append the current block snapshot and return ``sequence``.

        The one-line form of the canonical loop. Use the explicit
        ``for block in module: seq.add_block(*block)`` instead whenever a
        block needs individual handling.
        """
        for block in self:
            sequence.add_block(*block)
        return sequence

    def get(self):
        """Return a standalone enhanced Pulseq sequence for this module."""
        from pulserver.pypulseq import Sequence as PulseqSequence

        return self.add_to(PulseqSequence(self.system))

    def plot(self, **kwargs):
        """Plot this module alone, as a one-module sequence diagram.

        Replays the current block snapshot into a plain upstream
        :class:`pypulseq.Sequence.Sequence` — Pulserver's own ``Sequence`` is a
        write-only fast builder and cannot be read back — and hands it to
        :meth:`~pypulseq.Sequence.Sequence.plot`, so the RF, gradient and ADC
        axes are drawn exactly as they would be played.

        Call ``set_state`` first to inspect a particular shot. The state a
        factory leaves the module in is the full-amplitude one — the largest
        phase-encode step, the largest blip — which is the worst case for
        gradient and slew inspection.

        Parameters
        ----------
        **kwargs
            Forwarded to :meth:`~pypulseq.Sequence.Sequence.plot`
            (``time_range``, ``time_disp``, ``show_blocks``, ``stacked``,
            ...). ``stacked=True`` and ``plot_now=False`` are the defaults
            here, giving one self-contained figure.

        Returns
        -------
        pypulseq.plot.SeqPlot
            The plot handle returned by PyPulseq.

        Examples
        --------
        Inspect the ky = 0 shot of a Cartesian readout::

            readout = design.make_line_readout(system, (0.22, 0.22), (128, 128))
            readout.set_state(lin_idx=0).plot()
        """
        import pypulseq

        kwargs = {"stacked": True, "plot_now": False, "time_disp": "ms", **kwargs}
        try:
            self._current_blocks()
        except RuntimeError:
            # Never set: index 0 is the most negative phase-encode step, so
            # the default state is already the full-amplitude one.
            self.set_state()
        sequence = pypulseq.Sequence(system=self.system)
        for block in self:
            sequence.add_block(*block)
        return sequence.plot(**kwargs)
