"""Common protocol for reusable sequence modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from operator import attrgetter
from collections.abc import Iterator, Sequence
from typing import Any, TypeVar, overload

Block = tuple[Any, ...]
_ModuleT = TypeVar("_ModuleT", bound="SequenceModule")

#: Lowercase :meth:`SequenceModule.set_state` keywords that name a label rather
#: than a number the waveforms are rebuilt from. Everything else uppercase is
#: taken as a label name directly, so this is only for the two spellings that
#: read better in a loop than their label does.
_LABEL_ALIASES = {"once": "ONCE"}


def _name_values(declared: Any) -> dict[str, int]:
    """Normalise a ``labels=``/``flags=`` declaration to ``{name: value}``.

    A bare name declares the label at zero, which is what a counter a loop is
    about to write, or a flag that is off until something turns it on, wants.
    """
    if hasattr(declared, "items"):
        return {str(name): int(value) for name, value in declared.items()}
    if isinstance(declared, str):
        declared = (declared,)
    return {str(name): 0 for name in declared}


def _layout_match(previous: tuple, current: tuple) -> bool:
    """Whether two structure keys describe the same layout.

    Identity first, value equality only for the primitives a key is otherwise
    made of. A key entry can be a whole event -- a rotation carries a
    :class:`scipy.spatial.transform.Rotation`, whose ``==`` answers an array --
    so nothing but a primitive is ever compared by value.
    """
    if len(previous) != len(current):
        return False
    for old, new in zip(previous, current, strict=True):
        if old is new:
            continue
        if not isinstance(old, int | float | bool | str) or old != new:
            return False
    return True


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
    families. A subclass provides ``_set_state`` and ``_current_blocks``; the
    public :meth:`set_state`, the collection protocol, labels, flags,
    triggers, duration and plotting come free.

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

    Labels and flags are state too, and they go through the same call —
    uppercase keywords are label names, lowercase ones are numbers::

        readout.set_state(lin_idx=ky, phase_offset_rad=phase, SLC=s, REP=f)

    so one line per shot sets everything that moves. See :meth:`set_state`.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits the module was designed against.
    labels : sequence of str or mapping, optional
        Counters this module emits on its first block, in order. Names, or
        ``{name: initial value}``. A counter may also be introduced by naming
        it in :meth:`set_state`; declaring it here is how a module built by a
        factory arrives already carrying it.
    flags : sequence of str or mapping, optional
        Sticky flags this module carries — see :meth:`set_state` for the
        scoping rules.
    flag_scope : {'module', 'sticky'}, optional
        Override the per-flag default scope for every flag in ``flags``.
    triggers : optional
        Trigger and digital-output events. A flat sequence of events arms the
        module's first block; a ``{block index: events}`` mapping arms any
        block, with negative indices counting from the end. Which block a
        trigger belongs on is a property of the module's design, which is why
        this is a construction argument and not part of ``set_state``.

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

    See Also
    --------
    SequenceModule.set_state : the one setter — numbers, counters and flags.
    pulserver.ScanLoop : the encoding indices to feed ``set_state``.
    """

    #: Set through the ``duration`` setter by modules that know their timing.
    _duration: float | None = None
    #: Every counter and flag this module carries, as ``(name, is flag, opening
    #: event, closing event or None)``, **in declaration order** -- which is the
    #: order they are emitted in, so the file a module writes is fixed by how it
    #: was declared and not by an internal split. The events are the *same*
    #: objects every shot -- ``set_state`` writes their ``value`` -- so a TR
    #: template recognises them as its own and a payload cache survives a change
    #: of counter. See :meth:`set_state`.
    _labels: tuple[tuple[str, bool, Any, Any], ...] = ()
    _triggers: tuple[tuple[int, tuple[Any, ...]], ...] = ()
    #: ``name -> (is flag, opening, closing)`` over :attr:`_labels`; built with
    #: the tuple it indexes.
    _label_index: dict[str, Any] | None = None
    #: ``(payloads, ((event, values, index), ...))`` -- where each declared
    #: label's value sits in the payloads a subclass computes directly. See
    #: :meth:`payloads`.
    _label_slots: tuple | None = None
    #: ``(structure key, blocks, record)`` -- see :meth:`_retuned_blocks`.
    _layout: tuple | None = None
    #: Memoised label/flag/trigger merge; see :meth:`_rendered_blocks`.
    _rendered_cache: tuple | None = None
    #: Sample array -> its signed peak; see :meth:`payloads`.
    _peak_cache: dict | None = None
    #: Payloads a subclass builds once and rewrites in place; see
    #: :meth:`_direct_payloads`. Cleared by anything that changes what the
    #: blocks *are* rather than what the numbers in them are -- a structural
    #: change, or a new set of labels, flags or triggers, since those add
    #: entries the payload has to carry.
    _payload_cache: Any = None

    def __init__(
        self,
        system: Any,
        *,
        labels: Any = (),
        flags: Any = (),
        flag_scope: str | None = None,
        triggers: Any = (),
    ) -> None:
        self.system = system
        if labels:
            for name, value in _name_values(labels).items():
                self._declare_label(name, value, False, None)
        if flags:
            for name, value in _name_values(flags).items():
                self._declare_label(name, value, True, flag_scope)
        elif flag_scope is not None:
            raise ValueError("flag_scope was given without any flags to scope")
        if triggers:
            self._declare_triggers(triggers)

    def set_state(self: _ModuleT, *args: Any, **kwargs: Any) -> _ModuleT:
        """Set everything this shot moves — numbers, counters and flags.

        The module's only setter, and the only public method besides iteration.
        Two kinds of keyword, told apart by their spelling:

        - **lowercase** names are the numbers the blocks are re-rendered from
          — ``lin_idx``, ``phase_offset_rad``, ``rotation``. Which ones a
          module takes is its own business and is documented on the module;
          they are passed straight through, and like any state they are
          *replaced* wholesale, so an omitted one falls back to its default.
        - **UPPERCASE** names are labels: a counter (``SLC``, ``REP``,
          ``SET``, ``PHS``, ``AVG``, ``SEG``) saying where this acquisition
          belongs, or a flag (``NOROT``, ``NOPOS``, ``OFF``, ``ONCE``,
          ``NAV``, ``PMC``, ``TRID``) saying how these blocks are played.
          Unlike the numbers these **persist**: a label keeps its value until
          another call changes it, because the event carrying it is part of
          the module's structure from the moment it is first named. Pass
          ``0`` to clear one.

        Any uppercase name is accepted, not only the built-in set, because a
        sequence is allowed its own bookkeeping — a bin index, a preparation
        state — without abusing a built-in counter (see
        :func:`pulserver.pypulseq.make_label`). The cost of that is that a
        misspelled counter becomes a custom label rather than an error;
        :attr:`pulserver.pypulseq.Sequence.custom_labels` is where they show up.

        Two lowercase spellings are label sugar, because they read better in a
        loop than the label does: ``once=`` is ``ONCE``, and ``adc_flag=False``
        is ``OFF=1`` — an ADC that is still played, so timing is unchanged, but
        whose data is dropped. Either may be passed ``None`` to mean "leave it
        as it is".

        A call naming **only** labels leaves the numbers alone, so a counter
        can be updated without restating the shot::

            excitation.set_state(SLC=s)          # counter only, state kept
            excitation.set_state(freq_offset_hz=offsets[s], SLC=s)

        A bare ``set_state()`` is not that case: it replaces the state with
        its defaults, as it always has.

        Flags are sticky in the Pulseq file — a value set at one block holds
        until some later block sets it again — so a flag is *scoped* to the
        module that sets it. ``'module'`` scope (the default for most) emits
        the value on the first block and ``0`` on the last, so a fat-sat's
        ``NOPOS`` cannot escape into the readout that follows; ``'sticky'``
        scope (the default for ``ONCE`` and ``TRID``) emits it
        once and never resets, because those deliberately span more than one
        module. Pass ``flag_scope=`` at construction to override.

        Parameters
        ----------
        *args, **kwargs
            Lowercase keywords and positional arguments go to the module's own
            state; uppercase keywords are label values.

        Returns
        -------
        SequenceModule
            ``self``, so a shot is one expression.

        Raises
        ------
        ValueError
            If an uppercase keyword names a flag whose scope has already been
            fixed differently, or a label value is not an integer.

        Examples
        --------
        Play the readout but discard its data, as a dummy TR does:

        >>> import pulserver.design as design
        >>> import pulserver.pypulseq as pp
        >>> readout = design.make_line_readout(pp.Opts(), (0.22, 0.22), (64, 64))
        >>> blocks = readout.set_state(lin_idx=0, adc_flag=False).blocks
        >>> [(e.label, e.value) for e in blocks[0] if getattr(e, "type", "") == "labelset"]
        [('OFF', 1)]
        >>> [(e.label, e.value) for e in blocks[-1] if getattr(e, "type", "") == "labelset"]
        [('OFF', 0)]

        Tag every shot of a multi-slice, multi-frame loop::

            for f in range(len(frames)):
                for s in range(len(slices)):
                    excitation.set_state(
                        freq_offset_hz=offsets[s],
                        **frames.label_state(f),
                        **slices.label_state(s),
                    )
                    for _block in excitation:
                        seq.add_block(*_block)

        Mark the leading dummy TRs of a steady-state sequence as preparation::

            for _ in range(n_dummies):
                excitation.set_state(once=1)
                readout.set_state(lin_idx=0, once=1, adc_flag=False)

        ``ONCE`` stays set across both modules — that is the point, since the
        interpreter reads it as one contiguous prep *section* — while ``OFF``
        clears at the end of the readout it belongs to.

        See Also
        --------
        pulserver.ScanLoop.label_state : the counter values, straight from a loop.
        pulserver.pypulseq.get_supported_labels : every counter and flag.
        """
        state: dict[str, Any] = {}
        labels: dict[str, Any] = {}
        for key, value in kwargs.items():
            name = _LABEL_ALIASES.get(key)
            if name is not None:
                labels[name] = value
            elif key == "adc_flag":
                labels["OFF"] = None if value is None else int(not value)
            elif key.isupper():
                labels[key] = value
            else:
                state[key] = value

        # Only a call that names nothing *but* labels leaves the numbers alone;
        # a bare set_state() still means "back to the defaults".
        if args or state or not labels:
            self._set_state(*args, **state)
        for name, value in labels.items():
            if value is not None:
                self._set_label(name, value)
        return self

    @abstractmethod
    def _set_state(self: _ModuleT, *args: Any, **kwargs: Any) -> Any:
        """Replace the complete dynamic state.

        What :meth:`set_state` delegates the *numbers* to, once it has taken
        the labels out. A subclass names the state it accepts here; the return
        value is ignored, so it need not hand back ``self``.
        """

    @abstractmethod
    def _current_blocks(self) -> tuple[Block, ...]:
        """Return the immutable block snapshot for the current state."""

    def _set_label(self, name: str, value: Any) -> None:
        """Write one label's value, declaring the event that carries it if new."""
        from pulserver.pypulseq import FLAG_LABELS

        declared = self._label_index
        if declared is not None and name in declared:
            # The event object is part of the structure and stays put; only its
            # number moves, which is what keeps a TR template and a payload
            # cache valid across a change of counter.
            declared[name][2].value = int(value)
        else:
            self._declare_label(name, value, name in FLAG_LABELS, None)

    def _declare_label(self, name: str, value: Any, is_flag: bool, scope: str | None) -> None:
        from pulserver.pypulseq import STICKY_FLAGS, make_label

        if scope not in (None, "module", "sticky"):
            raise ValueError("flag_scope must be 'module' or 'sticky'")
        opening = make_label(label=name, type="SET", value=int(value))
        closing = None
        if is_flag:
            sticky = (name in STICKY_FLAGS) if scope is None else scope == "sticky"
            closing = None if sticky else make_label(label=name, type="SET", value=0)
        entry = (name, is_flag, opening, closing)
        self._labels = (*self._labels, entry)
        self._label_index = {**(self._label_index or {}), name: entry}
        self._declared()

    def _declare_triggers(self, triggers: Any) -> None:
        items = triggers.items() if hasattr(triggers, "items") else ((0, triggers),)
        for block, events in items:
            events = tuple(events) if isinstance(events, list | tuple) else (events,)
            remaining = tuple(item for item in self._triggers if item[0] != block)
            self._triggers = (*remaining, (int(block), events)) if events else remaining
        self._declared()

    def _declared(self) -> None:
        """A label, flag or trigger event was added: the blocks are now different."""
        self._rendered_cache = None
        self._payload_cache = None
        self._label_slots = None

    @property
    def flags(self) -> dict[str, int]:
        """Sticky flags currently set on this module, as ``{label: value}``."""
        return {name: opening.value for name, is_flag, opening, _ in self._labels if is_flag}

    @property
    def labels(self) -> dict[str, int]:
        """Counters currently set on this module, as ``{label: value}``."""
        return {name: opening.value for name, is_flag, opening, _ in self._labels if not is_flag}

    def payloads(self) -> tuple[dict[str, Any], ...]:
        """One dict per block: every value the current state can have moved.

        Keyed the way a pulseq block is (``duration``, ``rf``, ``gx``, ``gy``,
        ``gz``, ``adc``, ``ext``), with the *dynamic* fields of each event --
        the ones that are a number in an event library row rather than a shape
        or a duration.

        Deliberately exhaustive rather than minimal: it reports every dynamic
        field of every event, not the subset that happens to differ between two
        states. Which fields a given module actually moves is its own business
        and varies from one readout to the next, so a template derived by
        probing two states would be right only for the states probed.

        The peak of an RF envelope or an arbitrary gradient is the one field
        here that is not a plain attribute read, so it is cached by sample-array
        identity -- the same trick, and the same justification, as the sequence
        writer's shape caches: a module replays the *same* arrays every shot and
        never writes into them, so a scaled or re-indexed state moves the
        payload's numbers without producing a new waveform. Measured on a line
        readout, that cache is the difference between this being cheaper than
        rebuilding each event's library row and being twice its price; a
        20,000-shot loop fills three entries.
        """
        direct = self._direct_payloads()
        if direct is None:
            cache = self._peak_cache
            if cache is None:
                cache = self._peak_cache = {}
            direct = tuple(_block_payload(block, cache) for block in self._rendered_blocks())
        elif self._labels:
            # A subclass that computes its payloads directly builds them once
            # and rewrites numbers in place, so a counter set after that build
            # would otherwise be reported at the value it was built with. Find
            # each label's slot once and write the live values through it --
            # the same bridge a retuned amplitude uses.
            self._write_label_payloads(direct)
        # Bound once -- a module hands back the same payload objects every shot
        # -- so that ``payload.events`` can find its way back to this block.
        if getattr(direct[0], "_owner", None) is not self if direct else False:
            for index, payload in enumerate(direct):
                payload._owner, payload._index = self, index
        return direct

    def _write_label_payloads(self, payloads: tuple) -> None:
        """Publish the current label values into directly-computed payloads."""
        slots = self._label_slots
        if slots is None or slots[0] is not payloads:
            blocks = self._rendered_blocks()
            entries: list[tuple[Any, list, int]] = []
            # The closing half of a module-scoped flag is a constant zero, so
            # only the opening events are worth a slot.
            for event in (opening for _, _, opening, _ in self._labels):
                entries += [
                    (event, values, index)
                    for values, index in _locate_payloads(blocks, payloads, event, "value")
                ]
            slots = self._label_slots = (payloads, tuple(entries))
        for event, values, index in slots[1]:
            values[index] = event.value

    def waveform_inventory(self) -> tuple[dict[str, Any], ...] | None:
        """Every candidate waveform this module's arbitrary gradients can play.

        ``None`` -- the default -- means "the samples never move", which is true
        of every module that replays one designed waveform under an amplitude, a
        phase or a rotation.

        A module whose samples *do* move returns one entry per block of
        :attr:`blocks`, each a ``{channel: (samples, times)}`` mapping whose
        values are the **ordered, complete** set of candidates for that channel:
        ``samples[k]`` and ``times[k]`` are the waveform this block plays when
        the state selects index ``k``. Waveforms are built at construction, so
        this costs nothing to answer.

        That is the whole of what a moving waveform needs. The sequence registers
        each entry as a shape and a payload then names one **by index**, exactly
        as it names an amplitude -- there is no compressing samples inside
        ``add_block`` to recover an ID the module already knew. What varies per
        shot is a number; the TR is still the template.

        Returns
        -------
        tuple of dict or None
            Per block, ``{channel: (samples, times)}``, or ``None``.

        See Also
        --------
        _direct_payloads : where the selected index is published per shot.
        """
        return None

    def _direct_payloads(self) -> tuple[dict[str, Any], ...] | None:
        """Payloads computed from state, or ``None`` to derive them from events.

        The default is ``None``: derive. A module that overrides this stops
        paying for the round trip the default makes -- ``set_state`` rewrites
        numbers into event objects and :meth:`payloads` reads them straight back
        out, and rescaling a waveform to learn its new peak is the expensive
        half of that. A module knows ``lin_idx -> gradient amplitude`` and
        ``phase_offset_rad -> phase`` directly, so it can build its payload
        dicts **once**, at construction, and have ``set_state`` rewrite only the
        handful of entries that actually move.

        Returning the *same* dicts every call is not only allowed, it is the
        point; the sequence writer copies what it keeps
        (``Sequence._template_set_block``). Return ``None`` for any state the
        override does not cover -- a rotation, say, which changes the block
        structure rather than a number -- and the generic path takes over.
        """
        return None

    def _invalidate(self) -> None:
        """Drop every cached rendering, for a change of *structure* not of state.

        :meth:`set_state` deliberately keeps the layout — that is what makes a
        re-state cheap. Anything that changes which blocks or events a module
        emits (a new sampling mask, dropped rephasers) has to come through
        here instead.
        """
        self._layout = None
        self._rendered_cache = None
        # A structure change designs new waveforms, so the old peaks describe
        # arrays this module no longer holds.
        self._peak_cache = None
        self._payload_cache = None

    def _retuned_blocks(self, key: tuple, render, retune) -> tuple[Block, ...]:
        """Structure built once per ``key``; numbers rewritten on every later state.

        A module's block structure is fixed by its *design*: how many blocks,
        which events, which waveforms. Only a handful of payload numbers move
        from shot to shot -- an encoding amplitude, a transmit phase, a
        counter -- and rebuilding the events to change them re-solves no
        gradient and re-registers no shape.

        ``render`` is therefore called once, with a ``record`` dict to fill
        with the events it will want back, and ``retune`` writes the current
        state's numbers into those events. ``key`` names everything that would
        change the structure rather than a number, so a state that moves one of
        those rebuilds instead.

        The events are shared across states by construction, which is exactly
        what lets the sequence writer keep its shape caches: a retuned
        trapezoid has no waveform to recompress, and a retuned ADC or label is
        two scalars. A snapshot is read for the state that produced it, which
        is what this class has always promised.
        """
        layout = self._layout
        if layout is None or not _layout_match(layout[0], key):
            record: dict = {}
            blocks = tuple(tuple(block) for block in render(record))
            self._layout = (key, blocks, record)
            # A payload slot located by a _direct_payloads override points into
            # payloads built from the blocks that were just replaced, and the
            # peaks describe their waveforms. Neither survives a rebuild.
            self._payload_cache = None
            self._peak_cache = None
            return blocks
        retune(layout[2])
        return layout[1]

    def _rendered_blocks(self) -> tuple[Block, ...]:
        blocks = self._current_blocks()
        if not blocks or not (self._labels or self._triggers):
            return blocks

        # The merge allocates a list per block, and a loop asks for the same
        # snapshot several times per shot (iteration, len, payloads), so it is
        # worth remembering the last one. Identity is a sound key because a
        # declaration replaces these tuples wholesale, and a change of *value*
        # is written into an event the merged blocks already hold.
        cached = self._rendered_cache
        if (
            cached is not None
            and cached[0] is blocks
            and cached[1] is self._labels
            and cached[2] is self._triggers
        ):
            return cached[3]

        closing = [event for _, _, _, event in self._labels if event is not None]
        rendered = [list(block) for block in blocks]
        rendered[0] += [entry[2] for entry in self._labels]
        if closing and len(rendered) > 1:
            rendered[-1] += closing
        for index, events in self._triggers:
            rendered[index] += list(events)
        merged = tuple(tuple(block) for block in rendered)
        self._rendered_cache = (blocks, self._labels, self._triggers, merged)
        return merged

    def __len__(self) -> int:
        return len(self._rendered_blocks())

    @overload
    def __getitem__(self, index: int) -> Block: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[Block, ...]: ...

    def __getitem__(self, index: int | slice) -> Block | tuple[Block, ...]:
        return self._payload_blocks()[index]

    def __iter__(self) -> Iterator[Block]:
        return iter(self._payload_blocks())

    def _payload_blocks(self) -> tuple[Block, ...]:
        """What iterating a module yields: one ``add_block`` call per block.

        Each item is the argument tuple for one block, so the loop reads the way
        it always has::

            for block in module:
                seq.add_block(*block)

        and what crosses is that block's :class:`Payload` -- only the numbers
        this shot moves. Everything else about the block is fixed by the design,
        and a sequence given the TR structure up front already holds it.

        Use :attr:`blocks` to look at the *events* -- plotting, timing, any
        inspection. That is the structural view and it is what ``tr_struct``
        wants; this is the per-shot one.
        """
        return tuple((payload,) for payload in self.payloads())

    @property
    def blocks(self) -> tuple[Block, ...]:
        """Current immutable block snapshot, labels included.

        This is also the module's *structure*: what a plugin hands to
        :class:`pulserver.pypulseq.Sequence` as one subsequence's TR template.
        Everything in it a later shot can move is enumerated by
        :meth:`payloads`; everything else -- waveform shapes, sample counts,
        dwell times, delays -- is fixed here and never rebuilt.
        """
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

        return float(sum(calc_duration(*block) for block in self.blocks))

    @duration.setter
    def duration(self, value: float) -> None:
        self._duration = float(value)

    def get(self):
        """Return a standalone enhanced Pulseq sequence holding just this module.

        The module *is* the structure, played once, so that is what the sequence
        is told: a sequence is always built from the pattern that repeats and how
        many times it repeats, and here the pattern is this module and the count
        is one.
        """
        from pulserver.pypulseq import Sequence as PulseqSequence

        sequence = PulseqSequence(self.system, 1, self)
        for block in self:
            sequence.add_block(*block)
        return sequence

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
        for block in self.blocks:
            sequence.add_block(*block)
        return sequence.plot(**kwargs)


#: Per event type, the payload fields a later shot can move, **in order**.
#: A payload's value is a tuple in exactly this order, not a mapping: the
#: consumer knows which event type sits at each slot of its template, so a name
#: would be looked up per event per shot to learn something already fixed.
#: Everything absent from here -- shape IDs, sample counts, dwell times, delays,
#: rise/flat/fall -- is structure, owned by the module's constructor and never
#: re-read per shot.
_DYNAMIC_FIELDS: dict[str, tuple[str, ...]] = {
    "rf": ("amplitude", "freq_offset", "phase_offset", "freq_ppm", "phase_ppm"),
    "trap": ("amplitude",),
    # ``delay`` moves whenever pypulseq.align right- or left-aligns a solved
    # waveform into a fixed-duration segment, which is per-partition for a
    # stack-of-trajectories prewinder. Without it the template would keep
    # emitting the delay it was built at.
    "grad": ("amplitude", "first", "last", "delay"),
    "adc": ("freq_offset", "phase_offset", "freq_ppm", "phase_ppm"),
    "labelset": ("value",),
    "labelinc": ("value",),
    "trigger": ("delay", "duration"),
    "output": ("delay", "duration"),
}

#: One C-level call per event instead of a getattr per field. ``amplitude`` is
#: absent from these: it is the waveform's peak, which is derived rather than
#: read, and is prepended by hand.
_GETTERS = {
    "rf": attrgetter("freq_offset", "phase_offset", "freq_ppm", "phase_ppm"),
    "grad": attrgetter("first", "last", "delay"),
    "adc": attrgetter("freq_offset", "phase_offset", "freq_ppm", "phase_ppm"),
    "trigger": attrgetter("delay", "duration"),
    "output": attrgetter("delay", "duration"),
}

_CHANNEL_KEY = {"x": "gx", "y": "gy", "z": "gz"}


def _peak(samples, cache: dict) -> float:
    """Signed peak of a waveform: the amplitude its normalised shape is scaled by.

    Keyed by array identity, with the array itself kept alongside the result --
    which both pins the ``id`` against reuse and makes the hit check exact.
    """
    key = id(samples)
    hit = cache.get(key)
    if hit is not None and hit[0] is samples:
        return hit[1]

    import numpy as np

    values = np.asarray(samples)
    magnitude = float(np.max(np.abs(values))) if values.size else 0.0
    if magnitude > 0 and not np.iscomplexobj(values):
        nonzero = np.nonzero(values)[0]
        if nonzero.size:
            first = values[nonzero[0]]
            magnitude *= float(np.sign(first)) if first != 0 else 1.0
    cache[key] = (samples, magnitude)
    return magnitude


class Payload(dict):
    """One block's dynamic numbers, keyed the way a pulseq block is.

    What a module iterates. It carries only what ``set_state`` moves --
    amplitudes, phases, counters, a duration -- because everything else about
    the block is fixed by the design and a sequence given the TR structure
    up front already holds it.

    ``events`` is the block it describes, so a sequence *without* that
    structure can still take it: there is one way to write the loop, and it
    works either way.

    That attribute is deliberately **lazy**. A module that computes its
    payloads directly never writes the numbers into its events -- not doing so
    is the whole point -- so the events are only brought up to date if somebody
    asks for them. A sequence that was given the TR structure never does.
    """

    __slots__ = ("_index", "_owner")

    @property
    def events(self) -> Block:
        """The block this payload describes, retuned to the state it belongs to."""
        owner = getattr(self, "_owner", None)
        if owner is None:
            raise AttributeError(
                "this payload was not published by a module, so it has no events; "
                "give the sequence a tr_struct so the block can be built from the payload"
            )
        return owner._rendered_blocks()[self._index]


def _block_payload(block: Block, cache: dict) -> Payload:
    """Dynamic values of one block, keyed by pulseq slot, each a tuple."""
    payload: Payload = Payload()
    extensions: list[tuple[str, Any]] = []
    duration = 0.0

    for event in block:
        kind = getattr(event, "type", None)
        if kind is None:  # a bare float delay
            duration = max(duration, float(event))
            continue

        if kind == "rf":
            # The registered amplitude is the envelope's peak; the shape is
            # normalised by it, so a flip-angle schedule moves this number only.
            payload["rf"] = (_peak(event.signal, cache), *_GETTERS["rf"](event))
        elif kind == "trap":
            payload[_CHANNEL_KEY[event.channel]] = (event.amplitude,)
        elif kind == "grad":
            channel = _CHANNEL_KEY[event.channel]
            payload[channel] = (_peak(event.waveform, cache), *_GETTERS["grad"](event))
            # An arbitrary gradient's samples travel with it so the sequence
            # can tell "still the shape the template registered" from "this
            # shot re-solved the waveform". Only the answer matters, not the
            # samples: a module is handed every shot's waveform at
            # construction, so the shape is registered there and a payload
            # names it rather than carrying it. Trapezoids have no shape and
            # never take this branch.
            shapes = payload.get("shape")
            if shapes is None:
                shapes = payload["shape"] = {}
            shapes[channel] = event.waveform
        elif kind == "adc":
            payload["adc"] = _GETTERS["adc"](event)
        elif kind == "rot3D":
            extensions.append(("rot3D", tuple(
                event.rot_quaternion.as_quat(canonical=True, scalar_first=True).tolist()
            )))
        elif kind == "rf_shim":
            # Laid out as the library row is -- magnitude and phase interleaved
            # per channel -- so a shim is a payload like any other and a TR
            # template can write it straight into its claimed row. A complex
            # vector would have to be unpacked again on the way in.
            import numpy as _np

            vector = _np.asarray(event.shim_vector)
            extensions.append((
                "rf_shim",
                tuple(_np.stack((_np.abs(vector), _np.angle(vector)), axis=-1).ravel().tolist()),
            ))
        elif kind in ("labelset", "labelinc"):
            extensions.append((kind, (event.value,)))
        elif kind in _GETTERS:
            extensions.append((kind, _GETTERS[kind](event)))
        elif kind in ("delay", "soft_delay"):
            duration = max(duration, float(getattr(event, "delay", 0.0)))

    if extensions:
        payload["ext"] = tuple(extensions)
    if duration:
        payload["duration"] = duration
    return payload


#: Where each dynamic field sits inside its payload tuple. ``_DYNAMIC_FIELDS``
#: already spells the payload layout -- ``_block_payload`` builds every tuple in
#: that order -- so the position is the field's index in it.
_FIELD_INDEX = {
    kind: {name: index for index, name in enumerate(fields)}
    for kind, fields in _DYNAMIC_FIELDS.items()
}

#: Event kinds that reach a block through its extension chain rather than a
#: slot of its own, in the order ``_block_payload`` appends them.
_EXTENSION_TYPES = ("labelset", "labelinc", "trigger", "output", "rot3D", "rf_shim")


def _locate_payloads(blocks, payloads, target, field="amplitude"):
    """Every place one event's dynamic ``field`` sits: a list of ``(values, index)``.

    The bridge a :meth:`SequenceModule._direct_payloads` override is built on:
    it names the events it would otherwise have retuned, and gets back the
    position of each one's number so a later shot is a float write. Payload
    values arrive as tuples; the entry an event owns is swapped for a list, in
    place, so nothing has to be rebuilt to move it.

    An event may be played by more than one block -- one rotation shared by a
    whole segment, say -- and each of those blocks carries its own copy of the
    number, so every one of them is returned and every one has to be written.

    ``field`` is ignored for an extension whose whole value moves together --
    a rotation's quaternion, a shim's vector -- which is written as
    ``values[:] = ...`` against the returned index of 0.
    """
    kind = target.type
    found = []
    for block, payload in zip(blocks, payloads, strict=True):
        ordinal = -1
        for event in block:
            if getattr(event, "type", None) in _EXTENSION_TYPES:
                ordinal += 1
            if event is not target:
                continue
            if kind in _EXTENSION_TYPES:
                extensions = list(payload["ext"])
                name, values = extensions[ordinal]
                values = list(values)
                extensions[ordinal] = (name, values)
                payload["ext"] = tuple(extensions)
                found.append((values, _FIELD_INDEX.get(kind, {}).get(field, 0)))
            else:
                key = _CHANNEL_KEY[event.channel] if kind in ("trap", "grad") else kind
                values = payload[key] = list(payload[key])
                found.append((values, _FIELD_INDEX[kind][field]))
    if not found:
        raise RuntimeError(f"a retuned {kind} event is not in this module's own blocks")
    return found


def _locate_payload(blocks, payloads, target, field="amplitude"):
    """:func:`_locate_payloads` for an event only one block plays."""
    return _locate_payloads(blocks, payloads, target, field)[0]
