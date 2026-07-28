"""Common protocol for reusable sequence modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from typing import Any, TypeVar, overload

Block = tuple[Any, ...]
_ModuleT = TypeVar("_ModuleT", bound="SequenceModule")


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
    #: ``(structure key, blocks, record)`` -- see :meth:`_retuned_blocks`.
    _layout: tuple | None = None
    #: Memoised label/flag/trigger merge; see :meth:`_rendered_blocks`.
    _rendered_cache: tuple | None = None
    #: Sample array -> its signed peak; see :meth:`payloads`.
    _peak_cache: dict | None = None

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
        cache = self._peak_cache
        if cache is None:
            cache = self._peak_cache = {}
        return tuple(_block_payload(block, cache) for block in self._rendered_blocks())

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
            return blocks
        retune(layout[2])
        return layout[1]

    def _rendered_blocks(self) -> tuple[Block, ...]:
        blocks = self._current_blocks()
        if not blocks or not (self._labels or self._flags or self._triggers):
            return blocks

        # The merge allocates a list per block and a label event per flag, and
        # a loop asks for the same snapshot several times per shot (iteration,
        # len, add_to), so it is worth remembering the last one. Every input is
        # replaced wholesale rather than mutated, so identity is a sound key.
        cached = self._rendered_cache
        if (
            cached is not None
            and cached[0] is blocks
            and cached[1] is self._labels
            and cached[2] is self._flags
            and cached[3] is self._triggers
        ):
            return cached[4]

        from pulserver.pypulseq import make_label

        opening = [make_label(label=name, type="SET", value=value) for name, value, _ in self._flags]
        closing = [make_label(label=name, type="SET", value=0) for name, _, sticky in self._flags if not sticky]
        rendered = [list(block) for block in blocks]
        rendered[0] += [*opening, *self._labels]
        if closing and len(rendered) > 1:
            rendered[-1] += closing
        for index, events in self._triggers:
            rendered[index] += list(events)
        merged = tuple(tuple(block) for block in rendered)
        self._rendered_cache = (blocks, self._labels, self._flags, self._triggers, merged)
        return merged

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

    def add_range(self, sequence, **states):
        """Append one instance of this module per entry of the state arrays.

        The one-module form of :meth:`pulserver.pypulseq.Sequence.add_range`::

            readout.add_range(seq, lin_idx=np.arange(ny), phase_offset_rad=phases)

        is the loop it replaces, with arrays where the scalars were::

            for ky in range(ny):
                readout.set_state(lin_idx=ky, phase_offset_rad=phases[ky]).add_to(seq)

        Scalars are broadcast, so only what actually varies has to be an array.
        Use ``Sequence.add_range`` instead when a shot interleaves several
        modules — a group has to be batched as a whole, because event IDs are
        assigned in the order the blocks are visited.

        Parameters
        ----------
        sequence : pulserver.pypulseq.Sequence
            Sequence to append to.
        **states
            Arrays (or scalars) forwarded to :meth:`set_state` per shot.

        Returns
        -------
        pulserver.pypulseq.Sequence
            ``sequence``, so calls chain.

        See Also
        --------
        add_to : the single-shot form.
        """
        return sequence.add_range((self, states))

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


#: Per event type, the payload fields a later shot can move. Everything absent
#: from here -- shape IDs, sample counts, dwell times, delays, rise/flat/fall --
#: is structure, owned by the module's constructor and never re-read per shot.
#: The lists mirror the event library rows in
#: ``pulserver.pypulseq._sequence.Sequence._bulk_event``; an arbitrary gradient
#: carries three numbers because ``scale_grad`` moves ``first`` and ``last``
#: alongside the amplitude, while its normalised shape ID stays put.
_DYNAMIC_FIELDS: dict[str, tuple[str, ...]] = {
    "rf": ("freq_offset", "phase_offset", "freq_ppm", "phase_ppm"),
    "trap": ("amplitude",),
    "grad": ("first", "last"),
    "adc": ("freq_offset", "phase_offset", "freq_ppm", "phase_ppm"),
    "labelset": ("value",),
    "labelinc": ("value",),
    "trigger": ("delay", "duration"),
    "output": ("delay", "duration"),
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


def _block_payload(block: Block, cache: dict) -> dict[str, Any]:
    """Dynamic values of one block, keyed by pulseq slot."""
    payload: dict[str, Any] = {}
    extensions: list[tuple[str, Any]] = []
    duration = 0.0

    for event in block:
        kind = getattr(event, "type", None)
        if kind is None:  # a bare float delay
            duration = max(duration, float(event))
            continue

        if kind == "rf":
            values = {name: getattr(event, name, 0.0) for name in _DYNAMIC_FIELDS["rf"]}
            # The registered amplitude is the envelope's peak; the shape is
            # normalised by it, so a flip-angle schedule moves this number only.
            values["amplitude"] = _peak(event.signal, cache)
            payload["rf"] = values
        elif kind == "trap":
            payload[_CHANNEL_KEY[event.channel]] = {"amplitude": event.amplitude}
        elif kind == "grad":
            values = {name: getattr(event, name, 0.0) for name in _DYNAMIC_FIELDS["grad"]}
            values["amplitude"] = _peak(event.waveform, cache)
            payload[_CHANNEL_KEY[event.channel]] = values
        elif kind == "adc":
            payload["adc"] = {name: getattr(event, name, 0.0) for name in _DYNAMIC_FIELDS["adc"]}
        elif kind == "rot3D":
            extensions.append(("rot3D", tuple(
                event.rot_quaternion.as_quat(canonical=True, scalar_first=True).tolist()
            )))
        elif kind == "rf_shim":
            extensions.append(("rf_shim", tuple(event.shim_vector.tolist())))
        elif kind in _DYNAMIC_FIELDS:
            extensions.append((kind, tuple(getattr(event, name, 0) for name in _DYNAMIC_FIELDS[kind])))
        elif kind in ("delay", "soft_delay"):
            duration = max(duration, float(getattr(event, "delay", 0.0)))

    if extensions:
        payload["ext"] = tuple(extensions)
    if duration:
        payload["duration"] = duration
    return payload
