"""Common stateful interface for multi-block readout modules."""

from __future__ import annotations

__all__ = [
    "Block",
    "Readout",
    "adopt_waveform",
    "normalize_rotation",
    "normalize_slice_rephasing",
    "rescale_trap",
]

from abc import abstractmethod
from collections.abc import Iterable
from typing import Any, TypeVar

import numpy as np

from ..._core._module import Block, SequenceModule
from .._rotation import normalize_rotation


def _gradient_area(event: Any) -> float:
    """Zeroth moment of a trapezoid or arbitrary gradient event."""
    kind = getattr(event, "type", None)
    if kind == "trap":
        return float(event.area)
    if kind == "grad":
        return float(np.trapezoid(np.asarray(event.waveform), np.asarray(event.tt)))
    raise TypeError(f"slice_rephasing expects gradient events, got {kind!r}")


def normalize_slice_rephasing(events: Any, *, forbidden: Iterable[str] = ()) -> dict[str, float]:
    """Collapse slice-rephasing gradient event(s) into one area per axis.

    A readout that accepts ``slice_rephasing`` folds the excitation's
    rephasing moment into its own prewinder rather than letting it cost a
    separate block. Only the *area* survives that move: the prewinder
    re-solves the waveform over its own duration, so the event handed in is
    read for its moment and channel and is never replayed as given.

    Parameters
    ----------
    events : object
        A single gradient event, an iterable of them (an ``RfPulse``'s
        ``rephasers`` tuple), or ``None``.
    forbidden : iterable of str, optional
        Channels the readout cannot share — its own readout axis, whose
        waveform is fixed by the acquisition window.

    Returns
    -------
    dict
        ``{channel: total_area}``, empty when ``events`` is ``None``.

    Raises
    ------
    ValueError
        If a rephaser lands on a forbidden channel.
    TypeError
        If an entry is not a gradient event.
    """
    if events is None:
        return {}
    if hasattr(events, "type") or not isinstance(events, Iterable):
        events = (events,)
    forbidden = set(forbidden)
    areas: dict[str, float] = {}
    for event in events:
        channel = getattr(event, "channel", None)
        if channel is None:
            raise TypeError("slice_rephasing expects gradient events with a channel")
        if channel in forbidden:
            raise ValueError(
                f"slice_rephasing is on channel {channel!r}, which this readout already drives; "
                "the readout axis waveform is fixed by the acquisition window and cannot absorb it"
            )
        areas[channel] = areas.get(channel, 0.0) + _gradient_area(event)
    return {channel: area for channel, area in areas.items() if area != 0.0}


def rescale_trap(event, template, area, worst_mag) -> None:
    """Re-point an already-built scaled trapezoid at ``area``.

    The three fields :func:`pulserver.design.scale_grad` moves, moved by the
    same arithmetic on the same template, so a retuned event is bit-for-bit the
    event a fresh ``scale_grad`` would have returned. ``template`` is ``None``
    only when the worst-case area on this axis is ~0, in which case every
    shot's area is ~0 too and there is nothing to move.
    """
    if template is None:
        return
    scale = area / worst_mag
    event.amplitude = template.amplitude * scale
    event.flat_area = template.flat_area * scale
    event.area = template.area * scale


def adopt_waveform(target, source) -> None:
    """Give ``target`` every field of ``source`` except the delay it was aligned to.

    For the one thing a shot cannot express by rescaling: two gradients summed
    on the same axis are an arbitrary waveform, and the sum moves with the
    encode. The waveform *object* is adopted rather than copied, so a
    memoised source keeps the sequence writer's shape cache hitting -- and
    ``delay`` survives because :func:`pypulseq.align` is the only thing that
    ever set it, and alignment does not depend on the encode.
    """
    delay = target.delay
    target.__dict__.update(source.__dict__)
    target.delay = delay


_State = TypeVar("_State")
_STATE_UNSET = object()


class _BlockCollector:
    """Minimal ``Sequence.add_block`` sink used to snapshot a readout."""

    def __init__(self) -> None:
        self.blocks: list[Block] = []

    def add_block(self, *events: Any) -> None:
        block = tuple(event for event in events if event is not None)
        if block:
            self.blocks.append(block)


class Readout(SequenceModule):
    """A stateful readout exposed as an immutable sequence of Pulseq blocks.

    Concrete readouts replace their complete dynamic state through
    ``set_state()`` and implement ``_build_blocks()``.  The first collection
    operation materializes one block snapshot; later ``len()``, indexing,
    slicing and iteration all reuse that same snapshot until the state changes.
    """

    def __init__(self, system: Any) -> None:
        super().__init__(system)
        self._state: Any = _STATE_UNSET
        self._block_cache: tuple[Block, ...] | None = None

    def _replace_state(self, state: _State) -> None:
        self._state = state
        # Only the *snapshot* is invalidated: the layout survives a state
        # change, which is the whole point of it (see _retuned_blocks).
        self._block_cache = None

    def _require_state(self) -> Any:
        if self._state is _STATE_UNSET:
            raise RuntimeError("set_state() must be called before accessing readout blocks")
        return self._state

    @abstractmethod
    def set_state(self, *args: Any, **kwargs: Any) -> Readout:
        """Replace the complete dynamic state and return ``self``."""

    @abstractmethod
    def _build_blocks(self) -> Iterable[Block]:
        """Build the block snapshot for the current state."""

    def _collect_blocks(self, emit) -> tuple[Block, ...]:
        collector = _BlockCollector()
        emit(collector)
        return tuple(collector.blocks)

    def _current_blocks(self) -> tuple[Block, ...]:
        self._require_state()
        if self._block_cache is None:
            blocks = self._build_blocks()
            # A readout on the layout path already hands back tuples of tuples
            # and hands back the *same* ones every state; re-freezing them would
            # rebuild every block of every shot to change nothing.
            if type(blocks) is not tuple or any(type(block) is not tuple for block in blocks):
                blocks = tuple(tuple(block) for block in blocks)
            self._block_cache = blocks
        return self._block_cache
