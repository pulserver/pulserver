"""Common stateful collection protocol for RF and preparation modules."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

from ..._core._module import Block, Module
from .._rotation import normalize_rotation
from .._system import copy_event

_ModuleT = TypeVar("_ModuleT", bound="RfModule")


@dataclass(frozen=True)
class RfState:
    """Dynamic offsets applied to an RF module's immutable templates."""

    freq_offset_hz: float = 0.0
    phase_offset_rad: float = 0.0
    amplitude_scale: float = 1.0
    rotation: object | None = None


class RfModule(Module):
    """Stateful RF module exposed as an immutable sequence of Pulseq blocks.

    Factory-time events are retained as templates. ``set_state`` invalidates
    the cached block snapshot; the next collection operation deep-copies the
    templates and applies one global state while preserving relative values
    inside composite preparations.
    """

    def __init__(self, system, blocks: Sequence[Block]) -> None:
        super().__init__(system)
        self._template_blocks = tuple(tuple(block) for block in blocks if block)
        if not self._template_blocks:
            raise ValueError("an RF module must contain at least one block")
        self._state = RfState()
        self._block_cache: tuple[Block, ...] | None = None

    def set_state(
        self: _ModuleT,
        *,
        freq_offset_hz: float = 0.0,
        phase_offset_rad: float = 0.0,
        amplitude_scale: float = 1.0,
        rotation=None,
    ) -> _ModuleT:
        """Replace frequency, phase, amplitude, and explicit rotation state."""
        values = (freq_offset_hz, phase_offset_rad, amplitude_scale)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("RF state values must be finite")
        if amplitude_scale < 0:
            raise ValueError("amplitude_scale must be >= 0")
        self._state = RfState(
            freq_offset_hz=float(freq_offset_hz),
            phase_offset_rad=float(phase_offset_rad),
            amplitude_scale=float(amplitude_scale),
            rotation=normalize_rotation(rotation),
        )
        self._block_cache = None
        return self

    def _build_blocks(self) -> tuple[Block, ...]:
        blocks: list[Block] = []
        state = self._state
        for template in self._template_blocks:
            block = [copy_event(event) for event in template]
            has_gradient = False
            for event in block:
                event_type = getattr(event, "type", None)
                if event_type == "rf":
                    event.signal = np.asarray(event.signal) * state.amplitude_scale
                    event.freq_offset += state.freq_offset_hz
                    event.phase_offset += state.phase_offset_rad
                elif event_type in ("grad", "trap"):
                    has_gradient = True
            if has_gradient and state.rotation is not None:
                block.append(copy_event(state.rotation))
            blocks.append(tuple(block))
        return tuple(blocks)

    def _current_blocks(self) -> tuple[Block, ...]:
        if self._block_cache is None:
            self._block_cache = self._build_blocks()
        return self._block_cache

    @property
    def state(self) -> RfState:
        return self._state

    def __call__(self, seq=None, **state):
        """Apply ``state`` and either append to ``seq`` or return ``get()``."""
        self.set_state(**state)
        return self.get() if seq is None else self.add_to(seq)


class RfPulse(RfModule):
    """One RF/encoding block followed by zero or more rephasing events."""

    def __init__(self, system, rf, gradients: Sequence = (), rephasers: Sequence = ()) -> None:
        self.rf = rf
        self.gradients = tuple(gradients)
        self.rephasers = tuple(rephasers)
        blocks: list[Block] = [(rf, *self.gradients)]
        if self.rephasers:
            blocks.append(self.rephasers)
        super().__init__(system, blocks)
