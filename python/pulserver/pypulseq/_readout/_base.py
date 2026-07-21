"""Common stateful interface for multi-block readout modules."""

from __future__ import annotations

__all__ = ["Block", "Readout", "normalize_rotation"]

from abc import abstractmethod
from collections.abc import Iterable
from typing import Any, TypeVar

from ..._core._module import Block, Module
from .._rotation import normalize_rotation

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


class Readout(Module):
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
            self._block_cache = tuple(tuple(block) for block in self._build_blocks())
        return self._block_cache
