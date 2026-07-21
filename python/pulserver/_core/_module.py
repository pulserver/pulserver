"""Common protocol for reusable sequence modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from typing import Any, TypeVar, overload

Block = tuple[Any, ...]
_ModuleT = TypeVar("_ModuleT", bound="Module")


class Module(Sequence[Block], ABC):
    """Base class for a reusable, stateful sequence fragment.

    Modules expose their current state as an immutable sequence of Pulseq
    blocks. Concrete module classes are returned by the public ``make_*``
    factories. Subclass this type only to implement a new reusable RF,
    preparation, encoding, or readout module.
    """

    def __init__(self, system: Any) -> None:
        self.system = system

    @abstractmethod
    def set_state(self: _ModuleT, *args: Any, **kwargs: Any) -> _ModuleT:
        """Replace the complete dynamic state and return ``self``."""

    @abstractmethod
    def _current_blocks(self) -> tuple[Block, ...]:
        """Return the immutable block snapshot for the current state."""

    def __len__(self) -> int:
        return len(self._current_blocks())

    @overload
    def __getitem__(self, index: int) -> Block: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[Block, ...]: ...

    def __getitem__(self, index: int | slice) -> Block | tuple[Block, ...]:
        return self._current_blocks()[index]

    def __iter__(self) -> Iterator[Block]:
        return iter(self._current_blocks())

    @property
    def blocks(self) -> tuple[Block, ...]:
        """Current immutable block snapshot."""
        return self._current_blocks()

    @property
    def num_blocks(self) -> int:
        """Number of blocks in the current state snapshot."""
        return len(self)

    def add_to(self, sequence):
        """Append the current block snapshot and return ``sequence``."""
        for block in self:
            sequence.add_block(*block)
        return sequence

    def get(self):
        """Return a standalone enhanced Pulseq sequence for this module."""
        from pulserver.pypulseq import Sequence as PulseqSequence

        return self.add_to(PulseqSequence(self.system))

    def _add_or_get(self, sequence):
        return self.get() if sequence is None else self.add_to(sequence)
