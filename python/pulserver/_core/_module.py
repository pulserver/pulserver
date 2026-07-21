"""Common protocol for reusable sequence modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from typing import Any, TypeVar, overload

Block = tuple[Any, ...]
_ModuleT = TypeVar("_ModuleT", bound="SequenceModule")


class SequenceModule(Sequence[Block], ABC):
    """Base class for a reusable, stateful sequence fragment.

    Every ``make_*`` factory in :mod:`pulserver.pypulseq` returns one of these:
    an excitation with its gradients, a preparation with its delays and
    spoiler, a whole readout shot. Waveforms are designed **once**, at
    construction; per-shot variation (phase-encode index, RF phase, slice
    frequency, rotation) is applied through ``set_state`` and re-rendered
    lazily, so a loop over thousands of shots costs no redesign.

    A module *is* an immutable sequence of Pulseq blocks for its current state:
    ``len(module)`` is the block count, ``module[i]`` one block tuple, and
    iteration yields them in order. ``add_to(seq)`` appends them; ``get()``
    returns them as a standalone sequence and ``plot()`` draws it. Concrete
    modules also accept ``module(seq, **state)`` as shorthand for
    set-then-append.

    Subclass this only to implement a *new* reusable RF, preparation, encoding
    or readout module — the shipped factories cover the standard families.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits the module was designed against.

    Examples
    --------
    >>> import numpy as np
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    >>> from pulserver import SequenceModule
    >>> readout = pp.make_line_readout(system, (0.22, 0.22), (128, 128))
    >>> isinstance(readout, SequenceModule)
    True
    >>> readout.set_state(lin_idx=0).num_blocks
    3

    Design once, re-index per shot::

        excitation = pp.make_slice_selective_pulse(np.deg2rad(15), 5e-3, system=system)
        readout = pp.make_line_readout(system, fov, matrix)
        for ky in range(matrix[1]):
            excitation(seq, phase_offset_rad=phases[ky])
            readout(seq, pe_idx=ky, rf_phase_rad=phases[ky])
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

            readout = pp.make_line_readout(system, (0.22, 0.22), (128, 128))
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

    def _add_or_get(self, sequence):
        return self.get() if sequence is None else self.add_to(sequence)
