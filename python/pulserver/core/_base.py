"""Abstract base class for pulserver sequence plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pypulseq as pp

from ._params import Protocol


class PulseqSequence(ABC):
    """Contract that every sequence plugin must satisfy.

    Subclass this and implement the three abstract methods.
    The bridge discovers the subclass automatically via ``inspect``.
    """

    @abstractmethod
    def get_default_protocol(self, opts: pp.Opts) -> Protocol:
        """Return the default protocol for this sequence.

        Called once when the plugin is loaded.
        """
        ...

    @abstractmethod
    def validate_protocol(self, opts: pp.Opts, protocol: Protocol) -> dict:
        """Validate *protocol* against hardware *opts*.

        Called on every parameter change — must be fast, no file I/O.

        Returns ``{"valid": bool, "duration": float | None, "info": str | None}``.
        """
        ...

    @abstractmethod
    def make_sequence(
        self, opts: pp.Opts, protocol: Protocol, output_path: str
    ) -> None:
        """Build the full sequence and write the ``.seq`` file to *output_path*."""
        ...
