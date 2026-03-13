"""Fast sequence helpers for production bridge execution."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
from pypulseq.Sequence import block
from pypulseq.Sequence.sequence import Sequence


class FastSequence(Sequence):
    """Sequential-only Sequence variant with reduced per-block overhead.

    Intended for production generation where events are always appended in order
    and continuity checks are performed by the native interpreter path.
    """

    def __init__(
        self,
        system=None,
        *,
        disable_event_dedup: bool = True,
        disable_continuity_checks: bool = True,
        use_block_cache: bool = False,
    ):
        super().__init__(system=system, use_block_cache=use_block_cache)
        self.disable_event_dedup = disable_event_dedup
        self.disable_continuity_checks = disable_continuity_checks

    @contextmanager
    def _dedup_disabled(self) -> Iterator[None]:
        if not self.disable_event_dedup:
            yield
            return

        libraries = [
            self.adc_library,
            self.delay_library,
            self.extensions_library,
            self.grad_library,
            self.label_inc_library,
            self.label_set_library,
            self.rf_library,
            self.shape_library,
            self.trigger_library,
            self.soft_delay_library,
        ]
        original_finds = [lib.find for lib in libraries]
        original_find_or_inserts = [lib.find_or_insert for lib in libraries]

        def _always_insert(lib, *args, **kwargs):
            new_data = kwargs.get("new_data", args[0] if args else None)
            data_type = kwargs.get("data_type", args[1] if len(args) > 1 else str())
            key_id = lib.next_free_ID
            lib.insert(key_id, new_data, data_type)
            return key_id, False

        try:
            for lib in libraries:
                lib.find = lambda _data, _lib=lib: (_lib.next_free_ID, False)
                lib.find_or_insert = lambda *a, _lib=lib, **k: _always_insert(_lib, *a, **k)
            yield
        finally:
            for lib, original_find in zip(libraries, original_finds, strict=True):
                lib.find = original_find
            for lib, original_find_or_insert in zip(libraries, original_find_or_inserts, strict=True):
                lib.find_or_insert = original_find_or_insert

    @contextmanager
    def _continuity_checks_disabled(self) -> Iterator[None]:
        if not self.disable_continuity_checks:
            yield
            return

        original_max_slew = self.system.max_slew
        self.system.max_slew = np.inf
        try:
            yield
        finally:
            self.system.max_slew = original_max_slew

    def add_block(self, *args):
        """Append a block assuming strictly sequential insertion."""
        with self._dedup_disabled(), self._continuity_checks_disabled():
            block.set_block(self, self.next_free_block_ID, *args)
            self.next_free_block_ID += 1

    def set_block(self, _block_index: int, *args):  # noqa: ARG002
        """Disable positional insertion/update in fast mode."""
        raise NotImplementedError(
            "FastSequence only supports sequential add_block(). " "Use pypulseq.Sequence for random block updates."
        )
