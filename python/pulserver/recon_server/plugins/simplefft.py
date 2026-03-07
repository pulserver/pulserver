"""Reference minimal plugin mirroring the classic simplefft behavior."""

from __future__ import annotations

from typing import Any

from ..plugin_api import ReconPlugin


class SimpleFFTPlugin(ReconPlugin):
    """Drain acquisitions and close without modifying transport behavior."""

    def process(self, connection: Any, config: Any, metadata: Any) -> None:
        # Minimal baseline: consume all incoming messages until stream ends.
        for _ in connection:
            pass


def process(connection: Any, config: Any, metadata: Any) -> None:
    """Legacy function-style entrypoint for compatibility."""
    SimpleFFTPlugin().process(connection, config, metadata)
