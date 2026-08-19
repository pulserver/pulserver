"""Private data-sink reconstruction app."""

from __future__ import annotations

__all__ = ["PLUGIN", "SaveDataOnly"]

from ...plugin import ReconPlugin, ReconContext


class SaveDataOnly(ReconPlugin):
    """Consume and optionally persist a measurement without emitting images."""

    def recon(self, branch: str, context: ReconContext) -> None:
        """Emit nothing; the connection saver has already seen the stream."""
        del branch, context


PLUGIN = SaveDataOnly()
