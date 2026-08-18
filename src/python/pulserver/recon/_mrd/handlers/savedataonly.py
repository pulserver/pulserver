"""Private data-sink reconstruction app."""

from __future__ import annotations

__all__ = ["PLUGIN", "SaveDataOnly"]

from ...plugin import AcquisitionBucket, ReconPlugin, ReconContext


class SaveDataOnly(ReconPlugin):
    """Consume and optionally persist a measurement without emitting images."""

    def recon(
        self,
        bucket: AcquisitionBucket,
        context: ReconContext,
    ) -> None:
        """Discard the in-memory bucket after the connection saver sees it."""
        del bucket, context


PLUGIN = SaveDataOnly()
