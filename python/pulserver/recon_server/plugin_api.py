"""Plugin contract for reconstruction modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ReconPlugin(ABC):
    """Base class for reconstruction plugins."""

    @property
    def name(self) -> str:
        """Public plugin name used by registry and config lookup."""
        return self.__class__.__name__

    def validate_config(self, config: dict[str, Any]) -> None:
        """Hook for plugin-specific config checks."""

    @abstractmethod
    def process(self, connection: Any, config: Any, metadata: Any) -> None:
        """Run reconstruction for a single connection/session."""
