"""Host half of the orchestrator: design sessions for scanner PSD host processes."""

from ._sessions import Session, SessionKey, SessionStore, revision_hash
from .client import HostClient, HostError

__all__ = [
    "HostClient",
    "HostDaemon",
    "HostError",
    "Session",
    "SessionKey",
    "SessionStore",
    "revision_hash",
]


def __getattr__(name: str):
    # The daemon imports the design engine; a client process never loads it.
    if name == "HostDaemon":
        from ._daemon import HostDaemon

        return HostDaemon
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
