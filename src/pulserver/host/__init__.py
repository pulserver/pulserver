"""Host half of the orchestrator: the design calls of scanner PSD host processes."""

from ._sessions import Session, SessionKey, SessionStore, revision_hash
from ._store import DesignStore, design_id, design_identity

__all__ = [
    "DesignStore",
    "HostClient",
    "HostDaemon",
    "HostError",
    "Session",
    "SessionKey",
    "SessionStore",
    "design_id",
    "design_identity",
    "revision_hash",
]


def __getattr__(name: str):
    # The daemon imports the design engine and the client the protocol; a
    # call forwarded from a shell loads neither.
    if name == "HostDaemon":
        from ._daemon import HostDaemon

        return HostDaemon
    if name in ("HostClient", "HostError"):
        from . import client

        return getattr(client, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
