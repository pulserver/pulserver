"""Host half of the orchestrator: the design calls of interpreter host processes."""

from ._store import DesignStore, design_id, design_identity

__all__ = ["DesignStore", "call", "design_id", "design_identity"]


def __getattr__(name: str):
    # A call imports the design engine; a call forwarded from a shell does not.
    if name == "call":
        from ._service import call

        return call
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
