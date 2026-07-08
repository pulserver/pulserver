"""GE vendor extensions gateway (private package pulserver-gehc)."""

try:
    from pulserver_gehc import *  # noqa: F401,F403
except ImportError as e:
    raise ImportError(
        "GE extensions require the private 'pulserver-gehc' package "
        "(pulserver-interpreter repo)."
    ) from e
