"""The ``pulserver`` command."""

from __future__ import annotations

import sys

_USAGE = (
    "usage: pulserver design {list,validate,generate,import,prune,push,serve} ...\n"
)


def main(argv: list[str] | None = None) -> int:
    """Run ``pulserver <command> ...``; return its exit status."""
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] != "design":
        sys.stderr.write(_USAGE)
        return 2
    from .host._command import main as design

    return design(argv[1:])


if __name__ == "__main__":
    sys.exit(main())
