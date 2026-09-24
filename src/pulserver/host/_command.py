"""The ``pulserver design`` command: one design call, answered on standard output."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from ._blocks import parse_limits

_DAY = 86400.0

_DESCRIPTION = """\
Answer one design call of a PSD host process. The protocol or import block
is read from standard input; the reply is written to standard output, as the
interpreter parses it, and ends with exit status 0, or 1 after an ERROR line.
"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pulserver design", description=_DESCRIPTION)
    calls = parser.add_subparsers(dest="call", required=True, metavar="CALL")

    def call(name: str, help_text: str, *, plugin: bool, limits: bool, store: bool):
        sub = calls.add_parser(name, help=help_text, description=help_text)
        if plugin:
            sub.add_argument(
                "--plugins", type=Path, required=True, help="directory of <plugin>.py"
            )
            sub.add_argument("--plugin", required=True, help="scanner-sequence plugin")
        if limits:
            sub.add_argument(
                "--limits",
                type=Path,
                required=True,
                help="file holding the [Limits] block",
            )
        if store:
            sub.add_argument(
                "--store", type=Path, required=True, help="directory of designs"
            )
        return sub

    call(
        "list",
        "Reply the plugin's protocol listing.",
        plugin=True,
        limits=False,
        store=False,
    )
    call(
        "validate",
        "Resolve the value block on standard input.",
        plugin=True,
        limits=True,
        store=False,
    )
    call(
        "generate",
        "Design, check and store the value block on standard input.",
        plugin=True,
        limits=True,
        store=True,
    )
    call(
        "import",
        "Check and store the sequence file the import block on standard input names.",
        plugin=False,
        limits=True,
        store=True,
    )
    prune = call(
        "prune",
        "Remove designs from the store, least recently used first.",
        plugin=False,
        limits=False,
        store=True,
    )
    prune.add_argument(
        "--max-age-days", type=float, help="remove designs unused for longer"
    )
    prune.add_argument(
        "--max-bytes",
        type=int,
        help="then remove designs until the store holds at most",
    )
    return parser


def inputs(args: argparse.Namespace, block: str) -> dict[str, Any]:
    """Return the keyword arguments of the call ``args`` names."""
    found: dict[str, Any] = {}
    if getattr(args, "plugin", None) is not None:
        found["plugins"] = args.plugins
        found["plugin"] = args.plugin
    if getattr(args, "limits", None) is not None:
        found["limits"] = parse_limits(args.limits.read_text())
    if args.call in ("validate", "generate", "import"):
        found["block"] = block
    return found


def main(argv: list[str] | None = None) -> int:
    """Run one call from ``argv``; return its exit status."""
    args = _parser().parse_args(argv)
    if args.call == "prune":
        from ._store import DesignStore

        removed = DesignStore(args.store).prune(
            max_age=None if args.max_age_days is None else args.max_age_days * _DAY,
            max_bytes=args.max_bytes,
        )
        sys.stdout.write(f"PRUNED {len(removed)}\n")
        return 0
    try:
        found = inputs(args, sys.stdin.read() if args.call != "list" else "")
    except OSError as error:
        sys.stdout.write(f"ERROR {error}\n")
        return 1
    from . import _service
    from ._store import DesignStore

    if getattr(args, "store", None) is not None:
        found["store"] = DesignStore(args.store)
    status, reply = _service.reply(args.call, **found)
    sys.stdout.write(reply)
    return status
