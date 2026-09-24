"""The ``pulserver design`` command: one design call, answered on standard output."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path
from typing import Any

#: The environment variable naming the socket of a warm server, when
#: ``--socket`` does not.
SOCKET_VARIABLE = "PULSERVER_DESIGN_SOCKET"

_DAY = 86400.0

_DESCRIPTION = """\
Answer one design call of a PSD host process. The protocol or import block
is read from standard input; the reply is written to standard output, as the
interpreter parses it, and ends with exit status 0, or 1 after an ERROR line.
A call is forwarded to the warm server listening on --socket, or on
$PULSERVER_DESIGN_SOCKET, and answered in this process when none listens.
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
        if name != "prune":
            sub.add_argument(
                "--socket", type=Path, help="Unix socket of a warm design server"
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
    serve = calls.add_parser(
        "serve",
        help="Answer forwarded calls, one forked child per call.",
        description="Warm a design server, then answer the calls forwarded to "
        "its socket until SIGTERM or SIGINT.",
    )
    serve.add_argument(
        "--plugins", type=Path, required=True, help="directory of <plugin>.py to warm"
    )
    serve.add_argument(
        "--socket", type=Path, required=True, help="Unix socket to listen on"
    )
    return parser


def request(args: argparse.Namespace, block: str) -> dict[str, Any]:
    """Return the request of the call ``args`` names, as a warm server takes it.

    Paths are absolute, so a server running elsewhere reads the same files.

    Raises
    ------
    OSError
        If the limits file cannot be read.
    """
    found: dict[str, Any] = {"call": args.call}
    if getattr(args, "plugin", None) is not None:
        found["plugins"] = str(args.plugins.absolute())
        found["plugin"] = args.plugin
    if getattr(args, "limits", None) is not None:
        found["limits"] = args.limits.read_text()
    if getattr(args, "store", None) is not None:
        found["store"] = str(args.store.absolute())
    if args.call in ("validate", "generate", "import"):
        found["input"] = block
    return found


def forward(socket_path: Path | str, call: dict[str, Any]) -> tuple[int, str] | None:
    """Return a warm server's exit status and reply, or ``None`` when none listens."""
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        connection.connect(str(socket_path))
    except (FileNotFoundError, ConnectionRefusedError):
        connection.close()
        return None
    with connection:
        connection.sendall((json.dumps(call) + "\n").encode())
        with connection.makefile("rb") as reader:
            line = reader.readline()
    if not line:
        return 1, "ERROR the design server closed the call without a reply\n"
    reply = json.loads(line)
    return int(reply["status"]), str(reply["output"])


def main(argv: list[str] | None = None) -> int:
    """Run one call from ``argv``; return its exit status."""
    args = _parser().parse_args(argv)
    if args.call == "serve":
        from ._server import serve

        return serve(args.plugins, args.socket)
    if args.call == "prune":
        from ._store import DesignStore

        removed = DesignStore(args.store).prune(
            max_age=None if args.max_age_days is None else args.max_age_days * _DAY,
            max_bytes=args.max_bytes,
        )
        sys.stdout.write(f"PRUNED {len(removed)}\n")
        return 0
    try:
        call = request(args, sys.stdin.read() if args.call != "list" else "")
    except OSError as error:
        sys.stdout.write(f"ERROR {error}\n")
        return 1
    server = args.socket or os.environ.get(SOCKET_VARIABLE)
    answered = forward(server, call) if server else None
    if answered is None:
        from ._server import answer

        answered = answer(call)
    status, output = answered
    sys.stdout.write(output)
    return status
