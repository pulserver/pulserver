"""Run the host daemon: ``python -m pulserver.host --base DIR --socket PATH --plugins DIR``."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
from pathlib import Path

from ._daemon import HostDaemon


async def _serve(daemon: HostDaemon, socket_path: Path) -> None:
    """Serve until SIGTERM or SIGINT cancels this task."""
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, task.cancel)
    await daemon.serve(socket_path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m pulserver.host")
    parser.add_argument(
        "--base", type=Path, required=True, help="directory holding bucket/"
    )
    parser.add_argument(
        "--socket", type=Path, required=True, help="Unix socket to listen on"
    )
    parser.add_argument(
        "--plugins", type=Path, required=True, help="directory of <plugin>.py"
    )
    parser.add_argument(
        "--workers", type=int, default=2, help="design worker processes"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    daemon = HostDaemon(args.base, args.plugins, workers=args.workers)
    args.socket.unlink(missing_ok=True)
    try:
        asyncio.run(_serve(daemon, args.socket))
    except asyncio.CancelledError:
        pass
    finally:
        daemon.shutdown()


if __name__ == "__main__":
    main()
