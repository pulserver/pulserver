"""Run the reconstruction server: ``python -m pulserver.recon --plugins DIR --port N``."""

from __future__ import annotations

import argparse
import logging
import signal
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m pulserver.recon")
    parser.add_argument(
        "--plugins", type=Path, required=True, help="directory of <plugin>.py"
    )
    parser.add_argument("--port", type=int, required=True, help="TCP port to listen on")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="address to listen on; the loopback interface when unset",
    )
    parser.add_argument(
        "--slots",
        type=int,
        default=None,
        help="series reconstructed at once; memory and the GPUs decide when unset",
    )
    parser.add_argument(
        "--gpu-slots",
        type=int,
        default=1,
        help="series reconstructed at once on each GPU, when --slots is unset",
    )
    parser.add_argument(
        "--spares", type=int, default=1, help="warm worker processes kept waiting"
    )
    parser.add_argument(
        "--queue",
        type=Path,
        default=None,
        help="directory the series waiting for a slot are written to; a "
        "temporary directory when unset",
    )
    parser.add_argument(
        "--recon-timeout",
        type=float,
        default=None,
        help="seconds a reconstruction may run after its series ends; unlimited when unset",
    )
    args = parser.parse_args(argv)

    from ..vre import ReconServer

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    server = ReconServer(
        args.plugins,
        slots=args.slots,
        gpu_slots=args.gpu_slots,
        spares=args.spares,
        recon_timeout=args.recon_timeout,
        queue=args.queue,
    )
    server.bind(args.port, args.host)
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda _signum, _frame: server.stop())
    try:
        server.serve()
    finally:
        server.close()


if __name__ == "__main__":
    main()
