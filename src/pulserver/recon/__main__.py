"""Run the reconstruction server: ``pulserver recon --plugins DIR --port N``."""

from __future__ import annotations

import argparse
import signal
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pulserver recon")
    parser.add_argument(
        "--plugins",
        type=Path,
        action="append",
        required=True,
        help="directory of <plugin>.py, repeatable; the first holding it is used",
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
    parser.add_argument(
        "--save-data",
        type=Path,
        default=None,
        metavar="DIR",
        help="keep each series there as the proxy forwards it, so one can be "
        "reconstructed again offline; kept nowhere when unset",
    )
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=None,
        help="seconds without a client after which the server closes. A server "
        "started for one scan outlives whoever started it, and this is how it "
        "ends on its own; it never closes while a reconstruction is running",
    )
    parser.add_argument(
        "--logfile",
        type=Path,
        default=None,
        help="file the log is written to; the standard error when unset",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="DEBUG, INFO, WARNING or ERROR",
    )
    args = parser.parse_args(argv)

    from .._logs import configure
    from ..proxy import ReconServer

    configure(args.log_level, args.logfile)
    server = ReconServer(
        args.plugins,
        slots=args.slots,
        gpu_slots=args.gpu_slots,
        spares=args.spares,
        recon_timeout=args.recon_timeout,
        queue=args.queue,
        save_to=args.save_data,
        idle_timeout=args.idle_timeout,
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
