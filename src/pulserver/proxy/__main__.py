"""Run the reconstruction proxy: ``pulserver proxy --store DIR --port N (--plugins DIR | --forward HOST:PORT)``."""

from __future__ import annotations

import argparse
import signal
from pathlib import Path

from .._logs import configure
from .._memory import without_huge_page_advice
from ._intake import DesignIntake
from ._proxy import ReconProxy


def main(argv: list[str] | None = None) -> None:
    without_huge_page_advice()
    parser = argparse.ArgumentParser(prog="pulserver proxy")
    parser.add_argument(
        "--store",
        type=Path,
        required=True,
        help="directory of designs; the intake writes it, the proxy reads it",
    )
    parser.add_argument("--port", type=int, required=True, help="TCP port to listen on")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="address to listen on; the loopback interface when unset",
    )
    parser.add_argument(
        "--plugins",
        type=Path,
        action="append",
        default=None,
        help="directory of <plugin>.py, repeatable, the first holding it used; "
        "required unless --forward is given",
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
        "--intake-port",
        type=int,
        default=None,
        help="TCP port the design calls push designs to, on --host; none when unset",
    )
    parser.add_argument(
        "--queue",
        type=Path,
        default=None,
        help="directory the series waiting for a slot are written to; a "
        "temporary directory when unset",
    )
    parser.add_argument(
        "--exams",
        type=Path,
        default=None,
        help="directory the proxies of this host share the caches of an exam "
        "in; a location under the system temporary directory when unset",
    )
    parser.add_argument(
        "--recon-timeout",
        type=float,
        default=None,
        help="seconds a reconstruction may run after its series ends; unlimited when unset",
    )
    parser.add_argument(
        "--forward",
        type=_address,
        default=None,
        metavar="HOST:PORT",
        help="MRD server that reconstructs every series instead of local workers",
    )
    parser.add_argument(
        "--forward-config",
        default=None,
        help="config name sent to the --forward server; the series' reconstruction "
        "plugin when unset",
    )
    parser.add_argument(
        "--dicom",
        action="store_true",
        help="convert each image to DICOM before it is relayed, whatever "
        "reconstructed it; for a client that reads DICOM alone",
    )
    parser.add_argument(
        "--save-data",
        type=Path,
        default=None,
        metavar="DIR",
        help="keep each series there as the scanner sends it, so one can be "
        "reconstructed again offline; kept nowhere when unset",
    )
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=None,
        help="seconds without a client after which the proxy closes. A proxy "
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
    if args.forward is None and args.plugins is None:
        parser.error("--plugins is required unless --forward is given")

    configure(args.log_level, args.logfile)
    proxy = ReconProxy(
        args.store,
        args.plugins,
        slots=args.slots,
        gpu_slots=args.gpu_slots,
        spares=args.spares,
        recon_timeout=args.recon_timeout,
        queue=args.queue,
        exam_directory=args.exams,
        forward=args.forward,
        forward_config=args.forward_config,
        dicom=args.dicom,
        save_to=args.save_data,
        idle_timeout=args.idle_timeout,
    )
    proxy.bind(args.port, args.host)
    intake = None
    if args.intake_port is not None:
        intake = DesignIntake(
            args.store, args.host, args.intake_port, received=proxy.designs.read
        )
        intake.start()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda _signum, _frame: proxy.stop())
    try:
        proxy.serve()
    finally:
        if intake is not None:
            intake.close()
        proxy.close()


def _address(text: str) -> tuple[str, int]:
    host, separator, port = text.rpartition(":")
    if not separator or not host or not port.isdigit():
        raise argparse.ArgumentTypeError(f"{text!r} is not HOST:PORT")
    return host, int(port)


if __name__ == "__main__":
    main()
