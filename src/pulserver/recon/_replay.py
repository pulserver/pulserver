"""Reconstruct a kept series in this process: ``pulserver replay SERIES --plugin FILE``."""

from __future__ import annotations

import argparse
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pulserver replay",
        description="Reconstruct a series as the scanner sent it, in this process, "
        "enriched from the design it was played from as the proxy enriches it.",
    )
    parser.add_argument(
        "series",
        type=Path,
        help="ISMRMRD file of the series, as the proxy's --save-data keeps it",
    )
    parser.add_argument(
        "--plugin", type=Path, required=True, help="reconstruction plugin file"
    )
    played = parser.add_mutually_exclusive_group(required=True)
    played.add_argument(
        "--store", type=Path, help="design store holding the design the header names"
    )
    played.add_argument(
        "--design",
        type=Path,
        help="the design's sequence file, the first of its chain, with its "
        "manifest beside it",
    )
    parser.add_argument(
        "--config", help="configuration payload, as the scanner's stream carries it"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(),
        help="directory the reconstruction's images are written to",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run ``pulserver replay`` with ``argv``; return its exit status.

    Images go to ``images.h5`` in ``--output`` and DICOM datasets to the files
    they are named by. A plugin's exception is not caught, so its traceback
    names the line that raised it.
    """
    from ..virtual._command import _received
    from ._loader import load_plugin

    args = _parser().parse_args(argv)
    emitted = load_plugin(args.plugin).run(
        str(args.series),
        config=args.config,
        store=None if args.store is None else str(args.store),
        design=None if args.design is None else str(args.design),
    )
    return _received(emitted, args.output)
