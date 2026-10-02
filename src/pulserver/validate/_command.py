"""The ``pulserver validate`` command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ._validate import VENDORS, validate


def main(argv: list[str] | None = None) -> int:
    """Check a sequence against a recording of it, or against its own cache."""
    parser = argparse.ArgumentParser(
        prog="pulserver validate",
        description=(
            "Check that a sequence plays as it was written: its gradients "
            "against a recording of a machine playing it, or against the "
            "waveforms its own cache holds."
        ),
    )
    parser.add_argument("seq", type=Path, help="the sequence file")
    parser.add_argument(
        "--vendor",
        choices=VENDORS,
        default=None,
        help="which machine the recording came from; without one the check is "
        "against the cache, which establishes that the conversion kept the "
        "sequence but not that a machine plays it",
    )
    parser.add_argument(
        "--played",
        type=Path,
        default=None,
        help="the recording to check against; without one, and with a vendor "
        "named, the recording is asked for from that vendor's tooling",
    )
    parser.add_argument(
        "--shift-us",
        type=float,
        default=0.0,
        help="move the recording in time before comparing, in microseconds: a "
        "machine drives its transmit and gradient channels on separate "
        "timelines and records each as it was driven",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="the largest difference that still counts as agreement, in mT/m",
    )
    parser.add_argument(
        "--limits",
        type=Path,
        default=None,
        help="file holding the [Limits] block, for a sequence with no cache yet",
    )
    parser.add_argument(
        "--cache-ext", default=".pseg", help="extension of the cache, dot included"
    )
    args = parser.parse_args(argv)

    system = None
    if args.limits is not None:
        from ..host._blocks import parse_limits
        from ..host._limits import design_system

        system = design_system(parse_limits(args.limits.read_text()))

    try:
        comparison = validate(
            args.seq,
            vendor=args.vendor,
            played=args.played,
            shift_us=args.shift_us,
            tolerance_mt_per_m=args.tolerance,
            cache_ext=args.cache_ext,
            system=system,
        )
    except (FileNotFoundError, NotImplementedError, ValueError) as error:
        sys.stderr.write(f"pulserver validate: {error}\n")
        return 2

    sys.stdout.write(f"{comparison}\n")
    return 0 if comparison.agrees else 1


if __name__ == "__main__":
    sys.exit(main())
