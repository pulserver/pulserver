"""Generic offline CLI runner for sequence plugins."""

from __future__ import annotations

import argparse
import sys

import pulserver.pypulseq as pp

from ._params import set_protocol_value


def run_cli(
    plugin,
    argv: list[str],
    *,
    arg_map: list[tuple],
    description: str | None = None,
    default_output: str = "sequence.seq",
) -> int:
    """Run a plugin's offline CLI from a declarative flag→protocol mapping.

    Owns the shared options (``-o/--output``, ``--max-grad-mtm``,
    ``--max-slew-tm-s``, ``--validate-only``), protocol defaulting, per-flag
    overrides via :func:`pulserver.set_protocol_value`,
    validation (prints ``info``, returns 2 when invalid), and the final
    ``make_sequence`` + "Wrote sequence" print — exactly the behavior of the
    per-plugin ``_cli`` functions it replaces.

    Parameters
    ----------
    plugin : Sequence
        Plugin instance with ``get_default_protocol`` / ``validate_protocol``
        / ``make_sequence``.
    argv : list of str
        Command-line arguments (without the program name).
    arg_map : list of tuple
        Entries ``(flag, key, kind, help)`` where ``kind`` is:

        - ``float`` or ``int`` — plain typed option; set ``key`` when given.
        - a ``dict`` — choices option; the stored value is ``kind[choice]``
          (use an identity mapping for direct string pass-through).
        - ``("const", value)`` — ``store_true`` flag; set ``key`` to
          ``value`` when present.
    description : str or None, optional
        Parser description.
    default_output : str, optional
        Default ``--output`` path.

    Returns
    -------
    int
        Process exit code: 0 on success, 2 on invalid protocol.

    Examples
    --------
    >>> from pulserver import UIParam, run_cli
    >>> _ARG_MAP = [
    ...     ("--te-ms", UIParam.TE, float, "Echo time [ms]"),
    ...     ("--inversion-mode", UIParam.user_value(1),
    ...      {"hard": 0.0, "adiabatic": 1.0}, "Inversion pulse shape"),
    ...     ("--swap-phase-freq", UIParam.SWAP_PHASE_FREQ, ("const", True),
    ...      "Swap readout/phase axes"),
    ... ]
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-o", "--output", default=default_output, help="Output .seq file path")

    dests: list[tuple[str, object, object]] = []
    for flag, key, kind, help_text in arg_map:
        dest = flag.lstrip("-").replace("-", "_")
        if isinstance(kind, dict):
            parser.add_argument(flag, choices=list(kind), help=help_text)
        elif isinstance(kind, tuple) and kind[0] == "const":
            parser.add_argument(flag, action="store_true", help=help_text)
        else:
            parser.add_argument(flag, type=kind, help=help_text)
        dests.append((dest, key, kind))

    parser.add_argument("--max-grad-mtm", type=float, help="System max gradient amplitude [mT/m] for offline generation")
    parser.add_argument("--max-slew-tm-s", type=float, help="System max slew [T/m/s] for offline generation")
    parser.add_argument("--validate-only", action="store_true", help="Run protocol validation only (do not write sequence)")

    args = parser.parse_args(argv)

    opts_kwargs = {}
    if args.max_grad_mtm is not None:
        opts_kwargs["max_grad"] = args.max_grad_mtm
        opts_kwargs["grad_unit"] = "mT/m"
    if args.max_slew_tm_s is not None:
        opts_kwargs["max_slew"] = args.max_slew_tm_s
        opts_kwargs["slew_unit"] = "T/m/s"
    opts = pp.Opts(**opts_kwargs)

    protocol = plugin.get_default_protocol(opts)

    for dest, key, kind in dests:
        value = getattr(args, dest)
        if isinstance(kind, tuple) and kind[0] == "const":
            if value:
                set_protocol_value(protocol, key, kind[1])
        elif value is not None:
            if isinstance(kind, dict):
                value = kind[value]
            set_protocol_value(protocol, key, value)

    result = plugin.validate_protocol(opts, protocol)
    if not result.get("valid", False):
        print(f"ERROR: {result.get('info', 'Protocol invalid')}", file=sys.stderr)
        return 2

    print(result.get("info", "Protocol valid"))
    if args.validate_only:
        return 0

    plugin.make_sequence(opts, protocol, args.output)
    print(f"Wrote sequence: {args.output}")
    return 0
