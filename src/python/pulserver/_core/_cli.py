"""Generic offline CLI runner for sequence plugins, and the writer it shares."""

from __future__ import annotations

import argparse
import sys
import warnings

import pulserver.pypulseq as pp

from ._params import set_protocol_value


def write_sequence(seq: pp.Sequence, output_path: str, *, offline: bool) -> str | None:
    """Write a finished sequence in the form its destination reads.

    The two destinations want opposite things, and a plugin should not have to
    remember which is which.

    ``offline=False`` -- straight to the scanner. The binary format, which the
    interpreter parses an order of magnitude faster than the text one and
    which loses none of the precision the text one rounds away. No checks and
    no signature: the interpreter checks the timing against its own rasters
    and the gradients against its own limits at predownload, which is the
    authoritative pass.

    ``offline=True`` -- anywhere else: a bench, a foreign toolbox, a file
    someone will read. ``.seq`` text, signed, and with every check run here
    because nothing downstream will run them.

    Both are deduplicated. It costs a millisecond or two and takes several
    times the size off the file, which the interpreter then does not have to
    parse.

    Parameters
    ----------
    seq : pulserver.pypulseq.Sequence
        The finished sequence.
    output_path : str
        Where to write.
    offline : bool
        Which of the two forms to write.

    Returns
    -------
    str or None
        The signature, when one was created.

    Warns
    -----
    UserWarning
        Under ``offline``, once per failed check.
    """
    if not offline:
        seq.remove_duplicates().write_binary(output_path)
        return None

    for is_ok, message in (
        seq.check_gradient_continuity(),
        seq.check_hardware_limits(),
    ):
        if not is_ok:
            warnings.warn(f"write_sequence(): {message}", stacklevel=2)
    # `write` deduplicates and runs the timing check itself, and the gradient
    # continuity one this has just run.
    return seq.write(output_path, check_gradient_continuity=False)


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
    ``make_sequence`` + "Wrote sequence" print.

    ``make_sequence`` is called with ``offline=True``: a file written here is
    not going straight to a scanner, so it is written as `.seq` text with
    every check run and a signature appended. See
    :func:`pulserver.write_sequence`.

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
    parser.add_argument(
        "-o", "--output", default=default_output, help="Output .seq file path"
    )

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

    parser.add_argument(
        "--max-grad-mtm",
        type=float,
        help="System max gradient amplitude [mT/m] for offline generation",
    )
    parser.add_argument(
        "--max-slew-tm-s",
        type=float,
        help="System max slew [T/m/s] for offline generation",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run protocol validation only (do not write sequence)",
    )

    args = parser.parse_args(argv)

    system_kwargs = {}
    if args.max_grad_mtm is not None:
        system_kwargs["max_grad"] = args.max_grad_mtm
        system_kwargs["grad_unit"] = "mT/m"
    if args.max_slew_tm_s is not None:
        system_kwargs["max_slew"] = args.max_slew_tm_s
        system_kwargs["slew_unit"] = "T/m/s"
    system = pp.Opts(**system_kwargs)

    protocol = plugin.get_default_protocol(system)

    for dest, key, kind in dests:
        value = getattr(args, dest)
        if isinstance(kind, tuple) and kind[0] == "const":
            if value:
                set_protocol_value(protocol, key, kind[1])
        elif value is not None:
            if isinstance(kind, dict):
                value = kind[value]
            set_protocol_value(protocol, key, value)

    result = plugin.validate_protocol(system, protocol)
    if not result.get("valid", False):
        print(f"ERROR: {result.get('info', 'Protocol invalid')}", file=sys.stderr)
        return 2

    print(result.get("info", "Protocol valid"))
    if args.validate_only:
        return 0

    plugin.make_sequence(system, protocol, args.output, offline=True)
    print(f"Wrote sequence: {args.output}")
    return 0
