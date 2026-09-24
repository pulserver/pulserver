"""The timing and gradient checks a sequence chain passes before its IR is built."""

from __future__ import annotations

import contextlib
import copy
import io
from pathlib import Path

import pypulseqpp as pp
from pypulseqpp import safety

from ..mrd._sequence import read_chain

#: Timing problems listed per file; the report counts the rest.
_LISTED = 5
_RASTERS = (
    "rf_raster_time",
    "grad_raster_time",
    "adc_raster_time",
    "block_duration_raster",
)


def check(seq_path: Path | str, system: pp.Opts) -> list[str]:
    """Return the timing and gradient problems of a chain under a scanner's limits.

    Each file of the ``NextSequence`` chain is checked with
    ``pypulseqpp.check_timing``, gradient continuity across blocks included,
    and with ``pypulseqpp.safety.check_max_grad`` and ``check_max_slew``,
    against the gradient limits, dead times and ringdown time of ``system``.
    The waveforms are timed by the file's own rasters, and are those of the
    logical frame with each block's own rotation: the prescription's rotation
    is applied by the scanner, after these checks.

    Returns
    -------
    list of str
        One line per problem, naming the file when the chain has more than
        one; empty when every check passes.

    Raises
    ------
    ValueError
        If a file of the chain cannot be read.
    """
    try:
        chain_read = read_chain(seq_path, verify=False)
    except RuntimeError as failure:
        raise ValueError(f"cannot read {seq_path}: {failure}") from failure
    problems = []
    for path, sequence in chain_read:
        found = _problems(sequence, system)
        if len(chain_read) > 1:
            found = [f"{path.name}: {problem}" for problem in found]
        problems += found
    return problems


def _problems(sequence: pp.Sequence, system: pp.Opts) -> list[str]:
    limits = copy.copy(system)
    for name in _RASTERS:
        setattr(limits, name, getattr(sequence, name))
    sequence.system = limits
    problems = []
    is_ok, report = pp.check_timing(sequence)
    if not is_ok:
        problems.append(_timing(sequence, report))
    for check, name, unit, scale in (
        (safety.check_max_grad, "gradient amplitude", "mT/m", 1e3),
        (safety.check_max_slew, "slew rate", "T/m/s", 1.0),
    ):
        is_ok, found = check(sequence, limits)
        if not is_ok:
            peak = found.per_axis
            problems.append(
                f"{name} of {peak.value / limits.gamma * scale:.1f} {unit} on "
                f"{peak.axis} in block {peak.block} exceeds "
                f"{found.limit / limits.gamma * scale:.1f} {unit}"
            )
    return problems


def _timing(sequence: pp.Sequence, report: list) -> str:
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        pp.print_error_report(sequence, report, max_errors=_LISTED, colored=False)
    return "timing: " + " ".join(printed.getvalue().split())
