"""The checks a sequence chain passes before its IR is built, and its SAR against a reference."""

from __future__ import annotations

import contextlib
import copy
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
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

#: The reference pulse every pulse of a repetition is replaced by: hard,
#: 180 degrees, 1 ms, in the default channel shim.
REFERENCE_FLIP = np.pi
REFERENCE_DURATION = 1e-3


@dataclass(frozen=True)
class CheckLimits:
    """The nerve and resonance limits of a chain, and the VOPs of its SAR ratios, besides ``pypulseqpp.Opts``.

    Attributes
    ----------
    pns
        Nerve model of ``pypulseqpp.safety.check_pns``: a ``ChronaxieModel``
        or a SAFE description. ``None`` leaves out the PNS check.
    pns_limit
        Largest PNS response allowed, as a fraction of the model's threshold.
    bands
        Forbidden gradient bands of ``check_mech_resonance``, on the physical
        axes. Without any, the resonance check is left out.
    vops
        VOPs, or the ``.mat`` or ``.npz`` file holding them, which
        :func:`sar_ratios` reads; nothing is refused on SAR.
    drive_per_hz
        Channel drive per Hz of RF amplitude, in the VOPs' drive unit: one
        value, or one per channel. A scale common to every channel cancels in
        the ratios.
    default_shim
        Complex channel weights of a pulse played without an RF shim, and of
        the reference pulse; equal weights when None.

    Raises
    ------
    ValueError
        If the PNS limit is not positive.
    """

    pns: Any = None
    pns_limit: float = 1.0
    bands: tuple[safety.ForbiddenBand, ...] = ()
    vops: safety.VopModel | Path | str | None = None
    drive_per_hz: float | tuple[float, ...] = 1.0
    default_shim: tuple[complex, ...] | None = None

    def __post_init__(self) -> None:
        if self.pns_limit <= 0.0:
            raise ValueError("the PNS limit must be positive")


@dataclass(frozen=True)
class SarRatio:
    """A subsequence's RF energy at the VOPs, against the reference pulse.

    ``local_sar`` is the largest, over the subsequence's repetitions and the
    VOPs, of the energy a repetition deposits at a VOP over the energy there
    of the same repetition with each of its pulses replaced by the reference
    pulse; ``global_sar`` is that ratio through the global SAR matrix. A
    scanner's SAR for the subsequence is the ratio times its SAR for that
    reference repetition. Both are 0 without RF, and ``global_sar`` is 0
    without a global matrix.
    """

    local_sar: float
    global_sar: float


def check(
    seq_path: Path | str,
    system: pp.Opts,
    *,
    rotation: np.ndarray | None = None,
    limits: CheckLimits | None = None,
) -> list[str]:
    """Return the problems of a chain under a scanner's limits, in the physical frame.

    Each file of the ``NextSequence`` chain is rotated by ``rotation``,
    composed after each block's own rotation with blocks labelled ``NOROT``
    exempt, as the scanner plays it. It is then checked with
    ``pypulseqpp.check_timing``, gradient continuity included, and with
    ``pypulseqpp.safety.check_max_grad`` and ``check_max_slew``, against the
    gradient limits, dead times and ringdown time of ``system``; and with
    ``check_pns`` and ``check_mech_resonance`` where ``limits`` carries a
    nerve model or forbidden bands. The waveforms are timed by the file's own
    rasters. VOPs are not checked here: see :func:`sar_ratios`.

    Parameters
    ----------
    seq_path
        The first file of the chain.
    system
        The scanner's gradient limits, dead times and ringdown time.
    rotation
        ``(3, 3)`` prescription rotation from logical to physical axes, a
        reflection included; the identity by default.
    limits
        The nerve and resonance limits; none by default.

    Returns
    -------
    list of str
        One line per problem, naming the file when the chain has more than
        one; empty when every check passes.

    Raises
    ------
    ValueError
        If a file of the chain cannot be read, or ``rotation`` is not
        orthonormal.
    """
    limits = CheckLimits() if limits is None else limits
    try:
        chain_read = read_chain(seq_path, verify=False)
    except RuntimeError as failure:
        raise ValueError(f"cannot read {seq_path}: {failure}") from failure
    turn = None if rotation is None else _prescription(rotation)
    problems = []
    for path, sequence in chain_read:
        found = _problems(sequence, system, turn, limits)
        if len(chain_read) > 1:
            found = [f"{path.name}: {problem}" for problem in found]
        problems += found
    return problems


def _prescription(rotation: np.ndarray) -> np.ndarray | None:
    """Return ``rotation`` as a matrix, or ``None`` for the identity."""
    turn = np.array(rotation, dtype=float)
    if turn.shape != (3, 3) or not np.allclose(turn @ turn.T, np.eye(3), atol=1e-9):
        raise ValueError(f"the rotation {turn.tolist()} is not orthonormal")
    return None if np.allclose(turn, np.eye(3)) else turn


def _problems(
    sequence: pp.Sequence,
    system: pp.Opts,
    rotation: np.ndarray | None,
    limits: CheckLimits,
) -> list[str]:
    opts = copy.copy(system)
    for name in _RASTERS:
        setattr(opts, name, getattr(sequence, name))
    sequence.system = opts
    if rotation is not None:
        pp.TransformFOV(rotation=rotation).apply_to_sequence(sequence, in_place=True)
    problems = []
    is_ok, report = pp.check_timing(sequence)
    if not is_ok:
        problems.append(_timing(sequence, report))
    for check, name, unit, scale in (
        (safety.check_max_grad, "gradient amplitude", "mT/m", 1e3),
        (safety.check_max_slew, "slew rate", "T/m/s", 1.0),
    ):
        is_ok, found = check(sequence, opts)
        if not is_ok:
            peak = found.per_axis
            problems.append(
                f"{name} of {peak.value / opts.gamma * scale:.1f} {unit} on "
                f"{peak.axis} in block {peak.block} exceeds "
                f"{found.limit / opts.gamma * scale:.1f} {unit}"
            )
    if limits.pns is not None:
        problems += _pns(sequence, opts, limits)
    if limits.bands:
        problems += _resonance(sequence, opts, limits)
    return problems


def _pns(sequence: pp.Sequence, system: pp.Opts, limits: CheckLimits) -> list[str]:
    _, found = safety.check_pns(sequence, limits.pns, system=system)
    peak = found.peak
    if peak.value < limits.pns_limit:
        return []
    axis = max(found.axes, key=lambda reading: reading.value).axis
    return [
        f"PNS of {peak.value:.0%} of the {found.model} threshold, largest on "
        f"{axis}, at {peak.time:.4f} s in block {peak.block} reaches "
        f"{limits.pns_limit:.0%}"
    ]


def _resonance(
    sequence: pp.Sequence, system: pp.Opts, limits: CheckLimits
) -> list[str]:
    _, found = safety.check_mech_resonance(sequence, limits.bands, system=system)
    return [
        f"{band.peak:.1f} mT/m at {band.frequency:.0f} Hz on {band.peak_axis} in "
        f"the window at {band.window_start:.3f} s exceeds the {band.threshold:.1f} "
        f"mT/m of the forbidden band {band.f_min:.0f}-{band.f_max:.0f} Hz"
        + ("" if band.axis is None else f" on {band.axis}")
        for band in found.bands
        if band.violations
    ]


def sar_ratios(
    seq_path: Path | str, system: pp.Opts, limits: CheckLimits
) -> list[SarRatio]:
    """Return the SAR ratios of each file of a chain against the reference pulse.

    The reference pulse is hard, 180 degrees and 1 ms, played in the default
    shim of ``limits``; a pulse played in an RF shim is weighed through it. A
    repetition is a window ``pypulseqpp.safety.check_sar`` averages over: each
    repetition the block definitions repeat with, and the blocks before the
    first and after the last.

    Parameters
    ----------
    seq_path
        The first file of the chain.
    system
        The rasters and RF dead times the reference pulse is made with.
    limits
        The VOPs, channel drive and default shim.

    Returns
    -------
    list of SarRatio
        One per file, in play order.

    Raises
    ------
    ValueError
        If ``limits`` carries no VOPs, a file of the chain cannot be read, or a
        pulse or shim weighs another number of channels than the VOPs.
    """
    if limits.vops is None:
        raise ValueError("SAR ratios need VOPs")
    vops = limits.vops
    if not isinstance(vops, safety.VopModel):
        vops = safety.read_vops(vops)
    drive = {"drive_per_hz": limits.drive_per_hz, "default_shim": limits.default_shim}
    reference = pp.Sequence(system)
    reference.add_block(
        pp.make_block_pulse(
            flip_angle=REFERENCE_FLIP, duration=REFERENCE_DURATION, system=system
        )
    )
    _, pulse = safety.check_sar(reference, vops, **drive)
    try:
        chain_read = read_chain(seq_path, verify=False)
    except RuntimeError as failure:
        raise ValueError(f"cannot read {seq_path}: {failure}") from failure
    ratios = []
    for _, sequence in chain_read:
        _, found = safety.check_sar(sequence, vops, reference=pulse, **drive)
        ratios.append(_ratio(sequence, found, pulse))
    return ratios


def _ratio(sequence: pp.Sequence, found: Any, pulse: Any) -> SarRatio:
    """Divide each window's energy against one reference pulse by the pulses it plays."""
    windows = found.windows
    rf = np.array([row[1] for row in sequence.block_events.values()], dtype=int)
    counted = np.concatenate([[0], np.cumsum(rf > 0)])
    pulses = counted[windows.last] - counted[windows.first - 1]
    if not windows.first.size or not pulses.any():
        return SarRatio(0.0, 0.0)
    with_rf = pulses > 0
    unit = float(pulse.windows.duration[0])
    per_pulse = windows.duration / (unit * np.maximum(pulses, 1))
    local = float((windows.reference_ratio * per_pulse)[with_rf].max())
    whole = 0.0
    if windows.global_sar is not None and pulse.windows.global_sar[0] > 0.0:
        against = float(pulse.windows.global_sar[0])
        whole = float((windows.global_sar / against * per_pulse)[with_rf].max())
    return SarRatio(local, whole)


def _timing(sequence: pp.Sequence, report: list) -> str:
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        pp.print_error_report(sequence, report, max_errors=_LISTED, colored=False)
    return "timing: " + " ".join(printed.getvalue().split())
