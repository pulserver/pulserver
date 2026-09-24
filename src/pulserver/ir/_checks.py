"""The checks a sequence chain passes before its IR is built."""

from __future__ import annotations

import contextlib
import copy
import io
from dataclasses import dataclass, replace
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


@dataclass(frozen=True)
class CheckLimits:
    """The nerve, resonance and SAR limits a chain is checked against, besides ``pypulseqpp.Opts``.

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
        VOPs of ``check_sar``, or the ``.mat`` or ``.npz`` file holding them,
        which is read when a chain is checked. ``None`` leaves out the SAR
        check.
    drive_per_hz
        Channel drive per Hz of RF amplitude, in the VOPs' drive unit: one
        value, or one per channel.
    local_sar_limit, global_sar_limit
        SAR allowed over each window ``check_sar`` averages, in W/kg.

    Raises
    ------
    ValueError
        If VOPs are given without a drive, or the PNS limit is not positive.
    """

    pns: Any = None
    pns_limit: float = 1.0
    bands: tuple[safety.ForbiddenBand, ...] = ()
    vops: safety.VopModel | Path | str | None = None
    drive_per_hz: float | tuple[float, ...] | None = None
    local_sar_limit: float = 10.0
    global_sar_limit: float = 3.2

    def __post_init__(self) -> None:
        if self.vops is not None and self.drive_per_hz is None:
            raise ValueError("a VOP check needs the channel drive per Hz")
        if self.pns_limit <= 0.0:
            raise ValueError("the PNS limit must be positive")


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
    ``check_pns``, ``check_mech_resonance`` and ``check_sar`` where ``limits``
    carries a nerve model, forbidden bands or VOPs. The waveforms are timed by
    the file's own rasters.

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
        The nerve, resonance and SAR limits; none by default.

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
    if isinstance(limits.vops, (str, Path)):
        limits = replace(limits, vops=safety.read_vops(limits.vops))
    try:
        chain_read = read_chain(seq_path, verify=False)
    except RuntimeError as failure:
        raise ValueError(f"cannot read {seq_path}: {failure}") from failure
    turn = None if rotation is None else _proper(rotation)
    problems = []
    for path, sequence in chain_read:
        found = _problems(sequence, system, turn, limits)
        if len(chain_read) > 1:
            found = [f"{path.name}: {problem}" for problem in found]
        problems += found
    return problems


def _proper(rotation: np.ndarray) -> np.ndarray | None:
    """Return a rotation with the per-axis magnitudes of ``rotation``, or ``None`` for the identity.

    A reflection is made a rotation by reversing the physical z axis, which
    changes the sign of what it plays and no check reads a sign.
    """
    turn = np.array(rotation, dtype=float)
    if turn.shape != (3, 3) or not np.allclose(turn @ turn.T, np.eye(3), atol=1e-9):
        raise ValueError(f"the rotation {turn.tolist()} is not orthonormal")
    if np.linalg.det(turn) < 0.0:
        turn[2] *= -1.0
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
    if limits.vops is not None:
        problems += _sar(sequence, limits)
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


def _sar(sequence: pp.Sequence, limits: CheckLimits) -> list[str]:
    _, found = safety.check_sar(
        sequence,
        limits.vops,
        drive_per_hz=limits.drive_per_hz,
        local_limit=limits.local_sar_limit,
        global_limit=limits.global_sar_limit,
    )
    problems = []
    local, whole = found.worst_local, found.worst_global
    if local is not None and local.sar > limits.local_sar_limit:
        problems.append(
            f"local SAR of {local.sar:.2f} W/kg at VOP {local.vop} over blocks "
            f"{local.first}-{local.last} exceeds {limits.local_sar_limit:.2f} W/kg"
        )
    if whole is not None and whole.sar > limits.global_sar_limit:
        problems.append(
            f"global SAR of {whole.sar:.2f} W/kg over blocks {whole.first}-"
            f"{whole.last} exceeds {limits.global_sar_limit:.2f} W/kg"
        )
    return problems


def _timing(sequence: pp.Sequence, report: list) -> str:
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        pp.print_error_report(sequence, report, max_errors=_LISTED, colored=False)
    return "timing: " + " ".join(printed.getvalue().split())
