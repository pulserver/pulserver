"""The checks a sequence chain passes before its IR is built, and its SAR against a reference."""

from __future__ import annotations

import contextlib
import copy
import io
from concurrent.futures import ThreadPoolExecutor
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
        VOPs and the global SAR matrices of their body models, or the ``.mat``
        or ``.npz`` file holding them, which :func:`sar_ratios` reads; nothing
        is refused on SAR.
    drive_per_hz
        Channel drive per Hz of RF amplitude, in the VOPs' drive unit: one
        value, or one per channel. A scale common to every channel cancels in
        the ratios.
    default_shim
        Complex channel weights of a pulse played without an RF shim, and of
        the reference pulse; equal weights when None.
    vop_coil
        The transmit configuration the scanner reports. When given, the VOP
        file's ``metadata["transmit"]`` must name the same one.
    vop_head_limit, vop_local_limit
        The scanner's head and local SAR limits, W/kg, in its current
        operating mode; :func:`sar_ratios` needs both.

    Raises
    ------
    ValueError
        If the PNS limit or a SAR limit is not positive.
    """

    pns: Any = None
    pns_limit: float = 1.0
    bands: tuple[safety.ForbiddenBand, ...] = ()
    vops: safety.VopModel | Path | str | None = None
    drive_per_hz: float | tuple[float, ...] = 1.0
    default_shim: tuple[complex, ...] | None = None
    vop_coil: str | None = None
    vop_head_limit: float | None = None
    vop_local_limit: float | None = None

    def __post_init__(self) -> None:
        if self.pns_limit <= 0.0:
            raise ValueError("the PNS limit must be positive")
        for limit in (self.vop_head_limit, self.vop_local_limit):
            if limit is not None and not limit > 0.0:
                raise ValueError(f"a SAR limit must be positive, not {limit}")


@dataclass(frozen=True)
class SarRatio:
    """A subsequence's RF energy per pulse, against the reference pulse at the head SAR limit.

    Each is the largest over the subsequence's repetitions, per pulse the
    repetition plays, at the drive scale where the reference pulse meets the
    head SAR limit. ``local_sar`` is the repetition's peak local energy,
    through the VOPs and times the VOP file's safety factor, over the
    reference pulse's head energy in the body model where it is smallest,
    times the head limit over the local limit. ``global_sar`` is the
    repetition's head energy over the reference pulse's, body model by body
    model. Given the time the scanner gives the reference pulse, times the
    larger of the two, each pulse of the subsequence stays within both
    limits. Both are 0 without RF.
    """

    local_sar: float
    global_sar: float


def check(
    seq_path: Path | str,
    system: pp.Opts,
    *,
    rotation: np.ndarray | None = None,
    limits: CheckLimits | None = None,
    designed: list[tuple[Path, pp.Sequence]] | None = None,
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
    designed
        The chain as :func:`pulserver.mrd.designed_chain` returns it, checked
        in place of reading the files, and rotated in place.

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
    chain_read = designed if designed is not None else _read(seq_path)
    turn = None if rotation is None else _prescription(rotation)
    problems = []
    for path, sequence in chain_read:
        found = _problems(sequence, system, turn, limits)
        if len(chain_read) > 1:
            found = [f"{path.name}: {problem}" for problem in found]
        problems += found
    return problems


def _read(seq_path: Path | str) -> list[tuple[Path, pp.Sequence]]:
    try:
        return read_chain(seq_path, verify=False)
    except RuntimeError as failure:
        raise ValueError(f"cannot read {seq_path}: {failure}") from failure


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
    checks = [
        (_limit, safety.check_max_grad, "gradient amplitude", "mT/m", 1e3),
        (_limit, safety.check_max_slew, "slew rate", "T/m/s", 1.0),
    ]
    if limits.pns is not None:
        checks.append((_pns, limits))
    if limits.bands:
        checks.append((_resonance, limits))
    # The native checks read the sequence and release the GIL; check_timing
    # records TotalDuration, so it runs before them.
    with ThreadPoolExecutor(len(checks)) as pool:
        found = [pool.submit(run, sequence, opts, *args) for run, *args in checks]
    for result in found:
        problems += result.result()
    return problems


def _limit(
    sequence: pp.Sequence,
    system: pp.Opts,
    check: Any,
    name: str,
    unit: str,
    scale: float,
) -> list[str]:
    is_ok, found = check(sequence, system)
    if is_ok:
        return []
    peak = found.per_axis
    return [
        f"{name} of {peak.value / system.gamma * scale:.1f} {unit} on "
        f"{peak.axis} in block {peak.block} exceeds "
        f"{found.limit / system.gamma * scale:.1f} {unit}"
    ]


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
    seq_path: Path | str,
    system: pp.Opts,
    limits: CheckLimits,
    designed: list[tuple[Path, pp.Sequence]] | None = None,
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
        The VOPs, the head and local SAR limits, the channel drive, the
        default shim and, when given, the transmit configuration the VOP file
        must name.
    designed
        The chain as :func:`pulserver.mrd.designed_chain` returns it, weighed
        in place of reading the files.

    Returns
    -------
    list of SarRatio
        One per file, in play order.

    Raises
    ------
    ValueError
        If ``limits`` carries no VOPs or not both SAR limits, the VOPs carry
        no global SAR matrices, the VOP file names another transmit
        configuration than ``limits.vop_coil``, a file of the chain cannot be
        read, or a pulse or shim weighs another number of channels than the
        VOPs.
    """
    if limits.vops is None:
        raise ValueError("SAR ratios need VOPs")
    if limits.vop_head_limit is None or limits.vop_local_limit is None:
        raise ValueError("SAR ratios need the scanner's head and local SAR limits")
    vops = limits.vops
    if not isinstance(vops, safety.VopModel):
        vops = safety.read_vops(vops)
    if limits.vop_coil is not None:
        written_for = (vops.metadata or {}).get("transmit")
        if written_for != limits.vop_coil:
            raise ValueError(
                f"the VOPs are for the transmit configuration {written_for!r}, "
                f"and the scanner reports {limits.vop_coil!r}"
            )
    drive = {"drive_per_hz": limits.drive_per_hz, "default_shim": limits.default_shim}
    reference = pp.Sequence(system)
    reference.add_block(
        pp.make_block_pulse(
            flip_angle=REFERENCE_FLIP, duration=REFERENCE_DURATION, system=system
        )
    )
    limit_ratio = limits.vop_head_limit / limits.vop_local_limit
    chain_read = designed if designed is not None else _read(seq_path)
    ratios = []
    for _, sequence in chain_read:
        _, found = safety.check_sar(sequence, vops, reference=reference, **drive)
        ratios.append(_ratio(sequence, found, limit_ratio))
    return ratios


def _ratio(sequence: pp.Sequence, found: Any, limit_ratio: float) -> SarRatio:
    """Divide each window's energy against one reference pulse by the pulses it plays."""
    windows = found.windows
    rf = np.array([row[1] for row in sequence.block_events.values()], dtype=int)
    counted = np.concatenate([[0], np.cumsum(rf > 0)])
    pulses = counted[windows.last] - counted[windows.first - 1]
    if not windows.first.size or not pulses.any():
        return SarRatio(0.0, 0.0)
    with_rf = pulses > 0
    per_pulse = windows.duration / (found.reference.duration * np.maximum(pulses, 1))
    local = float((windows.local_to_head * per_pulse)[with_rf].max()) * limit_ratio
    whole = float((windows.global_ratio * per_pulse)[with_rf].max())
    return SarRatio(local, whole)


def _timing(sequence: pp.Sequence, report: list) -> str:
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        pp.print_error_report(sequence, report, max_errors=_LISTED, colored=False)
    return "timing: " + " ".join(printed.getvalue().split())
